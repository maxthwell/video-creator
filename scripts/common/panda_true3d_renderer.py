from __future__ import annotations

import hashlib
import io
import math
from collections import deque
from pathlib import Path
from typing import Any, Optional

import generate_actions_pose_reconstruction as poseviz
from PIL import Image, ImageChops, ImageDraw

from .io import CHARACTERS_DIR, TMP_DIR, _cached_remote_asset, manifest_index

MAX_EFFECT_ALPHA = 150
MIN_EFFECT_ALPHA = 100
PROP_WHITE_BG_VERSION = 2


class PandaTrue3DRenderer:
    def __init__(self, story: dict[str, Any], prefer_gpu: bool = True):
        try:
            from panda3d.core import (
                AmbientLight,
                CardMaker,
                Filename,
                GraphicsOutput,
                PNMImage,
                PerspectiveLens,
                StringStream,
                TextNode,
                Texture,
                TextureStage,
                TransparencyAttrib,
                loadPrcFileData,
            )
            from direct.showbase.ShowBase import ShowBase
        except ModuleNotFoundError as exc:
            raise RuntimeError("Panda3D is required. Install `panda3d` and rerun.") from exc

        self._core = {
            "AmbientLight": AmbientLight,
            "CardMaker": CardMaker,
            "Filename": Filename,
            "GraphicsOutput": GraphicsOutput,
            "PNMImage": PNMImage,
            "PerspectiveLens": PerspectiveLens,
            "StringStream": StringStream,
            "TextNode": TextNode,
            "Texture": Texture,
            "TextureStage": TextureStage,
            "TransparencyAttrib": TransparencyAttrib,
            "ShowBase": ShowBase,
            "loadPrcFileData": loadPrcFileData,
        }
        width = int(story["video"]["width"])
        height = int(story["video"]["height"])
        video_options = story.get("video", {})
        self.renderer_kind = str(video_options.get("renderer") or "true_3d").strip().lower()
        self.fast_card_mode = self.renderer_kind == "panda_card_fast"
        self.speed_mode = str(video_options.get("speed_mode") or "normal").strip().lower()
        self.extreme_speed_mode = self.speed_mode == "extreme"
        self.frame_width = width
        self.frame_height = height
        cache_dir = TMP_DIR / "panda_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        if prefer_gpu:
            loadPrcFileData("", "load-display pandagl")
            loadPrcFileData("", "aux-display p3tinydisplay")
        else:
            loadPrcFileData("", "load-display p3tinydisplay")
        loadPrcFileData("", "window-type offscreen")
        loadPrcFileData("", f"win-size {width} {height}")
        loadPrcFileData("", "audio-library-name null")
        loadPrcFileData("", "sync-video false")
        loadPrcFileData("", f"model-cache-dir {cache_dir}")
        loadPrcFileData("", "framebuffer-multisample 0")
        loadPrcFileData("", "multisamples 0")
        loadPrcFileData("", "texture-anisotropic-degree 0")
        if self.fast_card_mode:
            loadPrcFileData("", "basic-shaders-only true")
        if prefer_gpu:
            loadPrcFileData("", "textures-power-2 none")

        self.base = ShowBase(windowType="offscreen")
        self.pipe_name = self.base.pipe.getType().getName()
        self.base.disableMouse()
        self._capture_texture = Texture()
        self._capture_texture.setKeepRamImage(True)
        self.base.win.addRenderTexture(self._capture_texture, GraphicsOutput.RTM_copy_ram)
        self.story = story
        self.cast = {item["id"]: item for item in story["cast"]}
        self.backgrounds = manifest_index("backgrounds")
        self.floors = manifest_index("floors")
        self.props = manifest_index("props")
        self.characters = manifest_index("characters")
        self.effects = manifest_index("effects")
        self.foregrounds = manifest_index("foregrounds")
        self._shared_skins_dir = CHARACTERS_DIR / "_shared_skins"

        lens = PerspectiveLens()
        lens.setNearFar(0.1, 500.0)
        lens.setFov(44.0)
        self.base.cam.node().setLens(lens)
        self._lens = lens

        self.skybox_root = self.base.render.attachNewNode("skybox-root")
        self.stage_root = self.base.render.attachNewNode("true3d-stage")
        self.outside_root = self.stage_root.attachNewNode("outside-root")
        self.room_root = self.stage_root.attachNewNode("room-root")
        self.prop_root = self.stage_root.attachNewNode("prop-root")
        self.actor_root = self.stage_root.attachNewNode("actor-root")
        self.overlay_root = self.base.aspect2d.attachNewNode("overlay-root")
        self.effect_root = self.overlay_root.attachNewNode("effect-root")
        self.foreground_root = self.overlay_root.attachNewNode("foreground-root")
        self.subtitle_root = self.overlay_root.attachNewNode("subtitle-root")
        self.label_root = self.overlay_root.attachNewNode("label-root")

        self._box_model = self.base.loader.loadModel("models/box")
        self._texture_cache_dir = TMP_DIR / "true3d_asset_cache"
        self._texture_cache_dir.mkdir(parents=True, exist_ok=True)
        self._shape_texture_cache: dict[str, Any] = {}
        self._gradient_texture_cache: dict[str, Any] = {}
        self._texture_sequence_cache: dict[str, Any] = {}
        self._frame_duration_cache: dict[str, list[int]] = {}
        self._pose_track_cache: dict[str, Any] = {}
        self._text_font = self._load_text_font()
        self._prepared_scene_id: Optional[str] = None
        self._current_scene: Optional[dict[str, Any]] = None
        self._room_dims: dict[str, float] = {}
        self._actor_instances: dict[str, dict[str, Any]] = {}
        self._prop_instances: list[dict[str, Any]] = []
        self._foreground_instances: list[dict[str, Any]] = []
        self._effect_instances: list[dict[str, Any]] = []
        self._sky_prop_instances: list[dict[str, Any]] = []
        self._last_frame_signature: Any = None
        self._last_frame_rgb: Optional[bytes] = None
        self._build_lighting()
        self.base.render.setShaderOff()
        self.skybox_root.setShaderOff()
        self.skybox_root.setLightOff()
        self.stage_root.setShaderOff()
        self.stage_root.setLightOff()
        self.show_actor_labels = bool(story.get("video", {}).get("show_actor_labels", not self.fast_card_mode))

    def _load_text_font(self):
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                font = self.base.loader.loadFont(candidate)
                if font:
                    return font
        return None

    def _build_lighting(self) -> None:
        ambient = self.base.render.attachNewNode(self._core["AmbientLight"]("ambient"))
        ambient.node().setColor((1.0, 1.0, 1.0, 1.0))
        self.base.render.setLight(ambient)

    def _detach_children(self, node) -> None:
        node.getChildren().detach()

    def _normalized_rgba(self, value: Any, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if not isinstance(value, (list, tuple)):
            return default
        items = [float(item) for item in value[:4]]
        if any(item > 1.0 for item in items):
            items = [item / 255.0 for item in items]
        while len(items) < 4:
            items.append(default[len(items)])
        return tuple(items[:4])

    def _ease_ratio(self, ratio: float, ease_name: str) -> float:
        ratio = max(0.0, min(1.0, ratio))
        ease = str(ease_name or "linear").lower()
        if ease in {"inout", "ease-in-out"}:
            return 0.5 - math.cos(ratio * math.pi) * 0.5
        if ease in {"out", "ease-out"}:
            return 1.0 - (1.0 - ratio) * (1.0 - ratio)
        if ease in {"in", "ease-in"}:
            return ratio * ratio
        return ratio

    def _camera_state(self, scene: dict[str, Any], time_ms: int) -> dict[str, float]:
        camera = scene.get("camera") or {}
        duration_ms = max(1, int(scene.get("duration_ms", 1) or 1))
        ratio = self._ease_ratio(time_ms / duration_ms, str(camera.get("ease") or "linear"))
        x0 = float(camera.get("x", 0.0) or 0.0)
        z0 = float(camera.get("z", 0.0) or 0.0)
        zoom0 = float(camera.get("zoom", 1.0) or 1.0)
        x1 = float(camera.get("to_x", x0) or x0)
        z1 = float(camera.get("to_z", z0) or z0)
        zoom1 = float(camera.get("to_zoom", zoom0) or zoom0)
        return {
            "x": x0 + (x1 - x0) * ratio,
            "z": z0 + (z1 - z0) * ratio,
            "zoom": zoom0 + (zoom1 - zoom0) * ratio,
        }

    def _shape_texture(self, key: str, size: tuple[int, int], alpha: float = 1.0):
        cached = self._shape_texture_cache.get(key)
        if cached is not None:
            return cached
        width = max(32, int(size[0]))
        height = max(32, int(size[1]))
        image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((1, 1, width - 2, height - 2), fill=(255, 255, 255, max(1, int(255 * alpha))))
        path = self._texture_cache_dir / f"{key}.png"
        if not path.exists():
            image.save(path)
        texture = self.base.loader.loadTexture(str(path))
        self._shape_texture_cache[key] = texture
        return texture

    def _rounded_rect_texture(self, key: str, size: tuple[int, int], radius: int, alpha: float = 1.0):
        cached = self._shape_texture_cache.get(key)
        if cached is not None:
            return cached
        width = max(32, int(size[0]))
        height = max(32, int(size[1]))
        image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (1, 1, width - 2, height - 2),
            radius=max(2, int(radius)),
            fill=(255, 255, 255, max(1, int(255 * alpha))),
        )
        path = self._texture_cache_dir / f"{key}.png"
        if not path.exists():
            image.save(path)
        texture = self.base.loader.loadTexture(str(path))
        self._shape_texture_cache[key] = texture
        return texture

    def _active_beat(self, scene: dict[str, Any], actor_id: str, time_ms: int) -> Optional[dict[str, Any]]:
        for item in scene.get("beats", []):
            if str(item.get("actor_id") or "") != actor_id:
                continue
            if int(item.get("start_ms", -1)) <= time_ms <= int(item.get("end_ms", -1)):
                return item
        return None

    def _pose_track(self, path_value: str | None):
        path = str(path_value or "").strip()
        if not path:
            return None
        cached = self._pose_track_cache.get(path)
        if cached is not None:
            return cached
        track = poseviz._load_track(Path(path), width=max(960, self.frame_width), height=max(540, self.frame_height))
        self._pose_track_cache[path] = track
        return track

    @staticmethod
    def _pose_head_point(points: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
        markers = [points[name] for name in ("left_eye", "right_eye", "left_ear", "right_ear") if name in points]
        if len(markers) >= 2:
            return (
                sum(point[0] for point in markers) / len(markers),
                sum(point[1] for point in markers) / len(markers),
            )
        if "nose" in points:
            return points["nose"]
        return None

    def _pose_body_state(
        self,
        active_beat: dict[str, Any],
        time_ms: int,
        facing_sign: float,
    ) -> Optional[dict[str, Any]]:
        track = self._pose_track(active_beat.get("pose_track_path"))
        if track is None or track.total_duration_s <= 0.0:
            return None
        start_ms = int(active_beat.get("start_ms", 0) or 0)
        end_ms = max(start_ms + 1, int(active_beat.get("end_ms", start_ms + 1) or start_ms + 1))
        ratio = max(0.0, min(1.0, (time_ms - start_ms) / max(1.0, float(end_ms - start_ms))))
        pose_points = poseviz._sample_track(track, ratio * track.total_duration_s)
        if not pose_points:
            return None
        points = {name: (float(value[0]), float(value[1])) for name, value in pose_points.items()}
        if not {"left_shoulder", "right_shoulder", "left_hip", "right_hip"} <= set(points):
            return None
        left_shoulder = points["left_shoulder"]
        right_shoulder = points["right_shoulder"]
        left_hip = points["left_hip"]
        right_hip = points["right_hip"]
        shoulder_center_src = ((left_shoulder[0] + right_shoulder[0]) * 0.5, (left_shoulder[1] + right_shoulder[1]) * 0.5)
        hip_center_src = ((left_hip[0] + right_hip[0]) * 0.5, (left_hip[1] + right_hip[1]) * 0.5)
        torso_center_src = ((shoulder_center_src[0] + hip_center_src[0]) * 0.5, (shoulder_center_src[1] + hip_center_src[1]) * 0.5)
        torso_height = max(1.0, math.hypot(shoulder_center_src[0] - hip_center_src[0], shoulder_center_src[1] - hip_center_src[1]))
        scale = 0.64 / torso_height

        def _local(point: tuple[float, float]) -> tuple[float, float]:
            return (
                (point[0] - torso_center_src[0]) * scale * facing_sign,
                (hip_center_src[1] - point[1]) * scale + 0.70,
            )

        pelvis_center = _local(hip_center_src)
        chest_center = _local(shoulder_center_src)
        up_x = chest_center[0] - pelvis_center[0]
        up_z = chest_center[1] - pelvis_center[1]
        up_len = max(0.001, math.hypot(up_x, up_z))
        up_unit = (up_x / up_len, up_z / up_len)
        head_src = self._pose_head_point(points)
        if head_src is not None:
            head_center = _local(head_src)
        else:
            head_center = (chest_center[0] + up_unit[0] * 0.92, chest_center[1] + up_unit[1] * 0.92)
        min_head_gap = 0.56
        current_gap = (head_center[0] - chest_center[0]) * up_unit[0] + (head_center[1] - chest_center[1]) * up_unit[1]
        if current_gap < min_head_gap:
            head_center = (
                chest_center[0] + up_unit[0] * min_head_gap,
                chest_center[1] + up_unit[1] * min_head_gap,
            )
        neck_center = (
            chest_center[0] + (head_center[0] - chest_center[0]) * 0.42,
            chest_center[1] + (head_center[1] - chest_center[1]) * 0.42,
        )

        def _named(name: str, fallback: tuple[float, float]) -> tuple[float, float]:
            point = points.get(name)
            return _local(point) if point is not None else fallback

        def _stretch(anchor: tuple[float, float], point: tuple[float, float], ratio: float) -> tuple[float, float]:
            return (
                anchor[0] + (point[0] - anchor[0]) * ratio,
                anchor[1] + (point[1] - anchor[1]) * ratio,
            )

        shoulder_left = _named("left_shoulder", (chest_center[0] - 0.25, chest_center[1] + 0.22))
        shoulder_right = _named("right_shoulder", (chest_center[0] + 0.25, chest_center[1] + 0.22))
        hip_left = _named("left_hip", (pelvis_center[0] - 0.11, pelvis_center[1] - 0.02))
        hip_right = _named("right_hip", (pelvis_center[0] + 0.11, pelvis_center[1] - 0.02))
        elbow_left = _named("left_elbow", shoulder_left)
        elbow_right = _named("right_elbow", shoulder_right)
        hand_left = _named("left_wrist", elbow_left)
        hand_right = _named("right_wrist", elbow_right)
        knee_left = _named("left_knee", hip_left)
        knee_right = _named("right_knee", hip_right)
        foot_left = _named("left_ankle", knee_left)
        foot_right = _named("right_ankle", knee_right)
        elbow_left = _stretch(shoulder_left, elbow_left, 1.10)
        elbow_right = _stretch(shoulder_right, elbow_right, 1.10)
        hand_left = _stretch(elbow_left, hand_left, 1.16)
        hand_right = _stretch(elbow_right, hand_right, 1.16)
        knee_left = _stretch(hip_left, knee_left, 1.14)
        knee_right = _stretch(hip_right, knee_right, 1.14)
        foot_left = _stretch(knee_left, foot_left, 1.18)
        foot_right = _stretch(knee_right, foot_right, 1.18)
        head_center = _stretch(chest_center, head_center, 1.06)
        neck_center = (
            chest_center[0] + (head_center[0] - chest_center[0]) * 0.36,
            chest_center[1] + (head_center[1] - chest_center[1]) * 0.36,
        )
        torso_r = math.degrees(math.atan2(chest_center[0] - pelvis_center[0], chest_center[1] - pelvis_center[1]))
        head_r = math.degrees(math.atan2(head_center[0] - neck_center[0], head_center[1] - neck_center[1]))
        return {
            "pelvis_center": pelvis_center,
            "chest_center": chest_center,
            "neck_center": neck_center,
            "head_center": head_center,
            "torso_r": torso_r,
            "head_r": head_r,
            "shoulder_left": shoulder_left,
            "shoulder_right": shoulder_right,
            "hip_left": hip_left,
            "hip_right": hip_right,
            "elbow_left": elbow_left,
            "elbow_right": elbow_right,
            "hand_left": hand_left,
            "hand_right": hand_right,
            "knee_left": knee_left,
            "knee_right": knee_right,
            "foot_left": foot_left,
            "foot_right": foot_right,
        }

    @staticmethod
    def _static_body_state(talking: bool) -> dict[str, float]:
        talk_wave = math.sin(0.0) if not talking else 0.0
        return {
            "root_dx": 0.0,
            "jump": 0.0,
            "pelvis_x": 0.0,
            "pelvis_z": 0.70,
            "pelvis_r": 0.0,
            "chest_x": 0.0,
            "chest_z": 1.34,
            "chest_r": 0.0,
            "neck_x": 0.0,
            "neck_z": 1.97,
            "neck_r": 0.0,
            "head_x": 0.0,
            "head_z": 2.31,
            "head_r": talk_wave,
        }

    @staticmethod
    def _lerp(start: float, end: float, ratio: float) -> float:
        return start + (end - start) * max(0.0, min(1.0, ratio))

    @staticmethod
    def _ease_in_out(ratio: float) -> float:
        ratio = max(0.0, min(1.0, ratio))
        return 0.5 - 0.5 * math.cos(math.pi * ratio)

    def _vertical_gradient_texture(
        self,
        key: str,
        size: tuple[int, int],
        top_rgba: tuple[int, int, int, int],
        bottom_rgba: tuple[int, int, int, int],
    ):
        cached = self._gradient_texture_cache.get(key)
        if cached is not None:
            return cached
        width = max(8, int(size[0]))
        height = max(8, int(size[1]))
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        for row in range(height):
            ratio = row / max(1, height - 1)
            rgba = tuple(
                int(top_rgba[index] + (bottom_rgba[index] - top_rgba[index]) * ratio)
                for index in range(4)
            )
            draw.rectangle((0, row, width, row), fill=rgba)
        path = self._texture_cache_dir / f"{key}.png"
        if not path.exists():
            image.save(path)
        texture = self.base.loader.loadTexture(str(path))
        self._gradient_texture_cache[key] = texture
        return texture

    def _procedural_floor_texture(self, floor_id: str, base_color: tuple[float, float, float, float]) -> str:
        key = f"floor-proc-{floor_id}"
        cached = self._gradient_texture_cache.get(key)
        if cached is not None:
            return cached
        width, height = 512, 512
        r = int(max(0.0, min(1.0, base_color[0])) * 255)
        g = int(max(0.0, min(1.0, base_color[1])) * 255)
        b = int(max(0.0, min(1.0, base_color[2])) * 255)
        image = Image.new("RGBA", (width, height), (r, g, b, 255))
        draw = ImageDraw.Draw(image)
        floor_key = str(floor_id or "").strip().lower()
        if floor_key == "stone-court":
            block = 112
            mortar = (max(0, r - 42), max(0, g - 42), max(0, b - 42), 255)
            for y in range(0, height, block):
                offset = 0 if (y // block) % 2 == 0 else block // 2
                for x in range(-offset, width, block):
                    x0 = x + 4
                    y0 = y + 4
                    x1 = min(width - 4, x + block - 6)
                    y1 = min(height - 4, y + block - 6)
                    shade = ((x // block) + (y // block)) % 3
                    fill = (
                        min(255, r + shade * 8),
                        min(255, g + shade * 8),
                        min(255, b + shade * 8),
                        255,
                    )
                    draw.rounded_rectangle((x0, y0, x1, y1), radius=6, fill=fill, outline=mortar, width=3)
        elif floor_key == "wood-plank":
            plank_h = 112
            seam = (max(0, r - 55), max(0, g - 40), max(0, b - 24), 255)
            for y in range(0, height, plank_h):
                tone = ((y // plank_h) % 3) * 10
                fill = (min(255, r + tone), min(255, g + tone // 2), min(255, b), 255)
                draw.rectangle((0, y, width, y + plank_h - 2), fill=fill)
                draw.line((0, y, width, y), fill=seam, width=2)
                for x in range(0, width, 224):
                    draw.line((x + 26, y + 14, x + 40, y + plank_h - 16), fill=(seam[0], seam[1], seam[2], 72), width=1)
        else:
            stripe = (max(0, r - 10), max(0, g - 10), max(0, b - 10), 255)
            band_h = 160
            for y in range(0, height, band_h):
                draw.rectangle((0, y, width, y + band_h // 2), fill=(r, g, b, 255))
                draw.rectangle((0, y + band_h // 2, width, y + band_h), fill=stripe)
        path = self._texture_cache_dir / f"{key}.png"
        if not path.exists():
            image.save(path)
        self._gradient_texture_cache[key] = str(path)
        return str(path)

    def _make_card(self, width: float, height: float, name: str):
        cm = self._core["CardMaker"](name)
        cm.setFrame(-width / 2.0, width / 2.0, -height / 2.0, height / 2.0)
        return cm.generate()

    def _attach_card(
        self,
        parent,
        width: float,
        height: float,
        name: str,
        pos: tuple[float, float, float],
        color: tuple[float, ...],
    ):
        node = parent.attachNewNode(self._make_card(width, height, name))
        node.setPos(*pos)
        node.setColor(*tuple(color))
        node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        node.setTwoSided(True)
        return node

    def _place_segment_card(
        self,
        node,
        start: tuple[float, float],
        end: tuple[float, float],
        width: float,
        y: Optional[float] = None,
    ) -> None:
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length = max(0.001, math.hypot(dx, dz))
        angle = math.degrees(math.atan2(dx, dz))
        node.setPos((start[0] + end[0]) / 2.0, node.getY() if y is None else y, (start[1] + end[1]) / 2.0)
        node.setScale(width, 1.0, length)
        node.setR(angle)

    def _place_joint_card(self, node, point: tuple[float, float], diameter: float, y: Optional[float] = None) -> None:
        node.setPos(point[0], node.getY() if y is None else y, point[1])
        node.setScale(diameter, 1.0, diameter)
        node.setR(0.0)

    def _apply_texture(self, node, texture) -> None:
        if texture is None:
            node.clearTexture()
            return
        node.setTexture(texture, 1)
        node.setTransparency(self._core["TransparencyAttrib"].MAlpha)

    def _load_texture_sequence(self, path_value: str | None) -> dict[str, Any]:
        path = str(path_value or "").strip()
        if not path:
            return {"frames": [], "durations": []}
        normalized_path = path.replace("\\", "/")
        is_prop_asset = "/assets/props/" in normalized_path
        is_character_asset = "/assets/characters/" in normalized_path
        is_face_asset = "/skins/face_" in normalized_path
        try:
            stat = Path(path).stat()
            bg_version = PROP_WHITE_BG_VERSION if (is_prop_asset or is_character_asset) else 0
            face_crop_version = 13 if is_face_asset else 0
            cache_key = f"{path}|{stat.st_size}|{int(stat.st_mtime_ns)}|propbgv={bg_version}|facecrop={face_crop_version}"
        except OSError:
            cache_key = path
        cached = self._texture_sequence_cache.get(cache_key)
        if cached is not None:
            return cached
        frames: list[Any] = []
        durations: list[int] = []
        suffix = Path(path).suffix.lower()
        static_suffixes = {".png", ".jpg", ".jpeg"}
        if suffix in static_suffixes:
            digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
            try:
                with Image.open(path) as image:
                    rgba = image.convert("RGBA")
                    if is_prop_asset or is_character_asset:
                        rgba = self._remove_white_prop_background(rgba)
                    if is_face_asset:
                        rgba = self._crop_visible_face_region(rgba)
                    frame_path = self._texture_cache_dir / f"{digest}-0.png"
                    if not frame_path.exists():
                        rgba.save(frame_path)
                    texture = self.base.loader.loadTexture(self._core["Filename"].fromOsSpecific(str(frame_path)))
                    if texture:
                        frames = [texture]
                        durations = [100]
            except Exception:
                texture = self.base.loader.loadTexture(self._core["Filename"].fromOsSpecific(path))
                if texture:
                    frames = [texture]
                    durations = [100]
        else:
            digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
            try:
                with Image.open(path) as image:
                    frame_count = max(1, int(getattr(image, "n_frames", 1)))
                    for index in range(frame_count):
                        image.seek(index)
                        durations.append(max(20, int(image.info.get("duration", 0) or 100)))
                        rgba = image.convert("RGBA")
                        if is_prop_asset or is_character_asset:
                            rgba = self._remove_white_prop_background(rgba)
                        if is_face_asset:
                            rgba = self._crop_visible_face_region(rgba)
                        frame_path = self._texture_cache_dir / f"{digest}-{index}.png"
                        if not frame_path.exists():
                            rgba.save(frame_path)
                        texture = self.base.loader.loadTexture(self._core["Filename"].fromOsSpecific(str(frame_path)))
                        if texture:
                            frames.append(texture)
            except Exception:
                texture = self.base.loader.loadTexture(self._core["Filename"].fromOsSpecific(path))
                if texture:
                    frames = [texture]
                    durations = [100]
        payload = {"frames": frames, "durations": durations[: len(frames)]}
        self._texture_sequence_cache[cache_key] = payload
        self._frame_duration_cache[path] = payload["durations"]
        return payload

    @staticmethod
    def _crop_visible_face_region(image: Image.Image) -> Image.Image:
        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is None:
            return image
        left, top, right, bottom = bbox
        width = right - left
        height = bottom - top
        pad_left_right = int(round(width * 0.10))
        pad_top = int(round(height * 0.06))
        pad_bottom = int(round(height * 0.18))
        x0 = left - pad_left_right
        y0 = top - pad_top
        x1 = right + pad_left_right
        y1 = bottom + pad_bottom
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if x1 > image.width:
            shift = x1 - image.width
            x0 = max(0, x0 - shift)
            x1 = image.width
        if y1 > image.height:
            shift = y1 - image.height
            y0 = max(0, y0 - shift)
            y1 = image.height
        cropped = image.crop((x0, y0, x1, y1)).convert("RGBA")
        target_size = (240, 184)
        scale = min(
            target_size[0] / max(1, cropped.width),
            target_size[1] / max(1, cropped.height),
        )
        resized_width = max(1, int(round(cropped.width * scale)))
        resized_height = max(1, int(round(cropped.height * scale)))
        resized = cropped.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        composed = Image.new("RGBA", target_size, (0, 0, 0, 0))
        paste_x = (target_size[0] - resized_width) // 2
        paste_y = max(0, int(round((target_size[1] - resized_height) * 0.10)))
        composed.alpha_composite(resized, (paste_x, paste_y))
        mask = Image.new("L", target_size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, max(0, target_size[0] - 1), max(0, target_size[1] - 1)), fill=255)
        composed_alpha = composed.getchannel("A")
        composed.putalpha(ImageChops.multiply(composed_alpha, mask))
        return composed

    @staticmethod
    def _is_near_white(pixel: tuple[int, int, int, int]) -> bool:
        r, g, b, a = pixel
        return a >= 200 and r >= 245 and g >= 245 and b >= 245

    def _remove_white_prop_background(self, image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        width, height = rgba.size
        if width <= 0 or height <= 0:
            return rgba
        pixels = rgba.load()
        visited = set()
        queue = deque()

        def try_seed(x: int, y: int) -> None:
            if (x, y) in visited:
                return
            if self._is_near_white(pixels[x, y]):
                visited.add((x, y))
                queue.append((x, y))

        for x in range(width):
            try_seed(x, 0)
            try_seed(x, height - 1)
        for y in range(height):
            try_seed(0, y)
            try_seed(width - 1, y)

        while queue:
            x, y = queue.popleft()
            pixels[x, y] = (255, 255, 255, 0)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if (nx, ny) in visited:
                    continue
                if not self._is_near_white(pixels[nx, ny]):
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
        return rgba

    def _animation_timeline_ms(self, path_value: str | None, frame_count: int, period_ms: int | None = None) -> list[int]:
        if frame_count <= 0:
            return []
        durations = list(self._frame_duration_cache.get(str(path_value or ""), []))
        if len(durations) < frame_count:
            durations.extend([100] * (frame_count - len(durations)))
        durations = durations[:frame_count]
        total_native_ms = sum(durations)
        if total_native_ms <= 0:
            return [100] * frame_count
        if period_ms is None:
            return durations
        target_total_ms = max(1, int(period_ms))
        if target_total_ms == total_native_ms:
            return durations
        scaled: list[int] = []
        accumulated = 0
        for index, duration_ms in enumerate(durations):
            if index == frame_count - 1:
                scaled_duration = max(1, target_total_ms - accumulated)
            else:
                scaled_duration = max(1, int(round(duration_ms / total_native_ms * target_total_ms)))
                remaining_min = frame_count - index - 1
                scaled_duration = min(scaled_duration, max(1, target_total_ms - accumulated - remaining_min))
            scaled.append(scaled_duration)
            accumulated += scaled_duration
        return scaled

    @staticmethod
    def _timeline_frame_index(durations: list[int], position_ms: float) -> int:
        if not durations:
            return 0
        elapsed = 0.0
        for index, duration_ms in enumerate(durations):
            elapsed += max(1, duration_ms)
            if position_ms < elapsed:
                return index
        return len(durations) - 1

    def _texture_at_time(self, image_path: Optional[str], time_ms: int, *, period_ms: int | None = None):
        path = str(image_path or "").strip()
        if not path:
            return None
        cached = self._load_texture_sequence(path)
        frames = cached.get("frames", [])
        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        durations = self._animation_timeline_ms(path, len(frames), period_ms=period_ms)
        timeline_total_ms = max(1, sum(durations) if durations else int(period_ms or 900))
        position_ms = int(time_ms) % timeline_total_ms
        fallback = [max(1, timeline_total_ms // max(1, len(frames)))] * len(frames)
        frame_index = self._timeline_frame_index(durations or fallback, position_ms)
        return frames[frame_index % len(frames)]

    def _texture_at_progress(
        self,
        image_path: Optional[str],
        progress: float,
        *,
        playback_speed: float = 1.0,
        period_ms: int | None = None,
    ):
        path = str(image_path or "").strip()
        if not path:
            return None
        cached = self._load_texture_sequence(path)
        frames = cached.get("frames", [])
        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        durations = self._animation_timeline_ms(path, len(frames), period_ms=period_ms)
        timeline_total_ms = max(1, sum(durations) if durations else int(period_ms or 900))
        cycle_count = max(1.0, float(playback_speed or 1.0))
        adjusted_progress = max(0.0, min(1.0, float(progress))) * cycle_count
        position_ms = min(timeline_total_ms - 1, adjusted_progress * timeline_total_ms)
        fallback = [max(1, timeline_total_ms // max(1, len(frames)))] * len(frames)
        frame_index = self._timeline_frame_index(durations or fallback, position_ms)
        return frames[frame_index % len(frames)]

    @staticmethod
    def _clamp_ratio(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _effect_alpha_ratio(self, alpha_ratio: float) -> float:
        clamped = self._clamp_ratio(alpha_ratio)
        boosted = max(0.88, clamped)
        alpha_value = int(255 * boosted)
        alpha_value = max(MIN_EFFECT_ALPHA, min(MAX_EFFECT_ALPHA, alpha_value))
        return alpha_value / 255.0

    def _set_actor_layer(self, node, sort: int) -> None:
        node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        node.setDepthWrite(False)
        node.setDepthTest(False)
        node.setBin("fixed", sort)

    def _attach_overlay_card(
        self,
        parent,
        name: str,
        width: float,
        height: float,
        pos: tuple[float, float],
        texture_path: Optional[str] = None,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ):
        aspect = self.frame_width / max(1.0, float(self.frame_height))
        node = parent.attachNewNode(self._make_card(width * 2.0 * aspect, height * 2.0, name))
        node.setPos(pos[0] * 2.0 * aspect, 0.0, pos[1] * 2.0)
        node.setColor(*color)
        node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        if texture_path:
            self._apply_texture(node, self._texture_at_time(texture_path, 0))
        return node

    def _resolve_effect_asset_path(self, item: dict[str, Any]) -> Optional[str]:
        raw = str(item.get("asset_path") or "").strip()
        if raw:
            return raw
        effect_id = str(item.get("type") or "").strip()
        meta = self.effects.get(effect_id, {})
        return str(meta.get("asset_path") or "").strip() or None

    def _resolve_foreground_asset_path(self, item: dict[str, Any]) -> Optional[str]:
        raw = str(item.get("asset_path") or "").strip()
        if raw:
            return raw
        foreground_id = str(item.get("foreground_id") or "").strip()
        meta = self.foregrounds.get(foreground_id, {})
        return str(meta.get("asset_path") or "").strip() or None

    def _character_skin_path(self, character: dict[str, Any], stem_candidates: list[str]) -> Optional[str]:
        search_dirs = []
        if character.get("character_dir"):
            search_dirs.append(Path(character["character_dir"]) / "skins")
        search_dirs.append(self._shared_skins_dir)
        for directory in search_dirs:
            if not directory.exists():
                continue
            for stem in stem_candidates:
                for suffix in (".png", ".webp", ".gif", ".jpg", ".jpeg"):
                    candidate = directory / f"{stem}{suffix}"
                    if candidate.exists():
                        return str(candidate)
        return None

    def _face_skin_path(self, character: dict[str, Any], expression: str, time_ms: int, *, talking: bool = False) -> Optional[str]:
        normalized = str(expression or "default").strip().lower().replace("-", "_")
        mouth_open = (time_ms // 160) % 2 == 0
        talk_suffix = "open" if mouth_open else "closed"
        candidates: list[str] = []
        if talking:
            candidates.extend(
                [
                    f"face_talk_{normalized}_{talk_suffix}",
                    f"face_talk_neutral_{talk_suffix}",
                    f"face_talk_smile_{talk_suffix}",
                    f"face_talk_angry_{talk_suffix}",
                    f"face_{normalized}_{talk_suffix}",
                    f"face_neutral_{talk_suffix}",
                ]
            )
        if normalized == "default":
            candidates.extend(["face_default", "face_neutral"])
        else:
            candidates.extend([f"face_{normalized}", "face_default", "face_neutral"])
        return self._character_skin_path(character, candidates)

    def _outfit_skin_path(self, character: dict[str, Any]) -> Optional[str]:
        outfit_style = str(character.get("outfit_style") or "").strip()
        garment = str(character.get("garment") or "").strip()
        candidates = ["outfit"]
        if outfit_style:
            candidates.append(f"outfit_{outfit_style}")
        if garment:
            candidates.append(f"outfit_{garment}")
        candidates.append("outfit_default")
        return self._character_skin_path(character, candidates)

    def _attach_box(
        self,
        parent,
        name: str,
        sx: float,
        sy: float,
        sz: float,
        pos: tuple[float, float, float],
        color: tuple[float, float, float, float],
        texture_path: Optional[str] = None,
    ):
        node = self._box_model.copyTo(parent)
        node.setName(name)
        node.setScale(sx, sy, sz)
        node.setPos(*pos)
        node.setColor(*color)
        if texture_path:
            texture = self.base.loader.loadTexture(self._core["Filename"].fromOsSpecific(str(texture_path)))
            if texture:
                node.setTexture(texture, 1)
        return node

    def _attach_plane(
        self,
        parent,
        name: str,
        width: float,
        height: float,
        pos: tuple[float, float, float],
        hpr: tuple[float, float, float],
        texture_path: Optional[str],
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ):
        node = self._attach_card(parent, width, height, name, pos, color)
        node.setHpr(*hpr)
        node.setTwoSided(True)
        if texture_path:
            texture = self.base.loader.loadTexture(self._core["Filename"].fromOsSpecific(str(texture_path)))
            if texture:
                node.setTexture(texture, 1)
        return node

    def _resolve_surface_path(self, scene: dict[str, Any], key: str) -> Optional[str]:
        raw = str((scene.get("box") or {}).get(key) or "").strip()
        if not raw:
            return None
        cached = _cached_remote_asset(raw, "true3d")
        return str(cached) if cached else None

    def _wall_openings(self, scene: dict[str, Any], surface: str) -> list[dict[str, float]]:
        items: list[dict[str, float]] = []
        for item in scene.get("props", []):
            prop = self.props.get(item.get("prop_id"))
            if not prop:
                continue
            mount = str(item.get("mount") or prop.get("default_mount") or "free")
            render_style = str(prop.get("render_style") or "")
            if mount != surface or render_style not in {"window", "door", "double-door"}:
                continue
            scale = float(item.get("scale", 1.0) or 1.0)
            width = float(prop.get("width") or 0.0) or float(prop.get("base_width", 160)) / 140.0
            height = float(prop.get("height") or 0.0) or float(prop.get("base_height", 120)) / 140.0
            items.append(
                {
                    "x": float(item.get("x", 0.0) or 0.0),
                    "z": float(item.get("z", 0.0) or 0.0),
                    "width": width * scale,
                    "height": height * scale,
                }
            )
        return items

    def _make_wall_segments(
        self,
        parent,
        name: str,
        span: float,
        height: float,
        thickness: float,
        color: tuple[float, float, float, float],
        openings: list[dict[str, float]],
        texture_path: Optional[str],
    ):
        root = parent.attachNewNode(name)
        half_span = span / 2.0
        half_height = height / 2.0
        rects: list[tuple[float, float, float, float]] = []
        x_cuts = {-half_span, half_span}
        z_cuts = {-half_height, half_height}
        for opening in openings:
            ox = max(-half_span + 0.2, min(half_span - 0.2, float(opening["x"])))
            oz = max(-half_height + 0.2, min(half_height - 0.2, float(opening["z"])))
            ow = max(0.2, min(span - 0.4, float(opening["width"])))
            oh = max(0.2, min(height - 0.4, float(opening["height"])))
            left = ox - ow / 2.0
            right = ox + ow / 2.0
            bottom = oz - oh / 2.0
            top = oz + oh / 2.0
            rects.append((left, right, bottom, top))
            x_cuts.update({left, right})
            z_cuts.update({bottom, top})
        xs = sorted(x_cuts)
        zs = sorted(z_cuts)
        for x0, x1 in zip(xs, xs[1:]):
            for z0, z1 in zip(zs, zs[1:]):
                cx = (x0 + x1) / 2.0
                cz = (z0 + z1) / 2.0
                if any(left <= cx <= right and bottom <= cz <= top for left, right, bottom, top in rects):
                    continue
                seg_w = x1 - x0
                seg_h = z1 - z0
                if seg_w <= 0.01 or seg_h <= 0.01:
                    continue
                self._attach_box(root, f"{name}-{cx:.2f}-{cz:.2f}", seg_w, thickness, seg_h, (cx, 0.0, cz), color, texture_path=texture_path)
        return root

    def _attach_room_plane(
        self,
        parent,
        name: str,
        width: float,
        height: float,
        pos: tuple[float, float, float],
        hpr: tuple[float, float, float],
        color: tuple[float, float, float, float],
        texture_path: Optional[str],
    ):
        node = self._attach_plane(parent, name, width, height, pos, hpr, texture_path, color=color)
        node.setTwoSided(True)
        return node

    @staticmethod
    def _is_sky_prop(item: dict[str, Any], prop: dict[str, Any]) -> bool:
        mount = str(item.get("mount") or prop.get("default_mount") or "").strip().lower()
        category = str(item.get("category") or prop.get("category") or "").strip().lower()
        return mount == "sky" or category in {"sky", "celestial"}

    @staticmethod
    def _prop_world_scale(prop_id: str, prop: dict[str, Any]) -> tuple[float, float]:
        category = str(prop.get("category") or "").strip().lower()
        pid = str(prop_id or "").strip().lower()
        explicit_scales: dict[str, tuple[float, float]] = {
            "airplane": (6.80, 3.20),
            "cat": (0.72, 0.58),
            "cow": (3.70, 2.95),
            "dog": (1.35, 1.05),
            "donkey": (2.95, 2.35),
            "horse": (4.80, 3.85),
            "house": (9.60, 10.40),
            "lantern": (0.52, 0.76),
            "panda": (1.18, 1.02),
            "rabbit": (0.62, 0.52),
            "star": (1.00, 1.00),
            "moon": (1.00, 1.00),
            "t-rex": (8.20, 7.00),
            "tiger": (6.20, 4.80),
            "training-drum": (1.35, 1.45),
            "wall-door": (2.70, 3.40),
            "wall-double-door": (3.70, 3.40),
            "wall-window": (2.50, 2.20),
            "weapon-rack": (1.60, 1.95),
        }
        if pid in explicit_scales:
            return explicit_scales[pid]
        if category == "building":
            return 8.4, 9.2
        if category == "architecture":
            return 2.6, 3.0
        if category == "vehicle":
            return 5.2, 3.0
        if category in {"sky", "celestial"}:
            return 1.0, 1.0
        if category == "animal":
            return 2.2, 1.8
        return 1.0, 1.0

    def _attach_skybox(
        self,
        texture_path: Optional[str],
        color: tuple[float, float, float, float],
        *,
        bottom_texture_path: Optional[str] = None,
        ground_color: Optional[tuple[float, float, float, float]] = None,
    ) -> None:
        self._detach_children(self.skybox_root)
        sky_radius = 300.0
        dome_height = sky_radius * 0.88
        ground_span = sky_radius * 3.2
        horizon_radius = sky_radius * 0.92
        horizon_band_height = sky_radius * 0.26
        horizon_base_z = -sky_radius * 0.16
        land_tint = ground_color or (0.42, 0.38, 0.32, 1.0)

        def _send_to_back(node, sort: int) -> None:
            node.setDepthWrite(False)
            node.setDepthTest(False)
            node.setBin("background", sort)
            node.setTwoSided(True)

        backdrop = self._attach_plane(
            self.skybox_root,
            "sky-backdrop",
            sky_radius * 2.2,
            sky_radius * 1.18,
            (0.0, horizon_radius * 1.02, sky_radius * 0.04),
            (180.0, 0.0, 0.0),
            texture_path,
            (1.0, 1.0, 1.0, 0.96),
        )
        _send_to_back(backdrop, -103)

        dome_segments = 6 if self.extreme_speed_mode else 10
        for index in range(dome_segments):
            angle_deg = index * (360.0 / dome_segments)
            angle = math.radians(angle_deg)
            x = math.sin(angle) * horizon_radius
            y = math.cos(angle) * horizon_radius
            segment = self._attach_plane(
                self.skybox_root,
                f"sky-dome-{index}",
                sky_radius * 1.05,
                dome_height,
                (x, y, sky_radius * 0.12),
                (180.0 - angle_deg, -12.0, 0.0),
                None,
                color,
            )
            _send_to_back(segment, -100)

        top_cap = self._attach_plane(
            self.skybox_root,
            "sky-top",
            sky_radius * 2.1,
            sky_radius * 2.1,
            (0.0, 0.0, sky_radius * 0.86),
            (0.0, -90.0, 0.0),
            None,
            color,
        )
        _send_to_back(top_cap, -101)

        if not self.extreme_speed_mode:
            fog_texture = self._vertical_gradient_texture(
                "horizon-fog",
                (32, 256),
                (220, 226, 232, 0),
                (176, 184, 190, 180),
            )
            fog_band = self._attach_plane(
                self.skybox_root,
                "horizon-fog",
                sky_radius * 2.55,
                horizon_band_height * 1.12,
                (0.0, horizon_radius * 1.01, horizon_base_z - sky_radius * 0.08),
                (180.0, 0.0, 0.0),
                None,
                (1.0, 1.0, 1.0, 0.92),
            )
            self._apply_texture(fog_band, fog_texture)
            fog_band.setTransparency(self._core["TransparencyAttrib"].MAlpha)
            _send_to_back(fog_band, -96)

    def _active_expression(self, scene: dict[str, Any], actor_id: str, time_ms: int) -> Optional[str]:
        for item in scene.get("expressions", []):
            if str(item.get("actor_id") or "") != actor_id:
                continue
            if int(item.get("start_ms", -1)) <= time_ms <= int(item.get("end_ms", -1)):
                return str(item.get("expression") or "").strip() or None
        return None

    def _is_actor_talking(self, scene: dict[str, Any], actor_id: str, time_ms: int) -> bool:
        for dialogue in scene.get("dialogues", []):
            if str(dialogue.get("speaker_id") or "") != actor_id:
                continue
            if int(dialogue.get("start_ms", -1)) <= time_ms <= int(dialogue.get("end_ms", -1)):
                return True
        return False

    def _expression_for_actor(self, scene: dict[str, Any], actor_id: str, time_ms: int) -> str:
        explicit = self._active_expression(scene, actor_id, time_ms)
        if explicit:
            return explicit
        return "neutral"

    def _prepare_scene(self, scene: dict[str, Any]) -> None:
        scene_id = str(scene.get("id") or "")
        if self._prepared_scene_id == scene_id:
            return
        self._prepared_scene_id = scene_id
        self._current_scene = scene
        self._detach_children(self.outside_root)
        self._detach_children(self.room_root)
        self._detach_children(self.prop_root)
        self._detach_children(self.actor_root)
        self._detach_children(self.effect_root)
        self._detach_children(self.foreground_root)
        self._detach_children(self.label_root)
        self._prop_instances = []
        self._foreground_instances = []
        self._effect_instances = []
        self._sky_prop_instances = []

        background = self.backgrounds[scene["background"]]
        floor_id = scene.get("floor") or background.get("floor_id")
        floor = self.floors.get(floor_id) if floor_id else None
        box = scene.get("box") or {}
        room_width = float(box.get("width", 12.0) or 12.0)
        room_height = float(box.get("height", 7.0) or 7.0)
        room_depth = float(box.get("depth", 7.0) or 7.0)
        wall_thickness = 0.18
        floor_thickness = 0.18
        self._room_dims = {
            "width": room_width,
            "height": room_height,
            "depth": room_depth,
            "wall_thickness": wall_thickness,
            "floor_thickness": floor_thickness,
        }

        default_sky = tuple(background.get("sky_color") or (0.72, 0.78, 0.84, 1.0))
        default_accent = tuple(background.get("accent_color") or default_sky)
        default_ground = tuple((floor or {}).get("color") or background.get("ground_color") or (0.42, 0.38, 0.32, 1.0))
        back_color = self._normalized_rgba(box.get("back_wall_color") or box.get("wall_color"), default_sky)
        side_color = self._normalized_rgba(box.get("left_wall_color") or box.get("wall_color"), default_accent)
        right_color = self._normalized_rgba(box.get("right_wall_color") or box.get("wall_color"), default_accent)
        floor_color = self._normalized_rgba(box.get("floor_color"), default_ground)
        ceiling_color = self._normalized_rgba(box.get("ceiling_color"), default_accent)

        wall_texture = self._resolve_surface_path(scene, "wall_image_url")
        back_texture = self._resolve_surface_path(scene, "back_wall_image_url") or wall_texture
        left_texture = self._resolve_surface_path(scene, "left_wall_image_url") or wall_texture
        right_texture = self._resolve_surface_path(scene, "right_wall_image_url") or wall_texture
        floor_texture = self._resolve_surface_path(scene, "floor_image_url") or str((floor or {}).get("asset_path") or "").strip() or None
        if not floor_texture and floor_id:
            floor_texture = self._procedural_floor_texture(str(floor_id), floor_color)
        ceiling_texture = self._resolve_surface_path(scene, "ceiling_image_url")
        outside_back = self._resolve_surface_path(scene, "outside_back_image_url") or self._resolve_surface_path(scene, "outside_image_url")
        outside_left = self._resolve_surface_path(scene, "outside_left_image_url") or self._resolve_surface_path(scene, "outside_image_url")
        outside_right = self._resolve_surface_path(scene, "outside_right_image_url") or self._resolve_surface_path(scene, "outside_image_url")
        background_texture = str(background.get("asset_path") or "").strip() or None
        if not back_texture and background_texture:
            back_texture = background_texture
        if self.fast_card_mode:
            sky_texture = background_texture or back_texture or outside_back or outside_left or outside_right
            self._attach_skybox(
                sky_texture,
                back_color,
                bottom_texture_path=floor_texture,
                ground_color=floor_color,
            )

        wall_span = room_depth * 1.18
        back_openings = self._wall_openings(scene, "back-wall")
        left_openings = self._wall_openings(scene, "left-wall")
        right_openings = self._wall_openings(scene, "right-wall")

        has_openings = bool(back_openings or left_openings or right_openings)
        if self.fast_card_mode:
            local_ground = self._attach_room_plane(
                self.room_root,
                "local-ground",
                room_width * 18.0,
                room_depth * 18.0,
                (0.0, 0.0, -room_height / 2.0 - 1.05),
                (0.0, 90.0, 0.0),
                floor_color,
                floor_texture,
            )
            if floor_texture:
                local_ground.setTexScale(self._core["TextureStage"].getDefault(), 2.0, 2.0)
        elif has_openings:
            if outside_back:
                self._attach_plane(self.outside_root, "outside-back", room_width + 8.0, room_height + 4.0, (0.0, room_depth / 2.0 + 4.4, 0.0), (0.0, 0.0, 0.0), outside_back)
            if outside_left:
                self._attach_plane(self.outside_root, "outside-left", room_depth + 6.0, room_height + 4.0, (-room_width / 2.0 - 4.2, 0.0, 0.0), (90.0, 0.0, 0.0), outside_left)
            if outside_right:
                self._attach_plane(self.outside_root, "outside-right", room_depth + 6.0, room_height + 4.0, (room_width / 2.0 + 4.2, 0.0, 0.0), (-90.0, 0.0, 0.0), outside_right)

            self._attach_box(self.room_root, "floor", room_width, room_depth, floor_thickness, (0.0, 0.0, -room_height / 2.0), floor_color, texture_path=floor_texture)
            self._attach_box(self.room_root, "ceiling", room_width, room_depth, floor_thickness * 0.85, (0.0, 0.0, room_height / 2.0), ceiling_color, texture_path=ceiling_texture)

            back_root = self.room_root.attachNewNode("back-wall-root")
            back_root.setPos(0.0, room_depth / 2.0, 0.0)
            self._make_wall_segments(back_root, "back-wall", room_width, room_height, wall_thickness, back_color, back_openings, back_texture)

            left_root = self.room_root.attachNewNode("left-wall-root")
            left_root.setPos(-room_width / 2.0, 0.0, 0.0)
            left_root.setH(90.0)
            self._make_wall_segments(left_root, "left-wall", wall_span, room_height, wall_thickness, side_color, left_openings, left_texture)

            right_root = self.room_root.attachNewNode("right-wall-root")
            right_root.setPos(room_width / 2.0, 0.0, 0.0)
            right_root.setH(-90.0)
            self._make_wall_segments(right_root, "right-wall", wall_span, room_height, wall_thickness, right_color, right_openings, right_texture)
        else:
            self._attach_room_plane(self.room_root, "floor", room_width, room_depth, (0.0, 0.0, -room_height / 2.0 + 0.01), (0.0, 90.0, 0.0), floor_color, floor_texture)
            self._attach_room_plane(self.room_root, "ceiling", room_width, room_depth, (0.0, 0.0, room_height / 2.0 - 0.01), (0.0, -90.0, 0.0), ceiling_color, ceiling_texture)
            self._attach_room_plane(self.room_root, "back-wall", room_width, room_height, (0.0, room_depth / 2.0 - 0.01, 0.0), (180.0, 0.0, 0.0), back_color, back_texture)
            self._attach_room_plane(self.room_root, "left-wall", room_depth, room_height, (-room_width / 2.0 + 0.01, 0.0, 0.0), (90.0, 0.0, 0.0), side_color, left_texture)
            self._attach_room_plane(self.room_root, "right-wall", room_depth, room_height, (room_width / 2.0 - 0.01, 0.0, 0.0), (-90.0, 0.0, 0.0), right_color, right_texture)

        for item in scene.get("props", []):
            prop = self.props.get(item.get("prop_id"))
            if not prop:
                continue
            mount = str(item.get("mount") or prop.get("default_mount") or "free")
            render_style = str(prop.get("render_style") or "")
            if mount in {"back-wall", "left-wall", "right-wall"} and render_style in {"window", "door", "double-door"}:
                continue
            self._attach_simple_prop(item, prop)

        self._actor_instances = {}
        for actor in scene.get("actors", []):
            actor_id = str(actor["actor_id"])
            actor_root = self.actor_root.attachNewNode(f"actor-{actor_id}")
            if self.fast_card_mode:
                actor_root.setBillboardAxis()
            else:
                actor_root.setBillboardPointEye()
            upper_arm_left = self._attach_card(actor_root, 1.0, 1.0, f"upper-arm-left-{actor_id}", (-0.27, -0.03, 1.18), (1.0, 1.0, 1.0, 1.0))
            lower_arm_left = self._attach_card(actor_root, 1.0, 1.0, f"lower-arm-left-{actor_id}", (-0.34, -0.03, 0.78), (1.0, 1.0, 1.0, 1.0))
            upper_arm_right = self._attach_card(actor_root, 1.0, 1.0, f"upper-arm-right-{actor_id}", (0.27, -0.03, 1.18), (1.0, 1.0, 1.0, 1.0))
            lower_arm_right = self._attach_card(actor_root, 1.0, 1.0, f"lower-arm-right-{actor_id}", (0.34, -0.03, 0.78), (1.0, 1.0, 1.0, 1.0))
            upper_leg_left = self._attach_card(actor_root, 1.0, 1.0, f"upper-leg-left-{actor_id}", (-0.11, -0.04, 0.40), (1.0, 1.0, 1.0, 1.0))
            lower_leg_left = self._attach_card(actor_root, 1.0, 1.0, f"lower-leg-left-{actor_id}", (-0.11, -0.04, -0.16), (1.0, 1.0, 1.0, 1.0))
            upper_leg_right = self._attach_card(actor_root, 1.0, 1.0, f"upper-leg-right-{actor_id}", (0.11, -0.04, 0.40), (1.0, 1.0, 1.0, 1.0))
            lower_leg_right = self._attach_card(actor_root, 1.0, 1.0, f"lower-leg-right-{actor_id}", (0.11, -0.04, -0.16), (1.0, 1.0, 1.0, 1.0))
            pelvis_card = self._attach_card(actor_root, 0.34, 0.34, f"pelvis-{actor_id}", (0.0, 0.0, 0.78), (1.0, 1.0, 1.0, 0.0))
            body_card = self._attach_card(actor_root, 0.64, 2.24, f"body-{actor_id}", (0.0, 0.0, 1.42), (1.0, 1.0, 1.0, 1.0))
            neck_card = self._attach_card(actor_root, 0.06, 0.38, f"neck-{actor_id}", (0.0, 0.0, 2.03), (1.0, 1.0, 1.0, 1.0))
            head_base = self._attach_card(actor_root, 0.98, 0.82, f"head-base-{actor_id}", (0.0, 0.01, 2.08), (1.0, 1.0, 1.0, 1.0))
            self._apply_texture(body_card, self._rounded_rect_texture(f"body-{actor_id}", (182, 448), 78, alpha=1.0))
            self._apply_texture(pelvis_card, self._rounded_rect_texture(f"pelvis-{actor_id}", (32, 32), 16, alpha=0.0))
            self._apply_texture(neck_card, self._rounded_rect_texture(f"neck-{actor_id}", (34, 112), 16, alpha=1.0))
            self._apply_texture(head_base, self._shape_texture(f"head-{actor_id}", (240, 184), alpha=1.0))
            face_card = self._attach_card(actor_root, 0.98, 0.82, f"face-{actor_id}", (0.0, 0.03, 2.08), (1.0, 1.0, 1.0, 1.0))
            ear_left = self._attach_card(actor_root, 0.42, 0.42, f"ear-left-{actor_id}", (-0.27, -0.02, 2.40), (0.08, 0.08, 0.08, 1.0))
            ear_right = self._attach_card(actor_root, 0.42, 0.42, f"ear-right-{actor_id}", (0.27, -0.02, 2.40), (0.08, 0.08, 0.08, 1.0))
            joints = {}
            for name, pos, size in (
                ("shoulder_left", (-0.21, -0.01, 1.42), 0.12),
                ("shoulder_right", (0.21, -0.01, 1.42), 0.12),
                ("elbow_left", (-0.30, -0.01, 0.98), 0.10),
                ("elbow_right", (0.30, -0.01, 0.98), 0.10),
                ("hip_left", (-0.11, -0.01, 0.78), 0.13),
                ("hip_right", (0.11, -0.01, 0.78), 0.13),
                ("knee_left", (-0.11, -0.01, 0.08), 0.10),
                ("knee_right", (0.11, -0.01, 0.08), 0.10),
            ):
                joint = self._attach_card(actor_root, 1.0, 1.0, f"{name}-{actor_id}", pos, (1.0, 1.0, 1.0, 1.0))
                self._apply_texture(joint, self._shape_texture(f"{name}-{actor_id}", (64, 64), alpha=1.0))
                joints[name] = joint
            circle = self._shape_texture(f"ear-{actor_id}", (96, 96), alpha=1.0)
            self._apply_texture(ear_left, circle)
            self._apply_texture(ear_right, circle)
            for node, sort in (
                (upper_leg_left, 10),
                (upper_leg_right, 10),
                (lower_leg_left, 10),
                (lower_leg_right, 10),
                (pelvis_card, 18),
                (body_card, 20),
                (neck_card, 23),
                (upper_arm_left, 26),
                (upper_arm_right, 26),
                (lower_arm_left, 27),
                (lower_arm_right, 27),
                (ear_left, 24),
                (ear_right, 24),
                (head_base, 30),
                (face_card, 40),
            ):
                self._set_actor_layer(node, sort)
            for joint in joints.values():
                self._set_actor_layer(joint, 28)
            self._actor_instances[actor_id] = {
                "root": actor_root,
                "upper_arm_left": upper_arm_left,
                "upper_arm_right": upper_arm_right,
                "lower_arm_left": lower_arm_left,
                "lower_arm_right": lower_arm_right,
                "upper_leg_left": upper_leg_left,
                "upper_leg_right": upper_leg_right,
                "lower_leg_left": lower_leg_left,
                "lower_leg_right": lower_leg_right,
                "pelvis": pelvis_card,
                "body": body_card,
                "neck": neck_card,
                "head_base": head_base,
                "face": face_card,
                "ear_left": ear_left,
                "ear_right": ear_right,
                "joints": joints,
            }

        # Foregrounds were originally screen-space overlays for fake doorway/window masking.
        # In the 3D path we intentionally do not render them; equivalent structure should be
        # modeled as world geometry such as doors, windows, curtains, or room shells.

        scene_duration_ms = max(1, int(scene.get("duration_ms", 1) or 1))
        for index, item in enumerate(scene.get("effects", []), start=1):
            asset_path = self._resolve_effect_asset_path(item)
            if not asset_path:
                continue
            node = self._attach_overlay_card(
                self.effect_root,
                f"effect-{index}",
                1.04,
                1.04,
                (0.0, 0.0),
                asset_path,
            )
            node.hide()
            self._effect_instances.append(
                {
                    "node": node,
                    "asset_path": asset_path,
                    "alpha": float(item.get("alpha", 1.0) or 1.0),
                    "start_ms": int(item.get("start_ms", 0) or 0),
                    "end_ms": int(item.get("end_ms", scene_duration_ms) or scene_duration_ms),
                    "playback_speed": float(item.get("playback_speed", 1.0) or 1.0),
                }
            )

    def _attach_simple_prop(self, item: dict[str, Any], prop: dict[str, Any]) -> None:
        prop_id = str(item.get("prop_id") or "")
        if self._is_sky_prop(item, prop):
            asset_path = str(prop.get("asset_path") or "").strip()
            if not asset_path:
                return
            scale = float(item.get("scale", 1.0) or 1.0)
            width = (float(prop.get("width") or 0.0) or float(prop.get("base_width", 160)) / 140.0) * scale
            height = (float(prop.get("height") or 0.0) or float(prop.get("base_height", 120)) / 140.0) * scale
            x = float(item.get("x", 0.0) or 0.0)
            z = float(item.get("z", 0.0) or 0.0)
            node = self._attach_plane(
                self.skybox_root,
                f"sky-prop-{item['prop_id']}",
                width * 24.0,
                height * 24.0,
                (x * 18.0, 252.0 + max(-18.0, min(18.0, -z * 26.0)), 128.0 + max(-24.0, min(36.0, -z * 42.0))),
                (180.0, 0.0, 0.0),
                asset_path,
                (1.0, 1.0, 1.0, 1.0),
            )
            node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
            node.setDepthWrite(False)
            node.setDepthTest(False)
            node.setBin("background", -94)
            node.setTwoSided(True)
            self._sky_prop_instances.append(
                {
                    "node": node,
                    "asset_path": asset_path,
                    "motion_period_ms": 1200,
                }
            )
            return
        room = self._room_dims
        width = (float(prop.get("width") or 0.0) or float(prop.get("base_width", 160)) / 140.0) * float(item.get("scale", 1.0) or 1.0)
        height = (float(prop.get("height") or 0.0) or float(prop.get("base_height", 120)) / 140.0) * float(item.get("scale", 1.0) or 1.0)
        width_scale, height_scale = self._prop_world_scale(prop_id, prop)
        width *= width_scale
        height *= height_scale
        x = float(item.get("x", 0.0) or 0.0)
        z = float(item.get("z", 0.0) or 0.0)
        layer = str(item.get("layer") or prop.get("default_layer") or "front")
        y = -0.55 if layer == "front" else 0.75
        root = self.prop_root.attachNewNode(f"prop-{prop_id}")
        root.setPos(x, y, -room["height"] / 2.0 + room["floor_thickness"] / 2.0 + max(0.12, height * 0.5) + z * 0.15)
        color = tuple(prop.get("color") or (0.84, 0.84, 0.84, 1.0))
        asset_path = str(prop.get("asset_path") or "").strip()
        if asset_path:
            plane = self._attach_card(root, width, height, f"prop-plane-{prop_id}", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
            if self.fast_card_mode:
                plane.setBillboardAxis()
            else:
                plane.setBillboardPointEye()
            texture = self._texture_at_time(asset_path, 0)
            self._apply_texture(plane, texture)
            self._prop_instances.append(
                {
                    "node": plane,
                    "asset_path": asset_path,
                    "motion_period_ms": int(prop.get("motion_period_ms", 1400) or 1400),
                }
            )
            return
        self._attach_box(root, f"prop-box-{prop_id}", max(0.18, width * 0.7), max(0.18, width * 0.7), max(0.18, height), (0.0, 0.0, 0.0), color)

    def _attach_overlay_text(self, parent, text: str, pos: tuple[float, float], scale: float, fg=(1.0, 1.0, 1.0, 1.0), shadow=(0.0, 0.0, 0.0, 0.50)):
        text_node = self._core["TextNode"]("overlay-text")
        text_node.setText(str(text))
        text_node.setTextColor(*fg)
        text_node.setAlign(self._core["TextNode"].ACenter)
        if self._text_font:
            text_node.setFont(self._text_font)
        shadow_np = parent.attachNewNode(text_node)
        shadow_np.setScale(scale)
        shadow_np.setPos(pos[0] + 0.003, 0.0, pos[1] - 0.003)
        shadow_np.setColor(*shadow)
        text_np = parent.attachNewNode(text_node)
        text_np.setScale(scale)
        text_np.setPos(pos[0], 0.0, pos[1])

    def _place_actor(self, actor_id: str, scene: dict[str, Any], time_ms: int) -> None:
        actor_item = next((item for item in scene.get("actors", []) if str(item["actor_id"]) == actor_id), None)
        if not actor_item:
            return
        instance = self._actor_instances[actor_id]
        root = instance["root"]
        upper_arm_left = instance["upper_arm_left"]
        upper_arm_right = instance["upper_arm_right"]
        lower_arm_left = instance["lower_arm_left"]
        lower_arm_right = instance["lower_arm_right"]
        upper_leg_left = instance["upper_leg_left"]
        upper_leg_right = instance["upper_leg_right"]
        lower_leg_left = instance["lower_leg_left"]
        lower_leg_right = instance["lower_leg_right"]
        pelvis = instance["pelvis"]
        body = instance["body"]
        neck = instance["neck"]
        head_base = instance["head_base"]
        face = instance["face"]
        ear_left = instance["ear_left"]
        ear_right = instance["ear_right"]
        joints = instance["joints"]
        room = self._room_dims
        spawn = actor_item.get("spawn", {}) or {}
        x = float(spawn.get("x", 0.0) or 0.0)
        z = float(spawn.get("z", 0.0) or 0.0)
        layer = str(actor_item.get("layer") or "front")
        y = -0.75 if layer == "front" else 0.55
        expression = self._expression_for_actor(scene, actor_id, time_ms)
        active_beat = self._active_beat(scene, actor_id, time_ms)
        is_talking = self._is_actor_talking(scene, actor_id, time_ms)
        has_action = active_beat is not None
        facing = str((active_beat or {}).get("facing") or actor_item.get("facing") or "right").strip().lower()
        facing_sign = -1.0 if facing == "left" else 1.0
        if active_beat:
            start_ms = int(active_beat.get("start_ms", 0) or 0)
            end_ms = max(start_ms + 1, int(active_beat.get("end_ms", start_ms + 1) or start_ms + 1))
            beat_ratio = (time_ms - start_ms) / max(1.0, float(end_ms - start_ms))
            if active_beat.get("x0") is not None and active_beat.get("x1") is not None:
                x = self._lerp(float(active_beat.get("x0", x) or x), float(active_beat.get("x1", x) or x), self._ease_in_out(beat_ratio))
        motion_strength = 0.0 if has_action else (0.02 if is_talking else 0.0)
        bob = abs(math.sin(time_ms / 220.0 + x * 0.3)) * 0.02 * motion_strength
        pose_state = self._pose_body_state(active_beat, time_ms, facing_sign) if active_beat and active_beat.get("pose_track_path") else None
        body_state = self._static_body_state(is_talking) if pose_state is None else {
            "root_dx": 0.0,
            "jump": 0.0,
            "pelvis_x": pose_state["pelvis_center"][0],
            "pelvis_z": pose_state["pelvis_center"][1],
            "pelvis_r": pose_state["torso_r"],
            "chest_x": pose_state["chest_center"][0],
            "chest_z": pose_state["chest_center"][1],
            "chest_r": pose_state["torso_r"],
            "neck_x": pose_state["neck_center"][0],
            "neck_z": pose_state["neck_center"][1],
            "neck_r": pose_state["head_r"],
            "head_x": pose_state["head_center"][0],
            "head_z": pose_state["head_center"][1],
            "head_r": pose_state["head_r"],
        }
        root.setPos(
            x + body_state["root_dx"] * facing_sign,
            y,
            -room["height"] / 2.0 + room["floor_thickness"] / 2.0 + z + bob + body_state["jump"],
        )
        base_scale = 1.55 * float(actor_item.get("scale", 1.0) or 1.0)
        root.setScale(base_scale * 0.96, base_scale, base_scale * 1.50)

        character = self.characters.get(self.cast.get(actor_id, {}).get("asset_id", ""), {})
        face_texture = self._texture_at_time(self._face_skin_path(character, expression, time_ms, talking=is_talking), time_ms)
        outfit_texture = self._texture_at_time(self._outfit_skin_path(character), time_ms)
        body_color = tuple(character.get("body_color") or (0.24, 0.34, 0.46, 1.0))
        trim_color = tuple(character.get("body_secondary_color") or (0.82, 0.82, 0.82, 1.0))
        head_color = tuple(character.get("head_color") or (0.97, 0.97, 0.95, 1.0))
        if pose_state is None:
            pelvis_center = (body_state["pelvis_x"] * facing_sign, body_state["pelvis_z"])
            chest_center = (body_state["chest_x"] * facing_sign, body_state["chest_z"])
            neck_center = (body_state["neck_x"] * facing_sign, body_state["neck_z"])
            head_center = (body_state["head_x"] * facing_sign, body_state["head_z"])
            shoulder_left = (chest_center[0] - 0.25 * facing_sign, chest_center[1] + 0.22)
            shoulder_right = (chest_center[0] + 0.25 * facing_sign, chest_center[1] + 0.22)
            hip_left = (pelvis_center[0] - 0.11 * facing_sign, pelvis_center[1] - 0.02)
            hip_right = (pelvis_center[0] + 0.11 * facing_sign, pelvis_center[1] - 0.02)
            elbow_left = (shoulder_left[0] - 0.10 * facing_sign, shoulder_left[1] - 0.50)
            hand_left = (elbow_left[0] - 0.06 * facing_sign, elbow_left[1] - 0.52)
            elbow_right = (shoulder_right[0] + 0.10 * facing_sign, shoulder_right[1] - 0.50)
            hand_right = (elbow_right[0] + 0.06 * facing_sign, elbow_right[1] - 0.52)
            knee_left = (hip_left[0] - 0.03 * facing_sign, hip_left[1] - 0.82)
            foot_left = (knee_left[0], knee_left[1] - 0.78)
            knee_right = (hip_right[0] + 0.03 * facing_sign, hip_right[1] - 0.82)
            foot_right = (knee_right[0], knee_right[1] - 0.78)
            head_center = (head_center[0], head_center[1] + 0.08)
            neck_center = (
                chest_center[0] + (head_center[0] - chest_center[0]) * 0.36,
                chest_center[1] + (head_center[1] - chest_center[1]) * 0.36,
            )
        else:
            pelvis_center = pose_state["pelvis_center"]
            chest_center = pose_state["chest_center"]
            neck_center = pose_state["neck_center"]
            head_center = pose_state["head_center"]
            shoulder_left = pose_state["shoulder_left"]
            shoulder_right = pose_state["shoulder_right"]
            hip_left = pose_state["hip_left"]
            hip_right = pose_state["hip_right"]
            elbow_left = pose_state["elbow_left"]
            elbow_right = pose_state["elbow_right"]
            hand_left = pose_state["hand_left"]
            hand_right = pose_state["hand_right"]
            knee_left = pose_state["knee_left"]
            knee_right = pose_state["knee_right"]
            foot_left = pose_state["foot_left"]
            foot_right = pose_state["foot_right"]
        upper_arm_left.setColor(*body_color)
        upper_arm_right.setColor(*body_color)
        lower_arm_left.setColor(*trim_color)
        lower_arm_right.setColor(*trim_color)
        upper_leg_left.setColor(*body_color)
        upper_leg_right.setColor(*body_color)
        lower_leg_left.setColor(*trim_color)
        lower_leg_right.setColor(*trim_color)
        self._place_segment_card(upper_arm_left, shoulder_left, elbow_left, 0.14)
        self._place_segment_card(lower_arm_left, elbow_left, hand_left, 0.10)
        self._place_segment_card(upper_arm_right, shoulder_right, elbow_right, 0.14)
        self._place_segment_card(lower_arm_right, elbow_right, hand_right, 0.10)
        self._place_segment_card(upper_leg_left, hip_left, knee_left, 0.16)
        self._place_segment_card(lower_leg_left, knee_left, foot_left, 0.12)
        self._place_segment_card(upper_leg_right, hip_right, knee_right, 0.16)
        self._place_segment_card(lower_leg_right, knee_right, foot_right, 0.12)
        joint_positions = {
            "shoulder_left": shoulder_left,
            "shoulder_right": shoulder_right,
            "elbow_left": elbow_left,
            "elbow_right": elbow_right,
            "hip_left": hip_left,
            "hip_right": hip_right,
            "knee_left": knee_left,
            "knee_right": knee_right,
        }
        for joint_name, joint in joints.items():
            self._place_joint_card(joint, joint_positions[joint_name], 0.11 if "elbow" in joint_name or "knee" in joint_name else 0.17)
            if joint_name.startswith("shoulder") or joint_name.startswith("hip"):
                joint.setColor(*body_color)
            else:
                joint.setColor(*trim_color)
        pelvis.setPos(pelvis_center[0], 0.0, pelvis_center[1])
        pelvis.setR(body_state["pelvis_r"])
        body.setPos(chest_center[0], 0.0, chest_center[1])
        body.setR(body_state["chest_r"])
        neck.setPos(neck_center[0], 0.0, neck_center[1])
        neck.setR(body_state["neck_r"])
        if outfit_texture is not None:
            self._apply_texture(body, outfit_texture)
            body.setColor(1.0, 1.0, 1.0, 1.0)
        else:
            self._apply_texture(body, self._rounded_rect_texture("torso-body-mask", (182, 424), 76, alpha=1.0))
            body.setColor(*body_color)
        self._apply_texture(pelvis, self._rounded_rect_texture("torso-pelvis-mask-hidden", (32, 32), 16, alpha=0.0))
        pelvis.setColor(1.0, 1.0, 1.0, 0.0)
        self._apply_texture(neck, self._rounded_rect_texture("torso-neck-mask", (34, 112), 16, alpha=1.0))
        neck.setColor(*head_color)
        head_rotation = -body_state["head_r"]
        face_vertical_offset = 0.0
        head_base.setColor(*head_color)
        head_base.setPos(head_center[0], 0.01, head_center[1])
        head_base.setR(-head_rotation)
        face.setPos(head_center[0], 0.03, head_center[1] + face_vertical_offset)
        face.setR(-head_rotation)
        left_ear_offset = self._rotate_offset(-0.33 * facing_sign, 0.36, head_rotation)
        right_ear_offset = self._rotate_offset(0.33 * facing_sign, 0.36, head_rotation)
        ear_left.setPos(head_center[0] + left_ear_offset[0], -0.02, head_center[1] + left_ear_offset[1])
        ear_right.setPos(head_center[0] + right_ear_offset[0], -0.02, head_center[1] + right_ear_offset[1])
        ear_left.setR(head_rotation)
        ear_right.setR(head_rotation)
        if face_texture is not None:
            self._apply_texture(face, face_texture)
            face.setColor(1.0, 1.0, 1.0, 1.0)
        else:
            self._apply_texture(face, None)
            face.setColor(*head_color)

    def _render_subtitles(self, scene: dict[str, Any], time_ms: int) -> None:
        self._detach_children(self.subtitle_root)
        current = None
        for dialogue in scene.get("dialogues", []):
            if int(dialogue.get("start_ms", -1)) <= time_ms <= int(dialogue.get("end_ms", -1)):
                current = dialogue
                break
        if not current:
            return
        plate = self._attach_card(self.subtitle_root, 1.72, 0.18, "subtitle-bg", (0.0, 0.0, -0.82), (0.04, 0.04, 0.04, 0.74))
        plate.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        self._attach_overlay_text(self.subtitle_root, str(current.get("subtitle") or current.get("text") or ""), (0.0, -0.84), 0.07)

    def _render_foregrounds(self, time_ms: int) -> None:
        for item in self._foreground_instances:
            texture = self._texture_at_time(
                item.get("asset_path"),
                time_ms,
                period_ms=int(item.get("motion_period_ms", 1400) or 1400),
            )
            self._apply_texture(item["node"], texture)
            item["node"].setColorScale(1.0, 1.0, 1.0, float(item.get("opacity", 1.0) or 1.0))

    def _render_sky_props(self, time_ms: int) -> None:
        for item in self._sky_prop_instances:
            texture = self._texture_at_time(
                item.get("asset_path"),
                time_ms,
                period_ms=int(item.get("motion_period_ms", 1200) or 1200),
            )
            self._apply_texture(item["node"], texture)

    def _render_props(self, time_ms: int) -> None:
        for item in self._prop_instances:
            texture = self._texture_at_time(
                item.get("asset_path"),
                time_ms,
                period_ms=int(item.get("motion_period_ms", 1400) or 1400),
            )
            self._apply_texture(item["node"], texture)

    def _render_effects(self, time_ms: int) -> None:
        for item in self._effect_instances:
            node = item["node"]
            start_ms = int(item.get("start_ms", 0) or 0)
            end_ms = max(start_ms + 1, int(item.get("end_ms", start_ms + 1) or start_ms + 1))
            if time_ms < start_ms or time_ms > end_ms:
                node.hide()
                continue
            progress = (time_ms - start_ms) / max(1.0, float(end_ms - start_ms))
            texture = self._texture_at_progress(
                item.get("asset_path"),
                progress,
                playback_speed=float(item.get("playback_speed", 1.0) or 1.0),
            )
            if texture is None:
                node.hide()
                continue
            self._apply_texture(node, texture)
            node.setColorScale(1.0, 1.0, 1.0, self._effect_alpha_ratio(float(item.get("alpha", 1.0) or 1.0)))
            node.show()

    def _capture_scene_frame_rgb(self) -> Optional[bytes]:
        texture = self._capture_texture
        if not texture.hasRamImage():
            return None
        payload = texture.getRamImageAs("RGB")
        if not payload:
            return None
        frame_bytes = bytes(payload)
        expected_size = self.frame_width * self.frame_height * 3
        if len(frame_bytes) != expected_size:
            return None
        row_stride = self.frame_width * 3
        if row_stride <= 0:
            return frame_bytes
        return b"".join(
            frame_bytes[row_start : row_start + row_stride]
            for row_start in range(expected_size - row_stride, -1, -row_stride)
        )

    @staticmethod
    def _round_signature(value: float, step: float = 0.05) -> float:
        if step <= 0:
            return round(float(value), 4)
        return round(round(float(value) / step) * step, 3)

    @staticmethod
    def _rotate_offset(offset_x: float, offset_z: float, angle_deg: float) -> tuple[float, float]:
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return (
            offset_x * cos_a - offset_z * sin_a,
            offset_x * sin_a + offset_z * cos_a,
        )

    def _active_subtitle_signature(self, scene: dict[str, Any], time_ms: int) -> tuple[str, int, int] | None:
        for dialogue in scene.get("dialogues", []):
            start_ms = int(dialogue.get("start_ms", -1))
            end_ms = int(dialogue.get("end_ms", -1))
            if start_ms <= time_ms <= end_ms:
                return (str(dialogue.get("speaker_id") or ""), start_ms, end_ms)
        return None

    def _has_active_effect(self, time_ms: int) -> bool:
        for item in self._effect_instances:
            start_ms = int(item.get("start_ms", 0) or 0)
            end_ms = max(start_ms + 1, int(item.get("end_ms", start_ms + 1) or start_ms + 1))
            if start_ms <= time_ms <= end_ms:
                return True
        return False

    def _actor_frame_signature(self, scene: dict[str, Any], actor_item: dict[str, Any], time_ms: int) -> tuple[Any, ...]:
        actor_id = str(actor_item.get("actor_id") or "")
        spawn = actor_item.get("spawn", {}) or {}
        x = float(spawn.get("x", 0.0) or 0.0)
        z = float(spawn.get("z", 0.0) or 0.0)
        expression = self._expression_for_actor(scene, actor_id, time_ms)
        talking = self._is_actor_talking(scene, actor_id, time_ms)
        active_beat = self._active_beat(scene, actor_id, time_ms)
        facing = str((active_beat or {}).get("facing") or actor_item.get("facing") or "right").strip().lower()
        facing_sign = -1.0 if facing == "left" else 1.0
        if active_beat:
            start_ms = int(active_beat.get("start_ms", 0) or 0)
            end_ms = max(start_ms + 1, int(active_beat.get("end_ms", start_ms + 1) or start_ms + 1))
            beat_ratio = (time_ms - start_ms) / max(1.0, float(end_ms - start_ms))
            if active_beat.get("x0") is not None and active_beat.get("x1") is not None:
                x = self._lerp(float(active_beat.get("x0", x) or x), float(active_beat.get("x1", x) or x), self._ease_in_out(beat_ratio))
        pose_state = self._pose_body_state(active_beat, time_ms, facing_sign) if active_beat and active_beat.get("pose_track_path") else None
        if pose_state is None:
            return (
                actor_id,
                self._round_signature(x, 0.08),
                self._round_signature(z, 0.08),
                expression,
                "talk" if talking else "still",
                facing,
                int(time_ms // 120) if talking else 0,
            )
        return (
            actor_id,
            self._round_signature(x, 0.08),
            self._round_signature(z, 0.08),
            expression,
            "talk" if talking else "pose",
            facing,
            int(time_ms // 120) if talking else 0,
            self._round_signature(pose_state["pelvis_center"][0], 0.06),
            self._round_signature(pose_state["pelvis_center"][1], 0.06),
            self._round_signature(pose_state["chest_center"][0], 0.06),
            self._round_signature(pose_state["chest_center"][1], 0.06),
            self._round_signature(pose_state["head_center"][0], 0.06),
            self._round_signature(pose_state["head_center"][1], 0.06),
            self._round_signature(pose_state["torso_r"], 4.0),
            self._round_signature(pose_state["head_r"], 4.0),
        )

    def _frame_cache_signature(self, scene: dict[str, Any], time_ms: int) -> tuple[Any, ...] | None:
        if self._has_active_effect(time_ms):
            return None
        camera = self._camera_state(scene, time_ms)
        return (
            str(scene.get("id") or ""),
            self._active_subtitle_signature(scene, time_ms),
            self._round_signature(camera["x"], 0.04),
            self._round_signature(camera["z"], 0.04),
            self._round_signature(camera["zoom"], 0.04),
            tuple(self._actor_frame_signature(scene, actor, time_ms) for actor in scene.get("actors", [])),
        )

    def _apply_camera(self, scene: dict[str, Any], time_ms: int) -> None:
        state = self._camera_state(scene, time_ms)
        room = self._room_dims
        distance = max(room["depth"] * 3.10, room["width"] * 1.34)
        floor_z = -room["height"] / 2.0 + room["floor_thickness"] / 2.0
        shoulder_level_z = floor_z + room["height"] * 0.29
        cam_x = state["x"] * 0.70
        cam_y = -distance / max(0.55, state["zoom"])
        cam_z = shoulder_level_z + state["z"] * 0.12
        self._lens.setFov(max(23.5, 40.0 / max(0.5, state["zoom"])))
        self.base.camera.setPos(cam_x, cam_y, cam_z)
        self.base.camera.lookAt(state["x"] * 0.22, 0.0, shoulder_level_z + room["height"] * 0.18 + state["z"] * 0.10)
        if self.fast_card_mode:
            self.skybox_root.setPos(cam_x, cam_y, cam_z)

    def capture_scene_frame(self, scene: dict[str, Any], time_ms: int, raw_rgb: bool = False) -> bytes:
        self._prepare_scene(scene)
        self._current_scene = scene
        frame_signature = None
        if self.fast_card_mode and raw_rgb:
            frame_signature = self._frame_cache_signature(scene, time_ms)
            if frame_signature is not None and frame_signature == self._last_frame_signature and self._last_frame_rgb is not None:
                return self._last_frame_rgb
        if self.show_actor_labels:
            self._detach_children(self.label_root)
        self._apply_camera(scene, time_ms)
        self._render_sky_props(time_ms)
        self._render_props(time_ms)
        for actor in scene.get("actors", []):
            actor_id = str(actor["actor_id"])
            self._place_actor(actor_id, scene, time_ms)
        self._render_effects(time_ms)
        self._render_subtitles(scene, time_ms)
        if self.show_actor_labels:
            for index, actor in enumerate(scene.get("actors", [])):
                actor_id = str(actor["actor_id"])
                label = self.cast.get(actor_id, {}).get("display_name") or actor_id
                self._attach_overlay_text(self.label_root, label, (-0.42 + index * 0.42, 0.72), 0.06)
        self.base.graphicsEngine.renderFrame()
        if raw_rgb:
            payload = self._capture_scene_frame_rgb()
            if payload:
                if self.fast_card_mode:
                    self._last_frame_signature = frame_signature
                    self._last_frame_rgb = payload
                return payload
            self.base.graphicsEngine.renderFrame()
            payload = self._capture_scene_frame_rgb()
            if payload:
                if self.fast_card_mode:
                    self._last_frame_signature = frame_signature
                    self._last_frame_rgb = payload
                return payload
            raise RuntimeError("failed to capture raw RGB scene frame")
        image = self._core["PNMImage"]()
        for _ in range(2):
            if self.base.win.getScreenshot(image):
                stream = self._core["StringStream"]()
                if image.write(stream, "png"):
                    data = stream.getData()
                    if data:
                        return bytes(data)
            self.base.graphicsEngine.renderFrame()
        raise RuntimeError("failed to capture frame bytes")

    def close(self) -> None:
        self.base.destroy()
