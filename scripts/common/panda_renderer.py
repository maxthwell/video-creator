from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageChops, ImageDraw

from .io import ASSETS_DIR, CHARACTERS_DIR, TMP_DIR, _cached_remote_asset, manifest_index


class PandaSceneRenderer:
    def __init__(self, story: dict[str, Any], prefer_gpu: bool = True):
        try:
            from panda3d.core import (
                AmbientLight,
                CardMaker,
                ClockObject,
                DirectionalLight,
                Filename,
                GraphicsOutput,
                OrthographicLens,
                PNMImage,
                StringStream,
                TextNode,
                Texture,
                TransparencyAttrib,
                loadPrcFileData,
            )
            from panda3d.ai import AICharacter, AIWorld
            from direct.showbase.ShowBase import ShowBase
        except ModuleNotFoundError as exc:
            raise RuntimeError("Panda3D is required. Install `panda3d` and rerun.") from exc

        self._core = {
            "AmbientLight": AmbientLight,
            "AICharacter": AICharacter,
            "AIWorld": AIWorld,
            "CardMaker": CardMaker,
            "ClockObject": ClockObject,
            "DirectionalLight": DirectionalLight,
            "Filename": Filename,
            "GraphicsOutput": GraphicsOutput,
            "OrthographicLens": OrthographicLens,
            "PNMImage": PNMImage,
            "StringStream": StringStream,
            "TextNode": TextNode,
            "Texture": Texture,
            "TransparencyAttrib": TransparencyAttrib,
            "loadPrcFileData": loadPrcFileData,
            "ShowBase": ShowBase,
        }
        width = int(story["video"]["width"])
        height = int(story["video"]["height"])
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
        self.base = ShowBase(windowType="offscreen")
        self.pipe_name = self.base.pipe.getType().getName()
        self.base.disableMouse()
        self._capture_texture = Texture()
        self._capture_texture.setKeepRamImage(True)
        self.base.win.addRenderTexture(self._capture_texture, GraphicsOutput.RTM_copy_ram)
        self.frame_center_z = float(story["video"].get("frame_center_z", 0.85) or 0.85)
        self.camera_base_pos = (0.0, -20.0, self.frame_center_z)
        lens = OrthographicLens()
        self._base_film_width = 12.0
        self._base_film_height = 12.0 * height / width
        lens.setFilmSize(self._base_film_width, self._base_film_height)
        self.base.cam.node().setLens(lens)
        self._lens = lens
        self.base.camera.setPos(*self.camera_base_pos)
        self.base.camera.lookAt(0, 0, self.frame_center_z)

        self.stage_root = self.base.render.attachNewNode("stage-root")
        self.overlay_root = self.base.aspect2d.attachNewNode("overlay-root")
        self.background_root = self.stage_root.attachNewNode("background-root")
        self.props_root = self.stage_root.attachNewNode("props-root")
        self.actors_root = self.stage_root.attachNewNode("actors-root")
        self.npcs_root = self.stage_root.attachNewNode("npcs-root")
        self.effects_root = self.stage_root.attachNewNode("effects-root")
        self.scene_label_root = self.overlay_root.attachNewNode("scene-label-root")
        self.actor_label_root = self.overlay_root.attachNewNode("actor-label-root")
        self.overlay_fx_root = self.overlay_root.attachNewNode("overlay-fx-root")
        self.subtitle_root = self.overlay_root.attachNewNode("subtitle-root")
        self.backgrounds = manifest_index("backgrounds")
        self.characters = manifest_index("characters")
        self.floors = manifest_index("floors")
        self.props = manifest_index("props")
        self.effects = manifest_index("effects")
        self.story = story
        self.cast = {item["id"]: item for item in story["cast"]}
        self.force_pose_skeleton = bool(story["video"].get("force_pose_skeleton", False))
        self.show_outfit_overlay = bool(story["video"].get("show_outfit_overlay", True))
        self.show_head_overlay = bool(story["video"].get("show_head_overlay", True))
        self.actor_ground_offset = float(story["video"].get("actor_ground_offset", 0.0) or 0.0)
        self.actor_front_bias = float(story["video"].get("actor_front_bias", 0.90) or 0.90)
        self.actor_scale_base = float(story["video"].get("actor_scale_base", 0.667) or 0.667)
        self.room_wall_angle = float(story["video"].get("room_wall_angle", 64.0) or 64.0)
        self.room_floor_pitch = float(story["video"].get("room_floor_pitch", 64.0) or 64.0)
        self.room_wall_inset = float(story["video"].get("room_wall_inset", 1.72) or 1.72)
        self.room_wall_thickness = float(story["video"].get("room_wall_thickness", 0.26) or 0.26)
        self.room_floor_thickness = float(story["video"].get("room_floor_thickness", 0.24) or 0.24)
        stage_layout = story["video"].get("stage_layout") or {}
        self.stage_layout = {
            "background_width": float(stage_layout.get("background_width", 14.0) or 14.0),
            "background_height": float(stage_layout.get("background_height", 7.2) or 7.2),
            "background_y": float(stage_layout.get("background_y", 8.0) or 8.0),
            "background_z": float(stage_layout.get("background_z", 0.98) or 0.98),
            "ground_width": float(stage_layout.get("ground_width", 17.6) or 17.6),
            "ground_height": float(stage_layout.get("ground_height", 12.6) or 12.6),
            "ground_y": float(stage_layout.get("ground_y", 7.0) or 7.0),
            "ground_z": float(stage_layout.get("ground_z", -1.2) or -1.2),
            "ground_pitch": float(stage_layout.get("ground_pitch", -78.0) or -78.0),
            "ground_slope": float(stage_layout.get("ground_slope", 0.22) or 0.22),
        }
        self.text_font = self._load_text_font()
        self._texture_sequences: dict[str, dict[str, Any]] = {}
        self._shape_texture_cache: dict[str, Any] = {}
        self._shared_skins_dir = CHARACTERS_DIR / "_shared_skins"
        self._prepared_scene_key: Optional[str] = None
        self._background_nodes: dict[str, Any] = {}
        self._prop_instances: list[dict[str, Any]] = []
        self._actor_instances: dict[str, dict[str, Any]] = {}
        self._npc_instances: list[dict[str, Any]] = []
        self._impact_instances: list[dict[str, Any]] = []
        self._npc_scene_states: dict[str, dict[str, Any]] = {}
        self._scene_combat_cache: dict[str, dict[str, Any]] = {}
        self._clock = ClockObject.getGlobalClock()
        self._clock.setMode(ClockObject.M_non_real_time)
        self._clock.setDt(1.0 / max(1, int(story["video"].get("fps", 12) or 12)))
        self._build_lighting()

    def _load_text_font(self):
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                font = self.base.loader.loadFont(candidate)
                if font:
                    return font
        return None

    def _build_lighting(self) -> None:
        AmbientLight = self._core["AmbientLight"]
        ambient = self.base.render.attachNewNode(AmbientLight("ambient"))
        ambient.node().setColor((1.0, 1.0, 1.0, 1.0))
        self.base.render.setLight(ambient)

    def _detach_children(self, node) -> None:
        node.getChildren().detach()

    def _clear_dynamic_frame(self) -> None:
        self._detach_children(self.overlay_fx_root)
        self._detach_children(self.subtitle_root)

    def _reset_scene(self) -> None:
        for scene_key in list(self._npc_scene_states.keys()):
            self._cleanup_npc_scene_state(scene_key)
        self._detach_children(self.background_root)
        self._detach_children(self.props_root)
        self._detach_children(self.actors_root)
        self._detach_children(self.npcs_root)
        self._detach_children(self.effects_root)
        self._detach_children(self.scene_label_root)
        self._detach_children(self.actor_label_root)
        self._detach_children(self.overlay_fx_root)
        self._detach_children(self.subtitle_root)
        self._background_nodes = {}
        self._prop_instances = []
        self._actor_instances = {}
        self._npc_instances = []
        self._impact_instances = []
        self._prepared_scene_key = None

    def _make_card(self, width: float, height: float, name: str):
        CardMaker = self._core["CardMaker"]
        cm = CardMaker(name)
        cm.setFrame(-width / 2, width / 2, -height / 2, height / 2)
        return cm.generate()

    def _attach_card(
        self,
        parent,
        width: float,
        height: float,
        name: str,
        pos: tuple[float, float, float],
        color: tuple[float, ...],
        h: float = 0.0,
        p: float = 0.0,
        r: float = 0.0,
    ):
        node = parent.attachNewNode(self._make_card(width, height, name))
        node.setPos(*pos)
        node.setColor(*tuple(color))
        if len(color) == 4 and color[3] < 0.999:
            node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        node.setHpr(h, p, r)
        return node

    def _attach_rect_frame(self, parent, width: float, height: float, border: float, name: str):
        frame_root = parent.attachNewNode(name)
        half_w = width / 2.0
        half_h = height / 2.0
        self._attach_card(frame_root, width, border, f"{name}-top", (0.0, 0.0, half_h - border / 2.0), (1.0, 1.0, 1.0, 1.0))
        self._attach_card(frame_root, width, border, f"{name}-bottom", (0.0, 0.0, -half_h + border / 2.0), (1.0, 1.0, 1.0, 1.0))
        self._attach_card(frame_root, border, max(0.1, height - border * 2.0), f"{name}-left", (-half_w + border / 2.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
        self._attach_card(frame_root, border, max(0.1, height - border * 2.0), f"{name}-right", (half_w - border / 2.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
        return frame_root

    def _set_subtree_color(self, node, color: tuple[float, ...]) -> None:
        node.setColor(*color)
        for child in node.getChildren():
            child.setColor(*color)
            self._set_subtree_color(child, color)

    def _color_luma(self, color: tuple[float, ...]) -> float:
        return color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722

    def _ensure_floor_contrast(
        self,
        back_color: tuple[float, ...],
        floor_color: tuple[float, ...],
    ) -> tuple[float, ...]:
        dr = floor_color[0] - back_color[0]
        dg = floor_color[1] - back_color[1]
        db = floor_color[2] - back_color[2]
        distance = math.sqrt(dr * dr + dg * dg + db * db)
        luma_gap = abs(self._color_luma(floor_color) - self._color_luma(back_color))
        if distance >= 0.26 and luma_gap >= 0.16:
            return floor_color
        alpha = floor_color[3] if len(floor_color) > 3 else 1.0
        back_luma = self._color_luma(back_color)
        if back_luma >= 0.48:
            contrasted = (
                max(0.08, floor_color[0] * 0.48),
                max(0.07, floor_color[1] * 0.42),
                max(0.06, floor_color[2] * 0.34),
                alpha,
            )
        else:
            contrasted = (
                min(0.82, floor_color[0] * 0.58 + 0.22),
                min(0.74, floor_color[1] * 0.52 + 0.18),
                min(0.62, floor_color[2] * 0.46 + 0.12),
                alpha,
            )
        return contrasted

    def _attach_extruded_panel(
        self,
        parent,
        width: float,
        height: float,
        depth: float,
        name: str,
        pos: tuple[float, float, float],
        color: tuple[float, ...],
        side_color: Optional[tuple[float, ...]] = None,
        include_lr_sides: bool = True,
    ):
        panel_root = parent.attachNewNode(name)
        panel_root.setPos(*pos)
        shade = side_color or color
        front = self._attach_card(panel_root, width, height, f"{name}-front", (0.0, -depth / 2.0, 0.0), color)
        front.setTwoSided(True)
        back = self._attach_card(panel_root, width, height, f"{name}-back", (0.0, depth / 2.0, 0.0), shade, h=180.0)
        back.setTwoSided(True)
        faces = [front, back]
        if include_lr_sides:
            left = self._attach_card(panel_root, depth, height, f"{name}-left", (-width / 2.0, 0.0, 0.0), shade, h=90.0)
            left.setTwoSided(True)
            right = self._attach_card(panel_root, depth, height, f"{name}-right", (width / 2.0, 0.0, 0.0), shade, h=-90.0)
            right.setTwoSided(True)
            faces.extend([left, right])
        top = self._attach_card(panel_root, width, depth, f"{name}-top", (0.0, 0.0, height / 2.0), shade, p=90.0)
        top.setTwoSided(True)
        bottom = self._attach_card(panel_root, width, depth, f"{name}-bottom", (0.0, 0.0, -height / 2.0), shade, p=-90.0)
        bottom.setTwoSided(True)
        faces.extend([top, bottom])
        return front, faces

    def _attach_opening_depth(self, surface_root, opening: dict[str, float], name: str, frame_color: tuple[float, ...], wall_color: tuple[float, ...]) -> None:
        width = float(opening["width"])
        height = float(opening["height"])
        x = float(opening["x"])
        z = float(opening["z"])
        border = max(0.08, min(width, height) * 0.09)
        front = self._attach_rect_frame(surface_root, width + border * 0.8, height + border * 0.8, border, f"{name}-front")
        front.setPos(x, -(self.room_wall_thickness * 0.55 + 0.01), z)
        front.setColor(*frame_color)
        front.setTransparency(self._core["TransparencyAttrib"].MAlpha)

    def _attach_room_shadows(self, back_root, left_root, right_root, ground_root, room_width: float, room_height: float, floor_span: float) -> None:
        back_base = self._attach_card(back_root, room_width * 0.98, 0.44, "back-base-shadow", (0.0, -2.35, -room_height / 2.0 + 0.22), (0.01, 0.01, 0.02, 0.34))
        back_base.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        ground_back = self._attach_card(ground_root, room_width * 0.98, 0.54, "ground-back-shadow", (0.0, -0.02, floor_span / 2.0 - 0.44), (0.01, 0.01, 0.02, 0.26))
        ground_back.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        ground_left = self._attach_card(ground_root, 0.54, floor_span * 0.84, "ground-left-shadow", (-room_width / 2.0 + 0.28, -0.01, 0.06), (0.01, 0.01, 0.02, 0.16))
        ground_left.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        ground_right = self._attach_card(ground_root, 0.54, floor_span * 0.84, "ground-right-shadow", (room_width / 2.0 - 0.28, -0.01, 0.06), (0.01, 0.01, 0.02, 0.16))
        ground_right.setTransparency(self._core["TransparencyAttrib"].MAlpha)

    def _attach_back_corner_caps(self, back_root, room_width: float, room_height: float, color: tuple[float, ...]) -> None:
        cap_x = room_width / 2.0 - self.room_wall_inset + 0.04
        for direction in (-1.0, 1.0):
            cap = self._attach_card(back_root, 0.34, room_height * 0.98, f"back-corner-cap-{direction:+.0f}", (direction * cap_x, -self.room_wall_thickness * 0.72, 0.0), color)
            cap.setTransparency(self._core["TransparencyAttrib"].MAlpha)

    def _normalized_rgba(self, value: Any, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if not isinstance(value, (list, tuple)):
            return default
        items = [float(item) for item in value[:4]]
        if any(item > 1.0 for item in items):
            items = [item / 255.0 for item in items]
        while len(items) < 4:
            items.append(default[len(items)])
        return tuple(items[:4])

    def _resolve_box_surface_path(self, scene: dict[str, Any], key: str) -> Optional[str]:
        box = scene.get("box") or {}
        url = box.get(key)
        if not url:
            return None
        cached = _cached_remote_asset(url, "scene_box")
        return str(cached) if cached else None

    def _surface_root(self, parent, name: str, pos: tuple[float, float, float], h: float = 0.0, p: float = 0.0, r: float = 0.0):
        node = parent.attachNewNode(name)
        node.setPos(*pos)
        node.setHpr(h, p, r)
        return node

    def _architectural_openings(self, scene: dict[str, Any], surface: str) -> list[dict[str, Any]]:
        openings: list[dict[str, Any]] = []
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
            openings.append(
                {
                    "x": float(item.get("x", 0.0)),
                    "z": float(item.get("z", 0.0)),
                    "width": width * scale,
                    "height": height * scale,
                    "render_style": render_style,
                    "frame_color": tuple(prop.get("frame_color") or (0.40, 0.28, 0.16, 1.0)),
                    "glass_color": tuple(prop.get("glass_color") or (0.78, 0.90, 1.0, 0.24)),
                    "door_color": tuple(prop.get("door_color") or prop.get("frame_color") or (0.34, 0.22, 0.11, 0.82)),
                    "frame_padding": float(prop.get("frame_padding", 0.10) or 0.10),
                    "prop_id": str(item.get("prop_id") or ""),
                }
            )
        return openings

    def _opening_specs(self, scene: dict[str, Any], surface: str) -> list[dict[str, float]]:
        return [
            {
                "x": float(item["x"]),
                "z": float(item["z"]),
                "width": float(item["width"]),
                "height": float(item["height"]),
            }
            for item in self._architectural_openings(scene, surface)
        ]

    def _render_structural_opening(self, surface_root, opening: dict[str, Any], name: str) -> None:
        width = float(opening["width"])
        height = float(opening["height"])
        x = float(opening["x"])
        z = float(opening["z"])
        render_style = str(opening["render_style"])
        frame_color = tuple(opening["frame_color"])
        glass_color = tuple(opening["glass_color"])
        door_color = tuple(opening["door_color"])
        padding = min(0.26, max(0.05, float(opening["frame_padding"])))
        border = max(0.08, min(width, height) * padding)

        opening_root = surface_root.attachNewNode(name)
        opening_root.setPos(x, -(self.room_wall_thickness * 0.55 + 0.01), z)

        if render_style != "window":
            outer = self._attach_rect_frame(opening_root, width + border * 0.68, height + border * 0.68, border, f"{name}-outer")
            self._set_subtree_color(outer, frame_color)
            outer.setTransparency(self._core["TransparencyAttrib"].MAlpha)

        inner_w = max(0.18, width - border * 0.55)
        inner_h = max(0.18, height - border * (1.20 if render_style == "window" else 0.72))

        if render_style == "window":
            glass = self._attach_card(opening_root, width * 0.96, height * 0.96, f"{name}-glass", (0.0, 0.002, 0.0), glass_color)
            glass.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        elif render_style == "double-door":
            leaf_w = max(0.12, inner_w * 0.40)
            leaf_h = max(0.20, inner_h)
            left_leaf = self._attach_card(opening_root, leaf_w, leaf_h, f"{name}-left-leaf", (-inner_w * 0.34, 0.04, -border * 0.08), door_color)
            left_leaf.setTransparency(self._core["TransparencyAttrib"].MAlpha)
            left_leaf.setH(-28.0)
            right_leaf = self._attach_card(opening_root, leaf_w, leaf_h, f"{name}-right-leaf", (inner_w * 0.34, 0.04, -border * 0.08), door_color)
            right_leaf.setTransparency(self._core["TransparencyAttrib"].MAlpha)
            right_leaf.setH(28.0)
            seam = self._attach_card(opening_root, max(0.03, border * 0.22), leaf_h, f"{name}-seam", (0.0, 0.03, -border * 0.08), frame_color)
            seam.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        else:
            panel = self._attach_card(opening_root, inner_w, inner_h, f"{name}-panel", (0.0, 0.02, -border * 0.06), door_color)
            panel.setTransparency(self._core["TransparencyAttrib"].MAlpha)

    def _attach_surface_segments(
        self,
        parent,
        name: str,
        width: float,
        height: float,
        pos: tuple[float, float, float],
        color: tuple[float, ...],
        h: float = 0.0,
        p: float = 0.0,
        r: float = 0.0,
        openings: Optional[list[dict[str, float]]] = None,
        thickness: float = 0.0,
        side_color: Optional[tuple[float, ...]] = None,
        include_lr_sides: bool = True,
    ) -> dict[str, Any]:
        root = self._surface_root(parent, name, pos, h=h, p=p, r=r)
        segments = []
        solids = []
        if not openings:
            if thickness > 0.0:
                front, faces = self._attach_extruded_panel(root, width, height, thickness, f"{name}-full", (0.0, 0.0, 0.0), color, side_color=side_color, include_lr_sides=include_lr_sides)
                segments.append(front)
                solids.extend(faces)
            else:
                node = self._attach_card(root, width, height, f"{name}-full", (0.0, 0.0, 0.0), color)
                segments.append(node)
                solids.append(node)
            return {"root": root, "segments": segments, "solids": solids}
        half_w = width / 2.0
        half_h = height / 2.0
        rects: list[tuple[float, float, float, float]] = []
        x_cuts = {-half_w, half_w}
        z_cuts = {-half_h, half_h}
        for opening in openings:
            ox = max(-half_w + 0.25, min(half_w - 0.25, float(opening["x"])))
            oz = max(-half_h + 0.25, min(half_h - 0.25, float(opening["z"])))
            ow = max(0.6, min(width - 0.5, float(opening["width"])))
            oh = max(0.8, min(height - 0.5, float(opening["height"])))
            left = ox - ow / 2.0
            right = ox + ow / 2.0
            bottom = oz - oh / 2.0
            top = oz + oh / 2.0
            rects.append((left, right, bottom, top))
            x_cuts.update({left, right})
            z_cuts.update({bottom, top})
        xs = sorted(x_cuts)
        zs = sorted(z_cuts)
        seg_index = 0
        for x0, x1 in zip(xs, xs[1:]):
            for z0, z1 in zip(zs, zs[1:]):
                cx = (x0 + x1) / 2.0
                cz = (z0 + z1) / 2.0
                if any(left <= cx <= right and bottom <= cz <= top for left, right, bottom, top in rects):
                    continue
                seg_w = x1 - x0
                seg_h = z1 - z0
                if seg_w <= 0.02 or seg_h <= 0.02:
                    continue
                if thickness > 0.0:
                    front, faces = self._attach_extruded_panel(root, seg_w, seg_h, thickness, f"{name}-seg-{seg_index}", (cx, 0.0, cz), color, side_color=side_color, include_lr_sides=include_lr_sides)
                    segments.append(front)
                    solids.extend(faces)
                else:
                    node = self._attach_card(root, seg_w, seg_h, f"{name}-seg-{seg_index}", (cx, 0.0, cz), color)
                    segments.append(node)
                    solids.append(node)
                seg_index += 1
        return {"root": root, "segments": segments, "solids": solids}

    def _character_palette(self, character: dict[str, Any]) -> dict[str, tuple[float, ...]]:
        return {
            "bone": tuple(character.get("bone_color", (0.10, 0.10, 0.11, 1.0))),
            "robe": tuple(character["body_color"]),
            "robe_inner": tuple(character.get("body_secondary_color", character["accent_color"])),
            "face": tuple(character.get("head_color", (0.97, 0.97, 0.95, 1.0))),
            "patch": tuple(character.get("patch_color", (0.08, 0.08, 0.09, 1.0))),
            "accent": tuple(character["accent_color"]),
            "blush": tuple(character.get("blush_color", (0.96, 0.71, 0.74, 0.30))),
            "mouth": tuple(character.get("mouth_color", (0.66, 0.28, 0.30, 1.0))),
        }

    def _ellipse_mask_image(self, image: Image.Image, padding_x: float = 0.04, padding_y: float = 0.02) -> Image.Image:
        rgba = image.convert("RGBA")
        width, height = rgba.size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        inset_x = max(2, int(width * padding_x))
        inset_y = max(2, int(height * padding_y))
        draw.ellipse((inset_x, inset_y, width - inset_x - 1, height - inset_y - 1), fill=255)
        rgba.putalpha(ImageChops.multiply(rgba.getchannel("A"), mask))
        return rgba

    def _limb_mask_image(self, image: Image.Image, inset_x_ratio: float = 0.14, inset_y_ratio: float = 0.03) -> Image.Image:
        rgba = image.convert("RGBA")
        width, height = rgba.size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        inset_x = max(2, int(width * inset_x_ratio))
        inset_y = max(1, int(height * inset_y_ratio))
        left = inset_x
        top = inset_y
        right = max(left + 2, width - inset_x - 1)
        bottom = max(top + 2, height - inset_y - 1)
        radius = max(2, min((right - left) // 2, int(width * 0.34)))
        draw.rounded_rectangle((left, top, right, bottom), radius=radius, fill=255)
        rgba.putalpha(ImageChops.multiply(rgba.getchannel("A"), mask))
        return rgba

    def _impact_mask_image(self, image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        pixels = rgba.load()
        width, height = rgba.size
        corners = [
            rgba.getpixel((0, 0)),
            rgba.getpixel((max(0, width - 1), 0)),
            rgba.getpixel((0, max(0, height - 1))),
            rgba.getpixel((max(0, width - 1), max(0, height - 1))),
        ]
        bg_r = int(sum(pixel[0] for pixel in corners) / len(corners))
        bg_g = int(sum(pixel[1] for pixel in corners) / len(corners))
        bg_b = int(sum(pixel[2] for pixel in corners) / len(corners))
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if a == 0:
                    continue
                dr = abs(r - bg_r)
                dg = abs(g - bg_g)
                db = abs(b - bg_b)
                distance = dr + dg + db
                if b > r and b > g:
                    if distance <= 54:
                        pixels[x, y] = (r, g, b, 0)
                    elif distance < 170:
                        alpha_scale = (distance - 54) / (170 - 54)
                        pixels[x, y] = (r, g, b, int(a * alpha_scale))
        bbox = rgba.getbbox()
        if bbox is None:
            return rgba
        left, top, right, bottom = bbox
        pad_x = max(2, int((right - left) * 0.08))
        pad_y = max(2, int((bottom - top) * 0.08))
        left = max(0, left - pad_x)
        top = max(0, top - pad_y)
        right = min(width, right + pad_x)
        bottom = min(height, bottom + pad_y)
        return rgba.crop((left, top, right, bottom))

    def _load_texture_sequence(self, image_path: Optional[str], variant_key: Optional[str] = None) -> Optional[dict[str, Any]]:
        if not image_path:
            return None
        resolved = str(Path(image_path).resolve())
        variant_version = {
            "ellipse-face": "v1",
            "impact-fx": "v3",
            "limb-skin": "v1",
        }.get(variant_key or "", "v1")
        cache_key = resolved if not variant_key else f"{resolved}::{variant_key}::{variant_version}"
        if cache_key in self._texture_sequences:
            return self._texture_sequences[cache_key]

        source = Path(resolved)
        if not source.exists():
            self._texture_sequences[cache_key] = None
            return None

        durations: list[int] = []
        textures = []
        with Image.open(source) as image:
            frame_count = max(1, int(getattr(image, "n_frames", 1)))
            for frame_index in range(frame_count):
                image.seek(frame_index)
                rgba = image.convert("RGBA")
                if variant_key == "ellipse-face":
                    rgba = self._ellipse_mask_image(rgba)
                elif variant_key == "impact-fx":
                    rgba = self._impact_mask_image(rgba)
                elif variant_key == "limb-skin":
                    rgba = self._limb_mask_image(rgba)
                textures.append(self._texture_from_pil_image(rgba, f"{cache_key}:{frame_index}"))
                durations.append(max(40, int(image.info.get("duration", 100) or 100)))

        sequence = {
            "textures": textures,
            "durations": durations,
            "size": textures[0].getOrigFileXSize(),
            "height": textures[0].getOrigFileYSize(),
        }
        self._texture_sequences[cache_key] = sequence
        return sequence

    def _texture_from_pil_image(self, image: Image.Image, key: str):
        Texture = self._core["Texture"]
        PNMImage = self._core["PNMImage"]
        StringStream = self._core["StringStream"]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        stream = StringStream(buffer.getvalue())
        pnm = PNMImage()
        if not pnm.read(stream, f"{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}.png"):
            raise RuntimeError(f"failed to decode in-memory texture for {key}")
        texture = Texture()
        if not texture.load(pnm):
            raise RuntimeError(f"failed to load in-memory texture for {key}")
        return texture

    def _texture_at_time(self, image_path: Optional[str], time_ms: int, variant_key: Optional[str] = None):
        sequence = self._load_texture_sequence(image_path, variant_key=variant_key)
        if not sequence:
            return None
        textures = sequence["textures"]
        durations = sequence["durations"]
        if len(textures) == 1:
            return textures[0]
        total = sum(durations)
        if total <= 0:
            return textures[0]
        position = time_ms % total
        cursor = 0
        for texture, duration in zip(textures, durations):
            cursor += duration
            if position < cursor:
                return texture
        return textures[-1]

    def _impact_effect_path(self) -> Optional[str]:
        candidate = ASSETS_DIR / "effects" / "hit.gif"
        if candidate.exists():
            return str(candidate)
        return None

    def _apply_texture(self, node, texture) -> None:
        if texture is None:
            return
        node.setTexture(texture, 1)
        node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        node.setColor(1.0, 1.0, 1.0, 1.0)

    def _shape_texture(self, key: str, size: tuple[int, int], alpha: float = 1.0):
        cached = self._shape_texture_cache.get(key)
        if cached is not None:
            return cached

        width = max(32, int(size[0]))
        height = max(32, int(size[1]))
        image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((1, 1, width - 2, height - 2), fill=(255, 255, 255, max(1, int(255 * alpha))))
        texture = self._texture_from_pil_image(image, f"shape:{key}:{width}x{height}:{alpha:.3f}")
        self._shape_texture_cache[key] = texture
        return texture

    def _character_skin_path(self, character: dict[str, Any], stem_candidates: list[str]) -> Optional[str]:
        search_dirs = []
        if character.get("character_dir"):
            skins_dir = Path(character["character_dir"]) / "skins"
            search_dirs.append(skins_dir)
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

    def _face_skin_path(self, character: dict[str, Any], expression: str, time_ms: int, speaking: bool = False) -> Optional[str]:
        mouth_open = speaking and (time_ms // 120) % 2 == 0
        talk_suffix = "open" if mouth_open else "closed"
        if expression == "fierce":
            candidates = [f"face_talk_angry_{talk_suffix}", "face_angry"]
        elif expression == "talk":
            candidates = [f"face_talk_neutral_{talk_suffix}", "face_neutral", "face_default"]
        elif expression == "explain":
            candidates = [f"face_talk_thinking_{talk_suffix}", f"face_talk_skeptical_{talk_suffix}", "face_thinking", "face_skeptical"]
        elif expression == "smirk":
            candidates = ["face_skeptical", "face_smile"]
        elif expression == "awkward":
            candidates = ["face_thinking", "face_skeptical"]
        elif expression == "deadpan":
            candidates = ["face_neutral", "face_default"]
        else:
            candidates = ["face_default", "face_neutral", "face_smile"]
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

    def _limb_skin_path(self, character: dict[str, Any], limb_name: str) -> Optional[str]:
        normalized = str(limb_name or "").strip().lower()
        if not normalized:
            return None
        aliases = {
            "upper_arm": ["upper_arm", "arm_upper", "limb_upper_arm"],
            "lower_arm": ["lower_arm", "arm_lower", "limb_lower_arm"],
            "upper_leg": ["upper_leg", "leg_upper", "limb_upper_leg"],
            "lower_leg": ["lower_leg", "leg_lower", "limb_lower_leg"],
        }
        candidates = aliases.get(normalized, [normalized])
        return self._character_skin_path(character, candidates)

    def _attach_segment(
        self,
        parent,
        start: tuple[float, float],
        end: tuple[float, float],
        width: float,
        name: str,
        color: tuple[float, ...],
    ):
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length = max(0.001, math.hypot(dx, dz))
        angle = math.degrees(math.atan2(dx, dz))
        return self._attach_card(
            parent,
            width,
            length,
            name,
            ((start[0] + end[0]) / 2.0, 0.0, (start[1] + end[1]) / 2.0),
            color,
            r=angle,
        )

    def _attach_joint(
        self,
        parent,
        point: tuple[float, float],
        radius: float,
        name: str,
        color: tuple[float, ...],
    ):
        return self._attach_card(parent, radius, radius, name, (point[0], 0.0, point[1]), color)

    def _attach_circle(
        self,
        parent,
        center: tuple[float, float],
        radius: float,
        name: str,
        color: tuple[float, ...],
        depth_y: float = 0.0,
        slices: int = 10,
    ):
        texture = self._shape_texture(f"{name}-circle-{int(radius * 256)}", (int(radius * 512), int(radius * 512)))
        node = self._attach_card(
            parent,
            radius * 2.0,
            radius * 2.0,
            name,
            (center[0], depth_y, center[1]),
            color,
        )
        self._apply_texture(node, texture)
        node.setColor(*tuple(color))

    def _attach_ellipse(
        self,
        parent,
        center: tuple[float, float],
        radius_x: float,
        radius_z: float,
        name: str,
        color: tuple[float, ...],
        depth_y: float = 0.0,
        slices: int = 12,
    ):
        texture = self._shape_texture(
            f"{name}-ellipse-{int(radius_x * 256)}-{int(radius_z * 256)}",
            (int(radius_x * 512), int(radius_z * 512)),
        )
        node = self._attach_card(
            parent,
            radius_x * 2.0,
            radius_z * 2.0,
            name,
            (center[0], depth_y, center[1]),
            color,
        )
        self._apply_texture(node, texture)
        node.setColor(*tuple(color))

    def _make_unit_card_node(self, parent, name: str):
        node = parent.attachNewNode(self._make_card(1.0, 1.0, name))
        node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        return node

    def _update_card_node(
        self,
        node,
        width: float,
        height: float,
        pos: tuple[float, float, float],
        color: tuple[float, ...],
        r: float = 0.0,
        texture=None,
        visible: bool = True,
    ) -> None:
        if not visible:
            node.hide()
            return
        node.show()
        node.setPos(*pos)
        node.setScale(max(0.001, width), 1.0, max(0.001, height))
        node.setR(r)
        if texture is None:
            node.clearTexture()
        else:
            node.setTexture(texture, 1)
        node.setColor(*color)
        node.setTransparency(self._core["TransparencyAttrib"].MAlpha)

    def _update_segment_node(
        self,
        node,
        start: tuple[float, float],
        end: tuple[float, float],
        width: float,
        color: tuple[float, ...],
        y: float = 0.0,
        texture=None,
        visible: bool = True,
    ) -> None:
        if not visible:
            node.hide()
            return
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length = max(0.001, math.hypot(dx, dz))
        angle = math.degrees(math.atan2(dx, dz))
        self._update_card_node(
            node,
            width,
            length,
            ((start[0] + end[0]) / 2.0, y, (start[1] + end[1]) / 2.0),
            color,
            r=angle,
            texture=texture,
            visible=True,
        )

    def _update_joint_node(
        self,
        node,
        point: tuple[float, float],
        radius: float,
        color: tuple[float, ...],
        y: float = 0.0,
        texture=None,
        visible: bool = True,
    ) -> None:
        self._update_card_node(
            node,
            radius,
            radius,
            (point[0], y, point[1]),
            color,
            texture=texture,
            visible=visible,
        )

    def _head_angle(self, expression: str, motion: str, mirror: float, time_ms: int) -> float:
        angle = math.sin(time_ms / 220) * 3.0 if expression in {"talk", "explain"} else 0.0
        if expression == "smirk":
            angle += -8.0 * mirror
        elif expression == "fierce":
            angle += -14.0 * mirror + math.sin(time_ms / 90) * 4.0
        elif expression == "awkward":
            angle += 10.0 * mirror
        elif expression == "deadpan":
            angle += -3.0 * mirror

        if motion == "point":
            angle += -6.0 * mirror
        elif motion in {"dragon-palm", "thunder-strike", "sword-arc"}:
            angle += -10.0 * mirror
        elif motion == "somersault":
            angle += math.sin(time_ms / 80) * 18.0
        elif motion == "handstand-walk":
            angle += 180.0
        elif motion == "big-jump":
            angle += math.sin(time_ms / 160) * 10.0
        elif motion == "dunk":
            angle += -12.0 * mirror
        return angle

    def _body_points(self, motion: str, mirror: float, time_ms: int) -> dict[str, tuple[float, float]]:
        swing = math.sin(time_ms / 130)
        step = math.sin(time_ms / 180)
        bob = abs(math.sin(time_ms / 120)) * 0.04 if motion in {"talk", "enter", "exit"} else 0.0

        points = {
            "pelvis": (0.0, 0.04 + bob * 0.3),
            "chest": (0.0, 0.72 + bob),
            "neck": (0.0, 1.00 + bob),
            "hip_left": (-0.16, 0.02 + bob * 0.2),
            "hip_right": (0.16, 0.02 + bob * 0.2),
            "knee_left": (-0.20, -0.50 - bob * 0.18),
            "knee_right": (0.20, -0.50 - bob * 0.18),
            "foot_left": (-0.22, -1.04),
            "foot_right": (0.22, -1.04),
            "shoulder_left": (-0.25, 0.82 + bob),
            "shoulder_right": (0.25, 0.82 + bob),
            "elbow_left": (-0.40, 0.46 + bob * 0.25),
            "elbow_right": (0.40, 0.46 + bob * 0.25),
            "hand_left": (-0.42, 0.02),
            "hand_right": (0.42, 0.02),
        }

        if motion == "talk":
            points["elbow_left"] = (-0.42, 0.54 + bob)
            points["hand_left"] = (-0.32, 0.10 + swing * 0.10)
            points["elbow_right"] = (0.54, 0.44 + bob)
            points["hand_right"] = (0.66, 0.14 - swing * 0.08)
        elif motion == "point":
            points["elbow_right"] = (0.64, 0.68 + bob)
            points["hand_right"] = (0.98, 0.82 + bob * 0.3)
            points["elbow_left"] = (-0.42, 0.48)
            points["hand_left"] = (-0.54, 0.08)
        elif motion in {"dragon-palm", "thunder-strike", "sword-arc"}:
            points["elbow_left"] = (-0.56, 0.68 + step * 0.04)
            points["hand_left"] = (-0.80, 0.92 + step * 0.05)
            points["elbow_right"] = (0.54, 0.72 - step * 0.04)
            points["hand_right"] = (0.82, 0.92 - step * 0.04)
            points["knee_left"] = (-0.10, -0.30)
            points["knee_right"] = (0.26, -0.56)
            points["foot_left"] = (-0.02, -0.90)
            points["foot_right"] = (0.34, -1.00)
        elif motion in {"enter", "exit"}:
            points["knee_left"] = (-0.12, -0.40 - step * 0.10)
            points["knee_right"] = (0.24, -0.60 + step * 0.10)
            points["foot_left"] = (-0.04, -0.98 + step * 0.05)
            points["foot_right"] = (0.32, -1.02 - step * 0.05)
        elif motion == "somersault":
            tuck = 0.10 + abs(math.sin(time_ms / 90)) * 0.08
            points["chest"] = (0.0, 0.52)
            points["neck"] = (0.0, 0.74)
            points["shoulder_left"] = (-0.22, 0.60)
            points["shoulder_right"] = (0.22, 0.60)
            points["elbow_left"] = (-0.44, 0.46)
            points["elbow_right"] = (0.44, 0.46)
            points["hand_left"] = (-0.28, 0.18)
            points["hand_right"] = (0.28, 0.18)
            points["hip_left"] = (-0.16, -0.02)
            points["hip_right"] = (0.16, -0.02)
            points["knee_left"] = (-0.20, -0.08 + tuck)
            points["knee_right"] = (0.20, -0.08 + tuck)
            points["foot_left"] = (-0.10, 0.18 + tuck)
            points["foot_right"] = (0.10, 0.18 + tuck)
        elif motion == "handstand-walk":
            stride = math.sin(time_ms / 110)
            points["pelvis"] = (0.0, -0.14)
            points["chest"] = (0.0, -0.62)
            points["neck"] = (0.0, -0.92)
            points["shoulder_left"] = (-0.24, -0.74)
            points["shoulder_right"] = (0.24, -0.74)
            points["elbow_left"] = (-0.34, -0.24 + stride * 0.04)
            points["elbow_right"] = (0.34, -0.24 - stride * 0.04)
            points["hand_left"] = (-0.24, 0.26 - stride * 0.08)
            points["hand_right"] = (0.24, 0.26 + stride * 0.08)
            points["hip_left"] = (-0.16, -0.12)
            points["hip_right"] = (0.16, -0.12)
            points["knee_left"] = (-0.24, 0.34 + stride * 0.14)
            points["knee_right"] = (0.24, 0.34 - stride * 0.14)
            points["foot_left"] = (-0.18, 0.90 + stride * 0.10)
            points["foot_right"] = (0.18, 0.90 - stride * 0.10)
        elif motion == "big-jump":
            kick = math.sin(time_ms / 130)
            points["pelvis"] = (0.0, 0.16)
            points["chest"] = (0.0, 0.88)
            points["neck"] = (0.0, 1.18)
            points["shoulder_left"] = (-0.30, 0.96)
            points["shoulder_right"] = (0.30, 0.96)
            points["elbow_left"] = (-0.52, 1.22)
            points["elbow_right"] = (0.52, 1.22)
            points["hand_left"] = (-0.36, 1.50)
            points["hand_right"] = (0.36, 1.50)
            points["knee_left"] = (-0.36, -0.14 + kick * 0.12)
            points["knee_right"] = (0.36, -0.14 - kick * 0.12)
            points["foot_left"] = (-0.66, -0.70 + kick * 0.08)
            points["foot_right"] = (0.66, -0.70 - kick * 0.08)
        elif motion == "dunk":
            points["pelvis"] = (0.0, 0.10)
            points["chest"] = (0.0, 0.84)
            points["neck"] = (0.0, 1.10)
            points["elbow_right"] = (0.40, 1.18)
            points["hand_right"] = (0.18, 1.68)
            points["elbow_left"] = (-0.62, 0.38)
            points["hand_left"] = (-0.80, -0.02)
            points["knee_left"] = (-0.30, -0.20)
            points["knee_right"] = (0.24, -0.32)
            points["foot_left"] = (-0.48, -0.82)
            points["foot_right"] = (0.28, -0.92)
        elif motion == "stagger":
            points["chest"] = (0.0, 0.62)
            points["neck"] = (0.0, 0.92)
            points["shoulder_left"] = (-0.26, 0.72)
            points["shoulder_right"] = (0.26, 0.76)
            points["elbow_left"] = (-0.44, 0.34)
            points["elbow_right"] = (0.48, 0.30)
            points["hand_left"] = (-0.28, -0.02)
            points["hand_right"] = (0.66, 0.16)
            points["knee_left"] = (-0.16, -0.42)
            points["knee_right"] = (0.28, -0.54)
            points["foot_left"] = (-0.10, -0.96)
            points["foot_right"] = (0.34, -0.92)
        elif motion == "knockback":
            points["pelvis"] = (0.0, -0.02)
            points["chest"] = (0.0, 0.56)
            points["neck"] = (0.0, 0.82)
            points["shoulder_left"] = (-0.26, 0.64)
            points["shoulder_right"] = (0.26, 0.64)
            points["elbow_left"] = (-0.46, 0.22)
            points["elbow_right"] = (0.46, 0.24)
            points["hand_left"] = (-0.60, -0.10)
            points["hand_right"] = (0.60, -0.04)
            points["knee_left"] = (-0.08, -0.26)
            points["knee_right"] = (0.24, -0.44)
            points["foot_left"] = (-0.24, -0.80)
            points["foot_right"] = (0.40, -0.90)
        elif motion == "knockdown":
            points["pelvis"] = (0.0, -0.24)
            points["chest"] = (0.0, 0.12)
            points["neck"] = (0.0, 0.36)
            points["shoulder_left"] = (-0.24, 0.16)
            points["shoulder_right"] = (0.24, 0.16)
            points["elbow_left"] = (-0.50, -0.06)
            points["elbow_right"] = (0.50, -0.06)
            points["hand_left"] = (-0.70, -0.18)
            points["hand_right"] = (0.70, -0.18)
            points["hip_left"] = (-0.16, -0.24)
            points["hip_right"] = (0.16, -0.24)
            points["knee_left"] = (-0.42, -0.28)
            points["knee_right"] = (0.42, -0.28)
            points["foot_left"] = (-0.74, -0.22)
            points["foot_right"] = (0.74, -0.20)

        mirrored: dict[str, tuple[float, float]] = {}
        for name, (x, z) in points.items():
            mirrored[name] = (x * mirror, z)
        return mirrored

    def _build_actor_instance(self, root, actor_id: str, label_text: Optional[str] = None) -> dict[str, Any]:
        runtime: dict[str, Any] = {"root": root}
        runtime["aura"] = self._make_unit_card_node(root, f"aura-{actor_id}")
        runtime["body_root"] = root.attachNewNode(f"body-root-{actor_id}")
        runtime["head_root"] = root.attachNewNode(f"head-root-{actor_id}")

        segment_names = [
            "spine-lower",
            "spine-upper",
            "clavicle",
            "arm-upper-left",
            "arm-lower-left",
            "arm-upper-right",
            "arm-lower-right",
            "leg-upper-left",
            "leg-lower-left",
            "leg-upper-right",
            "leg-lower-right",
            "hip-bar",
            "skin-spine-lower",
            "skin-spine-upper",
            "skin-clavicle",
            "skin-arm-upper-left",
            "skin-arm-lower-left",
            "skin-arm-upper-right",
            "skin-arm-lower-right",
            "skin-leg-upper-left",
            "skin-leg-lower-left",
            "skin-leg-upper-right",
            "skin-leg-lower-right",
        ]
        runtime["segments"] = {name: self._make_unit_card_node(runtime["body_root"], f"{name}-{actor_id}") for name in segment_names}

        circle_texture_64 = self._shape_texture(f"joint-circle-64", (64, 64))
        circle_texture_96 = self._shape_texture(f"joint-circle-96", (96, 96))
        ellipse_head_texture = self._shape_texture(f"head-ellipse-actor", (300, 228))
        ellipse_face_texture = self._shape_texture(f"face-core-actor", (256, 216), alpha=0.38)

        joint_names = [
            "neck",
            "shoulder_left",
            "shoulder_right",
            "elbow_left",
            "elbow_right",
            "hand_left",
            "hand_right",
            "hip_left",
            "hip_right",
            "knee_left",
            "knee_right",
            "foot_left",
            "foot_right",
            "hand_left_skin",
            "hand_right_skin",
            "foot_left_skin",
            "foot_right_skin",
        ]
        runtime["joints"] = {name: self._make_unit_card_node(runtime["body_root"], f"{name}-{actor_id}") for name in joint_names}
        runtime["joint_texture_small"] = circle_texture_64
        runtime["joint_texture_large"] = circle_texture_96
        runtime["waist_wrap"] = self._make_unit_card_node(runtime["body_root"], f"waist-wrap-{actor_id}")
        runtime["outfit_overlay"] = self._make_unit_card_node(runtime["body_root"], f"outfit-overlay-{actor_id}")

        runtime["head"] = self._make_unit_card_node(runtime["head_root"], f"head-{actor_id}")
        runtime["head_texture"] = ellipse_head_texture
        runtime["face_core"] = self._make_unit_card_node(runtime["head_root"], f"face-core-{actor_id}")
        runtime["face_core_texture"] = ellipse_face_texture
        runtime["headband"] = self._make_unit_card_node(runtime["head_root"], f"headband-{actor_id}")
        runtime["headband_tag"] = self._make_unit_card_node(runtime["head_root"], f"headband-tag-{actor_id}")
        runtime["head_radius"] = self._make_unit_card_node(runtime["head_root"], f"head-radius-{actor_id}")
        runtime["face_skin"] = self._make_unit_card_node(runtime["head_root"], f"face-skin-{actor_id}")
        runtime["ear_left"] = self._make_unit_card_node(runtime["head_root"], f"ear-left-{actor_id}")
        runtime["ear_right"] = self._make_unit_card_node(runtime["head_root"], f"ear-right-{actor_id}")
        runtime["patch_left"] = self._make_unit_card_node(runtime["head_root"], f"patch-left-{actor_id}")
        runtime["patch_right"] = self._make_unit_card_node(runtime["head_root"], f"patch-right-{actor_id}")
        runtime["eye_white_left"] = self._make_unit_card_node(runtime["head_root"], f"eye-white-left-{actor_id}")
        runtime["eye_white_right"] = self._make_unit_card_node(runtime["head_root"], f"eye-white-right-{actor_id}")
        runtime["pupil_left"] = self._make_unit_card_node(runtime["head_root"], f"pupil-left-{actor_id}")
        runtime["pupil_right"] = self._make_unit_card_node(runtime["head_root"], f"pupil-right-{actor_id}")
        runtime["brow_left"] = self._make_unit_card_node(runtime["head_root"], f"brow-left-{actor_id}")
        runtime["brow_right"] = self._make_unit_card_node(runtime["head_root"], f"brow-right-{actor_id}")
        runtime["nose"] = self._make_unit_card_node(runtime["head_root"], f"nose-{actor_id}")
        runtime["muzzle"] = self._make_unit_card_node(runtime["head_root"], f"muzzle-{actor_id}")
        runtime["cheek_left"] = self._make_unit_card_node(runtime["head_root"], f"cheek-left-{actor_id}")
        runtime["cheek_right"] = self._make_unit_card_node(runtime["head_root"], f"cheek-right-{actor_id}")
        runtime["mouth"] = self._make_unit_card_node(runtime["head_root"], f"mouth-{actor_id}")
        runtime["tongue"] = self._make_unit_card_node(runtime["head_root"], f"tongue-{actor_id}")
        runtime["sweat"] = self._make_unit_card_node(runtime["head_root"], f"sweat-{actor_id}")
        runtime["circle_texture_head"] = self._shape_texture(f"circle-head", (96, 96))
        runtime["circle_texture_patch"] = self._shape_texture(f"circle-patch", (144, 144))
        runtime["circle_texture_cheek"] = self._shape_texture(f"circle-cheek", (96, 64), alpha=0.72)

        label_runtime = None
        if label_text is not None:
            TextNode = self._core["TextNode"]
            text_node = TextNode(f"label-{actor_id}")
            text_node.setText(label_text)
            text_node.setAlign(TextNode.ACenter)
            text_node.setTextColor(1, 1, 1, 0.82)
            if self.text_font is not None:
                text_node.setFont(self.text_font)
            label_node = self.actor_label_root.attachNewNode(text_node)
            label_node.setScale(0.05)
            label_node.setBin("fixed", 15)
            label_runtime = {"text_node": text_node, "node": label_node}
        runtime["label"] = label_runtime
        return runtime

    def _update_actor_visuals(
        self,
        runtime: dict[str, Any],
        actor_id: str,
        character: dict[str, Any],
        palette: dict[str, tuple[float, ...]],
        motion: str,
        expression: str,
        mirror: float,
        time_ms: int,
        speaking: bool,
    ) -> None:
        points = self._body_points(motion, mirror, time_ms)
        show_bones = self.force_pose_skeleton or bool(character.get("show_bones", False))
        show_skin = not self.force_pose_skeleton
        bone_segments = [
            ("spine-lower", "pelvis", "chest"),
            ("spine-upper", "chest", "neck"),
            ("clavicle", "shoulder_left", "shoulder_right"),
            ("arm-upper-left", "shoulder_left", "elbow_left"),
            ("arm-lower-left", "elbow_left", "hand_left"),
            ("arm-upper-right", "shoulder_right", "elbow_right"),
            ("arm-lower-right", "elbow_right", "hand_right"),
            ("leg-upper-left", "hip_left", "knee_left"),
            ("leg-lower-left", "knee_left", "foot_left"),
            ("leg-upper-right", "hip_right", "knee_right"),
            ("leg-lower-right", "knee_right", "foot_right"),
            ("hip-bar", "hip_left", "hip_right"),
        ]
        for name, start_key, end_key in bone_segments:
            self._update_segment_node(runtime["segments"][name], points[start_key], points[end_key], 0.06, palette["bone"], visible=show_bones)
        for name in [
            "neck",
            "shoulder_left",
            "shoulder_right",
            "elbow_left",
            "elbow_right",
            "hand_left",
            "hand_right",
            "hip_left",
            "hip_right",
            "knee_left",
            "knee_right",
            "foot_left",
            "foot_right",
        ]:
            self._update_joint_node(
                runtime["joints"][name],
                points[name],
                0.10,
                palette["bone"],
                texture=runtime["joint_texture_small"],
                visible=show_bones,
            )

        skin_segments = [
            ("skin-spine-lower", "pelvis", "chest", 0.20, palette["robe"], None),
            ("skin-spine-upper", "chest", "neck", 0.14, palette["robe_inner"], None),
            ("skin-clavicle", "shoulder_left", "shoulder_right", 0.10, palette["accent"], None),
            ("skin-arm-upper-left", "shoulder_left", "elbow_left", 0.14, palette["robe"], "upper_arm"),
            ("skin-arm-lower-left", "elbow_left", "hand_left", 0.10, palette["robe_inner"], "lower_arm"),
            ("skin-arm-upper-right", "shoulder_right", "elbow_right", 0.14, palette["robe"], "upper_arm"),
            ("skin-arm-lower-right", "elbow_right", "hand_right", 0.10, palette["robe_inner"], "lower_arm"),
            ("skin-leg-upper-left", "hip_left", "knee_left", 0.18, palette["robe"], "upper_leg"),
            ("skin-leg-lower-left", "knee_left", "foot_left", 0.12, palette["robe_inner"], "lower_leg"),
            ("skin-leg-upper-right", "hip_right", "knee_right", 0.18, palette["robe"], "upper_leg"),
            ("skin-leg-lower-right", "knee_right", "foot_right", 0.12, palette["robe_inner"], "lower_leg"),
        ]
        for name, start_key, end_key, width, color, limb_skin in skin_segments:
            texture = None
            render_color = color
            if limb_skin:
                texture = self._texture_at_time(self._limb_skin_path(character, limb_skin), time_ms, variant_key="limb-skin")
                if texture is not None:
                    render_color = (1.0, 1.0, 1.0, 1.0)
            self._update_segment_node(
                runtime["segments"][name],
                points[start_key],
                points[end_key],
                width,
                render_color,
                texture=texture,
                visible=show_skin,
            )

        self._update_card_node(
            runtime["waist_wrap"],
            0.42,
            0.16,
            (points["pelvis"][0], -0.02, points["pelvis"][1] + 0.02),
            palette["accent"],
            visible=show_skin,
        )
        for hand_key in ("hand_left", "hand_right"):
            self._update_joint_node(
                runtime["joints"][f"{hand_key}_skin"],
                points[hand_key],
                0.12,
                palette["face"],
                texture=runtime["joint_texture_large"],
                visible=show_skin,
            )
        for foot_key in ("foot_left", "foot_right"):
            self._update_joint_node(
                runtime["joints"][f"{foot_key}_skin"],
                points[foot_key],
                0.14,
                palette["patch"],
                texture=runtime["joint_texture_large"],
                visible=show_skin,
            )

        outfit_texture = self._texture_at_time(self._outfit_skin_path(character), time_ms) if self.show_outfit_overlay else None
        shoulder_span = abs(points["shoulder_right"][0] - points["shoulder_left"][0])
        hip_span = abs(points["hip_right"][0] - points["hip_left"][0])
        torso_width = max(0.58, max(shoulder_span * 0.82, hip_span * 1.08))
        torso_height = max(0.86, abs(points["neck"][1] - points["pelvis"][1]) * 1.06)
        outfit_width = torso_width
        outfit_height = torso_height
        outfit_z = ((points["chest"][1] + points["pelvis"][1]) * 0.5) + 0.04
        if motion == "somersault":
            outfit_width = torso_width * 0.92
            outfit_height = torso_height * 0.82
            outfit_z = ((points["chest"][1] + points["pelvis"][1]) * 0.5) + 0.08
        elif motion == "handstand-walk":
            outfit_width = torso_width * 0.92
            outfit_height = torso_height * 0.88
            outfit_z = ((points["chest"][1] + points["pelvis"][1]) * 0.5)
        elif motion == "big-jump":
            outfit_width = torso_width * 0.96
            outfit_height = torso_height * 0.90
            outfit_z = ((points["chest"][1] + points["pelvis"][1]) * 0.5) + 0.05
        elif motion == "dunk":
            outfit_width = torso_width * 0.94
            outfit_height = torso_height * 0.88
            outfit_z = ((points["chest"][1] + points["pelvis"][1]) * 0.5) + 0.05
        self._update_card_node(
            runtime["outfit_overlay"],
            outfit_width,
            outfit_height,
            (0.0, -0.10, outfit_z),
            (1.0, 1.0, 1.0, 1.0),
            texture=outfit_texture,
            visible=outfit_texture is not None,
        )

        head_root = runtime["head_root"]
        head_root.setPos(points["neck"][0], -0.16, points["neck"][1] + 0.28)
        head_angle = self._head_angle(expression, motion, mirror, time_ms)
        head_root.setR(head_angle)
        head_radius_x = 0.58
        head_radius_z = 0.44
        if not self.show_head_overlay:
            self._update_card_node(
                runtime["head"],
                0.34,
                0.34,
                (0.0, 0.0, 0.0),
                palette["bone"],
                texture=runtime["circle_texture_head"],
            )
            for node_name in (
                "face_core",
                "headband",
                "headband_tag",
                "head_radius",
                "face_skin",
                "ear_left",
                "ear_right",
                "patch_left",
                "patch_right",
                "eye_white_left",
                "eye_white_right",
                "pupil_left",
                "pupil_right",
                "brow_left",
                "brow_right",
                "nose",
                "muzzle",
                "cheek_left",
                "cheek_right",
                "mouth",
                "tongue",
                "sweat",
            ):
                runtime[node_name].hide()
            return
        self._update_card_node(
            runtime["head"],
            head_radius_x * 2.0,
            head_radius_z * 2.0,
            (0.0, 0.0, 0.0),
            palette["face"],
            texture=runtime["head_texture"],
        )

        face_texture = self._texture_at_time(
            self._face_skin_path(character, expression, time_ms, speaking=speaking),
            time_ms,
            variant_key="ellipse-face",
        )
        self._update_card_node(
            runtime["face_core"],
            head_radius_x * 1.64,
            head_radius_z * 1.60,
            (0.0, -0.01, -0.01),
            (0.99, 0.99, 0.98, 0.38),
            texture=runtime["face_core_texture"],
            visible=face_texture is None,
        )
        self._update_card_node(runtime["headband"], 0.72, 0.10, (0.0, -0.05, 0.22), (*palette["accent"][:3], 0.92))
        self._update_card_node(runtime["headband_tag"], 0.08, 0.18, (0.31 * mirror, -0.08, 0.12), (*palette["accent"][:3], 0.92), r=8.0 * mirror)
        self._update_segment_node(runtime["head_radius"], (0.0, -0.06), (0.0, head_radius_z), 0.035, (*palette["patch"][:3], 0.72), y=0.0)
        anchor = character.get("head_anchor") or {}
        offset = anchor.get("offset") or [0, 0]
        face_scale = float(anchor.get("scale", 1.0) or 1.0)
        self._update_card_node(
            runtime["face_skin"],
            head_radius_x * 1.86 * face_scale,
            head_radius_z * 1.86 * face_scale,
            (
                float(offset[0]) / 256.0 * head_radius_x * 2.0,
                -0.19,
                float(offset[1]) / -256.0 * head_radius_z * 2.0,
            ),
            (1.0, 1.0, 1.0, 1.0),
            texture=face_texture,
            visible=face_texture is not None,
        )
        self._update_joint_node(runtime["ear_left"], (-0.40, 0.32), 0.36, palette["patch"], y=0.08, texture=runtime["circle_texture_head"])
        self._update_joint_node(runtime["ear_right"], (0.40, 0.32), 0.36, palette["patch"], y=0.08, texture=runtime["circle_texture_head"])

        procedural_face = face_texture is None
        self._update_card_node(runtime["patch_left"], 0.28, 0.24, (-0.23, -0.06, 0.08), palette["patch"], r=24.0, visible=procedural_face)
        self._update_card_node(runtime["patch_right"], 0.28, 0.24, (0.23, -0.06, 0.08), palette["patch"], r=-24.0, visible=procedural_face)

        blink_factor = 0.18 if time_ms % 2100 > 1930 else 1.0
        eye_width = 0.12
        eye_height = 0.05 * blink_factor
        pupil_height = 0.05 * blink_factor
        mouth_width = 0.18
        mouth_height = 0.03
        mouth_y = -0.23
        mouth_x = 0.0
        mouth_r = 0.0
        left_brow_r = 8.0
        right_brow_r = -8.0
        left_brow_z = 0.28
        right_brow_z = 0.28
        blush_alpha = palette["blush"][3]
        sweat_drop = False

        if expression == "deadpan":
            eye_height = 0.026 * blink_factor
            pupil_height = 0.020 * blink_factor
            mouth_height = 0.02
        elif expression == "talk":
            eye_height = 0.052 * blink_factor
            pupil_height = 0.048 * blink_factor
            mouth_width = 0.20
            mouth_height = 0.08 + (abs(math.sin(time_ms / 95)) * 0.04 if speaking else 0.0)
            left_brow_r = 4.0
            right_brow_r = -4.0
        elif expression == "explain":
            eye_height = 0.044 * blink_factor
            pupil_height = 0.036 * blink_factor
            mouth_width = 0.18
            mouth_height = 0.06 + (abs(math.sin(time_ms / 130)) * 0.02 if speaking else 0.0)
            left_brow_r = 5.0
            right_brow_r = -5.0
        elif expression == "smirk":
            eye_height = 0.030 * blink_factor
            pupil_height = 0.022 * blink_factor
            mouth_width = 0.18
            mouth_height = 0.022
            mouth_x = 0.08 * mirror
            mouth_r = -12.0 * mirror
            left_brow_r = 18.0 * mirror
            right_brow_r = -6.0 * mirror
        elif expression == "fierce":
            eye_height = 0.020 * blink_factor
            pupil_height = 0.016 * blink_factor
            mouth_width = 0.24
            mouth_height = 0.11 + (abs(math.sin(time_ms / 70)) * 0.04 if speaking else 0.0)
            mouth_y = -0.20
            left_brow_r = -26.0
            right_brow_r = 26.0
            left_brow_z = 0.25
            right_brow_z = 0.25
            blush_alpha = max(blush_alpha, 0.48)
        elif expression == "awkward":
            eye_height = 0.030 * blink_factor
            pupil_height = 0.026 * blink_factor
            mouth_width = 0.16
            mouth_height = 0.026
            mouth_y = -0.25
            blush_alpha = max(blush_alpha, 0.42)
            sweat_drop = True
        elif expression == "hurt":
            eye_height = 0.018 * blink_factor
            pupil_height = 0.014 * blink_factor
            mouth_width = 0.22
            mouth_height = 0.05
            mouth_y = -0.22
            mouth_r = -10.0 * mirror
            left_brow_r = -18.0
            right_brow_r = 18.0
            blush_alpha = max(blush_alpha, 0.52)
            sweat_drop = True

        eye_color = (0.97, 0.97, 0.98, 1.0)
        self._update_card_node(runtime["eye_white_left"], eye_width, eye_height, (-0.21, -0.10, 0.08), eye_color, visible=procedural_face)
        self._update_card_node(runtime["eye_white_right"], eye_width, eye_height, (0.21, -0.10, 0.08), eye_color, visible=procedural_face)
        self._update_card_node(runtime["pupil_left"], 0.04, max(0.012, pupil_height), (-0.20 + 0.01 * mirror, -0.12, 0.08), palette["patch"], visible=procedural_face)
        self._update_card_node(runtime["pupil_right"], 0.04, max(0.012, pupil_height), (0.22 + 0.01 * mirror, -0.12, 0.08), palette["patch"], visible=procedural_face)
        self._update_card_node(runtime["brow_left"], 0.18, 0.03, (-0.20, -0.14, left_brow_z), palette["patch"], r=left_brow_r, visible=procedural_face)
        self._update_card_node(runtime["brow_right"], 0.18, 0.03, (0.20, -0.14, right_brow_z), palette["patch"], r=right_brow_r, visible=procedural_face)
        self._update_card_node(runtime["nose"], 0.08, 0.06, (0.0, -0.13, -0.05), palette["patch"], visible=procedural_face)
        self._update_card_node(runtime["muzzle"], 0.24, 0.10, (0.0, -0.15, -0.15), (0.96, 0.95, 0.93, 0.62), visible=procedural_face)
        self._update_card_node(runtime["cheek_left"], 0.15, 0.08, (-0.28, -0.16, -0.10), (*palette["blush"][:3], blush_alpha), visible=procedural_face)
        self._update_card_node(runtime["cheek_right"], 0.15, 0.08, (0.28, -0.16, -0.10), (*palette["blush"][:3], blush_alpha), visible=procedural_face)
        self._update_card_node(
            runtime["mouth"],
            mouth_width,
            mouth_height,
            (mouth_x, -0.18, mouth_y),
            palette["mouth"] if mouth_height > 0.03 else palette["patch"],
            r=mouth_r,
            visible=procedural_face,
        )
        self._update_card_node(
            runtime["tongue"],
            mouth_width * 0.56,
            mouth_height * 0.45,
            (mouth_x, -0.19, mouth_y - 0.02),
            (0.96, 0.58, 0.64, 0.84),
            r=mouth_r,
            visible=procedural_face and mouth_height > 0.05,
        )
        self._update_card_node(
            runtime["sweat"],
            0.07,
            0.14,
            (0.36 * mirror, -0.16, 0.02),
            (0.72, 0.89, 1.0, 0.95),
            r=10.0 * mirror,
            visible=procedural_face and sweat_drop,
        )

    def _build_impact_instances(self, scene: dict[str, Any]) -> None:
        self._impact_instances = []
        effect_path = self._impact_effect_path()
        effect_sequence = self._load_texture_sequence(effect_path, variant_key="impact-fx") if effect_path else None
        circle_texture = self._shape_texture("impact-ring", (196, 196), alpha=1.0)
        for event in self._scene_combat_data(scene)["events"]:
            root = self.effects_root.attachNewNode(f"impact-{event['target_id']}-{event['time_ms']}")
            gif_node = self._make_unit_card_node(root, "impact-gif")
            rays = [self._make_unit_card_node(root, f"impact-ray-{idx}") for idx in range(12)]
            rings = [self._make_unit_card_node(root, f"impact-ring-{idx}") for idx in range(2)]
            self._impact_instances.append(
                {
                    "event": event,
                    "root": root,
                    "gif_node": gif_node,
                    "effect_path": effect_path,
                    "effect_sequence": effect_sequence,
                    "rays": rays,
                    "rings": rings,
                    "ring_texture": circle_texture,
                }
            )

    def _scene_key(self, scene: dict[str, Any]) -> str:
        return str(scene.get("id") or scene.get("summary") or id(scene))

    def _prepare_background(self, scene: dict[str, Any]) -> None:
        background = self.backgrounds[scene["background"]]
        floor = self.floors.get(scene.get("floor") or background.get("floor_id")) if (scene.get("floor") or background.get("floor_id")) else None
        back_color = tuple(background["sky_color"])
        floor_color = tuple((floor or background).get("color", background["ground_color"]) if floor else background["ground_color"])
        floor_color = self._ensure_floor_contrast(back_color, floor_color)
        accent_color = tuple(background.get("accent_color", background["sky_color"]))
        background_width = float(self.stage_layout["background_width"])
        background_height = float(self.stage_layout["background_height"])
        background_y = float(self.stage_layout["background_y"])
        background_z = float(self.stage_layout["background_z"])
        ground_width = float(self.stage_layout["ground_width"])
        ground_height = float(self.stage_layout["ground_height"])
        ground_y = float(self.stage_layout["ground_y"])
        ground_z = float(self.stage_layout["ground_z"])
        ground_pitch = float(self.stage_layout["ground_pitch"])
        back_np = {
            "root": self.background_root.attachNewNode("background-flat-root"),
            "segments": [],
        }
        back_card = self._attach_card(back_np["root"], background_width, background_height, "background-flat", (0.0, background_y, background_z), back_color)
        back_np["segments"].append(back_card)

        ground_np = {
            "root": self.background_root.attachNewNode("ground-flat-root"),
            "segments": [],
        }
        ground_card = self._attach_card(ground_np["root"], ground_width, ground_height, "ground-flat", (0.0, ground_y, ground_z), floor_color, p=ground_pitch)
        ground_np["segments"].append(ground_card)

        left_np = {"root": self.background_root.attachNewNode("left-flat-root"), "segments": []}
        right_np = {"root": self.background_root.attachNewNode("right-flat-root"), "segments": []}
        ceiling_np = {"root": self.background_root.attachNewNode("ceiling-flat-root"), "segments": []}
        outside_back_np = {"root": self.background_root.attachNewNode("outside-back-flat-root"), "segments": []}
        outside_left_np = {"root": self.background_root.attachNewNode("outside-left-flat-root"), "segments": []}
        outside_right_np = {"root": self.background_root.attachNewNode("outside-right-flat-root"), "segments": []}

        veil_np = self.background_root.attachNewNode(self._make_card(14.0, 8.4, "background-veil"))
        veil_np.setPos(0.0, 7.98, 0.20)
        veil_np.setColor(0.0, 0.0, 0.0, 0.0)
        veil_np.setTransparency(self._core["TransparencyAttrib"].MAlpha)

        self._background_nodes = {
            "background_id": scene["background"],
            "floor_id": scene.get("floor") or background.get("floor_id"),
            "outside_back": outside_back_np,
            "outside_left": outside_left_np,
            "outside_right": outside_right_np,
            "background": back_np,
            "left_wall": left_np,
            "right_wall": right_np,
            "ground": ground_np,
            "ceiling": ceiling_np,
            "opening_counts": {"background": 0, "left_wall": 0, "right_wall": 0},
            "veil": veil_np,
            "surface_positions": {
                "back-wall": (0.0, 7.96, 0.20),
                "left-wall": (-5.2, 7.2, 0.10),
                "right-wall": (5.2, 7.2, 0.10),
                "outside-back": (0.0, 8.4, 0.08),
                "outside-left": (-6.4, 7.5, 0.08),
                "outside-right": (6.4, 7.5, 0.08),
            },
            "surface_defaults": {
                "background": back_color,
                "left_wall": accent_color,
                "right_wall": accent_color,
                "ground": floor_color,
                "ceiling": accent_color,
            },
        }

    def _update_background(self, scene: dict[str, Any], time_ms: int) -> None:
        nodes = self._background_nodes
        surface_defaults = nodes.get("surface_defaults", {})
        surface_paths = {
            "outside_back": None,
            "outside_left": None,
            "outside_right": None,
            "background": None,
            "left_wall": None,
            "right_wall": None,
            "ground": None,
            "ceiling": None,
        }
        for key in ("outside_back", "outside_left", "outside_right", "background", "left_wall", "right_wall", "ground", "ceiling"):
            bundle = nodes[key]
            if nodes.get("opening_counts", {}).get(key, 0) > 0:
                texture = None
            else:
                texture = self._texture_at_time(surface_paths.get(key), time_ms)
            for node in bundle["segments"]:
                if texture is not None:
                    self._apply_texture(node, texture)
                    tint = surface_defaults.get(key, (1.0, 1.0, 1.0, 1.0))
                    node.setColorScale(tint[0], tint[1], tint[2], tint[3])
                    node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                    node.setAlphaScale(1.0)
                else:
                    node.clearTexture()
                    fallback = surface_defaults.get(key, (0.08, 0.12, 0.18, 1.0))
                    node.setColor(*fallback)
                    node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                    node.setAlphaScale(fallback[3] if len(fallback) == 4 else 1.0)
        nodes["veil"].setAlphaScale(0.0)

    def _prepare_props(self, scene: dict[str, Any]) -> None:
        self._prop_instances = []
        for item in scene.get("props", []):
            prop = self.props[item["prop_id"]]
            mount = str(item.get("mount") or prop.get("default_mount") or "free")
            render_style = str(prop.get("render_style") or "sprite")
            if mount in {"back-wall", "left-wall", "right-wall"} and render_style in {"window", "door", "double-door"}:
                continue
            scale = float(item.get("scale", 1.0))
            base_width = float(prop.get("width") or 0.0) or float(prop.get("base_width", 160)) / 140.0
            base_height = float(prop.get("height") or 0.0) or float(prop.get("base_height", 120)) / 140.0
            width = base_width * scale
            height = base_height * scale
            root = self.props_root.attachNewNode(f"prop-{item['prop_id']}")
            frame_node = None
            mat_node = None
            glass_node = None
            art_width = width
            art_height = height
            if render_style == "frame":
                padding = min(0.35, max(0.02, float(prop.get("frame_padding", 0.10) or 0.10)))
                border = max(0.06, min(width, height) * padding)
                frame_node = self._attach_rect_frame(root, width, height, border, f"prop-frame-{item['prop_id']}")
                frame_node.setPos(0, 0.03, 0)
                frame_node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                mat_scale = max(0.1, 1.0 - padding * 1.35)
                art_scale = max(0.1, 1.0 - padding * 2.1)
                mat_node = root.attachNewNode(self._make_card(width * mat_scale, height * mat_scale, f"prop-mat-{item['prop_id']}"))
                mat_node.setPos(0, 0.01, 0)
                mat_node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                node = root.attachNewNode(self._make_card(width * art_scale, height * art_scale, f"prop-art-{item['prop_id']}"))
                node.setPos(0, 0.02, 0)
                node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                art_width = width * art_scale
                art_height = height * art_scale
            elif render_style in {"window", "door", "double-door"}:
                padding = min(0.28, max(0.04, float(prop.get("frame_padding", 0.10) or 0.10)))
                border = max(0.08, min(width, height) * padding * 0.95)
                frame_node = self._attach_rect_frame(root, width, height, border, f"prop-frame-{item['prop_id']}")
                frame_node.setPos(0, 0.03, 0.0)
                frame_node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                opening_scale_x = max(0.1, 1.0 - padding * 2.0)
                opening_scale_z = max(0.1, 1.0 - padding * (1.6 if render_style == "window" else 1.2))
                if render_style == "window":
                    opening_scale_z *= 0.74
                if render_style == "double-door":
                    opening_scale_x *= 0.96
                glass_node = root.attachNewNode(self._make_card(width * opening_scale_x, height * opening_scale_z, f"prop-glass-{item['prop_id']}"))
                glass_node.setPos(0, 0.02, 0.04 if render_style == "window" else -0.08)
                glass_node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                node = root.attachNewNode(self._make_card(width * opening_scale_x, height * opening_scale_z, f"prop-view-{item['prop_id']}"))
                node.setPos(0, 0.01, 0.04 if render_style == "window" else -0.08)
                node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                if render_style == "window":
                    mullion_v = frame_node.attachNewNode(self._make_card(width * 0.06, height * opening_scale_z * 0.92, f"prop-window-v-{item['prop_id']}"))
                    mullion_v.setPos(0, 0.03, 0.04)
                    mullion_h = frame_node.attachNewNode(self._make_card(width * opening_scale_x * 0.92, height * 0.06, f"prop-window-h-{item['prop_id']}"))
                    mullion_h.setPos(0, 0.031, 0.04)
                elif render_style == "double-door":
                    leaf_width = width * opening_scale_x * 0.44
                    leaf_height = height * opening_scale_z * 0.98
                    left_leaf = root.attachNewNode(self._make_card(leaf_width, leaf_height, f"prop-door-left-{item['prop_id']}"))
                    left_leaf.setPos(-width * 0.22, 0.028, -0.08)
                    left_leaf.setColor(tuple(prop.get("door_color") or prop.get("frame_color") or (0.34, 0.22, 0.11, 0.80)))
                    left_leaf.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                    left_leaf.setH(-8.0)
                    right_leaf = root.attachNewNode(self._make_card(leaf_width, leaf_height, f"prop-door-right-{item['prop_id']}"))
                    right_leaf.setPos(width * 0.22, 0.028, -0.08)
                    right_leaf.setColor(tuple(prop.get("door_color") or prop.get("frame_color") or (0.34, 0.22, 0.11, 0.80)))
                    right_leaf.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                    right_leaf.setH(8.0)
                    seam = root.attachNewNode(self._make_card(width * 0.03, height * opening_scale_z, f"prop-door-seam-{item['prop_id']}"))
                    seam.setPos(0, 0.034, -0.08)
                    seam.setColor(tuple(prop.get("frame_color") or (0.34, 0.22, 0.11, 1.0)))
                elif render_style == "door":
                    door_panel = root.attachNewNode(self._make_card(width * opening_scale_x * 0.94, height * opening_scale_z * 0.98, f"prop-door-panel-{item['prop_id']}"))
                    door_panel.setPos(0, 0.028, -0.08)
                    door_panel.setColor(tuple(prop.get("door_color") or prop.get("frame_color") or (0.34, 0.22, 0.11, 0.56)))
                    door_panel.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                art_width = width * opening_scale_x
                art_height = height * opening_scale_z
            else:
                node = root.attachNewNode(self._make_card(width, height, f"prop-{item['prop_id']}"))
                node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
            self._prop_instances.append(
                {
                    "item": item,
                    "prop": prop,
                    "root": root,
                    "node": node,
                    "frame_node": frame_node,
                    "mat_node": mat_node,
                    "glass_node": glass_node,
                    "width": width,
                    "height": height,
                    "art_width": art_width,
                    "art_height": art_height,
                    "anchor": prop.get("anchor") or [0.5, 1.0],
                }
            )

    def _update_props(self, time_ms: int) -> None:
        camera_state = self._camera_state(self._current_scene, time_ms) if hasattr(self, "_current_scene") else {"x": 0.0, "z": 0.0}
        for instance in self._prop_instances:
            item = instance["item"]
            prop = instance["prop"]
            root = instance["root"]
            node = instance["node"]
            width = instance["width"]
            height = instance["height"]
            anchor = instance["anchor"]
            layer = item.get("layer") or prop.get("default_layer") or "front"
            y = 6.4 if layer == "back" else 5.5
            x = float(item.get("x", 0.0))
            z = float(item.get("z", -1.0))
            if prop.get("motion") == "float":
                z += math.sin(time_ms / max(200, int(prop.get("motion_period_ms", 1400)))) * 0.12
            elif prop.get("motion") == "drift":
                x += math.sin(time_ms / max(200, int(prop.get("motion_period_ms", 1800)))) * 0.20
            x += float(prop.get("motion_x", 0.0) or 0.0)
            z += float(prop.get("motion_y", 0.0) or 0.0)
            mount = str(item.get("mount") or prop.get("default_mount") or "free")
            if mount == "free":
                root.setPos(
                    x + (0.5 - float(anchor[0])) * width,
                    y,
                    z + (0.5 - float(anchor[1])) * height,
                )
            frame_node = instance.get("frame_node")
            if frame_node is not None:
                self._set_subtree_color(frame_node, tuple(prop.get("frame_color") or (0.36, 0.24, 0.12, 1.0)))
            mat_node = instance.get("mat_node")
            if mat_node is not None:
                mat_node.setColor(tuple(prop.get("mat_color") or (0.94, 0.92, 0.88, 1.0)))
            glass_node = instance.get("glass_node")
            if glass_node is not None:
                glass_node.setColor(tuple(prop.get("glass_color") or (0.78, 0.90, 1.0, 0.26)))
            asset_path = prop.get("asset_path")
            if item.get("image_url"):
                asset_path = str(_cached_remote_asset(item.get("image_url"), "props") or asset_path or "")
            if mount in {"back-wall", "left-wall", "right-wall", "outside-back", "outside-left", "outside-right"}:
                surface_positions = self._background_nodes.get("surface_positions", {})
                base_x, base_y, base_z = surface_positions.get(mount, (0.0, y, 0.0))
                if mount == "back-wall":
                    root.setPos(base_x + x, base_y, base_z + z)
                    root.setHpr(0.0, 0.0, 0.0)
                elif mount == "outside-back":
                    root.setPos(base_x + x - camera_state["x"] * 0.28, base_y, base_z + z - camera_state["z"] * 0.12)
                    root.setHpr(0.0, 0.0, 0.0)
                elif mount == "left-wall":
                    root.setPos(base_x, base_y + x * 0.12, base_z + z)
                    root.setHpr(self.room_wall_angle, 0.0, 0.0)
                elif mount == "outside-left":
                    root.setPos(base_x, base_y + x * 0.12 + camera_state["x"] * 0.10, base_z + z - camera_state["z"] * 0.08)
                    root.setHpr(self.room_wall_angle, 0.0, 0.0)
                elif mount == "right-wall":
                    root.setPos(base_x, base_y - x * 0.12, base_z + z)
                    root.setHpr(-self.room_wall_angle, 0.0, 0.0)
                elif mount == "outside-right":
                    root.setPos(base_x, base_y - x * 0.12 - camera_state["x"] * 0.10, base_z + z - camera_state["z"] * 0.08)
                    root.setHpr(-self.room_wall_angle, 0.0, 0.0)
            texture = self._texture_at_time(asset_path, time_ms)
            if texture is not None:
                self._apply_texture(node, texture)
                node.setColorScale(1.0, 1.0, 1.0, 1.0)
            else:
                node.clearTexture()
                if str(prop.get("render_style") or "") in {"window", "door", "double-door"}:
                    node.setColor(1.0, 1.0, 1.0, 0.0)
                    node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                else:
                    node.setColor(tuple(prop["color"]))

    def _prepare_actors(self, scene: dict[str, Any]) -> None:
        self._actor_instances = {}
        for actor in scene.get("actors", []):
            actor_id = actor["actor_id"]
            actor_meta = self.cast.get(actor_id, {})
            self._actor_instances[actor_id] = {
                "root": self.actors_root.attachNewNode(f"actor-{actor_id}"),
                "runtime": None,
            }
            self._actor_instances[actor_id]["runtime"] = self._build_actor_instance(
                self._actor_instances[actor_id]["root"],
                actor_id,
                actor_meta.get("display_name") or actor_id,
            )

    def _stable_ratio(self, *parts: Any) -> float:
        digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def _lerp(self, start: float, end: float, ratio: float) -> float:
        return start + (end - start) * ratio

    def _npc_layer_bias(self, layer: str) -> float:
        return {
            "back": -0.85,
            "mid": 0.0,
            "front": 0.80,
        }.get(str(layer or "mid"), 0.0)

    def _cleanup_npc_scene_state(self, scene_key: str) -> None:
        state = self._npc_scene_states.pop(scene_key, None)
        if state is None:
            return
        world = state.get("world")
        if world is not None:
            for npc in state.get("npcs", []):
                try:
                    world.removeAiChar(str(npc.get("ai_name") or npc.get("id") or ""))
                except Exception:
                    pass
        ai_root = state.get("ai_root")
        if ai_root is not None:
            ai_root.detachNode()

    def _npc_spawn_state(self, scene: dict[str, Any], group: dict[str, Any], index: int) -> dict[str, float]:
        area = group.get("area") or {}
        rx = self._stable_ratio(scene.get("id"), group.get("id"), index, "x")
        rf = self._stable_ratio(scene.get("id"), group.get("id"), index, "front")
        rs = self._stable_ratio(scene.get("id"), group.get("id"), index, "scale")
        x = self._lerp(float(area.get("x_min", -4.8)), float(area.get("x_max", 4.8)), rx)
        frontness = self._lerp(float(area.get("front_min", -0.8)), float(area.get("front_max", 0.9)), rf)
        scale = self._lerp(float(group.get("scale_min", 0.72)), float(group.get("scale_max", 0.88)), rs)
        return {"x": x, "frontness": frontness, "scale": scale}

    def _npc_target_offset(self, group: dict[str, Any], index: int, count: int) -> tuple[float, float]:
        angle = (math.tau * index) / max(1, count)
        radius_x = 0.55 + (index % 3) * 0.22
        radius_front = 0.18 + (index % 4) * 0.08
        if str(group.get("behavior") or "") == "guard":
            radius_x *= 0.72
            radius_front *= 0.65
        return (math.cos(angle) * radius_x, math.sin(angle) * radius_front)

    def _npc_target_point(self, scene: dict[str, Any], group: dict[str, Any], npc: dict[str, Any], time_ms: int) -> tuple[float, float]:
        target_actor_id = str(group.get("target_actor_id") or "").strip()
        if target_actor_id:
            pose = self._find_actor_pose(scene, target_actor_id, time_ms)
            target_x = float(pose.get("x", 0.0))
            target_frontness = self._actor_frontness(scene, target_actor_id)
        else:
            anchor = group.get("anchor") or {}
            target_x = float(anchor.get("x", 0.0) or 0.0)
            target_frontness = float(anchor.get("frontness", 0.0) or 0.0)
        offset_x, offset_front = npc.get("target_offset", (0.0, 0.0))
        area = group.get("area") or {}
        target_x += offset_x
        target_frontness += offset_front
        target_x = max(float(area.get("x_min", -4.8)), min(float(area.get("x_max", 4.8)), target_x))
        target_frontness = max(float(area.get("front_min", -0.8)), min(float(area.get("front_max", 0.9)), target_frontness))
        return (target_x, target_frontness)

    def _npc_target_cache(self, scene: dict[str, Any], state: dict[str, Any], time_ms: int) -> dict[str, tuple[float, float]]:
        cache = state.setdefault("target_cache", {})
        cached = cache.get(time_ms)
        if cached is not None:
            return cached
        target_map: dict[str, tuple[float, float]] = {}
        for npc in state.get("npcs", []):
            target_map[str(npc["id"])] = self._npc_target_point(scene, npc["group"], npc, time_ms)
        cache.clear()
        cache[time_ms] = target_map
        return target_map

    def _prepare_npcs(self, scene: dict[str, Any], force_rebuild: bool = False) -> None:
        scene_key = self._scene_key(scene)
        if force_rebuild:
            self._detach_children(self.npcs_root)
            self._npc_instances = []
            self._cleanup_npc_scene_state(scene_key)
        if scene_key in self._npc_scene_states:
            self._npc_instances = list(self._npc_scene_states[scene_key].get("npcs", []))
            return
        self._npc_instances = []
        groups = scene.get("npc_groups") or []
        if not groups:
            return
        AIWorld = self._core["AIWorld"]
        AICharacter = self._core["AICharacter"]
        fps = int(self.story["video"].get("fps", 12) or 12)
        ai_root = self.stage_root.attachNewNode(f"npc-ai-{scene_key}")
        ai_root.hide()
        world = AIWorld(ai_root)
        state = {
            "ai_root": ai_root,
            "world": world,
            "step_ms": max(40, round(1000 / max(1, fps))),
            "last_time_ms": 0,
            "target_cache": {},
            "npcs": [],
        }
        for group_index, group in enumerate(groups):
            count = int(group.get("count", 0) or 0)
            asset_ids = list(group.get("asset_ids") or [])
            for npc_index in range(count):
                spawn = self._npc_spawn_state(scene, group, npc_index)
                npc_id = f"{group.get('id')}-{npc_index+1:02d}"
                asset_id = asset_ids[npc_index % len(asset_ids)] if asset_ids else next(iter(self.characters))
                visual_root = self.npcs_root.attachNewNode(f"npc-{npc_id}")
                ai_node = ai_root.attachNewNode(f"npc-ai-node-{npc_id}")
                ai_node.setPos(float(spawn["x"]), 0.0, float(spawn["frontness"]))
                move_force = max(0.10, 0.16 * float(group.get("speed", 1.0) or 1.0))
                max_force = max(6.0, 12.0 * float(group.get("speed", 1.0) or 1.0))
                ai_name = f"npc-{npc_id}"
                ai_char = AICharacter(ai_name, ai_node, 100.0, move_force, max_force)
                world.addAiChar(ai_char)
                behaviors = ai_char.getAiBehaviors()
                target_node = ai_root.attachNewNode(f"npc-target-{npc_id}")
                target_offset = self._npc_target_offset(group, npc_index, count)
                behavior = str(group.get("behavior") or "wander")
                if behavior == "wander":
                    behaviors.wander(float(group.get("wander_radius", 0.9) or 0.9), 0, float(group.get("wander_aoi", 65.0) or 65.0), float(group.get("seek_weight", 1.0) or 1.0))
                elif behavior == "seek":
                    behaviors.seek(target_node, float(group.get("seek_weight", 1.0) or 1.0))
                    behaviors.arrival(float(group.get("arrival_distance", 0.65) or 0.65))
                elif behavior == "pursue":
                    behaviors.pursue(target_node, float(group.get("seek_weight", 1.0) or 1.0))
                    behaviors.arrival(float(group.get("arrival_distance", 0.65) or 0.65))
                elif behavior == "evade":
                    behaviors.evade(
                        target_node,
                        float(group.get("evade_distance", 1.8) or 1.8),
                        float(group.get("relax_distance", 2.8) or 2.8),
                        float(group.get("seek_weight", 1.0) or 1.0),
                    )
                elif behavior == "guard":
                    behaviors.seek(target_node, float(group.get("seek_weight", 1.0) or 1.0))
                    behaviors.arrival(float(group.get("arrival_distance", 0.65) or 0.65))
                npc = {
                    "id": npc_id,
                    "group_id": str(group.get("id") or f"group-{group_index+1:02d}"),
                    "group": group,
                    "ai_name": ai_name,
                    "asset_id": asset_id,
                    "root": visual_root,
                    "ai_node": ai_node,
                    "ai_char": ai_char,
                    "target_node": target_node,
                    "target_offset": target_offset,
                    "spawn": spawn,
                    "scale": float(spawn["scale"]),
                    "facing": "right" if target_offset[0] >= 0 else "left",
                    "last_x": float(spawn["x"]),
                    "last_frontness": float(spawn["frontness"]),
                    "watch": bool(group.get("watch", True)),
                }
                npc["runtime"] = self._build_actor_instance(visual_root, npc_id, None)
                state["npcs"].append(npc)
                self._npc_instances.append(npc)
        self._npc_scene_states[scene_key] = state

    def _advance_npc_ai(self, scene: dict[str, Any], time_ms: int) -> None:
        scene_key = self._scene_key(scene)
        state = self._npc_scene_states.get(scene_key)
        if state is None or not state.get("npcs"):
            return
        last_time_ms = int(state.get("last_time_ms", 0) or 0)
        if time_ms < last_time_ms:
            self._prepare_npcs(scene, force_rebuild=True)
            state = self._npc_scene_states.get(scene_key)
            if state is None:
                return
            last_time_ms = 0
            state["last_time_ms"] = 0
        target_ms = int(time_ms)
        if target_ms == last_time_ms:
            return
        current_ms = int(last_time_ms or 0)
        step_ms = int(state.get("step_ms", 80) or 80)
        while current_ms < target_ms:
            next_ms = min(target_ms, current_ms + step_ms)
            target_map = self._npc_target_cache(scene, state, next_ms)
            for npc in state.get("npcs", []):
                target_x, target_frontness = target_map[str(npc["id"])]
                npc["target_node"].setPos(target_x, 0.0, target_frontness)
            self._clock.setDt(max(1, next_ms - current_ms) / 1000.0)
            self._clock.tick()
            state["world"].update()
            for npc in state.get("npcs", []):
                area = npc["group"].get("area") or {}
                ai_pos = npc["ai_node"].getPos()
                clamped_x = max(float(area.get("x_min", -4.8)), min(float(area.get("x_max", 4.8)), float(ai_pos.x)))
                clamped_front = max(float(area.get("front_min", -0.8)), min(float(area.get("front_max", 0.9)), float(ai_pos.z)))
                npc["ai_node"].setPos(clamped_x, 0.0, clamped_front)
            current_ms = next_ms
        state["last_time_ms"] = target_ms

    def _prepare_scene_label(self, scene: dict[str, Any]) -> None:
        title = f"{self.story['meta'].get('title', '')} | {scene.get('summary', '')}".strip(" |")
        self._overlay_text(title[:72], pos=(0.0, 0.90), scale=0.045, fg=(1.0, 0.98, 0.92, 0.88), parent=self.scene_label_root)

    def _prepare_scene(self, scene: dict[str, Any]) -> None:
        scene_key = self._scene_key(scene)
        if self._prepared_scene_key == scene_key:
            return
        self._reset_scene()
        self._prepare_background(scene)
        self._prepare_props(scene)
        self._prepare_actors(scene)
        self._prepare_npcs(scene)
        self._build_impact_instances(scene)
        self._prepare_scene_label(scene)
        self._scene_combat_data(scene)
        self._prepared_scene_key = scene_key

    def _base_actor_pose(self, scene: dict[str, Any], actor_id: str, time_ms: int) -> dict[str, Any]:
        actor = next(item for item in scene["actors"] if item["actor_id"] == actor_id)
        pose = {
            "x": float(actor["spawn"]["x"]),
            "z": float(actor["spawn"]["z"]),
            "scale": float(actor.get("scale", 1.0)),
            "facing": actor.get("facing") or "right",
            "motion": "idle",
            "effect": None,
            "emotion": "neutral",
            "expression": None,
        }
        active_beats = [
            beat
            for beat in scene.get("beats", [])
            if beat.get("actor_id") == actor_id and int(beat["start_ms"]) <= time_ms <= int(beat["end_ms"])
        ]
        for beat in active_beats:
            pose["motion"] = beat.get("motion") or pose["motion"]
            pose["effect"] = beat.get("effect") or pose["effect"]
            pose["emotion"] = beat.get("emotion") or pose["emotion"]
            if beat.get("expression") is not None:
                pose["expression"] = beat.get("expression")
            if beat.get("facing"):
                pose["facing"] = beat["facing"]
            if beat.get("from") and beat.get("to"):
                duration = max(1, int(beat["end_ms"]) - int(beat["start_ms"]))
                ratio = (time_ms - int(beat["start_ms"])) / duration
                pose["x"] = float(beat["from"]["x"]) + (float(beat["to"]["x"]) - float(beat["from"]["x"])) * ratio
                pose["z"] = float(beat["from"]["z"]) + (float(beat["to"]["z"]) - float(beat["from"]["z"])) * ratio
        return pose

    def _combat_profile(self, motion: str) -> Optional[dict[str, Any]]:
        profiles = {
            "dragon-palm": {"active": (0.26, 0.72), "forward": 1.55, "height": 0.90, "width": 1.25, "reaction": "knockback", "stun_ms": 720, "push_x": 1.10, "push_z": 0.38},
            "thunder-strike": {"active": (0.24, 0.58), "forward": 0.90, "height": 1.65, "width": 0.78, "reaction": "stagger", "stun_ms": 640, "push_x": 0.62, "push_z": 0.22},
            "sword-arc": {"active": (0.30, 0.62), "forward": 1.22, "height": 1.02, "width": 1.36, "reaction": "stagger", "stun_ms": 580, "push_x": 0.82, "push_z": 0.18},
            "somersault": {"active": (0.44, 0.82), "forward": 0.92, "height": 1.04, "width": 1.00, "reaction": "knockback", "stun_ms": 680, "push_x": 0.96, "push_z": 0.30},
            "big-jump": {"active": (0.40, 0.82), "forward": 0.88, "height": 1.30, "width": 0.96, "reaction": "knockback", "stun_ms": 760, "push_x": 0.90, "push_z": 0.48},
            "dunk": {"active": (0.46, 0.86), "forward": 1.04, "height": 1.44, "width": 0.98, "reaction": "knockdown", "stun_ms": 1080, "push_x": 1.15, "push_z": 0.62},
            "point": {"active": (0.50, 0.76), "forward": 0.74, "height": 0.80, "width": 0.44, "reaction": "stagger", "stun_ms": 380, "push_x": 0.34, "push_z": 0.04},
        }
        return profiles.get(motion)

    def _rects_intersect(self, a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
        return not (a[1] < b[0] or a[0] > b[1] or a[3] < b[2] or a[2] > b[3])

    def _attack_box(self, pose: dict[str, Any], time_ms: int) -> Optional[tuple[float, float, float, float]]:
        profile = self._combat_profile(str(pose.get("motion") or ""))
        beat = pose.get("_beat")
        if not profile or beat is None:
            return None
        progress = self._beat_progress(beat, time_ms)
        active_start, active_end = profile["active"]
        if not (active_start <= progress <= active_end):
            return None
        mirror = -1.0 if pose.get("facing") == "left" else 1.0
        center_x = float(pose["x"]) + mirror * float(profile["forward"])
        center_z = float(pose["z"]) + float(profile["height"]) * 0.5
        half_width = float(profile["width"]) * 0.5
        half_height = float(profile["height"]) * 0.5
        return (center_x - half_width, center_x + half_width, center_z - half_height, center_z + half_height)

    def _hurt_boxes(self, pose: dict[str, Any]) -> list[tuple[float, float, float, float]]:
        x = float(pose["x"])
        z = float(pose["z"])
        return [
            (x - 0.38, x + 0.38, z - 0.08, z + 0.98),
            (x - 0.30, x + 0.30, z + 0.84, z + 1.82),
        ]

    def _scene_combat_data(self, scene: dict[str, Any]) -> dict[str, Any]:
        scene_key = self._scene_key(scene)
        cached = self._scene_combat_cache.get(scene_key)
        if cached is not None:
            return cached

        actor_ids = [item["actor_id"] for item in scene.get("actors", [])]
        fps = int(self.story["video"].get("fps", 12) or 12)
        step_ms = max(40, round(1000 / max(1, fps)))
        events: list[dict[str, Any]] = []
        hit_index: dict[str, list[dict[str, Any]]] = {actor_id: [] for actor_id in actor_ids}
        seen_pairs: set[tuple[str, int, str]] = set()

        for beat_index, beat in enumerate(scene.get("beats", [])):
            actor_id = str(beat.get("actor_id") or "")
            motion = str(beat.get("motion") or "")
            profile = self._combat_profile(motion)
            if actor_id not in actor_ids or profile is None:
                continue
            start_ms = int(beat.get("start_ms", 0) or 0)
            end_ms = int(beat.get("end_ms", start_ms) or start_ms)
            if end_ms <= start_ms:
                continue
            current_ms = start_ms
            while current_ms <= end_ms:
                attacker_pose = self._base_actor_pose(scene, actor_id, current_ms)
                attacker_pose["_beat"] = beat
                attack_box = self._attack_box(attacker_pose, current_ms)
                if attack_box is not None:
                    candidates: list[tuple[float, str, dict[str, Any]]] = []
                    for target_id in actor_ids:
                        if target_id == actor_id:
                            continue
                        hit_key = (actor_id, beat_index, target_id)
                        if hit_key in seen_pairs:
                            continue
                        target_pose = self._base_actor_pose(scene, target_id, current_ms)
                        if any(self._rects_intersect(attack_box, hurt_box) for hurt_box in self._hurt_boxes(target_pose)):
                            candidates.append((abs(float(target_pose["x"]) - float(attacker_pose["x"])), target_id, target_pose))
                    if candidates:
                        _, target_id, target_pose = min(candidates, key=lambda item: item[0])
                        seen_pairs.add((actor_id, beat_index, target_id))
                        mirror = -1.0 if attacker_pose.get("facing") == "left" else 1.0
                        event = {
                            "attacker_id": actor_id,
                            "target_id": target_id,
                            "beat_index": beat_index,
                            "time_ms": current_ms,
                            "motion": motion,
                            "reaction": profile["reaction"],
                            "stun_ms": int(profile["stun_ms"]),
                            "push_x": float(profile["push_x"]) * mirror,
                            "push_z": float(profile["push_z"]),
                            "impact_x": (attack_box[0] + attack_box[1]) / 2.0,
                            "impact_z": (attack_box[2] + attack_box[3]) / 2.0,
                            "target_start_x": float(target_pose["x"]),
                            "target_start_z": float(target_pose["z"]),
                        }
                        events.append(event)
                        hit_index[target_id].append(event)
                        break
                current_ms += step_ms

        data = {"events": events, "by_target": hit_index}
        self._scene_combat_cache[scene_key] = data
        return data

    def _reaction_state(self, scene: dict[str, Any], actor_id: str, time_ms: int) -> Optional[dict[str, Any]]:
        active: Optional[dict[str, Any]] = None
        for event in self._scene_combat_data(scene)["by_target"].get(actor_id, []):
            start_ms = int(event["time_ms"])
            end_ms = start_ms + int(event["stun_ms"])
            if start_ms <= time_ms <= end_ms:
                active = event
        if active is None:
            return None
        duration = max(1, int(active["stun_ms"]))
        progress = max(0.0, min(1.0, (time_ms - int(active["time_ms"])) / duration))
        push_curve = math.sin(progress * math.pi)
        return {
            **active,
            "progress": progress,
            "dx": float(active["push_x"]) * push_curve,
            "dz": float(active["push_z"]) * math.sin(progress * math.pi * 0.8),
            "motion": str(active["reaction"]),
        }

    def _find_actor_pose(self, scene: dict[str, Any], actor_id: str, time_ms: int) -> dict[str, Any]:
        pose = self._base_actor_pose(scene, actor_id, time_ms)
        active_beat = self._active_actor_beat(scene, actor_id, time_ms)
        if active_beat is not None:
            pose["_beat"] = active_beat
        reaction = self._reaction_state(scene, actor_id, time_ms)
        if reaction is not None:
            pose["x"] = float(reaction["target_start_x"]) + float(reaction["dx"])
            pose["z"] = float(reaction["target_start_z"]) + float(reaction["dz"])
            pose["motion"] = reaction["motion"]
            pose["emotion"] = "hurt"
            pose["_reaction"] = reaction
        return pose

    def _active_actor_beat(self, scene: dict[str, Any], actor_id: str, time_ms: int) -> Optional[dict[str, Any]]:
        for beat in scene.get("beats", []):
            if beat.get("actor_id") != actor_id:
                continue
            if int(beat["start_ms"]) <= time_ms <= int(beat["end_ms"]):
                return beat
        return None

    def _beat_progress(self, beat: Optional[dict[str, Any]], time_ms: int) -> float:
        if not beat:
            return 0.0
        start_ms = int(beat["start_ms"])
        end_ms = int(beat["end_ms"])
        duration = max(1, end_ms - start_ms)
        return max(0.0, min(1.0, (time_ms - start_ms) / duration))

    def _actor_meta_in_scene(self, scene: dict[str, Any], actor_id: str) -> dict[str, Any]:
        return next(item for item in scene["actors"] if item["actor_id"] == actor_id)

    def _actor_frontness(self, scene: dict[str, Any], actor_id: str) -> float:
        actor = self._actor_meta_in_scene(scene, actor_id)
        layer = str(actor.get("layer") or "mid")
        layer_frontness = {
            "back": -1.0,
            "mid": 0.0,
            "front": 1.0,
        }.get(layer, 0.0)
        explicit = actor.get("frontness")
        if explicit is not None:
            return float(explicit)
        return layer_frontness + self.actor_front_bias

    def _actor_stage_y(self, frontness: float) -> float:
        return 6.0 - max(-1.0, min(2.0, frontness)) * 0.38

    def _ground_plane_z(self, stage_y: float) -> float:
        return float(self.stage_layout["ground_z"]) + (stage_y - float(self.stage_layout["ground_y"])) * float(self.stage_layout["ground_slope"])

    def _stage_actor_z(self, pose_z: float, stage_y: float, scale: float) -> float:
        foot_anchor = 1.18 * scale
        foot_clearance = 0.14 * scale
        return self._ground_plane_z(stage_y) + foot_anchor + foot_clearance + pose_z + self.actor_ground_offset

    def _stage_actor_visual_z(self, pose_z: float, frontness: float, scale: float) -> float:
        stage_y = self._actor_stage_y(frontness)
        return self._stage_actor_z(pose_z, stage_y, scale)

    def _stage_actor_scale(self, scale: float, frontness: float) -> float:
        return scale * self.actor_scale_base * (1.0 + max(-1.0, min(2.0, frontness)) * 0.14)

    def _motion_offsets(self, motion: str, progress: float, mirror: float) -> dict[str, float]:
        jump_arc = math.sin(progress * math.pi)
        if motion == "somersault":
            return {
                "dx": (progress - 0.5) * 1.4 * mirror,
                "dz": jump_arc * 1.25,
                "roll": -360.0 * progress * mirror,
                "lean": 0.0,
            }
        if motion == "handstand-walk":
            stride = math.sin(progress * math.pi * 4.0)
            return {
                "dx": (progress - 0.5) * 1.0 * mirror,
                "dz": 0.06 * abs(stride),
                "roll": 180.0,
                "lean": stride * 10.0,
            }
        if motion == "big-jump":
            return {
                "dx": 0.0,
                "dz": jump_arc * 1.55,
                "roll": math.sin(progress * math.pi) * 18.0 * mirror,
                "lean": 0.0,
            }
        if motion == "dunk":
            return {
                "dx": progress * 1.1 * mirror,
                "dz": jump_arc * 1.70,
                "roll": -18.0 * mirror + math.sin(progress * math.pi) * -12.0 * mirror,
                "lean": -8.0 * mirror,
            }
        if motion == "stagger":
            recoil = math.sin(progress * math.pi)
            return {
                "dx": -0.12 * recoil * mirror,
                "dz": recoil * 0.12,
                "roll": 10.0 * recoil * mirror,
                "lean": -16.0 * recoil * mirror,
            }
        if motion == "knockback":
            recoil = math.sin(progress * math.pi)
            return {
                "dx": -0.18 * recoil * mirror,
                "dz": recoil * 0.18,
                "roll": 18.0 * recoil * mirror,
                "lean": -24.0 * recoil * mirror,
            }
        if motion == "knockdown":
            recoil = math.sin(progress * math.pi)
            return {
                "dx": -0.22 * recoil * mirror,
                "dz": recoil * 0.22,
                "roll": 86.0 * min(1.0, progress * 1.35) * mirror,
                "lean": -18.0 * recoil * mirror,
            }
        return {"dx": 0.0, "dz": 0.0, "roll": 0.0, "lean": 0.0}

    def _expression_for_actor(self, scene: dict[str, Any], actor_id: str, time_ms: int, pose: dict[str, Any]) -> str:
        explicit = self._active_expression(scene, actor_id, time_ms)
        if explicit:
            return explicit
        beat_expression = pose.get("expression")
        if beat_expression:
            return str(beat_expression)
        active_dialogue = self._active_dialogue(scene, time_ms)
        speaking = bool(active_dialogue and active_dialogue.get("speaker_id") == actor_id)
        motion = pose.get("motion") or "idle"
        emotion = (pose.get("emotion") or "neutral").lower()

        if emotion in {"charged", "angry", "furious"} or motion in {"dragon-palm", "thunder-strike", "sword-arc", "somersault", "dunk"}:
            return "fierce"
        if motion in {"stagger", "knockback", "knockdown"} or emotion in {"hurt", "pain"}:
            return "hurt"
        if motion == "point":
            return "smirk"
        if emotion in {"awkward", "embarrassed", "nervous"}:
            return "awkward"
        # Only open the mouth during an actual active dialogue window.
        # Fight stories may still keep "talk" beats as a neutral stance cue.
        if speaking:
            if emotion in {"calm", "serious", "cold"}:
                return "explain"
            return "talk"
        if motion == "talk":
            if emotion in {"calm", "serious", "cold"}:
                return "explain"
            return "neutral"
        if motion in {"handstand-walk", "big-jump"}:
            return "deadpan"
        if emotion in {"calm", "serious"}:
            return "deadpan"
        return "neutral"

    def _active_expression(self, scene: dict[str, Any], actor_id: str, time_ms: int) -> Optional[str]:
        for item in scene.get("expressions", []):
            if str(item.get("actor_id") or "") != actor_id:
                continue
            if int(item.get("start_ms", -1)) <= time_ms <= int(item.get("end_ms", -1)):
                expression = str(item.get("expression") or "").strip()
                if expression:
                    return expression
        return None

    def _render_panda_body(
        self,
        actor_root,
        actor_id: str,
        character: dict[str, Any],
        palette: dict[str, tuple[float, ...]],
        motion: str,
        mirror: float,
        time_ms: int,
    ) -> tuple[float, float]:
        points = self._body_points(motion, mirror, time_ms)
        show_skin = True
        show_bones = bool(character.get("show_bones", not show_skin))
        bone_width = 0.06

        bone_segments = [
            ("spine-lower", "pelvis", "chest"),
            ("spine-upper", "chest", "neck"),
            ("clavicle", "shoulder_left", "shoulder_right"),
            ("arm-upper-left", "shoulder_left", "elbow_left"),
            ("arm-lower-left", "elbow_left", "hand_left"),
            ("arm-upper-right", "shoulder_right", "elbow_right"),
            ("arm-lower-right", "elbow_right", "hand_right"),
            ("leg-upper-left", "hip_left", "knee_left"),
            ("leg-lower-left", "knee_left", "foot_left"),
            ("leg-upper-right", "hip_right", "knee_right"),
            ("leg-lower-right", "knee_right", "foot_right"),
            ("hip-bar", "hip_left", "hip_right"),
        ]
        if show_bones:
            for name, start_key, end_key in bone_segments:
                self._attach_segment(
                    actor_root,
                    points[start_key],
                    points[end_key],
                    bone_width,
                    f"{name}-{actor_id}",
                    palette["bone"],
                )

            joint_points = [
                "neck",
                "shoulder_left",
                "shoulder_right",
                "elbow_left",
                "elbow_right",
                "hand_left",
                "hand_right",
                "hip_left",
                "hip_right",
                "knee_left",
                "knee_right",
                "foot_left",
                "foot_right",
            ]
            for name in joint_points:
                self._attach_joint(actor_root, points[name], 0.10, f"{name}-{actor_id}", palette["bone"])

        if show_skin:
            skin_segments = [
                ("skin-spine-lower", "pelvis", "chest", 0.20, palette["robe"]),
                ("skin-spine-upper", "chest", "neck", 0.14, palette["robe_inner"]),
                ("skin-clavicle", "shoulder_left", "shoulder_right", 0.10, palette["accent"]),
                ("skin-arm-upper-left", "shoulder_left", "elbow_left", 0.18, palette["robe"]),
                ("skin-arm-lower-left", "elbow_left", "hand_left", 0.16, palette["robe_inner"]),
                ("skin-arm-upper-right", "shoulder_right", "elbow_right", 0.18, palette["robe"]),
                ("skin-arm-lower-right", "elbow_right", "hand_right", 0.16, palette["robe_inner"]),
                ("skin-leg-upper-left", "hip_left", "knee_left", 0.24, palette["robe"]),
                ("skin-leg-lower-left", "knee_left", "foot_left", 0.16, palette["robe_inner"]),
                ("skin-leg-upper-right", "hip_right", "knee_right", 0.24, palette["robe"]),
                ("skin-leg-lower-right", "knee_right", "foot_right", 0.16, palette["robe_inner"]),
            ]
            for name, start_key, end_key, width, color in skin_segments:
                self._attach_segment(actor_root, points[start_key], points[end_key], width, f"{name}-{actor_id}", color)

            self._attach_card(
                actor_root,
                0.42,
                0.16,
                f"waist-wrap-{actor_id}",
                (points["pelvis"][0], -0.02, points["pelvis"][1] + 0.02),
                palette["accent"],
            )
            for hand_key in ("hand_left", "hand_right"):
                self._attach_joint(actor_root, points[hand_key], 0.15, f"{hand_key}-skin-{actor_id}", palette["face"])
            for foot_key in ("foot_left", "foot_right"):
                self._attach_joint(actor_root, points[foot_key], 0.16, f"{foot_key}-skin-{actor_id}", palette["patch"])

            outfit_texture = self._texture_at_time(self._outfit_skin_path(character), time_ms)
            if outfit_texture is not None:
                shoulder_span = abs(points["shoulder_right"][0] - points["shoulder_left"][0])
                hip_span = abs(points["hip_right"][0] - points["hip_left"][0])
                torso_width = max(0.58, max(shoulder_span * 0.82, hip_span * 1.08))
                torso_height = max(0.86, abs(points["neck"][1] - points["pelvis"][1]) * 1.06)
                outfit_width = torso_width
                outfit_height = torso_height
                outfit_z = ((points["chest"][1] + points["pelvis"][1]) * 0.5) + 0.04
                if motion == "somersault":
                    outfit_width = torso_width * 0.92
                    outfit_height = torso_height * 0.82
                    outfit_z = ((points["chest"][1] + points["pelvis"][1]) * 0.5) + 0.08
                elif motion == "handstand-walk":
                    outfit_width = torso_width * 0.92
                    outfit_height = torso_height * 0.88
                    outfit_z = ((points["chest"][1] + points["pelvis"][1]) * 0.5)
                elif motion == "big-jump":
                    outfit_width = torso_width * 0.96
                    outfit_height = torso_height * 0.90
                    outfit_z = ((points["chest"][1] + points["pelvis"][1]) * 0.5) + 0.05
                elif motion == "dunk":
                    outfit_width = torso_width * 0.94
                    outfit_height = torso_height * 0.88
                    outfit_z = ((points["chest"][1] + points["pelvis"][1]) * 0.5) + 0.05
                outfit_card = self._attach_card(
                    actor_root,
                    outfit_width,
                    outfit_height,
                    f"outfit-overlay-{actor_id}",
                    (0.0, -0.10, outfit_z),
                    (1.0, 1.0, 1.0, 1.0),
                )
                self._apply_texture(outfit_card, outfit_texture)

        return points["neck"]

    def _render_panda_head(
        self,
        actor_root,
        actor_id: str,
        character: dict[str, Any],
        palette: dict[str, tuple[float, ...]],
        expression: str,
        motion: str,
        neck_point: tuple[float, float],
        mirror: float,
        time_ms: int,
        speaking: bool,
    ) -> None:
        head_root = actor_root.attachNewNode(f"head-root-{actor_id}")
        head_root.setPos(neck_point[0], -0.16, neck_point[1] + 0.28)
        head_radius_x = 0.58
        head_radius_z = 0.44
        head_angle = self._head_angle(expression, motion, mirror, time_ms)
        head_root.setR(head_angle)
        face_texture = self._texture_at_time(
            self._face_skin_path(character, expression, time_ms, speaking=speaking),
            time_ms,
            variant_key="ellipse-face",
        )

        self._attach_ellipse(head_root, (0.0, 0.0), head_radius_x, head_radius_z, f"head-{actor_id}", palette["face"], depth_y=0.00, slices=16)
        if face_texture is None:
            self._attach_ellipse(
                head_root,
                (0.0, -0.01),
                head_radius_x * 0.82,
                head_radius_z * 0.80,
                f"face-core-{actor_id}",
                (0.99, 0.99, 0.98, 0.38),
                depth_y=-0.01,
                slices=10,
            )
        self._attach_card(
            head_root,
            0.72,
            0.10,
            f"headband-{actor_id}",
            (0.0, -0.05, 0.22),
            (*palette["accent"][:3], 0.92),
        )
        self._attach_card(
            head_root,
            0.08,
            0.18,
            f"headband-tag-{actor_id}",
            (0.31 * mirror, -0.08, 0.12),
            (*palette["accent"][:3], 0.92),
            r=8.0 * mirror,
        )

        # Directed radius: nose direction -> head top. This angle is the head pose source
        # of truth, so future portrait textures can be attached to this same transform.
        self._attach_segment(
            head_root,
            (0.0, -0.06),
            (0.0, head_radius_z),
            0.035,
            f"head-radius-{actor_id}",
            (*palette["patch"][:3], 0.72),
        )

        if face_texture is not None:
            anchor = character.get("head_anchor") or {}
            offset = anchor.get("offset") or [0, 0]
            scale = float(anchor.get("scale", 1.0) or 1.0)
            face_card = self._attach_card(
                head_root,
                head_radius_x * 1.86 * scale,
                head_radius_z * 1.86 * scale,
                f"face-skin-{actor_id}",
                (
                    float(offset[0]) / 256.0 * head_radius_x * 2.0,
                    -0.19,
                    float(offset[1]) / -256.0 * head_radius_z * 2.0,
                ),
                (1.0, 1.0, 1.0, 1.0),
            )
            self._apply_texture(face_card, face_texture)

        for ear_x in (-0.40, 0.40):
            self._attach_circle(
                head_root,
                (ear_x, 0.32),
                0.18,
                f"ear-{actor_id}-{ear_x:+.0f}",
                palette["patch"],
                depth_y=0.08,
                slices=8,
            )
        if face_texture is not None:
            return
        for patch_x, rotation in ((-0.23, 24.0), (0.23, -24.0)):
            self._attach_card(
                head_root,
                0.28,
                0.24,
                f"patch-{actor_id}-{patch_x:+.0f}",
                (patch_x, -0.06, 0.08),
                palette["patch"],
                r=rotation,
            )

        blink_factor = 0.18 if time_ms % 2100 > 1930 else 1.0
        eye_width = 0.12
        eye_height = 0.05 * blink_factor
        pupil_height = 0.05 * blink_factor
        mouth_width = 0.18
        mouth_height = 0.03
        mouth_y = -0.23
        mouth_x = 0.0
        mouth_r = 0.0
        left_brow_r = 8.0
        right_brow_r = -8.0
        left_brow_z = 0.28
        right_brow_z = 0.28
        blush_alpha = palette["blush"][3]
        sweat_drop = False

        if expression == "deadpan":
            eye_height = 0.026 * blink_factor
            pupil_height = 0.020 * blink_factor
            mouth_height = 0.02
        elif expression == "talk":
            eye_height = 0.052 * blink_factor
            pupil_height = 0.048 * blink_factor
            mouth_width = 0.20
            mouth_height = 0.08 + (abs(math.sin(time_ms / 95)) * 0.04 if speaking else 0.0)
            left_brow_r = 4.0
            right_brow_r = -4.0
        elif expression == "explain":
            eye_height = 0.044 * blink_factor
            pupil_height = 0.036 * blink_factor
            mouth_width = 0.18
            mouth_height = 0.06 + (abs(math.sin(time_ms / 130)) * 0.02 if speaking else 0.0)
            left_brow_r = 5.0
            right_brow_r = -5.0
        elif expression == "smirk":
            eye_height = 0.030 * blink_factor
            pupil_height = 0.022 * blink_factor
            mouth_width = 0.18
            mouth_height = 0.022
            mouth_x = 0.08 * mirror
            mouth_r = -12.0 * mirror
            left_brow_r = 18.0 * mirror
            right_brow_r = -6.0 * mirror
        elif expression == "fierce":
            eye_height = 0.020 * blink_factor
            pupil_height = 0.016 * blink_factor
            mouth_width = 0.24
            mouth_height = 0.11 + (abs(math.sin(time_ms / 70)) * 0.04 if speaking else 0.0)
            mouth_y = -0.20
            left_brow_r = -26.0
            right_brow_r = 26.0
            left_brow_z = 0.25
            right_brow_z = 0.25
            blush_alpha = max(blush_alpha, 0.48)
        elif expression == "awkward":
            eye_height = 0.030 * blink_factor
            pupil_height = 0.026 * blink_factor
            mouth_width = 0.16
            mouth_height = 0.026
            mouth_y = -0.25
            blush_alpha = max(blush_alpha, 0.42)
            sweat_drop = True
        elif expression == "hurt":
            eye_height = 0.018 * blink_factor
            pupil_height = 0.014 * blink_factor
            mouth_width = 0.22
            mouth_height = 0.05
            mouth_y = -0.22
            mouth_r = -10.0 * mirror
            left_brow_r = -18.0
            right_brow_r = 18.0
            blush_alpha = max(blush_alpha, 0.52)
            sweat_drop = True

        for eye_x in (-0.21, 0.21):
            self._attach_card(
                head_root,
                eye_width,
                eye_height,
                f"eye-white-{actor_id}-{eye_x:+.0f}",
                (eye_x, -0.10, 0.08),
                (0.97, 0.97, 0.98, 1.0),
            )
            self._attach_card(
                head_root,
                0.04,
                max(0.012, pupil_height),
                f"pupil-{actor_id}-{eye_x:+.0f}",
                (eye_x + 0.01 * mirror, -0.12, 0.08),
                palette["patch"],
            )

        for brow_x, brow_r, brow_z in ((-0.20, left_brow_r, left_brow_z), (0.20, right_brow_r, right_brow_z)):
            self._attach_card(
                head_root,
                0.18,
                0.03,
                f"brow-{actor_id}-{brow_x:+.0f}",
                (brow_x, -0.14, brow_z),
                palette["patch"],
                r=brow_r,
            )

        self._attach_card(
            head_root,
            0.08,
            0.06,
            f"nose-{actor_id}",
            (0.0, -0.13, -0.05),
            palette["patch"],
        )
        self._attach_card(
            head_root,
            0.24,
            0.10,
            f"muzzle-{actor_id}",
            (0.0, -0.15, -0.15),
            (0.96, 0.95, 0.93, 0.62),
        )

        for cheek_x in (-0.28, 0.28):
            self._attach_card(
                head_root,
                0.15,
                0.08,
                f"cheek-{actor_id}-{cheek_x:+.0f}",
                (cheek_x, -0.16, -0.10),
                (*palette["blush"][:3], blush_alpha),
            )

        self._attach_card(
            head_root,
            mouth_width,
            mouth_height,
            f"mouth-{actor_id}",
            (mouth_x, -0.18, mouth_y),
            palette["mouth"] if mouth_height > 0.03 else palette["patch"],
            r=mouth_r,
        )
        if mouth_height > 0.05:
            self._attach_card(
                head_root,
                mouth_width * 0.56,
                mouth_height * 0.45,
                f"tongue-{actor_id}",
                (mouth_x, -0.19, mouth_y - 0.02),
                (0.96, 0.58, 0.64, 0.84),
                r=mouth_r,
            )

        if sweat_drop:
            self._attach_card(
                head_root,
                0.07,
                0.14,
                f"sweat-{actor_id}",
                (0.36 * mirror, -0.16, 0.02),
                (0.72, 0.89, 1.0, 0.95),
                r=10.0 * mirror,
            )

    def _render_actor(self, scene: dict[str, Any], actor_id: str, time_ms: int) -> None:
        actor_meta = self.cast[actor_id]
        character = self.characters[actor_meta["asset_id"]]
        pose = self._find_actor_pose(scene, actor_id, time_ms)
        active_beat = self._active_actor_beat(scene, actor_id, time_ms)
        motion_progress = self._beat_progress(active_beat, time_ms)
        expression = self._expression_for_actor(scene, actor_id, time_ms, pose)
        palette = self._character_palette(character)
        x = pose["x"]
        frontness = self._actor_frontness(scene, actor_id)
        scale = self._stage_actor_scale(pose["scale"], frontness)
        z = self._stage_actor_visual_z(pose["z"], frontness, scale)
        motion = pose["motion"]
        bob = 0.0
        lean = 0.0
        aura_alpha = 0.0
        active_dialogue = self._active_dialogue(scene, time_ms)
        speaking = bool(active_dialogue and active_dialogue.get("speaker_id") == actor_id)

        if motion == "talk" and speaking:
            bob = math.sin(time_ms / 120) * 0.08
            aura_alpha = 0.08
        elif motion == "point":
            lean = -10 if pose["facing"] == "right" else 10
        elif motion in {"dragon-palm", "thunder-strike", "sword-arc"}:
            bob = math.sin(time_ms / 80) * 0.12
            aura_alpha = 0.22
            lean = -15 if pose["facing"] == "right" else 15

        if motion == "enter":
            x -= 1.8 if pose["facing"] == "right" else -1.8
        if motion == "exit":
            x += 1.8 if pose["facing"] == "right" else -1.8

        mirror = -1.0 if pose["facing"] == "left" else 1.0
        motion_offsets = self._motion_offsets(motion, motion_progress, mirror)
        x += motion_offsets["dx"]
        z += motion_offsets["dz"]
        lean += motion_offsets["lean"]

        instance = self._actor_instances[actor_id]
        actor_root = instance["root"]
        actor_root.setPos(x, self._actor_stage_y(frontness), z + bob)
        actor_root.setScale(scale, 1.0, scale)
        actor_root.setR(lean + motion_offsets["roll"])

        runtime = instance["runtime"]
        self._update_card_node(
            runtime["aura"],
            1.7,
            2.5,
            (0.0, 0.2, 0.05),
            (*palette["accent"][:3], aura_alpha),
            visible=aura_alpha > 0.0,
        )
        self._update_actor_visuals(
            runtime,
            actor_id,
            character,
            palette,
            motion,
            expression,
            mirror,
            time_ms,
            speaking,
        )
        label_runtime = runtime.get("label")
        if label_runtime is not None:
            label_runtime["node"].setPos(x / 6.0, 0, -0.02 - z / 8.0)

    def _render_npc(self, scene: dict[str, Any], npc: dict[str, Any], time_ms: int) -> None:
        character = self.characters.get(str(npc.get("asset_id") or ""), {})
        if not character:
            return
        ai_pos = npc["ai_node"].getPos()
        x = float(ai_pos.x)
        frontness = self._npc_layer_bias(str(npc["group"].get("layer") or "mid")) + float(ai_pos.z)
        frontness = max(-1.0, min(2.0, frontness))
        scale = self._stage_actor_scale(float(npc.get("scale", 0.8) or 0.8), frontness)
        z = self._stage_actor_visual_z(0.0, frontness, scale)
        dx = x - float(npc.get("last_x", x))
        dfront = float(ai_pos.z) - float(npc.get("last_frontness", ai_pos.z))
        npc["last_x"] = x
        npc["last_frontness"] = float(ai_pos.z)
        if abs(dx) > 0.008:
            npc["facing"] = "right" if dx >= 0.0 else "left"
        elif npc.get("watch"):
            target_x, _ = self._npc_target_point(scene, npc["group"], npc, time_ms)
            if abs(target_x - x) > 0.12:
                npc["facing"] = "right" if target_x >= x else "left"
        speed = math.hypot(dx, dfront)
        motion = "talk" if speed > 0.008 else "idle"
        expression = "awkward" if str(npc["group"].get("behavior") or "") == "evade" else "deadpan"
        if motion == "idle" and npc.get("watch"):
            expression = "deadpan"
        elif motion == "talk":
            expression = "neutral"
        mirror = -1.0 if npc.get("facing") == "left" else 1.0
        actor_root = npc["root"]
        actor_root.setPos(x, self._actor_stage_y(frontness), z)
        actor_root.setScale(scale, 1.0, scale)
        self._update_card_node(
            npc["runtime"]["aura"],
            1.7,
            2.5,
            (0.0, 0.2, 0.05),
            (*self._character_palette(character)["accent"][:3], 0.0),
            visible=False,
        )
        self._update_actor_visuals(
            npc["runtime"],
            str(npc["id"]),
            character,
            self._character_palette(character),
            motion,
            expression,
            mirror,
            time_ms,
            False,
        )

    def _overlay_text(
        self,
        text: str,
        pos: tuple[float, float],
        scale: float,
        fg: tuple[float, float, float, float],
        parent=None,
    ):
        TextNode = self._core["TextNode"]
        text_node = TextNode("text")
        text_node.setText(text)
        text_node.setAlign(TextNode.ACenter)
        text_node.setTextColor(*fg)
        if self.text_font is not None:
            text_node.setFont(self.text_font)
        text_parent = parent if parent is not None else self.overlay_root
        node = text_parent.attachNewNode(text_node)
        node.setScale(scale)
        node.setPos(pos[0], 0, pos[1])
        return node

    def _active_dialogue(self, scene: dict[str, Any], time_ms: int) -> Optional[dict[str, Any]]:
        for dialogue in scene.get("dialogues", []):
            if int(dialogue["start_ms"]) <= time_ms <= int(dialogue["end_ms"]):
                return dialogue
        return None

    def _render_subtitle(self, scene: dict[str, Any], time_ms: int) -> None:
        subtitle_mode = self.story["video"].get("subtitle_mode")
        if subtitle_mode == "none":
            return
        active = self._active_dialogue(scene, time_ms)
        if not active:
            return
        text = active.get("subtitle") or active.get("text") or ""
        plate = self.subtitle_root.attachNewNode(self._make_card(1.72, 0.22, "subtitle-bg"))
        plate.setPos(0, 0, -0.82)
        plate.setColor(0.02, 0.02, 0.02, 0.68)
        plate.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        self._overlay_text(text, pos=(0.0, -0.86), scale=0.065, fg=(1.0, 0.98, 0.95, 1.0), parent=self.subtitle_root)

    def _effect_target_pose(self, scene: dict[str, Any], actor_id: str, time_ms: int):
        candidates = [item["actor_id"] for item in scene.get("actors", []) if item.get("actor_id") != actor_id]
        if not candidates:
            return None
        source_pose = self._find_actor_pose(scene, actor_id, time_ms)
        ranked = []
        for target_id in candidates:
            target_pose = self._find_actor_pose(scene, target_id, time_ms)
            ranked.append((abs(float(target_pose["x"]) - float(source_pose["x"])), {"actor_id": target_id, "pose": target_pose}))
        ranked.sort(key=lambda item: item[0])
        return ranked[0][1] if ranked else None

    def _glow_color(self, color: tuple[float, ...], lift: float = 0.55, alpha: float = 1.0) -> tuple[float, float, float, float]:
        r = min(1.0, color[0] * (1.0 - lift) + lift)
        g = min(1.0, color[1] * (1.0 - lift) + lift)
        b = min(1.0, color[2] * (1.0 - lift) + lift)
        return (r, g, b, alpha)

    def _attach_glow_outline(self, parent, width: float, height: float, name: str, pos: tuple[float, float, float], r: float, alpha: float) -> None:
        outline = parent.attachNewNode(self._make_card(width, height, f"{name}-outline"))
        outline.setPos(*pos)
        outline.setColor(1.0, 1.0, 1.0, alpha)
        outline.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        outline.setR(r)

    def _attach_fx_blob(
        self,
        parent,
        x: float,
        y: float,
        z: float,
        radius_x: float,
        radius_z: float,
        name: str,
        color: tuple[float, ...],
        r: float = 0.0,
        glow: float = 0.0,
        glow_alpha: float = 0.0,
    ):
        blob_root = parent.attachNewNode(name)
        blob_root.setPos(x, y, z)
        blob_root.setR(r)
        if glow > 0.0 and glow_alpha > 0.0:
            self._attach_ellipse(
                blob_root,
                (0.0, 0.0),
                radius_x * glow,
                radius_z * glow,
                f"{name}-glow",
                self._glow_color(color, lift=0.98, alpha=glow_alpha),
                depth_y=-0.02,
            )
        self._attach_ellipse(blob_root, (0.0, 0.0), radius_x, radius_z, f"{name}-core", color, depth_y=0.0)
        return blob_root

    def _attach_fx_circle(
        self,
        parent,
        x: float,
        y: float,
        z: float,
        radius: float,
        name: str,
        color: tuple[float, ...],
        glow: float = 0.0,
        glow_alpha: float = 0.0,
    ):
        node = parent.attachNewNode(name)
        node.setPos(x, y, z)
        if glow > 0.0 and glow_alpha > 0.0:
            self._attach_circle(node, (0.0, 0.0), radius * glow, f"{name}-glow", self._glow_color(color, lift=0.98, alpha=glow_alpha), depth_y=-0.02)
        self._attach_circle(node, (0.0, 0.0), radius, f"{name}-core", color, depth_y=0.0)
        return node

    def _attach_fx_shard(
        self,
        parent,
        x: float,
        y: float,
        z: float,
        width: float,
        height: float,
        name: str,
        color: tuple[float, ...],
        r: float,
        alpha: float,
    ):
        shard = parent.attachNewNode(self._make_card(width, height, name))
        shard.setPos(x, y, z)
        shard.setColor(*self._glow_color(color, lift=0.90, alpha=alpha))
        shard.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        shard.setR(r)
        self._attach_fx_circle(parent, x, y - 0.01, z, max(width, height) * 0.28, f"{name}-joint", color, glow=2.1, glow_alpha=max(0.0, alpha * 0.32))
        return shard

    def _attach_fx_beam(
        self,
        parent,
        start: tuple[float, float],
        end: tuple[float, float],
        y: float,
        width: float,
        name: str,
        color: tuple[float, ...],
        alpha: float,
        glow: float = 2.2,
    ):
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length = max(0.04, math.hypot(dx, dz))
        angle = math.degrees(math.atan2(dx, dz if abs(dz) > 0.001 else 0.001))
        cx = (start[0] + end[0]) / 2.0
        cz = (start[1] + end[1]) / 2.0
        if glow > 0.0:
            glow_beam = parent.attachNewNode(self._make_card(width * glow, length, f"{name}-glow"))
            glow_beam.setPos(cx, y - 0.02, cz)
            glow_beam.setColor(*self._glow_color(color, lift=0.98, alpha=max(0.0, alpha * 0.28)))
            glow_beam.setTransparency(self._core["TransparencyAttrib"].MAlpha)
            glow_beam.setR(angle)
        beam = parent.attachNewNode(self._make_card(width, length, name))
        beam.setPos(cx, y, cz)
        beam.setColor(*self._glow_color(color, lift=0.92, alpha=alpha))
        beam.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        beam.setR(angle)
        return beam

    def _render_beam_path(
        self,
        points: list[tuple[float, float]],
        y: float,
        width: float,
        color: tuple[float, ...],
        name: str,
        alpha_start: float = 0.96,
        alpha_decay: float = 0.08,
    ) -> None:
        for idx, (start, end) in enumerate(zip(points, points[1:])):
            alpha = max(0.18, alpha_start - idx * alpha_decay)
            self._attach_fx_beam(
                self.effects_root,
                start,
                end,
                y - idx * 0.01,
                max(0.02, width - idx * 0.004),
                f"{name}-beam-{idx}",
                color,
                alpha,
            )
            self._attach_fx_circle(
                self.effects_root,
                end[0],
                y - idx * 0.01,
                end[1],
                max(0.018, width * 0.55),
                f"{name}-node-{idx}",
                self._glow_color(color, lift=0.98, alpha=alpha),
                glow=2.2,
                glow_alpha=alpha * 0.22,
            )

    def _render_dragon_head(self, x: float, y: float, z: float, angle: float, color: tuple[float, ...], name: str, scale: float = 1.0) -> None:
        head_root = self.effects_root.attachNewNode(name)
        head_root.setPos(x, y, z)
        head_root.setR(angle)
        self._attach_fx_beam(head_root, (-0.08 * scale, -0.02 * scale), (0.18 * scale, 0.02 * scale), 0.0, 0.06 * scale, f"{name}-snout", color, 0.96, glow=2.8)
        self._attach_fx_beam(head_root, (-0.10 * scale, 0.06 * scale), (0.12 * scale, 0.10 * scale), -0.01, 0.05 * scale, f"{name}-crest", color, 0.82, glow=2.4)
        self._attach_fx_beam(head_root, (0.04 * scale, -0.01 * scale), (0.20 * scale, -0.10 * scale), 0.01, 0.04 * scale, f"{name}-jaw", color, 0.72, glow=2.0)
        self._attach_fx_circle(head_root, 0.06 * scale, -0.01, 0.04 * scale, 0.018 * scale, f"{name}-eye", (1.0, 0.98, 0.92, 0.98), glow=2.6, glow_alpha=0.26)
        for horn_dir in (-1.0, 1.0):
            self._attach_fx_shard(
                head_root,
                -0.08 * scale,
                -0.01,
                0.15 * scale * horn_dir,
                0.05 * scale,
                0.18 * scale,
                f"{name}-horn-{int(horn_dir)}",
                color,
                42.0 * horn_dir,
                0.58,
            )

    def _render_lightning_path(self, points: list[tuple[float, float]], y: float, color: tuple[float, ...], name: str) -> None:
        for idx, point in enumerate(points):
            alpha = max(0.26, 0.94 - idx * 0.10)
            self._attach_fx_circle(
                self.effects_root,
                point[0],
                y - 0.50 - idx * 0.01,
                point[1],
                max(0.04, 0.10 - idx * 0.006),
                f"{name}-node-{idx}",
                self._glow_color(color, lift=0.98, alpha=alpha),
                glow=2.4,
                glow_alpha=alpha * 0.30,
            )
            if idx == 0:
                continue
            prev = points[idx - 1]
            dx = point[0] - prev[0]
            dz = point[1] - prev[1]
            length = max(0.06, math.hypot(dx, dz))
            angle = math.degrees(math.atan2(dx, dz if abs(dz) > 0.001 else 0.001))
            self._attach_fx_shard(
                self.effects_root,
                (point[0] + prev[0]) / 2.0,
                y - 0.49 - idx * 0.01,
                (point[1] + prev[1]) / 2.0,
                0.055,
                length,
                f"{name}-seg-{idx}",
                color,
                angle,
                alpha * 0.92,
            )

    def _render_dragon_charge(self, x: float, y: float, z: float, ratio: float, mirror: float, color: tuple[float, ...]) -> None:
        orbit = ratio * math.tau * 1.35
        for ring in range(3):
            points: list[tuple[float, float]] = []
            for idx in range(8):
                phase = orbit + ring * 0.78 + idx * 0.42
                px = x + math.cos(phase) * (0.34 + ring * 0.12) * mirror
                pz = z + 0.76 + math.sin(phase) * (0.22 + ring * 0.08) + idx * 0.04
                points.append((px, pz))
            self._render_beam_path(points, y - 0.48 - ring * 0.02, 0.07 + ring * 0.01, color, f"dragon-charge-{ring}", alpha_start=0.86 - ring * 0.08, alpha_decay=0.06)
        head_x = x + math.cos(orbit + 0.84) * 0.54 * mirror
        head_z = z + 1.00 + math.sin(orbit + 0.84) * 0.22
        self._render_dragon_head(head_x, y - 0.44, head_z, 18.0 * mirror + math.degrees(orbit), color, "dragon-charge-head", scale=0.82)

    def _render_dragon_flight(self, x0: float, y0: float, z0: float, x1: float, z1: float, ratio: float, color: tuple[float, ...]) -> None:
        travel_x = x0 + (x1 - x0) * ratio
        travel_z = z0 + (z1 - z0) * ratio + math.sin(ratio * math.pi) * 0.42
        tail_dx = x1 - x0
        tail_dz = z1 - z0
        angle = math.degrees(math.atan2(tail_dx, tail_dz if abs(tail_dz) > 0.001 else 0.001))
        points: list[tuple[float, float]] = []
        for idx in range(10):
            trail_ratio = max(0.0, ratio - idx * 0.065)
            trail_x = x0 + (x1 - x0) * trail_ratio
            trail_z = z0 + (z1 - z0) * trail_ratio + math.sin(trail_ratio * math.pi) * 0.38 + math.sin((trail_ratio + idx * 0.12) * math.pi * 4.0) * 0.07
            points.append((trail_x, trail_z))
            if idx < 8:
                fin_start = (trail_x - 0.03, trail_z + 0.03)
                fin_end = (trail_x - 0.16, trail_z + 0.12)
                self._attach_fx_beam(self.effects_root, fin_start, fin_end, y0 - 0.53 - idx * 0.012, 0.028, f"dragon-fin-{idx}", color, max(0.0, 0.46 - idx * 0.04), glow=1.8)
        self._render_beam_path(points, y0 - 0.50, 0.08, color, "dragon-flight", alpha_start=0.98, alpha_decay=0.07)
        self._render_dragon_head(travel_x, y0 - 0.44, travel_z, angle, color, "dragon-flight-head", scale=0.88)

    def _render_thunder_charge(self, x: float, y: float, z: float, ratio: float, color: tuple[float, ...]) -> None:
        pulse = 0.40 + math.sin(ratio * math.pi) * 0.18
        for idx in range(10):
            angle = ratio * 180.0 + idx * 36.0
            rad = math.radians(angle)
            start = (x + math.cos(rad) * (0.10 + pulse * 0.10), z + 0.86 + math.sin(rad) * (0.10 + pulse * 0.10))
            end = (x + math.cos(rad) * (0.48 + pulse * 0.24), z + 0.86 + math.sin(rad) * (0.48 + pulse * 0.24))
            self._attach_fx_beam(self.effects_root, start, end, y - 0.58 - idx * 0.002, 0.03, f"thunder-ray-{idx}", color, 0.34, glow=2.6)
        for idx in range(3):
            start_x = x + (-0.34 + idx * 0.34)
            start_z = z + 1.68 + idx * 0.04
            points = [(start_x, start_z)]
            current_x = start_x
            current_z = start_z
            for step in range(4):
                current_x += math.sin((ratio + idx * 0.09 + step * 0.17) * math.pi * 5.0) * 0.24
                current_z -= 0.30 + step * 0.06
                points.append((current_x, current_z))
            self._render_lightning_path(points, y, color, f"thunder-charge-{idx}")

    def _render_sword_charge(self, x: float, y: float, z: float, ratio: float, mirror: float, color: tuple[float, ...]) -> None:
        sweep = -46.0 * mirror + ratio * 92.0 * mirror
        for idx in range(5):
            arc_ratio = idx / 4.0
            start = (x + 0.12 * mirror, z + 0.56)
            end = (x + (0.34 + arc_ratio * 0.54) * mirror, z + 0.66 + math.sin(arc_ratio * math.pi) * 0.52 + idx * 0.05)
            self._attach_fx_beam(self.effects_root, start, end, y - 0.56 - idx * 0.015, 0.04 + idx * 0.004, f"sword-charge-{idx}", color, max(0.0, 0.82 - idx * 0.10), glow=2.4)

    def _render_charge_screen_dim(self, ratio: float, color: tuple[float, ...]) -> None:
        return

    def _render_avatar_backplate(self, x: float, y: float, z: float, ratio: float, color: tuple[float, ...], mirror: float, shape: str) -> None:
        scale = 1.0 + ratio * 0.9
        if shape == "dragon":
            for idx in range(3):
                plate = self.effects_root.attachNewNode(self._make_card(1.8 * scale - idx * 0.18, 2.3 * scale - idx * 0.22, f"avatar-dragon-{idx}"))
                plate.setPos(x - 0.22 * mirror + idx * 0.08 * mirror, y - 0.78 - idx * 0.02, z + 0.88 + idx * 0.10)
                plate.setColor(*self._glow_color(color, lift=0.78, alpha=max(0.0, 0.48 - idx * 0.06)))
                plate.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                plate.setR(-18.0 * mirror + idx * 10.0 * mirror)
            head = self.effects_root.attachNewNode(self._make_card(0.64 * scale, 0.46 * scale, "avatar-dragon-head"))
            head.setPos(x - 0.62 * mirror, y - 0.74, z + 1.08)
            head.setColor(*self._glow_color(color, lift=0.92, alpha=0.64))
            head.setTransparency(self._core["TransparencyAttrib"].MAlpha)
            head.setR(-22.0 * mirror)
            for horn_dir in (-1.0, 1.0):
                horn = self.effects_root.attachNewNode(self._make_card(0.12 * scale, 0.42 * scale, f"avatar-dragon-horn-{horn_dir:+.0f}"))
                horn.setPos(x - 0.84 * mirror + horn_dir * 0.10 * mirror, y - 0.76, z + 1.36)
                horn.setColor(*self._glow_color(color, lift=0.94, alpha=0.58))
                horn.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                horn.setR((-32.0 + horn_dir * 20.0) * mirror)
            jaw = self.effects_root.attachNewNode(self._make_card(0.48 * scale, 0.16 * scale, "avatar-dragon-jaw"))
            jaw.setPos(x - 0.82 * mirror, y - 0.70, z + 0.86)
            jaw.setColor(*self._glow_color(color, lift=0.88, alpha=0.48))
            jaw.setTransparency(self._core["TransparencyAttrib"].MAlpha)
            jaw.setR(-8.0 * mirror)
        elif shape == "thunder":
            for idx in range(2):
                plate = self.effects_root.attachNewNode(self._make_card(1.2 * scale, 2.7 * scale - idx * 0.24, f"avatar-thunder-{idx}"))
                plate.setPos(x, y - 0.82 - idx * 0.02, z + 1.02 + idx * 0.08)
                plate.setColor(*self._glow_color(color, lift=0.82, alpha=0.44 - idx * 0.06))
                plate.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                plate.setR(idx * 12.0)
            crown = self.effects_root.attachNewNode(self._make_card(1.42 * scale, 0.22 * scale, "avatar-thunder-crown"))
            crown.setPos(x, y - 0.76, z + 1.72)
            crown.setColor(*self._glow_color(color, lift=0.94, alpha=0.52))
            crown.setTransparency(self._core["TransparencyAttrib"].MAlpha)
            for idx, offset in enumerate((-0.48, 0.0, 0.48)):
                spike = self.effects_root.attachNewNode(self._make_card(0.12 * scale, 0.74 * scale, f"avatar-thunder-spike-{idx}"))
                spike.setPos(x + offset, y - 0.78, z + 1.94 + idx * 0.04)
                spike.setColor(*self._glow_color(color, lift=0.98, alpha=0.50 - idx * 0.06))
                spike.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                spike.setR((-18.0 + idx * 18.0))
        else:
            for idx in range(3):
                plate = self.effects_root.attachNewNode(self._make_card(1.7 * scale - idx * 0.16, 1.9 * scale - idx * 0.18, f"avatar-sword-{idx}"))
                plate.setPos(x + 0.06 * mirror, y - 0.76 - idx * 0.02, z + 0.80 + idx * 0.08)
                plate.setColor(*self._glow_color(color, lift=0.84, alpha=0.44 - idx * 0.06))
                plate.setTransparency(self._core["TransparencyAttrib"].MAlpha)
                plate.setR(24.0 * mirror - idx * 14.0 * mirror)
            blade = self.effects_root.attachNewNode(self._make_card(0.22 * scale, 2.20 * scale, "avatar-sword-blade"))
            blade.setPos(x + 0.80 * mirror, y - 0.78, z + 1.16)
            blade.setColor(*self._glow_color(color, lift=0.96, alpha=0.68))
            blade.setTransparency(self._core["TransparencyAttrib"].MAlpha)
            blade.setR(24.0 * mirror)
            guard = self.effects_root.attachNewNode(self._make_card(0.74 * scale, 0.18 * scale, "avatar-sword-guard"))
            guard.setPos(x + 0.52 * mirror, y - 0.76, z + 0.42)
            guard.setColor(*self._glow_color(color, lift=0.90, alpha=0.50))
            guard.setTransparency(self._core["TransparencyAttrib"].MAlpha)
            guard.setR(14.0 * mirror)

    def _render_impact_smoke(self, x: float, y: float, z: float, ratio: float) -> None:
        for idx in range(6):
            spread = -0.42 + idx * 0.17
            rise = 0.22 + idx * 0.05 + ratio * 0.14
            self._attach_fx_blob(
                self.effects_root,
                x + spread,
                y - 0.58 - idx * 0.01,
                z + rise,
                0.16 + ratio * 0.22 + idx * 0.02,
                0.10 + ratio * 0.08,
                f"impact-smoke-{idx}",
                (0.96, 0.97, 1.0, max(0.0, 0.62 - ratio * 0.34 - idx * 0.05)),
                r=-18.0 + idx * 10.0,
                glow=1.4,
                glow_alpha=0.08,
            )

    def _render_thunder_bolt(self, x: float, y: float, z: float, ratio: float, color: tuple[float, ...]) -> None:
        start_z = z + 2.10 - ratio * 0.10
        points = [(x, start_z)]
        current_x = x
        current_z = start_z
        for idx in range(6):
            current_x += math.sin((ratio + idx * 0.18) * math.pi * 7.0) * (0.12 if idx < 4 else 0.08)
            current_z -= 0.34 + (0.02 if idx % 2 else 0.0)
            points.append((current_x, current_z))
        self._render_lightning_path(points, y - 0.02, color, "thunder-bolt")

    def _render_sword_slash(self, x: float, y: float, z: float, ratio: float, mirror: float, color: tuple[float, ...]) -> None:
        for idx in range(7):
            arc_ratio = idx / 6.0
            start = (x + (0.20 + arc_ratio * 0.30) * mirror, z + 0.34 + arc_ratio * 0.12)
            end = (x + (0.54 + arc_ratio * 1.12) * mirror, z + 0.44 + math.sin(arc_ratio * math.pi) * (0.64 + ratio * 0.32))
            self._attach_fx_beam(self.effects_root, start, end, y - 0.58 - idx * 0.01, 0.038 + idx * 0.003, f"sword-slash-{idx}", color, max(0.0, 0.92 - idx * 0.08), glow=2.6)

    def _render_impact_burst(self, x: float, y: float, z: float, ratio: float, color: tuple[float, ...], name: str) -> None:
        radius = 0.10 + ratio * 0.18
        for idx in range(12):
            angle = idx * 30.0 + ratio * 24.0
            rad = math.radians(angle)
            start = (x + math.cos(rad) * radius * 0.18, z + 0.86 + math.sin(rad) * radius * 0.18)
            end = (x + math.cos(rad) * (radius + 0.18 + ratio * 0.20), z + 0.86 + math.sin(rad) * (radius + 0.18 + ratio * 0.20))
            self._attach_fx_beam(self.effects_root, start, end, y - 0.56, 0.026, f"{name}-ray-{idx}", color, max(0.0, 0.76 - ratio * 0.70), glow=2.8)
        for idx in range(2):
            self._attach_fx_circle(
                self.effects_root,
                x,
                y - 0.55 - idx * 0.01,
                z + 0.86,
                radius + idx * 0.08,
                f"{name}-ring-{idx}",
                self._glow_color(color, lift=0.98, alpha=max(0.0, 0.42 - idx * 0.10 - ratio * 0.34)),
                glow=1.4,
                glow_alpha=max(0.0, 0.14 - idx * 0.03),
            )

    def _render_impact_gif(self, x: float, y: float, z: float, impact_age: int, ratio: float, name: str) -> bool:
        image_path = self._impact_effect_path()
        sequence = self._load_texture_sequence(image_path, variant_key="impact-fx")
        if not sequence:
            return False
        texture = self._texture_at_time(image_path, impact_age, variant_key="impact-fx")
        if texture is None:
            return False
        frame_height = max(1, int(sequence.get("height", 1) or 1))
        frame_width = max(1, int(sequence.get("size", 1) or 1))
        target_height = 1.55 + max(0.0, 1.0 - ratio) * 0.36
        target_width = target_height * (frame_width / frame_height)
        node = self.effects_root.attachNewNode(self._make_card(target_width, target_height, f"{name}-gif"))
        node.setPos(x, y - 0.36, z + 0.90)
        node.setTransparency(self._core["TransparencyAttrib"].MAlpha)
        node.setColor(1.0, 0.94, 0.94, max(0.0, 1.0 - ratio * 0.08))
        self._apply_texture(node, texture)
        return True

    def _render_impact_flash(self, ratio: float) -> None:
        return

    def _render_impact_blackout(self, ratio: float) -> None:
        return

    def _impact_state(self, scene: dict[str, Any], time_ms: int) -> Optional[dict[str, Any]]:
        active: Optional[dict[str, Any]] = None
        for event in self._scene_combat_data(scene)["events"]:
            age = time_ms - int(event["time_ms"])
            if 0 <= age <= 420:
                active = {"event": event, "age": age}
        return active

    def _effective_time_ms(self, scene: dict[str, Any], time_ms: int) -> int:
        impact = self._impact_state(scene, time_ms)
        if impact is None:
            return time_ms
        age = int(impact["age"])
        if age <= 90:
            return int(impact["event"]["time_ms"])
        return time_ms - min(70, max(0, 140 - age))

    def _camera_ratio(self, scene: dict[str, Any], time_ms: int) -> float:
        duration = max(1, int(scene.get("duration_ms", 1) or 1))
        return max(0.0, min(1.0, float(time_ms) / duration))

    def _ease_ratio(self, ratio: float, ease: str) -> float:
        mode = str(ease or "linear").lower()
        if mode in {"inout", "ease-in-out", "smooth"}:
            return 0.5 - 0.5 * math.cos(math.pi * ratio)
        if mode in {"in", "ease-in"}:
            return ratio * ratio
        if mode in {"out", "ease-out"}:
            return 1.0 - (1.0 - ratio) * (1.0 - ratio)
        return ratio

    def _camera_state(self, scene: dict[str, Any], time_ms: int) -> dict[str, float]:
        camera = scene.get("camera") or {}
        ratio = self._camera_ratio(scene, time_ms)
        eased = self._ease_ratio(ratio, str(camera.get("ease") or "linear"))
        start_x = float(camera.get("x", 0.0) or 0.0)
        start_z = float(camera.get("z", 0.0) or 0.0)
        start_zoom = max(0.4, float(camera.get("zoom", 1.0) or 1.0))
        camera_type = str(camera.get("type") or "static").lower()
        if camera_type in {"pan", "move", "truck", "dolly"}:
            x = start_x + (float(camera.get("to_x", start_x) or start_x) - start_x) * eased
            z = start_z + (float(camera.get("to_z", start_z) or start_z) - start_z) * eased
            zoom = start_zoom + (max(0.4, float(camera.get("to_zoom", start_zoom) or start_zoom)) - start_zoom) * eased
        else:
            x = start_x
            z = start_z
            zoom = start_zoom
        return {"x": x, "z": z, "zoom": zoom}

    def _apply_camera_shake(self, scene: dict[str, Any], time_ms: int) -> None:
        impact = self._impact_state(scene, time_ms)
        shake_x = 0.0
        shake_z = 0.0
        state = self._camera_state(scene, time_ms)
        film_scale = 1.0 / max(0.4, state["zoom"])
        self._lens.setFilmSize(self._base_film_width * film_scale, self._base_film_height * film_scale)
        if impact is not None:
            age = float(impact["age"])
            decay = max(0.0, 1.0 - age / 240.0)
            shake_x = math.sin(age * 0.35) * 0.14 * decay
            shake_z = math.cos(age * 0.43) * 0.10 * decay
        cam_x = self.camera_base_pos[0] + state["x"] + shake_x
        cam_z = self.camera_base_pos[2] + state["z"] + shake_z
        look_x = state["x"] * 0.92 + shake_x * 0.4
        look_z = self.frame_center_z + state["z"] * 0.92 + shake_z * 0.3
        self.base.camera.setPos(cam_x, self.camera_base_pos[1], cam_z)
        self.base.camera.lookAt(look_x, 0, look_z)

    def _render_effects(self, scene: dict[str, Any], time_ms: int) -> None:
        for instance in self._impact_instances:
            event = instance["event"]
            impact_age = time_ms - int(event["time_ms"])
            if not (0 <= impact_age <= 420):
                instance["root"].hide()
                continue
            instance["root"].show()
            ratio = impact_age / 420.0
            actor_pose = self._find_actor_pose(scene, event["target_id"], time_ms)
            frontness = self._actor_frontness(scene, event["target_id"])
            impact_scale = self._stage_actor_scale(actor_pose["scale"], frontness)
            impact_x = float(event["impact_x"])
            impact_y = self._actor_stage_y(frontness)
            impact_z = self._stage_actor_visual_z(actor_pose["z"], frontness, impact_scale)
            texture = None
            effect_path = instance.get("effect_path")
            if effect_path:
                texture = self._texture_at_time(effect_path, impact_age, variant_key="impact-fx")
            if texture is not None:
                sequence = instance.get("effect_sequence") or {}
                frame_height = max(1, int(sequence.get("height", 1) or 1))
                frame_width = max(1, int(sequence.get("size", 1) or 1))
                target_height = 1.55 + max(0.0, 1.0 - ratio) * 0.36
                target_width = target_height * (frame_width / frame_height)
                self._update_card_node(
                    instance["gif_node"],
                    target_width,
                    target_height,
                    (impact_x, impact_y - 0.36, impact_z + 0.90),
                    (1.0, 1.0, 1.0, max(0.0, 1.0 - ratio * 0.08)),
                    texture=texture,
                    visible=True,
                )
                for node in instance["rays"]:
                    node.hide()
                for node in instance["rings"]:
                    node.hide()
            else:
                instance["gif_node"].hide()
                radius = 0.10 + ratio * 0.18
                color = (1.0, 0.08, 0.08, max(0.0, 0.76 - ratio * 0.70))
                for idx, node in enumerate(instance["rays"]):
                    angle = idx * 30.0 + ratio * 24.0
                    rad = math.radians(angle)
                    start = (impact_x + math.cos(rad) * radius * 0.18, impact_z + 0.86 + math.sin(rad) * radius * 0.18)
                    end = (impact_x + math.cos(rad) * (radius + 0.18 + ratio * 0.20), impact_z + 0.86 + math.sin(rad) * (radius + 0.18 + ratio * 0.20))
                    self._update_segment_node(node, start, end, 0.026, color, y=impact_y - 0.56, visible=True)
                for idx, node in enumerate(instance["rings"]):
                    alpha = max(0.0, 0.42 - idx * 0.10 - ratio * 0.34)
                    self._update_card_node(
                        node,
                        (radius + idx * 0.08) * 2.0,
                        (radius + idx * 0.08) * 2.0,
                        (impact_x, impact_y - 0.55 - idx * 0.01, impact_z + 0.86),
                        (1.0, 0.28, 0.28, alpha),
                        texture=instance["ring_texture"],
                        visible=alpha > 0.01,
                    )

    def _render_scene(self, scene: dict[str, Any], time_ms: int) -> None:
        effective_time_ms = self._effective_time_ms(scene, time_ms)
        self._current_scene = scene
        self._prepare_scene(scene)
        self._clear_dynamic_frame()
        self._apply_camera_shake(scene, time_ms)
        self._update_background(scene, effective_time_ms)
        self._update_props(effective_time_ms)
        self._advance_npc_ai(scene, effective_time_ms)
        for npc in self._npc_instances:
            self._render_npc(scene, npc, effective_time_ms)
        for actor in scene.get("actors", []):
            self._render_actor(scene, actor["actor_id"], effective_time_ms)
        self._render_effects(scene, time_ms)
        self._render_subtitle(scene, effective_time_ms)

    def _capture_scene_frame_ppm(self) -> Optional[bytes]:
        PNMImage = self._core["PNMImage"]
        StringStream = self._core["StringStream"]
        image = PNMImage()
        if not self.base.win.getScreenshot(image):
            return None
        stream = StringStream()
        if not image.write(stream, "ppm"):
            return None
        payload = stream.getData()
        return bytes(payload) if payload else None

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

    def _capture_scene_frame_png(self) -> Optional[bytes]:
        PNMImage = self._core["PNMImage"]
        StringStream = self._core["StringStream"]
        image = PNMImage()
        if not self.base.win.getScreenshot(image):
            return None
        stream = StringStream()
        if not image.write(stream, "png"):
            return None
        payload = stream.getData()
        return bytes(payload) if payload else None

    def capture_scene_frame(self, scene: dict[str, Any], time_ms: int, *, raw_rgb: bool = False) -> bytes:
        self._render_scene(scene, time_ms)
        self.base.graphicsEngine.renderFrame()
        if raw_rgb:
            payload = self._capture_scene_frame_rgb()
            if payload:
                return payload
            self.base.graphicsEngine.renderFrame()
            payload = self._capture_scene_frame_rgb()
            if payload:
                return payload
            raise RuntimeError("failed to capture raw RGB scene frame")
        payload = self._capture_scene_frame_png()
        if payload:
            return payload
        self.base.graphicsEngine.renderFrame()
        payload = self._capture_scene_frame_png()
        if payload:
            return payload
        raise RuntimeError("failed to capture scene frame")

    def render_scene_frame(self, scene: dict[str, Any], time_ms: int, output_path: Path) -> None:
        Filename = self._core["Filename"]
        self._render_scene(scene, time_ms)
        self.base.graphicsEngine.renderFrame()
        self.base.win.saveScreenshot(Filename.fromOsSpecific(str(output_path)))

    def close(self) -> None:
        for scene_key in list(self._npc_scene_states.keys()):
            self._cleanup_npc_scene_state(scene_key)
        self.base.destroy()
