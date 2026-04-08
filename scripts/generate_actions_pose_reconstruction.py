#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageOps


ROOT_DIR = Path(__file__).resolve().parents[1]
POSE_DIR = ROOT_DIR / "assets" / "actions" / ".cache" / "poses"
PREVIEW_DIR = ROOT_DIR / "assets" / "actions" / ".cache" / "pose_previews"
CHARACTER_DIR = ROOT_DIR / "assets" / "characters"
PANDA_PACK_DIR = CHARACTER_DIR / "_shared_skins" / "panda_head_pack"
DEFAULT_FACE_TEXTURE_PATH = CHARACTER_DIR / "face-1" / "skins" / "face_default.png"
DEFAULT_OUTFIT_TEXTURE_PATH = CHARACTER_DIR / "narrator" / "skins" / "outfit.png"
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540
DEFAULT_FPS = 24
FAST_FPS = 12
FAST2_WIDTH = 640
FAST2_HEIGHT = 360
FAST2_FPS = 8
FAST3_WIDTH = 480
FAST3_HEIGHT = 270
FAST3_FPS = 6

POSE_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
POSE_EDGES = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]
HEAD_FILL = (250, 249, 242, 255)
HEAD_OUTLINE = (54, 48, 44, 255)
EAR_FILL = (28, 28, 28, 255)
EAR_OUTLINE = (12, 12, 12, 255)
EYE_PATCH_FILL = (36, 36, 36, 255)
NOSE_FILL = (83, 61, 55, 255)
LIMB_WIDTH = 24
JOINT_RADIUS = 12


def render_scale_for_size(width: int, height: int) -> float:
    return min(width / DEFAULT_WIDTH, height / DEFAULT_HEIGHT)


@dataclass(frozen=True)
class TexturePack:
    character_id: str
    head_base: Image.Image
    face: Image.Image
    body: Image.Image
    arm: Image.Image
    leg: Image.Image
    hand: Image.Image
    foot: Image.Image
    neck: Image.Image
    outfit: Image.Image


@dataclass(frozen=True)
class CharacterPalette:
    torso_fill: tuple[int, int, int, int]
    torso_outline: tuple[int, int, int, int]
    arm_color: tuple[int, int, int]
    leg_color: tuple[int, int, int]
    joint_base: tuple[int, int, int]


@dataclass(frozen=True)
class PoseFrame:
    keypoints: dict[str, np.ndarray]
    people: tuple["PosePerson", ...] = ()


@dataclass(frozen=True)
class PosePerson:
    track_id: int
    keypoints: dict[str, np.ndarray]


@dataclass(frozen=True)
class PoseTrack:
    name: str
    source_path: str
    durations_ms: list[float]
    frames: list[PoseFrame]
    total_duration_s: float
    x_center: float
    y_bottom: float
    scale: float
    head_size: int
    people_count: int


DEFAULT_PALETTE = CharacterPalette(
    torso_fill=(212, 168, 92, 255),
    torso_outline=(157, 110, 58, 255),
    arm_color=(223, 148, 91),
    leg_color=(176, 111, 66),
    joint_base=(154, 96, 57),
)

CHARACTER_PALETTES: dict[str, CharacterPalette] = {
    "general-guard": CharacterPalette((206, 154, 90, 255), (145, 97, 55, 255), (214, 138, 84), (173, 108, 66), (166, 104, 62)),
    "farmer-old": CharacterPalette((176, 142, 96, 255), (123, 92, 58, 255), (170, 126, 82), (136, 98, 66), (141, 102, 71)),
    "narrator": CharacterPalette((164, 168, 182, 255), (92, 98, 114, 255), (152, 160, 178), (122, 132, 151), (128, 137, 156)),
    "npc-boy": CharacterPalette((215, 172, 108, 255), (155, 112, 62, 255), (222, 160, 92), (182, 122, 70), (176, 118, 68)),
    "npc-girl": CharacterPalette((232, 177, 160, 255), (169, 108, 98, 255), (226, 159, 142), (192, 128, 118), (186, 126, 116)),
    "witness-strolling": CharacterPalette((151, 176, 168, 255), (88, 111, 105, 255), (142, 175, 166), (110, 143, 136), (117, 149, 141)),
    "detective-sleek": CharacterPalette((138, 152, 178, 255), (76, 88, 109, 255), (135, 150, 176), (99, 113, 140), (108, 122, 149)),
    "emperor-ming": CharacterPalette((196, 138, 96, 255), (136, 77, 43, 255), (204, 122, 76), (168, 92, 58), (170, 100, 65)),
    "official-minister": CharacterPalette((155, 170, 128, 255), (94, 104, 71, 255), (145, 160, 115), (116, 132, 94), (122, 139, 98)),
    "office-worker-modern": CharacterPalette((180, 162, 201, 255), (116, 95, 144, 255), (174, 150, 197), (140, 118, 171), (149, 127, 178)),
    "reporter-selfie": CharacterPalette((183, 171, 128, 255), (124, 110, 74, 255), (176, 160, 113), (142, 128, 88), (149, 135, 96)),
}


def _load_texture(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


@lru_cache(maxsize=256)
def _load_texture_cached(path_str: str) -> Image.Image:
    return _load_texture(Path(path_str))


def _crop_visible_region(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    side = int(round(max(width, height) * 1.32))
    cx = (left + right) * 0.5
    cy = (top + bottom) * 0.5
    x0 = int(round(cx - side * 0.5))
    y0 = int(round(cy - side * 0.5))
    x1 = x0 + side
    y1 = y0 + side

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
    return image.crop((x0, y0, x1, y1))


def _mix_rgb(a: tuple[int, int, int], b: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    return tuple(int(round(a[i] * (1.0 - ratio) + b[i] * ratio)) for i in range(3))


def _palette_for_character(character_id: str | None) -> CharacterPalette:
    if not character_id:
        return DEFAULT_PALETTE
    return CHARACTER_PALETTES.get(character_id, DEFAULT_PALETTE)


def _edge_color(start: str, palette: CharacterPalette) -> tuple[int, int, int]:
    if "shoulder" in start or "elbow" in start:
        return palette.arm_color
    return palette.leg_color


def _joint_color(name: str, palette: CharacterPalette) -> tuple[int, int, int]:
    if "shoulder" in name:
        return palette.arm_color
    if "elbow" in name:
        return palette.arm_color
    if "wrist" in name:
        return palette.arm_color
    if "hip" in name:
        return palette.leg_color
    if "knee" in name:
        return palette.leg_color
    if "ankle" in name:
        return palette.leg_color
    return palette.joint_base


def _resolve_face_texture_path(
    character_id: str | None = None,
    *,
    expression: str = "default",
    talking: bool = False,
    mouth_open: bool = False,
    face_texture_path: Path | None = None,
) -> Path:
    if face_texture_path is not None:
        return face_texture_path
    skins_dir = CHARACTER_DIR / character_id / "skins" if character_id else None
    if skins_dir is None or not skins_dir.exists():
        return DEFAULT_FACE_TEXTURE_PATH

    normalized = (expression or "default").strip().lower().replace("-", "_")
    state = "open" if mouth_open else "closed"
    candidates: list[str] = []
    if talking:
        candidates.extend(
            [
                f"face_talk_{normalized}_{state}.png",
                f"face_talk_{normalized}_{state}.webp",
                f"face_talk_neutral_{state}.png",
                f"face_talk_neutral_{state}.webp",
            ]
        )
    candidates.extend(
        [
            f"face_{normalized}.png",
            f"face_{normalized}.webp",
            "face_default.png",
            "face_default.webp",
            "face_neutral.png",
            "face_neutral.webp",
        ]
    )
    for candidate_name in candidates:
        candidate = skins_dir / candidate_name
        if candidate.exists():
            return candidate
    return DEFAULT_FACE_TEXTURE_PATH


def _load_face_texture(
    character_id: str | None = None,
    *,
    expression: str = "default",
    talking: bool = False,
    mouth_open: bool = False,
    face_texture_path: Path | None = None,
) -> Image.Image:
    path = _resolve_face_texture_path(
        character_id,
        expression=expression,
        talking=talking,
        mouth_open=mouth_open,
        face_texture_path=face_texture_path,
    )
    return _crop_visible_region(_load_texture_cached(str(path)))


def _resolve_character_skin(
    character_id: str | None = None,
    *,
    face_texture_path: Path | None = None,
    outfit_texture_path: Path | None = None,
) -> tuple[str, Path, Path]:
    resolved_id = character_id or "custom"
    if face_texture_path is None:
        if character_id:
            candidate = CHARACTER_DIR / character_id / "skins" / "face_default.png"
            if not candidate.exists():
                raise FileNotFoundError(f"missing face skin for {character_id}: {candidate}")
            face_texture_path = candidate
        else:
            face_texture_path = DEFAULT_FACE_TEXTURE_PATH
    if outfit_texture_path is None:
        if character_id:
            candidate = CHARACTER_DIR / character_id / "skins" / "outfit.png"
            outfit_texture_path = candidate if candidate.exists() else DEFAULT_OUTFIT_TEXTURE_PATH
        else:
            outfit_texture_path = DEFAULT_OUTFIT_TEXTURE_PATH
    return resolved_id, face_texture_path, outfit_texture_path


def _load_texture_pack(
    character_id: str | None = None,
    *,
    face_texture_path: Path | None = None,
    outfit_texture_path: Path | None = None,
) -> TexturePack:
    resolved_id, face_texture_path, outfit_texture_path = _resolve_character_skin(
        character_id,
        face_texture_path=face_texture_path,
        outfit_texture_path=outfit_texture_path,
    )
    return TexturePack(
        character_id=resolved_id,
        head_base=_load_texture(PANDA_PACK_DIR / "head_base.png"),
        face=_load_texture_cached(str(face_texture_path)),
        body=_load_texture(PANDA_PACK_DIR / "body.png"),
        arm=_load_texture(PANDA_PACK_DIR / "arm.png"),
        leg=_load_texture(PANDA_PACK_DIR / "leg.png"),
        hand=_load_texture(PANDA_PACK_DIR / "hand.png"),
        foot=_load_texture(PANDA_PACK_DIR / "foot.png"),
        neck=_load_texture(PANDA_PACK_DIR / "neck.png"),
        outfit=_load_texture_cached(str(outfit_texture_path)),
    )


TEXTURES = _load_texture_pack()


def _encoding_profile(*, fast: bool = False, fast2: bool = False, fast3: bool = False) -> tuple[str, int]:
    if fast3:
        return ("ultrafast", 32)
    if fast2:
        return ("ultrafast", 30)
    if fast:
        return ("ultrafast", 26)
    return ("medium", 18)


def _open_ffmpeg_stream(
    fps: int,
    width: int,
    height: int,
    output_path: Path,
    *,
    preset: str = "medium",
    crf: int = 18,
) -> subprocess.Popen[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        str(crf),
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def _parse_keypoint_items(
    raw_items: list[dict[str, object]] | None,
    *,
    xs: list[float],
    ys: list[float],
) -> dict[str, np.ndarray]:
    keypoints: dict[str, np.ndarray] = {}
    for item in raw_items or []:
        score = float(item.get("score") or 0.0)
        if score <= 0.05:
            continue
        point = np.array([float(item["x"]), float(item["y"]), score], dtype=np.float32)
        keypoints[str(item["name"])] = point
        xs.append(float(point[0]))
        ys.append(float(point[1]))
    return keypoints


def _load_track(path: Path, *, width: int, height: int) -> PoseTrack:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames: list[PoseFrame] = []
    xs: list[float] = []
    ys: list[float] = []
    people_count = 0
    durations_ms = [float(value) for value in payload.get("durations_ms") or []]
    for raw_frame in payload.get("frames") or []:
        keypoints = _parse_keypoint_items(raw_frame.get("keypoints"), xs=xs, ys=ys)
        people: list[PosePerson] = []
        for person_index, raw_person in enumerate(raw_frame.get("people") or []):
            person_keypoints = _parse_keypoint_items(raw_person.get("keypoints"), xs=xs, ys=ys)
            if not person_keypoints:
                continue
            track_id = int(raw_person.get("track_id", person_index))
            people.append(PosePerson(track_id=track_id, keypoints=person_keypoints))
            people_count = max(people_count, track_id + 1)
        if not people and keypoints:
            people_count = max(people_count, 1)
        frames.append(PoseFrame(keypoints=keypoints, people=tuple(people)))
    if not frames:
        raise ValueError(f"no frames in {path}")
    if not xs or not ys:
        raise ValueError(f"no visible keypoints in {path}")
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span_x = max(1.0, x_max - x_min)
    span_y = max(1.0, y_max - y_min)
    scale = min((width * 0.44) / span_x, (height * 0.62) / span_y)
    total_duration_s = sum(durations_ms) / 1000.0
    head_sizes: list[float] = []
    for frame in frames:
        keypoints = frame.keypoints
        if "left_ear" in keypoints and "right_ear" in keypoints:
            head_sizes.append(abs(float(keypoints["right_ear"][0]) - float(keypoints["left_ear"][0])) * scale * 2.1)
            continue
        if "left_eye" in keypoints and "right_eye" in keypoints:
            head_sizes.append(abs(float(keypoints["right_eye"][0]) - float(keypoints["left_eye"][0])) * scale * 3.2)
            continue
        if "left_shoulder" in keypoints and "right_shoulder" in keypoints:
            head_sizes.append(abs(float(keypoints["right_shoulder"][0]) - float(keypoints["left_shoulder"][0])) * scale * 0.62)
    stable_head_size = int(round(float(np.median(head_sizes)))) if head_sizes else 42
    min_head_size = max(16, int(round(height * 0.04)))
    return PoseTrack(
        name=path.stem.replace(".pose", ""),
        source_path=str(payload.get("source_path") or ""),
        durations_ms=durations_ms,
        frames=frames,
        total_duration_s=total_duration_s,
        x_center=(x_min + x_max) * 0.5,
        y_bottom=y_max,
        scale=scale,
        head_size=max(min_head_size, int(round(stable_head_size * 0.64))),
        people_count=max(1, people_count),
    )


def _frame_people_map(frame: PoseFrame) -> dict[int, dict[str, np.ndarray]]:
    if frame.people:
        return {person.track_id: person.keypoints for person in frame.people if person.keypoints}
    if frame.keypoints:
        return {0: frame.keypoints}
    return {}
def _interpolate_point(a: np.ndarray | None, b: np.ndarray | None, alpha: float) -> np.ndarray | None:
    if a is None and b is None:
        return None
    if a is None:
        return b.copy()
    if b is None:
        return a.copy()
    mixed = a * (1.0 - alpha) + b * alpha
    mixed[2] = max(float(a[2]), float(b[2]))
    return mixed.astype(np.float32)


def _sample_track(track: PoseTrack, t_s: float) -> dict[str, np.ndarray]:
    people = _sample_people_tracks(track, t_s)
    return people[0] if people else {}


def _sample_people_tracks(track: PoseTrack, t_s: float) -> list[dict[str, np.ndarray]]:
    if not track.frames:
        return []
    if len(track.frames) == 1 or t_s <= 0.0:
        return [{name: value.copy() for name, value in people.items()} for _, people in sorted(_frame_people_map(track.frames[0]).items())]
    if t_s >= track.total_duration_s:
        return [{name: value.copy() for name, value in people.items()} for _, people in sorted(_frame_people_map(track.frames[-1]).items())]

    elapsed = 0.0
    for index in range(len(track.frames) - 1):
        duration_s = (track.durations_ms[index] if index < len(track.durations_ms) else 83.0) / 1000.0
        next_elapsed = elapsed + duration_s
        if t_s <= next_elapsed:
            alpha = 0.0 if duration_s <= 1e-6 else (t_s - elapsed) / duration_s
            current_people = _frame_people_map(track.frames[index])
            next_people = _frame_people_map(track.frames[index + 1])
            sampled_people: list[dict[str, np.ndarray]] = []
            for track_id in sorted(set(current_people) | set(next_people)):
                current = current_people.get(track_id, {})
                nxt = next_people.get(track_id, {})
                sampled: dict[str, np.ndarray] = {}
                for name in set(current) | set(nxt):
                    point = _interpolate_point(current.get(name), nxt.get(name), alpha)
                    if point is not None:
                        sampled[name] = point
                if sampled:
                    sampled_people.append(sampled)
            return sampled_people
        elapsed = next_elapsed
    return [{name: value.copy() for name, value in people.items()} for _, people in sorted(_frame_people_map(track.frames[-1]).items())]


def _stage_point(track: PoseTrack, point: np.ndarray, *, width: int, height: int) -> tuple[float, float]:
    x = (float(point[0]) - track.x_center) * track.scale + width * 0.5
    y = (float(point[1]) - track.y_bottom) * track.scale + height * 0.82
    return (x, y)


def _draw_grid(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    grid_color = (218, 224, 232)
    for x in range(0, width, 80):
        draw.line((x, 0, x, height), fill=grid_color, width=1)
    for y in range(0, height, 80):
        draw.line((0, y, width, y), fill=grid_color, width=1)
    draw.line((0, int(height * 0.82), width, int(height * 0.82)), fill=(176, 184, 196), width=2)


def _draw_preview(image: Image.Image, track: PoseTrack, width: int, height: int) -> None:
    preview_path = PREVIEW_DIR / f"{track.name}.preview.jpg"
    if not preview_path.exists():
        return
    preview = Image.open(preview_path).convert("RGB")
    preview.thumbnail((240, 160))
    frame = Image.new("RGB", (preview.width + 12, preview.height + 12), (20, 24, 32))
    frame.paste(preview, (6, 6))
    image.paste(frame, (width - frame.width - 24, 24))


def _draw_label(draw: ImageDraw.ImageDraw, track: PoseTrack, width: int) -> None:
    source_name = Path(track.source_path).name if track.source_path else f"{track.name}.gif"
    draw.rounded_rectangle((24, 20, 360, 92), radius=18, fill=(14, 18, 26))
    draw.text((42, 34), f"DNN Pose: {track.name}", fill=(245, 245, 245))
    draw.text((42, 60), f"source: {source_name}", fill=(150, 160, 180))
    draw.text((width - 222, 24), "YOLOv8 Pose JSON", fill=(180, 190, 210))


def _head_center(points: dict[str, tuple[float, float]], size: int | None = None) -> tuple[float, float] | None:
    shoulders = [points[key] for key in ("left_shoulder", "right_shoulder") if key in points]
    hips = [points[key] for key in ("left_hip", "right_hip") if key in points]
    if shoulders and hips:
        shoulder_center = (
            sum(point[0] for point in shoulders) / len(shoulders),
            sum(point[1] for point in shoulders) / len(shoulders),
        )
        hip_center = (
            sum(point[0] for point in hips) / len(hips),
            sum(point[1] for point in hips) / len(hips),
        )
        up_x = shoulder_center[0] - hip_center[0]
        up_y = shoulder_center[1] - hip_center[1]
        up_len = max(1.0, float(np.hypot(up_x, up_y)))
        up_unit = (up_x / up_len, up_y / up_len)
        torso_center = (
            (shoulder_center[0] + hip_center[0]) * 0.5,
            (shoulder_center[1] + hip_center[1]) * 0.5,
        )
        torso_height = max(18.0, float(np.hypot(hip_center[0] - shoulder_center[0], hip_center[1] - shoulder_center[1])))
        body_ry = torso_height * 0.85
        head_ry = float(size or 42) * 0.80
        tangent_gap = max(1.0, float(size or 42) * 0.04)
        return (
            torso_center[0] + up_unit[0] * (body_ry + head_ry + tangent_gap),
            torso_center[1] + up_unit[1] * (body_ry + head_ry + tangent_gap),
        )

    head_markers = [points[key] for key in ("left_eye", "right_eye", "left_ear", "right_ear") if key in points]
    if len(head_markers) >= 2:
        return (
            sum(point[0] for point in head_markers) / len(head_markers),
            sum(point[1] for point in head_markers) / len(head_markers),
        )
    if "nose" in points:
        return points["nose"]
    ears = [points[key] for key in ("left_ear", "right_ear") if key in points]
    if ears:
        return (
            sum(point[0] for point in ears) / len(ears),
            sum(point[1] for point in ears) / len(ears),
        )
    return None


def _paste_texture(canvas: Image.Image, texture: Image.Image, box: tuple[int, int, int, int], mask: Image.Image | None = None) -> None:
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    fitted = ImageOps.fit(texture, (x1 - x0, y1 - y0), method=Image.Resampling.LANCZOS)
    if mask is not None:
        alpha = fitted.getchannel("A")
        composed_mask = ImageChops.multiply(alpha, mask)
        canvas.paste(fitted, (x0, y0), composed_mask)
    else:
        canvas.paste(fitted, (x0, y0), fitted)


def _paste_segment_texture(
    canvas: Image.Image,
    texture: Image.Image,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    thickness: int,
    length_scale: float = 1.08,
    width_scale: float = 1.7,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(1.0, float(np.hypot(dx, dy)))
    angle = np.degrees(np.arctan2(dy, dx)) - 90.0
    target_w = max(16, int(round(thickness * width_scale)))
    target_h = max(24, int(round(length * length_scale)))
    fitted = ImageOps.fit(texture, (target_w, target_h), method=Image.Resampling.LANCZOS)
    rotated = fitted.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    mid_x = int(round((start[0] + end[0]) * 0.5))
    mid_y = int(round((start[1] + end[1]) * 0.5))
    canvas.paste(rotated, (mid_x - rotated.width // 2, mid_y - rotated.height // 2), rotated)


def _paste_rotated_texture(
    canvas: Image.Image,
    texture: Image.Image,
    center: tuple[float, float],
    *,
    width: float,
    height: float,
    angle_deg: float,
) -> None:
    target_w = max(12, int(round(width)))
    target_h = max(12, int(round(height)))
    fitted = ImageOps.fit(texture, (target_w, target_h), method=Image.Resampling.LANCZOS)
    rotated = fitted.rotate(angle_deg, expand=True, resample=Image.Resampling.BICUBIC)
    cx = int(round(center[0]))
    cy = int(round(center[1]))
    canvas.paste(rotated, (cx - rotated.width // 2, cy - rotated.height // 2), rotated)


def _paste_joint_texture(canvas: Image.Image, texture: Image.Image, center: tuple[float, float], *, size: int) -> None:
    radius = max(8, size // 2)
    x = int(round(center[0] - radius))
    y = int(round(center[1] - radius))
    fitted = ImageOps.fit(texture, (radius * 2, radius * 2), method=Image.Resampling.LANCZOS)
    canvas.paste(fitted, (x, y), fitted)


def _draw_torso(
    draw: ImageDraw.ImageDraw,
    stage_points: dict[str, tuple[float, float]],
    palette: CharacterPalette = DEFAULT_PALETTE,
) -> None:
    required = ("left_shoulder", "right_shoulder", "right_hip", "left_hip")
    if any(key not in stage_points for key in required):
        return
    ls = stage_points["left_shoulder"]
    rs = stage_points["right_shoulder"]
    rh = stage_points["right_hip"]
    lh = stage_points["left_hip"]
    polygon = [ls, rs, rh, lh]
    draw.polygon(polygon, fill=palette.torso_fill, outline=palette.torso_outline)


def _draw_torso_texture(image: Image.Image, stage_points: dict[str, tuple[float, float]], textures: TexturePack = TEXTURES) -> None:
    required = ("left_shoulder", "right_shoulder", "right_hip", "left_hip")
    if any(key not in stage_points for key in required):
        return
    ls = stage_points["left_shoulder"]
    rs = stage_points["right_shoulder"]
    rh = stage_points["right_hip"]
    lh = stage_points["left_hip"]
    polygon = [
        ls,
        rs,
        rh,
        lh,
    ]
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    pad = 18
    x0 = max(0, int(np.floor(min(xs) - pad)))
    y0 = max(0, int(np.floor(min(ys) - pad)))
    x1 = min(image.width, int(np.ceil(max(xs) + pad)))
    y1 = min(image.height, int(np.ceil(max(ys) + pad)))
    if x1 <= x0 or y1 <= y0:
        return
    mask = Image.new("L", (x1 - x0, y1 - y0), 0)
    mask_draw = ImageDraw.Draw(mask)
    shifted = [(x - x0, y - y0) for x, y in polygon]
    mask_draw.polygon(shifted, fill=255)
    _paste_texture(image, textures.body, (x0, y0, x1, y1), mask=mask)
    shoulder_center = ((ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5)
    hip_center = ((lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5)
    torso_center = ((shoulder_center[0] + hip_center[0]) * 0.5, (shoulder_center[1] + hip_center[1]) * 0.5)
    torso_width = max(24.0, float(np.hypot(rs[0] - ls[0], rs[1] - ls[1])))
    torso_height = max(28.0, float(np.hypot(hip_center[0] - shoulder_center[0], hip_center[1] - shoulder_center[1])))
    angle = 90.0 - np.degrees(np.arctan2(hip_center[1] - shoulder_center[1], hip_center[0] - shoulder_center[0]))
    _paste_rotated_texture(
        image,
        textures.outfit,
        (torso_center[0], torso_center[1] + torso_height * 0.20),
        width=torso_width * 1.26,
        height=torso_height * 2.24,
        angle_deg=angle,
    )


def _head_rotation_deg(stage_points: dict[str, tuple[float, float]]) -> float:
    shoulders = [stage_points[key] for key in ("left_shoulder", "right_shoulder") if key in stage_points]
    hips = [stage_points[key] for key in ("left_hip", "right_hip") if key in stage_points]
    if shoulders and hips:
        shoulder_center = (
            sum(point[0] for point in shoulders) / len(shoulders),
            sum(point[1] for point in shoulders) / len(shoulders),
        )
        hip_center = (
            sum(point[0] for point in hips) / len(hips),
            sum(point[1] for point in hips) / len(hips),
        )
        return 90.0 - float(np.degrees(np.arctan2(hip_center[1] - shoulder_center[1], hip_center[0] - shoulder_center[0])))
    if len(shoulders) == 2:
        left, right = shoulders
        return -float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0]))) * 0.55
    return 0.0


def _draw_panda_head(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    stage_points: dict[str, tuple[float, float]],
    *,
    size: int,
    textures: TexturePack = TEXTURES,
    face_texture: Image.Image | None = None,
) -> None:
    center = _head_center(stage_points, size=size)
    if center is None:
        return
    cx, cy = center
    angle_deg = _head_rotation_deg(stage_points)
    rx = size * 0.98
    ry = size * 0.80
    ear_r = size * 0.34
    local_w = max(12, int(round(rx * 2.0 + ear_r * 2.6 + 18.0)))
    local_h = max(12, int(round(ry * 2.0 + ear_r * 2.2 + 18.0)))
    head_layer = Image.new("RGBA", (local_w, local_h), (0, 0, 0, 0))
    head_draw = ImageDraw.Draw(head_layer, "RGBA")
    local_cx = local_w * 0.5
    local_cy = local_h * 0.57
    ear_y = local_cy - ry * 0.98
    left_ear_x = local_cx - rx * 0.64
    right_ear_x = local_cx + rx * 0.64
    outline_w = max(2, int(round(size * 0.05)))
    head_draw.ellipse((left_ear_x - ear_r, ear_y - ear_r, left_ear_x + ear_r, ear_y + ear_r), fill=EAR_FILL, outline=EAR_OUTLINE, width=outline_w)
    head_draw.ellipse((right_ear_x - ear_r, ear_y - ear_r, right_ear_x + ear_r, ear_y + ear_r), fill=EAR_FILL, outline=EAR_OUTLINE, width=outline_w)
    x0 = max(0, int(round(local_cx - rx)))
    y0 = max(0, int(round(local_cy - ry)))
    x1 = min(head_layer.width, int(round(local_cx + rx)))
    y1 = min(head_layer.height, int(round(local_cy + ry)))
    if x1 <= x0 or y1 <= y0:
        return
    mask = Image.new("L", (x1 - x0, y1 - y0), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((0, 0, x1 - x0, y1 - y0), fill=255)
    head_draw.ellipse((local_cx - rx, local_cy - ry, local_cx + rx, local_cy + ry), fill=HEAD_FILL)
    _paste_texture(head_layer, face_texture or textures.face, (x0, y0, x1, y1), mask=mask)
    head_draw.ellipse((local_cx - rx, local_cy - ry, local_cx + rx, local_cy + ry), outline=HEAD_OUTLINE, width=outline_w)
    rotated = head_layer.rotate(angle_deg, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(rotated, (int(round(cx - rotated.width * 0.5)), int(round(cy - rotated.height * 0.5))))


def _draw_pose_actor(
    image: Image.Image,
    stage_points: dict[str, tuple[float, float]],
    *,
    head_size: int,
    textures: TexturePack = TEXTURES,
) -> None:
    palette = _palette_for_character(textures.character_id)
    draw = ImageDraw.Draw(image)
    _draw_torso(draw, stage_points, palette)
    _draw_torso_texture(image, stage_points, textures)
    draw = ImageDraw.Draw(image)
    for start, end in POSE_EDGES:
        if start not in stage_points or end not in stage_points:
            continue
        draw.line((*stage_points[start], *stage_points[end]), fill=_edge_color(start, palette), width=LIMB_WIDTH, joint="curve")
    for name, (x, y) in stage_points.items():
        if name in {"nose", "left_hip", "right_hip"}:
            continue
        radius = JOINT_RADIUS if name != "nose" else JOINT_RADIUS - 1
        color = _joint_color(name, palette)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    _draw_panda_head(image, draw, stage_points, size=head_size, textures=textures)


def _render_frame(
    track: PoseTrack,
    points: dict[str, np.ndarray],
    *,
    width: int,
    height: int,
    textures: TexturePack = TEXTURES,
) -> Image.Image:
    image = Image.new("RGBA", (width, height), (247, 244, 236, 255))
    draw = ImageDraw.Draw(image)
    _draw_grid(draw, width, height)
    _draw_preview(image, track, width, height)
    _draw_label(draw, track, width)
    stage_points = {name: _stage_point(track, point, width=width, height=height) for name, point in points.items()}
    _draw_pose_actor(image, stage_points, head_size=track.head_size, textures=textures)
    return image.convert("RGB")


def _render_people_frame(
    track: PoseTrack,
    people_points: list[dict[str, np.ndarray]],
    *,
    width: int,
    height: int,
    textures: TexturePack = TEXTURES,
) -> Image.Image:
    image = Image.new("RGBA", (width, height), (247, 244, 236, 255))
    draw = ImageDraw.Draw(image)
    _draw_grid(draw, width, height)
    _draw_preview(image, track, width, height)
    _draw_label(draw, track, width)
    for points in people_points:
        stage_points = {name: _stage_point(track, point, width=width, height=height) for name, point in points.items()}
        _draw_pose_actor(image, stage_points, head_size=track.head_size, textures=textures)
    return image.convert("RGB")


def render_video(
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    hold_s: float,
    character_id: str | None = None,
    face_texture_path: Path | None = None,
    outfit_texture_path: Path | None = None,
    fast: bool = False,
    fast2: bool = False,
    fast3: bool = False,
) -> None:
    tracks = [_load_track(path, width=width, height=height) for path in sorted(POSE_DIR.glob("*.pose.json"))]
    if not tracks:
        raise SystemExit(f"no pose tracks found in {POSE_DIR}")
    textures = _load_texture_pack(
        character_id,
        face_texture_path=face_texture_path,
        outfit_texture_path=outfit_texture_path,
    )
    preset, crf = _encoding_profile(fast=fast, fast2=fast2, fast3=fast3)
    ffmpeg_proc = _open_ffmpeg_stream(
        fps,
        width,
        height,
        output_path,
        preset=preset,
        crf=crf,
    )
    try:
        assert ffmpeg_proc.stdin is not None
        for track in tracks:
            hold_frames = max(1, int(round(hold_s * fps)))
            opening = _render_people_frame(track, _sample_people_tracks(track, 0.0), width=width, height=height, textures=textures)
            opening_bytes = opening.tobytes()
            for _ in range(hold_frames):
                ffmpeg_proc.stdin.write(opening_bytes)
            total_frames = max(1, int(round(track.total_duration_s * fps)))
            for frame_index in range(total_frames):
                t_s = min(track.total_duration_s, frame_index / fps)
                frame = _render_people_frame(track, _sample_people_tracks(track, t_s), width=width, height=height, textures=textures)
                ffmpeg_proc.stdin.write(frame.tobytes())
    finally:
        if ffmpeg_proc.stdin is not None:
            ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        if ffmpeg_proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {ffmpeg_proc.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a true DNN pose-track skeleton video from assets/actions/.cache/poses/*.pose.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/actions_dnn_pose_pandahead.mp4"))
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--hold", type=float, default=0.6)
    parser.add_argument("--character", default=None)
    parser.add_argument("--face-texture", type=Path, default=None)
    parser.add_argument("--outfit-texture", type=Path, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--fast2", action="store_true")
    parser.add_argument("--fast3", action="store_true")
    args = parser.parse_args()
    width = args.width
    height = args.height
    fps = args.fps
    hold_s = args.hold
    if args.fast3:
        if width == DEFAULT_WIDTH:
            width = FAST3_WIDTH
        if height == DEFAULT_HEIGHT:
            height = FAST3_HEIGHT
        if fps == DEFAULT_FPS:
            fps = FAST3_FPS
        if hold_s == 0.6:
            hold_s = 0.2
    elif args.fast2:
        if width == DEFAULT_WIDTH:
            width = FAST2_WIDTH
        if height == DEFAULT_HEIGHT:
            height = FAST2_HEIGHT
        if fps == DEFAULT_FPS:
            fps = FAST2_FPS
        if hold_s == 0.6:
            hold_s = 0.3
    elif args.fast and fps == DEFAULT_FPS:
        fps = FAST_FPS
    render_video(
        args.output,
        width=width,
        height=height,
        fps=fps,
        hold_s=hold_s,
        character_id=args.character,
        face_texture_path=args.face_texture,
        outfit_texture_path=args.outfit_texture,
        fast=args.fast,
        fast2=args.fast2,
        fast3=args.fast3,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
