#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from common.io import CHARACTERS_DIR, ROOT_DIR, write_json
from extract_action_poses import (
    DEFAULT_MODEL_PATH,
    POSE_NAMES,
    _parse_yolov8_pose,
    _pick_primary_detection,
    _prepare_tensor,
)


PEOPLE_DIR = ROOT_DIR / "assets" / "people"
PEOPLE_CACHE_DIR = PEOPLE_DIR / ".cache"
POSE_CACHE_DIR = PEOPLE_CACHE_DIR / "poses"
INDEX_PATH = PEOPLE_CACHE_DIR / "white_model_index.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _iter_people_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.name.endswith(":Zone.Identifier")
    )


def _filename_gender(stem: str) -> tuple[str, str]:
    lowered = stem.lower()
    if any(token in stem for token in ("男", "男人", "男生")) or "man" in lowered or "male" in lowered:
        return "masculine", "zh-CN-YunxiNeural"
    if any(token in stem for token in ("女", "女生", "女性")) or "woman" in lowered or "female" in lowered:
        return "feminine", "zh-CN-XiaoxiaoNeural"
    return "neutral", "zh-CN-XiaoyiNeural"


def _asset_id_for(path: Path, counters: dict[str, int]) -> str:
    category = "person"
    stem = path.stem
    lowered = stem.lower()
    if any(token in stem for token in ("男", "男人", "男生")) or "man" in lowered or "male" in lowered:
        category = "man"
    elif any(token in stem for token in ("女", "女生", "女性")) or "woman" in lowered or "female" in lowered:
        category = "woman"
    counters[category] += 1
    return f"white-{category}-{counters[category]:02d}"


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def _split_vertical_views(image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = image.shape[:2]
    if width < int(height * 0.72):
        return None
    mid = width // 2
    gutter = max(2, int(width * 0.02))
    left = image[:, : max(1, mid - gutter)].copy()
    right = image[:, min(width - 1, mid + gutter) :].copy()
    if left.size == 0 or right.size == 0:
        return None
    return left, right


def _detect_primary_pose(
    image: np.ndarray,
    session: ort.InferenceSession,
) -> tuple[Any | None, list[Any]]:
    tensor, scale, pad = _prepare_tensor(image)
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    detections = _parse_yolov8_pose(
        outputs[0],
        image_shape=image.shape[:2],
        scale=scale,
        pad=pad,
        conf_threshold=0.35,
        iou_threshold=0.45,
    )
    return _pick_primary_detection(detections), detections


def _front_view_score(detection: Any | None) -> float:
    if detection is None:
        return -1.0
    face_indices = [0, 1, 2, 3, 4]
    face_score = float(np.sum(detection.keypoints[face_indices, 2]))
    bbox = detection.bbox
    area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return face_score * 10.0 + area * 0.001 + float(detection.score)


def _save_rgba(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    Image.fromarray(rgba).save(path)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    largest_index = max(range(1, count), key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_index, 255, 0).astype(np.uint8)


def _foreground_mask(image: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    height, width = image.shape[:2]
    border = max(8, min(width, height) // 18)
    samples = [
        image[:border, :, :].reshape(-1, 3),
        image[-border:, :, :].reshape(-1, 3),
        image[:, :border, :].reshape(-1, 3),
        image[:, -border:, :].reshape(-1, 3),
    ]
    background = np.median(np.concatenate(samples, axis=0), axis=0)
    distance = np.linalg.norm(image.astype(np.float32) - background.reshape(1, 1, 3).astype(np.float32), axis=2)
    mask = np.where(distance > 24.0, 255, 0).astype(np.uint8)

    x0, y0, x1, y1 = bbox
    pad_x = max(12, int((x1 - x0) * 0.18))
    pad_y = max(12, int((y1 - y0) * 0.12))
    roi = np.zeros_like(mask)
    ix0 = max(0, int(round(x0)) - pad_x)
    iy0 = max(0, int(round(y0)) - pad_y)
    ix1 = min(width, int(round(x1)) + pad_x)
    iy1 = min(height, int(round(y1)) + pad_y)
    roi[iy0:iy1, ix0:ix1] = 255
    mask = cv2.bitwise_and(mask, roi)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = np.where(mask >= 80, 255, 0).astype(np.uint8)
    mask = _largest_component(mask)
    return mask


def _mask_bbox(mask: np.ndarray, fallback: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        x0, y0, x1, y1 = fallback
        return int(x0), int(y0), int(x1), int(y1)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _slice_width(mask: np.ndarray, y: int, bbox: tuple[int, int, int, int]) -> int:
    x0, _, x1, _ = bbox
    line = mask[max(0, min(mask.shape[0] - 1, y)), max(0, x0) : min(mask.shape[1], x1 + 1)]
    xs = np.where(line > 0)[0]
    if len(xs) == 0:
        return max(1, x1 - x0)
    return int(xs.max() - xs.min() + 1)


def _row_segments(mask: np.ndarray, y: int, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    x0, _, x1, _ = bbox
    line = mask[max(0, min(mask.shape[0] - 1, y)), max(0, x0) : min(mask.shape[1], x1 + 1)]
    xs = np.where(line > 0)[0]
    if len(xs) == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = int(xs[0])
    prev = int(xs[0])
    for value in xs[1:]:
        current = int(value)
        if current > prev + 1:
            segments.append((x0 + start, x0 + prev))
            start = current
        prev = current
    segments.append((x0 + start, x0 + prev))
    return segments


def _component_width(mask: np.ndarray, y: int, bbox: tuple[int, int, int, int]) -> int:
    segments = _row_segments(mask, y, bbox)
    if not segments:
        return max(1, bbox[2] - bbox[0])
    widths = [right - left + 1 for left, right in segments]
    if len(widths) == 1:
        return widths[0]
    return int(round(sum(widths) / len(widths)))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point(points: np.ndarray, index: int) -> tuple[float, float]:
    return float(points[index, 0]), float(points[index, 1])


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _white_model_profile(points: np.ndarray, bbox: tuple[int, int, int, int], mask: np.ndarray) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    visible_height = max(1.0, float(y1 - y0))
    shoulders = (_point(points, 5), _point(points, 6))
    hips = (_point(points, 11), _point(points, 12))
    chest = _midpoint(*shoulders)
    pelvis = _midpoint(*hips)
    shoulder_span = _distance(*shoulders)
    hip_span = _distance(*hips)
    torso_length = _distance(chest, pelvis)

    chest_y = int(round(y0 + (y1 - y0) * 0.34))
    waist_y = int(round(y0 + (y1 - y0) * 0.50))
    hip_y = int(round(y0 + (y1 - y0) * 0.61))
    chest_width = _slice_width(mask, chest_y, bbox)
    waist_width = _slice_width(mask, waist_y, bbox)
    hip_width = _slice_width(mask, hip_y, bbox)
    head_width = _slice_width(mask, int(round(y0 + (y1 - y0) * 0.10)), bbox)
    thigh_width = _component_width(mask, int(round(y0 + (y1 - y0) * 0.70)), bbox)
    calf_width = _component_width(mask, int(round(y0 + (y1 - y0) * 0.83)), bbox)
    ankle_width = _component_width(mask, int(round(y0 + (y1 - y0) * 0.94)), bbox)
    head_height = max(24.0, chest[1] - y0)

    chest_ratio = chest_width / visible_height
    waist_ratio = waist_width / visible_height
    hip_ratio = hip_width / visible_height
    head_ratio = head_height / visible_height
    torso_ratio = torso_length / visible_height
    upper_arm_ratio = (
        (_distance(_point(points, 5), _point(points, 7)) + _distance(_point(points, 6), _point(points, 8))) * 0.5
    ) / visible_height
    lower_arm_ratio = (
        (_distance(_point(points, 7), _point(points, 9)) + _distance(_point(points, 8), _point(points, 10))) * 0.5
    ) / visible_height
    upper_leg_ratio = (
        (_distance(_point(points, 11), _point(points, 13)) + _distance(_point(points, 12), _point(points, 14))) * 0.5
    ) / visible_height
    lower_leg_ratio = (
        (_distance(_point(points, 13), _point(points, 15)) + _distance(_point(points, 14), _point(points, 16))) * 0.5
    ) / visible_height

    profile = {
        "head_scale": round(_clamp(head_ratio * 0.96, 0.16, 0.26), 4),
        "head_width_ratio": round(_clamp(head_width / visible_height, 0.08, 0.22), 4),
        "torso_width": round(_clamp(max(chest_ratio * 0.74, waist_ratio * 0.86, hip_ratio * 0.82, shoulder_span / visible_height * 0.72), 0.24, 0.42), 4),
        "torso_height_scale": round(_clamp(torso_ratio * 2.1, 1.00, 1.46), 4),
        "neck_width": round(_clamp(head_ratio * 0.10, 0.03, 0.05), 4),
        "arm_width": round(_clamp((chest_ratio + waist_ratio) * 0.08, 0.03, 0.055), 4),
        "leg_width": round(_clamp(max(thigh_width / visible_height, hip_span / visible_height) * 0.16, 0.03, 0.08), 4),
        "hand_width": round(_clamp(chest_ratio * 0.16, 0.045, 0.075), 4),
        "hand_height": round(_clamp(chest_ratio * 0.24, 0.07, 0.12), 4),
        "foot_width": round(_clamp(hip_ratio * 0.24, 0.08, 0.14), 4),
        "foot_height": round(_clamp(hip_ratio * 0.10, 0.035, 0.065), 4),
        "shoulder_span": round(_clamp(shoulder_span / visible_height, 0.14, 0.30), 4),
        "hip_span": round(_clamp(hip_span / visible_height, 0.10, 0.24), 4),
        "upper_arm_length": round(_clamp(upper_arm_ratio, 0.10, 0.24), 4),
        "lower_arm_length": round(_clamp(lower_arm_ratio, 0.10, 0.24), 4),
        "upper_leg_length": round(_clamp(upper_leg_ratio, 0.16, 0.34), 4),
        "lower_leg_length": round(_clamp(lower_leg_ratio, 0.16, 0.34), 4),
        "image_bbox": [int(x0), int(y0), int(x1), int(y1)],
        "visible_height_px": int(round(visible_height)),
        "chest_width_px": int(chest_width),
        "waist_width_px": int(waist_width),
        "hip_width_px": int(hip_width),
        "head_width_px": int(head_width),
        "thigh_width_px": int(thigh_width),
        "calf_width_px": int(calf_width),
        "ankle_width_px": int(ankle_width),
        "silhouette_rows": _silhouette_rows(mask, bbox),
    }
    return profile


def _silhouette_rows(mask: np.ndarray, bbox: tuple[int, int, int, int], *, samples: int = 48) -> list[dict[str, float]]:
    x0, y0, x1, y1 = bbox
    visible_height = max(1.0, float(y1 - y0))
    center_x = (x0 + x1) * 0.5
    rows: list[dict[str, float]] = []
    for index in range(samples):
        t = index / max(1, samples - 1)
        y = int(round(y0 + t * (y1 - y0)))
        segments = _row_segments(mask, y, bbox)
        if segments:
            left = min(segment[0] for segment in segments)
            right = max(segment[1] for segment in segments)
            width = right - left + 1
            segment_count = len(segments)
            center = (left + right) * 0.5
        else:
            width = 0
            segment_count = 0
            center = center_x
        rows.append(
            {
                "t": round(float(t), 4),
                "width_ratio": round(width / visible_height, 4),
                "center_offset_ratio": round((center - center_x) / visible_height, 4),
                "segment_count": float(segment_count),
            }
        )
    return rows


def _write_mask_preview(image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(path)


def _write_pose_payload(points: np.ndarray, source_path: Path, output_path: Path) -> None:
    payload = {
        "source_path": str(source_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "model_path": str(DEFAULT_MODEL_PATH.relative_to(ROOT_DIR)).replace("\\", "/"),
        "frame_count": 1,
        "durations_ms": [2000],
        "frames": [
            {
                "keypoints": [
                    {
                        "name": POSE_NAMES[index],
                        "x": float(points[index, 0]),
                        "y": float(points[index, 1]),
                        "score": float(points[index, 2]),
                    }
                    for index in range(len(POSE_NAMES))
                ],
                "angles_deg": {},
            }
        ],
    }
    write_json(output_path, payload)


def _extract_single(path: Path, session: ort.InferenceSession, counters: dict[str, int]) -> dict[str, Any]:
    asset_id = _asset_id_for(path, counters)
    character_dir = CHARACTERS_DIR / asset_id
    skins_dir = character_dir / "skins"
    skins_dir.mkdir(parents=True, exist_ok=True)

    image = _load_image(path)
    front_image = image
    back_image: np.ndarray | None = None
    detection = None
    full_detection, full_detections = _detect_primary_pose(image, session)
    split_views = _split_vertical_views(image)
    should_split = False
    if len(full_detections) >= 2:
        centers = sorted(((float((det.bbox[0] + det.bbox[2]) * 0.5), det) for det in full_detections), key=lambda item: item[0])
        if abs(centers[-1][0] - centers[0][0]) >= image.shape[1] * 0.18:
            should_split = True
    elif split_views is not None and image.shape[1] >= int(image.shape[0] * 0.9):
        should_split = True
    if should_split and split_views is not None:
        left_image, right_image = split_views
        left_detection, _ = _detect_primary_pose(left_image, session)
        right_detection, _ = _detect_primary_pose(right_image, session)
        if _front_view_score(left_detection) >= _front_view_score(right_detection):
            front_image = left_image
            back_image = right_image
            detection = left_detection
        else:
            front_image = right_image
            back_image = left_image
            detection = right_detection
    if detection is None:
        detection = full_detection
    if detection is None and front_image is not image:
        detection, _ = _detect_primary_pose(front_image, session)
    if detection is None:
        raise RuntimeError(f"no person pose detected in {path}")

    mask = _foreground_mask(front_image, detection.bbox)
    bbox = _mask_bbox(mask, detection.bbox)
    profile = _white_model_profile(detection.keypoints, bbox, mask)
    gender_presentation, speaker = _filename_gender(path.stem)

    reference_path = skins_dir / "source_reference.png"
    _save_rgba(front_image, reference_path)
    back_reference_path = skins_dir / "source_back_reference.png"
    if back_image is not None:
        _save_rgba(back_image, back_reference_path)
    mask_path = skins_dir / "source_mask.png"
    _write_mask_preview(front_image, mask, mask_path)

    pose_path = POSE_CACHE_DIR / f"{asset_id}.pose.json"
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pose_payload(detection.keypoints, path, pose_path)

    character_payload = {
        "display_name": path.stem,
        "gender_presentation": gender_presentation,
        "tts_speaker_id": speaker,
        "model_style": "stickman",
        "source_person_image": str(reference_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "source_people_sheet": str(path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "source_pose_track": str(pose_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "head_style": "panda_head_stickman",
        "body_color": [0.98, 0.98, 0.97, 1.0],
        "body_secondary_color": [0.12, 0.12, 0.12, 1.0],
        "head_color": [0.985, 0.985, 0.975, 1.0],
        "bone_color": [0.08, 0.08, 0.08, 1.0],
        "accent_color": [0.10, 0.10, 0.10, 1.0],
        "white_model_profile": profile,
        "reference_assets": {
            "source_reference": str(reference_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "source_mask": str(mask_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "source_back_reference": str(back_reference_path.relative_to(ROOT_DIR)).replace("\\", "/") if back_image is not None else None,
        },
    }
    write_json(character_dir / "character.json", character_payload)
    return {
        "asset_id": asset_id,
        "display_name": path.stem,
        "source_path": str(reference_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "source_sheet_path": str(path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "character_dir": str(character_dir.relative_to(ROOT_DIR)).replace("\\", "/"),
        "pose_track_path": str(pose_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "gender_presentation": gender_presentation,
        "white_model_profile": profile,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create white-model character assets from local full-body people images.")
    parser.add_argument("--input-dir", type=Path, default=PEOPLE_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.model.exists():
        print(f"missing pose model: {args.model}")
        return 2
    files = _iter_people_files(args.input_dir)
    if not files:
        print(f"no people images found under {args.input_dir}")
        return 0
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    counters: dict[str, int] = defaultdict(int)
    items: list[dict[str, Any]] = []
    for path in files:
        item = _extract_single(path, session, counters)
        items.append(item)
        print(item["character_dir"])
    write_json(args.index_path, {"items": items})
    print(args.index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
