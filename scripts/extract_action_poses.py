#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageSequence


ROOT_DIR = Path(__file__).resolve().parents[1]
ACTIONS_DIR = ROOT_DIR / "assets" / "actions"
MODELS_DIR = ROOT_DIR / "assets" / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "yolov8n-pose.onnx"
DEFAULT_CACHE_DIR = ACTIONS_DIR / ".cache" / "poses"
DEFAULT_PREVIEW_DIR = ACTIONS_DIR / ".cache" / "pose_previews"
IMAGE_SUFFIXES = {".gif", ".webp", ".png", ".jpg", ".jpeg", ".bmp"}
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
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass
class Detection:
    score: float
    bbox: tuple[float, float, float, float]
    keypoints: np.ndarray


def _iter_action_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and ".cache" not in path.parts
    )


def _load_frames(path: Path) -> tuple[list[np.ndarray], list[int]]:
    frames: list[np.ndarray] = []
    durations: list[int] = []
    with Image.open(path) as image:
        frame_total = int(getattr(image, "n_frames", 1))
        for index in range(frame_total):
            image.seek(index)
            rgba = image.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=rgba.getchannel("A"))
            frames.append(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
            durations.append(max(20, int(image.info.get("duration", 83) or 83)))
    return frames, durations


def _letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    scale = min(size / max(1, width), size / max(1, height))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - resized_w) / 2.0
    pad_y = (size - resized_h) / 2.0
    left = int(math.floor(pad_x))
    top = int(math.floor(pad_y))
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas, scale, (pad_x, pad_y)


def _prepare_tensor(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    canvas, scale, pad = _letterbox(image, size=size)
    tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]
    return tensor, scale, pad


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def _parse_yolov8_pose(
    output: np.ndarray,
    *,
    image_shape: tuple[int, int],
    scale: float,
    pad: tuple[float, float],
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    raw = np.asarray(output)
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim != 2:
        raise ValueError(f"unsupported pose output shape: {tuple(output.shape)}")
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T
    if raw.shape[1] < 56:
        raise ValueError(f"expected YOLOv8 pose output width >= 56, got {raw.shape[1]}")

    boxes_xywh = raw[:, :4]
    scores = raw[:, 4]
    kps = raw[:, 5:56].reshape(-1, 17, 3)
    valid = scores >= conf_threshold
    if not np.any(valid):
        return []

    boxes_xywh = boxes_xywh[valid]
    scores = scores[valid]
    kps = kps[valid]

    boxes = np.empty((boxes_xywh.shape[0], 4), dtype=np.float32)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] * 0.5
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] * 0.5
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] * 0.5
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] * 0.5

    keep = _nms(boxes, scores, iou_threshold)
    width = float(image_shape[1])
    height = float(image_shape[0])
    detections: list[Detection] = []
    for index in keep:
        bbox = boxes[index].copy()
        bbox[[0, 2]] = (bbox[[0, 2]] - pad[0]) / max(scale, 1e-6)
        bbox[[1, 3]] = (bbox[[1, 3]] - pad[1]) / max(scale, 1e-6)
        bbox[0::2] = np.clip(bbox[0::2], 0.0, width - 1.0)
        bbox[1::2] = np.clip(bbox[1::2], 0.0, height - 1.0)

        points = kps[index].copy()
        points[:, 0] = (points[:, 0] - pad[0]) / max(scale, 1e-6)
        points[:, 1] = (points[:, 1] - pad[1]) / max(scale, 1e-6)
        points[:, 0] = np.clip(points[:, 0], 0.0, width - 1.0)
        points[:, 1] = np.clip(points[:, 1], 0.0, height - 1.0)
        detections.append(Detection(score=float(scores[index]), bbox=tuple(float(v) for v in bbox), keypoints=points))
    return detections


def _pick_primary_detection(detections: list[Detection]) -> Detection | None:
    if not detections:
        return None
    return max(
        detections,
        key=lambda item: (
            item.score,
            max(0.0, item.bbox[2] - item.bbox[0]) * max(0.0, item.bbox[3] - item.bbox[1]),
        ),
    )


def _smooth_tracks(track: list[np.ndarray | None], alpha: float) -> list[np.ndarray | None]:
    smoothed: list[np.ndarray | None] = []
    previous: np.ndarray | None = None
    for frame in track:
        if frame is None:
            smoothed.append(previous.copy() if previous is not None else None)
            continue
        current = frame.copy()
        if previous is not None:
            visible = current[:, 2] > 0.05
            current[visible, :2] = previous[visible, :2] * (1.0 - alpha) + current[visible, :2] * alpha
            current[:, 2] = np.maximum(previous[:, 2] * (1.0 - alpha), current[:, 2])
        smoothed.append(current)
        previous = current
    return smoothed


def _joint_angle(points: np.ndarray, a: int, b: int, c: int) -> float | None:
    if points[a, 2] <= 0.05 or points[b, 2] <= 0.05 or points[c, 2] <= 0.05:
        return None
    ba = points[a, :2] - points[b, :2]
    bc = points[c, :2] - points[b, :2]
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= 1e-6:
        return None
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _frame_summary(points: np.ndarray | None) -> dict[str, Any]:
    if points is None:
        return {"keypoints": None, "angles_deg": {}}
    angles = {
        "left_elbow": _joint_angle(points, 5, 7, 9),
        "right_elbow": _joint_angle(points, 6, 8, 10),
        "left_knee": _joint_angle(points, 11, 13, 15),
        "right_knee": _joint_angle(points, 12, 14, 16),
        "left_shoulder": _joint_angle(points, 7, 5, 11),
        "right_shoulder": _joint_angle(points, 8, 6, 12),
        "left_hip": _joint_angle(points, 5, 11, 13),
        "right_hip": _joint_angle(points, 6, 12, 14),
    }
    return {
        "keypoints": [
            {
                "name": POSE_NAMES[index],
                "x": float(points[index, 0]),
                "y": float(points[index, 1]),
                "score": float(points[index, 2]),
            }
            for index in range(len(POSE_NAMES))
        ],
        "angles_deg": {key: (None if value is None else round(float(value), 3)) for key, value in angles.items()},
    }


def _draw_preview(frames: list[np.ndarray], track: list[np.ndarray | None], path: Path) -> None:
    sample_count = min(9, len(frames))
    if sample_count <= 0:
        return
    indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
    thumbs: list[Image.Image] = []
    for frame_index in indices:
        frame = cv2.cvtColor(frames[int(frame_index)], cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(image)
        points = track[int(frame_index)]
        if points is not None:
            for start, end in POSE_EDGES:
                if points[start, 2] > 0.1 and points[end, 2] > 0.1:
                    draw.line(
                        (float(points[start, 0]), float(points[start, 1]), float(points[end, 0]), float(points[end, 1])),
                        fill=(255, 120, 40),
                        width=4,
                    )
            for point in points:
                if point[2] > 0.1:
                    radius = 4
                    draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=(40, 240, 255))
        image.thumbnail((320, 320))
        thumbs.append(image)

    cols = 3
    rows = int(math.ceil(len(thumbs) / cols))
    cell_w = max(image.width for image in thumbs)
    cell_h = max(image.height for image in thumbs)
    canvas = Image.new("RGB", (cell_w * cols, cell_h * rows), (18, 18, 18))
    for index, thumb in enumerate(thumbs):
        x = (index % cols) * cell_w
        y = (index // cols) * cell_h
        canvas.paste(thumb, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _extract_file(
    path: Path,
    session: ort.InferenceSession,
    *,
    output_dir: Path,
    preview_dir: Path,
    conf_threshold: float,
    iou_threshold: float,
    smooth_alpha: float,
) -> Path:
    frames, durations = _load_frames(path)
    track: list[np.ndarray | None] = []
    input_name = session.get_inputs()[0].name
    for frame in frames:
        tensor, scale, pad = _prepare_tensor(frame)
        outputs = session.run(None, {input_name: tensor})
        detections = _parse_yolov8_pose(
            outputs[0],
            image_shape=frame.shape[:2],
            scale=scale,
            pad=pad,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        primary = _pick_primary_detection(detections)
        track.append(primary.keypoints if primary is not None else None)

    smoothed = _smooth_tracks(track, alpha=smooth_alpha)
    relative = path.relative_to(ACTIONS_DIR)
    output_path = output_dir / relative.with_suffix(".pose.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_path": str(relative).replace("\\", "/"),
        "model_path": str(DEFAULT_MODEL_PATH if session is not None else ""),
        "frame_count": len(frames),
        "durations_ms": durations,
        "frames": [_frame_summary(points) for points in smoothed],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    preview_path = preview_dir / relative.with_suffix(".preview.jpg")
    _draw_preview(frames, smoothed, preview_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DNN body poses from local action GIF/WebP assets.")
    parser.add_argument("--input-dir", type=Path, default=ACTIONS_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--conf-threshold", type=float, default=0.35)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--smooth-alpha", type=float, default=0.42)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.model.exists():
        print(f"missing pose model: {args.model}")
        print("expected a YOLOv8 pose ONNX file such as assets/models/yolov8n-pose.onnx")
        return 2

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.model), providers=providers)
    files = _iter_action_files(args.input_dir)
    if not files:
        print(f"no action files found under {args.input_dir}")
        return 0

    written: list[Path] = []
    for path in files:
        written.append(
            _extract_file(
                path,
                session,
                output_dir=args.output_dir,
                preview_dir=args.preview_dir,
                conf_threshold=float(args.conf_threshold),
                iou_threshold=float(args.iou_threshold),
                smooth_alpha=float(args.smooth_alpha),
            )
        )
        print(written[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
