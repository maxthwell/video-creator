#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen
import re

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFilter

from common.io import CHARACTERS_DIR, ROOT_DIR, write_json


FACES_DIR = ROOT_DIR / "assets" / "faces"
MODELS_DIR = ROOT_DIR / "assets" / "models"
CACHE_DIR = FACES_DIR / ".cache"
EXTRACTED_DIR = CACHE_DIR / "extracted"
MANIFEST_PATH = CACHE_DIR / "face_index.json"
FACE_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
EMOTION_MODEL_PATH = MODELS_DIR / "emotion-ferplus-8.onnx"
FAIRFACE_MODEL_PATH = MODELS_DIR / "fairface.onnx"
FACE_MODEL_URL = "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
EMOTION_MODEL_URL = "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
RAW_EMOTION_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]
EXPRESSION_ALIASES = {
    "default": "neutral",
    "neutral": "neutral",
    "talk": "talk_neutral_closed",
    "talk_neutral": "talk_neutral_closed",
    "talk_neutral_open": "talk_neutral_open",
    "talk_neutral_closed": "talk_neutral_closed",
    "happy": "happy",
    "smile": "happy",
    "grin": "happy",
    "talk_smile_open": "talk_smile_open",
    "talk_smile_closed": "talk_smile_closed",
    "angry": "angry",
    "anger": "angry",
    "fierce": "angry",
    "talk_angry_open": "talk_angry_open",
    "talk_angry_closed": "talk_angry_closed",
    "skeptical": "skeptical",
    "sceptical": "skeptical",
    "disgust": "skeptical",
    "contempt": "skeptical",
    "talk_skeptical_open": "talk_skeptical_open",
    "talk_skeptical_closed": "talk_skeptical_closed",
    "thinking": "thinking",
    "think": "thinking",
    "talk_thinking_open": "talk_thinking_open",
    "talk_thinking_closed": "talk_thinking_closed",
    "excited": "excited",
    "surprise": "excited",
    "fear": "excited",
    "sad": "sad",
}
PROMOTION_TARGETS = {
    "neutral": ["face_default", "face_neutral"],
    "happy": ["face_smile"],
    "angry": ["face_angry"],
    "skeptical": ["face_skeptical"],
    "thinking": ["face_thinking"],
    "excited": ["face_excited"],
    "sad": ["face_sad"],
    "talk_neutral_open": ["face_talk_neutral_open"],
    "talk_neutral_closed": ["face_talk_neutral_closed"],
    "talk_smile_open": ["face_talk_smile_open"],
    "talk_smile_closed": ["face_talk_smile_closed"],
    "talk_angry_open": ["face_talk_angry_open"],
    "talk_angry_closed": ["face_talk_angry_closed"],
    "talk_skeptical_open": ["face_talk_skeptical_open"],
    "talk_skeptical_closed": ["face_talk_skeptical_closed"],
    "talk_thinking_open": ["face_talk_thinking_open"],
    "talk_thinking_closed": ["face_talk_thinking_closed"],
}
MODEL_TO_CANONICAL_LABEL = {
    "neutral": "neutral",
    "happiness": "happy",
    "surprise": "excited",
    "sadness": "sad",
    "anger": "angry",
    "disgust": "skeptical",
    "fear": "excited",
    "contempt": "skeptical",
}
TALK_FALLBACKS = {
    "talk_neutral_open": "neutral",
    "talk_neutral_closed": "neutral",
    "talk_smile_open": "happy",
    "talk_smile_closed": "happy",
    "talk_angry_open": "angry",
    "talk_angry_closed": "angry",
    "talk_skeptical_open": "skeptical",
    "talk_skeptical_closed": "skeptical",
    "talk_thinking_open": "thinking",
    "talk_thinking_closed": "thinking",
}
FAIRFACE_RACE_LABELS = [
    "white",
    "black",
    "latino_hispanic",
    "east_asian",
    "southeast_asian",
    "indian",
    "middle_eastern",
]
FAIRFACE_GENDER_LABELS = ["masculine", "feminine"]
FAIRFACE_AGE_LABELS = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+",
]
SOURCE_SENTINELS = {"raw", "input", "inputs", "images", "image", "source", "sources", "photos"}


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    landmarks: np.ndarray | None


@dataclass
class DemographicsEstimate:
    gender: str | None
    gender_confidence: float
    age_group: str | None
    age_confidence: float
    race: str | None
    race_confidence: float


@dataclass
class ExtractedFace:
    image: Image.Image
    landmarks: np.ndarray | None


@dataclass
class PromotedCandidate:
    score: float
    base_path: Path
    talk_open_path: Path | None = None
    talk_closed_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect all faces under assets/faces, classify emotion labels, cache crops, and promote reusable skins.")
    parser.add_argument("--input-dir", default=str(FACES_DIR))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--size", type=int, default=512, help="Output square size for extracted PNG faces.")
    parser.add_argument("--min-face", type=int, default=60, help="Minimum detected face size in pixels.")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached extracted crops and rebuild everything.")
    parser.add_argument("--no-download", action="store_true", help="Do not attempt to download missing models.")
    parser.add_argument("--no-promote", action="store_true", help="Skip writing selected crops into assets/characters/*/skins.")
    return parser.parse_args()


def ensure_dirs() -> None:
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)


def _download_if_missing(path: Path, url: str, *, allow_download: bool) -> bool:
    if path.exists():
        return True
    if not allow_download:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=30) as response:
            data = response.read()
    except Exception:
        return False
    path.write_bytes(data)
    return path.exists() and path.stat().st_size > 0


def _source_hash(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(str(path.relative_to(ROOT_DIR)).encode("utf-8"))
    stat = path.stat()
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()[:16]


def _all_source_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and ".cache" not in path.parts]


def _character_ids() -> set[str]:
    if not CHARACTERS_DIR.exists():
        return set()
    return {path.name for path in CHARACTERS_DIR.iterdir() if path.is_dir() and not path.name.startswith("_")}


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", value.strip().lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "face"


def _normalize_requested_expression(raw: str | None) -> str | None:
    if not raw:
        return None
    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    return EXPRESSION_ALIASES.get(key)


def _infer_requested_expression(source: Path, input_dir: Path, character_ids: set[str]) -> tuple[str | None, str | None]:
    rel = source.relative_to(input_dir)
    parts = list(rel.parts[:-1])
    character_id = parts[0] if parts and parts[0] in character_ids else None
    candidates: list[str] = []
    if character_id is not None:
        candidates.extend(parts[1:])
    else:
        candidates.extend(parts)
    candidates.append(source.stem)
    for candidate in reversed(candidates):
        for token in candidate.replace("__", "/").replace("-", "_").split("/"):
            normalized = _normalize_requested_expression(token)
            if normalized:
                return character_id, normalized
    return character_id, None


def _infer_group_character_id(source: Path, input_dir: Path) -> tuple[str, str]:
    rel = source.relative_to(input_dir)
    parts = list(rel.parts[:-1])
    stem = _slugify(source.stem)
    if parts:
        first = _slugify(parts[0])
        if first and first not in SOURCE_SENTINELS and _normalize_requested_expression(first) is None:
            character_id = f"face-{first}"
            display_name = parts[0]
            return character_id, display_name
    character_id = f"face-{stem}"
    display_name = source.stem
    return character_id, display_name


def _load_image_bgr(path: Path) -> np.ndarray | None:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def _detect_faces_yunet(image: np.ndarray, model_path: Path, min_face: int) -> list[Detection]:
    height, width = image.shape[:2]
    detector = cv2.FaceDetectorYN_create(str(model_path), "", (width, height), 0.7, 0.3, 5000)
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None:
        return []
    detections: list[Detection] = []
    for row in faces:
        x, y, w, h = [float(v) for v in row[:4]]
        if min(w, h) < min_face:
            continue
        landmarks = np.array(row[4:14], dtype=np.float32).reshape(5, 2)
        score = float(row[14]) if len(row) > 14 else 1.0
        detections.append(Detection((x, y, w, h), score, landmarks))
    return detections


def _detect_faces_haar(image: np.ndarray, min_face: int) -> list[Detection]:
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face))
    detections: list[Detection] = []
    for (x, y, w, h) in faces:
        detections.append(Detection((float(x), float(y), float(w), float(h)), 0.5, None))
    return detections


def _align_image(image: np.ndarray, landmarks: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    if landmarks is None or len(landmarks) < 2:
        return image, landmarks
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    center = tuple(np.mean([left_eye, right_eye], axis=0))
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if landmarks is None:
        return aligned, None
    pts = np.hstack([landmarks, np.ones((landmarks.shape[0], 1), dtype=np.float32)])
    rotated = pts @ matrix.T
    return aligned, rotated.astype(np.float32)


def _expanded_square(bbox: tuple[float, float, float, float], image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    side = max(w * 1.26, h * 1.34)
    cx = x + w * 0.5
    cy = y + h * 0.46
    left = int(round(cx - side * 0.5))
    top = int(round(cy - side * 0.42))
    right = int(round(left + side))
    bottom = int(round(top + side))
    height, width = image_shape[:2]
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    return left, top, right, bottom


def _soft_ellipse_mask(size: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    arr = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    cx = size * 0.5
    cy = size * 0.48
    rx = size * 0.40
    ry = size * 0.47
    dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    arr[dist <= 1.0] = 255
    mask = Image.fromarray(arr, mode="L").filter(ImageFilter.GaussianBlur(radius=max(4, size // 40)))
    return mask


def _sorted_face_landmarks(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eyes = sorted(landmarks[:2], key=lambda point: float(point[0]))
    mouth = sorted(landmarks[3:5], key=lambda point: float(point[0]))
    left_eye = np.asarray(eyes[0], dtype=np.float32)
    right_eye = np.asarray(eyes[1], dtype=np.float32)
    nose = np.asarray(landmarks[2], dtype=np.float32)
    mouth_left = np.asarray(mouth[0], dtype=np.float32)
    mouth_right = np.asarray(mouth[1], dtype=np.float32)
    return left_eye, right_eye, nose, mouth_left, mouth_right


def _landmark_crop_rect(landmarks: np.ndarray, bbox: tuple[float, float, float, float], image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    left_eye, right_eye, nose, mouth_left, mouth_right = _sorted_face_landmarks(landmarks)
    eye_center = (left_eye + right_eye) * 0.5
    mouth_center = (mouth_left + mouth_right) * 0.5
    eye_dist = max(1.0, float(np.linalg.norm(right_eye - left_eye)))
    mouth_width = max(1.0, float(mouth_right[0] - mouth_left[0]))
    eye_to_mouth = max(1.0, float(mouth_center[1] - eye_center[1]))
    face_width = max(float(bbox[2]) * 0.92, mouth_width * 1.75, eye_dist * 2.15)
    face_height = max(float(bbox[3]) * 0.92, eye_to_mouth * 2.55, eye_dist * 2.45)
    center_x = float((eye_center[0] + nose[0] + mouth_center[0]) / 3.0)
    center_y = float(eye_center[1] + eye_to_mouth * 0.62)
    left = int(round(center_x - face_width * 0.50))
    right = int(round(center_x + face_width * 0.50))
    top = int(round(eye_center[1] - face_height * 0.34))
    bottom = int(round(eye_center[1] + face_height * 0.66))
    height, width = image_shape[:2]
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    return left, top, right, bottom


def _transform_landmarks_to_crop(
    landmarks: np.ndarray | None,
    crop_rect: tuple[int, int, int, int],
    size: int,
) -> np.ndarray | None:
    if landmarks is None:
        return None
    left, top, right, bottom = crop_rect
    width = max(1, right - left)
    height = max(1, bottom - top)
    scaled = landmarks.copy().astype(np.float32)
    scaled[:, 0] = (scaled[:, 0] - float(left)) * (float(size) / float(width))
    scaled[:, 1] = (scaled[:, 1] - float(top)) * (float(size) / float(height))
    return scaled


def _head_polygon_mask(size: int, landmarks: np.ndarray | None) -> Image.Image:
    if landmarks is None or len(landmarks) < 5:
        return _soft_ellipse_mask(size).filter(ImageFilter.GaussianBlur(radius=max(2, size // 96)))
    left_eye, right_eye, nose, mouth_left, mouth_right = _sorted_face_landmarks(landmarks)
    eye_center = (left_eye + right_eye) * 0.5
    mouth_center = (mouth_left + mouth_right) * 0.5
    eye_dist = max(1.0, float(np.linalg.norm(right_eye - left_eye)))
    mouth_width = max(1.0, float(mouth_right[0] - mouth_left[0]))
    eye_to_mouth = max(1.0, float(mouth_center[1] - eye_center[1]))
    cx = float((eye_center[0] + nose[0] + mouth_center[0]) / 3.0)
    cy = float(eye_center[1] + eye_to_mouth * 0.58)
    rx = max(mouth_width * 0.96, eye_dist * 1.08)
    ry = max(eye_to_mouth * 1.48, eye_dist * 1.36)
    yy, xx = np.ogrid[:size, :size]
    dist = ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[dist <= 1.0] = 255
    arr[int(min(size - 1, cy + ry * 0.96)) :, :] = 0
    arr[:, : max(0, int(cx - rx * 1.18))] = 0
    arr[:, int(min(size, cx + rx * 1.18)) :] = 0
    mask = Image.fromarray(arr, mode="L").filter(ImageFilter.GaussianBlur(radius=max(2, size // 96)))
    return mask


def _alpha_cutout(image_bgr: np.ndarray, landmarks: np.ndarray | None) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).convert("RGBA")
    size = pil.size[0]
    pil.putalpha(_head_polygon_mask(size, landmarks))
    return pil


def _extract_crop(image: np.ndarray, detection: Detection, size: int) -> ExtractedFace:
    aligned, aligned_landmarks = _align_image(image, detection.landmarks)
    bbox = detection.bbox
    if aligned_landmarks is not None and len(aligned_landmarks) >= 5:
        left, top, right, bottom = _landmark_crop_rect(aligned_landmarks, bbox, aligned.shape)
    else:
        left, top, right, bottom = _expanded_square(bbox, aligned.shape)
    crop = aligned[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("empty crop")
    crop_landmarks = _transform_landmarks_to_crop(aligned_landmarks, (left, top, right, bottom), size)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return ExtractedFace(image=_alpha_cutout(crop, crop_landmarks), landmarks=crop_landmarks)


class EmotionClassifier:
    def __init__(self, model_path: Path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def classify(self, face_rgba: Image.Image) -> tuple[str, dict[str, float]]:
        gray = face_rgba.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
        data = np.asarray(gray, dtype=np.float32)[None, None, :, :]
        outputs = self.session.run(None, {self.input_name: data})[0].reshape(-1)
        exp = np.exp(outputs - np.max(outputs))
        probs = exp / np.maximum(np.sum(exp), 1e-6)
        raw_scores = {label: float(prob) for label, prob in zip(RAW_EMOTION_LABELS, probs)}
        raw_label = max(raw_scores.items(), key=lambda item: item[1])[0]
        canonical_scores: dict[str, float] = {}
        for raw_label_name, score in raw_scores.items():
            canonical = MODEL_TO_CANONICAL_LABEL.get(raw_label_name, raw_label_name)
            canonical_scores[canonical] = canonical_scores.get(canonical, 0.0) + float(score)
        canonical_label = MODEL_TO_CANONICAL_LABEL.get(raw_label, raw_label)
        return canonical_label, canonical_scores


class FairFaceClassifier:
    def __init__(self, model_path: Path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def classify(self, face_rgba: Image.Image) -> DemographicsEstimate:
        image = face_rgba.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        array = ((array - mean) / std).transpose(2, 0, 1)[None, :, :, :]
        raw = self.session.run(None, {self.input_name: array})
        if len(raw) == 3 and all(np.asarray(item).size == 1 for item in raw):
            race_index = int(np.asarray(raw[0]).reshape(-1)[0])
            gender_index = int(np.asarray(raw[1]).reshape(-1)[0])
            age_index = int(np.asarray(raw[2]).reshape(-1)[0])
            if not (0 <= race_index < len(FAIRFACE_RACE_LABELS) and 0 <= gender_index < len(FAIRFACE_GENDER_LABELS) and 0 <= age_index < len(FAIRFACE_AGE_LABELS)):
                return DemographicsEstimate(None, 0.0, None, 0.0, None, 0.0)
            return DemographicsEstimate(
                gender=FAIRFACE_GENDER_LABELS[gender_index],
                gender_confidence=1.0,
                age_group=FAIRFACE_AGE_LABELS[age_index],
                age_confidence=1.0,
                race=FAIRFACE_RACE_LABELS[race_index],
                race_confidence=1.0,
            )

        logits = np.asarray(raw[0]).reshape(-1)
        if logits.size < 18:
            return DemographicsEstimate(None, 0.0, None, 0.0, None, 0.0)
        race_scores = _softmax(logits[:7])
        gender_scores = _softmax(logits[7:9])
        age_scores = _softmax(logits[9:18])
        race_index = int(np.argmax(race_scores))
        gender_index = int(np.argmax(gender_scores))
        age_index = int(np.argmax(age_scores))
        return DemographicsEstimate(
            gender=FAIRFACE_GENDER_LABELS[gender_index],
            gender_confidence=float(gender_scores[gender_index]),
            age_group=FAIRFACE_AGE_LABELS[age_index],
            age_confidence=float(age_scores[age_index]),
            race=FAIRFACE_RACE_LABELS[race_index],
            race_confidence=float(race_scores[race_index]),
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp), 1e-6)


def _save_face_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def _sample_median_rgba(image: Image.Image, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    rgba = image.convert("RGBA")
    width, height = rgba.size
    left = max(0, min(width - 1, int(round(box[0]))))
    top = max(0, min(height - 1, int(round(box[1]))))
    right = max(left + 1, min(width, int(round(box[2]))))
    bottom = max(top + 1, min(height, int(round(box[3]))))
    region = np.asarray(rgba.crop((left, top, right, bottom)), dtype=np.uint8)
    if region.size == 0:
        return (180, 120, 120, 255)
    flat = region.reshape(-1, 4)
    opaque = flat[flat[:, 3] > 16]
    if len(opaque) == 0:
        opaque = flat
    values = np.median(opaque, axis=0)
    return tuple(int(np.clip(v, 0, 255)) for v in values)


def _mouth_geometry(image: Image.Image, landmarks: np.ndarray | None) -> dict[str, float]:
    size = float(image.size[0])
    if landmarks is not None and len(landmarks) >= 5:
        left_eye, right_eye, nose, mouth_left, mouth_right = _sorted_face_landmarks(landmarks)
        mouth_center = (mouth_left + mouth_right) * 0.5
        mouth_width = max(24.0, float(np.linalg.norm(mouth_right - mouth_left)) * 1.04)
        eye_dist = max(28.0, float(np.linalg.norm(right_eye - left_eye)))
        center_x = float(mouth_center[0])
        center_y = float(mouth_center[1] + eye_dist * 0.045)
    else:
        center_x = size * 0.5
        center_y = size * 0.62
        mouth_width = size * 0.18
        eye_dist = size * 0.18
    return {
        "cx": center_x,
        "cy": center_y,
        "mouth_width": mouth_width,
        "eye_dist": eye_dist,
    }


def _mouth_patch_overlay(image: Image.Image, mouth: dict[str, float], *, emotion: str, state: str) -> Image.Image:
    width, height = image.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    cx = mouth["cx"]
    cy = mouth["cy"]
    mouth_width = mouth["mouth_width"]
    eye_dist = mouth["eye_dist"]
    patch_half_w = mouth_width * 0.58
    patch_half_h = eye_dist * (0.15 if state == "closed" else 0.18)
    skin_color = _sample_median_rgba(image, (cx - patch_half_w, cy - patch_half_h * 1.8, cx + patch_half_w, cy + patch_half_h * 1.9))
    lip_color = _sample_median_rgba(image, (cx - mouth_width * 0.34, cy - eye_dist * 0.06, cx + mouth_width * 0.34, cy + eye_dist * 0.12))
    mouth_fill = (
        max(18, int(lip_color[0] * 0.28)),
        max(10, int(lip_color[1] * 0.18)),
        max(10, int(lip_color[2] * 0.20)),
        235,
    )
    lip_line = (
        min(255, int(lip_color[0] * 0.95 + 12)),
        min(255, int(lip_color[1] * 0.68 + 8)),
        min(255, int(lip_color[2] * 0.70 + 10)),
        228,
    )
    patch_alpha = 212 if state == "closed" else 160
    draw.ellipse(
        (
            cx - patch_half_w,
            cy - patch_half_h,
            cx + patch_half_w,
            cy + patch_half_h,
        ),
        fill=(skin_color[0], skin_color[1], skin_color[2], patch_alpha),
    )
    if state == "closed":
        line_half_w = mouth_width * (0.34 if emotion in {"happy", "smile"} else 0.3)
        line_h = max(3.0, mouth_width * 0.035)
        line_y = cy + (mouth_width * (0.025 if emotion in {"happy", "smile"} else 0.01))
        draw.rounded_rectangle(
            (
                cx - line_half_w,
                line_y - line_h,
                cx + line_half_w,
                line_y + line_h,
            ),
            radius=max(2, int(line_h)),
            fill=lip_line,
        )
        shadow_y = line_y + line_h * 1.4
        draw.rounded_rectangle(
            (
                cx - line_half_w * 0.78,
                shadow_y - line_h * 0.6,
                cx + line_half_w * 0.78,
                shadow_y + line_h * 0.6,
            ),
            radius=max(1, int(line_h * 0.8)),
            fill=(60, 40, 40, 58),
        )
    else:
        outer_half_w = mouth_width * (0.34 if emotion in {"happy", "smile"} else 0.31)
        outer_half_h = eye_dist * (0.16 if emotion in {"happy", "smile"} else 0.15)
        mouth_y = cy + eye_dist * 0.035
        draw.ellipse(
            (
                cx - outer_half_w,
                mouth_y - outer_half_h,
                cx + outer_half_w,
                mouth_y + outer_half_h,
            ),
            fill=(lip_line[0], lip_line[1], lip_line[2], 188),
        )
        inner_half_w = outer_half_w * 0.82
        inner_half_h = outer_half_h * (0.72 if emotion in {"happy", "smile"} else 0.78)
        draw.ellipse(
            (
                cx - inner_half_w,
                mouth_y - inner_half_h,
                cx + inner_half_w,
                mouth_y + inner_half_h,
            ),
            fill=mouth_fill,
        )
        teeth_y = mouth_y - inner_half_h * 0.3
        draw.rounded_rectangle(
            (
                cx - inner_half_w * 0.58,
                teeth_y - inner_half_h * 0.32,
                cx + inner_half_w * 0.58,
                teeth_y + inner_half_h * 0.08,
            ),
            radius=max(2, int(inner_half_h * 0.25)),
            fill=(236, 226, 220, 120),
        )
    return overlay.filter(ImageFilter.GaussianBlur(radius=max(1.2, width / 180.0)))


def _synthesize_talk_variant(image: Image.Image, landmarks: np.ndarray | None, *, emotion: str, state: str) -> Image.Image:
    base = image.convert("RGBA")
    mouth = _mouth_geometry(base, landmarks)
    overlay = _mouth_patch_overlay(base, mouth, emotion=emotion, state=state)
    composed = Image.alpha_composite(base, overlay)
    composed.putalpha(base.getchannel("A"))
    return composed


def _variant_paths(extracted_path: Path) -> tuple[Path, Path]:
    stem = extracted_path.stem
    return (
        extracted_path.with_name(f"{stem}_talk_open.png"),
        extracted_path.with_name(f"{stem}_talk_closed.png"),
    )


def _ensure_talk_variants(
    extracted_path: Path,
    face_image: Image.Image,
    landmarks: np.ndarray | None,
    *,
    emotion: str,
    refresh: bool,
) -> tuple[Path, Path]:
    talk_open_path, talk_closed_path = _variant_paths(extracted_path)
    if refresh or not talk_open_path.exists():
        open_image = _synthesize_talk_variant(face_image, landmarks, emotion=emotion, state="open")
        _save_face_image(open_image, talk_open_path)
    if refresh or not talk_closed_path.exists():
        closed_image = _synthesize_talk_variant(face_image, landmarks, emotion=emotion, state="closed")
        _save_face_image(closed_image, talk_closed_path)
    return talk_open_path, talk_closed_path


def _source_path_for_slot(slot: str, candidate: PromotedCandidate) -> Path:
    if slot.endswith("_open") and candidate.talk_open_path is not None:
        return candidate.talk_open_path
    if slot.endswith("_closed") and candidate.talk_closed_path is not None:
        return candidate.talk_closed_path
    return candidate.base_path


def _promote_slot(character_id: str, slot: str, source_path: Path) -> list[str]:
    stems = PROMOTION_TARGETS.get(slot, [])
    if not stems:
        return []
    skins_dir = CHARACTERS_DIR / character_id / "skins"
    skins_dir.mkdir(parents=True, exist_ok=True)
    promoted: list[str] = []
    for stem in stems:
        destination = skins_dir / f"{stem}.png"
        shutil.copy2(source_path, destination)
        promoted.append(_relative(destination))
    return promoted


def _ensure_talk_fallbacks(character_id: str, chosen_slots: dict[str, PromotedCandidate]) -> list[str]:
    promoted: list[str] = []
    for talk_slot, base_slot in TALK_FALLBACKS.items():
        if talk_slot in chosen_slots:
            continue
        candidate = chosen_slots.get(base_slot)
        if candidate is None:
            continue
        promoted.extend(_promote_slot(character_id, talk_slot, _source_path_for_slot(talk_slot, candidate)))
    return promoted


def _weighted_vote(records: list[tuple[str | None, float]]) -> tuple[str | None, float]:
    totals: dict[str, float] = {}
    for label, weight in records:
        if not label:
            continue
        totals[label] = totals.get(label, 0.0) + max(0.0, float(weight))
    if not totals:
        return None, 0.0
    label, score = max(totals.items(), key=lambda item: item[1])
    total = sum(totals.values())
    confidence = score / total if total > 0 else 0.0
    return label, float(confidence)


def _default_voice_for_gender(gender: str | None) -> str:
    if gender == "feminine":
        return "zh-CN-XiaoxiaoNeural"
    return "zh-CN-YunxiNeural"


def _write_character_metadata(
    *,
    character_id: str,
    display_name: str,
    source_path: str,
    detection_count: int,
    promoted_paths: list[str],
    demographics: DemographicsEstimate | None,
    emotion_counts: dict[str, int],
) -> str:
    character_dir = CHARACTERS_DIR / character_id
    character_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "display_name": display_name,
        "gender_presentation": demographics.gender if demographics and demographics.gender else "unspecified",
        "tts_speaker_id": _default_voice_for_gender(demographics.gender if demographics else None),
        "head_style": "expressive_head",
        "head_anchor": {"offset": [0, -10], "scale": 1.0},
        "source_face_image": source_path,
        "detected_face_count": int(detection_count),
        "generated_face_skins": promoted_paths,
        "emotion_distribution": emotion_counts,
        "demographics_estimate": {
            "age_group": demographics.age_group if demographics else None,
            "age_confidence": round(float(demographics.age_confidence), 4) if demographics else 0.0,
            "gender": demographics.gender if demographics else None,
            "gender_confidence": round(float(demographics.gender_confidence), 4) if demographics else 0.0,
            "race": demographics.race if demographics else None,
            "race_confidence": round(float(demographics.race_confidence), 4) if demographics else 0.0,
            "is_estimated": True,
        },
    }
    target = character_dir / "character.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return _relative(target)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    extracted_dir = cache_dir / "extracted"
    manifest_path = cache_dir / "face_index.json"
    face_model_path = Path(args.models_dir).resolve() / FACE_MODEL_PATH.name
    emotion_model_path = Path(args.models_dir).resolve() / EMOTION_MODEL_PATH.name
    allow_download = not args.no_download

    ensure_dirs()
    cache_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    has_yunet = _download_if_missing(face_model_path, FACE_MODEL_URL, allow_download=allow_download)
    has_emotion_model = _download_if_missing(emotion_model_path, EMOTION_MODEL_URL, allow_download=allow_download)
    classifier = EmotionClassifier(emotion_model_path) if has_emotion_model else None
    fairface_classifier: FairFaceClassifier | None = None
    if FAIRFACE_MODEL_PATH.exists():
        try:
            fairface_classifier = FairFaceClassifier(FAIRFACE_MODEL_PATH)
        except Exception:
            fairface_classifier = None

    character_ids = _character_ids()
    sources = _all_source_images(input_dir)
    manifest: dict[str, Any] = {
        "input_dir": _relative(input_dir),
        "models": {
            "face_detector": _relative(face_model_path) if face_model_path.exists() else None,
            "emotion_classifier": _relative(emotion_model_path) if emotion_model_path.exists() else None,
            "demographics_classifier": _relative(FAIRFACE_MODEL_PATH) if FAIRFACE_MODEL_PATH.exists() else None,
        },
        "entries": [],
        "generated_characters": [],
    }
    best_by_character_slot: dict[tuple[str, str], PromotedCandidate] = {}
    character_groups: dict[str, dict[str, Any]] = {}

    for source in sources:
        image = _load_image_bgr(source)
        if image is None:
            continue
        character_id, requested_slot = _infer_requested_expression(source, input_dir, character_ids)
        generated_character_id, display_name = _infer_group_character_id(source, input_dir)
        if has_yunet:
            detections = _detect_faces_yunet(image, face_model_path, args.min_face)
            detector_name = "yunet"
        else:
            detections = _detect_faces_haar(image, args.min_face)
            detector_name = "haar"
        detections.sort(key=lambda item: (item.score, item.bbox[2] * item.bbox[3]), reverse=True)
        source_hash = _source_hash(source)
        group = character_groups.setdefault(
            generated_character_id,
            {
                "character_id": generated_character_id,
                "display_name": display_name,
                "source_path": _relative(source),
                "entries": [],
                "emotion_counts": {},
                "gender_votes": [],
                "age_votes": [],
                "race_votes": [],
            },
        )
        for face_index, detection in enumerate(detections):
            bbox = detection.bbox
            cache_name = f"{source_hash}_f{face_index:02d}.png"
            extracted_path = extracted_dir / cache_name
            crop_landmarks: np.ndarray | None = None
            if args.refresh or not extracted_path.exists():
                try:
                    extracted_face = _extract_crop(image, detection, args.size)
                    face_image = extracted_face.image
                    crop_landmarks = extracted_face.landmarks
                    _save_face_image(face_image, extracted_path)
                except Exception:
                    continue
            else:
                face_image = Image.open(extracted_path).convert("RGBA")
            if crop_landmarks is None and detection.landmarks is not None:
                try:
                    crop_landmarks = _extract_crop(image, detection, args.size).landmarks
                except Exception:
                    crop_landmarks = None

            if classifier is not None:
                predicted_label, emotion_scores = classifier.classify(face_image)
            else:
                predicted_label, emotion_scores = "unknown", {}
            demographics = fairface_classifier.classify(face_image) if fairface_classifier is not None else None
            talk_open_path, talk_closed_path = _ensure_talk_variants(
                extracted_path,
                face_image,
                crop_landmarks,
                emotion=predicted_label,
                refresh=args.refresh,
            )

            slot = requested_slot or (predicted_label if predicted_label in PROMOTION_TARGETS else None)
            promoted_paths: list[str] = []
            score_value = float(detection.score) * float(emotion_scores.get(predicted_label, 1.0))
            if slot and not args.no_promote:
                current = best_by_character_slot.get((generated_character_id, slot))
                if current is None or score_value >= current.score:
                    best_by_character_slot[(generated_character_id, slot)] = PromotedCandidate(
                        score=score_value,
                        base_path=extracted_path,
                        talk_open_path=talk_open_path,
                        talk_closed_path=talk_closed_path,
                    )
            group["entries"].append({"cache_path": extracted_path, "slot": slot})
            if predicted_label and predicted_label != "unknown":
                emotion_counts = group["emotion_counts"]
                emotion_counts[predicted_label] = int(emotion_counts.get(predicted_label, 0)) + 1
            if demographics is not None:
                if demographics.gender:
                    group["gender_votes"].append((demographics.gender, float(demographics.gender_confidence) * float(detection.score)))
                if demographics.age_group:
                    group["age_votes"].append((demographics.age_group, float(demographics.age_confidence) * float(detection.score)))
                if demographics.race:
                    group["race_votes"].append((demographics.race, float(demographics.race_confidence) * float(detection.score)))

            manifest["entries"].append(
                {
                    "source_path": _relative(source),
                    "character_id": generated_character_id,
                    "requested_character_id": character_id,
                    "display_name": display_name,
                    "requested_slot": requested_slot,
                    "face_index": face_index,
                    "detector": detector_name,
                    "bbox": {
                        "x": round(float(bbox[0]), 2),
                        "y": round(float(bbox[1]), 2),
                        "w": round(float(bbox[2]), 2),
                        "h": round(float(bbox[3]), 2),
                    },
                    "score": round(float(detection.score), 4),
                    "emotion_label": predicted_label,
                    "emotion_scores": {key: round(value, 4) for key, value in emotion_scores.items()},
                    "demographics_estimate": {
                        "gender": demographics.gender if demographics else None,
                        "gender_confidence": round(float(demographics.gender_confidence), 4) if demographics else 0.0,
                        "age_group": demographics.age_group if demographics else None,
                        "age_confidence": round(float(demographics.age_confidence), 4) if demographics else 0.0,
                        "race": demographics.race if demographics else None,
                        "race_confidence": round(float(demographics.race_confidence), 4) if demographics else 0.0,
                    },
                    "cache_path": _relative(extracted_path),
                    "talk_open_cache_path": _relative(talk_open_path),
                    "talk_closed_cache_path": _relative(talk_closed_path),
                    "promoted_paths": promoted_paths,
                }
            )

    promoted_lookup: dict[str, list[str]] = {}
    chosen_slots: dict[str, dict[str, PromotedCandidate]] = {}
    if not args.no_promote:
        for (character_id, slot), candidate in best_by_character_slot.items():
            source_path = _source_path_for_slot(slot, candidate)
            promoted = _promote_slot(character_id, slot, source_path)
            promoted_lookup[_relative(source_path)] = promoted_lookup.get(_relative(source_path), []) + promoted
            chosen_slots.setdefault(character_id, {})[slot] = candidate
        for character_id, slots in chosen_slots.items():
            extras = _ensure_talk_fallbacks(character_id, slots)
            if extras:
                fallback_source = next(iter(slots.values()))
                promoted_lookup[_relative(fallback_source.base_path)] = promoted_lookup.get(_relative(fallback_source.base_path), []) + extras

    for entry in manifest["entries"]:
        entry["promoted_paths"] = promoted_lookup.get(entry["cache_path"], [])

    if not args.no_promote:
        for character_id, group in sorted(character_groups.items()):
            gender, gender_conf = _weighted_vote(group["gender_votes"])
            age_group, age_conf = _weighted_vote(group["age_votes"])
            race, race_conf = _weighted_vote(group["race_votes"])
            demographics = DemographicsEstimate(gender=gender, gender_confidence=gender_conf, age_group=age_group, age_confidence=age_conf, race=race, race_confidence=race_conf)
            promoted_paths = sorted({path for entry in manifest["entries"] if entry["character_id"] == character_id for path in entry.get("promoted_paths", [])})
            meta_path = _write_character_metadata(
                character_id=character_id,
                display_name=str(group["display_name"]),
                source_path=str(group["source_path"]),
                detection_count=len(group["entries"]),
                promoted_paths=promoted_paths,
                demographics=demographics,
                emotion_counts=dict(group["emotion_counts"]),
            )
            manifest["generated_characters"].append(
                {
                    "character_id": character_id,
                    "display_name": group["display_name"],
                    "source_path": group["source_path"],
                    "character_meta_path": meta_path,
                    "promoted_paths": promoted_paths,
                    "detected_face_count": len(group["entries"]),
                    "demographics_estimate": {
                        "gender": demographics.gender,
                        "gender_confidence": round(float(demographics.gender_confidence), 4),
                        "age_group": demographics.age_group,
                        "age_confidence": round(float(demographics.age_confidence), 4),
                        "race": demographics.race,
                        "race_confidence": round(float(demographics.race_confidence), 4),
                    },
                }
            )

    write_json(manifest_path, manifest)
    print(f"faces_dir={_relative(input_dir)}")
    print(f"sources={len(sources)}")
    print(f"detections={len(manifest['entries'])}")
    print(f"characters_generated={len(manifest['generated_characters'])}")
    print(f"detector={'yunet' if has_yunet else 'haar'}")
    print(f"emotion_model={'enabled' if classifier is not None else 'missing'}")
    print(f"demographics_model={'enabled' if fairface_classifier is not None else 'missing'}")
    print(_relative(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
