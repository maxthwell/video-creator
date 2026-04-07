#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cv2
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    GraphicsOutput,
    PerspectiveLens,
    Texture,
    TransparencyAttrib,
    loadPrcFileData,
)
from direct.showbase.ShowBase import ShowBase
from smplx import SMPL

from common.io import ROOT_DIR, TMP_DIR, ensure_runtime_dirs


INDEX_PATH = ROOT_DIR / "assets" / "people" / ".cache" / "white_model_index.json"
SMPL_MODEL_DIR = Path("/root/.cache/4DHumans/data/smpl")
PART_ORDER = ["head", "torso", "arm_left", "arm_right", "leg_left", "leg_right"]


def _configure_panda3d(width: int, height: int) -> None:
    cache_dir = TMP_DIR / "panda_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    loadPrcFileData("", "load-display p3tinydisplay")
    loadPrcFileData("", "window-type offscreen")
    loadPrcFileData("", f"win-size {width} {height}")
    loadPrcFileData("", "audio-library-name null")
    loadPrcFileData("", "sync-video false")
    loadPrcFileData("", f"model-cache-dir {cache_dir}")


def _open_ffmpeg_stream(fps: int, width: int, height: int, output_path: Path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "20",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def _capture_frame_bytes(base: ShowBase) -> bytes:
    capture_texture = getattr(base, "_codex_capture_texture", None)
    if capture_texture is None:
        capture_texture = Texture()
        capture_texture.setKeepRamImage(True)
        base.win.addRenderTexture(capture_texture, GraphicsOutput.RTM_copy_ram)
        base._codex_capture_texture = capture_texture
    for _ in range(2):
        base.graphicsEngine.renderFrame()
        if capture_texture.hasRamImage():
            payload = capture_texture.getRamImageAs("RGB")
            if payload:
                frame_bytes = bytes(payload)
                row_stride = base.win.getXSize() * 3
                return b"".join(
                    frame_bytes[row_start : row_start + row_stride]
                    for row_start in range(len(frame_bytes) - row_stride, -1, -row_stride)
                )
    raise RuntimeError("failed to capture frame")


def _rotation_matrix(axis: str, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _load_entries() -> list[dict[str, Any]]:
    payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    items_by_asset = {str(item.get("asset_id")): item for item in payload.get("items") or []}
    entries = []
    for mesh in payload.get("hmr2_meshes") or []:
        asset_id = str(mesh.get("asset_id"))
        item = items_by_asset.get(asset_id, {})
        character_path = ROOT_DIR / str(item.get("character_dir", "")) / "character.json"
        character = json.loads(character_path.read_text(encoding="utf-8"))
        params = json.loads((ROOT_DIR / str(mesh["hmr2_params_path"])).read_text(encoding="utf-8"))
        texture_path = character.get("hmr2_texture_path") or character.get("reference_assets", {}).get("source_reference")
        entries.append(
            {
                "asset_id": asset_id,
                "display_name": item.get("display_name", asset_id),
                "character": character,
                "params": params,
                "texture_path": ROOT_DIR / str(texture_path),
                "reference_path": ROOT_DIR / str(character.get("reference_assets", {}).get("source_reference", texture_path)),
                "back_reference_path": (
                    ROOT_DIR / str(character.get("reference_assets", {}).get("source_back_reference"))
                    if character.get("reference_assets", {}).get("source_back_reference")
                    else None
                ),
                "mask_path": ROOT_DIR / str(character.get("reference_assets", {}).get("source_mask", texture_path)),
            }
        )
    return entries


def _prepare_projected_texture(
    reference_path: Path,
    mask_path: Path,
    cache_key: str,
    *,
    back_reference_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    texture_dir = TMP_DIR / "hmr2_textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    output_path = texture_dir / f"{cache_key}.png"
    reference = Image.open(reference_path).convert("RGBA")
    auto_front_mask = _auto_subject_mask(reference)
    if mask_path.exists():
        provided_mask = Image.open(mask_path).convert("L").resize(reference.size, Image.Resampling.BILINEAR)
        front_mask = Image.fromarray(
            np.maximum(np.asarray(provided_mask, dtype=np.uint8), np.asarray(auto_front_mask, dtype=np.uint8)),
            mode="L",
        )
    else:
        front_mask = auto_front_mask
    reference_masked = reference.copy()
    reference_masked.putalpha(front_mask)
    if back_reference_path is not None and back_reference_path.exists():
        mirrored = Image.open(back_reference_path).convert("RGBA").resize(reference.size, Image.Resampling.BILINEAR)
        mirrored_mask = _auto_subject_mask(mirrored)
    else:
        mirrored = reference.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mirrored_rgb = np.asarray(mirrored, dtype=np.uint8).copy()
        mirrored_rgb[..., :3] = np.clip(mirrored_rgb[..., :3].astype(np.float32) * 0.92, 0, 255).astype(np.uint8)
        mirrored = Image.fromarray(mirrored_rgb, mode="RGBA")
        mirrored_mask = front_mask
    mirrored_masked = mirrored.copy()
    mirrored_masked.putalpha(mirrored_mask)
    return output_path, {
        "front": _solidify_rgba(reference_masked),
        "back": _solidify_rgba(mirrored_masked),
        "front_masked": reference_masked,
        "back_masked": mirrored_masked,
    }


def _auto_subject_mask(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    near_white = np.all(rgb >= 246, axis=2)
    edges = cv2.Canny(gray, 40, 120) > 0
    protected = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    background_candidates = (near_white & ~protected).astype(np.uint8)
    _, labels = cv2.connectedComponents(background_candidates, connectivity=4)
    border_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    background = np.isin(labels, border_labels) & (background_candidates > 0)
    mask = np.where(background, 0, 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return Image.fromarray(mask, mode="L")


def _solidify_rgba(image: Image.Image) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    kernel = np.ones((5, 5), np.uint8)
    for _ in range(64):
        unknown = alpha == 0
        if not np.any(unknown):
            break
        dilated_alpha = cv2.dilate(alpha, kernel, iterations=1)
        if not np.any(dilated_alpha > 0):
            break
        dilated_rgb = np.stack([cv2.dilate(rgb[:, :, channel], kernel, iterations=1) for channel in range(3)], axis=2)
        newly = unknown & (dilated_alpha > 0)
        rgb[newly] = dilated_rgb[newly]
        alpha[newly] = 255
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def _blend_rgba(a: Image.Image, b: Image.Image, alpha: float) -> Image.Image:
    arr_a = np.asarray(a.convert("RGBA"), dtype=np.float32)
    arr_b = np.asarray(b.convert("RGBA"), dtype=np.float32)
    mixed = arr_a * (1.0 - alpha) + arr_b * alpha
    mixed[:, :, 3] = 255.0
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8), mode="RGBA")


def _rgba_arrays(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return rgba[:, :, :3].astype(np.float32), (rgba[:, :, 3].astype(np.float32) / 255.0)


def _estimate_part_fill(front_crop: Image.Image, back_crop: Image.Image, default_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    pixels = []
    for crop in (front_crop, back_crop):
        rgba = np.asarray(crop.convert("RGBA"), dtype=np.uint8)
        alpha = rgba[:, :, 3] > 32
        rgb = rgba[:, :, :3]
        valid = (
            alpha
            & (rgb[:, :, 0] > 120)
            & (rgb[:, :, 1] > 90)
            & (rgb[:, :, 2] > 70)
            & (rgb[:, :, 0] < 245)
            & (rgb[:, :, 1] < 240)
            & (rgb[:, :, 2] < 235)
        )
        if np.any(valid):
            pixels.append(rgb[valid])
    if not pixels:
        return default_rgb
    merged = np.concatenate(pixels, axis=0)
    median = np.median(merged, axis=0)
    return tuple(int(v) for v in np.clip(median, 0, 255))


def _compute_projected_uvs(
    vertices: np.ndarray,
    *,
    normals: np.ndarray,
    camera_translation: np.ndarray,
    focal_length: float,
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    verts_cam = vertices + camera_translation[None, :]
    z = np.clip(verts_cam[:, 2], 1e-4, None)
    x = focal_length * (verts_cam[:, 0] / z) + (width * 0.5)
    y = focal_length * (verts_cam[:, 1] / z) + (height * 0.5)
    u_front = np.clip(x / max(1.0, float(width - 1)), 0.0, 1.0)
    v = np.clip(1.0 - (y / max(1.0, float(height - 1))), 0.0, 1.0)
    front_mask = normals[:, 2] < -0.32
    back_mask = normals[:, 2] > 0.32
    side_mask = ~(front_mask | back_mask)
    left_side = side_mask & (normals[:, 0] < 0.0)
    right_side = side_mask & ~left_side
    side_u = np.clip(0.5 + (u_front - 0.5) * 0.36, 0.18, 0.82)
    u = np.empty_like(u_front)
    u[front_mask] = u_front[front_mask] * 0.25
    u[left_side] = 0.25 + side_u[left_side] * 0.25
    u[back_mask] = 0.50 + (1.0 - u_front[back_mask]) * 0.25
    u[right_side] = 0.75 + side_u[right_side] * 0.25
    return np.stack([u, v], axis=1).astype(np.float32)


def _rotate_y(vertices: np.ndarray, angle: float) -> np.ndarray:
    return vertices @ _rotation_matrix("y", angle).T


def _project_vertices(vertices: np.ndarray, camera_translation: np.ndarray, focal_length: float, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    verts_cam = vertices + camera_translation[None, :]
    z = np.clip(verts_cam[:, 2], 1e-4, None)
    x = focal_length * (verts_cam[:, 0] / z) + (width * 0.5)
    y = focal_length * (verts_cam[:, 1] / z) + (height * 0.5)
    return np.stack([x, y], axis=1).astype(np.float32)


def _camera_space_depths(vertices: np.ndarray, camera_translation: np.ndarray) -> np.ndarray:
    return (vertices + camera_translation[None, :])[:, 2].astype(np.float32)


def _vertex_parts(vertices: np.ndarray) -> np.ndarray:
    x = vertices[:, 0]
    y = vertices[:, 1]
    y_min = float(y.min())
    y_max = float(y.max())
    y_norm = (y - y_min) / max(1e-6, y_max - y_min)
    abs_x = np.abs(x)
    max_abs_x = max(1e-6, float(abs_x.max()))
    labels = np.full(len(vertices), 1, dtype=np.int32)  # torso
    labels[y_norm > 0.82] = 0  # head
    leg_mask = y_norm < 0.42
    labels[leg_mask & (x < 0.0)] = 4
    labels[leg_mask & (x >= 0.0)] = 5
    arm_mask = (y_norm >= 0.36) & (y_norm < 0.82) & (abs_x / max_abs_x > 0.46)
    labels[arm_mask & (x < 0.0)] = 2
    labels[arm_mask & (x >= 0.0)] = 3
    return labels


def _face_parts(faces: np.ndarray, vertex_parts: np.ndarray) -> np.ndarray:
    labels = np.empty(len(faces), dtype=np.int32)
    for index, face in enumerate(faces):
        part_ids = vertex_parts[np.asarray(face, dtype=np.int32)]
        labels[index] = int(np.bincount(part_ids, minlength=len(PART_ORDER)).argmax())
    return labels


def _mask_from_faces(points_2d: np.ndarray, faces: np.ndarray, face_parts: np.ndarray, part_id: int, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    for face, label in zip(faces, face_parts):
        if int(label) != part_id:
            continue
        polygon = np.round(points_2d[np.asarray(face, dtype=np.int32)]).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def _crop_to_mask(image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    width, height = image.size
    if len(xs) == 0 or len(ys) == 0:
        return image.copy(), (0, 0, width, height)
    x0 = max(0, int(xs.min()) - 2)
    y0 = max(0, int(ys.min()) - 2)
    x1 = min(width, int(xs.max()) + 3)
    y1 = min(height, int(ys.max()) + 3)
    return image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)


def _compute_cylindrical_part_uvs(vertices: np.ndarray, vertex_parts: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    tile_width = 256.0
    tile_height = 256.0
    atlas_width = tile_width
    atlas_height = tile_height * len(PART_ORDER)
    uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    part_meta: dict[str, Any] = {
        "tile_width": tile_width,
        "tile_height": tile_height,
        "atlas_width": atlas_width,
        "atlas_height": atlas_height,
    }
    for part_index, part_name in enumerate(PART_ORDER):
        mask = vertex_parts == part_index
        if not np.any(mask):
            continue
        coords = vertices[mask]
        center_x = float(np.median(coords[:, 0]))
        center_z = float(np.median(coords[:, 2]))
        y_min = float(coords[:, 1].min())
        y_max = float(coords[:, 1].max())
        angles = np.arctan2(coords[:, 2] - center_z, coords[:, 0] - center_x)
        local_u = (angles + math.pi) / (2.0 * math.pi)
        local_v = 1.0 - ((coords[:, 1] - y_min) / max(1e-6, y_max - y_min))
        row_top = part_index * tile_height
        uvs[mask, 0] = local_u.astype(np.float32)
        uvs[mask, 1] = (1.0 - ((row_top + local_v * tile_height) / atlas_height)).astype(np.float32)
        part_meta[part_name] = {
            "row": part_index,
            "y_min": y_min,
            "y_max": y_max,
            "center_x": center_x,
            "center_z": center_z,
        }
    return uvs, part_meta


def _canonical_dest_triangles(face_uv: np.ndarray, tile_width: float, tile_height: float, row_index: int) -> list[np.ndarray]:
    local_u = face_uv[:, 0].astype(np.float32).copy()
    local_v = (((1.0 - face_uv[:, 1]) * len(PART_ORDER)) - row_index).astype(np.float32)
    if float(local_u.max() - local_u.min()) > 0.5:
        local_u[local_u < 0.5] += 1.0
    triangles = []
    base_x = local_u * tile_width
    base_y = local_v * tile_height
    triangles.append(np.stack([base_x, base_y], axis=1))
    if np.any(base_x > tile_width):
        triangles.append(np.stack([base_x - tile_width, base_y], axis=1))
    if np.any(base_x < 0.0):
        triangles.append(np.stack([base_x + tile_width, base_y], axis=1))
    return triangles


def _face_visibility(points_2d: np.ndarray, depths: np.ndarray, faces: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    zbuffer = np.full((height, width), np.inf, dtype=np.float32)
    samples_per_face: list[tuple[np.ndarray, float]] = []
    for face in faces:
        tri = points_2d[np.asarray(face, dtype=np.int32)].astype(np.float32)
        depth = float(depths[np.asarray(face, dtype=np.int32)].mean())
        samples = np.stack(
            [
                tri.mean(axis=0),
                (tri[0] + tri[1]) * 0.5,
                (tri[1] + tri[2]) * 0.5,
                (tri[2] + tri[0]) * 0.5,
            ],
            axis=0,
        )
        samples_per_face.append((samples, depth))
        for sample in samples:
            x = int(np.clip(round(float(sample[0])), 0, width - 1))
            y = int(np.clip(round(float(sample[1])), 0, height - 1))
            if depth < zbuffer[y, x]:
                zbuffer[y, x] = depth
    visible = np.zeros(len(faces), dtype=bool)
    for face_index, (samples, depth) in enumerate(samples_per_face):
        for sample in samples:
            x = int(np.clip(round(float(sample[0])), 0, width - 1))
            y = int(np.clip(round(float(sample[1])), 0, height - 1))
            if depth <= zbuffer[y, x] + 0.03:
                visible[face_index] = True
                break
    return visible


def _paint_triangle(
    atlas_rgb: np.ndarray,
    atlas_weight: np.ndarray,
    atlas_offset: tuple[int, int],
    dest_tri: np.ndarray,
    source_rgb: np.ndarray,
    source_alpha: np.ndarray,
    source_tri: np.ndarray,
    strength: float,
) -> None:
    if strength <= 1e-4:
        return
    tri = np.asarray(dest_tri, dtype=np.float32)
    tri[:, 0] += atlas_offset[0]
    tri[:, 1] += atlas_offset[1]
    x0 = max(0, int(math.floor(float(tri[:, 0].min()))))
    y0 = max(0, int(math.floor(float(tri[:, 1].min()))))
    x1 = min(atlas_rgb.shape[1], int(math.ceil(float(tri[:, 0].max()))) + 1)
    y1 = min(atlas_rgb.shape[0], int(math.ceil(float(tri[:, 1].max()))) + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    local_tri = tri - np.array([x0, y0], dtype=np.float32)
    src_tri = np.asarray(source_tri, dtype=np.float32)
    warp_rgb = cv2.getAffineTransform(src_tri, local_tri)
    warp_alpha = cv2.getAffineTransform(src_tri, local_tri)
    patch_size = (x1 - x0, y1 - y0)
    patch_rgb = cv2.warpAffine(
        source_rgb,
        warp_rgb,
        patch_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    patch_alpha = cv2.warpAffine(
        source_alpha,
        warp_alpha,
        patch_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    tri_mask = np.zeros((patch_size[1], patch_size[0]), dtype=np.float32)
    cv2.fillConvexPoly(tri_mask, np.round(local_tri).astype(np.int32), 1.0)
    patch_weight = np.clip(patch_alpha * tri_mask * strength, 0.0, 1.0)
    if not np.any(patch_weight > 1e-5):
        return
    atlas_rgb[y0:y1, x0:x1] += patch_rgb * patch_weight[:, :, None]
    atlas_weight[y0:y1, x0:x1] += patch_weight


def _finalize_atlas(atlas_rgb: np.ndarray, atlas_weight: np.ndarray, fallback: Image.Image) -> Image.Image:
    safe_weight = np.where(atlas_weight > 1e-5, atlas_weight, 1.0)
    filled = atlas_rgb / safe_weight[:, :, None]
    fallback_rgb = np.asarray(fallback.convert("RGBA"), dtype=np.uint8)
    low_weight_mask = atlas_weight <= 0.22
    filled[low_weight_mask] = fallback_rgb[:, :, :3][low_weight_mask].astype(np.float32)
    pale_mask = (
        (filled[:, :, 0] > 238.0)
        & (filled[:, :, 1] > 238.0)
        & (filled[:, :, 2] > 238.0)
        & (atlas_weight <= 0.65)
    )
    filled[pale_mask] = fallback_rgb[:, :, :3][pale_mask].astype(np.float32)
    output = np.zeros((atlas_rgb.shape[0], atlas_rgb.shape[1], 4), dtype=np.uint8)
    output[:, :, :3] = np.clip(filled, 0, 255).astype(np.uint8)
    output[:, :, 3] = 255
    return _solidify_rgba(Image.fromarray(output, mode="RGBA"))


def _build_triangle_baked_atlas(
    cache_key: str,
    *,
    front_image: Image.Image,
    back_image: Image.Image,
    front_masked: Image.Image,
    back_masked: Image.Image,
    front_points: np.ndarray,
    back_points: np.ndarray,
    faces: np.ndarray,
    vertex_parts: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    front_visible: np.ndarray,
    back_visible: np.ndarray,
) -> Path:
    texture_dir = TMP_DIR / "hmr2_textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    atlas_path = texture_dir / f"{cache_key}.png"
    tile_width = 256
    tile_height = 256
    atlas_width = tile_width
    atlas_height = tile_height * len(PART_ORDER)
    atlas_rgb = np.zeros((atlas_height, atlas_width, 3), dtype=np.float32)
    atlas_weight = np.zeros((atlas_height, atlas_width), dtype=np.float32)
    fallback = Image.new("RGBA", (atlas_width, atlas_height), (0, 0, 0, 255))

    front_rgb, _ = _rgba_arrays(front_image)
    back_rgb, _ = _rgba_arrays(back_image)
    _, front_alpha = _rgba_arrays(front_masked)
    _, back_alpha = _rgba_arrays(back_masked)
    face_parts = _face_parts(faces, vertex_parts)
    part_source_rgb: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for part_index, part_name in enumerate(PART_ORDER):
        front_mask = _mask_from_faces(front_points, faces, face_parts, part_index, front_image.size)
        back_mask = _mask_from_faces(back_points, faces, face_parts, part_index, back_image.size)
        front_crop, _ = _crop_to_mask(front_image, front_mask)
        back_crop, _ = _crop_to_mask(back_image, back_mask)
        preview = _blend_rgba(
            _solidify_rgba(front_crop).resize((tile_width, tile_height), Image.Resampling.BILINEAR),
            _solidify_rgba(back_crop).resize((tile_width, tile_height), Image.Resampling.BILINEAR),
            0.5,
        )
        if part_name != "head":
            fill_rgb = _estimate_part_fill(front_crop, back_crop, (224, 190, 164))
            front_part_rgb = front_rgb.copy()
            back_part_rgb = back_rgb.copy()
            front_pale = (
                (front_part_rgb[:, :, 0] > 236.0)
                & (front_part_rgb[:, :, 1] > 232.0)
                & (front_part_rgb[:, :, 2] > 226.0)
                & (front_alpha > 0.05)
            )
            back_pale = (
                (back_part_rgb[:, :, 0] > 236.0)
                & (back_part_rgb[:, :, 1] > 232.0)
                & (back_part_rgb[:, :, 2] > 226.0)
                & (back_alpha > 0.05)
            )
            front_part_rgb[front_pale, 0] = fill_rgb[0]
            front_part_rgb[front_pale, 1] = fill_rgb[1]
            front_part_rgb[front_pale, 2] = fill_rgb[2]
            back_part_rgb[back_pale, 0] = fill_rgb[0]
            back_part_rgb[back_pale, 1] = fill_rgb[1]
            back_part_rgb[back_pale, 2] = fill_rgb[2]
            part_source_rgb[part_index] = (front_part_rgb, back_part_rgb)
            preview_arr = np.asarray(preview.convert("RGBA"), dtype=np.uint8).copy()
            pale = (
                (preview_arr[:, :, 0] > 236)
                & (preview_arr[:, :, 1] > 232)
                & (preview_arr[:, :, 2] > 226)
            )
            preview_arr[pale, 0] = fill_rgb[0]
            preview_arr[pale, 1] = fill_rgb[1]
            preview_arr[pale, 2] = fill_rgb[2]
            preview = Image.fromarray(preview_arr, mode="RGBA")
        else:
            part_source_rgb[part_index] = (front_rgb, back_rgb)
        fallback.paste(preview, (0, part_index * tile_height))

    for face_index, face in enumerate(faces):
        part_index = int(face_parts[face_index])
        face_uv = uvs[np.asarray(face, dtype=np.int32)]
        front_tri = front_points[np.asarray(face, dtype=np.int32)].astype(np.float32)
        back_tri = back_points[np.asarray(face, dtype=np.int32)].astype(np.float32)
        face_normal = normals[np.asarray(face, dtype=np.int32)].mean(axis=0)
        front_strength = max(0.0, float(-face_normal[2]))
        back_strength = max(0.0, float(face_normal[2]))
        if not bool(front_visible[face_index]):
            front_strength = 0.0
        if not bool(back_visible[face_index]):
            back_strength = 0.0
        total_strength = front_strength + back_strength
        if total_strength < 0.15:
            front_strength = 0.5
            back_strength = 0.5
        else:
            front_strength /= total_strength
            back_strength /= total_strength
        face_front_rgb, face_back_rgb = part_source_rgb.get(part_index, (front_rgb, back_rgb))
        for dest_tri in _canonical_dest_triangles(face_uv, float(tile_width), float(tile_height), part_index):
            _paint_triangle(
                atlas_rgb,
                atlas_weight,
                (0, part_index * tile_height),
                dest_tri,
                face_front_rgb,
                front_alpha,
                front_tri,
                front_strength,
            )
            _paint_triangle(
                atlas_rgb,
                atlas_weight,
                (0, part_index * tile_height),
                dest_tri,
                face_back_rgb,
                back_alpha,
                back_tri,
                back_strength,
            )

    _finalize_atlas(atlas_rgb, atlas_weight, fallback).save(atlas_path)
    return atlas_path


def _build_part_atlas(
    cache_key: str,
    *,
    front_image: Image.Image,
    back_image: Image.Image,
    front_points: np.ndarray,
    back_points: np.ndarray,
    faces: np.ndarray,
    vertex_parts: np.ndarray,
) -> tuple[Path, dict[str, Any]]:
    texture_dir = TMP_DIR / "hmr2_textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    atlas_path = texture_dir / f"{cache_key}.png"
    face_parts = _face_parts(faces, vertex_parts)
    image_size = front_image.size
    part_tiles: dict[str, dict[str, Any]] = {}
    rows: list[list[Image.Image]] = []
    part_height = 192
    tile_width = 96
    for part_index, part_name in enumerate(PART_ORDER):
        front_mask = _mask_from_faces(front_points, faces, face_parts, part_index, image_size)
        back_mask = _mask_from_faces(back_points, faces, face_parts, part_index, image_size)
        front_crop, front_bbox = _crop_to_mask(front_image, front_mask)
        back_crop, back_bbox = _crop_to_mask(back_image, back_mask)
        front_tile = _solidify_rgba(front_crop).resize((tile_width, part_height), Image.Resampling.BILINEAR)
        back_tile = _solidify_rgba(back_crop).resize((tile_width, part_height), Image.Resampling.BILINEAR)
        side_left = _blend_rgba(front_tile, back_tile.transpose(Image.Transpose.FLIP_LEFT_RIGHT), 0.5)
        side_right = side_left.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        rows.append([front_tile, side_left, back_tile, side_right])
        part_tiles[part_name] = {
            "front_bbox": front_bbox,
            "back_bbox": back_bbox,
            "row": part_index,
        }

    atlas = Image.new("RGBA", (tile_width * 4, part_height * len(PART_ORDER)), (0, 0, 0, 255))
    for row_index, tiles in enumerate(rows):
        for col_index, tile in enumerate(tiles):
            atlas.paste(tile, (col_index * tile_width, row_index * part_height))
    atlas.save(atlas_path)
    part_tiles["tile_width"] = tile_width
    part_tiles["tile_height"] = part_height
    part_tiles["atlas_width"] = tile_width * 4
    part_tiles["atlas_height"] = part_height * len(PART_ORDER)
    return atlas_path, part_tiles


def _compute_part_baked_uvs(
    vertices: np.ndarray,
    *,
    normals: np.ndarray,
    vertex_parts: np.ndarray,
    part_meta: dict[str, Any],
    front_points: np.ndarray,
    back_points: np.ndarray,
) -> np.ndarray:
    atlas_width = float(part_meta["atlas_width"])
    atlas_height = float(part_meta["atlas_height"])
    tile_width = float(part_meta["tile_width"])
    tile_height = float(part_meta["tile_height"])
    uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    for part_index, part_name in enumerate(PART_ORDER):
        mask = vertex_parts == part_index
        if not np.any(mask):
            continue
        front_bbox = part_meta[part_name]["front_bbox"]
        back_bbox = part_meta[part_name]["back_bbox"]
        fx0, fy0, fx1, fy1 = [float(v) for v in front_bbox]
        bx0, by0, bx1, by1 = [float(v) for v in back_bbox]
        fw = max(1.0, fx1 - fx0)
        fh = max(1.0, fy1 - fy0)
        bw = max(1.0, bx1 - bx0)
        bh = max(1.0, by1 - by0)
        front_local_u = np.clip((front_points[mask, 0] - fx0) / fw, 0.0, 1.0)
        front_local_v = np.clip((front_points[mask, 1] - fy0) / fh, 0.0, 1.0)
        back_local_u = np.clip((back_points[mask, 0] - bx0) / bw, 0.0, 1.0)
        back_local_v = np.clip((back_points[mask, 1] - by0) / bh, 0.0, 1.0)
        row_top = part_index * tile_height
        part_normals = normals[mask]
        front_mask = part_normals[:, 2] < -0.32
        back_mask = part_normals[:, 2] > 0.32
        side_mask = ~(front_mask | back_mask)
        left_side = side_mask & (part_normals[:, 0] < 0.0)
        right_side = side_mask & ~left_side
        col_offset = np.zeros(np.count_nonzero(mask), dtype=np.float32)
        local_u = np.zeros_like(col_offset)
        local_v = np.zeros_like(col_offset)
        col_offset[front_mask] = 0.0
        local_u[front_mask] = front_local_u[front_mask]
        local_v[front_mask] = front_local_v[front_mask]
        col_offset[back_mask] = 2.0
        local_u[back_mask] = 1.0 - back_local_u[back_mask]
        local_v[back_mask] = back_local_v[back_mask]
        side_mix_u = np.clip(0.5 + (front_local_u - 0.5) * 0.4, 0.18, 0.82)
        side_mix_v = front_local_v * 0.5 + back_local_v * 0.5
        col_offset[left_side] = 1.0
        local_u[left_side] = side_mix_u[left_side]
        local_v[left_side] = side_mix_v[left_side]
        col_offset[right_side] = 3.0
        local_u[right_side] = 1.0 - side_mix_u[right_side]
        local_v[right_side] = side_mix_v[right_side]
        atlas_u = (col_offset * tile_width + local_u * tile_width) / atlas_width
        atlas_v = 1.0 - ((row_top + local_v * tile_height) / atlas_height)
        uvs[mask, 0] = atlas_u.astype(np.float32)
        uvs[mask, 1] = atlas_v.astype(np.float32)
    return uvs


def _to_panda_coords(vertices: np.ndarray) -> np.ndarray:
    transformed = np.empty_like(vertices)
    transformed[:, 0] = vertices[:, 0]
    transformed[:, 1] = -vertices[:, 2]
    transformed[:, 2] = vertices[:, 1]
    return transformed


def _make_geom_node(vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray, texture_path: Path) -> GeomNode:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    vformat = GeomVertexFormat.getV3n3t2()
    vdata = GeomVertexData("textured_body", vformat, Geom.UHStatic)
    vdata.setNumRows(len(vertices))
    writer_v = GeomVertexWriter(vdata, "vertex")
    writer_n = GeomVertexWriter(vdata, "normal")
    writer_t = GeomVertexWriter(vdata, "texcoord")
    for vertex, normal, uv in zip(vertices, normals, uvs):
        writer_v.addData3f(float(vertex[0]), float(vertex[1]), float(vertex[2]))
        writer_n.addData3f(float(normal[0]), float(normal[1]), float(normal[2]))
        writer_t.addData2f(float(uv[0]), float(uv[1]))
    primitive = GeomTriangles(Geom.UHStatic)
    for face in faces:
        primitive.addVertices(int(face[0]), int(face[1]), int(face[2]))
    primitive.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(primitive)
    node = GeomNode("mesh")
    node.addGeom(geom)
    return node


def _apply_pose_deltas(base_body_pose: np.ndarray, phase: float, *, style: str) -> np.ndarray:
    pose = base_body_pose.copy()
    if style == "wave":
        pose[15] = _rotation_matrix("z", 0.55 + math.sin(phase * 2.0) * 0.35) @ pose[15]
        pose[17] = _rotation_matrix("z", -0.45 - math.sin(phase * 2.0) * 0.25) @ pose[17]
        pose[16] = _rotation_matrix("z", -0.18 + math.sin(phase) * 0.08) @ pose[16]
        pose[1] = _rotation_matrix("x", 0.10 * math.sin(phase * 2.0)) @ pose[1]
        pose[4] = _rotation_matrix("x", -0.14 * math.sin(phase * 2.0)) @ pose[4]
    else:
        pose[14] = _rotation_matrix("z", -0.45 + math.sin(phase * 1.5) * 0.18) @ pose[14]
        pose[16] = _rotation_matrix("z", 0.40 - math.sin(phase * 1.5) * 0.18) @ pose[16]
        pose[17] = _rotation_matrix("z", -0.35 + math.sin(phase * 2.4) * 0.18) @ pose[17]
        pose[18] = _rotation_matrix("z", 0.28 - math.sin(phase * 2.4) * 0.15) @ pose[18]
        pose[0] = _rotation_matrix("x", -0.10 * (0.5 + 0.5 * math.sin(phase))) @ pose[0]
        pose[3] = _rotation_matrix("x", 0.18 * (0.5 + 0.5 * math.sin(phase))) @ pose[3]
        pose[4] = _rotation_matrix("x", 0.18 * (0.5 + 0.5 * math.sin(phase + 0.6))) @ pose[4]
    return pose


def _upright_global_orient(global_orient: np.ndarray) -> np.ndarray:
    # HMR2 often returns a camera-facing body with a near-180deg X flip in world space.
    # Cancel that so the Panda scene sees an upright person.
    return _rotation_matrix("x", math.pi) @ global_orient


def render_action_showcase(output_path: Path, *, width: int, height: int, fps: int, duration: float) -> None:
    ensure_runtime_dirs()
    _configure_panda3d(width, height)
    entries = _load_entries()[:2]
    if not entries:
        raise SystemExit("no hmr2 meshes found")

    smpl_model = SMPL(model_path=str(SMPL_MODEL_DIR), gender="NEUTRAL", batch_size=1)
    faces = np.asarray(smpl_model.faces, dtype=np.int32)

    base_assets = []
    for entry in entries:
        rotmats = entry["params"]["smpl_params_rotmat"]
        body_pose = np.asarray(rotmats["body_pose"], dtype=np.float32)
        raw_global_orient = np.asarray(rotmats["global_orient"], dtype=np.float32)
        global_orient = _upright_global_orient(raw_global_orient)
        betas = np.asarray(rotmats["betas"], dtype=np.float32)
        texture_path, view_images = _prepare_projected_texture(
            entry["reference_path"],
            entry["mask_path"],
            cache_key=str(entry["asset_id"]),
            back_reference_path=entry["back_reference_path"],
        )
        texture_size = view_images["front"].size
        camera_translation = np.asarray(entry["params"]["camera_translation"], dtype=np.float32)
        focal_length = float(entry["params"]["focal_length"])
        with torch.no_grad():
            uv_output = smpl_model(
                global_orient=torch.from_numpy(raw_global_orient).unsqueeze(0),
                body_pose=torch.from_numpy(body_pose).unsqueeze(0),
                betas=torch.from_numpy(betas).unsqueeze(0),
                pose2rot=False,
            )
        uv_vertices = uv_output.vertices[0].detach().cpu().numpy().astype(np.float32)
        uv_mesh = trimesh.Trimesh(vertices=uv_vertices, faces=faces, process=False)
        uv_normals = np.asarray(uv_mesh.vertex_normals, dtype=np.float32)
        vertex_parts = _vertex_parts(uv_vertices)
        uvs, _ = _compute_cylindrical_part_uvs(uv_vertices, vertex_parts)
        front_points = _project_vertices(
            uv_vertices,
            camera_translation=camera_translation,
            focal_length=focal_length,
            image_size=texture_size,
        )
        front_depths = _camera_space_depths(uv_vertices, camera_translation)
        back_points = _project_vertices(
            _rotate_y(uv_vertices, math.pi),
            camera_translation=camera_translation,
            focal_length=focal_length,
            image_size=texture_size,
        )
        back_depths = _camera_space_depths(_rotate_y(uv_vertices, math.pi), camera_translation)
        front_visible = _face_visibility(front_points, front_depths, faces, texture_size)
        back_visible = _face_visibility(back_points, back_depths, faces, texture_size)
        atlas_path = _build_triangle_baked_atlas(
            str(entry["asset_id"]),
            front_image=view_images["front"],
            back_image=view_images["back"],
            front_masked=view_images["front_masked"],
            back_masked=view_images["back_masked"],
            front_points=front_points,
            back_points=back_points,
            faces=faces,
            vertex_parts=vertex_parts,
            normals=uv_normals,
            uvs=uvs,
            front_visible=front_visible,
            back_visible=back_visible,
        )
        base_assets.append(
            {
                "entry": entry,
                "body_pose": body_pose,
                "global_orient": global_orient,
                "betas": betas,
                "uvs": uvs,
                "texture_path": atlas_path,
                "style": "wave" if "man" in entry["asset_id"] else "salute",
            }
        )

    base = ShowBase(windowType="offscreen")
    ffmpeg_proc = _open_ffmpeg_stream(fps, width, height, output_path)
    try:
        base.disableMouse()
        lens = PerspectiveLens()
        lens.setFov(28)
        base.cam.node().setLens(lens)
        base.setBackgroundColor(0.08, 0.09, 0.11, 1.0)

        ambient = base.render.attachNewNode(AmbientLight("ambient"))
        ambient.node().setColor((0.34, 0.34, 0.36, 1.0))
        base.render.setLight(ambient)
        key = base.render.attachNewNode(DirectionalLight("key"))
        key.node().setColor((1.0, 0.96, 0.92, 1.0))
        key.setHpr(-22, -28, 0)
        base.render.setLight(key)
        rim = base.render.attachNewNode(DirectionalLight("rim"))
        rim.node().setColor((0.42, 0.58, 0.82, 1.0))
        rim.setHpr(148, -18, 0)
        base.render.setLight(rim)

        textures = [base.loader.loadTexture(str(asset["texture_path"])) for asset in base_assets]
        nodes = []
        total_frames = max(1, int(round(duration * fps)))
        for frame_index in range(total_frames):
            t = frame_index / fps
            phase = t * math.tau * 0.5
            cycle = (t / max(duration, 1e-6)) * len(base_assets)
            active_index = min(len(base_assets) - 1, int(cycle))
            local_mix = cycle - active_index
            for node in nodes:
                node.removeNode()
            nodes = []
            for index, asset in enumerate(base_assets):
                if index != active_index:
                    continue
                body_pose = _apply_pose_deltas(asset["body_pose"], phase + index * 0.8, style=asset["style"])
                turn = -math.pi * 0.72 + local_mix * math.pi * 1.44
                global_orient = _rotation_matrix("y", turn) @ asset["global_orient"]
                with torch.no_grad():
                    smpl_out = smpl_model(
                        global_orient=torch.from_numpy(global_orient).unsqueeze(0),
                        body_pose=torch.from_numpy(body_pose).unsqueeze(0),
                        betas=torch.from_numpy(asset["betas"]).unsqueeze(0),
                        pose2rot=False,
                    )
                vertices = smpl_out.vertices[0].detach().cpu().numpy().astype(np.float32)
                vertices = _to_panda_coords(vertices)
                vertices -= vertices.mean(axis=0, keepdims=True)
                vertices[:, 2] -= vertices[:, 2].min()
                geom_node = _make_geom_node(vertices, faces, asset["uvs"], asset["texture_path"])
                node = base.render.attachNewNode(geom_node)
                node.setTexture(textures[index], 1)
                node.setTransparency(TransparencyAttrib.MAlpha)
                node.setScale(2.8)
                node.setPos(0.0, 8.5, -1.6)
                nodes.append(node)

            cam_orbit = -0.6 + 1.2 * local_mix
            base.camera.setPos(cam_orbit * 1.6, -18.8 + 0.5 * math.cos(local_mix * math.pi), 1.1)
            base.camera.lookAt(0.0, 8.5, 0.75)
            assert ffmpeg_proc.stdin is not None
            ffmpeg_proc.stdin.write(_capture_frame_bytes(base))
    finally:
        if ffmpeg_proc.stdin is not None:
            ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        base.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a skinned HMR2 action showcase video.")
    parser.add_argument("--output", type=Path, default=Path("outputs/people_hmr2_action_showcase.mp4"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()
    render_action_showcase(args.output, width=args.width, height=args.height, fps=args.fps, duration=args.duration)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
