#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from common.io import CHARACTERS_DIR, ROOT_DIR, write_json


INDEX_PATH = ROOT_DIR / "assets" / "people" / ".cache" / "white_model_index.json"


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-8:
        return v.copy()
    return v / norm


def _append_faces(faces: list[tuple[int, int, int]], offset: int, tris: list[tuple[int, int, int]]) -> None:
    for a, b, c in tris:
        faces.append((a + offset, b + offset, c + offset))


def _ellipsoid(
    center: np.ndarray,
    rx: float,
    ry: float,
    rz: float,
    *,
    lat_steps: int = 10,
    lon_steps: int = 16,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for lat_index in range(lat_steps + 1):
        theta = math.pi * lat_index / lat_steps
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for lon_index in range(lon_steps):
            phi = math.tau * lon_index / lon_steps
            x = center[0] + rx * sin_theta * math.cos(phi)
            y = center[1] + ry * sin_theta * math.sin(phi)
            z = center[2] + rz * cos_theta
            vertices.append((float(x), float(y), float(z)))
    for lat_index in range(lat_steps):
        row0 = lat_index * lon_steps
        row1 = (lat_index + 1) * lon_steps
        for lon_index in range(lon_steps):
            next_lon = (lon_index + 1) % lon_steps
            a = row0 + lon_index
            b = row0 + next_lon
            c = row1 + lon_index
            d = row1 + next_lon
            if lat_index != 0:
                faces.append((a, c, b))
            if lat_index != lat_steps - 1:
                faces.append((b, c, d))
    return vertices, faces


def _oriented_cylinder(start: np.ndarray, end: np.ndarray, radius_x: float, radius_y: float, *, sides: int = 12) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length <= 1e-6:
        return _ellipsoid(start, radius_x, radius_y, max(radius_x, radius_y), lat_steps=6, lon_steps=12)
    forward = _normalize(axis)
    reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    side = _normalize(np.cross(forward, reference))
    up = _normalize(np.cross(side, forward))
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def ring(center: np.ndarray) -> list[int]:
        indices: list[int] = []
        for index in range(sides):
            angle = math.tau * index / sides
            point = center + side * (math.cos(angle) * radius_x) + up * (math.sin(angle) * radius_y)
            indices.append(len(vertices))
            vertices.append((float(point[0]), float(point[1]), float(point[2])))
        return indices

    ring0 = ring(start)
    ring1 = ring(end)
    for index in range(sides):
        nxt = (index + 1) % sides
        a, b = ring0[index], ring0[nxt]
        c, d = ring1[index], ring1[nxt]
        faces.append((a, c, b))
        faces.append((b, c, d))
    cap0_center = len(vertices)
    vertices.append((float(start[0]), float(start[1]), float(start[2])))
    cap1_center = len(vertices)
    vertices.append((float(end[0]), float(end[1]), float(end[2])))
    for index in range(sides):
        nxt = (index + 1) % sides
        faces.append((cap0_center, ring0[nxt], ring0[index]))
        faces.append((cap1_center, ring1[index], ring1[nxt]))
    return vertices, faces


def _oriented_box(center: np.ndarray, axis_forward: np.ndarray, axis_up: np.ndarray, axis_side: np.ndarray, half_forward: float, half_up: float, half_side: float) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    corners = []
    for sf in (-1.0, 1.0):
        for su in (-1.0, 1.0):
            for ss in (-1.0, 1.0):
                point = center + axis_forward * (sf * half_forward) + axis_up * (su * half_up) + axis_side * (ss * half_side)
                corners.append((float(point[0]), float(point[1]), float(point[2])))
    faces = [
        (0, 4, 6), (0, 6, 2),
        (1, 3, 7), (1, 7, 5),
        (0, 1, 5), (0, 5, 4),
        (2, 6, 7), (2, 7, 3),
        (0, 2, 3), (0, 3, 1),
        (4, 5, 7), (4, 7, 6),
    ]
    return corners, faces


def _append_mesh(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    mesh_vertices: list[tuple[float, float, float]],
    mesh_faces: list[tuple[int, int, int]],
) -> None:
    offset = len(vertices)
    vertices.extend(mesh_vertices)
    _append_faces(faces, offset, mesh_faces)


def _load_pose_track(path: Path) -> dict[str, tuple[float, float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = (payload.get("frames") or [{}])[0]
    points = {}
    for item in frame.get("keypoints") or []:
        points[str(item.get("name"))] = (
            float(item.get("x", 0.0) or 0.0),
            float(item.get("y", 0.0) or 0.0),
            float(item.get("score", 0.0) or 0.0),
        )
    return points


def _normalize_pose(points: dict[str, tuple[float, float, float]]) -> dict[str, np.ndarray]:
    left_hip = np.array(points["left_hip"][:2], dtype=np.float32)
    right_hip = np.array(points["right_hip"][:2], dtype=np.float32)
    left_shoulder = np.array(points["left_shoulder"][:2], dtype=np.float32)
    right_shoulder = np.array(points["right_shoulder"][:2], dtype=np.float32)
    pelvis = (left_hip + right_hip) * 0.5
    chest = (left_shoulder + right_shoulder) * 0.5
    torso_length = max(12.0, float(np.linalg.norm(chest - pelvis)))
    scale = 0.64 / torso_length

    def project(name: str) -> np.ndarray:
        x, y, _ = points[name]
        return np.array([(x - float(pelvis[0])) * scale, 0.0, 0.70 + (float(pelvis[1]) - y) * scale], dtype=np.float32)

    names = [
        "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    projected = {name: project(name) for name in names}
    projected["pelvis"] = (projected["left_hip"] + projected["right_hip"]) * 0.5
    projected["chest"] = (projected["left_shoulder"] + projected["right_shoulder"]) * 0.5
    projected["neck"] = projected["chest"] * 0.6 + projected["nose"] * 0.4
    projected["head"] = projected["nose"] + np.array([0.0, 0.0, 0.06], dtype=np.float32)
    return projected


def _segment_box(start: np.ndarray, end: np.ndarray, width: float, depth: float) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    axis_forward = _normalize(end - start)
    axis_side = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    axis_up = _normalize(np.cross(axis_side, axis_forward))
    if float(np.linalg.norm(axis_up)) <= 1e-6:
        axis_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    center = (start + end) * 0.5
    half_forward = float(np.linalg.norm(end - start)) * 0.5
    return _oriented_box(center, axis_forward, axis_up, axis_side, half_forward, width * 0.5, depth * 0.5)


def _smooth_rows(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    if not rows:
        return []
    smoothed: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        width_total = 0.0
        center_total = 0.0
        weight_total = 0.0
        for neighbor in range(max(0, index - 2), min(len(rows), index + 3)):
            weight = 1.0 / (1.0 + abs(neighbor - index))
            width_total += float(rows[neighbor].get("width_ratio", 0.0)) * weight
            center_total += float(rows[neighbor].get("center_offset_ratio", 0.0)) * weight
            weight_total += weight
        smoothed.append(
            {
                "t": float(row.get("t", 0.0)),
                "width_ratio": width_total / max(weight_total, 1e-6),
                "center_offset_ratio": center_total / max(weight_total, 1e-6),
                "segment_count": float(row.get("segment_count", 1.0)),
            }
        )
    return smoothed


def _depth_scale_for_t(t: float) -> float:
    if t <= 0.14:
        return 0.76
    if t <= 0.34:
        return 0.68
    if t <= 0.62:
        return 0.58
    if t <= 0.86:
        return 0.42
    return 0.34


def _append_profile_shell(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    rows: list[dict[str, float]],
    *,
    world_height: float,
    bottom_z: float,
    ring_sides: int = 24,
) -> None:
    if not rows:
        return
    rings: list[list[int]] = []
    for row in rows:
        t = float(row.get("t", 0.0))
        width = max(0.018, float(row.get("width_ratio", 0.0)) * world_height)
        center_offset = float(row.get("center_offset_ratio", 0.0)) * world_height
        z = bottom_z + (1.0 - t) * world_height
        rx = width * 0.5
        ry = max(width * _depth_scale_for_t(t) * 0.5, 0.012)
        ring: list[int] = []
        for side_index in range(ring_sides):
            angle = math.tau * side_index / ring_sides
            x = center_offset + math.cos(angle) * rx
            y = math.sin(angle) * ry
            ring.append(len(vertices))
            vertices.append((float(x), float(y), float(z)))
        rings.append(ring)
    if len(rings) < 2:
        return
    for ring_index in range(len(rings) - 1):
        current = rings[ring_index]
        nxt = rings[ring_index + 1]
        for side_index in range(ring_sides):
            next_index = (side_index + 1) % ring_sides
            a = current[side_index]
            b = current[next_index]
            c = nxt[side_index]
            d = nxt[next_index]
            faces.append((a, c, b))
            faces.append((b, c, d))
    top_center_x = float(rows[0].get("center_offset_ratio", 0.0)) * world_height
    top_center = len(vertices)
    vertices.append((top_center_x, 0.0, bottom_z + world_height + 0.01))
    for side_index in range(ring_sides):
        next_index = (side_index + 1) % ring_sides
        faces.append((top_center, rings[0][side_index], rings[0][next_index]))
    bottom_center_x = float(rows[-1].get("center_offset_ratio", 0.0)) * world_height
    bottom_center = len(vertices)
    vertices.append((bottom_center_x, 0.0, bottom_z - 0.01))
    for side_index in range(ring_sides):
        next_index = (side_index + 1) % ring_sides
        faces.append((bottom_center, rings[-1][next_index], rings[-1][side_index]))


def _build_mesh(profile: dict[str, Any], pose: dict[str, np.ndarray]) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], dict[str, Any]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    silhouette_rows = _smooth_rows(list(profile.get("silhouette_rows") or []))
    ankle_floor = min(float(pose["left_ankle"][2]), float(pose["right_ankle"][2]))
    head_top = float(pose["head"][2]) + float(profile.get("head_scale", 0.20)) * 0.90
    world_height = max(1.6, head_top - ankle_floor)
    bottom_z = ankle_floor - float(profile.get("foot_height", 0.04)) * 0.75

    if silhouette_rows:
        _append_profile_shell(vertices, faces, silhouette_rows, world_height=world_height, bottom_z=bottom_z)
    else:
        torso_width = float(profile.get("torso_width", 0.32))
        torso_height_scale = float(profile.get("torso_height_scale", 1.18))
        torso_center = (pose["pelvis"] + pose["chest"]) * 0.5 + np.array([0.0, 0.0, 0.02], dtype=np.float32)
        torso_height = max(0.92, float(np.linalg.norm(pose["chest"] - pose["pelvis"])) * torso_height_scale)
        torso_depth = torso_width * 0.60
        _append_mesh(vertices, faces, *_ellipsoid(torso_center, torso_width * 0.5, torso_depth * 0.5, torso_height * 0.5))

    rig = {
        "joints": {name: [round(float(value[0]), 6), round(float(value[1]), 6), round(float(value[2]), 6)] for name, value in pose.items()},
        "bones": [
            ["pelvis", "left_hip"], ["pelvis", "right_hip"], ["pelvis", "chest"], ["chest", "neck"], ["neck", "head"],
            ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"], ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
            ["left_hip", "left_knee"], ["left_knee", "left_ankle"], ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
        ],
    }
    return vertices, faces, rig


def _write_obj(path: Path, vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> None:
    lines = ["o human_proxy"]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _process_item(item: dict[str, Any]) -> dict[str, Any]:
    character_dir = ROOT_DIR / str(item["character_dir"])
    character = json.loads((character_dir / "character.json").read_text(encoding="utf-8"))
    pose_track_path = ROOT_DIR / str(item["pose_track_path"])
    pose = _normalize_pose(_load_pose_track(pose_track_path))
    vertices, faces, rig = _build_mesh(character.get("white_model_profile") or {}, pose)
    mesh_dir = character_dir / "mesh"
    mesh_path = mesh_dir / "body.obj"
    rig_path = mesh_dir / "rig.json"
    _write_obj(mesh_path, vertices, faces)
    write_json(rig_path, rig)
    character["mesh_asset_path"] = str(mesh_path.relative_to(ROOT_DIR)).replace("\\", "/")
    character["mesh_rig_path"] = str(rig_path.relative_to(ROOT_DIR)).replace("\\", "/")
    write_json(character_dir / "character.json", character)
    return {
        "asset_id": item["asset_id"],
        "mesh_asset_path": character["mesh_asset_path"],
        "mesh_rig_path": character["mesh_rig_path"],
        "vertex_count": len(vertices),
        "face_count": len(faces),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate actual OBJ mesh proxies from people pose/profile assets.")
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    args = parser.parse_args()
    payload = json.loads(args.index.read_text(encoding="utf-8"))
    items = payload.get("items") or []
    results = []
    for item in items:
        result = _process_item(item)
        results.append(result)
        print(result["mesh_asset_path"])
    write_json(args.index, {"items": payload.get("items") or [], "meshes": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
