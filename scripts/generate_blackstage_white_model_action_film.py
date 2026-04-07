#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    CardMaker,
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
from smplx import SMPL

from common.io import ROOT_DIR, TMP_DIR, ensure_runtime_dirs


INDEX_PATH = ROOT_DIR / "assets" / "people" / ".cache" / "white_model_index.json"
POSE_DIR = ROOT_DIR / "assets" / "actions" / ".cache" / "poses"
SMPL_MODEL_DIR = Path("/root/.cache/4DHumans/data/smpl")
BGM_PATH = ROOT_DIR / "assets" / "bgm" / "男儿当自强.mp3"


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
        "-crf", "19",
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


def _to_panda_coords(vertices: np.ndarray) -> np.ndarray:
    transformed = np.empty_like(vertices)
    transformed[:, 0] = vertices[:, 0]
    transformed[:, 1] = -vertices[:, 2]
    transformed[:, 2] = vertices[:, 1]
    return transformed


def _upright_global_orient(global_orient: np.ndarray) -> np.ndarray:
    return _rotation_matrix("x", math.pi) @ global_orient


def _make_white_geom_node(vertices: np.ndarray, faces: np.ndarray) -> GeomNode:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    vformat = GeomVertexFormat.getV3n3()
    vdata = GeomVertexData("white_body", vformat, Geom.UHStatic)
    vdata.setNumRows(len(vertices))
    writer_v = GeomVertexWriter(vdata, "vertex")
    writer_n = GeomVertexWriter(vdata, "normal")
    for vertex, normal in zip(vertices, normals):
        writer_v.addData3f(float(vertex[0]), float(vertex[1]), float(vertex[2]))
        writer_n.addData3f(float(normal[0]), float(normal[1]), float(normal[2]))
    primitive = GeomTriangles(Geom.UHStatic)
    for face in faces:
        primitive.addVertices(int(face[0]), int(face[1]), int(face[2]))
    primitive.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(primitive)
    node = GeomNode("mesh")
    node.addGeom(geom)
    return node


def _load_entries() -> list[dict[str, Any]]:
    payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    items_by_asset = {str(item.get("asset_id")): item for item in payload.get("items") or []}
    entries = []
    for mesh in payload.get("hmr2_meshes") or []:
        asset_id = str(mesh.get("asset_id"))
        item = items_by_asset.get(asset_id, {})
        params = json.loads((ROOT_DIR / str(mesh["hmr2_params_path"])).read_text(encoding="utf-8"))
        entries.append(
            {
                "asset_id": asset_id,
                "display_name": item.get("display_name", asset_id),
                "params": params,
            }
        )
    return entries


def _action_duration_seconds(name: str) -> float:
    payload = json.loads((POSE_DIR / f"{name}.pose.json").read_text(encoding="utf-8"))
    return sum(float(value) for value in payload.get("durations_ms") or []) / 1000.0


def _identity_body_pose() -> np.ndarray:
    return np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 23, axis=0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _load_pose_track(name: str) -> dict[str, Any]:
    payload = json.loads((POSE_DIR / f"{name}.pose.json").read_text(encoding="utf-8"))
    durations_ms = [float(value) for value in payload.get("durations_ms") or []]
    frames: list[dict[str, Any]] = []
    times = [0.0]
    elapsed = 0.0
    for duration_ms, frame in zip(durations_ms, payload.get("frames") or []):
        keypoints: dict[str, np.ndarray] = {}
        for point in frame.get("keypoints") or []:
            score = float(point.get("score") or 0.0)
            if score <= 0.05:
                continue
            keypoints[str(point["name"])] = np.array(
                [float(point["x"]), float(point["y"]), score],
                dtype=np.float32,
            )
        frames.append({"keypoints": keypoints, "angles_deg": dict(frame.get("angles_deg") or {})})
        elapsed += duration_ms / 1000.0
        times.append(elapsed)
    return {"duration": elapsed, "frames": frames, "times": times}


def _interpolate_point(a: np.ndarray | None, b: np.ndarray | None, t: float) -> np.ndarray | None:
    if a is None and b is None:
        return None
    if a is None:
        return b.copy()
    if b is None:
        return a.copy()
    mixed = a * (1.0 - t) + b * t
    mixed[2] = max(float(a[2]), float(b[2]))
    return mixed.astype(np.float32)


def _sample_pose_track(track: dict[str, Any], t: float) -> dict[str, np.ndarray]:
    frames: list[dict[str, Any]] = track["frames"]
    if not frames:
        return {}
    if len(frames) == 1 or t <= 0.0:
        return {name: value.copy() for name, value in frames[0]["keypoints"].items()}
    duration = float(track["duration"] or 0.0)
    if duration <= 1e-6 or t >= duration:
        return {name: value.copy() for name, value in frames[-1]["keypoints"].items()}

    times: list[float] = track["times"]
    for index in range(len(frames) - 1):
        start = times[index]
        end = times[index + 1]
        if t <= end:
            alpha = 0.0 if end <= start else (t - start) / (end - start)
            current = frames[index]["keypoints"]
            nxt = frames[index + 1]["keypoints"]
            names = set(current) | set(nxt)
            return {
                name: _interpolate_point(current.get(name), nxt.get(name), alpha)
                for name in names
                if _interpolate_point(current.get(name), nxt.get(name), alpha) is not None
            }
    return {name: value.copy() for name, value in frames[-1]["keypoints"].items()}


def _pose_point(points: dict[str, np.ndarray], name: str) -> np.ndarray | None:
    point = points.get(name)
    if point is None or float(point[2]) <= 0.05:
        return None
    return point


def _midpoint(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is None and b is None:
        return None
    if a is None:
        return b.copy()
    if b is None:
        return a.copy()
    mixed = (a + b) * 0.5
    mixed[2] = max(float(a[2]), float(b[2]))
    return mixed.astype(np.float32)


def _vector_angle_own_side(anchor: np.ndarray | None, child: np.ndarray | None, *, own_side: str) -> float | None:
    if anchor is None or child is None:
        return None
    dx = float(child[0] - anchor[0])
    dy = float(child[1] - anchor[1])
    if own_side == "left":
        dx = -dx
    if own_side == "right":
        dx = dx
    return math.atan2(dx, max(6.0, dy))


def _limb_bend(points: dict[str, np.ndarray], a: str, b: str, c: str) -> float | None:
    pa = _pose_point(points, a)
    pb = _pose_point(points, b)
    pc = _pose_point(points, c)
    if pa is None or pb is None or pc is None:
        return None
    ba = pa[:2] - pb[:2]
    bc = pc[:2] - pb[:2]
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 1e-6:
        return None
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.acos(cosine)


def _retarget_pose_from_track(
    points: dict[str, np.ndarray],
    neutral_body_pose: np.ndarray,
    base_global_orient: np.ndarray,
    *,
    action_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    pose = neutral_body_pose.copy()
    global_orient = base_global_orient.copy()

    left_shoulder = _pose_point(points, "left_shoulder")
    right_shoulder = _pose_point(points, "right_shoulder")
    left_elbow = _pose_point(points, "left_elbow")
    right_elbow = _pose_point(points, "right_elbow")
    left_wrist = _pose_point(points, "left_wrist")
    right_wrist = _pose_point(points, "right_wrist")
    left_hip = _pose_point(points, "left_hip")
    right_hip = _pose_point(points, "right_hip")
    left_knee = _pose_point(points, "left_knee")
    right_knee = _pose_point(points, "right_knee")
    left_ankle = _pose_point(points, "left_ankle")
    right_ankle = _pose_point(points, "right_ankle")

    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)
    if shoulder_mid is not None and hip_mid is not None:
        dx = float(shoulder_mid[0] - hip_mid[0])
        dy = float(hip_mid[1] - shoulder_mid[1])
        torso_lean = math.atan2(dx, max(10.0, dy))
        global_orient = _rotation_matrix("z", -0.55 * torso_lean) @ global_orient
        pose[2] = _rotation_matrix("z", 0.26 * torso_lean) @ pose[2]
        pose[5] = _rotation_matrix("z", 0.18 * torso_lean) @ pose[5]
        if action_name == "翻跟头gif":
            torso_pitch = _clamp((220.0 - dy) / 210.0, -0.2, 1.15)
            global_orient = _rotation_matrix("x", 0.65 * torso_pitch) @ global_orient

    left_shoulder_splay = _vector_angle_own_side(left_shoulder, left_elbow, own_side="left")
    right_shoulder_splay = _vector_angle_own_side(right_shoulder, right_elbow, own_side="right")
    if left_shoulder_splay is not None:
        left_shoulder_splay = _clamp(left_shoulder_splay, -1.4, 1.55)
        pose[12] = _rotation_matrix("z", 0.28 * left_shoulder_splay) @ pose[12]
        pose[15] = _rotation_matrix("z", 0.92 * left_shoulder_splay) @ pose[15]
    if right_shoulder_splay is not None:
        right_shoulder_splay = _clamp(right_shoulder_splay, -1.55, 1.4)
        pose[13] = _rotation_matrix("z", -0.28 * right_shoulder_splay) @ pose[13]
        pose[16] = _rotation_matrix("z", -0.92 * right_shoulder_splay) @ pose[16]

    left_elbow_bend = _limb_bend(points, "left_shoulder", "left_elbow", "left_wrist")
    right_elbow_bend = _limb_bend(points, "right_shoulder", "right_elbow", "right_wrist")
    if left_elbow_bend is not None:
        left_elbow_flex = _clamp(math.pi - left_elbow_bend, 0.0, 1.65)
        pose[17] = _rotation_matrix("z", -1.05 * left_elbow_flex) @ pose[17]
    if right_elbow_bend is not None:
        right_elbow_flex = _clamp(math.pi - right_elbow_bend, 0.0, 1.65)
        pose[18] = _rotation_matrix("z", 1.05 * right_elbow_flex) @ pose[18]

    left_hip_splay = _vector_angle_own_side(left_hip, left_knee, own_side="left")
    right_hip_splay = _vector_angle_own_side(right_hip, right_knee, own_side="right")
    if left_hip_splay is not None:
        left_hip_splay = _clamp(left_hip_splay, -0.8, 0.9)
        pose[0] = _rotation_matrix("z", 0.55 * left_hip_splay) @ pose[0]
    if right_hip_splay is not None:
        right_hip_splay = _clamp(right_hip_splay, -0.9, 0.8)
        pose[1] = _rotation_matrix("z", -0.55 * right_hip_splay) @ pose[1]

    left_knee_bend = _limb_bend(points, "left_hip", "left_knee", "left_ankle")
    right_knee_bend = _limb_bend(points, "right_hip", "right_knee", "right_ankle")
    if left_knee_bend is not None:
        left_knee_flex = _clamp(math.pi - left_knee_bend, 0.0, 1.7)
        pose[3] = _rotation_matrix("x", 1.10 * left_knee_flex) @ pose[3]
        pose[0] = _rotation_matrix("x", 0.38 * left_knee_flex) @ pose[0]
    if right_knee_bend is not None:
        right_knee_flex = _clamp(math.pi - right_knee_bend, 0.0, 1.7)
        pose[4] = _rotation_matrix("x", 1.10 * right_knee_flex) @ pose[4]
        pose[1] = _rotation_matrix("x", 0.38 * right_knee_flex) @ pose[1]

    if action_name == "舞剑" and right_wrist is not None and right_elbow is not None and right_shoulder is not None:
        wrist_drive = _vector_angle_own_side(right_elbow, right_wrist, own_side="right")
        if wrist_drive is not None:
            pose[20] = _rotation_matrix("z", -0.55 * _clamp(wrist_drive, -1.2, 1.2)) @ pose[20]
        pose[14] = _rotation_matrix("z", -0.15) @ pose[14]
    if action_name == "太极" and left_wrist is not None and right_wrist is not None:
        spread = _clamp((float(right_wrist[0]) - float(left_wrist[0])) / 120.0, -0.8, 0.8)
        pose[14] = _rotation_matrix("y", 0.12 * spread) @ pose[14]
    if action_name == "翻跟头gif":
        if left_shoulder is not None and left_hip is not None:
            tuck_left = _clamp((float(left_hip[1]) - float(left_knee[1])) / 90.0, -0.6, 0.2) if left_knee is not None else 0.0
            pose[15] = _rotation_matrix("x", -0.25 + 0.35 * tuck_left) @ pose[15]
        if right_shoulder is not None and right_hip is not None:
            tuck_right = _clamp((float(right_hip[1]) - float(right_knee[1])) / 90.0, -0.6, 0.2) if right_knee is not None else 0.0
            pose[16] = _rotation_matrix("x", -0.25 + 0.35 * tuck_right) @ pose[16]

    return pose, global_orient


def _blend(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return tuple((1.0 - t) * x + t * y for x, y in zip(a, b))


def _mix_bgm(video_path: Path, final_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not BGM_PATH.exists():
        if video_path != final_path:
            shutil.copy2(video_path, final_path)
        return
    command = [
        ffmpeg,
        "-y",
        "-stream_loop", "-1",
        "-i", str(BGM_PATH),
        "-i", str(video_path),
        "-shortest",
        "-filter:a", "volume=0.18",
        "-c:v", "copy",
        "-c:a", "aac",
        str(final_path),
    ]
    subprocess.run(command, check=True)


def render_film(output_path: Path, *, width: int, height: int, fps: int) -> None:
    ensure_runtime_dirs()
    _configure_panda3d(width, height)
    entries = _load_entries()
    if len(entries) < 2:
        raise SystemExit("need at least two hmr2 human entries")

    pose_tracks = {
        "taiji": _load_pose_track("太极"),
        "sword": _load_pose_track("舞剑"),
        "flip": _load_pose_track("翻跟头gif"),
    }
    durations = {
        "taiji": float(pose_tracks["taiji"]["duration"]),
        "sword": float(pose_tracks["sword"]["duration"]),
        "flip": float(pose_tracks["flip"]["duration"]),
    }
    segments = [
        ("taiji", durations["taiji"]),
        ("sword", durations["sword"]),
        ("flip", durations["flip"]),
        ("finale", 2.8),
    ]
    total_duration = sum(duration for _, duration in segments)

    smpl_model = SMPL(model_path=str(SMPL_MODEL_DIR), gender="NEUTRAL", batch_size=1)
    faces = np.asarray(smpl_model.faces, dtype=np.int32)

    assets = []
    for entry in entries[:2]:
        rotmats = entry["params"]["smpl_params_rotmat"]
        assets.append(
            {
                "entry": entry,
                "body_pose": _identity_body_pose(),
                "global_orient": _upright_global_orient(np.asarray(rotmats["global_orient"], dtype=np.float32)),
                "betas": np.asarray(rotmats["betas"], dtype=np.float32),
            }
        )

    base = ShowBase(windowType="offscreen")
    silent_output = TMP_DIR / "people_white_action_film_silent.mp4"
    ffmpeg_proc = _open_ffmpeg_stream(fps, width, height, silent_output)
    try:
        base.disableMouse()
        lens = PerspectiveLens()
        lens.setFov(32)
        base.cam.node().setLens(lens)
        base.setBackgroundColor(0.0, 0.0, 0.0, 1.0)

        ambient = base.render.attachNewNode(AmbientLight("ambient"))
        ambient.node().setColor((0.16, 0.16, 0.17, 1.0))
        base.render.setLight(ambient)
        key = base.render.attachNewNode(DirectionalLight("key"))
        key.node().setColor((0.95, 0.95, 0.98, 1.0))
        key.setHpr(-15, -34, 0)
        base.render.setLight(key)
        rim = base.render.attachNewNode(DirectionalLight("rim"))
        rim.node().setColor((0.50, 0.58, 0.72, 1.0))
        rim.setHpr(150, -10, 0)
        base.render.setLight(rim)

        floor = CardMaker("floor")
        floor.setFrame(-6.2, 6.2, -3.8, 3.8)
        floor_node = base.render.attachNewNode(floor.generate())
        floor_node.setP(-90)
        floor_node.setZ(-1.65)
        floor_node.setColor(0.05, 0.05, 0.06, 1.0)

        pool = CardMaker("pool")
        pool.setFrame(-2.6, 2.6, -1.6, 1.6)
        pool_node = base.render.attachNewNode(pool.generate())
        pool_node.setP(-90)
        pool_node.setZ(-1.62)
        pool_node.setColor(0.18, 0.18, 0.20, 0.26)
        pool_node.setTransparency(TransparencyAttrib.MAlpha)

        total_frames = max(1, int(round(total_duration * fps)))
        nodes = []
        for frame_index in range(total_frames):
            t = frame_index / fps
            cursor = 0.0
            scene_name = "finale"
            local_t = 0.0
            scene_duration = 1.0
            for name, duration in segments:
                if t < cursor + duration:
                    scene_name = name
                    local_t = t - cursor
                    scene_duration = duration
                    break
                cursor += duration

            local_phase = (local_t / max(scene_duration, 1e-6)) * math.tau
            for node in nodes:
                node.removeNode()
            nodes = []

            performer_count = 1 if scene_name in {"taiji", "sword", "flip"} else 2
            for index, asset in enumerate(assets[:performer_count]):
                style = scene_name
                if scene_name == "sword" and index == 0:
                    style = "sword"
                    asset = assets[1]
                elif scene_name == "flip" and index == 0:
                    style = "flip"
                    asset = assets[0]
                elif scene_name == "taiji":
                    asset = assets[0]
                elif scene_name == "finale":
                    style = "finale"

                global_orient = asset["global_orient"].copy()
                if style in {"taiji", "sword", "flip"}:
                    action_map = {"taiji": "太极", "sword": "舞剑", "flip": "翻跟头gif"}
                    track = pose_tracks[style]
                    sampled_points = _sample_pose_track(track, local_t)
                    body_pose, global_orient = _retarget_pose_from_track(
                        sampled_points,
                        asset["body_pose"],
                        global_orient,
                        action_name=action_map[style],
                    )
                    if style == "sword":
                        global_orient = _rotation_matrix("y", 0.08 * math.sin(local_phase)) @ global_orient
                    elif style == "taiji":
                        global_orient = _rotation_matrix("y", -0.06 * math.sin(local_phase * 0.5)) @ global_orient
                else:
                    body_pose = asset["body_pose"].copy()
                    body_pose[15] = _rotation_matrix("z", 0.18 + 0.08 * math.sin(local_phase + index * 0.45)) @ body_pose[15]
                    body_pose[16] = _rotation_matrix("z", -0.18 - 0.08 * math.sin(local_phase + index * 0.45)) @ body_pose[16]
                    body_pose[3] = _rotation_matrix("x", 0.12) @ body_pose[3]
                    body_pose[4] = _rotation_matrix("x", 0.12) @ body_pose[4]
                    global_orient = _rotation_matrix("y", (-0.22 if index == 0 else 0.22)) @ global_orient

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
                geom_node = _make_white_geom_node(vertices, faces)
                node = base.render.attachNewNode(geom_node)
                node.setScale(2.6)
                if scene_name == "taiji":
                    node.setPos(0.0, 9.0, -1.65)
                elif scene_name == "sword":
                    node.setPos(0.4, 8.7, -1.65)
                elif scene_name == "flip":
                    travel = -3.5 + 7.0 * min(1.0, local_t / max(scene_duration, 1e-6))
                    jump = 1.1 * max(0.0, math.sin(local_phase))
                    node.setPos(travel, 8.8, -1.65 + jump)
                else:
                    node.setPos(-1.8 + index * 3.6, 8.9, -1.65)
                node.setColor(*( (0.97, 0.97, 0.97, 1.0) if index == 0 else (0.88, 0.90, 0.94, 1.0) ))
                nodes.append(node)

            if scene_name == "taiji":
                camera_pos = _blend((-2.4, -18.0, 1.4), (0.0, -14.0, 1.8), min(1.0, local_t / scene_duration))
                look_at = (0.0, 9.0, 0.65)
                pool_node.setScale(1.1)
            elif scene_name == "sword":
                orbit = local_t / scene_duration
                camera_pos = (
                    math.sin(orbit * math.pi) * 2.6,
                    -15.5 + math.cos(orbit * math.pi) * 1.2,
                    1.25 + 0.3 * math.sin(orbit * math.pi),
                )
                look_at = (0.2, 8.7, 0.7)
                pool_node.setScale(1.0)
            elif scene_name == "flip":
                follow = -2.5 + 5.0 * min(1.0, local_t / scene_duration)
                camera_pos = (follow, -13.0, 1.7)
                look_at = (follow + 0.5, 8.8, 1.0)
                pool_node.setScale(1.25)
            else:
                orbit = local_t / scene_duration
                camera_pos = (
                    math.sin(orbit * math.pi * 0.8) * 1.4,
                    -18.2 + 2.2 * orbit,
                    1.5,
                )
                look_at = (0.0, 8.9, 0.72)
                pool_node.setScale(1.35)

            base.camera.setPos(*camera_pos)
            base.camera.lookAt(*look_at)

            assert ffmpeg_proc.stdin is not None
            ffmpeg_proc.stdin.write(_capture_frame_bytes(base))
    finally:
        if ffmpeg_proc.stdin is not None:
            ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        base.destroy()

    _mix_bgm(silent_output, output_path)
    print(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a black-stage white-model action short film.")
    parser.add_argument("--output", type=Path, default=Path("outputs/blackstage_white_model_action_film.mp4"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    render_film(args.output, width=args.width, height=args.height, fps=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
