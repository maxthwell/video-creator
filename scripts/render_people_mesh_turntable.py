#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from common.io import ROOT_DIR, TMP_DIR, ensure_runtime_dirs


INDEX_PATH = ROOT_DIR / "assets" / "people" / ".cache" / "white_model_index.json"


def _configure_panda3d(width: int, height: int) -> None:
    from panda3d.core import loadPrcFileData

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
        "-crf", "23",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def _capture_frame_bytes(base) -> bytes:
    from panda3d.core import GraphicsOutput, Texture

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
                return b"".join(frame_bytes[row_start : row_start + row_stride] for row_start in range(len(frame_bytes) - row_stride, -1, -row_stride))
    raise RuntimeError("failed to capture frame")


def _load_mesh_entries() -> list[dict]:
    payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    entries = payload.get("hmr2_meshes") or payload.get("meshes") or []
    if not entries:
        raise SystemExit("run scripts/generate_people_mesh_models.py first")
    items_by_asset_id = {str(item.get("asset_id")): item for item in payload.get("items") or []}
    merged = []
    for entry in entries:
        asset_id = str(entry.get("asset_id"))
        item = items_by_asset_id.get(asset_id, {})
        merged.append(
            {
                **entry,
                "source_path": item.get("source_path"),
                "display_name": item.get("display_name", asset_id),
            }
        )
    return merged


def _prepare_reference_panel(path: Path, *, panel_width: int, panel_height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((panel_width - 24, panel_height - 24))
    panel = Image.new("RGB", (panel_width, panel_height), (244, 245, 248))
    offset_x = (panel_width - image.width) // 2
    offset_y = (panel_height - image.height) // 2
    panel.paste(image, (offset_x, offset_y))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, panel_width - 1, panel_height - 1), radius=18, outline=(180, 184, 194), width=3)
    return panel


def _composite_reference_panels(
    frame_bytes: bytes,
    *,
    width: int,
    height: int,
    reference_panels: list[tuple[Image.Image, tuple[int, int]]],
) -> bytes:
    frame = Image.frombytes("RGB", (width, height), frame_bytes)
    for panel, (x, y) in reference_panels:
        frame.paste(panel, (x, y))
    return frame.tobytes()


def render_turntable(output_path: Path, *, width: int, height: int, fps: int, duration: float) -> None:
    ensure_runtime_dirs()
    _configure_panda3d(width, height)
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import AmbientLight, DirectionalLight, OrthographicLens

    entries = _load_mesh_entries()
    base = ShowBase(windowType="offscreen")
    ffmpeg_proc = _open_ffmpeg_stream(fps, width, height, output_path)
    try:
        base.disableMouse()
        lens = OrthographicLens()
        lens.setFilmSize(10.0, 10.0 * height / width)
        base.cam.node().setLens(lens)
        base.setBackgroundColor(0.82, 0.84, 0.88, 1.0)

        ambient = base.render.attachNewNode(AmbientLight("ambient"))
        ambient.node().setColor((0.72, 0.72, 0.75, 1.0))
        base.render.setLight(ambient)
        sun = base.render.attachNewNode(DirectionalLight("sun"))
        sun.node().setColor((1.0, 0.99, 0.96, 1.0))
        sun.setHpr(-18, -28, 0)
        base.render.setLight(sun)

        models = []
        reference_panels: list[tuple[Image.Image, tuple[int, int]]] = []
        for index, entry in enumerate(entries[:2]):
            mesh_path = entry.get("hmr2_mesh_asset_path") or entry.get("mesh_asset_path")
            if not mesh_path:
                continue
            model = base.loader.loadModel(str((ROOT_DIR / str(mesh_path)).resolve()))
            model.reparentTo(base.render)
            model.setColor(0.95, 0.95, 0.945, 1.0)
            minimum, maximum = model.getTightBounds()
            height_span = max(0.001, float(maximum.z - minimum.z))
            width_span = max(0.001, float(maximum.x - minimum.x))
            target_height = 3.65
            target_width = 2.10
            scale = min(target_height / height_span, target_width / width_span)
            ground_z = -1.55
            panel_x = -1.9 + index * 3.8
            center_x = float((minimum.x + maximum.x) * 0.5) * scale
            base_z = float(minimum.z) * scale
            model.setScale(scale)
            model.setPos(panel_x - center_x, 8.5, ground_z - base_z)
            model.setH(0.0)
            models.append(model)
            source_path = entry.get("source_path")
            if source_path:
                reference_panel = _prepare_reference_panel((ROOT_DIR / str(source_path)).resolve(), panel_width=170, panel_height=250)
                panel_x_px = 24 if index == 0 else width - 194
                reference_panels.append((reference_panel, (panel_x_px, 18)))

        total_frames = max(1, int(round(duration * fps)))
        for frame_index in range(total_frames):
            t = frame_index / fps
            spin = t * 20.0
            sway = math.sin(t * 0.7) * 0.10
            for index, model in enumerate(models):
                model.setH(spin + index * 26.0)
                model.setP(math.sin(t * 0.9 + index) * 1.4)
                model.setR(math.cos(t * 0.8 + index) * 1.1)
            base.camera.setPos(sway, -22.0, 0.8)
            base.camera.lookAt(sway, 8.5, 0.35)
            assert ffmpeg_proc.stdin is not None
            frame_bytes = _capture_frame_bytes(base)
            if reference_panels:
                frame_bytes = _composite_reference_panels(
                    frame_bytes,
                    width=width,
                    height=height,
                    reference_panels=reference_panels,
                )
            ffmpeg_proc.stdin.write(frame_bytes)
    finally:
        if ffmpeg_proc.stdin is not None:
            ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        base.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a Panda3D turntable preview for generated people meshes.")
    parser.add_argument("--output", type=Path, default=Path("outputs/people_mesh_turntable.mp4"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()
    render_turntable(args.output, width=args.width, height=args.height, fps=args.fps, duration=args.duration)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
