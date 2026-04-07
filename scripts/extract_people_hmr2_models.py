#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from common.io import ROOT_DIR, write_json


INDEX_PATH = ROOT_DIR / "assets" / "people" / ".cache" / "white_model_index.json"


def _load_people_items(index_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return list(payload.get("items") or [])


def _load_hmr2(checkpoint: str | None = None):
    from hmr2.configs import CACHE_DIR_4DHUMANS, get_config
    from hmr2.models import DEFAULT_CHECKPOINT, HMR2, check_smpl_exists

    checkpoint_path = checkpoint or DEFAULT_CHECKPOINT
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"HMR2 checkpoint not found at {checkpoint_path}. "
            f"Download it first or pass --checkpoint."
        )
    model_cfg_path = str(Path(checkpoint_path).parent.parent / "model_config.yaml")
    model_cfg = get_config(model_cfg_path, update_cachedir=True)
    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    check_smpl_exists()
    model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, init_renderer=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    model.checkpoint_path = checkpoint_path
    return model, model_cfg, device, Path(CACHE_DIR_4DHUMANS)


def _bbox_from_item(item: dict[str, Any], image_shape: tuple[int, int, int]) -> np.ndarray:
    profile = item.get("white_model_profile") or {}
    bbox = profile.get("image_bbox") or [0, 0, image_shape[1] - 1, image_shape[0] - 1]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    pad_x = max(18.0, (x1 - x0) * 0.08)
    pad_y = max(18.0, (y1 - y0) * 0.06)
    x0 = max(0.0, x0 - pad_x)
    y0 = max(0.0, y0 - pad_y)
    x1 = min(float(image_shape[1] - 1), x1 + pad_x)
    y1 = min(float(image_shape[0] - 1), y1 + pad_y)
    return np.array([[x0, y0, x1, y1]], dtype=np.float32)


def _run_single(
    item: dict[str, Any],
    *,
    model,
    model_cfg,
    device,
) -> dict[str, Any]:
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.utils import recursive_to
    from hmr2.utils.renderer import Renderer, cam_crop_to_full

    source_path = ROOT_DIR / str(item["source_path"])
    character_dir = ROOT_DIR / str(item["character_dir"])
    character_path = character_dir / "character.json"
    character = json.loads(character_path.read_text(encoding="utf-8"))

    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(f"failed to read image: {source_path}")

    boxes = _bbox_from_item(item, image.shape)
    dataset = ViTDetDataset(model_cfg, image, boxes)
    batch = dataset[0]
    batch = {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else torch.as_tensor(value).unsqueeze(0)
        for key, value in batch.items()
    }
    batch = recursive_to(batch, device)

    with torch.no_grad():
        output = model(batch)

    pred_vertices = output["pred_vertices"][0].detach().cpu().numpy()
    pred_cam = output["pred_cam"]
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)[0].detach().cpu().numpy()

    mesh_dir = character_dir / "hmr2_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    obj_path = mesh_dir / "body_hmr2.obj"
    params_path = mesh_dir / "hmr2_params.json"
    texture_path = mesh_dir / "source_texture.png"

    renderer = Renderer(model_cfg, faces=model.smpl.faces)
    trimesh_mesh = renderer.vertices_to_trimesh(pred_vertices, pred_cam_t_full, mesh_base_color=(0.92, 0.92, 0.93))
    trimesh_mesh.export(obj_path)
    Image.open(source_path).convert("RGBA").save(texture_path)

    pred_smpl_params = output["pred_smpl_params"]

    payload = {
        "source_path": str(source_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "checkpoint": str(getattr(model, "checkpoint_path", "") or ""),
        "box_xyxy": [round(float(v), 3) for v in boxes[0]],
        "camera_translation": [round(float(v), 6) for v in pred_cam_t_full],
        "vertex_count": int(pred_vertices.shape[0]),
        "focal_length": float(scaled_focal_length.detach().cpu().item() if hasattr(scaled_focal_length, "detach") else scaled_focal_length),
        "smpl_params_rotmat": {
            "global_orient": pred_smpl_params["global_orient"][0].detach().cpu().numpy().tolist(),
            "body_pose": pred_smpl_params["body_pose"][0].detach().cpu().numpy().tolist(),
            "betas": pred_smpl_params["betas"][0].detach().cpu().numpy().tolist(),
        },
        "texture_image_path": str(texture_path.relative_to(ROOT_DIR)).replace("\\", "/"),
    }
    write_json(params_path, payload)

    character["model_style"] = "stickman"
    character["head_style"] = "panda_head_stickman"
    character["hmr2_mesh_asset_path"] = str(obj_path.relative_to(ROOT_DIR)).replace("\\", "/")
    character["hmr2_params_path"] = str(params_path.relative_to(ROOT_DIR)).replace("\\", "/")
    character["hmr2_texture_path"] = str(texture_path.relative_to(ROOT_DIR)).replace("\\", "/")
    write_json(character_path, character)

    return {
        "asset_id": str(item["asset_id"]),
        "hmr2_mesh_asset_path": character["hmr2_mesh_asset_path"],
        "hmr2_params_path": character["hmr2_params_path"],
        "vertex_count": int(pred_vertices.shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HMR2 on local people images and export real human meshes.")
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    items = _load_people_items(args.index)
    if not items:
        print("no people items found")
        return 0

    model, model_cfg, device, _ = _load_hmr2(args.checkpoint)
    results = []
    for item in items:
        result = _run_single(item, model=model, model_cfg=model_cfg, device=device)
        results.append(result)
        print(result["hmr2_mesh_asset_path"])

    payload = json.loads(args.index.read_text(encoding="utf-8"))
    payload["hmr2_meshes"] = results
    write_json(args.index, payload)
    print(args.index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
