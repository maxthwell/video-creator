#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from storyboard import (
    BaseVideoScript,
    actor,
    audio_bgm,
    audio_sfx,
    beat,
    camera_pan,
    camera_static,
    cast_member,
    effect,
    scene,
    scene_audio,
)


INDEX_PATH = Path("assets/people/.cache/white_model_index.json")

VIDEO = {
    "width": 960,
    "height": 540,
    "fps": 12,
    "renderer": "true_3d",
    "character_model_style": "stickman",
    "show_actor_labels": False,
    "video_codec": "mpeg4",
    "encoder_preset": "ultrafast",
    "crf": 26,
    "audio_bitrate": "64k",
    "subtitle_mode": "bottom",
    "tts_enabled": False,
}


def _load_models() -> list[dict]:
    if not INDEX_PATH.exists():
        raise SystemExit(f"missing index: {INDEX_PATH}. run scripts/extract_people_white_models.py first")
    payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    items = payload.get("items") or []
    if not items:
        raise SystemExit(f"no models found in {INDEX_PATH}")
    return items[:2]


MODELS = _load_models()
PRIMARY = MODELS[0]
SECONDARY = MODELS[1] if len(MODELS) > 1 else MODELS[0]

CAST = [
    cast_member("model_a", PRIMARY["display_name"], PRIMARY["asset_id"]),
    cast_member("model_b", SECONDARY["display_name"], SECONDARY["asset_id"]),
]


def pose_beat(start_ms: int, end_ms: int, actor_id: str, pose_track_path: str, motion: str = "talk", **kwargs) -> dict:
    item = beat(start_ms, end_ms, actor_id, motion, **kwargs)
    item["pose_track_path"] = pose_track_path
    return item


SCENES = [
    scene(
        "scene-001",
        background="theatre-stage",
        floor="dark-stage",
        duration_ms=7600,
        summary="两个人物白模从两侧入场，先给整体建模印象。",
        camera=camera_pan(x=0.0, z=0.18, zoom=1.58, to_x=0.0, to_z=0.18, to_zoom=1.70),
        actors=[
            actor("model_a", -2.8, facing="right", scale=1.02),
            actor("model_b", 2.8, facing="left", scale=1.0),
        ],
        beats=[
            beat(600, 2500, "model_a", "enter", x0=-3.4, x1=-1.3, facing="right"),
            beat(1100, 3200, "model_b", "enter", x0=3.4, x1=1.4, facing="left"),
            beat(4200, 6800, "model_a", "point", facing="right"),
            beat(4300, 6900, "model_b", "point", facing="left"),
        ],
        effects=[],
        audio=scene_audio(
            bgm=audio_bgm("assets/bgm/历史的天空-古筝-三国演义片尾曲.mp3", volume=0.38),
            sfx=[],
        ),
    ),
    scene(
        "scene-002",
        background="museum-gallery",
        floor="wood-plank",
        duration_ms=6200,
        summary="第一张全身图拟合出的白模保持源姿态，强调头身和体块比例。",
        camera=camera_static(x=-0.02, z=0.22, zoom=1.82),
        actors=[actor("model_a", -0.7, facing="right", scale=1.06)],
        beats=[pose_beat(1, 6200, "model_a", PRIMARY["pose_track_path"], motion="idle", facing="right")],
        effects=[],
        audio=scene_audio(
            bgm=audio_bgm("assets/bgm/天府乐-许镜清.mp3", volume=0.30),
            sfx=[],
        ),
    ),
    scene(
        "scene-003",
        background="museum-gallery",
        floor="wood-plank",
        duration_ms=6200,
        summary="第二张全身图拟合出的白模保持源姿态，和前一个角色形成对比。",
        camera=camera_static(x=0.02, z=0.22, zoom=1.82),
        actors=[actor("model_b", 0.7, facing="left", scale=1.0)],
        beats=[pose_beat(1, 6200, "model_b", SECONDARY["pose_track_path"], motion="idle", facing="left")],
        effects=[],
        audio=scene_audio(
            bgm=audio_bgm("assets/bgm/天府乐-许镜清.mp3", volume=0.30),
            sfx=[],
        ),
    ),
    scene(
        "scene-004",
        background="training-ground",
        floor="stone-court",
        duration_ms=7600,
        summary="白模开始做基础行走和指向动作，验证骨架可动画化。",
        camera=camera_pan(x=-0.02, z=0.18, zoom=1.46, to_x=0.02, to_z=0.20, to_zoom=1.58),
        actors=[
            actor("model_a", -2.7, facing="right", scale=1.02),
            actor("model_b", 2.6, facing="left", scale=1.0),
        ],
        beats=[
            beat(500, 2200, "model_a", "enter", x0=-2.7, x1=-1.0, facing="right"),
            beat(1100, 2900, "model_b", "enter", x0=2.6, x1=1.0, facing="left"),
            beat(3400, 5600, "model_a", "talk", facing="right"),
            beat(3600, 5900, "model_b", "point", facing="left"),
        ],
        effects=[],
        audio=scene_audio(
            bgm=audio_bgm("assets/bgm/仙剑情缘.mp3", volume=0.32),
            sfx=[audio_sfx("assets/audio/格斗打中.wav", start_ms=3980, volume=0.35)],
        ),
    ),
    scene(
        "scene-005",
        background="night-bridge",
        floor="dark-stage",
        duration_ms=8200,
        summary="夜景收尾镜头，两个白模再回到源姿态，作为生成结果确认。",
        camera=camera_pan(x=0.0, z=0.20, zoom=1.62, to_x=0.0, to_z=0.24, to_zoom=1.78),
        actors=[
            actor("model_a", -1.8, facing="right", scale=1.02),
            actor("model_b", 1.9, facing="left", scale=1.0),
        ],
        beats=[
            pose_beat(1, 3600, "model_a", PRIMARY["pose_track_path"], motion="idle", facing="right"),
            pose_beat(1, 3600, "model_b", SECONDARY["pose_track_path"], motion="idle", facing="left"),
            beat(4300, 7200, "model_a", "point", facing="right"),
            beat(4300, 7200, "model_b", "talk", facing="left"),
        ],
        effects=[],
        audio=scene_audio(
            bgm=audio_bgm("assets/bgm/最后之战-热血-卢冠廷.mp3", volume=0.30),
            sfx=[],
        ),
    ),
]


class PeopleWhiteModelShowcase(BaseVideoScript):
    def get_title(self) -> str:
        return "人物熊猫头火柴人展示"

    def get_theme(self) -> str:
        return "single-image-panda-stickman"

    def get_default_output(self) -> str:
        return "outputs/people_white_model_showcase.mp4"

    def get_video_options(self) -> dict:
        return VIDEO

    def get_cast(self):
        return CAST

    def get_scenes(self):
        return SCENES


SCRIPT = PeopleWhiteModelShowcase()
VIDEO_SCRIPT = SCRIPT


if __name__ == "__main__":
    raise SystemExit(SCRIPT())
