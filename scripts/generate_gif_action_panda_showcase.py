#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from storyboard import (
    BaseVideoScript,
    actor,
    audio_bgm,
    audio_sfx,
    beat,
    camera_pan,
    camera_static,
    cast_member,
    dialogue,
    effect,
    expression,
    foreground,
    scene,
    scene_audio,
)


VIDEO = {
    "width": 960,
    "height": 540,
    "fps": 12,
    "renderer": "panda_card_fast",
    "character_model_style": "stickman",
    "force_pose_skeleton": True,
    "show_outfit_overlay": False,
    "show_head_overlay": False,
    "video_codec": "mpeg4",
    "encoder_preset": "ultrafast",
    "crf": 26,
    "audio_bitrate": "64k",
    "subtitle_mode": "bottom",
    "tts_enabled": True,
    "stage_layout": {
        "effect_overlay_alpha": 0.88,
    },
}

CAST = [
    cast_member("host", "馆主", "master-monk"),
    cast_member("taiji", "太极师兄", "young-hero"),
    cast_member("acrobat", "翻子手", "strategist"),
    cast_member("sword", "舞剑者", "swordswoman"),
    cast_member("target", "木桩陪练", "general-guard"),
]

SCENE_DURATION_MS = 14_800
DIALOGUE_WINDOWS = [
    (400, 2800),
    (3300, 5900),
    (6700, 9500),
]

FLOOR_BY_BACKGROUND = {
    "theatre-stage": "dark-stage",
    "training-ground": "stone-court",
    "temple-courtyard": "stone-court",
    "night-bridge": "dark-stage",
    "mountain-cliff": "stone-court",
}

FIST_AUDIO = "assets/audio/031_26_赤手空拳打斗的声音_爱给网_aigei_com.mp3"
HIT_AUDIO = "assets/audio/格斗打中.wav"
METAL_AUDIO = "assets/audio/刀剑、金属碰撞（带回音）_爱给网_aigei_com.mp3"
BOOM_AUDIO = "assets/audio/音效 爆炸 爆破 爆发 战斗_爱给网_aigei_com.mp3"

INTRO_BGM = "assets/bgm/历史的天空-古筝-三国演义片尾曲.mp3"
FORM_BGM = "assets/bgm/天府乐-许镜清.mp3"
ACTION_BGM = "assets/bgm/男儿当自强.mp3"
FINALE_BGM = "assets/bgm/最后之战-热血-卢冠廷.mp3"

SOURCE_TAIJI = "assets/actions/太极.gif"
SOURCE_FLIP = "assets/actions/翻跟头gif.gif"
SOURCE_SWORD = "assets/actions/舞剑.webp"
POSE_TAIJI = "assets/actions/.cache/poses/太极.pose.json"
POSE_FLIP = "assets/actions/.cache/poses/翻跟头gif.pose.json"
POSE_SWORD = "assets/actions/.cache/poses/舞剑.pose.json"


DialogueLine = tuple[str, str]


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    background: str
    summary: str
    actors: list[dict]
    lines: list[DialogueLine]
    action_beats: list[dict]
    effects: list[dict]
    sfx: list[dict]
    bgm: dict
    camera: dict
    foregrounds: list[dict]


def front_actor(actor_id: str, x: float, *, facing: str, scale: float = 1.0, z: float = 0.0) -> dict:
    return actor(actor_id, x, z=z, facing=facing, scale=scale)


def mid_actor(actor_id: str, x: float, *, facing: str, scale: float = 0.92, z: float = -0.20) -> dict:
    return actor(actor_id, x, z=z, facing=facing, scale=scale, layer="mid")


def line(speaker_id: str, text: str) -> DialogueLine:
    return (speaker_id, text)


def pose_beat(start_ms: int, end_ms: int, actor_id: str, motion: str, pose_track_path: str, **kwargs) -> dict:
    item = beat(start_ms, end_ms, actor_id, motion, **kwargs)
    item["pose_track_path"] = pose_track_path
    return item


def infer_expression(text: str) -> str:
    if any(token in text for token in ("稳", "慢", "圆", "收", "沉")):
        return "thinking"
    if any(token in text for token in ("翻", "冲", "劈", "斩", "爆", "快")):
        return "angry"
    if any(token in text for token in ("漂亮", "到了", "成了", "好")):
        return "smile"
    return "neutral"


def build_dialogue_bundle(lines: Sequence[DialogueLine]) -> tuple[list[dict], list[dict], list[dict]]:
    dialogue_items: list[dict] = []
    talk_beats: list[dict] = []
    expressions_track: list[dict] = []
    for (start_ms, end_ms), (speaker_id, text) in zip(DIALOGUE_WINDOWS, lines):
        dialogue_items.append(dialogue(start_ms, end_ms, speaker_id, text))
        talk_beats.append(beat(start_ms, end_ms, speaker_id, "talk", emotion="focused"))
        expressions_track.append(expression(speaker_id, start_ms, end_ms, infer_expression(text)))
    return dialogue_items, talk_beats, expressions_track


def trim_talk_beats_for_actions(talk_beats: Sequence[dict], action_beats: Sequence[dict]) -> list[dict]:
    trimmed: list[dict] = []
    for talk in talk_beats:
        segments = [(talk["start_ms"], talk["end_ms"])]
        for action in action_beats:
            if action["actor_id"] != talk["actor_id"]:
                continue
            next_segments: list[tuple[int, int]] = []
            for seg_start, seg_end in segments:
                if action["end_ms"] <= seg_start or action["start_ms"] >= seg_end:
                    next_segments.append((seg_start, seg_end))
                    continue
                if action["start_ms"] > seg_start:
                    next_segments.append((seg_start, action["start_ms"]))
                if action["end_ms"] < seg_end:
                    next_segments.append((action["end_ms"], seg_end))
            segments = next_segments
            if not segments:
                break
        for seg_start, seg_end in segments:
            if seg_end - seg_start < 220:
                continue
            trimmed.append(
                beat(
                    seg_start,
                    seg_end,
                    talk["actor_id"],
                    talk["motion"],
                    facing=talk.get("facing"),
                    emotion=talk.get("emotion", "focused"),
                )
            )
    return trimmed


def scene_card_foregrounds() -> list[dict]:
    return [
        foreground(asset_path=SOURCE_TAIJI, x=-0.27, y=-0.16, width=0.28, height=0.44, opacity=0.90),
        foreground(asset_path=SOURCE_FLIP, x=0.01, y=-0.16, width=0.25, height=0.41, opacity=0.90),
        foreground(asset_path=SOURCE_SWORD, x=0.29, y=-0.16, width=0.30, height=0.44, opacity=0.90),
    ]


def intro_scene() -> dict:
    dialogues, talk_beats, expressions_track = build_dialogue_bundle(
        [
            line("host", "你放进来的三段动作，我已经拆成太极、空翻和舞剑三条模板。"),
            line("host", "今天不放原人像上场，只看熊猫人把动作重新做一遍。"),
            line("sword", "慢的要稳，快的要脆，收势还要留得住。"),
        ]
    )
    action_beats = [
        beat(7600, 8400, "host", "point", facing="right", emotion="focused"),
        beat(7900, 9400, "taiji", "enter", x0=-3.0, x1=-1.9, facing="right", emotion="focused"),
        beat(8200, 9700, "acrobat", "enter", x0=0.0, x1=0.0, facing="right", emotion="charged"),
        beat(8500, 10000, "sword", "enter", x0=3.1, x1=1.8, facing="left", emotion="charged"),
    ]
    return scene(
        "scene-001",
        background="theatre-stage",
        floor=FLOOR_BY_BACKGROUND["theatre-stage"],
        duration_ms=SCENE_DURATION_MS,
        summary="开场说明三段动作素材已经被改写为 Panda3D 熊猫人的动作模板。",
        camera=camera_static(x=0.0, z=0.03, zoom=1.06),
        actors=[
            mid_actor("host", -3.2, facing="right", scale=0.92),
            front_actor("taiji", -1.9, facing="right", scale=1.02),
            front_actor("acrobat", 0.0, facing="right", scale=1.0),
            front_actor("sword", 2.0, facing="left", scale=1.02),
        ],
        beats=[*trim_talk_beats_for_actions(talk_beats, action_beats), *action_beats],
        expressions=expressions_track,
        dialogues=dialogues,
        foregrounds=scene_card_foregrounds(),
        effects=[],
        audio=scene_audio(
            bgm=audio_bgm(INTRO_BGM, volume=0.54, loop=True),
            sfx=[
                audio_sfx(FIST_AUDIO, start_ms=8160, volume=0.62),
                audio_sfx(METAL_AUDIO, start_ms=9020, volume=0.48),
            ],
        ),
        notes={"sources": [SOURCE_TAIJI, SOURCE_FLIP, SOURCE_SWORD]},
    )


def scene_from_spec(spec: SceneSpec) -> dict:
    dialogues, talk_beats, expressions_track = build_dialogue_bundle(spec.lines)
    return scene(
        spec.scene_id,
        background=spec.background,
        floor=FLOOR_BY_BACKGROUND[spec.background],
        duration_ms=SCENE_DURATION_MS,
        summary=spec.summary,
        camera=spec.camera,
        actors=spec.actors,
        beats=[*trim_talk_beats_for_actions(talk_beats, spec.action_beats), *spec.action_beats],
        expressions=expressions_track,
        dialogues=dialogues,
        foregrounds=spec.foregrounds,
        effects=spec.effects,
        audio=scene_audio(bgm=spec.bgm, sfx=spec.sfx),
    )


def build_specs() -> list[SceneSpec]:
    return [
        SceneSpec(
            scene_id="scene-002",
            background="temple-courtyard",
            summary="太极参考 GIF 被转成了 Panda3D 的 tai-chi-flow 动作模板，强调沉肩、转胯和圆手。",
            actors=[
                mid_actor("host", -3.1, facing="right", scale=0.90),
                front_actor("taiji", -0.6, facing="right", scale=1.06),
                front_actor("target", 2.2, facing="left", scale=0.98),
            ],
            lines=[
                line("host", "第一套取自太极 GIF，核心不是快，而是前后手始终走圆。"),
                line("host", "脚下换重心的时候，上身不能飘，腰要像轴一样稳。"),
                line("taiji", "我先打一遍整套慢架，让你看清手脚怎么一起转。"),
            ],
            action_beats=[
                pose_beat(7600, 11600, "taiji", "talk", POSE_TAIJI, x0=-0.8, x1=0.3, facing="right", emotion="focused"),
                beat(8200, 9000, "host", "point", facing="right", emotion="focused"),
            ],
            effects=[],
            sfx=[audio_sfx(FIST_AUDIO, start_ms=8720, volume=0.34)],
            bgm=audio_bgm(FORM_BGM, volume=0.50, loop=True),
            camera=camera_pan(x=-0.18, z=0.04, zoom=1.03, to_x=0.16, to_z=0.0, to_zoom=1.10, ease="ease-in-out"),
            foregrounds=[foreground("中式古典大门", x=-0.01, y=-0.02, width=1.02, height=1.05, opacity=0.92)],
        ),
        SceneSpec(
            scene_id="scene-003",
            background="temple-courtyard",
            summary="双人镜像验证太极模板，左右换势时骨架仍然能保持圆融和稳定。",
            actors=[
                front_actor("taiji", -1.7, facing="right", scale=1.02),
                front_actor("target", 1.9, facing="left", scale=0.98),
                mid_actor("host", -3.4, facing="right", scale=0.86),
            ],
            lines=[
                line("host", "第二遍看双人镜像，重点是左转右转都不能塌胯。"),
                line("target", "我不抢招，只陪你走一圈，看这个圆能不能闭合。"),
                line("taiji", "来吧，前手领路，后手守中，脚下慢慢把门关上。"),
            ],
            action_beats=[
                pose_beat(7600, 11800, "taiji", "talk", POSE_TAIJI, x0=-1.8, x1=-0.8, facing="right", emotion="focused"),
                pose_beat(7600, 11800, "target", "talk", POSE_TAIJI, x0=1.9, x1=0.9, facing="left", emotion="focused"),
            ],
            effects=[],
            sfx=[audio_sfx(FIST_AUDIO, start_ms=8960, volume=0.28)],
            bgm=audio_bgm(FORM_BGM, volume=0.48, loop=True),
            camera=camera_pan(x=-0.10, z=0.03, zoom=1.00, to_x=0.12, to_z=0.01, to_zoom=1.08, ease="ease-in-out"),
            foregrounds=[],
        ),
        SceneSpec(
            scene_id="scene-004",
            background="training-ground",
            summary="翻跟头 GIF 被改写成连续 somersault 链，强调腾空弧线和落点连续性。",
            actors=[
                mid_actor("host", -3.3, facing="right", scale=0.88),
                front_actor("acrobat", -2.5, facing="right", scale=1.02),
                front_actor("target", 2.1, facing="left", scale=0.96),
            ],
            lines=[
                line("host", "第二套来自翻跟头 GIF，我没有照抄，而是把节奏拆成连续空翻链。"),
                line("host", "每一翻都要有起、顶、落三段，落地还得立刻能接下一翻。"),
                line("acrobat", "你只看我的落点，它会一格一格往前咬，不会散。"),
            ],
            action_beats=[
                pose_beat(7600, 9000, "acrobat", "somersault", POSE_FLIP, x0=-2.7, x1=-1.1, facing="right", emotion="charged"),
                pose_beat(9040, 10340, "acrobat", "somersault", POSE_FLIP, x0=-1.1, x1=0.6, facing="right", emotion="charged"),
                pose_beat(10380, 11650, "acrobat", "somersault", POSE_FLIP, x0=0.6, x1=2.1, facing="right", emotion="charged"),
                beat(10650, 11800, "target", "exit", x0=2.1, x1=2.7, facing="left", emotion="hurt"),
            ],
            effects=[effect("explosion", start_ms=11020, end_ms=11820, alpha=0.06, playback_speed=1.00)],
            sfx=[
                audio_sfx(BOOM_AUDIO, start_ms=8020, volume=0.62),
                audio_sfx(BOOM_AUDIO, start_ms=9340, volume=0.60),
                audio_sfx(HIT_AUDIO, start_ms=11080, volume=0.86),
            ],
            bgm=audio_bgm(ACTION_BGM, volume=0.56, loop=True),
            camera=camera_pan(x=-0.34, z=0.04, zoom=1.05, to_x=0.30, to_z=0.01, to_zoom=1.15, ease="ease-in-out"),
            foregrounds=[],
        ),
        SceneSpec(
            scene_id="scene-005",
            background="night-bridge",
            summary="连续空翻的中段加入二次提速，让 Panda3D 熊猫人在同一 scene 内做出三段不同长度的翻转。",
            actors=[
                front_actor("acrobat", -2.8, facing="right", scale=1.04),
                front_actor("target", 2.6, facing="left", scale=0.94),
                mid_actor("host", -3.6, facing="right", scale=0.84),
            ],
            lines=[
                line("host", "再看一遍加速版，第一翻探路，第二翻抬高，第三翻直接穿身位。"),
                line("host", "这样做出来，就更接近你那张 GIF 里连续翻进的感觉。"),
                line("acrobat", "前两翻吃距离，最后一翻才拿来撞人。"),
            ],
            action_beats=[
                pose_beat(7600, 8800, "acrobat", "somersault", POSE_FLIP, x0=-2.8, x1=-1.5, facing="right", emotion="charged"),
                pose_beat(8840, 10020, "acrobat", "somersault", POSE_FLIP, x0=-1.5, x1=0.2, facing="right", emotion="charged"),
                pose_beat(10060, 11480, "acrobat", "somersault", POSE_FLIP, x0=0.2, x1=2.4, facing="right", emotion="charged"),
                beat(10920, 11800, "target", "big-jump", x0=2.5, x1=3.0, facing="left", emotion="hurt"),
            ],
            effects=[effect("hit", start_ms=10880, end_ms=11620, alpha=0.06, playback_speed=0.96)],
            sfx=[
                audio_sfx(BOOM_AUDIO, start_ms=7860, volume=0.54),
                audio_sfx(BOOM_AUDIO, start_ms=9120, volume=0.54),
                audio_sfx(BOOM_AUDIO, start_ms=10460, volume=0.62),
                audio_sfx(HIT_AUDIO, start_ms=10920, volume=0.92),
            ],
            bgm=audio_bgm(ACTION_BGM, volume=0.58, loop=True),
            camera=camera_pan(x=-0.28, z=0.05, zoom=1.08, to_x=0.34, to_z=0.00, to_zoom=1.18, ease="ease-in-out"),
            foregrounds=[foreground("中式古典大门", x=-0.01, y=-0.02, width=1.02, height=1.05, opacity=0.90)],
        ),
        SceneSpec(
            scene_id="scene-006",
            background="mountain-cliff",
            summary="舞剑 WebP 被抽象成 sword-dance，先保留跑位和提膝，再把斜斩和落地收势接进同一套动作。",
            actors=[
                mid_actor("host", -3.2, facing="right", scale=0.88),
                front_actor("sword", -1.8, facing="right", scale=1.04),
                front_actor("target", 2.0, facing="left", scale=0.98),
            ],
            lines=[
                line("host", "第三套参考是舞剑 WebP，它最难的是一整串跑、提、转、落不能断。"),
                line("host", "我把它压成 sword-dance，一个 beat 里把起势和斜斩都放进去。"),
                line("sword", "先看第一遍开架，不急着碰人，先把刀路走满。"),
            ],
            action_beats=[
                pose_beat(7600, 11600, "sword", "sword-arc", POSE_SWORD, x0=-1.9, x1=0.7, facing="right", effect="sword-arc", emotion="charged"),
            ],
            effects=[],
            sfx=[
                audio_sfx(METAL_AUDIO, start_ms=8980, volume=0.76),
                audio_sfx(FIST_AUDIO, start_ms=10460, volume=0.42),
            ],
            bgm=audio_bgm(ACTION_BGM, volume=0.56, loop=True),
            camera=camera_pan(x=-0.20, z=0.05, zoom=1.04, to_x=0.22, to_z=0.0, to_zoom=1.13, ease="ease-in-out"),
            foregrounds=[],
        ),
        SceneSpec(
            scene_id="scene-007",
            background="training-ground",
            summary="剑舞模板开始加入目标判定，斜斩和转身落点与陪练反应同步。",
            actors=[
                front_actor("sword", -1.7, facing="right", scale=1.04),
                front_actor("target", 1.8, facing="left", scale=0.98),
                mid_actor("host", -3.4, facing="right", scale=0.84),
            ],
            lines=[
                line("host", "这一遍开始带目标，看提膝之后的横切怎么压进对方胸线。"),
                line("target", "她一转开，刀路已经过半，我只能往后撤。"),
                line("sword", "对，就是先抢角，再把斜斩从肩上压下来。"),
            ],
            action_beats=[
                pose_beat(7600, 11400, "sword", "sword-arc", POSE_SWORD, x0=-1.8, x1=0.8, facing="right", effect="sword-arc", emotion="charged"),
                beat(9900, 11800, "target", "exit", x0=1.8, x1=2.8, facing="left", emotion="hurt"),
            ],
            effects=[effect("hit", start_ms=10060, end_ms=11040, alpha=0.06, playback_speed=0.94)],
            sfx=[
                audio_sfx(METAL_AUDIO, start_ms=9580, volume=0.84),
                audio_sfx(HIT_AUDIO, start_ms=10140, volume=0.82),
            ],
            bgm=audio_bgm(ACTION_BGM, volume=0.58, loop=True),
            camera=camera_pan(x=-0.22, z=0.03, zoom=1.08, to_x=0.20, to_z=0.0, to_zoom=1.16, ease="ease-in-out"),
            foregrounds=[],
        ),
        SceneSpec(
            scene_id="scene-008",
            background="training-ground",
            summary="剑舞模板的终版带有更明显的腾空和低收势，接近参考素材后段的转身落地。",
            actors=[
                front_actor("sword", -1.5, facing="right", scale=1.06),
                front_actor("target", 2.0, facing="left", scale=0.98),
                mid_actor("host", -3.3, facing="right", scale=0.84),
            ],
            lines=[
                line("host", "最后看终版，动作后段我故意压低，让它接近参考里那一下落地收身。"),
                line("host", "这样熊猫人看起来就不是瞎挥，而是真的在收刀。"),
                line("sword", "收势一稳，这套长动作才算有了根。"),
            ],
            action_beats=[
                pose_beat(7600, 11520, "sword", "sword-arc", POSE_SWORD, x0=-1.7, x1=0.9, facing="right", effect="sword-arc", emotion="charged"),
                beat(10380, 11800, "target", "big-jump", x0=2.0, x1=2.8, facing="left", emotion="hurt"),
            ],
            effects=[effect("explosion", start_ms=10440, end_ms=11180, alpha=0.05, playback_speed=1.04)],
            sfx=[
                audio_sfx(METAL_AUDIO, start_ms=9520, volume=0.82),
                audio_sfx(HIT_AUDIO, start_ms=10480, volume=0.90),
                audio_sfx(BOOM_AUDIO, start_ms=10620, volume=0.46),
            ],
            bgm=audio_bgm(ACTION_BGM, volume=0.58, loop=True),
            camera=camera_pan(x=-0.18, z=0.04, zoom=1.10, to_x=0.18, to_z=0.0, to_zoom=1.18, ease="ease-in-out"),
            foregrounds=[],
        ),
        SceneSpec(
            scene_id="scene-009",
            background="theatre-stage",
            summary="三套动作同时回到同一舞台，验证从参考素材抽出来的 Panda3D 模板能共存并复用。",
            actors=[
                front_actor("taiji", -2.6, facing="right", scale=0.98),
                front_actor("acrobat", 0.0, facing="right", scale=1.00),
                front_actor("sword", 2.4, facing="left", scale=1.00),
                mid_actor("host", -3.8, facing="right", scale=0.80),
            ],
            lines=[
                line("host", "现在把三套动作并排摆回舞台，你就能看出它们已经成了可复用模板。"),
                line("host", "左边走太极，中间接空翻，右边收剑舞，节奏不会互相打架。"),
                line("acrobat", "以后再给新 GIF，我就按这个方法继续扩动作库。"),
            ],
            action_beats=[
                pose_beat(7600, 11300, "taiji", "talk", POSE_TAIJI, x0=-2.7, x1=-1.8, facing="right", emotion="focused"),
                pose_beat(7920, 9160, "acrobat", "somersault", POSE_FLIP, x0=-0.4, x1=0.7, facing="right", emotion="charged"),
                pose_beat(9200, 10320, "acrobat", "somersault", POSE_FLIP, x0=0.7, x1=1.8, facing="right", emotion="charged"),
                pose_beat(7600, 11480, "sword", "sword-arc", POSE_SWORD, x0=2.6, x1=1.4, facing="left", effect="sword-arc", emotion="charged"),
            ],
            effects=[],
            sfx=[
                audio_sfx(BOOM_AUDIO, start_ms=8240, volume=0.46),
                audio_sfx(BOOM_AUDIO, start_ms=9460, volume=0.44),
                audio_sfx(METAL_AUDIO, start_ms=9620, volume=0.68),
            ],
            bgm=audio_bgm(FINALE_BGM, volume=0.56, loop=True),
            camera=camera_static(x=0.0, z=0.04, zoom=1.10),
            foregrounds=[foreground("敞开的红色帘子-窗帘或床帘皆可", x=-0.02, y=-0.04, width=1.04, height=1.10, opacity=1.0)],
        ),
        SceneSpec(
            scene_id="scene-010",
            background="theatre-stage",
            summary="结尾总结这三套由 GIF 派生而来的动作已经被做成可继续扩展的熊猫人动作库。",
            actors=[
                mid_actor("host", -3.0, facing="right", scale=0.92),
                front_actor("taiji", -1.6, facing="right", scale=0.96),
                front_actor("acrobat", 0.3, facing="right", scale=0.96),
                front_actor("sword", 2.1, facing="left", scale=0.96),
            ],
            lines=[
                line("host", "这支片子的重点，不只是成片，而是把你给的 GIF 真正落成了动作模板。"),
                line("host", "下一次你再丢进新动作，我就能继续往这套骨架库里扩。"),
                line("sword", "现在这三套已经能直接拿去写剧情、打斗和表演场景了。"),
            ],
            action_beats=[
                beat(7600, 8500, "host", "point", facing="right", emotion="focused"),
                pose_beat(8640, 10380, "taiji", "talk", POSE_TAIJI, x0=-1.7, x1=-1.0, facing="right", emotion="focused"),
                pose_beat(9040, 10340, "acrobat", "somersault", POSE_FLIP, x0=-0.1, x1=0.9, facing="right", emotion="charged"),
                pose_beat(9440, 11420, "sword", "sword-arc", POSE_SWORD, x0=2.2, x1=1.2, facing="left", effect="sword-arc", emotion="charged"),
            ],
            effects=[],
            sfx=[
                audio_sfx(BOOM_AUDIO, start_ms=9440, volume=0.40),
                audio_sfx(METAL_AUDIO, start_ms=10440, volume=0.64),
            ],
            bgm=audio_bgm(FINALE_BGM, volume=0.54, loop=True),
            camera=camera_pan(x=-0.10, z=0.03, zoom=1.04, to_x=0.12, to_z=0.0, to_zoom=1.10, ease="ease-in-out"),
            foregrounds=[foreground("敞开的红色帘子-窗帘或床帘皆可", x=-0.02, y=-0.04, width=1.04, height=1.10, opacity=1.0)],
        ),
    ]


class GifActionPandaShowcaseVideo(BaseVideoScript):
    def get_title(self) -> str:
        return "GIF 动作临摹熊猫演武"

    def get_theme(self) -> str:
        return "基于本地动作 GIF 与 WebP 参考重建 Panda3D 熊猫人动作模板"

    def has_tts(self) -> bool:
        return True

    def get_default_output(self) -> str:
        return "outputs/gif_action_panda_showcase.mp4"

    def get_description(self) -> str:
        return "Render a stickman action showcase that adapts the local taiji, flip, and sword references under assets/actions into reusable pose-driven motions."

    def get_video_options(self) -> dict:
        return VIDEO

    def get_notes(self) -> dict:
        return {
            "focus": "gif-action-panda-showcase",
            "source_actions": [SOURCE_TAIJI, SOURCE_FLIP, SOURCE_SWORD],
            "derived_motions": ["talk", "somersault", "sword-arc"],
            "bgm_assets": [INTRO_BGM, FORM_BGM, ACTION_BGM, FINALE_BGM],
        }

    def get_cast(self) -> list[dict]:
        return CAST

    def get_scenes(self) -> list[dict]:
        scenes = [intro_scene()]
        scenes.extend(scene_from_spec(spec) for spec in build_specs())
        return scenes


SCRIPT = GifActionPandaShowcaseVideo()


def build_story() -> dict:
    return SCRIPT.build_story()


def main() -> int:
    return SCRIPT()


if __name__ == "__main__":
    raise SystemExit(main())
