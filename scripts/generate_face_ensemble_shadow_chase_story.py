#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from storyboard import (
    BaseVideoScript,
    actor,
    beat,
    camera_pan,
    camera_static,
    cast_member,
    dialogue,
    effect,
    expression,
    foreground,
    prop,
    scene,
    scene_audio,
)


VIDEO = {
    "width": 960,
    "height": 540,
    "fps": 12,
    "renderer": "panda_card_fast",
    "video_codec": "mpeg4",
    "encoder_preset": "ultrafast",
    "crf": 26,
    "subtitle_mode": "bottom",
    "tts_enabled": True,
    "stage_layout": {
        "effect_overlay_alpha": 0.88,
    },
}

CAST = [
    cast_member("shen_li", "沈砺", "face-1"),
    cast_member("luo_cheng", "罗诚", "face-2"),
    cast_member("gu_yuan", "顾远", "face-3"),
    cast_member("wen_xia", "闻夏", "face-5"),
    cast_member("qiao_yu", "乔雨", "face-7"),
    cast_member("lin_zhi", "林枝", "face-8"),
    cast_member("xu_ning", "许宁", "face-13"),
    cast_member("song_cheng", "宋澄", "face-14"),
    cast_member("tang_ji", "唐霁", "face-15"),
    cast_member("huo_lin", "霍临", "face-17"),
]

SCENE_DURATION_MS = 14_800
DIALOGUE_WINDOWS = [
    (400, 3000),
    (3500, 6100),
    (7600, 10100),
    (11000, 14100),
]

FLOOR_BY_BACKGROUND = {
    "archive-library": "wood-plank",
    "bank-lobby": "wood-plank",
    "cafe-night": "wood-plank",
    "inn-hall": "wood-plank",
    "mountain-cliff": "stone-court",
    "museum-gallery": "wood-plank",
    "night-bridge": "dark-stage",
    "park-evening": "dark-stage",
    "restaurant-booth": "wood-plank",
    "room-day": "wood-plank",
    "school-yard": "stone-court",
    "shop-row": "stone-court",
    "street-day": "stone-court",
    "temple-courtyard": "stone-court",
    "theatre-stage": "dark-stage",
    "town-hall-records": "wood-plank",
    "training-ground": "stone-court",
}

FIST_AUDIO = "assets/audio/031_26_赤手空拳打斗的声音_爱给网_aigei_com.mp3"
METAL_AUDIO = "assets/audio/刀剑、金属碰撞（带回音）_爱给网_aigei_com.mp3"
BOOM_AUDIO = "assets/audio/音效 爆炸 爆破 爆发 战斗_爱给网_aigei_com.mp3"
HEART_AUDIO = "assets/audio/心脏怦怦跳.wav"
HIT_AUDIO = "assets/audio/格斗打中.wav"
THUNDER_AUDIO = "assets/audio/打雷闪电.wav"

PROLOGUE_BGM = "assets/bgm/误入迷失森林-少年包青天.mp3"
TRACE_BGM = "assets/bgm/御剑飞行.mp3"
CHASE_BGM = "assets/bgm/杀破狼.mp3"
CRISIS_BGM = "assets/bgm/观音降临-高潮版.mp3"
FINAL_BGM = "assets/bgm/最后之战-热血-卢冠廷.mp3"
EPILOGUE_BGM = "assets/bgm/仙剑情缘.mp3"


DialogueLine = tuple[str, str]


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    background: str
    summary: str
    actors: list[dict]
    props: list[dict]
    lines: list[DialogueLine]
    extra_beats: list[dict] = field(default_factory=list)
    effects: list[dict] = field(default_factory=list)
    foregrounds: list[dict] = field(default_factory=list)
    audio: dict = field(default_factory=scene_audio)
    camera: dict | None = None


def line(speaker_id: str, text: str) -> DialogueLine:
    return (speaker_id, text)


def front_actor(actor_id: str, x: float, *, facing: str, scale: float = 1.0, z: float = 0.0) -> dict:
    return actor(actor_id, x, z=z, facing=facing, scale=scale)


def mid_actor(actor_id: str, x: float, *, facing: str, scale: float = 0.94, z: float = -0.14) -> dict:
    return actor(actor_id, x, z=z, facing=facing, scale=scale, layer="mid")


def back_actor(actor_id: str, x: float, *, facing: str, scale: float = 0.88, z: float = -0.72) -> dict:
    return actor(actor_id, x, z=z, facing=facing, scale=scale, layer="back")


def infer_expression(text: str) -> str:
    if any(token in text for token in ("炸", "断电", "封锁", "追", "打", "抢", "冲", "埋伏")):
        return "angry"
    if any(token in text for token in ("快", "马上", "立刻", "机会", "上桥", "撤离")):
        return "excited"
    if any(token in text for token in ("芯片", "坐标", "记录", "内线", "证据", "信号")):
        return "skeptical"
    if any(token in text for token in ("守住", "没事", "回来了", "结束了", "赢")):
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


def scene_camera(scene_index: int, *, battle: bool, rooftop: bool = False) -> dict:
    if rooftop:
        return camera_pan(
            x=-0.3,
            z=0.08,
            zoom=1.1,
            to_x=0.26,
            to_z=0.02,
            to_zoom=1.18,
            ease="ease-in-out",
        )
    if battle:
        return camera_pan(
            x=-0.22 + 0.05 * (scene_index % 3),
            z=0.04,
            zoom=1.06,
            to_x=0.15 - 0.03 * (scene_index % 2),
            to_z=0.01,
            to_zoom=1.14,
            ease="ease-in-out",
        )
    if scene_index in {0, 19}:
        return camera_static(x=0.0, z=0.03, zoom=1.06)
    return camera_pan(
        x=-0.14,
        z=0.02,
        zoom=1.0,
        to_x=0.12,
        to_z=0.0,
        to_zoom=1.05,
        ease="ease-in-out",
    )


def city_props(scene_index: int, *, interior: bool = False, night: bool = False, open_space: bool = False) -> list[dict]:
    if interior:
        return [
            prop("wall-window", -3.8, -1.0, scale=0.92, layer="back"),
            prop("wall-door", 3.8, -1.02, scale=0.92, layer="back"),
            prop("lantern", 3.0, -0.93, scale=0.9, layer="front"),
            prop("weapon-rack" if scene_index % 2 == 0 else "training-drum", -0.2, -1.02, scale=0.88, layer="mid"),
        ]
    if open_space:
        props = [prop("house", 0.0, -1.08, scale=0.98, layer="back")]
        if night:
            props.extend(
                [
                    prop("moon", 3.7, -0.42, scale=0.72, layer="back"),
                    prop("star", -3.6, -0.55, scale=0.5, layer="back"),
                    prop("lantern", -3.2, -0.93, scale=0.88, layer="front"),
                ]
            )
        else:
            props.extend(
                [
                    prop("wall-door", 3.8, -1.02, scale=0.9, layer="back"),
                    prop("horse", -3.8, -0.95, scale=0.82, layer="front"),
                ]
            )
        return props
    return [
        prop("house", 0.0, -1.08, scale=0.98, layer="back"),
        prop("wall-door", 3.8, -1.02, scale=0.88, layer="back"),
    ]


def city_audio(*, chase: bool = False, metal: bool = False, boom: bool = False, heart: bool = False, thunder: bool = False) -> dict:
    sfx: list[dict] = []
    if chase:
        sfx.extend(
            [
                {"asset_path": FIST_AUDIO, "start_ms": 4100, "volume": 0.76, "loop": False},
                {"asset_path": HIT_AUDIO, "start_ms": 8350, "volume": 0.82, "loop": False},
            ]
        )
    if metal:
        sfx.extend(
            [
                {"asset_path": METAL_AUDIO, "start_ms": 5800, "volume": 0.7, "loop": False},
                {"asset_path": METAL_AUDIO, "start_ms": 10400, "volume": 0.68, "loop": False},
            ]
        )
    if boom:
        sfx.extend(
            [
                {"asset_path": BOOM_AUDIO, "start_ms": 7200, "volume": 0.7, "loop": False},
                {"asset_path": BOOM_AUDIO, "start_ms": 11600, "volume": 0.62, "loop": False},
            ]
        )
    if heart:
        sfx.append({"asset_path": HEART_AUDIO, "start_ms": 1900, "volume": 0.58, "loop": False})
    if thunder:
        sfx.append({"asset_path": THUNDER_AUDIO, "start_ms": 2800, "volume": 0.44, "loop": False})
    return scene_audio(sfx=sfx)


def scene_bgm(scene_index: int) -> dict:
    if scene_index <= 2:
        path = PROLOGUE_BGM
        volume = 0.5
    elif scene_index <= 6:
        path = TRACE_BGM
        volume = 0.56
    elif scene_index <= 10:
        path = CHASE_BGM
        volume = 0.62
    elif scene_index <= 14:
        path = CRISIS_BGM
        volume = 0.58
    elif scene_index <= 18:
        path = FINAL_BGM
        volume = 0.64
    else:
        path = EPILOGUE_BGM
        volume = 0.5
    return {"asset_path": path, "start_ms": 0, "volume": volume, "loop": True}


def default_foregrounds(scene_index: int, background: str) -> list[dict]:
    if background in {"temple-courtyard", "theatre-stage"}:
        return [
            foreground(
                "敞开的红色帘子-窗帘或床帘皆可",
                asset_path="assets/foreground/敞开的红色帘子-窗帘或床帘皆可.webp",
                x=-0.02,
                y=-0.04,
                width=1.04,
                height=1.08,
                opacity=1.0,
            )
        ]
    if background in {"room-day", "archive-library", "town-hall-records", "inn-hall", "museum-gallery", "bank-lobby", "restaurant-booth"}:
        fg_id = "开着门的室内" if scene_index % 2 == 0 else "古典木门木窗-有点日式风格"
        fg_path = "assets/foreground/开着门的室内.webp" if fg_id == "开着门的室内" else "assets/foreground/古典木门木窗-有点日式风格.webp"
        return [foreground(fg_id, asset_path=fg_path, x=-0.01, y=-0.02, width=1.02, height=1.06, opacity=1.0)]
    if background in {"night-bridge", "park-evening", "street-day", "shop-row", "school-yard"}:
        return [
            foreground(
                "中式古典大门",
                asset_path="assets/foreground/中式古典大门.webp",
                x=-0.01,
                y=-0.02,
                width=1.02,
                height=1.06,
                opacity=1.0,
            )
        ]
    return []


def duel_beats(left_id: str, right_id: str, *, left_x: float, right_x: float, airborne: bool = False, heavy: bool = False) -> list[dict]:
    jump_z = 0.16 if airborne else 0.05
    return [
        beat(3300, 4550, left_id, "straight-punch", x0=left_x, x1=left_x + 0.4, z0=0.0, z1=jump_z, facing="right", effect="hit"),
        beat(4700, 5950, right_id, "hook-punch", x0=right_x, x1=right_x - 0.34, z0=0.0, z1=jump_z, facing="left", effect="hit"),
        beat(6400, 7800, left_id, "combo-punch" if heavy else "swing-punch", x0=left_x + 0.2, x1=left_x + 0.76, z0=jump_z, z1=0.02, facing="right", effect="dragon-palm" if heavy else "hit"),
        beat(8250, 9500, right_id, "diagonal-kick" if airborne else "spin-kick", x0=right_x - 0.14, x1=right_x - 0.74, z0=jump_z, z1=0.02, facing="left", effect="thunder-strike"),
        beat(10100, 11450, left_id, "double-palm-push", x0=left_x + 0.4, x1=left_x + 0.98, z0=0.0, z1=0.0, facing="right", effect="sword-arc"),
    ]


SCENE_SPECS = [
    SceneSpec(
        scene_id="scene-001",
        background="museum-gallery",
        summary="夜里的美术馆警报骤停，展柜中的海雾镜匣被人取走，沈砺带队赶到空空如也的展厅。",
        actors=[
            front_actor("shen_li", -2.3, facing="right"),
            front_actor("wen_xia", 0.2, facing="left"),
            mid_actor("xu_ning", 2.9, facing="left"),
        ],
        props=city_props(0, interior=True),
        lines=[
            line("wen_xia", "警报不是被拆掉的，是被人整整静音了八秒，足够把镜匣带走。"),
            line("shen_li", "馆里还有热量残留，偷东西的人离开没多久。"),
            line("xu_ning", "监控最后一帧只有一道白闪，像有人故意把画面烧掉了。"),
            line("shen_li", "从这一秒开始，全城所有出口都不能让。"),
        ],
        effects=[
            effect("aura", start_ms=2600, end_ms=5200, alpha=0.16, playback_speed=0.9),
            effect("hit", start_ms=9300, end_ms=11500, alpha=0.14, playback_speed=0.85),
        ],
        audio=city_audio(heart=True),
    ),
    SceneSpec(
        scene_id="scene-002",
        background="archive-library",
        summary="许宁调出十年前的能源档案，发现镜匣并非文物，而是能接管整座城区电网的控制核心。",
        actors=[
            front_actor("xu_ning", -2.4, facing="right"),
            front_actor("song_cheng", 0.3, facing="left"),
            mid_actor("shen_li", 2.7, facing="left"),
        ],
        props=city_props(1, interior=True),
        lines=[
            line("xu_ning", "镜匣原名天穹七号，一旦接入主网，能让半座城在十分钟内断电。"),
            line("song_cheng", "那它根本不是收藏品，是一把藏在玻璃柜里的钥匙。"),
            line("shen_li", "谁能拿到旧档案，谁就是提前知道布展路线的人。"),
            line("xu_ning", "馆方名单里只有一个内线名字，被人手工刮掉了。"),
        ],
        effects=[
            effect("aura", start_ms=2100, end_ms=4500, alpha=0.12, playback_speed=0.8),
        ],
        audio=city_audio(),
    ),
    SceneSpec(
        scene_id="scene-003",
        background="cafe-night",
        summary="闻夏在咖啡馆接触线人霍临，霍临交出一张手写坐标，指向夜桥下方的旧信号井。",
        actors=[
            front_actor("wen_xia", -2.2, facing="right"),
            front_actor("huo_lin", 2.2, facing="left"),
            back_actor("song_cheng", 3.8, facing="left"),
        ],
        props=city_props(2, interior=True),
        lines=[
            line("huo_lin", "偷镜匣的人没走高速，他们走的是断桥下面那条维修通道。"),
            line("wen_xia", "你怎么知道。"),
            line("huo_lin", "因为昨晚我听见顾远在电话里说，零点前一定要把信号井点亮。"),
            line("song_cheng", "既然你敢露面，就跟我们一起把这条线走到底。"),
        ],
        effects=[
            effect("sword-arc", start_ms=8200, end_ms=10100, alpha=0.14, playback_speed=0.9),
        ],
        audio=city_audio(heart=True),
    ),
    SceneSpec(
        scene_id="scene-004",
        background="bank-lobby",
        summary="罗诚在银行大厅查到一笔异常保管箱记录，租箱人正是消失三年的顾远。",
        actors=[
            front_actor("luo_cheng", -2.3, facing="right"),
            front_actor("shen_li", 0.1, facing="left"),
            mid_actor("tang_ji", 2.8, facing="left"),
        ],
        props=city_props(3, interior=True),
        lines=[
            line("luo_cheng", "顾远昨天用了一个旧身份取物，时间恰好卡在美术馆闭馆前。"),
            line("shen_li", "他不是来取钱，是来拿备用接口。"),
            line("tang_ji", "保安说取箱的是个戴白手套的女人，离开时连门把都没碰。"),
            line("shen_li", "乔雨上线了，她向来只做最干净的活。"),
        ],
        effects=[
            effect("hit", start_ms=9300, end_ms=11600, alpha=0.1, playback_speed=0.75),
        ],
        audio=city_audio(),
    ),
    SceneSpec(
        scene_id="scene-005",
        background="street-day",
        summary="队伍赶到街口时，乔雨正把一只银色接口箱交给林枝，沈砺当街追击，第一轮短打爆发。",
        actors=[
            front_actor("shen_li", -2.7, facing="right"),
            front_actor("qiao_yu", 0.6, facing="left"),
            mid_actor("lin_zhi", 2.9, facing="left"),
        ],
        props=city_props(4, open_space=True),
        lines=[
            line("shen_li", "乔雨，箱子放下，你今天跑不出去。"),
            line("qiao_yu", "我既然敢站在街中央，就没打算让你追上。"),
            line("lin_zhi", "桥下信号井已经预热，你们现在回头还来得及。"),
            line("shen_li", "可惜我最擅长的，就是把来不及追回来。"),
        ],
        extra_beats=duel_beats("shen_li", "qiao_yu", left_x=-2.4, right_x=0.9, heavy=False),
        effects=[
            effect("thunder-strike", start_ms=5200, end_ms=8500, alpha=0.16, playback_speed=0.88),
            effect("hit", start_ms=8700, end_ms=10900, alpha=0.18, playback_speed=0.9),
        ],
        audio=city_audio(chase=True, metal=True),
    ),
    SceneSpec(
        scene_id="scene-006",
        background="room-day",
        summary="临时作战室里，许宁拆出接口箱的地图层，发现所有线路最终都汇向城北山脊上的旧中继塔。",
        actors=[
            front_actor("xu_ning", -2.3, facing="right"),
            front_actor("wen_xia", 0.3, facing="left"),
            mid_actor("huo_lin", 2.8, facing="left"),
        ],
        props=city_props(5, interior=True),
        lines=[
            line("xu_ning", "箱子里面不是钱，是一层离线地图，最终坐标指向北山中继塔。"),
            line("wen_xia", "顾远想在高处接管主网，这样整座城区的灯都会听他的话。"),
            line("huo_lin", "塔底还有一条检修索道，够他们把镜匣运上去。"),
            line("wen_xia", "那我们得比他更快，把塔口先占住。"),
        ],
        effects=[
            effect("aura", start_ms=2500, end_ms=4700, alpha=0.14, playback_speed=0.82),
        ],
        audio=city_audio(),
    ),
    SceneSpec(
        scene_id="scene-007",
        background="town-hall-records",
        summary="宋澄在政务档案里找到被删掉的合同，签字人竟然是如今负责城市供电维护的顾远。",
        actors=[
            front_actor("song_cheng", -2.4, facing="right"),
            front_actor("shen_li", 0.0, facing="left"),
            back_actor("tang_ji", 3.2, facing="left"),
        ],
        props=city_props(6, interior=True),
        lines=[
            line("song_cheng", "这份旧合同把天穹七号列为城市应急权限核心，顾远当年就是项目负责人。"),
            line("shen_li", "难怪他要亲自回来，因为只有他知道怎么让镜匣醒过来。"),
            line("tang_ji", "更麻烦的是，他还掌握全城检修井的备用钥匙。"),
            line("shen_li", "从现在起，顾远不是盗匣犯，他是要把城市当作人质。"),
        ],
        effects=[
            effect("sword-arc", start_ms=8600, end_ms=10400, alpha=0.12, playback_speed=0.82),
        ],
        audio=city_audio(heart=True),
    ),
    SceneSpec(
        scene_id="scene-008",
        background="park-evening",
        summary="傍晚公园里，霍临正式归队，队伍按中继塔、夜桥、信号井三条线拆分行动，准备包围顾远。",
        actors=[
            front_actor("huo_lin", -2.4, facing="right"),
            front_actor("song_cheng", 0.1, facing="left"),
            mid_actor("wen_xia", 2.7, facing="left"),
        ],
        props=city_props(7, open_space=True, night=True),
        lines=[
            line("huo_lin", "顾远的人手不会多，但每一个点位都能拖我们一分钟。"),
            line("song_cheng", "那就别跟他们耗，桥线归我，闻夏守井口。"),
            line("wen_xia", "沈砺和罗诚直接上山，我负责把后路给他们锁死。"),
            line("huo_lin", "今夜不是抓人，是把整座城从黑里拽回来。"),
        ],
        effects=[
            effect("aura", start_ms=1800, end_ms=3900, alpha=0.1, playback_speed=0.8),
        ],
        audio=city_audio(),
    ),
    SceneSpec(
        scene_id="scene-009",
        background="training-ground",
        summary="上山前最后一次合练，沈砺和罗诚用近身搏击把中继塔入口的守卫动作全部预演一遍。",
        actors=[
            front_actor("shen_li", -2.5, facing="right"),
            front_actor("luo_cheng", 0.8, facing="left"),
            mid_actor("xu_ning", 3.0, facing="left"),
        ],
        props=city_props(8, open_space=True),
        lines=[
            line("luo_cheng", "塔口只有三米宽，第一下得先把人撞离扶梯。"),
            line("shen_li", "第二下我补肘，你接转身踢，把通道清出来。"),
            line("xu_ning", "别恋战，镜匣一通电，整条山脊的备用线都会冒火。"),
            line("shen_li", "明白，今晚每一拳都只为争时间。"),
        ],
        extra_beats=duel_beats("shen_li", "luo_cheng", left_x=-2.2, right_x=1.0, heavy=True),
        effects=[
            effect("dragon-palm", start_ms=6400, end_ms=9300, alpha=0.18, playback_speed=0.92),
            effect("hit", start_ms=9800, end_ms=11600, alpha=0.16, playback_speed=0.9),
        ],
        audio=city_audio(chase=True, metal=True),
    ),
    SceneSpec(
        scene_id="scene-010",
        background="night-bridge",
        summary="宋澄在桥面遭遇林枝伏击，桥下信号井同时亮起蓝光，桥线与井线的战斗一起爆开。",
        actors=[
            front_actor("song_cheng", -2.6, facing="right"),
            front_actor("lin_zhi", 0.7, facing="left"),
            mid_actor("qiao_yu", 3.0, facing="left"),
        ],
        props=city_props(9, open_space=True, night=True),
        lines=[
            line("song_cheng", "林枝，你再拖一分钟，桥下那口井就会把主网接通。"),
            line("lin_zhi", "一分钟够了，顾远只需要一段稳定信号。"),
            line("qiao_yu", "你们盯着桥，我去把井口最后一层锁打开。"),
            line("song_cheng", "那就先从你们两个身上把时间打回来。"),
        ],
        extra_beats=duel_beats("song_cheng", "lin_zhi", left_x=-2.3, right_x=0.9, airborne=True),
        effects=[
            effect("thunder-strike", start_ms=3200, end_ms=8700, alpha=0.18, playback_speed=0.9),
            effect("explosion", start_ms=10300, end_ms=12800, alpha=0.14, playback_speed=0.82),
        ],
        audio=city_audio(chase=True, boom=True, thunder=True),
    ),
    SceneSpec(
        scene_id="scene-011",
        background="restaurant-booth",
        summary="短暂回合后，队伍在路边餐馆交换情报，罗诚说塔上还有第四个人，而这个名字所有人都没料到。",
        actors=[
            front_actor("luo_cheng", -2.4, facing="right"),
            front_actor("wen_xia", 0.2, facing="left"),
            mid_actor("shen_li", 2.8, facing="left"),
        ],
        props=city_props(10, interior=True),
        lines=[
            line("luo_cheng", "中继塔的检修名单里多出一个临时通行号，登记人是唐霁。"),
            line("wen_xia", "她一直在帮我们查银行记录，为什么会在塔上。"),
            line("shen_li", "如果唐霁在塔上，说明顾远从一开始就在借我们的视线换路。"),
            line("luo_cheng", "我们原以为抓三个人，实际上要拦下的是整条链。"),
        ],
        effects=[
            effect("aura", start_ms=2600, end_ms=4600, alpha=0.1, playback_speed=0.8),
        ],
        audio=city_audio(heart=True),
    ),
    SceneSpec(
        scene_id="scene-012",
        background="temple-courtyard",
        summary="闻夏守住井口时截到顾远的广播，顾远宣布零点一到便切断城区供电，让所有人看见真正的城市秩序。",
        actors=[
            front_actor("wen_xia", -2.5, facing="right"),
            front_actor("xu_ning", 0.1, facing="left"),
            back_actor("gu_yuan", 3.3, facing="left"),
        ],
        props=city_props(11, open_space=True, night=True),
        lines=[
            line("gu_yuan", "天穹七号本来就该由我启动，是你们把它锁进了玻璃柜里。"),
            line("wen_xia", "你不是要修城市，你只是想证明没有你这座城就会熄火。"),
            line("xu_ning", "顾远，主网一旦强切，医院和地铁都会同时停摆。"),
            line("gu_yuan", "那就让他们记住，谁才是真正握着开关的人。"),
        ],
        effects=[
            effect("aura", start_ms=1400, end_ms=4200, alpha=0.18, playback_speed=0.88),
            effect("thunder-strike", start_ms=9800, end_ms=12200, alpha=0.15, playback_speed=0.84),
        ],
        audio=city_audio(boom=True, heart=True),
    ),
    SceneSpec(
        scene_id="scene-013",
        background="theatre-stage",
        summary="队伍借一场闭馆演出潜入剧场后场，唐霁正在用舞台总电闸给山顶中继塔做最后一次模拟通电。",
        actors=[
            front_actor("tang_ji", -2.2, facing="right"),
            front_actor("huo_lin", 0.4, facing="left"),
            mid_actor("shen_li", 2.8, facing="left"),
        ],
        props=city_props(12, interior=True),
        lines=[
            line("tang_ji", "我只是在做测试，真正的通电要等顾远发话。"),
            line("huo_lin", "你拿一座剧场试电，等于提前让半条街冒火。"),
            line("shen_li", "把接线板放下，别逼我在台上跟你动手。"),
            line("tang_ji", "你如果想拦，现在就来。"),
        ],
        extra_beats=duel_beats("huo_lin", "tang_ji", left_x=0.2, right_x=-1.9, heavy=False),
        effects=[
            effect("sword-arc", start_ms=3500, end_ms=6200, alpha=0.14, playback_speed=0.9),
            effect("thunder-strike", start_ms=8150, end_ms=11050, alpha=0.16, playback_speed=0.9),
        ],
        audio=city_audio(chase=True, metal=True, thunder=True),
    ),
    SceneSpec(
        scene_id="scene-014",
        background="inn-hall",
        summary="剧场后的旧客栈通道忽然落闸，乔雨从暗门杀回，沈砺与闻夏在狭窄木廊里同时迎敌。",
        actors=[
            front_actor("shen_li", -2.7, facing="right"),
            front_actor("wen_xia", -0.4, facing="right"),
            front_actor("qiao_yu", 2.2, facing="left"),
        ],
        props=city_props(13, interior=True),
        lines=[
            line("qiao_yu", "顾远早就说过，你们一定会顺着剧场摸到这条暗廊。"),
            line("wen_xia", "可他没说，暗廊里等着你的会是两个人。"),
            line("shen_li", "你守右边，我逼她退到门框上。"),
            line("qiao_yu", "试试吧，我今天也想看看你们配合得有多快。"),
        ],
        extra_beats=[
            *duel_beats("shen_li", "qiao_yu", left_x=-2.4, right_x=2.0, heavy=True),
            beat(5000, 6600, "wen_xia", "flying-kick", x0=-0.5, x1=0.5, z0=0.0, z1=0.14, facing="right", effect="hit"),
        ],
        effects=[
            effect("dragon-palm", start_ms=6100, end_ms=9300, alpha=0.18, playback_speed=0.92),
            effect("explosion", start_ms=10800, end_ms=13000, alpha=0.12, playback_speed=0.8),
        ],
        audio=city_audio(chase=True, metal=True, boom=True),
    ),
    SceneSpec(
        scene_id="scene-015",
        background="bank-lobby",
        summary="顾远反手切掉银行周边街区的照明，黑暗中的大厅变成临时人质点，罗诚和宋澄强行破入控制区。",
        actors=[
            front_actor("luo_cheng", -2.5, facing="right"),
            front_actor("song_cheng", -0.1, facing="right"),
            front_actor("gu_yuan", 2.4, facing="left"),
        ],
        props=city_props(14, interior=True),
        lines=[
            line("gu_yuan", "看见了吗，只要我动一根线，这里的人就只能站在原地等。"),
            line("luo_cheng", "你以为黑下来的是大厅，其实是你最后的退路。"),
            line("song_cheng", "主控柜后面就是总闸，给我三秒，我把它撬开。"),
            line("gu_yuan", "那你们就先问问我答不答应。"),
        ],
        extra_beats=[
            *duel_beats("luo_cheng", "gu_yuan", left_x=-2.2, right_x=2.2, airborne=True, heavy=True),
            beat(5600, 7200, "song_cheng", "point", facing="right", emotion="charged"),
        ],
        effects=[
            effect("thunder-strike", start_ms=3800, end_ms=7600, alpha=0.18, playback_speed=0.88),
            effect("explosion", start_ms=10100, end_ms=12600, alpha=0.16, playback_speed=0.84),
        ],
        audio=city_audio(chase=True, metal=True, boom=True, thunder=True),
    ),
    SceneSpec(
        scene_id="scene-016",
        background="shop-row",
        summary="控制区被破后，林枝带着镜匣半成品冲向山道，霍临与闻夏沿着商铺长街一路追砍封路。",
        actors=[
            front_actor("lin_zhi", -2.2, facing="right"),
            front_actor("huo_lin", 0.7, facing="left"),
            mid_actor("wen_xia", 3.0, facing="left"),
        ],
        props=city_props(15, open_space=True, night=True),
        lines=[
            line("lin_zhi", "镜匣已经点火，你们现在抢回去也只是一块发热的空壳。"),
            line("huo_lin", "空壳也比放在顾远手里强。"),
            line("wen_xia", "前面路口别让她拐，我从侧墙切过去。"),
            line("lin_zhi", "那就看你们追得快，还是我扔得快。"),
        ],
        extra_beats=[
            beat(3200, 4700, "lin_zhi", "exit", x0=-2.1, x1=-0.8, facing="right", emotion="charged"),
            beat(4800, 6200, "huo_lin", "big-jump", x0=0.7, x1=0.1, z0=0.0, z1=0.18, facing="left", effect="thunder-strike"),
            beat(7700, 9300, "wen_xia", "flying-kick", x0=2.8, x1=1.2, z0=0.0, z1=0.16, facing="left", effect="hit"),
            beat(9800, 11600, "lin_zhi", "spin-kick", x0=-0.7, x1=-1.5, z0=0.12, z1=0.0, facing="right", effect="sword-arc"),
        ],
        effects=[
            effect("hit", start_ms=7700, end_ms=9800, alpha=0.16, playback_speed=0.9),
            effect("sword-arc", start_ms=9900, end_ms=11800, alpha=0.15, playback_speed=0.88),
        ],
        audio=city_audio(chase=True, metal=True),
    ),
    SceneSpec(
        scene_id="scene-017",
        background="museum-gallery",
        summary="众人折返美术馆确认真匣，却发现展柜里留下的是诱饵，真正的海雾镜匣已被顾远装上山顶索道。",
        actors=[
            front_actor("xu_ning", -2.4, facing="right"),
            front_actor("song_cheng", 0.1, facing="left"),
            mid_actor("shen_li", 2.8, facing="left"),
        ],
        props=city_props(16, interior=True),
        lines=[
            line("xu_ning", "这是仿品，里面只有散热片，没有主控核心。"),
            line("song_cheng", "顾远故意让我们在城里兜圈，他真正要去的还是北山塔顶。"),
            line("shen_li", "那就不再拆线了，所有人一起上山，最后一关在塔上解决。"),
            line("xu_ning", "我把主网延时拖到十分钟，十分钟后就只能靠你们硬抢。"),
        ],
        effects=[
            effect("aura", start_ms=2400, end_ms=4700, alpha=0.14, playback_speed=0.8),
            effect("explosion", start_ms=11100, end_ms=13200, alpha=0.12, playback_speed=0.78),
        ],
        audio=city_audio(heart=True, boom=True),
    ),
    SceneSpec(
        scene_id="scene-018",
        background="mountain-cliff",
        summary="北山风口，中继塔蓝火翻滚，顾远把真正的镜匣推入主接口，山脊上方的备用电网开始整片发亮。",
        actors=[
            front_actor("gu_yuan", -2.2, facing="right"),
            front_actor("shen_li", 0.4, facing="left"),
            mid_actor("luo_cheng", 2.9, facing="left"),
        ],
        props=city_props(17, open_space=True, night=True),
        lines=[
            line("gu_yuan", "你们还是慢了一步，塔一旦点满，整座城的灯都会按我的节奏呼吸。"),
            line("shen_li", "那我就在你按下最后一步之前，把你从接口前拖下来。"),
            line("luo_cheng", "风口线太乱，沈砺，我断左边，你直取镜匣。"),
            line("gu_yuan", "来吧，让我看看你们拿什么拦住一座亮起来的山。"),
        ],
        extra_beats=[
            *duel_beats("shen_li", "gu_yuan", left_x=0.2, right_x=-2.0, airborne=True, heavy=True),
            beat(5400, 7200, "luo_cheng", "thunder-strike", x0=2.8, x1=1.8, z0=0.0, z1=0.18, facing="left", effect="thunder-strike"),
        ],
        effects=[
            effect("aura", start_ms=1200, end_ms=4800, alpha=0.18, playback_speed=0.86),
            effect("thunder-strike", start_ms=5200, end_ms=9400, alpha=0.18, playback_speed=0.9),
            effect("explosion", start_ms=10300, end_ms=12900, alpha=0.16, playback_speed=0.84),
        ],
        audio=city_audio(chase=True, metal=True, boom=True, thunder=True),
    ),
    SceneSpec(
        scene_id="scene-019",
        background="night-bridge",
        summary="镜匣被抛离塔口后沿桥线滑落，闻夏和许宁在桥面完成人工断链，整座城区在最后一秒保住主网。",
        actors=[
            front_actor("wen_xia", -2.4, facing="right"),
            front_actor("xu_ning", 0.2, facing="left"),
            mid_actor("huo_lin", 2.8, facing="left"),
        ],
        props=city_props(18, open_space=True, night=True),
        lines=[
            line("wen_xia", "镜匣落桥了，我能按住外壳，许宁，你现在切最后一根链。"),
            line("xu_ning", "切了以后桥面会跳火，你们全部退到护栏外侧。"),
            line("huo_lin", "别废话了，灯已经在抖，再慢一秒全城都得陪着黑。"),
            line("wen_xia", "那就现在，给它断。"),
        ],
        extra_beats=[
            beat(3300, 4700, "wen_xia", "big-jump", x0=-2.1, x1=-1.0, z0=0.0, z1=0.2, facing="right", effect="thunder-strike"),
            beat(5200, 6900, "huo_lin", "point", facing="left", emotion="charged"),
            beat(7900, 9600, "xu_ning", "double-palm-push", x0=0.1, x1=-0.4, z0=0.0, z1=0.0, facing="left", effect="sword-arc"),
            beat(10100, 11700, "wen_xia", "exit", x0=-1.0, x1=-1.8, facing="left", emotion="charged"),
        ],
        effects=[
            effect("thunder-strike", start_ms=2900, end_ms=6200, alpha=0.18, playback_speed=0.88),
            effect("sword-arc", start_ms=7900, end_ms=10200, alpha=0.14, playback_speed=0.86),
            effect("explosion", start_ms=10400, end_ms=12800, alpha=0.12, playback_speed=0.8),
        ],
        audio=city_audio(chase=True, boom=True, thunder=True),
    ),
    SceneSpec(
        scene_id="scene-020",
        background="park-evening",
        summary="天亮前的公园仍有余电在草地边缘闪烁，众人确认主网恢复，海雾镜匣被重新封存，这一夜终于过去。",
        actors=[
            front_actor("shen_li", -2.4, facing="right"),
            front_actor("wen_xia", 0.0, facing="left"),
            mid_actor("xu_ning", 2.5, facing="left"),
            back_actor("huo_lin", 3.7, facing="left"),
        ],
        props=city_props(19, open_space=True, night=True),
        lines=[
            line("xu_ning", "主网恢复了，医院、地铁和北区配电房都重新亮起来了。"),
            line("wen_xia", "镜匣外壳已经锁死，顾远那套接口再也接不上去。"),
            line("shen_li", "这一夜够长，但总算把灯守住了。"),
            line("huo_lin", "城醒了，我们也该回去睡一觉了。"),
        ],
        effects=[
            effect("aura", start_ms=1800, end_ms=4200, alpha=0.1, playback_speed=0.78),
            effect("hit", start_ms=11000, end_ms=13100, alpha=0.08, playback_speed=0.72),
        ],
        audio=city_audio(),
    ),
]


class FaceEnsembleShadowChaseVideo(BaseVideoScript):
    def get_title(self) -> str:
        return "夜桥追光行动"

    def get_theme(self) -> str:
        return "群像追缉、都市断电危机、桥面追逐、山顶决战、多人接力"

    def get_cast(self) -> list[dict]:
        return CAST

    def get_video_options(self) -> dict:
        return VIDEO

    def has_tts(self) -> bool:
        return True

    def get_notes(self) -> dict:
        return {
            "scene_count": len(SCENE_SPECS),
            "format": "face-ensemble-shadow-chase",
            "bgm_assets": [PROLOGUE_BGM, TRACE_BGM, CHASE_BGM, CRISIS_BGM, FINAL_BGM, EPILOGUE_BGM],
            "featured_effects": ["aura", "hit", "dragon-palm", "sword-arc", "thunder-strike", "explosion"],
            "featured_characters": [member["asset_id"] for member in CAST],
        }

    def get_default_output(self) -> str:
        return "outputs/face_ensemble_shadow_chase.mp4"

    def get_description(self) -> str:
        return "Render a 20-scene ensemble city-crisis story built around the extracted face-* characters, with layered BGM, action SFX, and Panda3D effects."

    def get_scenes(self) -> list[dict]:
        scenes: list[dict] = []
        for scene_index, spec in enumerate(SCENE_SPECS):
            dialogue_items, talk_beats, expressions_track = build_dialogue_bundle(spec.lines)
            talk_beats = trim_talk_beats_for_actions(talk_beats, spec.extra_beats)
            beats = sorted([*talk_beats, *spec.extra_beats], key=lambda item: (item["start_ms"], item["actor_id"]))
            expressions_sorted = sorted(expressions_track, key=lambda item: (item["start_ms"], item["actor_id"]))
            battle = bool(spec.extra_beats or spec.effects)
            rooftop = spec.background in {"mountain-cliff", "night-bridge"}
            audio_payload = scene_audio(
                bgm=scene_bgm(scene_index),
                sfx=list(spec.audio.get("sfx", [])),
            )
            scenes.append(
                scene(
                    spec.scene_id,
                    background=spec.background,
                    floor=FLOOR_BY_BACKGROUND[spec.background],
                    duration_ms=SCENE_DURATION_MS,
                    summary=spec.summary,
                    camera=spec.camera or scene_camera(scene_index, battle=battle, rooftop=rooftop),
                    effects=spec.effects,
                    foregrounds=[*default_foregrounds(scene_index, spec.background), *spec.foregrounds],
                    props=spec.props,
                    actors=spec.actors,
                    beats=beats,
                    expressions=expressions_sorted,
                    dialogues=dialogue_items,
                    audio=audio_payload,
                )
            )
        return scenes


SCRIPT = FaceEnsembleShadowChaseVideo()


def build_story() -> dict:
    return SCRIPT.build_story()


def main() -> int:
    return SCRIPT()


if __name__ == "__main__":
    raise SystemExit(main())
