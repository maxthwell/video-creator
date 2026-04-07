#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
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
        "effect_overlay_alpha": 0.9,
    },
}

CAST = [
    cast_member("wu_kong", "孙悟空", "young-hero"),
    cast_member("er_lang", "二郎神", "emperor-ming"),
    cast_member("lao_jun", "太上老君", "farmer-old"),
    cast_member("narrator", "旁白", "narrator"),
    cast_member("tian_jiang", "天将", "general-guard"),
    cast_member("mei_shan", "梅山兄弟", "npc-boy"),
]

SCENE_DURATION_MS = 15_000
DIALOGUE_WINDOWS = [
    (400, 2900),
    (3500, 6100),
    (7300, 10000),
    (10900, 13800),
]

FLOOR_BY_BACKGROUND = {
    "mountain-cliff": "stone-court",
    "night-bridge": "dark-stage",
    "park-evening": "dark-stage",
    "temple-courtyard": "stone-court",
    "theatre-stage": "dark-stage",
    "training-ground": "stone-court",
}

FIST_AUDIO = "assets/audio/031_26_赤手空拳打斗的声音_爱给网_aigei_com.mp3"
METAL_AUDIO = "assets/audio/刀剑、金属碰撞（带回音）_爱给网_aigei_com.mp3"
BOOM_AUDIO = "assets/audio/音效 爆炸 爆破 爆发 战斗_爱给网_aigei_com.mp3"
THUNDER_AUDIO = "assets/audio/打雷闪电.wav"
DOG_AUDIO = "assets/audio/狗喘粗气.wav"

HEAVEN_BGM = "assets/bgm/天府乐-许镜清.mp3"
FLIGHT_BGM = "assets/bgm/御剑飞行.mp3"
DUEL_BGM = "assets/bgm/最后之战-热血-卢冠廷.mp3"
PRESSURE_BGM = "assets/bgm/观音降临-高潮版.mp3"
EPILOGUE_BGM = "assets/bgm/芦苇荡-赵季平-大话西游.mp3"

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


def mid_actor(actor_id: str, x: float, *, facing: str, scale: float = 0.92, z: float = -0.14) -> dict:
    return actor(actor_id, x, z=z, facing=facing, scale=scale, layer="mid")


def back_actor(actor_id: str, x: float, *, facing: str, scale: float = 0.84, z: float = -0.72) -> dict:
    return actor(actor_id, x, z=z, facing=facing, scale=scale, layer="back")


def infer_expression(text: str) -> str:
    if any(token in text for token in ("打", "斩", "压", "破", "拿", "咬", "擒", "退")):
        return "angry"
    if any(token in text for token in ("快", "来", "看招", "休走", "再来")):
        return "excited"
    if any(token in text for token in ("变", "法", "算", "眼", "机会", "围住")):
        return "thinking"
    if any(token in text for token in ("笑", "有趣", "不差")):
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


def scene_camera(scene_index: int, *, battle: bool, aerial: bool = False) -> dict:
    if aerial:
        return camera_pan(
            x=-0.30,
            z=0.07,
            zoom=1.10,
            to_x=0.24,
            to_z=0.02,
            to_zoom=1.20,
            ease="ease-in-out",
        )
    if battle:
        return camera_pan(
            x=-0.24 + 0.04 * (scene_index % 2),
            z=0.04,
            zoom=1.08,
            to_x=0.18 - 0.03 * (scene_index % 3),
            to_z=0.0,
            to_zoom=1.16,
            ease="ease-in-out",
        )
    if scene_index in {0, 9}:
        return camera_static(x=0.0, z=0.03, zoom=1.08)
    return camera_pan(
        x=-0.16,
        z=0.02,
        zoom=1.0,
        to_x=0.12,
        to_z=0.0,
        to_zoom=1.06,
        ease="ease-in-out",
    )


def myth_props(scene_index: int, *, celestial: bool = False, night: bool = False, dog: bool = False) -> list[dict]:
    items: list[dict] = []
    if celestial:
        items.extend(
            [
                prop("training-drum", -3.5, -1.02, scale=0.94, layer="back"),
                prop("weapon-rack", 3.4, -1.0, scale=0.94, layer="mid"),
                prop("lantern", -0.2, -0.92, scale=0.98, layer="front"),
            ]
        )
    if night:
        items.extend(
            [
                prop("moon", 3.7, -0.42, scale=0.74, layer="back"),
                prop("star", -3.7, -0.56, scale=0.55, layer="back"),
            ]
        )
    if dog:
        items.append(prop("dog", 3.1 if scene_index % 2 == 0 else -3.1, -1.0, scale=0.78, layer="front"))
    return items


def story_audio(*, metal: bool = False, boom: bool = False, thunder: bool = False, dog: bool = False) -> dict:
    sfx = [
        audio_sfx(FIST_AUDIO, start_ms=3900, volume=0.76),
        audio_sfx(FIST_AUDIO, start_ms=7600, volume=0.74),
    ]
    if metal:
        sfx.append(audio_sfx(METAL_AUDIO, start_ms=5600, volume=0.70))
        sfx.append(audio_sfx(METAL_AUDIO, start_ms=9800, volume=0.66))
    if boom:
        sfx.append(audio_sfx(BOOM_AUDIO, start_ms=6900, volume=0.64))
        sfx.append(audio_sfx(BOOM_AUDIO, start_ms=11200, volume=0.58))
    if thunder:
        sfx.append(audio_sfx(THUNDER_AUDIO, start_ms=3100, volume=0.58))
    if dog:
        sfx.append(audio_sfx(DOG_AUDIO, start_ms=9600, volume=0.54))
    return scene_audio(sfx=sfx)


def scene_bgm(scene_index: int) -> dict:
    if scene_index <= 1:
        return audio_bgm(HEAVEN_BGM, volume=0.48, loop=True)
    if scene_index <= 3:
        return audio_bgm(FLIGHT_BGM, volume=0.56, loop=True)
    if scene_index <= 7:
        return audio_bgm(DUEL_BGM, volume=0.64, loop=True)
    if scene_index == 8:
        return audio_bgm(PRESSURE_BGM, volume=0.58, loop=True)
    return audio_bgm(EPILOGUE_BGM, volume=0.48, loop=True)


def default_foregrounds(background: str) -> list[dict]:
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
    return []


def duel_beats(
    left_id: str,
    right_id: str,
    *,
    left_x: float,
    right_x: float,
    airborne: bool = False,
    heavy: bool = False,
) -> list[dict]:
    jump_z = 0.18 if airborne else 0.06
    return [
        beat(3300, 4500, left_id, "somersault" if airborne else "straight-punch", x0=left_x, x1=left_x + 0.45, z0=0.0, z1=jump_z, facing="right", effect="thunder-strike" if airborne else "hit"),
        beat(4700, 5900, right_id, "hook-punch", x0=right_x, x1=right_x - 0.40, z0=0.0, z1=jump_z, facing="left", effect="hit"),
        beat(6400, 7800, left_id, "combo-punch" if heavy else "swing-punch", x0=left_x + 0.25, x1=left_x + 0.90, z0=jump_z, z1=0.04, facing="right", effect="dragon-palm"),
        beat(8200, 9500, right_id, "spin-kick", x0=right_x - 0.18, x1=right_x - 0.82, z0=jump_z, z1=0.02, facing="left", effect="sword-arc"),
        beat(10100, 11400, left_id, "double-palm-push", x0=left_x + 0.55, x1=left_x + 1.18, z0=0.02, z1=0.06, facing="right", effect="dragon-palm" if heavy else "sword-arc"),
    ]


SCENE_SPECS = [
    SceneSpec(
        scene_id="scene-001",
        background="temple-courtyard",
        summary="花果山妖气冲天，天庭议战，二郎神主动请缨下界擒拿孙悟空。",
        actors=[
            front_actor("er_lang", -2.0, facing="right"),
            front_actor("lao_jun", 0.8, facing="left"),
            back_actor("narrator", 3.5, facing="left"),
        ],
        props=myth_props(0, celestial=True),
        lines=[
            line("narrator", "花果山外旌旗震动，天庭上下都知道，这一战非寻常天将可解。"),
            line("er_lang", "玉帝若要拿那泼猴，我去，正好看看他七十二变有多少真章。"),
            line("lao_jun", "此猴筋骨奇硬，切莫只图快胜，先逼出他的路数再收网。"),
            line("er_lang", "老君放心，我先与他明战一场，让三界都看个明白。"),
        ],
        effects=[
            effect("风起云涌", start_ms=200, end_ms=5200, alpha=0.14, playback_speed=0.92),
            effect("英雄出场", start_ms=900, end_ms=3000, alpha=0.16, playback_speed=0.92),
        ],
        audio=story_audio(),
    ),
    SceneSpec(
        scene_id="scene-002",
        background="mountain-cliff",
        summary="二郎神降临花果山顶，孙悟空擎棒迎战，两人尚未出手，山风已像刀一样卷起。",
        actors=[
            front_actor("wu_kong", -2.0, facing="right"),
            front_actor("er_lang", 1.9, facing="left"),
            back_actor("mei_shan", 3.6, facing="left"),
        ],
        props=myth_props(1, night=False),
        lines=[
            line("wu_kong", "二郎神，你若只是奉旨来吓人，就回去，老孙的花果山不吃这一套。"),
            line("er_lang", "你闹过天宫，打过天将，如今轮到我来看看你还能狂到几时。"),
            line("wu_kong", "说得好听，真有本事，先接我金箍棒一回。"),
            line("er_lang", "正合我意，你我今日只论胜负，不论嘴上高低。"),
        ],
        effects=[effect("英雄出场", start_ms=600, end_ms=2600, alpha=0.16, playback_speed=0.92)],
        audio=story_audio(metal=True),
    ),
    SceneSpec(
        scene_id="scene-003",
        background="park-evening",
        summary="两人踏云升空，金箍棒与三尖两刃刀在半空连撞，招招都逼着对方后退半步。",
        actors=[
            front_actor("wu_kong", -1.9, facing="right"),
            front_actor("er_lang", 1.9, facing="left"),
            back_actor("narrator", 3.5, facing="left"),
        ],
        props=myth_props(2, night=True),
        lines=[
            line("narrator", "一上高空，二人便像两道撕开的电光，棒影刀光连成一片。"),
            line("wu_kong", "来得好，老孙最爱这种不躲不闪的对手。"),
            line("er_lang", "你筋斗虽快，我刀势也不慢，想从我眼前脱身没有那么容易。"),
            line("wu_kong", "那你就跟紧些，看我把这天风都踩碎。"),
        ],
        extra_beats=duel_beats("wu_kong", "er_lang", left_x=-1.8, right_x=1.8, airborne=True),
        effects=[
            effect("御剑飞行", start_ms=300, end_ms=9800, alpha=0.18, playback_speed=0.95),
            effect("风起云涌", start_ms=9800, end_ms=14500, alpha=0.14, playback_speed=0.92),
        ],
        audio=story_audio(metal=True, boom=True),
        camera=scene_camera(2, battle=True, aerial=True),
    ),
    SceneSpec(
        scene_id="scene-004",
        background="night-bridge",
        summary="孙悟空忽地连翻筋斗施展七十二变，二郎神却凭天眼一路咬住，半点不失。",
        actors=[
            front_actor("wu_kong", -2.1, facing="right"),
            front_actor("er_lang", 1.8, facing="left"),
            back_actor("narrator", 3.6, facing="left"),
        ],
        props=myth_props(3, night=True),
        lines=[
            line("wu_kong", "老孙一变十变百变，你若认得出真身，再来夸口不迟。"),
            line("er_lang", "你变得再快，我这只天眼也盯得住，你休想借假形脱走。"),
            line("narrator", "桥上桥下都是残影，一个翻云，一个逐影，竟像两阵风在互相撕咬。"),
            line("wu_kong", "好个二郎神，眼倒是毒，再吃我一轮翻身棍！"),
        ],
        extra_beats=[
            beat(3400, 4700, "wu_kong", "somersault", x0=-2.0, x1=-1.0, z0=0.02, z1=0.24, facing="right", effect="thunder-strike"),
            beat(5000, 6300, "er_lang", "big-jump", x0=1.7, x1=0.7, z0=0.0, z1=0.20, facing="left", effect="sword-arc"),
            beat(7600, 9000, "wu_kong", "flying-kick", x0=-0.9, x1=0.2, z0=0.18, z1=0.10, facing="right", effect="飞踢"),
            beat(9800, 11200, "er_lang", "double-palm-push", x0=0.7, x1=-0.2, z0=0.08, z1=0.04, facing="left", effect="dragon-palm"),
        ],
        effects=[
            effect("御剑飞行", start_ms=300, end_ms=8200, alpha=0.18, playback_speed=0.95),
            effect("风起云涌", start_ms=2000, end_ms=14500, alpha=0.14, playback_speed=0.92),
            effect("飞踢", start_ms=7600, end_ms=9200, alpha=0.18, playback_speed=0.95),
        ],
        audio=story_audio(metal=True, boom=True, thunder=True),
        camera=scene_camera(3, battle=True, aerial=True),
    ),
    SceneSpec(
        scene_id="scene-005",
        background="training-ground",
        summary="僵持之中二郎神忽开天眼，神光直照孙悟空，逼得悟空横棒硬架，火花一路炸开。",
        actors=[
            front_actor("er_lang", -1.9, facing="right"),
            front_actor("wu_kong", 1.8, facing="left"),
            back_actor("mei_shan", 3.5, facing="left"),
        ],
        props=myth_props(4, celestial=True),
        lines=[
            line("er_lang", "我不与你拖了，这一道神光专照妖形，看你怎么遮。"),
            line("wu_kong", "照就照，老孙顶天立地，怕你一只眼么。"),
            line("mei_shan", "真君神光压下去了，那猴头脚下的石地都在裂。"),
            line("wu_kong", "裂得好，越裂越说明你这一招还压不住我。"),
        ],
        extra_beats=[
            beat(3500, 4900, "er_lang", "double-palm-push", x0=-1.8, x1=-0.8, z0=0.0, z1=0.06, facing="right", effect="死亡光线特效"),
            beat(5600, 7000, "wu_kong", "straight-punch", x0=1.6, x1=0.8, z0=0.0, z1=0.10, facing="left", effect="hit"),
            beat(8200, 9800, "wu_kong", "combo-punch", x0=0.7, x1=-0.1, z0=0.08, z1=0.02, facing="left", effect="dragon-palm"),
        ],
        effects=[
            effect("启动大招特效", start_ms=2500, end_ms=4400, alpha=0.16, playback_speed=0.92),
            effect("死亡光线特效", start_ms=3600, end_ms=9800, alpha=0.18, playback_speed=0.96),
            effect("爆炸特效", start_ms=9800, end_ms=11800, alpha=0.18, playback_speed=0.92),
        ],
        audio=story_audio(metal=True, boom=True, thunder=True),
    ),
    SceneSpec(
        scene_id="scene-006",
        background="theatre-stage",
        summary="孙悟空怒起法力，棒势越抡越大，逼得二郎神连换三路身法，台前台后都是炸开的气浪。",
        actors=[
            front_actor("wu_kong", -1.9, facing="right"),
            front_actor("er_lang", 1.9, facing="left"),
            back_actor("narrator", 3.5, facing="left"),
        ],
        props=myth_props(5, celestial=True),
        lines=[
            line("wu_kong", "接住了天眼，再接我这一路法天象地的棍势。"),
            line("er_lang", "力道确实惊人，可你每抡大一分，破绽也就跟着露一分。"),
            line("narrator", "这一回不是单拼快慢，而是硬碰硬地比谁能先把对方压弯。"),
            line("wu_kong", "破绽若真在，你便来取，老孙正嫌你退得不够近。"),
        ],
        extra_beats=duel_beats("wu_kong", "er_lang", left_x=-1.8, right_x=1.9, heavy=True),
        effects=[
            effect("启动大招特效", start_ms=2400, end_ms=4700, alpha=0.16, playback_speed=0.92),
            effect("龟派气功", start_ms=5200, end_ms=9800, alpha=0.16, playback_speed=0.94),
            effect("命中特效", start_ms=10400, end_ms=12200, alpha=0.18, playback_speed=0.92),
        ],
        audio=story_audio(metal=True, boom=True),
    ),
    SceneSpec(
        scene_id="scene-007",
        background="mountain-cliff",
        summary="大战拖到山巅，梅山兄弟和哮天犬逼近策应，孙悟空边战边退，局势第一次向二郎神倾斜。",
        actors=[
            front_actor("wu_kong", -1.9, facing="right"),
            front_actor("er_lang", 1.9, facing="left"),
            back_actor("mei_shan", 3.3, facing="left"),
        ],
        props=myth_props(6, night=True, dog=True),
        lines=[
            line("mei_shan", "真君，我们已经封住后坡，那猴头这回没有借势翻走的路。"),
            line("er_lang", "别急着上，只管逼住他的脚步，真正收招的人是我。"),
            line("wu_kong", "围得倒快，可惜靠人多拿不下老孙，终究还得看你自己。"),
            line("er_lang", "正合我意，我也不愿别人抢了这场胜负。"),
        ],
        extra_beats=[
            beat(3500, 4800, "wu_kong", "somersault", x0=-1.8, x1=-0.8, z0=0.02, z1=0.24, facing="right", effect="thunder-strike"),
            beat(5200, 6500, "er_lang", "hook-punch", x0=1.8, x1=0.9, z0=0.08, z1=0.14, facing="left", effect="hit"),
            beat(7600, 9000, "wu_kong", "double-palm-push", x0=-0.7, x1=0.3, z0=0.08, z1=0.06, facing="right", effect="dragon-palm"),
            beat(9800, 11200, "er_lang", "spin-kick", x0=0.8, x1=0.0, z0=0.12, z1=0.02, facing="left", effect="sword-arc"),
        ],
        effects=[
            effect("风起云涌", start_ms=200, end_ms=14800, alpha=0.14, playback_speed=0.92),
            effect("御剑飞行", start_ms=3000, end_ms=8600, alpha=0.16, playback_speed=0.95),
        ],
        audio=story_audio(metal=True, boom=True, dog=True),
        camera=scene_camera(6, battle=True, aerial=True),
    ),
    SceneSpec(
        scene_id="scene-008",
        background="park-evening",
        summary="两人仍旧难分高下，云头忽有金光破空而下，太上老君看准一瞬，暗中祭出金刚琢。",
        actors=[
            front_actor("lao_jun", -2.2, facing="right"),
            front_actor("er_lang", 0.6, facing="left"),
            back_actor("narrator", 3.5, facing="left"),
        ],
        props=myth_props(7, night=True),
        lines=[
            line("narrator", "眼看二郎神与孙悟空杀得平分秋色，云上忽有一道冷金之光无声转下。"),
            line("lao_jun", "这猴头太硬，不借外力难收，贫道这一琢只取他一瞬失手。"),
            line("er_lang", "机会只有这一回，我若再放过，他又要翻云而走。"),
            line("narrator", "话音未落，那金刚琢已从云顶砸下，直取悟空后脑。"),
        ],
        effects=[
            effect("英雄出场", start_ms=800, end_ms=2600, alpha=0.15, playback_speed=0.92),
            effect("命中特效", start_ms=9600, end_ms=10800, alpha=0.18, playback_speed=0.92),
            effect("爆炸特效", start_ms=10400, end_ms=12600, alpha=0.16, playback_speed=0.92),
        ],
        audio=story_audio(boom=True, thunder=True),
    ),
    SceneSpec(
        scene_id="scene-009",
        background="mountain-cliff",
        summary="孙悟空被金刚琢打得一个踉跄，哮天犬趁势扑上咬住他的腿弯，二郎神终于抢得擒拿先机。",
        actors=[
            front_actor("wu_kong", -1.9, facing="right"),
            front_actor("er_lang", 1.8, facing="left"),
            back_actor("tian_jiang", 3.4, facing="left"),
        ],
        props=myth_props(8, night=True, dog=True),
        lines=[
            line("wu_kong", "好个暗手，云上砸下一环，倒真叫老孙失了半步。"),
            line("er_lang", "胜负只争这一瞬，哮天犬，上！"),
            line("tian_jiang", "咬住了，真君已经压上去，那猴头再翻身就迟了！"),
            line("wu_kong", "今日不是输你一人，是输在天上地下都来围我。"),
        ],
        extra_beats=[
            beat(3400, 4700, "wu_kong", "big-jump", x0=-1.8, x1=-0.9, z0=0.0, z1=0.18, facing="right", effect="命中特效"),
            beat(5200, 6600, "er_lang", "straight-punch", x0=1.6, x1=0.7, z0=0.0, z1=0.10, facing="left", effect="hit"),
            beat(7600, 9300, "er_lang", "combo-punch", x0=0.6, x1=-0.1, z0=0.10, z1=0.04, facing="left", effect="thunder-strike"),
            beat(9800, 11300, "er_lang", "double-palm-push", x0=0.0, x1=-0.8, z0=0.04, z1=0.0, facing="left", effect="dragon-palm"),
        ],
        effects=[
            effect("命中特效", start_ms=3200, end_ms=5200, alpha=0.16, playback_speed=0.92),
            effect("飞踢", start_ms=7600, end_ms=9200, alpha=0.16, playback_speed=0.95),
            effect("爆炸特效", start_ms=10800, end_ms=12800, alpha=0.16, playback_speed=0.92),
        ],
        audio=story_audio(boom=True, thunder=True, dog=True),
    ),
    SceneSpec(
        scene_id="scene-010",
        background="temple-courtyard",
        summary="大战落幕，孙悟空终被押回天庭，但二郎神与他正面对决许久不分胜负，也让这一战成了三界名局。",
        actors=[
            front_actor("narrator", -2.3, facing="right", scale=0.90),
            front_actor("er_lang", 0.5, facing="left"),
            back_actor("lao_jun", 2.9, facing="left"),
        ],
        props=myth_props(9, celestial=True),
        lines=[
            line("narrator", "这一战打到最后，孙悟空虽被擒下，可他与二郎神正面对杀许久，始终不曾轻易服输。"),
            line("er_lang", "若只论你我单斗，你确是少见的强手。"),
            line("lao_jun", "猴头桀骜，真君神勇，此战之后，三界再无人敢轻看花果山那一位。"),
            line("narrator", "于是孙悟空大战二郎神，便成了《西游记》中最令人难忘的一场斗法。"),
        ],
        effects=[effect("英雄出场", start_ms=700, end_ms=2400, alpha=0.14, playback_speed=0.92)],
        audio=story_audio(),
    ),
]


class SunWukongVsErlangVideo(BaseVideoScript):
    def get_title(self) -> str:
        return "西游记之孙悟空大战二郎神"

    def get_theme(self) -> str:
        return "西游神话、花果山大战、云中斗法、七十二变、天眼神光、天庭围捕"

    def get_cast(self) -> list[dict]:
        return CAST

    def get_video_options(self) -> dict:
        return VIDEO

    def has_tts(self) -> bool:
        return True

    def get_notes(self) -> dict:
        return {
            "scene_count": len(SCENE_SPECS),
            "format": "sun-wukong-vs-erlang",
            "bgm_assets": [HEAVEN_BGM, FLIGHT_BGM, DUEL_BGM, PRESSURE_BGM, EPILOGUE_BGM],
            "featured_effects": [
                "风起云涌",
                "御剑飞行",
                "死亡光线特效",
                "龟派气功",
                "命中特效",
                "爆炸特效",
                "飞踢",
            ],
        }

    def get_default_output(self) -> str:
        return "outputs/sun_wukong_vs_erlang.mp4"

    def get_description(self) -> str:
        return "Render a 10-scene Journey to the West duel with narration, TTS, staged BGM shifts, effects, and combat SFX."

    def get_scenes(self) -> list[dict]:
        scenes: list[dict] = []
        for scene_index, spec in enumerate(SCENE_SPECS):
            dialogue_items, talk_beats, expressions_track = build_dialogue_bundle(spec.lines)
            talk_beats = trim_talk_beats_for_actions(talk_beats, spec.extra_beats)
            beats = sorted([*talk_beats, *spec.extra_beats], key=lambda item: (item["start_ms"], item["actor_id"]))
            expressions_sorted = sorted(expressions_track, key=lambda item: (item["start_ms"], item["actor_id"]))
            battle = bool(spec.extra_beats or spec.effects)
            aerial = any(item.get("type") in {"御剑飞行", "风起云涌", "死亡光线特效"} for item in spec.effects)
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
                    camera=spec.camera or scene_camera(scene_index, battle=battle, aerial=aerial),
                    effects=spec.effects,
                    foregrounds=[*default_foregrounds(spec.background), *spec.foregrounds],
                    props=spec.props,
                    actors=spec.actors,
                    beats=beats,
                    expressions=expressions_sorted,
                    dialogues=dialogue_items,
                    audio=audio_payload,
                )
            )
        return scenes


SCRIPT = SunWukongVsErlangVideo()


def build_story() -> dict:
    return SCRIPT.build_story()


def main() -> int:
    return SCRIPT()


if __name__ == "__main__":
    raise SystemExit(main())
