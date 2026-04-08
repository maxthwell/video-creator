#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import edge_tts

import generate_actions_pose_reconstruction as poseviz
from common.io import manifest_index, resolve_effect_asset
from common.panda_true3d_renderer import PandaTrue3DRenderer


ROOT_DIR = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT_DIR / "tmp" / "direct_runs" / "cangyun_escort_story"
TMP_DIR = TMP_ROOT / "normal"
OUTPUT_DEFAULT = ROOT_DIR / "outputs" / "cangyun_escort_story.mp4"
DEFAULT_FPS = 24
FAST_FPS = 12
FAST2_FPS = 8
FAST3_FPS = 5
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540
FAST_WIDTH = DEFAULT_WIDTH
FAST_HEIGHT = DEFAULT_HEIGHT
FAST2_WIDTH = 640
FAST2_HEIGHT = 360
FAST3_WIDTH = 480
FAST3_HEIGHT = 270
WIDTH = DEFAULT_WIDTH
HEIGHT = DEFAULT_HEIGHT
TITLE = "寒江断令"
TALK_GAP_S = 0.28
EXPRESSION_CYCLE_S = 2.4
ACTION_TRACKS = {"拳击", "翻跟头gif", "人物A飞踢倒人物B", "舞剑", "跑", "连续后空翻", "降龙十八掌", "鲤鱼打挺"}
EFFECT_PLAYBACK_RATE = 2.8
EFFECT_ALPHA_MIN = 100
EFFECT_ALPHA_MAX = 150
EFFECT_ONE_SHOT_DURATION_S: dict[str, float] = {
    "rain": 2.2,
    "wind": 2.0,
    "impact": 1.1,
    "slash": 1.0,
    "thunder": 1.3,
    "fire": 1.8,
    "burst": 1.1,
    "embers": 1.6,
    "dust": 1.5,
}
EFFECT_AUDIO_HINTS: dict[str, tuple[str, ...]] = {
    "rain": ("暴雨", "雨"),
    "wind": ("风",),
    "impact": ("击中", "打中", "拳"),
    "slash": ("刀", "剑", "金属"),
    "thunder": ("打雷", "雷"),
    "fire": ("爆炸", "爆破", "火"),
    "burst": ("打斗", "击中", "拳", "爆"),
    "embers": ("心脏", "怦怦"),
    "dust": ("心脏", "怦怦"),
}
EFFECT_ASSET_MAP: dict[str, tuple[str, tuple[float, float, float, float], float]] = {
    "rain": ("电闪雷鸣", (0.0, 0.0, 1.0, 1.0), 0.28),
    "wind": ("风起云涌", (0.0, 0.0, 1.0, 1.0), 0.22),
    "impact": ("命中特效", (0.0, 0.0, 1.0, 1.0), 0.55),
    "slash": ("银河旋转特效", (0.0, 0.0, 1.0, 1.0), 0.32),
    "thunder": ("电闪雷鸣", (0.0, 0.0, 1.0, 1.0), 0.30),
    "fire": ("熊熊大火", (0.0, 0.0, 1.0, 1.0), 0.34),
    "burst": ("爆炸特效", (0.0, 0.0, 1.0, 1.0), 0.32),
    "embers": ("启动大招特效", (0.0, 0.0, 1.0, 1.0), 0.18),
    "dust": ("夕阳武士", (0.0, 0.0, 1.0, 1.0), 0.16),
}
BACKGROUND_IDS = set(manifest_index("backgrounds").keys())


@dataclass(frozen=True)
class SfxCue:
    path: Path
    offset_s: float
    volume: float = 0.8


@dataclass(frozen=True)
class ActorSpec:
    actor_id: str
    label: str
    character_id: str
    voice: str
    track_name: str
    expression: str
    x_offset: int
    scale: float = 0.9
    mirror: bool = False
    visible: bool = True


@dataclass(frozen=True)
class LineSpec:
    speaker_id: str
    text: str
    expression: str
    track_name: str | None = None


@dataclass(frozen=True)
class ExpressionCue:
    actor_id: str
    start_s: float
    expression: str


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    title: str
    actors: tuple[ActorSpec, ...]
    lines: tuple[LineSpec, ...]
    bgm_path: Path
    background_top: tuple[int, int, int]
    background_bottom: tuple[int, int, int]
    accent: tuple[int, int, int]
    effect: str
    bgm_group: str | None = None
    sfx: tuple[SfxCue, ...] = field(default_factory=tuple)
    expression_cues: tuple[ExpressionCue, ...] = field(default_factory=tuple)
    hold_s: float = 0.42


@dataclass(frozen=True)
class ScheduledLine:
    speaker_id: str
    speaker_label: str
    text: str
    expression: str
    track_name: str | None
    voice: str
    tts_path: Path
    start_s: float
    end_s: float
    duration_s: float


@dataclass(frozen=True)
class ScheduledExpressionCue:
    actor_id: str
    start_s: float
    expression: str
    priority: int


SCENES: list[SceneSpec] = [
    SceneSpec(
        scene_id="01",
        title="雪夜遗令",
        actors=(
            ActorSpec("shen", "沈孤鸿", "farmer-old", "zh-CN-YunjianNeural", "放松站立", "serious", -220, 0.82),
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "focused", 0, 0.96),
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "站立", "neutral", 230, 0.88),
        ),
        lines=(
            LineSpec("shen", "断龙令今夜必须送到白鹿关，迟一刻，关外三营都会被假军令调走。", "serious"),
            LineSpec("lu", "师叔把令箭交给我，我便护它到天亮。", "focused"),
            LineSpec("ning", "我带药囊和封蜡，路上若有人查匣，我来替你周旋。", "neutral"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "铁血丹心.mp3",
        bgm_group="opening",
        background_top=(52, 47, 67),
        background_bottom=(15, 13, 24),
        accent=(242, 220, 173),
        effect="embers",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "心脏怦怦跳.wav", 0.8, 0.12),),
    ),
    SceneSpec(
        scene_id="02",
        title="镖局封门",
        actors=(
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "serious", -200, 0.96),
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "站立", "skeptical", 20, 0.88),
            ActorSpec("han", "韩照", "detective-sleek", "zh-CN-YunjianNeural", "放松站立", "focused", 250, 0.9),
        ),
        lines=(
            LineSpec("han", "前后门都换成了叶藏锋的人，明面上是护院，脚下站位却像围杀。", "focused"),
            LineSpec("ning", "那就不走门，后井有条旧水道，能通到灯市边上的染坊。", "skeptical"),
            LineSpec("lu", "你在前探路，我背令匣走井道。今夜谁也别回头。", "serious"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "铁血丹心.mp3",
        bgm_group="opening",
        background_top=(219, 203, 170),
        background_bottom=(149, 113, 73),
        accent=(121, 70, 34),
        effect="wind",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "潺潺流水声.wav", 0.1, 0.08),),
    ),
    SceneSpec(
        scene_id="03",
        title="灯市换匣",
        actors=(
            ActorSpec("shop", "染坊娘子", "face-15", "zh-CN-XiaoxiaoNeural", "坐下", "nervous", -200, 0.84),
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "站立", "smile", 20, 0.88),
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "focused", 250, 0.96),
        ),
        lines=(
            LineSpec("shop", "你们来得比约定早，假匣和旧封条都备好了，只是街口多了三拨生面孔。", "nervous"),
            LineSpec("ning", "越热闹越好，他们盯着我手里的假货，才看不见你背后的真匣。", "smile"),
            LineSpec("lu", "换完就走，灯一灭，整条街都会变成他们的网。", "focused"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "铁血丹心.mp3",
        bgm_group="opening",
        background_top=(203, 178, 143),
        background_bottom=(108, 79, 56),
        accent=(248, 230, 204),
        effect="dust",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "心脏怦怦跳.wav", 1.9, 0.1),),
    ),
    SceneSpec(
        scene_id="04",
        title="雨巷截杀",
        actors=(
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "angry", -190, 0.96, True),
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "跑", "focused", 20, 0.88),
            ActorSpec("yuan", "袁烈", "emperor-ming", "zh-CN-YunjianNeural", "拳击", "angry", 260, 0.92),
        ),
        lines=(
            LineSpec("yuan", "把匣子放下，我只断你们一只手。再跑，我就收两条命。", "angry", "拳击"),
            LineSpec("lu", "令匣你碰不到，今夜这条巷子就是你的坟。", "angry", "拳击"),
            LineSpec("ning", "右边墙头有空档，我扔火粉逼他抬头，你借势出拳。", "focused", "跑"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "最后之战-热血-卢冠廷.mp3",
        bgm_group="rain_fight",
        background_top=(57, 76, 107),
        background_bottom=(12, 16, 27),
        accent=(206, 228, 255),
        effect="rain",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "暴雨.wav", 0.0, 0.18),
            SfxCue(ROOT_DIR / "assets" / "audio" / "格斗打中.wav", 2.4, 0.82),
            SfxCue(ROOT_DIR / "assets" / "audio" / "一拳击中.wav", 4.1, 0.94),
        ),
    ),
    SceneSpec(
        scene_id="05",
        title="河仓疗伤",
        actors=(
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "蹲下", "pained", -190, 0.92),
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "朝右跪坐", "focused", 40, 0.86),
            ActorSpec("han", "韩照", "detective-sleek", "zh-CN-YunjianNeural", "放松站立", "serious", 260, 0.9),
        ),
        lines=(
            LineSpec("ning", "刀口里有麻骨散，袁烈不是来抢匣，是想先废你的右臂。", "focused"),
            LineSpec("han", "码头外沿多了军中暗哨，能调得动他们的人，只能是叶藏锋。", "serious"),
            LineSpec("lu", "那就顺着这条线往前查，今夜先活下来，明夜再拔他的根。", "pained"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "最后之战-热血-卢冠廷.mp3",
        bgm_group="rain_fight",
        background_top=(70, 59, 62),
        background_bottom=(22, 18, 24),
        accent=(255, 201, 166),
        effect="embers",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "心脏怦怦跳.wav", 1.3, 0.12),),
    ),
    SceneSpec(
        scene_id="06",
        title="义庄验尸",
        actors=(
            ActorSpec("han", "韩照", "detective-sleek", "zh-CN-YunjianNeural", "站立", "focused", -190, 0.9),
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "skeptical", 10, 0.96),
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "站立", "serious", 250, 0.88),
        ),
        lines=(
            LineSpec("han", "死者腰牌背面刻着白鹿关仓印，叶藏锋已经把手伸到关口。", "focused"),
            LineSpec("lu", "他若只是求财，不会动关仓和军印。断龙令背后还有更大的局。", "skeptical"),
            LineSpec("ning", "那我们就不能只逃，要拿到能钉死他的卷册和人证。", "serious"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "芦苇荡-赵季平-大话西游.mp3",
        background_top=(86, 116, 128),
        background_bottom=(31, 49, 59),
        accent=(220, 239, 227),
        effect="wind",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "潺潺流水声.wav", 0.0, 0.12),),
    ),
    SceneSpec(
        scene_id="07",
        title="竹海擒哨",
        actors=(
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "angry", -190, 0.96, True),
            ActorSpec("han", "韩照", "detective-sleek", "zh-CN-YunjianNeural", "站立", "focused", 20, 0.9),
            ActorSpec("qian", "钱哨头", "official-minister", "zh-CN-YunjianNeural", "人物A飞踢倒人物B", "fear", 270, 0.86),
        ),
        lines=(
            LineSpec("qian", "别打了，我只负责沿河递信，真正接匣的人在府衙案库等你们。", "fear", "人物A飞踢倒人物B"),
            LineSpec("han", "案库夜里只开一道偏门，钥匙归叶藏锋心腹管。", "focused"),
            LineSpec("lu", "很好，你带路。敢耍花样，我先折你的腿。", "angry", "拳击"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "杀破狼.mp3",
        bgm_group="prison_escape",
        background_top=(55, 94, 64),
        background_bottom=(15, 31, 18),
        accent=(185, 235, 190),
        effect="burst",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "031_26_赤手空拳打斗的声音_爱给网_aigei_com.mp3", 1.0, 0.82),
            SfxCue(ROOT_DIR / "assets" / "audio" / "格斗打中.wav", 3.0, 0.86),
        ),
    ),
    SceneSpec(
        scene_id="08",
        title="夜入案库",
        actors=(
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "focused", -190, 0.96),
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "站立", "smile", 20, 0.88),
            ActorSpec("clerk", "守库吏", "official-minister", "zh-CN-YunjianNeural", "放松站立", "skeptical", 260, 0.86),
        ),
        lines=(
            LineSpec("clerk", "三更以后不准翻册，谁给你们的胆子闯案库。", "skeptical"),
            LineSpec("ning", "你认清楚，是叶大人让我们来换封条。你若耽误时辰，掉脑袋的是你。", "smile"),
            LineSpec("lu", "找白鹿关仓册、调兵票底和押印名单，一页都不能漏。", "focused"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "杀破狼.mp3",
        bgm_group="prison_escape",
        background_top=(52, 61, 84),
        background_bottom=(15, 18, 33),
        accent=(224, 236, 250),
        effect="thunder",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "打雷闪电.wav", 1.7, 0.42),),
    ),
    SceneSpec(
        scene_id="09",
        title="狱中救证",
        actors=(
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "angry", -190, 0.96, True),
            ActorSpec("qin", "秦刀", "face-17", "zh-CN-YunxiNeural", "朝右跪坐", "serious", 20, 0.88),
            ActorSpec("guard", "狱卒", "official-minister", "zh-CN-YunjianNeural", "拳击", "angry", 260, 0.86),
        ),
        lines=(
            LineSpec("guard", "叶大人交代过，秦刀活不到天亮，谁来都一样。", "angry", "拳击"),
            LineSpec("qin", "仓门机括图在我脑子里，只要你们带我出去，我就能开白鹿关北门。", "serious"),
            LineSpec("lu", "先跟我杀出去，到了外头，你再把这笔旧账一条条说清。", "angry", "舞剑"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "杀破狼.mp3",
        bgm_group="prison_escape",
        background_top=(71, 35, 28),
        background_bottom=(18, 9, 12),
        accent=(255, 208, 184),
        effect="fire",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "刀剑、金属碰撞（带回音）_爱给网_aigei_com.mp3", 2.0, 0.84),
            SfxCue(ROOT_DIR / "assets" / "audio" / "格斗打中.wav", 4.0, 0.84),
        ),
    ),
    SceneSpec(
        scene_id="10",
        title="屋脊脱围",
        actors=(
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "跑", "angry", -180, 0.96, True),
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "跑", "focused", 20, 0.88),
            ActorSpec("qin", "秦刀", "face-17", "zh-CN-YunxiNeural", "跑", "serious", 240, 0.88),
        ),
        lines=(
            LineSpec("qin", "西巷全是弩手，正街又有封马，我熟悉屋檐走法，跟着我跳。", "serious"),
            LineSpec("ning", "我在后面撒药烟，他们看不清脚下，你们先过。", "focused"),
            LineSpec("lu", "一直翻到城西鼓楼，到了暗河口再分开换气。", "angry", "跑"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "杀破狼.mp3",
        bgm_group="prison_escape",
        background_top=(27, 36, 61),
        background_bottom=(7, 8, 18),
        accent=(208, 226, 255),
        effect="thunder",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "打雷闪电.wav", 1.0, 0.52),
            SfxCue(ROOT_DIR / "assets" / "audio" / "暴雨.wav", 0.0, 0.16),
        ),
    ),
    SceneSpec(
        scene_id="11",
        title="山亭对质",
        actors=(
            ActorSpec("qin", "秦刀", "face-17", "zh-CN-YunxiNeural", "坐下", "thinking", -210, 0.86),
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "serious", 10, 0.96),
            ActorSpec("han", "韩照", "detective-sleek", "zh-CN-YunjianNeural", "站立", "focused", 250, 0.9),
        ),
        lines=(
            LineSpec("qin", "叶藏锋要的不是令箭本身，而是借断龙令打开白鹿关北仓，偷换军械。", "thinking"),
            LineSpec("han", "所以他先灭案库旧卷，再灭你这个做过仓匠的人证。", "focused"),
            LineSpec("lu", "只要把你和卷册一起送到关前，他这条线就再也藏不住。", "serious"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "思君黯然-天龙八部-悲伤.mp3",
        background_top=(67, 61, 71),
        background_bottom=(22, 20, 28),
        accent=(231, 214, 188),
        effect="embers",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "心脏怦怦跳.wav", 2.0, 0.12),),
    ),
    SceneSpec(
        scene_id="12",
        title="绝壁采药",
        actors=(
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "蹲下", "pained", -180, 0.84),
            ActorSpec("yao", "药娘", "face-15", "zh-CN-XiaoxiaoNeural", "坐下", "neutral", 40, 0.82),
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "focused", 250, 0.96),
        ),
        lines=(
            LineSpec("yao", "你替陆青川挡了一记透骨针，再不解毒，明日拿刀的人就变成你自己。", "neutral"),
            LineSpec("lu", "药采到了就走，白鹿关只剩半日路程，我们耽误不起。", "focused"),
            LineSpec("ning", "我还能撑，等把令箭送进关门，你再逼我喝药不迟。", "pained"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "莫失莫忘.mp3",
        background_top=(170, 182, 200),
        background_bottom=(80, 97, 120),
        accent=(245, 247, 255),
        effect="wind",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "潺潺流水声.wav", 0.0, 0.1),),
    ),
    SceneSpec(
        scene_id="13",
        title="雪岭伏杀",
        actors=(
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "angry", -190, 0.96, True),
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "站立", "focused", 30, 0.88),
            ActorSpec("yuan", "袁烈", "emperor-ming", "zh-CN-YunjianNeural", "拳击", "angry", 270, 0.92),
        ),
        lines=(
            LineSpec("yuan", "叶大人算得真准，你们果然会走雪岭近道。把秦刀留下，我饶你们一个全尸。", "angry", "拳击"),
            LineSpec("ning", "你敢堵在这里，说明叶藏锋还没拿到令箭，他比你更急。", "focused"),
            LineSpec("lu", "那我就先拿你的命，去换他今晚的胆寒。", "angry", "拳击"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "杀破狼.mp3",
        bgm_group="siege_arc",
        background_top=(174, 184, 198),
        background_bottom=(92, 102, 118),
        accent=(255, 234, 204),
        effect="impact",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "一拳击中.wav", 2.6, 0.95),
            SfxCue(ROOT_DIR / "assets" / "audio" / "格斗打中.wav", 4.5, 0.88),
        ),
    ),
    SceneSpec(
        scene_id="14",
        title="断塔焚册",
        actors=(
            ActorSpec("han", "韩照", "detective-sleek", "zh-CN-YunjianNeural", "站立", "focused", -180, 0.9),
            ActorSpec("ye", "叶藏锋", "official-minister", "zh-CN-YunjianNeural", "放松站立", "cold", 20, 0.88),
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "站立", "angry", 260, 0.88),
        ),
        lines=(
            LineSpec("ye", "旧册一烧，北仓失印的事就只剩流言。你们拿什么进关告我。", "cold"),
            LineSpec("han", "卷册能烧，押印人和仓门图却在我们手里。你越急，越像做贼。", "focused"),
            LineSpec("ning", "火光照得满城都能看见，你这一烧，是替我们把人都喊醒了。", "angry"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "杀破狼.mp3",
        bgm_group="siege_arc",
        background_top=(96, 32, 24),
        background_bottom=(22, 8, 10),
        accent=(255, 190, 146),
        effect="fire",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "音效 爆炸 爆破 爆发 战斗_爱给网_aigei_com.mp3", 4.0, 0.2),),
    ),
    SceneSpec(
        scene_id="15",
        title="暗河换船",
        actors=(
            ActorSpec("boat", "乌篷翁", "farmer-old", "zh-CN-YunjianNeural", "坐下", "thinking", -190, 0.82),
            ActorSpec("qin", "秦刀", "face-17", "zh-CN-YunxiNeural", "坐下", "serious", 30, 0.86),
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "neutral", 250, 0.96),
        ),
        lines=(
            LineSpec("boat", "官道都断了，只剩暗河还能贴着山根走，天亮前就能把你们送到关下。", "thinking"),
            LineSpec("qin", "到了白鹿关南坡，我能认出藏机括钥的旧砖。", "serious"),
            LineSpec("lu", "过了这道水，前面就只剩硬闯。大家把最后的力气留到关前。", "neutral"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "芦苇荡-赵季平-大话西游.mp3",
        background_top=(88, 121, 141),
        background_bottom=(27, 46, 63),
        accent=(229, 241, 228),
        effect="wind",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "潺潺流水声.wav", 0.0, 0.15),),
    ),
    SceneSpec(
        scene_id="16",
        title="长街断后",
        actors=(
            ActorSpec("han", "韩照", "detective-sleek", "zh-CN-YunjianNeural", "拳击", "angry", -190, 0.9, True),
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "跑", "focused", 20, 0.96),
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "跑", "focused", 240, 0.88),
        ),
        lines=(
            LineSpec("han", "前街我来挡，你们带秦刀和令匣冲关，别让我的血白流。", "angry", "拳击"),
            LineSpec("ning", "韩照，最多一炷香，若你不来，我们就在关楼上替你点第一盏灯。", "focused", "跑"),
            LineSpec("lu", "守住自己这口气，关门一开，我回头接你。", "focused", "跑"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "杀破狼.mp3",
        background_top=(64, 55, 74),
        background_bottom=(20, 16, 27),
        accent=(244, 223, 193),
        effect="slash",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "刀剑、金属碰撞（带回音）_爱给网_aigei_com.mp3", 1.8, 0.84),
            SfxCue(ROOT_DIR / "assets" / "audio" / "031_26_赤手空拳打斗的声音_爱给网_aigei_com.mp3", 3.7, 0.88),
        ),
    ),
    SceneSpec(
        scene_id="17",
        title="寒寺托证",
        actors=(
            ActorSpec("shen", "沈孤鸿", "farmer-old", "zh-CN-YunjianNeural", "坐下", "thinking", -210, 0.82),
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "serious", 10, 0.96),
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "站立", "focused", 240, 0.88),
        ),
        lines=(
            LineSpec("shen", "二十年前我守过北仓，叶藏锋那时就偷换军械，死的人一直压在雪里。", "thinking"),
            LineSpec("lu", "今夜把令箭、仓册、人证和你的口供一并送上关楼，他就再也赖不掉。", "serious"),
            LineSpec("ning", "旧案埋得再深，只要见了天光，就会自己喊冤。", "focused"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "铁血丹心.mp3",
        bgm_group="finale",
        background_top=(58, 62, 89),
        background_bottom=(17, 20, 35),
        accent=(225, 221, 255),
        effect="embers",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "心脏怦怦跳.wav", 2.2, 0.12),),
    ),
    SceneSpec(
        scene_id="18",
        title="关前拒令",
        actors=(
            ActorSpec("feng", "封守毅", "general-guard", "zh-CN-YunxiNeural", "放松站立", "skeptical", -190, 0.92),
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "angry", 20, 0.96),
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "站立", "angry", 240, 0.88),
        ),
        lines=(
            LineSpec("feng", "关门夜封，任何私令都不能验。你们若再上前，我只能按闯关论。", "skeptical"),
            LineSpec("ning", "你若再迟半刻，明早开仓的人就会发现军械全成了废铁。", "angry"),
            LineSpec("lu", "后面追兵已到，你是守规矩，还是守白鹿关，今夜就得选。", "angry"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "铁血丹心.mp3",
        bgm_group="finale",
        background_top=(88, 103, 124),
        background_bottom=(27, 34, 49),
        accent=(251, 240, 204),
        effect="thunder",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "打雷闪电.wav", 1.4, 0.45),),
    ),
    SceneSpec(
        scene_id="19",
        title="关楼决战",
        actors=(
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "angry", -190, 0.96, True),
            ActorSpec("yuan", "袁烈", "emperor-ming", "zh-CN-YunjianNeural", "拳击", "angry", 170, 0.92),
            ActorSpec("ye", "叶藏锋", "official-minister", "zh-CN-YunjianNeural", "放松站立", "cold", 330, 0.84),
        ),
        lines=(
            LineSpec("ye", "只要关门不开，今晚的一切都能算成匪患。陆青川，你赢不了朝里的手。", "cold"),
            LineSpec("yuan", "你去夺令匣，我来打碎他的骨头。", "angry", "拳击"),
            LineSpec("lu", "你们要的是整座关城，我要的只是让所有人看清你们的脸。", "angry", "舞剑"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "铁血丹心.mp3",
        bgm_group="finale",
        background_top=(108, 30, 28),
        background_bottom=(20, 8, 10),
        accent=(255, 222, 206),
        effect="impact",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "刀剑、金属碰撞（带回音）_爱给网_aigei_com.mp3", 2.0, 0.86),
            SfxCue(ROOT_DIR / "assets" / "audio" / "格斗打中.wav", 4.6, 0.86),
            SfxCue(ROOT_DIR / "assets" / "audio" / "一拳击中.wav", 5.3, 0.95),
        ),
    ),
    SceneSpec(
        scene_id="20",
        title="关楼昭令",
        actors=(
            ActorSpec("feng", "封守毅", "general-guard", "zh-CN-YunxiNeural", "放松站立", "relieved", -200, 0.92),
            ActorSpec("ning", "宁听雪", "face-13", "zh-CN-XiaoxiaoNeural", "站立", "smile", 20, 0.88),
            ActorSpec("lu", "陆青川", "face-2", "zh-CN-YunxiNeural", "站立", "neutral", 250, 0.96),
        ),
        lines=(
            LineSpec("feng", "断龙令、仓册、机括图与叶藏锋亲笔押印俱在，白鹿关众军听令，今夜就地封仓拿人。", "relieved"),
            LineSpec("ning", "雪夜里埋了二十年的旧案，总算在天亮前见了人心。", "smile"),
            LineSpec("lu", "关门守住了，命也守住了。接下来，该把那些躲在城里的名字一个个挖出来。", "neutral"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "铁血丹心.mp3",
        bgm_group="finale",
        background_top=(163, 177, 204),
        background_bottom=(81, 100, 131),
        accent=(255, 242, 210),
        effect="wind",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "潺潺流水声.wav", 0.0, 0.08),),
    ),
]


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    return ffmpeg


def _ffprobe_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return max(0.0, float((result.stdout or "0").strip() or 0.0))


@lru_cache(maxsize=64)
def _track(track_name: str) -> poseviz.PoseTrack:
    return poseviz._load_track(poseviz.POSE_DIR / f"{track_name}.pose.json", width=WIDTH, height=HEIGHT)


@lru_cache(maxsize=64)
def _has_track(track_name: str) -> bool:
    return (poseviz.POSE_DIR / f"{track_name}.pose.json").exists()


@lru_cache(maxsize=64)
def _available_expressions(character_id: str) -> tuple[str, ...]:
    skins_dir = poseviz.CHARACTER_DIR / character_id / "skins"
    if not skins_dir.exists():
        return ("default", "neutral")
    names: set[str] = set()
    for path in skins_dir.glob("face_*.png"):
        stem = path.stem
        if stem.startswith("face_talk_"):
            continue
        if stem in {"face_default", "face_neutral_open", "face_neutral_closed"}:
            continue
        names.add(stem.removeprefix("face_"))
    if "default" not in names:
        names.add("default")
    return tuple(sorted(names))


def _resolve_expression(character_id: str, requested: str) -> str:
    available = set(_available_expressions(character_id))
    normalized = (requested or "default").strip().lower().replace("-", "_")
    aliases: dict[str, tuple[str, ...]] = {
        "serious": ("thinking", "skeptical", "neutral", "default"),
        "focused": ("thinking", "skeptical", "neutral", "default"),
        "skeptical": ("skeptical", "thinking", "neutral", "default"),
        "nervous": ("sad", "skeptical", "neutral", "default"),
        "pained": ("sad", "angry", "neutral", "default"),
        "shocked": ("excited", "skeptical", "neutral", "default"),
        "surprised": ("excited", "skeptical", "neutral", "default"),
        "excited": ("excited", "smile", "neutral", "default"),
        "smile": ("smile", "neutral", "default"),
        "sad": ("sad", "neutral", "default"),
        "angry": ("angry", "skeptical", "neutral", "default"),
        "thinking": ("thinking", "skeptical", "neutral", "default"),
        "neutral": ("neutral", "default"),
        "default": ("default", "neutral"),
    }
    for candidate in aliases.get(normalized, (normalized, "neutral", "default")):
        if candidate in available:
            return candidate
    return "default"


def _reaction_expression(actor: ActorSpec, driver_expression: str) -> str:
    normalized = (driver_expression or "default").strip().lower().replace("-", "_")
    if normalized in {"angry"}:
        requested = "skeptical"
    elif normalized in {"pained", "sad", "nervous"}:
        requested = "sad"
    elif normalized in {"excited", "smile"}:
        requested = "smile"
    elif normalized in {"focused", "serious", "thinking"}:
        requested = "thinking"
    else:
        requested = actor.expression
    return _resolve_expression(actor.character_id, requested)


def _ambient_expression_sequence(actor: ActorSpec) -> tuple[str, ...]:
    available = set(_available_expressions(actor.character_id))
    base = _resolve_expression(actor.character_id, actor.expression)
    candidates: list[str] = [base]
    for name in ("neutral", "thinking", "skeptical", "smile", "sad", "angry", "excited", "default"):
        resolved = _resolve_expression(actor.character_id, name)
        if resolved not in candidates:
            candidates.append(resolved)
    filtered = tuple(name for name in candidates if name in available or name == "default")
    return filtered or (base,)


def _expression_cycle_jitter(actor: ActorSpec, step: int) -> float:
    seed = sum(ord(ch) for ch in f"{actor.actor_id}:{actor.character_id}") * 0.017
    phase = seed + step * 1.61803398875
    return math.sin(phase) * 0.42 + math.cos(phase * 0.73) * 0.18

def _set_render_profile(*, fast: bool = False, fast2: bool = False, fast3: bool = False) -> int:
    global WIDTH, HEIGHT, TMP_DIR
    if fast3:
        WIDTH = FAST3_WIDTH
        HEIGHT = FAST3_HEIGHT
        TMP_DIR = TMP_ROOT / "fast3"
        fps = FAST3_FPS
    elif fast2:
        WIDTH = FAST2_WIDTH
        HEIGHT = FAST2_HEIGHT
        TMP_DIR = TMP_ROOT / "fast2"
        fps = FAST2_FPS
    elif fast:
        WIDTH = FAST_WIDTH
        HEIGHT = FAST_HEIGHT
        TMP_DIR = TMP_ROOT / "fast"
        fps = FAST_FPS
    else:
        WIDTH = DEFAULT_WIDTH
        HEIGHT = DEFAULT_HEIGHT
        TMP_DIR = TMP_ROOT / "normal"
        fps = DEFAULT_FPS
    _track.cache_clear()
    return fps


def _default_idle_track(actor: ActorSpec, requested: str) -> str:
    palette = poseviz.CHARACTER_PALETTES.get(actor.character_id)
    feminine = actor.character_id in {"npc-girl", "office-worker-modern", "reporter-selfie"} or actor.character_id.startswith("face-") and actor.character_id in {"face-5", "face-7", "face-8", "face-13", "face-14", "face-15", "face-16"}
    if feminine:
        if requested == "掐腰站立" and _has_track("女人单手掐腰站立"):
            return "女人单手掐腰站立"
        if requested in {"站立", "放松站立"} and _has_track("女人站立"):
            return "女人站立"
    if requested == "站立" and _has_track("放松站立"):
        return "放松站立"
    return requested


async def _synthesize_tts(text: str, voice: str, output_path: Path, *, refresh: bool = False) -> None:
    if not refresh and output_path.exists() and output_path.stat().st_size > 0:
        return
    existing_ok = output_path.exists() and output_path.stat().st_size > 0
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice, rate="+0%")
            if temp_path.exists():
                temp_path.unlink()
            await communicate.save(str(temp_path))
            if temp_path.stat().st_size <= 0:
                raise RuntimeError(f"TTS output is empty: {temp_path}")
            temp_path.replace(output_path)
            return
        except Exception as exc:
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()
            await asyncio.sleep(1.2 * (attempt + 1))
    if existing_ok:
        return
    assert last_error is not None
    raise last_error


def _scene_paths(scene: SceneSpec) -> dict[str, Path]:
    scene_dir = TMP_DIR / f"scene_{scene.scene_id}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": scene_dir,
        "audio": scene_dir / "scene_audio.m4a",
        "video": scene_dir / "scene_video.mp4",
        "scene_mp4": scene_dir / f"{scene.scene_id}.mp4",
    }


def _bgm_chain_key(scene: SceneSpec) -> tuple[str | None, Path]:
    return (scene.bgm_group or scene.scene_id, scene.bgm_path.resolve())


def _rgba01(rgb: tuple[int, int, int], alpha: float = 1.0) -> tuple[float, float, float, float]:
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, alpha)


def _panda_background_id(scene: SceneSpec) -> str:
    title = scene.title
    if scene.effect == "rain":
        return "night-bridge" if "night-bridge" in BACKGROUND_IDS else "shop-row"
    if any(token in title for token in ("库", "案", "卷", "阁", "厅", "堂", "庄", "义庄")):
        return "town-hall-records" if "town-hall-records" in BACKGROUND_IDS else "inn-hall"
    if any(token in title for token in ("巷", "市", "街", "门", "桥", "关", "坡")):
        return "shop-row" if "shop-row" in BACKGROUND_IDS else "street-day"
    if scene.effect in {"embers", "fire", "burst"}:
        return "mountain-cliff" if "mountain-cliff" in BACKGROUND_IDS else "temple-courtyard"
    return "temple-courtyard" if "temple-courtyard" in BACKGROUND_IDS else next(iter(BACKGROUND_IDS))


def _panda_effect_items(scene: SceneSpec, duration_s: float) -> list[dict[str, object]]:
    effect_spec = EFFECT_ASSET_MAP.get(scene.effect)
    if effect_spec is None:
        return []
    effect_name, _, alpha = effect_spec
    asset_path = resolve_effect_asset(effect_name)
    if asset_path is None:
        return []
    duration_ms = max(1, int(round(duration_s * 1000)))
    effect_duration_ms = int(round(EFFECT_ONE_SHOT_DURATION_S.get(scene.effect, 1.4) * 1000))
    items: list[dict[str, object]] = []
    for start_s in _effect_trigger_times(scene, duration_s):
        start_ms = max(0, int(round(start_s * 1000)))
        end_ms = min(duration_ms, start_ms + effect_duration_ms)
        items.append(
            {
                "type": scene.effect,
                "asset_path": str(asset_path),
                "alpha": alpha,
                "start_ms": start_ms,
                "end_ms": max(start_ms + 1, end_ms),
                "playback_speed": EFFECT_PLAYBACK_RATE,
            }
        )
    return items


def _panda_expression_items(
    scene: SceneSpec,
    expression_cues: tuple[ScheduledExpressionCue, ...],
    duration_s: float,
) -> list[dict[str, object]]:
    duration_ms = int(round(duration_s * 1000))
    items: list[dict[str, object]] = []
    for actor in scene.actors:
        actor_cues = [cue for cue in expression_cues if cue.actor_id == actor.actor_id]
        if not actor_cues:
            continue
        for index, cue in enumerate(actor_cues):
            start_ms = max(0, int(round(cue.start_s * 1000)))
            next_start_ms = duration_ms
            if index + 1 < len(actor_cues):
                next_start_ms = max(start_ms + 1, int(round(actor_cues[index + 1].start_s * 1000)))
            items.append(
                {
                    "actor_id": actor.actor_id,
                    "expression": cue.expression,
                    "start_ms": start_ms,
                    "end_ms": max(start_ms + 1, next_start_ms - 1),
                }
            )
    return items


def _panda_dialogue_items(schedule: list[ScheduledLine]) -> list[dict[str, object]]:
    return [
        {
            "speaker_id": item.speaker_id,
            "text": item.text,
            "subtitle": item.text,
            "start_ms": int(round(item.start_s * 1000)),
            "end_ms": int(round(item.end_s * 1000)),
        }
        for item in schedule
    ]


def _panda_beat_items(scene: SceneSpec, schedule: list[ScheduledLine], duration_s: float) -> list[dict[str, object]]:
    duration_ms = int(round(duration_s * 1000))
    beats: list[dict[str, object]] = []
    actor_map = {actor.actor_id: actor for actor in scene.actors}
    for item in schedule:
        actor = actor_map[item.speaker_id]
        facing = "left" if actor.mirror else "right"
        track_name = item.track_name or actor.track_name
        if _has_track(track_name):
            beats.append(
                {
                    "actor_id": actor.actor_id,
                    "start_ms": int(round(item.start_s * 1000)),
                    "end_ms": int(round(item.end_s * 1000)),
                    "motion": "pose",
                    "facing": facing,
                    "pose_track_path": str((poseviz.POSE_DIR / f"{track_name}.pose.json").resolve()),
                }
            )
    for actor in scene.actors:
        facing = "left" if actor.mirror else "right"
        idle_track = _default_idle_track(actor, actor.track_name)
        if _has_track(idle_track):
            beats.append(
                {
                    "actor_id": actor.actor_id,
                    "start_ms": 0,
                    "end_ms": duration_ms,
                    "motion": "pose",
                    "facing": facing,
                    "pose_track_path": str((poseviz.POSE_DIR / f"{idle_track}.pose.json").resolve()),
                }
            )
    return beats


def _panda_actor_items(scene: SceneSpec) -> list[dict[str, object]]:
    actors: list[dict[str, object]] = []
    for actor in scene.actors:
        if not actor.visible:
            continue
        actors.append(
            {
                "actor_id": actor.actor_id,
                "scale": max(0.72, actor.scale * 0.82),
                "facing": "left" if actor.mirror else "right",
                "layer": "front",
                "spawn": {
                    "x": max(-3.2, min(3.2, actor.x_offset / 95.0)),
                    "z": 0.0,
                },
            }
        )
    return actors


def _panda_scene_dict(
    scene: SceneSpec,
    schedule: list[ScheduledLine],
    expression_cues: tuple[ScheduledExpressionCue, ...],
    duration_s: float,
) -> dict[str, object]:
    background_id = _panda_background_id(scene)
    return {
        "id": scene.scene_id,
        "background": background_id,
        "duration_ms": int(round(duration_s * 1000)),
        "actors": _panda_actor_items(scene),
        "dialogues": _panda_dialogue_items(schedule),
        "expressions": _panda_expression_items(scene, expression_cues, duration_s),
        "beats": _panda_beat_items(scene, schedule, duration_s),
        "effects": _panda_effect_items(scene, duration_s),
        "box": {
            "width": 12.0,
            "height": 7.0,
            "depth": 7.5,
            "back_wall_color": _rgba01(scene.background_top, 1.0),
            "left_wall_color": _rgba01(scene.background_bottom, 1.0),
            "right_wall_color": _rgba01(scene.background_bottom, 1.0),
            "floor_color": _rgba01(tuple(max(0, int(channel * 0.72)) for channel in scene.background_bottom), 1.0),
            "ceiling_color": _rgba01(tuple(min(255, int(channel * 1.04)) for channel in scene.background_top), 1.0),
        },
        "camera": {
            "x": 0.0,
            "z": 0.2,
            "zoom": 1.42,
        },
    }


def _build_panda_story(*, fast: bool, fast2: bool, fast3: bool) -> dict[str, object]:
    cast_map: dict[str, dict[str, object]] = {}
    for scene in SCENES:
        for actor in scene.actors:
            cast_map.setdefault(
                actor.actor_id,
                {
                    "id": actor.actor_id,
                    "display_name": actor.label,
                    "asset_id": actor.character_id,
                },
            )
    return {
        "video": {
            "width": WIDTH,
            "height": HEIGHT,
            "renderer": "panda_card_fast" if (fast or fast2 or fast3) else "true_3d",
            "speed_mode": "extreme" if fast3 else ("fast" if (fast or fast2) else "normal"),
            "show_actor_labels": False,
        },
        "cast": list(cast_map.values()),
    }


def _build_schedule(scene: SceneSpec, scene_dir: Path, *, refresh_tts: bool = False) -> tuple[list[ScheduledLine], float]:
    actor_map = {actor.actor_id: actor for actor in scene.actors}
    schedule: list[ScheduledLine] = []
    cursor = scene.hold_s
    for index, line in enumerate(scene.lines, start=1):
        actor = actor_map[line.speaker_id]
        tts_path = scene_dir / f"line_{index:02d}.mp3"
        asyncio.run(_synthesize_tts(line.text, actor.voice, tts_path, refresh=refresh_tts))
        duration_s = _ffprobe_duration(tts_path)
        schedule.append(
            ScheduledLine(
                speaker_id=actor.actor_id,
                speaker_label=actor.label,
                text=line.text,
                expression=line.expression,
                track_name=line.track_name,
                voice=actor.voice,
                tts_path=tts_path,
                start_s=cursor,
                end_s=cursor + duration_s,
                duration_s=duration_s,
            )
        )
        cursor += duration_s + TALK_GAP_S
    scene_duration = max(cursor + 0.4, 5.0)
    return schedule, scene_duration


def _build_expression_schedule(scene: SceneSpec, schedule: list[ScheduledLine], duration_s: float) -> tuple[ScheduledExpressionCue, ...]:
    cues: list[ScheduledExpressionCue] = []
    actor_map = {actor.actor_id: actor for actor in scene.actors}
    for actor in scene.actors:
        cues.append(ScheduledExpressionCue(actor.actor_id, 0.0, _resolve_expression(actor.character_id, actor.expression), 0))
        sequence = _ambient_expression_sequence(actor)
        if len(sequence) > 1:
            total_steps = max(0, int(duration_s // EXPRESSION_CYCLE_S))
            base_offset = (abs(actor.x_offset) / max(1, WIDTH)) * 0.9
            for step in range(1, total_steps + 1):
                jitter = _expression_cycle_jitter(actor, step)
                start_s = min(duration_s, max(0.0, step * EXPRESSION_CYCLE_S + base_offset + jitter))
                expression = sequence[step % len(sequence)]
                cues.append(ScheduledExpressionCue(actor.actor_id, start_s, expression, 120 + step))
    for index, line in enumerate(schedule, start=1):
        speaker = actor_map[line.speaker_id]
        cues.append(
            ScheduledExpressionCue(
                line.speaker_id,
                line.start_s,
                _resolve_expression(speaker.character_id, line.expression),
                1000 + index,
            )
        )
        for actor in scene.actors:
            if actor.actor_id == line.speaker_id:
                continue
            cues.append(
                ScheduledExpressionCue(
                    actor.actor_id,
                    line.start_s + 0.06,
                    _reaction_expression(actor, line.expression),
                    500 + index,
                )
            )
    for index, cue in enumerate(scene.expression_cues, start=1):
        actor = actor_map[cue.actor_id]
        cues.append(
            ScheduledExpressionCue(
                cue.actor_id,
                max(0.0, cue.start_s),
                _resolve_expression(actor.character_id, cue.expression),
                2000 + index,
            )
        )
    return tuple(sorted(cues, key=lambda item: (item.start_s, item.priority)))


def _expression_at_time(
    actor_id: str,
    t_s: float,
    cues: tuple[ScheduledExpressionCue, ...],
    default_expression: str,
) -> str:
    expression = default_expression
    for cue in cues:
        if cue.actor_id != actor_id:
            continue
        if cue.start_s > t_s:
            break
        expression = cue.expression
    return expression


def _mix_scene_audio(scene: SceneSpec, schedule: list[ScheduledLine], duration_s: float, output_path: Path, *, bgm_offset_s: float = 0.0) -> None:
    ffmpeg = _ffmpeg()
    command = [ffmpeg, "-y", "-stream_loop", "-1", "-i", str(scene.bgm_path)]
    for line in schedule:
        command.extend(["-i", str(line.tts_path)])
    for cue in scene.sfx:
        command.extend(["-i", str(cue.path)])

    bgm_end_s = bgm_offset_s + duration_s
    filters = [f"[0:a]atrim={bgm_offset_s:.3f}:{bgm_end_s:.3f},asetpts=N/SR/TB,volume=0.16[bgm]"]
    mix_inputs = ["[bgm]"]
    for index, line in enumerate(schedule, start=1):
        delay_ms = int(line.start_s * 1000)
        label = f"tts{index}"
        filters.append(f"[{index}:a]adelay={delay_ms}|{delay_ms},volume=1.12[{label}]")
        mix_inputs.append(f"[{label}]")
    base_index = 1 + len(schedule)
    for cue_index, cue in enumerate(scene.sfx, start=base_index):
        label = f"sfx{cue_index}"
        delay_ms = int(cue.offset_s * 1000)
        filters.append(f"[{cue_index}:a]adelay={delay_ms}|{delay_ms},volume={cue.volume:.3f}[{label}]")
        mix_inputs.append(f"[{label}]")
    filters.append(
        f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)},"
        "loudnorm=I=-16:LRA=7:TP=-1.5:linear=true,"
        "alimiter=limit=0.92[aout]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[aout]",
            "-t",
            f"{duration_s:.3f}",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_path),
        ]
    )
    subprocess.run(command, check=True)


def _effect_trigger_times(scene: SceneSpec, duration_s: float) -> tuple[float, ...]:
    hints = EFFECT_AUDIO_HINTS.get(scene.effect, ())
    starts: list[float] = []
    if hints:
        for cue in scene.sfx:
            cue_name = cue.path.stem
            if any(hint in cue_name for hint in hints):
                starts.append(max(0.0, min(duration_s, cue.offset_s)))
    if not starts and scene.sfx:
        starts = [max(0.0, min(duration_s, cue.offset_s)) for cue in scene.sfx]
    if not starts:
        starts = [max(0.0, duration_s * 0.35)]
    deduped: list[float] = []
    for start_s in sorted(starts):
        if not deduped or abs(start_s - deduped[-1]) > 0.08:
            deduped.append(start_s)
    return tuple(deduped)
def _render_scene_video(
    scene: SceneSpec,
    schedule: list[ScheduledLine],
    expression_cues: tuple[ScheduledExpressionCue, ...],
    duration_s: float,
    output_path: Path,
    fps: int,
    panda_renderer: PandaTrue3DRenderer,
    *,
    fast: bool,
    fast2: bool,
    fast3: bool,
) -> None:
    preset, crf = poseviz._encoding_profile(fast=fast, fast2=fast2, fast3=fast3)
    proc = poseviz._open_ffmpeg_stream(fps, WIDTH, HEIGHT, output_path, preset=preset, crf=crf)
    try:
        assert proc.stdin is not None
        total_frames = max(1, int(math.ceil(duration_s * fps)))
        half_frame_s = 0.5 / max(1, fps)
        panda_scene = _panda_scene_dict(scene, schedule, expression_cues, duration_s)
        for frame_index in range(total_frames):
            # Sample slightly ahead of the frame boundary so visual events do not lag behind audio at low FPS.
            t_s = min(duration_s, frame_index / fps + half_frame_s)
            frame_rgb = panda_renderer.capture_scene_frame(panda_scene, int(round(t_s * 1000)), raw_rgb=True)
            proc.stdin.write(frame_rgb)
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")


def _mux_scene(video_path: Path, audio_path: Path, output_path: Path) -> None:
    temp_output = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    temp_output.unlink(missing_ok=True)
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(temp_output),
        ],
        check=True,
    )
    temp_output.replace(output_path)


def _concat_scenes(
    scene_files: list[Path],
    output_path: Path,
    *,
    fps: int,
    fast: bool = False,
    fast2: bool = False,
    fast3: bool = False,
) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False, dir=str(TMP_DIR)) as handle:
        for path in scene_files:
            handle.write(f"file '{path.resolve()}'\n")
        concat_list = Path(handle.name)
    temp_output = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    try:
        preset, crf = poseviz._encoding_profile(fast=fast, fast2=fast2, fast3=fast3)
        temp_output.unlink(missing_ok=True)
        subprocess.run(
            [
                _ffmpeg(),
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-vsync",
                "cfr",
                "-r",
                str(fps),
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-af",
                "aresample=async=1:first_pts=0",
                "-max_muxing_queue_size",
                "4096",
                str(temp_output),
            ],
            check=True,
        )
        temp_output.replace(output_path)
    finally:
        concat_list.unlink(missing_ok=True)


def render_story(output_path: Path, *, force: bool = False, fast: bool = False, fast2: bool = False, fast3: bool = False) -> None:
    fps = _set_render_profile(fast=fast, fast2=fast2, fast3=fast3)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    panda_renderer = PandaTrue3DRenderer(_build_panda_story(fast=fast, fast2=fast2, fast3=fast3), prefer_gpu=True)
    scene_outputs: list[Path] = []
    current_bgm_key: tuple[str | None, Path] | None = None
    current_bgm_offset_s = 0.0
    for scene in SCENES:
        paths = _scene_paths(scene)
        if not force and paths["scene_mp4"].exists() and paths["scene_mp4"].stat().st_size > 0:
            scene_outputs.append(paths["scene_mp4"])
            print(paths["scene_mp4"])
            continue
        schedule, duration_s = _build_schedule(scene, paths["dir"], refresh_tts=force)
        expression_cues = _build_expression_schedule(scene, schedule, duration_s)
        bgm_key = _bgm_chain_key(scene)
        bgm_offset_s = current_bgm_offset_s if bgm_key == current_bgm_key else 0.0
        _mix_scene_audio(scene, schedule, duration_s, paths["audio"], bgm_offset_s=bgm_offset_s)
        _render_scene_video(
            scene,
            schedule,
            expression_cues,
            duration_s,
            paths["video"],
            fps,
            panda_renderer,
            fast=fast,
            fast2=fast2,
            fast3=fast3,
        )
        _mux_scene(paths["video"], paths["audio"], paths["scene_mp4"])
        scene_outputs.append(paths["scene_mp4"])
        if bgm_key == current_bgm_key:
            current_bgm_offset_s += duration_s
        else:
            current_bgm_key = bgm_key
            current_bgm_offset_s = duration_s
        print(paths["scene_mp4"])
    _concat_scenes(scene_outputs, output_path, fps=fps, fast=fast, fast2=fast2, fast3=fast3)
    print(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a multi-character Water Margin dialogue story using DNN pose stickman actors with Chinese subtitles, TTS, BGM, SFX, and action blocking.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--fast2", action="store_true")
    parser.add_argument("--fast3", action="store_true")
    args = parser.parse_args()
    render_story(args.output.resolve(), force=args.force, fast=args.fast, fast2=args.fast2, fast3=args.fast3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
