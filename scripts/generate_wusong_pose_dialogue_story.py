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
from PIL import Image, ImageDraw, ImageFont

import generate_actions_pose_reconstruction as poseviz


ROOT_DIR = Path(__file__).resolve().parents[1]
TMP_DIR = ROOT_DIR / "tmp" / "direct_runs" / "wusong_pose_dialogue_story"
OUTPUT_DEFAULT = ROOT_DIR / "outputs" / "wusong_pose_dialogue_story.mp4"
FPS = 24
WIDTH = 960
HEIGHT = 540
TITLE = "水浒故事·武松对话版"
GROUND_Y = HEIGHT * 0.82
TALK_GAP_S = 0.28
TALK_MOUTH_CYCLE_FRAMES = 4
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
ACTION_TRACKS = {"拳击", "翻跟头gif", "人物A飞踢倒人物B", "舞剑", "跑", "连续后空翻", "降龙十八掌", "鲤鱼打挺"}
LEG_POINTS = {"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"}
ARM_POINTS = {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"}


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
    sfx: tuple[SfxCue, ...] = field(default_factory=tuple)
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


SCENES: list[SceneSpec] = [
    SceneSpec(
        scene_id="01",
        title="酒店相逢",
        actors=(
            ActorSpec("wusong", "武松", "general-guard", "zh-CN-YunxiNeural", "站立", "neutral", -150, 0.96),
            ActorSpec("innkeeper", "店家", "farmer-old", "zh-CN-YunjianNeural", "站立", "smile", 180, 0.84),
        ),
        lines=(
            LineSpec("innkeeper", "客官，景阳冈上近来闹虎，过往行人都不敢夜行。", "skeptical"),
            LineSpec("wusong", "先把酒拿来。酒若不够，老虎我替你收了。", "smile"),
            LineSpec("innkeeper", "别人只喝三碗，你却一连十八碗，真要上冈不成？", "excited"),
            LineSpec("wusong", "大丈夫说上就上。若真有虎，我便打虎下山。", "excited", "太极"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "王进打高俅-赵季平-水浒传.mp3",
        background_top=(243, 231, 210),
        background_bottom=(206, 171, 127),
        accent=(128, 74, 40),
        effect="dust",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "潺潺流水声.wav", 0.0, 0.16),),
    ),
    SceneSpec(
        scene_id="02",
        title="上冈夜话",
        actors=(
            ActorSpec("wusong", "武松", "general-guard", "zh-CN-YunxiNeural", "站立", "focused", -110, 0.96),
            ActorSpec("narrator", "榜文", "narrator", "zh-CN-YunjianNeural", "行走", "skeptical", 220, 0.82, False),
        ),
        lines=(
            LineSpec("narrator", "榜文写得分明，近来大虫伤人，过客须在白日结队而行。", "skeptical"),
            LineSpec("wusong", "我偏不信邪。一个人走，也照样翻过景阳冈。", "angry", "跑"),
            LineSpec("narrator", "山风紧得像兽喘，林子深处只剩你一个脚步声。", "neutral"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "历史的天空-古筝-三国演义片尾曲.mp3",
        background_top=(45, 52, 76),
        background_bottom=(11, 15, 28),
        accent=(202, 212, 242),
        effect="rain",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "暴雨.wav", 0.0, 0.18),
            SfxCue(ROOT_DIR / "assets" / "audio" / "打雷闪电.wav", 1.9, 0.5),
        ),
    ),
    SceneSpec(
        scene_id="03",
        title="景阳恶斗",
        actors=(
            ActorSpec("wusong", "武松", "general-guard", "zh-CN-YunxiNeural", "拳击", "angry", -150, 0.98),
            ActorSpec("tiger", "猛虎", "official-minister", "zh-CN-YunjianNeural", "人物A飞踢倒人物B", "angry", 170, 0.92),
        ),
        lines=(
            LineSpec("tiger", "嗷！", "angry", "人物A飞踢倒人物B"),
            LineSpec("wusong", "来得好！你若扑我，我便拿拳头收你！", "angry", "拳击"),
            LineSpec("wusong", "哨棒断了也无妨，今日只凭双拳，照样打你个半死！", "angry", "拳击"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "王进打高俅-赵季平-水浒传.mp3",
        background_top=(113, 80, 55),
        background_bottom=(33, 22, 16),
        accent=(255, 200, 125),
        effect="impact",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "狗喘粗气.wav", 0.15, 0.24),
            SfxCue(ROOT_DIR / "assets" / "audio" / "一拳击中.wav", 1.7, 0.92),
            SfxCue(ROOT_DIR / "assets" / "audio" / "格斗打中.wav", 3.1, 0.85),
        ),
    ),
    SceneSpec(
        scene_id="04",
        title="阳谷惊闻",
        actors=(
            ActorSpec("wusong", "武松", "general-guard", "zh-CN-YunxiNeural", "站立", "neutral", -170, 0.96),
            ActorSpec("wuda", "武大", "npc-boy", "zh-CN-YunjianNeural", "站立", "smile", 60, 0.82),
            ActorSpec("pan", "潘金莲", "npc-girl", "zh-CN-XiaoxiaoNeural", "站立", "neutral", 260, 0.86),
        ),
        lines=(
            LineSpec("wuda", "二郎，你总算回来了。阳谷县里人人都在说你打虎。", "smile"),
            LineSpec("wusong", "哥哥在，我心里就安稳。往后有我，谁也欺你不得。", "neutral"),
            LineSpec("pan", "叔叔既有本事，日后可要多照应家里。", "smile"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "武松杀嫂-水浒传-赵季平.mp3",
        background_top=(230, 218, 210),
        background_bottom=(132, 117, 116),
        accent=(136, 56, 66),
        effect="embers",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "心脏怦怦跳.wav", 2.7, 0.14),),
    ),
    SceneSpec(
        scene_id="05",
        title="狮子楼前",
        actors=(
            ActorSpec("wusong", "武松", "general-guard", "zh-CN-YunxiNeural", "站立", "angry", -190, 0.98, True),
            ActorSpec("ximen", "西门庆", "detective-sleek", "zh-CN-YunjianNeural", "站立", "skeptical", 180, 0.9),
        ),
        lines=(
            LineSpec("ximen", "武松，你一个配军，也敢闯我狮子楼？", "skeptical"),
            LineSpec("wusong", "我来不是喝酒，是替哥哥讨命。", "angry", "舞剑"),
            LineSpec("ximen", "要钱要官我都能给，你何必把路走绝？", "excited"),
            LineSpec("wusong", "害命的账，只能用命来还。", "angry", "拳击"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "武松杀嫂-水浒传-赵季平.mp3",
        background_top=(92, 25, 30),
        background_bottom=(25, 8, 12),
        accent=(244, 223, 201),
        effect="slash",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "刀剑、金属碰撞（带回音）_爱给网_aigei_com.mp3", 2.2, 0.78),
            SfxCue(ROOT_DIR / "assets" / "audio" / "格斗打中.wav", 5.0, 0.82),
        ),
    ),
    SceneSpec(
        scene_id="06",
        title="孟州受托",
        actors=(
            ActorSpec("shien", "施恩", "witness-strolling", "zh-CN-YunxiNeural", "站立", "sad", -170, 0.84),
            ActorSpec("wusong", "武松", "general-guard", "zh-CN-YunxiNeural", "站立", "neutral", 110, 0.96),
            ActorSpec("jms", "蒋门神", "emperor-ming", "zh-CN-YunjianNeural", "站立", "angry", 300, 0.9),
        ),
        lines=(
            LineSpec("shien", "蒋门神霸了快活林，我的生意尽被他抢走。", "sad"),
            LineSpec("wusong", "你既敬我一尺，我便替你把这口气争回来。", "neutral"),
            LineSpec("jms", "谁敢动我的场子，我就让谁横着出去！", "angry"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "男儿当自强.mp3",
        background_top=(243, 226, 183),
        background_bottom=(177, 112, 73),
        accent=(115, 53, 22),
        effect="wind",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "心脏怦怦跳.wav", 2.1, 0.13),),
    ),
    SceneSpec(
        scene_id="07",
        title="快活林斗口",
        actors=(
            ActorSpec("wusong", "武松", "general-guard", "zh-CN-YunxiNeural", "站立", "excited", -170, 0.98, True),
            ActorSpec("jms", "蒋门神", "emperor-ming", "zh-CN-YunjianNeural", "站立", "angry", 180, 0.92),
        ),
        lines=(
            LineSpec("jms", "你算什么东西，也敢替施恩出头？", "angry"),
            LineSpec("wusong", "我姓武名松，专打你这种仗势欺人的。", "excited", "人物A飞踢倒人物B"),
            LineSpec("jms", "给我上！", "angry", "拳击"),
            LineSpec("wusong", "来多少，倒多少！", "angry", "拳击"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "男儿当自强.mp3",
        background_top=(248, 227, 179),
        background_bottom=(178, 110, 69),
        accent=(120, 60, 26),
        effect="burst",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "031_26_赤手空拳打斗的声音_爱给网_aigei_com.mp3", 1.4, 0.85),
            SfxCue(ROOT_DIR / "assets" / "audio" / "一拳击中.wav", 3.6, 0.95),
        ),
    ),
    SceneSpec(
        scene_id="08",
        title="飞云浦围杀",
        actors=(
            ActorSpec("wusong", "武松", "general-guard", "zh-CN-YunxiNeural", "站立", "angry", -190, 0.98),
            ActorSpec("killer1", "军汉甲", "official-minister", "zh-CN-YunjianNeural", "蹲下", "angry", 90, 0.86),
            ActorSpec("killer2", "军汉乙", "witness-strolling", "zh-CN-YunxiNeural", "蹲下", "angry", 310, 0.82),
        ),
        lines=(
            LineSpec("killer1", "上头有令，今晚就在这里结果了他。", "angry"),
            LineSpec("killer2", "锁着他还不够？一刀下去最省事。", "angry"),
            LineSpec("wusong", "想要我的命，你们还差得远。", "angry", "翻跟头gif"),
            LineSpec("wusong", "今日谁来害我，我就先送谁上路！", "angry", "拳击"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "最后之战-热血-卢冠廷.mp3",
        background_top=(35, 46, 66),
        background_bottom=(9, 12, 23),
        accent=(204, 231, 255),
        effect="thunder",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "打雷闪电.wav", 0.9, 0.55),
            SfxCue(ROOT_DIR / "assets" / "audio" / "音效 爆炸 爆破 爆发 战斗_爱给网_aigei_com.mp3", 4.0, 0.22),
        ),
    ),
    SceneSpec(
        scene_id="09",
        title="鸳鸯楼夜决",
        actors=(
            ActorSpec("wusong", "武松", "general-guard", "zh-CN-YunxiNeural", "站立", "angry", -190, 0.98, True),
            ActorSpec("zhang", "张都监", "detective-sleek", "zh-CN-YunjianNeural", "站立", "skeptical", 180, 0.88),
        ),
        lines=(
            LineSpec("zhang", "武松，你若肯低头，我还能替你求活路。", "skeptical"),
            LineSpec("wusong", "飞云浦的杀局，是你布的；鸳鸯楼的血债，也由你来还。", "angry", "舞剑"),
            LineSpec("zhang", "你敢！", "angry"),
            LineSpec("wusong", "我已经杀到了这里，还怕再多这一刀么？", "angry", "拳击"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "最后之战-热血-卢冠廷.mp3",
        background_top=(114, 22, 25),
        background_bottom=(30, 6, 7),
        accent=(255, 226, 212),
        effect="fire",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "刀剑、金属碰撞（带回音）_爱给网_aigei_com.mp3", 1.8, 0.82),
            SfxCue(ROOT_DIR / "assets" / "audio" / "格斗打中.wav", 4.6, 0.82),
        ),
    ),
    SceneSpec(
        scene_id="10",
        title="夜走梁山路",
        actors=(
            ActorSpec("wusong", "武松", "general-guard", "zh-CN-YunxiNeural", "跑", "neutral", -130, 0.96),
            ActorSpec("narrator", "江湖回声", "narrator", "zh-CN-YunjianNeural", "行走", "thinking", 210, 0.84, False),
        ),
        lines=(
            LineSpec("narrator", "血路走尽，前面却还有更大的江湖在等你。", "thinking"),
            LineSpec("wusong", "我这一生，宁可站着死，也不肯跪着生。", "neutral"),
            LineSpec("narrator", "于是武松披夜色远去，名字也一步步走进梁山。", "smile"),
        ),
        bgm_path=ROOT_DIR / "assets" / "bgm" / "铁血丹心.mp3",
        background_top=(31, 34, 54),
        background_bottom=(7, 8, 14),
        accent=(224, 228, 242),
        effect="wind",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "暴雨.wav", 0.0, 0.12),),
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


@lru_cache(maxsize=16)
def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for ch in text:
        trial = current + ch
        width = font.getbbox(trial)[2] - font.getbbox(trial)[0]
        if current and width > max_width:
            lines.append(current)
            current = ch
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [text]


@lru_cache(maxsize=64)
def _track(track_name: str) -> poseviz.PoseTrack:
    return poseviz._load_track(poseviz.POSE_DIR / f"{track_name}.pose.json", width=WIDTH, height=HEIGHT)


@lru_cache(maxsize=32)
def _textures(character_id: str) -> poseviz.TexturePack:
    return poseviz._load_texture_pack(character_id)


def _all_head_size() -> int:
    values = [_track(actor.track_name).head_size for scene in SCENES for actor in scene.actors]
    base = int(round(sum(values) / len(values) * 0.88))
    return max(62, min(80, base))


async def _synthesize_tts(text: str, voice: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice, rate="+0%")
            await communicate.save(str(output_path))
            return
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(1.2 * (attempt + 1))
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


def _build_schedule(scene: SceneSpec, scene_dir: Path) -> tuple[list[ScheduledLine], float]:
    actor_map = {actor.actor_id: actor for actor in scene.actors}
    schedule: list[ScheduledLine] = []
    cursor = scene.hold_s
    for index, line in enumerate(scene.lines, start=1):
        actor = actor_map[line.speaker_id]
        tts_path = scene_dir / f"line_{index:02d}.mp3"
        asyncio.run(_synthesize_tts(line.text, actor.voice, tts_path))
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


def _mix_scene_audio(scene: SceneSpec, schedule: list[ScheduledLine], duration_s: float, output_path: Path) -> None:
    ffmpeg = _ffmpeg()
    command = [ffmpeg, "-y", "-stream_loop", "-1", "-i", str(scene.bgm_path)]
    for line in schedule:
        command.extend(["-i", str(line.tts_path)])
    for cue in scene.sfx:
        command.extend(["-i", str(cue.path)])

    filters = [f"[0:a]atrim=0:{duration_s:.3f},asetpts=N/SR/TB,volume=0.16[bgm]"]
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
    filters.append(f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)},alimiter=limit=0.92[aout]")
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


def _gradient_background(image: Image.Image, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    for y in range(image.height):
        alpha = y / max(1, image.height - 1)
        color = tuple(int(top[i] * (1.0 - alpha) + bottom[i] * alpha) for i in range(3))
        draw.line((0, y, image.width, y), fill=color, width=1)


def _draw_effect(draw: ImageDraw.ImageDraw, scene: SceneSpec, progress: float) -> None:
    accent = scene.accent
    if scene.effect == "rain":
        for index in range(16):
            x = (index * 73 + int(progress * 340)) % (WIDTH + 140) - 70
            y = (index * 37 + int(progress * 220)) % HEIGHT
            draw.line((x, y, x - 18, y + 38), fill=(*accent, 90), width=2)
    elif scene.effect == "wind":
        for index in range(4):
            y = 118 + index * 74 + math.sin(progress * 6.0 + index) * 10.0
            draw.arc((64, y - 18, WIDTH - 64, y + 22), 8, 170, fill=(*accent, 110), width=3)
    elif scene.effect == "impact":
        radius = 40 + progress * 130
        cx = WIDTH * 0.56
        cy = HEIGHT * 0.58
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(*accent, 88), width=4)
    elif scene.effect == "slash":
        x0 = WIDTH * (0.2 + progress * 0.34)
        draw.line((x0, HEIGHT * 0.18, x0 + 180, HEIGHT * 0.8), fill=(*accent, 155), width=10)
        draw.line((x0 + 16, HEIGHT * 0.23, x0 + 168, HEIGHT * 0.76), fill=(255, 247, 236, 175), width=4)
    elif scene.effect == "thunder":
        if 0.2 < progress < 0.3:
            draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(235, 242, 255, 90))
        bolt = [(WIDTH * 0.68, 0), (WIDTH * 0.62, 118), (WIDTH * 0.69, 118), (WIDTH * 0.57, 250), (WIDTH * 0.66, 250), (WIDTH * 0.55, 420)]
        draw.line(bolt, fill=(*accent, 180), width=6)
    elif scene.effect == "fire":
        for index in range(9):
            x = 90 + index * 92 + math.sin(progress * 7.0 + index) * 16.0
            flame_h = 80 + (index % 3) * 24
            draw.polygon([(x, HEIGHT), (x - 20, HEIGHT - flame_h * 0.4), (x, HEIGHT - flame_h), (x + 20, HEIGHT - flame_h * 0.4)], fill=(255, 140, 42, 92))
    elif scene.effect == "burst":
        cx = WIDTH * 0.56
        cy = HEIGHT * 0.54
        for index in range(8):
            angle = progress * math.tau + index * (math.tau / 8.0)
            x1 = cx + math.cos(angle) * 40
            y1 = cy + math.sin(angle) * 40
            x2 = cx + math.cos(angle) * 110
            y2 = cy + math.sin(angle) * 110
            draw.line((x1, y1, x2, y2), fill=(*accent, 138), width=5)
    elif scene.effect == "embers":
        for index in range(18):
            x = (index * 53 + int(progress * 250)) % WIDTH
            y = HEIGHT - ((index * 33 + int(progress * 310)) % HEIGHT)
            r = 3 + (index % 3)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(*accent, 122))
    elif scene.effect == "dust":
        for index in range(12):
            x = (index * 81 + int(progress * 180)) % WIDTH
            y = HEIGHT * 0.78 + math.sin(progress * 4.0 + index) * 12.0
            draw.ellipse((x - 18, y - 8, x + 18, y + 8), fill=(*accent, 46))


def _draw_caption(draw: ImageDraw.ImageDraw, scene: SceneSpec, progress: float) -> None:
    title_font = _font(26, bold=True)
    scene_font = _font(22, bold=True)
    draw.rounded_rectangle((28, 24, 470, 126), radius=20, fill=(14, 18, 28, 208))
    draw.text((48, 42), TITLE, fill=(248, 244, 235), font=title_font)
    draw.text((48, 78), f"{scene.scene_id}  {scene.title}", fill=scene.accent, font=scene_font)
    bar_w = int(320 * min(1.0, max(0.0, progress)))
    draw.rounded_rectangle((48, 110, 368, 118), radius=4, fill=(82, 88, 106))
    draw.rounded_rectangle((48, 110, 48 + bar_w, 118), radius=4, fill=scene.accent)


def _draw_subtitle(draw: ImageDraw.ImageDraw, label: str, text: str) -> None:
    name_font = _font(24, bold=True)
    text_font = _font(26, bold=False)
    lines = _wrap_text(f"{label}：{text}", text_font, WIDTH - 140)
    plate_h = 82 + max(0, len(lines) - 1) * 30
    y0 = HEIGHT - plate_h - 28
    draw.rounded_rectangle((38, y0, WIDTH - 38, HEIGHT - 28), radius=18, fill=(16, 18, 24, 220))
    draw.text((60, y0 + 18), label, fill=(255, 226, 172), font=name_font)
    text_y = y0 + 18
    for idx, line in enumerate(lines):
        x = 60 if idx == 0 else 60
        prefix = f"{label}：" if idx == 0 else ""
        content = line[len(prefix) :] if idx == 0 and line.startswith(prefix) else line
        offset_x = 44 if idx == 0 else 0
        draw.text((60 + offset_x, text_y), content, fill=(247, 244, 238), font=text_font)
        text_y += 30


def _active_line(schedule: list[ScheduledLine], t_s: float) -> ScheduledLine | None:
    for item in schedule:
        if item.start_s <= t_s <= item.end_s:
            return item
    return None


def _actor_stage_points(
    actor: ActorSpec,
    track: poseviz.PoseTrack,
    points: dict[str, poseviz.np.ndarray],
) -> dict[str, tuple[float, float]]:
    base = {name: poseviz._stage_point(track, point, width=WIDTH, height=HEIGHT) for name, point in points.items()}
    stage: dict[str, tuple[float, float]] = {}
    anchor_x = WIDTH * 0.5 + actor.x_offset
    for name, (x, y) in base.items():
        stage_x = WIDTH * 0.5 + (x - WIDTH * 0.5) * actor.scale + actor.x_offset
        if actor.mirror:
            stage_x = anchor_x - (stage_x - anchor_x)
        stage_y = GROUND_Y + (y - GROUND_Y) * actor.scale
        stage[name] = (stage_x, stage_y)
    return stage


def _draw_actor(
    image: Image.Image,
    actor: ActorSpec,
    active: ScheduledLine | None,
    t_s: float,
    head_size: int,
) -> None:
    speaking = active is not None and active.speaker_id == actor.actor_id
    track_name = active.track_name if speaking and active.track_name else actor.track_name
    track = _track(track_name)
    if not speaking:
        sample_t = 0.0
    elif track_name in ACTION_TRACKS:
        line_t = 0.0 if active is None else max(0.0, t_s - active.start_s)
        sample_t = (line_t * 0.55) % max(track.total_duration_s, 0.001)
    else:
        line_t = 0.0 if active is None else max(0.0, t_s - active.start_s)
        sample_t = min(track.total_duration_s * 0.12, line_t * 0.08)
    points = poseviz._sample_track(track, sample_t)
    stage_points = _actor_stage_points(actor, track, points)
    textures = _textures(actor.character_id)
    palette = poseviz._palette_for_character(actor.character_id)
    draw = ImageDraw.Draw(image, "RGBA")
    for start, end in poseviz.POSE_EDGES:
        if start not in LEG_POINTS or start not in stage_points or end not in stage_points:
            continue
        draw.line((*stage_points[start], *stage_points[end]), fill=poseviz._edge_color(start, palette), width=poseviz.LIMB_WIDTH, joint="curve")
    for name, (x, y) in stage_points.items():
        if name in {"nose", "left_hip", "right_hip"} or name not in LEG_POINTS:
            continue
        radius = poseviz.JOINT_RADIUS
        color = poseviz._joint_color(name, palette)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    poseviz._draw_torso(draw, stage_points, palette)
    poseviz._draw_torso_texture(image, stage_points, textures)
    draw = ImageDraw.Draw(image, "RGBA")
    for start, end in poseviz.POSE_EDGES:
        if start not in ARM_POINTS or start not in stage_points or end not in stage_points:
            continue
        draw.line((*stage_points[start], *stage_points[end]), fill=poseviz._edge_color(start, palette), width=poseviz.LIMB_WIDTH, joint="curve")
    for name, (x, y) in stage_points.items():
        if name in {"nose", "left_hip", "right_hip"} or name not in ARM_POINTS:
            continue
        radius = poseviz.JOINT_RADIUS
        color = poseviz._joint_color(name, palette)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    mouth_open = speaking and (int(t_s * FPS) % TALK_MOUTH_CYCLE_FRAMES) < (TALK_MOUTH_CYCLE_FRAMES // 2)
    expression = active.expression if speaking else actor.expression
    face_texture = poseviz._load_face_texture(actor.character_id, expression=expression, talking=speaking, mouth_open=mouth_open)
    poseviz._draw_panda_head(image, draw, stage_points, size=int(head_size * actor.scale), textures=textures, face_texture=face_texture)
    head_center = poseviz._head_center(stage_points)
    if head_center is not None:
        label_font = _font(20, bold=True)
        bubble_color = (255, 230, 172, 220) if speaking else (18, 22, 32, 188)
        text_color = (78, 42, 20) if speaking else (240, 240, 240)
        text_w = label_font.getbbox(actor.label)[2] - label_font.getbbox(actor.label)[0]
        cx, cy = head_center
        box = (cx - text_w * 0.55 - 18, cy - head_size * actor.scale * 1.18, cx + text_w * 0.55 + 18, cy - head_size * actor.scale * 0.86)
        draw.rounded_rectangle(box, radius=12, fill=bubble_color)
        draw.text((box[0] + 16, box[1] + 6), actor.label, fill=text_color, font=label_font)


def _render_scene_frame(scene: SceneSpec, schedule: list[ScheduledLine], duration_s: float, t_s: float, head_size: int) -> Image.Image:
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 255))
    _gradient_background(image, scene.background_top, scene.background_bottom)
    draw = ImageDraw.Draw(image, "RGBA")
    progress = 0.0 if duration_s <= 1e-6 else max(0.0, min(1.0, t_s / duration_s))
    _draw_effect(draw, scene, progress)
    active = _active_line(schedule, t_s)
    for actor in scene.actors:
        if not actor.visible:
            continue
        _draw_actor(image, actor, active, t_s, head_size)
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_caption(draw, scene, progress)
    if active is not None:
        _draw_subtitle(draw, active.speaker_label, active.text)
    return image.convert("RGB")


def _render_scene_video(scene: SceneSpec, schedule: list[ScheduledLine], duration_s: float, head_size: int, output_path: Path) -> None:
    proc = poseviz._open_ffmpeg_stream(FPS, WIDTH, HEIGHT, output_path)
    try:
        assert proc.stdin is not None
        total_frames = max(1, int(round(duration_s * FPS)))
        for frame_index in range(total_frames):
            t_s = frame_index / FPS
            frame = _render_scene_frame(scene, schedule, duration_s, t_s, head_size)
            proc.stdin.write(frame.tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")


def _mux_scene(video_path: Path, audio_path: Path, output_path: Path) -> None:
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
            str(output_path),
        ],
        check=True,
    )


def _concat_scenes(scene_files: list[Path], output_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False, dir=str(TMP_DIR)) as handle:
        for path in scene_files:
            handle.write(f"file '{path.resolve()}'\n")
        concat_list = Path(handle.name)
    try:
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
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
        )
    finally:
        concat_list.unlink(missing_ok=True)


def render_story(output_path: Path, *, force: bool = False) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    head_size = _all_head_size()
    scene_outputs: list[Path] = []
    for scene in SCENES:
        paths = _scene_paths(scene)
        if not force and paths["scene_mp4"].exists() and paths["scene_mp4"].stat().st_size > 0:
            scene_outputs.append(paths["scene_mp4"])
            print(paths["scene_mp4"])
            continue
        schedule, duration_s = _build_schedule(scene, paths["dir"])
        _mix_scene_audio(scene, schedule, duration_s, paths["audio"])
        _render_scene_video(scene, schedule, duration_s, head_size, paths["video"])
        _mux_scene(paths["video"], paths["audio"], paths["scene_mp4"])
        scene_outputs.append(paths["scene_mp4"])
        print(paths["scene_mp4"])
    _concat_scenes(scene_outputs, output_path)
    print(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a multi-character Water Margin dialogue story using DNN pose stickman actors with Chinese subtitles, TTS, BGM, SFX, and action blocking.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    render_story(args.output.resolve(), force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
