#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import edge_tts
from PIL import Image, ImageDraw

import generate_actions_pose_reconstruction as poseviz


ROOT_DIR = Path(__file__).resolve().parents[1]
TMP_DIR = ROOT_DIR / "tmp" / "direct_runs" / "wusong_pose_story"
OUTPUT_DEFAULT = ROOT_DIR / "outputs" / "wusong_pose_story.mp4"
FPS = 24
WIDTH = 960
HEIGHT = 540
VOICE = "zh-CN-YunjianNeural"
TITLE = "水浒故事·武松血路"


@dataclass(frozen=True)
class SfxCue:
    path: Path
    offset_s: float
    volume: float = 0.8


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    title: str
    narration: str
    track_name: str
    character_id: str
    expression: str
    bgm_path: Path
    background_top: tuple[int, int, int]
    background_bottom: tuple[int, int, int]
    accent: tuple[int, int, int]
    effect: str
    sfx: tuple[SfxCue, ...] = field(default_factory=tuple)
    hold_s: float = 0.45


SCENES: list[SceneSpec] = [
    SceneSpec(
        scene_id="01",
        title="归乡入店",
        narration="水浒传里，武松自柴进庄上归来，路过景阳冈前的酒店，酒意未起，豪气已先压住满堂看客。",
        track_name="行走",
        character_id="farmer-old",
        expression="smile",
        bgm_path=ROOT_DIR / "assets" / "bgm" / "王进打高俅-赵季平-水浒传.mp3",
        background_top=(246, 232, 211),
        background_bottom=(214, 182, 139),
        accent=(128, 76, 42),
        effect="dust",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "潺潺流水声.wav", 0.0, 0.18),),
    ),
    SceneSpec(
        scene_id="02",
        title="连饮十八碗",
        narration="店家见他气概逼人，只得连连斟酒。武松越喝越热，十八碗下肚，心中只剩一句，山上若有虎，我便去会它。",
        track_name="太极",
        character_id="farmer-old",
        expression="excited",
        bgm_path=ROOT_DIR / "assets" / "bgm" / "王进打高俅-赵季平-水浒传.mp3",
        background_top=(255, 241, 214),
        background_bottom=(221, 183, 126),
        accent=(171, 98, 54),
        effect="wind",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "心脏怦怦跳.wav", 1.4, 0.2),),
    ),
    SceneSpec(
        scene_id="03",
        title="独上景阳冈",
        narration="夜色压下山林，榜文在风里乱响。武松提着哨棒独自上冈，脚下越稳，四面风声却越像野兽的喘息。",
        track_name="跑",
        character_id="general-guard",
        expression="skeptical",
        bgm_path=ROOT_DIR / "assets" / "bgm" / "历史的天空-古筝-三国演义片尾曲.mp3",
        background_top=(47, 54, 78),
        background_bottom=(14, 18, 31),
        accent=(202, 212, 242),
        effect="rain",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "暴雨.wav", 0.0, 0.2),
            SfxCue(ROOT_DIR / "assets" / "audio" / "打雷闪电.wav", 2.2, 0.65),
        ),
    ),
    SceneSpec(
        scene_id="04",
        title="猛虎扑人",
        narration="忽然一阵腥风扑面，吊睛白额虎从乱树后猛然跃出。武松酒意尽散，脚下一拧，正面对上这头食人的恶兽。",
        track_name="人物A飞踢倒人物B",
        character_id="general-guard",
        expression="angry",
        bgm_path=ROOT_DIR / "assets" / "bgm" / "王进打高俅-赵季平-水浒传.mp3",
        background_top=(59, 49, 39),
        background_bottom=(19, 15, 12),
        accent=(238, 143, 73),
        effect="impact",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "狗喘粗气.wav", 0.2, 0.25),
            SfxCue(ROOT_DIR / "assets" / "audio" / "一拳击中.wav", 2.0, 0.9),
        ),
    ),
    SceneSpec(
        scene_id="05",
        title="拳打景阳虎",
        narration="哨棒打折之后，武松索性弃棒上手，拽住虎头连拳带肘狠狠干下去，硬把猛虎压进泥里，打出了景阳冈第一场惊雷。",
        track_name="拳击",
        character_id="general-guard",
        expression="angry",
        bgm_path=ROOT_DIR / "assets" / "bgm" / "王进打高俅-赵季平-水浒传.mp3",
        background_top=(116, 83, 57),
        background_bottom=(38, 26, 19),
        accent=(255, 202, 130),
        effect="impact",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "031_26_赤手空拳打斗的声音_爱给网_aigei_com.mp3", 0.6, 0.8),
            SfxCue(ROOT_DIR / "assets" / "audio" / "格斗打中.wav", 1.9, 0.8),
        ),
    ),
    SceneSpec(
        scene_id="06",
        title="兄仇血案",
        narration="打虎成名之后，武松回到阳谷县，与兄长武大重逢。可好景不长，潘金莲与西门庆勾连成祸，终究害死了武大郎。",
        track_name="行走",
        character_id="official-minister",
        expression="neutral",
        bgm_path=ROOT_DIR / "assets" / "bgm" / "武松杀嫂-水浒传-赵季平.mp3",
        background_top=(224, 215, 210),
        background_bottom=(114, 100, 102),
        accent=(113, 38, 44),
        effect="embers",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "心脏怦怦跳.wav", 2.0, 0.16),),
    ),
    SceneSpec(
        scene_id="07",
        title="狮子楼复仇",
        narration="灵前立誓之后，武松提刀直奔狮子楼。西门庆还想靠财势压人，可武松这一回，不打算给任何人留下退路。",
        track_name="舞剑",
        character_id="official-minister",
        expression="angry",
        bgm_path=ROOT_DIR / "assets" / "bgm" / "武松杀嫂-水浒传-赵季平.mp3",
        background_top=(96, 23, 29),
        background_bottom=(24, 9, 12),
        accent=(242, 219, 197),
        effect="slash",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "刀剑、金属碰撞（带回音）_爱给网_aigei_com.mp3", 1.2, 0.8),
            SfxCue(ROOT_DIR / "assets" / "audio" / "格斗打中.wav", 2.6, 0.75),
        ),
    ),
    SceneSpec(
        scene_id="08",
        title="醉打蒋门神",
        narration="后来武松被发配孟州，为报施恩之恩，醉闯快活林。蒋门神仗着横肉与人多势众，却撞上了武松真正的凶气。",
        track_name="人物A飞踢倒人物B",
        character_id="general-guard",
        expression="excited",
        bgm_path=ROOT_DIR / "assets" / "bgm" / "男儿当自强.mp3",
        background_top=(249, 227, 178),
        background_bottom=(179, 111, 71),
        accent=(116, 52, 22),
        effect="burst",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "031_26_赤手空拳打斗的声音_爱给网_aigei_com.mp3", 0.7, 0.85),
            SfxCue(ROOT_DIR / "assets" / "audio" / "一拳击中.wav", 2.3, 0.95),
        ),
    ),
    SceneSpec(
        scene_id="09",
        title="飞云浦反杀",
        narration="张都监设下毒计，要在飞云浦黑掉武松的命。可武松锁链在身，仍旧翻身暴起，把来害他的军汉一个个逼到绝路。",
        track_name="翻跟头gif",
        character_id="general-guard",
        expression="angry",
        bgm_path=ROOT_DIR / "assets" / "bgm" / "最后之战-热血-卢冠廷.mp3",
        background_top=(36, 47, 66),
        background_bottom=(9, 13, 23),
        accent=(203, 230, 255),
        effect="thunder",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "打雷闪电.wav", 1.4, 0.55),
            SfxCue(ROOT_DIR / "assets" / "audio" / "音效 爆炸 爆破 爆发 战斗_爱给网_aigei_com.mp3", 2.1, 0.28),
        ),
    ),
    SceneSpec(
        scene_id="10",
        title="血溅鸳鸯楼",
        narration="飞云浦反杀之后，武松不逃反进，提着血刃当夜直扑鸳鸯楼。那一夜，楼上楼下都是他压了太久的怒火。",
        track_name="舞剑",
        character_id="general-guard",
        expression="angry",
        bgm_path=ROOT_DIR / "assets" / "bgm" / "最后之战-热血-卢冠廷.mp3",
        background_top=(115, 23, 26),
        background_bottom=(31, 6, 8),
        accent=(255, 225, 211),
        effect="fire",
        sfx=(
            SfxCue(ROOT_DIR / "assets" / "audio" / "刀剑、金属碰撞（带回音）_爱给网_aigei_com.mp3", 0.8, 0.85),
            SfxCue(ROOT_DIR / "assets" / "audio" / "音效 爆炸 爆破 爆发 战斗_爱给网_aigei_com.mp3", 2.5, 0.22),
        ),
    ),
    SceneSpec(
        scene_id="11",
        title="夜走江湖",
        narration="杀尽仇敌之后，武松披风而去。前路依旧是刀山火海，可这条血路，也把他一步步逼向更辽阔的江湖与梁山。",
        track_name="跑",
        character_id="narrator",
        expression="skeptical",
        bgm_path=ROOT_DIR / "assets" / "bgm" / "铁血丹心.mp3",
        background_top=(32, 34, 54),
        background_bottom=(6, 7, 13),
        accent=(225, 228, 243),
        effect="wind",
        sfx=(SfxCue(ROOT_DIR / "assets" / "audio" / "暴雨.wav", 0.0, 0.14),),
    ),
]
TALK_MOUTH_CYCLE_FRAMES = 4


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


async def _synthesize_tts(text: str, output_path: Path) -> None:
    communicate = edge_tts.Communicate(text=text, voice=VOICE, rate="+0%")
    await communicate.save(str(output_path))


def _scene_paths(scene: SceneSpec) -> dict[str, Path]:
    scene_dir = TMP_DIR / f"scene_{scene.scene_id}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": scene_dir,
        "tts": scene_dir / "tts.mp3",
        "audio": scene_dir / "scene_audio.m4a",
        "video": scene_dir / "scene_video.mp4",
        "scene_mp4": scene_dir / f"{scene.scene_id}.mp4",
    }


def _mix_scene_audio(scene: SceneSpec, tts_path: Path, output_path: Path) -> tuple[float, float]:
    tts_duration = _ffprobe_duration(tts_path)
    scene_duration = max(tts_duration + scene.hold_s + 0.7, 2.8)
    ffmpeg = _ffmpeg()
    command = [ffmpeg, "-y", "-i", str(tts_path), "-stream_loop", "-1", "-i", str(scene.bgm_path)]
    for cue in scene.sfx:
        command.extend(["-i", str(cue.path)])
    filters = [
        f"[1:a]atrim=0:{scene_duration:.3f},asetpts=N/SR/TB,volume=0.18[bgm]",
        f"[0:a]adelay={int(scene.hold_s * 1000)}|{int(scene.hold_s * 1000)},volume=1.15[tts]",
    ]
    mix_inputs = ["[tts]", "[bgm]"]
    for index, cue in enumerate(scene.sfx, start=2):
        label = f"sfx{index}"
        delay_ms = int(cue.offset_s * 1000)
        filters.append(f"[{index}:a]adelay={delay_ms}|{delay_ms},volume={cue.volume:.3f}[{label}]")
        mix_inputs.append(f"[{label}]")
    filters.append(f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)},alimiter=limit=0.92[aout]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[aout]",
            "-t",
            f"{scene_duration:.3f}",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_path),
        ]
    )
    subprocess.run(command, check=True)
    return scene_duration, tts_duration


def _gradient_background(image: Image.Image, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    for y in range(image.height):
        alpha = y / max(1, image.height - 1)
        color = tuple(int(top[i] * (1.0 - alpha) + bottom[i] * alpha) for i in range(3))
        draw.line((0, y, image.width, y), fill=color, width=1)


def _draw_effect(draw: ImageDraw.ImageDraw, scene: SceneSpec, progress: float, width: int, height: int) -> None:
    accent = scene.accent
    if scene.effect == "rain":
        for index in range(16):
            x = (index * 71 + int(progress * 320)) % (width + 120) - 60
            y = (index * 37 + int(progress * 210)) % height
            draw.line((x, y, x - 18, y + 36), fill=(*accent, 90), width=2)
    elif scene.effect == "wind":
        for index in range(4):
            y = 120 + index * 72 + math.sin(progress * 6.0 + index) * 10.0
            draw.arc((60, y - 20, width - 60, y + 24), 8, 168, fill=(*accent, 110), width=3)
    elif scene.effect == "impact":
        radius = 40 + progress * 140
        cx = width * 0.56
        cy = height * 0.6
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(*accent, 80), width=4)
    elif scene.effect == "slash":
        x0 = width * (0.18 + progress * 0.36)
        draw.line((x0, height * 0.2, x0 + 180, height * 0.82), fill=(*accent, 160), width=10)
        draw.line((x0 + 18, height * 0.24, x0 + 166, height * 0.78), fill=(255, 245, 233, 180), width=4)
    elif scene.effect == "thunder":
        if 0.24 < progress < 0.32:
            draw.rectangle((0, 0, width, height), fill=(235, 242, 255, 92))
        bolt = [(width * 0.7, 0), (width * 0.62, 120), (width * 0.69, 120), (width * 0.57, 250), (width * 0.66, 250), (width * 0.54, 420)]
        draw.line(bolt, fill=(*accent, 180), width=6)
    elif scene.effect == "fire":
        for index in range(9):
            x = 90 + index * 92 + math.sin(progress * 7.0 + index) * 18.0
            flame_h = 80 + (index % 3) * 26
            draw.polygon(
                [(x, height), (x - 20, height - flame_h * 0.4), (x, height - flame_h), (x + 20, height - flame_h * 0.4)],
                fill=(255, 142, 42, 92),
            )
    elif scene.effect == "burst":
        cx = width * 0.58
        cy = height * 0.55
        for index in range(8):
            angle = progress * math.tau + index * (math.tau / 8.0)
            x1 = cx + math.cos(angle) * 40
            y1 = cy + math.sin(angle) * 40
            x2 = cx + math.cos(angle) * 110
            y2 = cy + math.sin(angle) * 110
            draw.line((x1, y1, x2, y2), fill=(*accent, 140), width=5)
    elif scene.effect == "embers":
        for index in range(18):
            x = (index * 53 + int(progress * 250)) % width
            y = height - ((index * 33 + int(progress * 310)) % height)
            r = 3 + (index % 3)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(*accent, 120))
    elif scene.effect == "dust":
        for index in range(12):
            x = (index * 81 + int(progress * 180)) % width
            y = height * 0.78 + math.sin(progress * 4.0 + index) * 12.0
            draw.ellipse((x - 18, y - 8, x + 18, y + 8), fill=(*accent, 48))


def _draw_caption(draw: ImageDraw.ImageDraw, scene: SceneSpec, progress: float) -> None:
    box_color = (14, 18, 28, 208)
    draw.rounded_rectangle((28, 26, 436, 116), radius=20, fill=box_color)
    draw.text((48, 42), TITLE, fill=(248, 244, 235))
    draw.text((48, 74), f"{scene.scene_id}  {scene.title}", fill=scene.accent)
    bar_w = int(280 * min(1.0, max(0.0, progress)))
    draw.rounded_rectangle((48, 104, 328, 112), radius=4, fill=(82, 88, 106))
    draw.rounded_rectangle((48, 104, 48 + bar_w, 112), radius=4, fill=scene.accent)


def _draw_subtitle(draw: ImageDraw.ImageDraw, text: str, width: int, height: int) -> None:
    draw.rounded_rectangle((42, height - 108, width - 42, height - 34), radius=18, fill=(16, 18, 24, 214))
    draw.text((62, height - 90), text, fill=(247, 244, 238))


def _fixed_story_head_size() -> int:
    head_sizes = []
    for scene in SCENES:
        track = poseviz._load_track(poseviz.POSE_DIR / f"{scene.track_name}.pose.json", width=WIDTH, height=HEIGHT)
        head_sizes.append(track.head_size)
    if not head_sizes:
        return 78
    return max(72, min(92, int(round(sum(head_sizes) / len(head_sizes)))))


def _scene_face_texture(scene: SceneSpec, t_s: float, speech_duration: float) -> Image.Image:
    speech_start = scene.hold_s
    speaking = speech_start <= t_s <= (speech_start + speech_duration)
    mouth_open = speaking and (int(t_s * FPS) % TALK_MOUTH_CYCLE_FRAMES) < (TALK_MOUTH_CYCLE_FRAMES // 2)
    return poseviz._load_face_texture(
        scene.character_id,
        expression=scene.expression,
        talking=speaking,
        mouth_open=mouth_open,
    )


def _render_scene_frame(
    scene: SceneSpec,
    track: poseviz.PoseTrack,
    textures: poseviz.TexturePack,
    t_s: float,
    duration_s: float,
    speech_duration: float,
    head_size: int,
) -> Image.Image:
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 255))
    _gradient_background(image, scene.background_top, scene.background_bottom)
    draw = ImageDraw.Draw(image, "RGBA")
    progress = 0.0 if duration_s <= 1e-6 else max(0.0, min(1.0, t_s / duration_s))
    _draw_effect(draw, scene, progress, WIDTH, HEIGHT)
    points = poseviz._sample_track(track, t_s % max(0.001, track.total_duration_s))
    stage_points = {name: poseviz._stage_point(track, point, width=WIDTH, height=HEIGHT) for name, point in points.items()}
    poseviz._draw_torso(draw, stage_points)
    poseviz._draw_torso_texture(image, stage_points, textures)
    draw = ImageDraw.Draw(image, "RGBA")
    for start, end, color in poseviz.POSE_EDGES:
        if start not in stage_points or end not in stage_points:
            continue
        draw.line((*stage_points[start], *stage_points[end]), fill=color, width=poseviz.LIMB_WIDTH, joint="curve")
    for name, (x, y) in stage_points.items():
        radius = poseviz.JOINT_RADIUS if name != "nose" else poseviz.JOINT_RADIUS - 1
        color = poseviz.KEYPOINT_COLORS.get(name, (245, 245, 245))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    poseviz._draw_panda_head(
        image,
        draw,
        stage_points,
        size=head_size,
        textures=textures,
        face_texture=_scene_face_texture(scene, t_s, speech_duration),
    )
    _draw_caption(draw, scene, progress)
    _draw_subtitle(draw, scene.narration, WIDTH, HEIGHT)
    return image.convert("RGB")


def _render_scene_video(scene: SceneSpec, duration_s: float, speech_duration: float, head_size: int, output_path: Path) -> None:
    track = poseviz._load_track(poseviz.POSE_DIR / f"{scene.track_name}.pose.json", width=WIDTH, height=HEIGHT)
    textures = poseviz._load_texture_pack(scene.character_id)
    proc = poseviz._open_ffmpeg_stream(FPS, WIDTH, HEIGHT, output_path)
    try:
        assert proc.stdin is not None
        total_frames = max(1, int(round(duration_s * FPS)))
        for frame_index in range(total_frames):
            t_s = frame_index / FPS
            frame = _render_scene_frame(scene, track, textures, t_s, duration_s, speech_duration, head_size)
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


def render_story(output_path: Path) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    scene_outputs: list[Path] = []
    head_size = _fixed_story_head_size()
    for scene in SCENES:
        paths = _scene_paths(scene)
        asyncio.run(_synthesize_tts(scene.narration, paths["tts"]))
        scene_duration, tts_duration = _mix_scene_audio(scene, paths["tts"], paths["audio"])
        _render_scene_video(scene, scene_duration, tts_duration, head_size, paths["video"])
        _mux_scene(paths["video"], paths["audio"], paths["scene_mp4"])
        scene_outputs.append(paths["scene_mp4"])
        print(paths["scene_mp4"])
    _concat_scenes(scene_outputs, output_path)
    print(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a 10+ scene Water Margin story using DNN pose stickman actors with TTS, BGM, effects, and SFX.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    render_story(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
