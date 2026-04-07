#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "characters" / "_shared_skins" / "panda_head_pack"


def _new(size: tuple[int, int]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", size, (255, 255, 255, 0))
    return image, ImageDraw.Draw(image)


def _save(name: str, image: Image.Image) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUT_DIR / name)


def build_head() -> None:
    image, draw = _new((720, 720))
    shadow = (0, 0, 0, 34)
    outline = (28, 28, 30, 255)
    head_fill = (250, 248, 242, 255)
    patch = (38, 38, 42, 255)
    inner = (255, 255, 255, 255)
    muzzle = (245, 241, 235, 255)

    draw.ellipse((122, 8, 302, 188), fill=shadow)
    draw.ellipse((418, 8, 598, 188), fill=shadow)
    draw.ellipse((128, 16, 296, 184), fill=patch, outline=outline, width=6)
    draw.ellipse((424, 16, 592, 184), fill=patch, outline=outline, width=6)
    draw.ellipse((56, 92, 664, 676), fill=head_fill, outline=outline, width=10)
    draw.ellipse((180, 224, 328, 416), fill=patch)
    draw.ellipse((392, 224, 540, 416), fill=patch)
    draw.ellipse((232, 270, 270, 320), fill=inner)
    draw.ellipse((450, 270, 488, 320), fill=inner)
    draw.ellipse((216, 392, 504, 556), fill=muzzle, outline=(214, 207, 198, 255), width=6)
    draw.ellipse((330, 430, 390, 474), fill=outline)
    draw.rounded_rectangle((302, 470, 418, 490), radius=10, fill=(70, 70, 74, 255))
    draw.line((360, 470, 360, 520), fill=(90, 90, 94, 255), width=6)
    draw.arc((300, 500, 360, 554), start=8, end=168, fill=(90, 90, 94, 255), width=6)
    draw.arc((360, 500, 420, 554), start=12, end=172, fill=(90, 90, 94, 255), width=6)
    _save("head_base.png", image)


def build_body() -> None:
    image, draw = _new((440, 620))
    outline = (26, 26, 28, 255)
    suit = (36, 36, 40, 255)
    belly = (245, 244, 238, 255)
    accent = (70, 70, 76, 255)

    shell = [
        (220, 14),
        (328, 44),
        (392, 156),
        (398, 292),
        (356, 438),
        (276, 560),
        (164, 586),
        (82, 544),
        (40, 430),
        (34, 260),
        (74, 120),
        (146, 42),
    ]
    draw.polygon(shell, fill=suit, outline=outline)
    draw.ellipse((116, 122, 324, 448), fill=belly, outline=(218, 214, 208, 255), width=6)
    draw.arc((78, 32, 362, 186), start=192, end=348, fill=accent, width=12)
    draw.rounded_rectangle((154, 470, 286, 548), radius=28, fill=(20, 20, 22, 180))
    _save("body.png", image)


def build_arm() -> None:
    image, draw = _new((160, 560))
    outline = (25, 25, 27, 255)
    fill = (34, 34, 38, 255)
    highlight = (82, 82, 90, 110)
    draw.rounded_rectangle((38, 18, 122, 542), radius=42, fill=fill, outline=outline, width=6)
    draw.rounded_rectangle((56, 36, 86, 520), radius=16, fill=highlight)
    _save("arm.png", image)


def build_leg() -> None:
    image, draw = _new((180, 620))
    outline = (23, 23, 24, 255)
    fill = (30, 30, 33, 255)
    highlight = (88, 88, 94, 100)
    points = [
        (68, 18),
        (116, 18),
        (138, 110),
        (144, 236),
        (138, 386),
        (122, 592),
        (58, 592),
        (42, 386),
        (36, 236),
        (42, 110),
    ]
    draw.polygon(points, fill=fill, outline=outline)
    draw.rounded_rectangle((70, 50, 96, 560), radius=12, fill=highlight)
    _save("leg.png", image)


def build_hand() -> None:
    image, draw = _new((220, 220))
    outline = (26, 26, 28, 255)
    fill = (248, 247, 242, 255)
    cuff = (38, 38, 42, 255)
    draw.rounded_rectangle((76, 54, 176, 182), radius=42, fill=fill, outline=outline, width=6)
    draw.ellipse((24, 74, 108, 152), fill=fill, outline=outline, width=6)
    draw.rounded_rectangle((104, 146, 176, 206), radius=18, fill=cuff, outline=outline, width=6)
    _save("hand.png", image)


def build_foot() -> None:
    image, draw = _new((280, 150))
    outline = (24, 24, 26, 255)
    fill = (30, 30, 34, 255)
    sole = (234, 232, 226, 255)
    upper = [
        (32, 108),
        (54, 56),
        (142, 32),
        (234, 40),
        (258, 76),
        (248, 120),
        (68, 124),
    ]
    draw.polygon(upper, fill=fill, outline=outline)
    draw.rounded_rectangle((44, 100, 250, 132), radius=14, fill=sole, outline=outline, width=4)
    _save("foot.png", image)


def build_neck() -> None:
    image, draw = _new((120, 220))
    draw.rounded_rectangle((34, 12, 86, 208), radius=26, fill=(32, 32, 36, 255))
    _save("neck.png", image)


def main() -> int:
    build_head()
    build_body()
    build_arm()
    build_leg()
    build_hand()
    build_foot()
    build_neck()
    print(f"generated pack in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
