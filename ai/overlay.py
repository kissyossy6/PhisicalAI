"""Drawing helpers: Gemini annotations on the camera image and the video layout."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_DIRS = ["/System/Library/Fonts", "/usr/share/fonts/opentype/noto", "/usr/share/fonts/truetype/noto"]
_FONT_FILES = {
    "bold": ["ヒラギノ角ゴシック W6.ttc", "NotoSansCJK-Bold.ttc"],
    "regular": ["ヒラギノ角ゴシック W3.ttc", "NotoSansCJK-Regular.ttc"],
}


def font(size: int, weight: str = "regular"):
    for d in _FONT_DIRS:
        for f in _FONT_FILES[weight]:
            p = os.path.join(d, f)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


BLUE = (41, 98, 255)
PINK = (255, 64, 129)
GREEN = (0, 200, 83)
RED = (230, 40, 40)
WHITE = (255, 255, 255)


def _xy(point, size):
    return point[1] / 1000 * size[0], point[0] / 1000 * size[1]


def draw_points(img: Image.Image, points, color=BLUE, radius=7) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    f = font(16, "bold")
    for p in points:
        if not p or "point" not in p:
            continue
        x, y = _xy(p["point"], img.size)
        d.ellipse([x - radius - 6, y - radius - 6, x + radius + 6, y + radius + 6], outline=(*color,), width=2)
        d.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=WHITE, width=2)
        label = p.get("label", "")
        if label:
            tw = d.textlength(label, font=f)
            d.rounded_rectangle([x + 14, y - 13, x + 26 + tw, y + 13], 5, fill=color)
            d.text((x + 20, y - 11), label, font=f, fill=WHITE)
    return img


def draw_plan(img: Image.Image, plan: dict) -> Image.Image:
    """Arrows from each item to the destination Gemini chose for it."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    dests = {x["label"]: x["point"] for x in plan.get("destinations", []) if "point" in x}
    for it in plan.get("items", []):
        dst = dests.get(it.get("destination"))
        if dst is None or "point" not in it:
            continue
        x0, y0 = _xy(it["point"], img.size)
        x1, y1 = _xy(dst, img.size)
        d.line([x0, y0, x1, y1], fill=PINK, width=3)
        ang = np.arctan2(y1 - y0, x1 - x0)
        for s in (2.6, -2.6):
            d.line([x1, y1, x1 + 16 * np.cos(ang + s), y1 + 16 * np.sin(ang + s)], fill=PINK, width=3)
    img = draw_points(img, [{"point": p, "label": l} for l, p in dests.items()], color=PINK)
    return draw_points(img, plan.get("items", []), color=BLUE)


def draw_badge(img: Image.Image, text: str, ok: bool) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    f = font(30, "bold")
    tw = d.textlength(text, font=f)
    d.rounded_rectangle([16, 16, 44 + tw, 66], 10, fill=GREEN if ok else RED)
    d.text((30, 22), text, font=f, fill=WHITE)
    return img


def compose_frame(main: np.ndarray, pip: Image.Image | None, title: str, caption: str, sub: str = "", tag: str = "") -> np.ndarray:
    """1280x720 frame: cinematic view + Gemini view (picture-in-picture) + captions."""
    frame = Image.fromarray(main).convert("RGBA")
    W, H = frame.size
    over = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    # title
    ft = font(22, "bold")
    tw = d.textlength(title, font=ft)
    d.rounded_rectangle([16, 14, 36 + tw, 52], 8, fill=(15, 18, 30, 200))
    d.text((26, 19), title, font=ft, fill=WHITE)
    if tag:
        fs = font(18, "bold")
        sw = d.textlength(tag, font=fs)
        d.rounded_rectangle([16, 60, 36 + sw, 92], 8, fill=(*PINK, 220))
        d.text((26, 64), tag, font=fs, fill=WHITE)
    # caption bar
    if caption or sub:
        d.rectangle([0, H - 104, W, H], fill=(10, 12, 20, 190))
        d.text((28, H - 94), caption, font=font(28, "bold"), fill=WHITE)
        if sub:
            d.text((28, H - 50), sub, font=font(20), fill=(200, 215, 255))
    frame = Image.alpha_composite(frame, over)
    if pip is not None:
        pw = 400
        ph = int(pip.size[1] * pw / pip.size[0])
        p = pip.convert("RGB").resize((pw, ph), Image.LANCZOS)
        x0, y0 = W - pw - 16, 14
        bg = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(bg)
        bd.rounded_rectangle([x0 - 4, y0 - 4, x0 + pw + 4, y0 + ph + 34], 8, fill=(15, 18, 30, 220))
        bd.text((x0 + 6, y0 + ph + 4), "Gemini Robotics-ER 1.6 が見ている画像", font=font(17, "bold"), fill=WHITE)
        frame = Image.alpha_composite(frame, bg)
        frame.paste(p, (x0, y0))
    return np.asarray(frame.convert("RGB"))
