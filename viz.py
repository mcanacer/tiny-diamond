"""Small image helpers for inspection and evaluation (PIL only)."""
from typing import List

import numpy as np
from PIL import Image, ImageDraw


def upscale(img: np.ndarray, factor: int = 4) -> np.ndarray:
    return img.repeat(factor, 0).repeat(factor, 1)


def labeled_row(frames: List[np.ndarray], labels: List[str], factor: int = 4, pad: int = 4) -> Image.Image:
    tiles = [upscale(f, factor) for f in frames]
    h, w = tiles[0].shape[:2]
    canvas = Image.new("RGB", (len(tiles) * (w + pad) + pad, h + 20 + pad), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        x = pad + i * (w + pad)
        canvas.paste(Image.fromarray(t), (x, 18))
        d.text((x, 3), lab, fill=(0, 0, 0))
    return canvas


def stack_rows(rows: List[Image.Image]) -> Image.Image:
    w = max(r.width for r in rows)
    out = Image.new("RGB", (w, sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out
