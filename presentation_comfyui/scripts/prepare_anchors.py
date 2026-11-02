"""Normalise the three supplied images for the demo.

The originals have different aspect ratios (1:1, 3:2, ~2:3). Rather than crop or
pad them, each anchor is resized so its long side is 1024 px (dimensions rounded
to a multiple of 8). Every model sees the full image in its native aspect, which
keeps the comparison honest and avoids letterboxing artefacts in the results.

Source images live in images/source/, prepared anchors are written to images/.
"""
import os

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "images", "source")
DST = os.path.join(ROOT, "images")

ANCHORS = [
    ("image0.jpeg", "anchor_1_surreal.png"),
    ("_TAK2500.jpg", "anchor_2_nature.png"),
    ("diffusion_target.png", "anchor_3_collage.png"),
]


def fit_long(im, long=1024):
    im = im.convert("RGB")
    w, h = im.size
    r = long / max(w, h)
    nw = max(256, round(w * r / 8) * 8)
    nh = max(256, round(h * r / 8) * 8)
    return im.resize((nw, nh), Image.LANCZOS)


def main():
    for src, dst in ANCHORS:
        im = Image.open(os.path.join(SRC, src))
        out = fit_long(im)
        path = os.path.join(DST, dst)
        out.save(path)
        print(f"{src} {im.size} -> {dst} {out.size}")


if __name__ == "__main__":
    main()
