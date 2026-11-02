"""Generate a prompt grid with the 2 GB micro LCM model and time each image.

The micro model is DreamShaper 8 LCM, a 2.0 GiB fp16 SD1.5 checkpoint with LCM
distillation baked in. It needs 4 steps at 512 px, which makes it fast enough for
live demos while still producing presentable images.

Examples::

    python scripts/run_micro.py                       # four built-in demo prompts
    python scripts/run_micro.py --prompt "a red fox in snow" --batch 4
    python scripts/run_micro.py --steps 6 --width 640 --height 640
"""
import argparse
import json
import os
import re
import sys
import time

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_client import queue, wait  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_PATH = os.path.join(ROOT, "workflows", "api", "04_micro_lcm.json")
OUT = os.path.join(ROOT, "outputs")

PROMPTS = [
    "portrait photo of an elderly fisherman, weathered face, golden hour, shallow depth of field",
    "a red fox in a snowy forest, soft morning light, wildlife photography",
    "a bowl of ramen on a wooden table, steam, soft window light, food photography",
    "a hot air balloon over a lavender field at sunrise, cinematic landscape",
]
FONT = "/usr/share/fonts/noto/NotoSans-Regular.ttf"
BOLD = "/usr/share/fonts/noto/NotoSans-Bold.ttf"


def slug(text, limit=48):
    value = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (value or "prompt")[:limit]


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def wrap(draw, text, width, face):
    words, lines, line = text.split(), [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=face) <= width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def build_sheet(rows, path, steps, size):
    """rows = [(prompt, [image paths], seconds)] -> labelled contact sheet."""
    items = [(prompt, image, elapsed) for prompt, paths, elapsed in rows for image in paths]
    if not items:
        return None
    cell, gap, margin = 384, 12, 24
    label_h, header = 78, 112
    columns = min(4, len(items))
    grid_rows = (len(items) + columns - 1) // columns
    content_w = columns * cell + (columns - 1) * gap
    width = max(content_w + margin * 2, 980)
    height = header + grid_rows * (cell + label_h + gap) + margin - gap
    sheet = Image.new("RGB", (width, height), "#0e0e12")
    draw = ImageDraw.Draw(sheet)
    x0 = (width - content_w) // 2

    draw.text((margin, 22), f"MICRO LCM — {steps} STEPS AT {size} PX",
              font=font(30, True), fill="#ffffff")
    draw.text(
        (margin, 64),
        "DreamShaper 8 LCM (2.0 GiB fp16) · sampler lcm · cfg 2.0 · live-demo model",
        font=font(16), fill="#a7a7b7",
    )

    for index, (prompt, image_path, elapsed) in enumerate(items):
        row, col = divmod(index, columns)
        x = x0 + col * (cell + gap)
        y = header + row * (cell + label_h + gap)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            scale = min(cell / image.width, cell / image.height)
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.LANCZOS,
            )
            tile = Image.new("RGB", (cell, cell), "#15151c")
            tile.paste(image, ((cell - image.width) // 2, (cell - image.height) // 2))
        sheet.paste(tile, (x, y))
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline="#26262f")
        lines = wrap(draw, prompt, cell - 8, font(13))[:2]
        for line_no, line in enumerate(lines):
            draw.text((x, y + cell + 6 + line_no * 18), line,
                      font=font(13), fill="#b9b9c9")
        draw.text((x, y + cell + 47), f"{elapsed:.2f}s", font=font(13, True), fill="#7ee0b0")
    sheet.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", action="append", default=None,
                        help="prompt to render; repeat for several prompts")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1,
                        help="images per prompt (EmptyLatentImage batch size)")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--grid", default=os.path.join(OUT, "00_showcase_micro.png"))
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()

    prompts = args.prompt or PROMPTS
    base = json.load(open(API_PATH))
    latent_id = next(nid for nid, node in base.items()
                     if node["class_type"] == "EmptyLatentImage")
    ks_id = next(nid for nid, node in base.items()
                 if node["class_type"] == "KSampler")
    save_id = next(nid for nid, node in base.items()
                   if node["class_type"] == "SaveImage")
    positive_id = base[ks_id]["inputs"]["positive"][0]
    base[latent_id]["inputs"].update({"width": args.width, "height": args.height,
                                      "batch_size": args.batch})
    base[ks_id]["inputs"].update({"steps": args.steps, "cfg": args.cfg, "seed": args.seed})

    os.makedirs(os.path.join(OUT, "micro"), exist_ok=True)
    rows, timings = [], []
    started_all = time.time()

    for index, prompt in enumerate(prompts, 1):
        api = json.loads(json.dumps(base))
        api[positive_id]["inputs"]["text"] = prompt
        api[save_id]["inputs"]["filename_prefix"] = f"micro/{index:02d}_{slug(prompt)}"
        started = time.time()
        prompt_id = queue(api)
        history = wait(prompt_id, timeout=args.timeout, poll=0.1)
        elapsed = time.time() - started

        paths = []
        for node in history.get("outputs", {}).values():
            for image in node.get("images", []):
                path = os.path.join(OUT, image.get("subfolder", ""), image["filename"])
                if os.path.isfile(path):
                    paths.append(path)
        rows.append((prompt, paths, elapsed))
        timings.append(elapsed)
        print(
            f"[{index}/{len(prompts)}] {elapsed:5.2f}s "
            f"({elapsed / max(1, len(paths)):4.2f}s/image)  {prompt}"
        )

    total = time.time() - started_all
    if not args.no_grid and any(paths for _, paths, _ in rows):
        grid = build_sheet(rows, args.grid, args.steps, f"{args.width}×{args.height}")
        print(f"grid: {grid}")
    print(
        f"total: {total:.2f}s for {len(prompts)} prompt(s) · "
        f"{sum(timings) / len(timings):.2f}s/prompt"
    )
    print("images: outputs/micro/")


if __name__ == "__main__":
    main()
