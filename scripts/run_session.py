"""Queue the process-capture workflow and wait for the saved GIF/final image.

This is the command-line equivalent of opening
``workflows/ui/01_sdxl_img2img_process.json`` in ComfyUI and pressing Queue.

Examples::

    python scripts/run_session.py --anchor 1 --model dreamshaper
    python scripts/run_session.py --anchor 2 --model noobai --seed 7
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_client import images, run  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_PATH = os.path.join(ROOT, "workflows", "api", "01_sdxl_img2img_process.json")

ANCHORS = {
    "1": ("anchor_1_surreal.png", "anchor_1_surreal", 1024, 1024),
    "2": ("anchor_2_nature.png", "anchor_2_nature", 1024, 680),
    "3": ("anchor_3_collage.png", "anchor_3_collage", 680, 1024),
}
MODELS = {
    "dreamshaper": ("dreamshaper-xl.safetensors", "DreamShaper XL"),
    "noobai": ("NoobAI-XL-v1.0.safetensors", "NoobAI XL"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", default="1", choices=sorted(ANCHORS))
    parser.add_argument("--model", default="dreamshaper", choices=sorted(MODELS))
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--denoise", type=float, default=0.75)
    parser.add_argument("--stride", type=int, default=2,
                        help="capture every Nth sampling step")
    parser.add_argument("--max-side", type=int, default=512,
                        help="longest side of the GIF frames")
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args()

    api = json.load(open(API_PATH))
    image_name, anchor_label, width, height = ANCHORS[args.anchor]
    ckpt_name, model_label = MODELS[args.model]

    api["1"]["inputs"]["ckpt_name"] = ckpt_name
    api["4"]["inputs"]["image"] = image_name
    api["5"]["inputs"]["width"] = width
    api["5"]["inputs"]["height"] = height
    api["7"]["inputs"].update({
        "seed": args.seed,
        "steps": args.steps,
        "denoise": args.denoise,
        "frame_stride": args.stride,
        "max_side": args.max_side,
        "fps": args.fps,
        "label": f"{model_label} · anchor {args.anchor}",
        "anchor_label": anchor_label,
    })

    started = time.time()
    history = run(api, timeout=900)
    elapsed = time.time() - started
    status = history.get("status", {})
    print(f"status: {status.get('status_str', 'unknown')}  ({elapsed:.1f}s)")
    for filename, subfolder, kind in images(history):
        print(f"  {kind:6s} {subfolder + '/' if subfolder else ''}{filename}")
    print(f"session gallery: {os.path.join(ROOT, 'outputs', 'session', 'session.html')}")
    print(f"export archive:  {os.path.join(ROOT, 'outputs', 'session', 'session-export.zip')}")


if __name__ == "__main__":
    main()
