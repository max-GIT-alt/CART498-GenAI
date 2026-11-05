"""Queue the process-capture workflow and wait for the saved GIF/final image.

This is the command-line equivalent of opening
``workflows/ui/01_sdxl_img2img_process.json`` (or ``01m_micro_img2img_process.json``
for the micro model) in ComfyUI and pressing Queue.

Examples::

    python scripts/run_session.py --anchor 1 --model dreamshaper
    python scripts/run_session.py --anchor 2 --model noobai --seed 7
    python scripts/run_session.py --anchor all --model micro
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_client import images, run  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(ROOT, "workflows", "api")

# file, label, SDXL size, micro (512 px) size — native aspect, no crop
ANCHORS = {
    "1": ("anchor_1_surreal.png", "anchor_1_surreal", (1024, 1024), (512, 512)),
    "2": ("anchor_2_nature.png", "anchor_2_nature", (1024, 680), (512, 340)),
    "3": ("anchor_3_collage.png", "anchor_3_collage", (680, 1024), (340, 512)),
}
# checkpoint, label, graph, default steps, default frame stride
MODELS = {
    "dreamshaper": ("dreamshaper-xl.safetensors", "DreamShaper XL",
                    "01_sdxl_img2img_process.json", 30, 2),
    "noobai": ("NoobAI-XL-v1.0.safetensors", "NoobAI XL",
               "01_sdxl_img2img_process.json", 30, 2),
    "micro": ("DreamShaper8_LCM.safetensors", "Micro LCM (DreamShaper 8)",
              "01m_micro_img2img_process.json", 4, 1),
}


def queue_one(args, anchor):
    image_name, anchor_label, sdxl_size, micro_size = ANCHORS[anchor]
    ckpt_name, model_label, graph, default_steps, default_stride = MODELS[args.model]
    width, height = micro_size if args.model == "micro" else sdxl_size
    with open(os.path.join(API_DIR, graph)) as f:
        api = json.load(f)

    api["1"]["inputs"]["ckpt_name"] = ckpt_name
    api["4"]["inputs"]["image"] = image_name
    api["5"]["inputs"]["width"] = width
    api["5"]["inputs"]["height"] = height
    api["7"]["inputs"].update({
        "seed": args.seed,
        "steps": args.steps or default_steps,
        "denoise": args.denoise,
        "frame_stride": args.stride or default_stride,
        "max_side": args.max_side,
        "fps": args.fps,
        "label": f"{model_label} · anchor {anchor}",
        "anchor_label": anchor_label,
    })

    print(f"[anchor {anchor}] {model_label} · {image_name} at {width}×{height}", flush=True)
    started = time.time()
    history = run(api, timeout=args.timeout)
    elapsed = time.time() - started
    status = history.get("status", {})
    print(f"status: {status.get('status_str', 'unknown')}  ({elapsed:.1f}s)")
    for filename, subfolder, kind in images(history):
        print(f"  {kind:6s} {subfolder + '/' if subfolder else ''}{filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", default="1", choices=sorted(ANCHORS) + ["all"])
    parser.add_argument("--model", default="dreamshaper", choices=sorted(MODELS))
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--steps", type=int, default=None,
                        help="sampling steps (default: 30 for SDXL, 4 for micro)")
    parser.add_argument("--denoise", type=float, default=0.75)
    parser.add_argument("--stride", type=int, default=None,
                        help="capture every Nth sampling step (default: 2 SDXL, 1 micro)")
    parser.add_argument("--max-side", type=int, default=512,
                        help="longest side of the GIF frames")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="seconds to wait for each run")
    args = parser.parse_args()

    anchors = sorted(ANCHORS) if args.anchor == "all" else [args.anchor]
    for anchor in anchors:
        queue_one(args, anchor)
    print(f"session gallery: {os.path.join(ROOT, 'outputs', 'session', 'session.html')}")
    print(f"export archive:  {os.path.join(ROOT, 'outputs', 'session', 'session-export.zip')}")


if __name__ == "__main__":
    main()
