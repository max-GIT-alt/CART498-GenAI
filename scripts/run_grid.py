"""Render 3 anchors through 4 diffusion models, with a process GIF per run.

    python scripts/run_grid.py                    # 12 GIFs + 12 PNGs
    python scripts/run_grid.py --models micro --anchors 1
    python scripts/run_grid.py --out export/results

The SDXL and Qwen models are optional; the micro model is always available after
``scripts/fetch_models.py``. Results are written as::

    <out>/gifs/<nn>_<model>__<anchor>.gif
    <out>/outputs/<nn>_<model>__<anchor>.png
    <out>/manifest.json
    <out>/index.html
"""
import argparse
import html
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_client import download, queue, wait  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "outputs", "grid4")

TARGET = "cinematic fantasy illustration, dramatic lighting, rich colours, highly detailed"
NEGATIVE = "lowres, bad anatomy, bad hands, blurry, watermark, text, jpeg artifacts"
QWEN_INSTRUCTION = (
    "Transform this image into a cinematic fantasy illustration with dramatic lighting "
    "and rich colours. Keep the composition and subject recognisable."
)

ANCHORS = {
    "anchor_1_surreal": {
        "file": "anchor_1_surreal.png",
        "label": "01 · Surreal digital art",
        "sdxl": (1024, 1024),
        "micro": (512, 512),
    },
    "anchor_2_nature": {
        "file": "anchor_2_nature.png",
        "label": "02 · Nature photograph",
        "sdxl": (1024, 680),
        "micro": (512, 340),
    },
    "anchor_3_collage": {
        "file": "anchor_3_collage.png",
        "label": "03 · Mixed-media collage",
        "sdxl": (680, 1024),
        "micro": (340, 512),
    },
}

MODELS = {
    "dreamshaper": {"label": "DreamShaper XL", "kind": "sdxl"},
    "noobai": {"label": "NoobAI XL", "kind": "sdxl"},
    "qwen_edit": {"label": "Qwen-Image-Edit 2511", "kind": "qwen"},
    "micro": {"label": "Micro LCM (DreamShaper 8)", "kind": "micro"},
}

MODEL_FILES = {
    "dreamshaper": "models/checkpoints/dreamshaper-xl.safetensors",
    "noobai": "models/checkpoints/NoobAI-XL-v1.0.safetensors",
    "qwen_edit": "models/unet/qwen-image-edit-2511-Q4_K_M.gguf",
    "micro": "models/checkpoints/DreamShaper8_LCM.safetensors",
}
QWEN_FILES = [
    "models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "models/vae/qwen_image_vae.safetensors",
    "models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
]


def sdxl_api(model_key, anchor_key, seed):
    anchor = ANCHORS[anchor_key]
    width, height = anchor["sdxl"]
    label = f"{MODELS[model_key]['label']} · {anchor['label']}"
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": MODEL_FILES[model_key].rsplit("/", 1)[-1]}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": TARGET, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": anchor["file"]}},
        "5": {"class_type": "ImageScale",
              "inputs": {"image": ["4", 0], "upscale_method": "lanczos",
                         "width": width, "height": height, "crop": "disabled"}},
        "6": {"class_type": "VAEEncode",
              "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "KSamplerFrames", "inputs": {
            "model": ["1", 0], "vae": ["1", 2], "seed": seed, "steps": 30,
            "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["6", 0],
            "denoise": 0.75, "frame_source": "x0 (denoised prediction)",
            "frame_stride": 2, "max_side": 512, "fps": 8.0,
            "session_name": "grid4", "label": label,
            "anchor_label": anchor_key, "include_anchor": True, "preview": True,
        }},
        "8": {"class_type": "SaveImage", "inputs": {
            "images": ["7", 0],
            "filename_prefix": f"grid4/{model_key}__{anchor_key}",
        }},
        "9": {"class_type": "ShowAnimatedGIF", "inputs": {"gif_path": ["7", 3]}},
    }


def qwen_api(anchor_key, seed):
    anchor = ANCHORS[anchor_key]
    label = f"{MODELS['qwen_edit']['label']} · {anchor['label']}"
    return {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_M.gguf"}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0],
            "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "strength_model": 1.0}},
        "5": {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": ["4", 0], "shift": 3.1}},
        "6": {"class_type": "CFGNorm",
              "inputs": {"model": ["5", 0], "strength": 1.0, "pre_cfg": False}},
        "7": {"class_type": "LoadImage", "inputs": {"image": anchor["file"]}},
        "8": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["7", 0]}},
        "9": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["2", 0], "prompt": QWEN_INSTRUCTION,
            "vae": ["3", 0], "image1": ["8", 0]}},
        "10": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": ["9", 0],
            "reference_latents_method": "index_timestep_zero"}},
        "11": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["2", 0], "prompt": "", "vae": ["3", 0], "image1": ["8", 0]}},
        "12": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": ["11", 0],
            "reference_latents_method": "index_timestep_zero"}},
        "13": {"class_type": "VAEEncode",
               "inputs": {"pixels": ["8", 0], "vae": ["3", 0]}},
        "14": {"class_type": "KSamplerFrames", "inputs": {
            "model": ["6", 0], "vae": ["3", 0], "seed": seed, "steps": 4,
            "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "positive": ["10", 0], "negative": ["12", 0],
            "latent_image": ["13", 0], "denoise": 1.0,
            "frame_source": "x0 (denoised prediction)", "frame_stride": 1,
            "max_side": 512, "fps": 8.0, "session_name": "grid4",
            "label": label, "anchor_label": anchor_key,
            "include_anchor": True, "preview": True,
        }},
        "15": {"class_type": "SaveImage", "inputs": {
            "images": ["14", 0],
            "filename_prefix": f"grid4/qwen_edit__{anchor_key}",
        }},
        "16": {"class_type": "ShowAnimatedGIF", "inputs": {"gif_path": ["14", 3]}},
    }


def micro_api(anchor_key, seed):
    anchor = ANCHORS[anchor_key]
    width, height = anchor["micro"]
    label = f"{MODELS['micro']['label']} · {anchor['label']}"
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "DreamShaper8_LCM.safetensors"}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": TARGET, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": anchor["file"]}},
        "5": {"class_type": "ImageScale",
              "inputs": {"image": ["4", 0], "upscale_method": "lanczos",
                         "width": width, "height": height, "crop": "disabled"}},
        "6": {"class_type": "VAEEncode",
              "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "KSamplerFrames", "inputs": {
            "model": ["1", 0], "vae": ["1", 2], "seed": seed, "steps": 4,
            "cfg": 2.0, "sampler_name": "lcm", "scheduler": "normal",
            "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["6", 0],
            "denoise": 0.75, "frame_source": "x0 (denoised prediction)",
            "frame_stride": 1, "max_side": 512, "fps": 8.0,
            "session_name": "grid4", "label": label,
            "anchor_label": anchor_key, "include_anchor": True, "preview": True,
        }},
        "8": {"class_type": "SaveImage", "inputs": {
            "images": ["7", 0],
            "filename_prefix": f"grid4/micro__{anchor_key}",
        }},
        "9": {"class_type": "ShowAnimatedGIF", "inputs": {"gif_path": ["7", 3]}},
    }


def build_api(model_key, anchor_key, seed):
    kind = MODELS[model_key]["kind"]
    if kind == "sdxl":
        return sdxl_api(model_key, anchor_key, seed)
    if kind == "qwen":
        return qwen_api(anchor_key, seed)
    return micro_api(anchor_key, seed)


def output_links(history, save_id, show_id):
    outputs = history.get("outputs", {})
    final = None
    gif = None
    for image in outputs.get(save_id, {}).get("images", []):
        if image["filename"].lower().endswith(".png"):
            final = (image["filename"], image.get("subfolder", ""))
            break
    for image in outputs.get(show_id, {}).get("images", []):
        if image["filename"].lower().endswith(".gif"):
            gif = (image["filename"], image.get("subfolder", ""))
            break
    if gif is None:
        for node in outputs.values():
            for image in node.get("images", []):
                if image["filename"].lower().endswith(".gif"):
                    gif = (image["filename"], image.get("subfolder", ""))
                    break
    return final, gif


def write_index(out, records):
    cards = []
    for record in records:
        cards.append(f"""
      <article class="card">
        <div class="media">
          <img src="{html.escape(record['gif'])}" alt="process GIF">
          <img src="{html.escape(record['output'])}" alt="final output">
        </div>
        <div class="meta">
          <strong>{html.escape(record['model_label'])}</strong>
          <div class="pills">
            <span>{html.escape(record['anchor_label'])}</span>
            <span>seed {record['seed']}</span>
            <span>{record['steps']} steps</span>
            <span>denoise {record['denoise']}</span>
            <span>{record['elapsed_s']:.1f}s</span>
          </div>
          <div class="links">
            <a href="{html.escape(record['gif'])}" download>GIF</a>
            <a href="{html.escape(record['output'])}" download>PNG</a>
          </div>
        </div>
      </article>""")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3 anchors × 4 models — process GIFs and outputs</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #0e0e12; color: #e8e8f0;
         font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  header {{ padding: 34px 30px 10px; }}
  h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: -0.02em; }}
  .sub {{ color: #9aa3b2; max-width: 900px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
           gap: 16px; padding: 20px 30px 50px; }}
  .card {{ background: #181b23; border: 1px solid #2a2f3a; border-radius: 12px;
           overflow: hidden; }}
  .media {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: #2a2f3a; }}
  .media img {{ width: 100%; height: 100%; object-fit: contain; display: block; background: #0a0b0e; }}
  .meta {{ padding: 11px 13px 13px; }}
  .meta strong {{ color: #fff; }}
  .pills {{ margin: 7px 0; }}
  .pills span {{ display: inline-block; padding: 2px 7px; margin: 0 4px 4px 0;
                 border-radius: 999px; background: #232838; color: #cfd6e4; font-size: 11px; }}
  .links a {{ display: inline-block; margin-right: 8px; color: #ffd166; font-size: 12px; }}
</style></head>
<body>
<header>
  <h1>Three anchors × four diffusion models</h1>
  <div class="sub">Each card pairs the saved denoising GIF with the final output.
  Twelve runs in total: DreamShaper XL, NoobAI XL and Qwen-Image-Edit at the
  original prompt, plus the 4-step micro LCM model.</div>
</header>
<main class="grid">{''.join(cards)}</main>
</body></html>
"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as handle:
        handle.write(doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS),
                        default=["dreamshaper", "noobai", "qwen_edit", "micro"])
    parser.add_argument("--anchors", nargs="+", choices=sorted(ANCHORS), default=sorted(ANCHORS))
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    required = [MODEL_FILES[key] for key in args.models if key in MODEL_FILES]
    if "qwen_edit" in args.models:
        required += QWEN_FILES
    missing = [name for name in required if not os.path.isfile(os.path.join(ROOT, name))]
    if missing:
        raise SystemExit("missing local models:\n  " + "\n  ".join(missing) +
                         "\nsee models/README.md")

    out = os.path.abspath(os.path.expanduser(args.out))
    gifs = os.path.join(out, "gifs")
    outputs = os.path.join(out, "outputs")
    os.makedirs(gifs, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)
    for folder in (gifs, outputs):
        for name in os.listdir(folder):
            if name.lower().endswith((".gif", ".png")):
                os.remove(os.path.join(folder, name))

    jobs = [(model, anchor) for model in args.models for anchor in args.anchors]
    records = []
    started_all = time.time()
    for index, (model_key, anchor_key) in enumerate(jobs, 1):
        api = build_api(model_key, anchor_key, args.seed)
        save_id = "8" if model_key != "qwen_edit" else "15"
        show_id = "9" if model_key != "qwen_edit" else "16"
        started = time.time()
        prompt_id = queue(api)
        history = wait(prompt_id, timeout=1800, poll=0.5)
        elapsed = time.time() - started
        final, gif = output_links(history, save_id, show_id)
        if not final or not gif:
            raise SystemExit(f"missing outputs for {model_key} / {anchor_key}: {history.get('outputs')}")

        stem = f"{index:02d}_{model_key}__{anchor_key}"
        gif_name = stem + ".gif"
        png_name = stem + ".png"
        with open(os.path.join(gifs, gif_name), "wb") as handle:
            handle.write(download(*gif))
        with open(os.path.join(outputs, png_name), "wb") as handle:
            handle.write(download(*final))

        record = {
            "index": index,
            "model": model_key,
            "model_label": MODELS[model_key]["label"],
            "anchor": anchor_key,
            "anchor_label": ANCHORS[anchor_key]["label"],
            "seed": args.seed,
            "steps": 4 if model_key in ("qwen_edit", "micro") else 30,
            "denoise": 1.0 if model_key == "qwen_edit" else 0.75,
            "gif": f"gifs/{gif_name}",
            "output": f"outputs/{png_name}",
            "elapsed_s": round(elapsed, 2),
        }
        records.append(record)
        print(f"[{index}/{len(jobs)}] {elapsed:6.1f}s  {record['model_label']} · {record['anchor_label']}")

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "runs": records,
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    write_index(out, records)
    print(f"\n{len(records)} GIFs + {len(records)} outputs in {time.time() - started_all:.1f}s")
    print(f"index: {os.path.join(out, 'index.html')}")


if __name__ == "__main__":
    main()
