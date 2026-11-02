"""Run the transformation matrix: 3 anchors x 3 diffusion models, plus a denoise ladder.

Column 1  DreamShaper XL  (SDXL, photoreal/versatile)  -- img2img, denoise 0.75
Column 2  NoobAI XL      (SDXL, anime)                -- img2img, denoise 0.75
Column 3  Qwen-Image-Edit 2511 (20B DiT, instruction edit, 4-step Lightning LoRA)

All three get the same *semantic* target:
    "cinematic fantasy illustration, dramatic lighting, rich colours"

SDXL models receive it as a prompt at a fixed denoise; Qwen receives it as an
instruction. That keeps the comparison about the model, not the wording.
"""
import json
import os
import sys

from PIL import Image

from comfy_client import download, images, run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
WF = os.path.join(ROOT, "workflows", "api")

ANCHORS = ["anchor_1_surreal", "anchor_2_nature", "anchor_3_collage"]
TARGET = "cinematic fantasy illustration, dramatic lighting, rich colours, highly detailed"
NEGATIVE = "lowres, bad anatomy, bad hands, blurry, watermark, text, jpeg artifacts"

SDXL_MODELS = {
    "dreamshaper": "dreamshaper-xl.safetensors",
    "noobai": "NoobAI-XL-v1.0.safetensors",
}
QWEN_MODEL = "qwen-image-edit-2511-Q4_K_M.gguf"
QWEN_INSTRUCTION = (
    "Transform this image into a cinematic fantasy illustration with dramatic lighting "
    "and rich colours. Keep the composition and subject recognisable."
)


def sdxl_img2img(ckpt, image, prompt, denoise=0.75, seed=4242, prefix="sdxl"):
    w, h = Image.open(os.path.join(ROOT, "images", image)).size
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image}},
        "5": {
            "class_type": "ImageScale",
            "inputs": {"image": ["4", 0], "upscale_method": "lanczos", "width": w, "height": h, "crop": "disabled"},
        },
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}},
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": 30,
                "cfg": 6.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


def qwen_edit(image, instruction, seed=4242, prefix="qwen_edit"):
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": QWEN_MODEL}},
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                "strength_model": 1.0,
            },
        },
        "5": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["4", 0], "shift": 3.1}},
        "6": {"class_type": "CFGNorm", "inputs": {"model": ["5", 0], "strength": 1.0, "pre_cfg": False}},
        "7": {"class_type": "LoadImage", "inputs": {"image": image}},
        "8": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["7", 0]}},
        "9": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"clip": ["2", 0], "prompt": instruction, "vae": ["3", 0], "image1": ["8", 0]},
        },
        "10": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["9", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "11": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"clip": ["2", 0], "prompt": "", "vae": ["3", 0], "image1": ["8", 0]},
        },
        "12": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["11", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "13": {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["3", 0]}},
        "14": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["6", 0],
                "positive": ["10", 0],
                "negative": ["12", 0],
                "latent_image": ["13", 0],
                "seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["3", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}},
    }


def save_workflow(name, wf):
    os.makedirs(WF, exist_ok=True)
    with open(os.path.join(WF, name + ".json"), "w") as f:
        json.dump(wf, f, indent=2)


def fetch(h, dest):
    for fn, sub, _ in images(h):
        download(fn, sub, dest)
        print("   saved", dest, flush=True)
        return dest
    raise RuntimeError("no output image in history")


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    sdxl_only = only == "sdxl"
    matrix = os.path.join(OUT, "matrix")
    ladder = os.path.join(OUT, "ladder")
    os.makedirs(matrix, exist_ok=True)
    os.makedirs(ladder, exist_ok=True)

    # Group runs by model so each checkpoint is loaded once.
    for short, ckpt in SDXL_MODELS.items():
        for anchor in ANCHORS:
            name = f"{short}__{anchor}"
            if only and not sdxl_only and only not in name:
                continue
            print(f"[matrix] {name}", flush=True)
            wf = sdxl_img2img(ckpt, anchor + ".png", TARGET, prefix=name)
            save_workflow(name, wf)
            fetch(run(wf), os.path.join(matrix, name + ".png"))

    # Denoise ladder: how much of the anchor survives as denoise rises.
    for d in (0.35, 0.55, 0.75, 0.95):
        if sdxl_only:
            continue
        name = f"noobai__anchor_1_surreal__denoise_{str(d).replace('.', '')}"
        if only and only not in name:
            continue
        print(f"[ladder] {name}", flush=True)
        wf = sdxl_img2img(SDXL_MODELS["noobai"], "anchor_1_surreal.png", TARGET, denoise=d, prefix=name)
        save_workflow(name, wf)
        fetch(run(wf), os.path.join(ladder, name + ".png"))

    for anchor in ANCHORS:
        name = f"qwen_edit__{anchor}"
        if sdxl_only or (only and only not in name):
            continue
        print(f"[matrix] {name}", flush=True)
        wf = qwen_edit(anchor + ".png", QWEN_INSTRUCTION, prefix=name)
        save_workflow(name, wf)
        fetch(run(wf, timeout=2400), os.path.join(matrix, name + ".png"))


if __name__ == "__main__":
    main()
