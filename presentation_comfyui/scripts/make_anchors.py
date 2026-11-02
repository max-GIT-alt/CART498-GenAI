"""Generate the three anchor images that the demo is built around.

Optional extra anchor set, kept in images/generated/ so it does not interfere with
the three supplied anchors. They are intentionally different *kinds* of picture so
that the effect of each diffusion model on the anchor is easy to read:

  1. a photorealistic portrait      (photographic detail, human subject)
  2. a flat anime illustration      (line art, cel shading, stylised face)
  3. a 3D diorama / environment     (geometry, lighting, no face)

Run with the demo ComfyUI server already up on port 8190.
"""
import os
import sys

from comfy_client import download, images, run

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images", "generated")

ANCHORS = [
    {
        "name": "gen_anchor_1_photo",
        "model": "dreamshaper-xl.safetensors",
        "positive": (
            "RAW photo, close portrait of an elderly fisherman with a weathered face and "
            "white beard, wearing a bright yellow raincoat, standing in a foggy harbour at "
            "dawn, fishing boats and coiled ropes behind him, soft overcast light, 85mm lens, "
            "shallow depth of field, ultra detailed skin texture, documentary photography"
        ),
        "negative": (
            "anime, cartoon, illustration, painting, 3d render, plastic skin, low quality, "
            "blurry, watermark, text"
        ),
        "steps": 30,
        "cfg": 6.0,
        "seed": 1101,
    },
    {
        "name": "gen_anchor_2_anime",
        "model": "NoobAI-XL-v1.0.safetensors",
        "positive": (
            "1girl, solo, long silver hair, blue eyes, sailor school uniform, red ribbon, "
            "standing under blooming cherry blossom trees, petals falling, gentle smile, "
            "looking at viewer, spring afternoon, soft sunlight, detailed background, "
            "masterpiece, best quality, anime illustration"
        ),
        "negative": (
            "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, "
            "worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry"
        ),
        "steps": 28,
        "cfg": 5.0,
        "seed": 2202,
    },
    {
        "name": "gen_anchor_3_diorama",
        "model": "dreamshaper-xl.safetensors",
        "positive": (
            "isometric 3D render of a cozy fantasy bookshop inside a giant hollow tree, warm "
            "glowing windows, winding wooden staircase, tiny crowded bookshelves, floating "
            "lanterns, moss and mushrooms, highly detailed miniature diorama, soft studio "
            "lighting, octane render, whimsical, depth of field"
        ),
        "negative": (
            "photo, realistic, anime, flat illustration, low quality, blurry, watermark, text"
        ),
        "steps": 30,
        "cfg": 6.0,
        "seed": 3303,
    },
]


def workflow(a):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": a["model"]}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": a["positive"], "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": a["negative"], "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": a["seed"],
                "steps": a["steps"],
                "cfg": a["cfg"],
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": a["name"]}},
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    only = set(sys.argv[1:])
    for a in ANCHORS:
        if only and a["name"] not in only:
            continue
        print(f"[anchor] {a['name']} <- {a['model']}", flush=True)
        h = run(workflow(a))
        for fn, sub, _ in images(h):
            dest = os.path.join(OUT, a["name"] + ".png")
            download(fn, sub, dest)
            print("   saved", dest, flush=True)


if __name__ == "__main__":
    main()
