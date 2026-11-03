"""Build the process-capturing version of workflow 01 (SDXL img2img)."""
import json
import os
import shutil

from to_ui import convert

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = os.path.join(ROOT, "workflows", "api")
UI = os.path.join(ROOT, "workflows", "ui")

TARGET = "cinematic fantasy illustration, dramatic lighting, rich colours, highly detailed"
NEGATIVE = "lowres, bad anatomy, bad hands, blurry, watermark, text, jpeg artifacts"


def build_api():
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "dreamshaper-xl.safetensors"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": TARGET, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": "anchor_1_surreal.png"},
        },
        "5": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["4", 0],
                "upscale_method": "lanczos",
                "width": 1024,
                "height": 1024,
                "crop": "disabled",
            },
        },
        "6": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "KSamplerFrames",
            "inputs": {
                "model": ["1", 0],
                "vae": ["1", 2],
                "seed": 4242,
                "steps": 30,
                "cfg": 6.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["6", 0],
                "denoise": 0.75,
                "frame_source": "x0 (denoised prediction)",
                "frame_stride": 2,
                "max_side": 512,
                "fps": 8.0,
                "session_name": "session",
                "label": "DreamShaper XL · anchor 01",
                "anchor_label": "anchor_1_surreal",
                "include_anchor": True,
                "preview": True,
            },
        },
        "8": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["7", 0]},
        },
        "9": {
            "class_type": "ShowAnimatedGIF",
            "inputs": {"gif_path": ["7", 3]},
        },
    }


def layout(ui):
    positions = {
        1: [80, 80],
        2: [510, 80],
        3: [510, 290],
        4: [80, 290],
        5: [510, 520],
        6: [940, 520],
        7: [1300, 500],
        8: [1780, 500],
        9: [1780, 1000],
    }
    sizes = {
        1: [360, 100],
        2: [360, 160],
        3: [360, 160],
        4: [360, 320],
        5: [360, 150],
        6: [300, 80],
        7: [420, 560],
        8: [400, 450],
        9: [400, 400],
    }
    titles = {
        4: "anchor 01",
        7: "KSampler (capture process)",
        8: "final image",
        9: "denoising process (GIF)",
    }
    for node in ui["nodes"]:
        nid = node["id"]
        if nid in positions:
            node["pos"] = positions[nid]
        if nid in sizes:
            node["size"] = sizes[nid]
        if nid in titles:
            node["title"] = titles[nid]

    ui["groups"] = [{
        "id": 1,
        "title": "01 · SDXL img2img — live process + session GIF",
        "bounding": [-50, -50, 2300, 1600],
        "color": "#3f789e",
        "font_size": 24,
        "flags": {},
    }]

    note_id = max(int(n["id"]) for n in ui["nodes"]) + 1
    note = {
        "id": note_id,
        "type": "MarkdownNote",
        "pos": [80, 1000],
        "size": [1200, 500],
        "flags": {},
        "order": len(ui["nodes"]),
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "title": "How to read this workflow",
        "properties": {"widget_ue_connectable": {}},
        "widgets_values": [(
            "# 01 · SDXL img2img — process capture\n\n"
            "This is the single-model graph, extended so you can see how the image is made.\n\n"
            "- **KSampler (capture process)** runs the normal SDXL img2img sampling "
            "(DreamShaper XL, 30 steps, denoise 0.75) and sends decoded frames to the UI "
            "while it works.\n"
            "- **final image** is the full-resolution result, also saved to "
            "`outputs/session/<run_id>/final.png`.\n"
            "- **denoising process (GIF)** is the saved animation, "
            "`outputs/session/<run_id>/process.gif`; it plays automatically in the node "
            "and in Queue History once the prompt finishes.\n"
            "- Every run appends to `outputs/session/session.html` — open it to see all "
            "previous generations from this session with their GIFs and final images.\n\n"
            "The GIF uses every 2nd sampling step at 512 px, 8 fps. Change `frame_stride`, "
            "`max_side` or `fps` on the sampler node to taste. To use NoobAI instead, "
            "change the checkpoint in the loader; the capture works the same way."
        )],
    }
    ui["nodes"].append(note)
    ui["last_node_id"] = note_id
    ui["extra"] = {"ds": {"scale": 0.5, "offset": [40, 40]}}
    return ui


def main():
    os.makedirs(API, exist_ok=True)
    os.makedirs(UI, exist_ok=True)
    api = build_api()
    api_path = os.path.join(API, "01_sdxl_img2img_process.json")
    with open(api_path, "w") as f:
        json.dump(api, f, indent=2)
    ui = layout(convert(api))
    ui_path = os.path.join(UI, "01_sdxl_img2img_process.json")
    with open(ui_path, "w") as f:
        json.dump(ui, f, indent=2)
    # Keep the historical 01 filenames pointing at the same process graph.
    shutil.copyfile(api_path, os.path.join(API, "01_sdxl_img2img.json"))
    shutil.copyfile(ui_path, os.path.join(UI, "01_sdxl_img2img.json"))
    print("wrote", api_path, "and", ui_path, f"({len(ui['nodes'])} nodes)")


if __name__ == "__main__":
    main()
