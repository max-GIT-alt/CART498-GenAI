"""Build the combined ComfyUI workflows for the anchor x model demo.

Three graphs are produced:

  00_matrix_all       3 anchors x 3 models (9 outputs) in one canvas
  03_denoise_ladder   NoobAI on anchor 01 at denoise 0.35 / 0.55 / 0.75 / 0.95
  04_micro_lcm        fast 4-step text-to-image demo with the 2 GB micro model

Each graph is written in both API format (what /prompt accepts) and UI format
(what the ComfyUI editor opens). The UI layout is laid out explicitly: model
columns, anchor rows, one group per model column, one group per anchor.
"""
import json
import os

from PIL import Image

from to_ui import convert

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images")
API = os.path.join(ROOT, "workflows", "api")
UI = os.path.join(ROOT, "workflows", "ui")

TARGET = "cinematic fantasy illustration, dramatic lighting, rich colours, highly detailed"
NEGATIVE = "lowres, bad anatomy, bad hands, blurry, watermark, text, jpeg artifacts"
QWEN_INSTRUCTION = (
    "Transform this image into a cinematic fantasy illustration with dramatic lighting "
    "and rich colours. Keep the composition and subject recognisable."
)

ANCHORS = [
    (1, "anchor_1_surreal", "01 · Surreal digital art"),
    (2, "anchor_2_nature", "02 · Nature photograph"),
    (3, "anchor_3_collage", "03 · Mixed-media collage"),
]
SDXL = [
    ("dreamshaper", "DreamShaper XL", "dreamshaper-xl.safetensors"),
    ("noobai", "NoobAI XL", "NoobAI-XL-v1.0.safetensors"),
]
QWEN = ("qwen_edit", "Qwen-Image-Edit 2511")
MICRO_CKPT = "DreamShaper8_LCM.safetensors"
MICRO_PROMPT = (
    "portrait photo of an elderly fisherman, weathered face, golden hour, "
    "shallow depth of field"
)
MICRO_NEGATIVE = "lowres, blurry, bad anatomy, bad hands, watermark, text, jpeg artifacts"

COL_X = {"dreamshaper": 2650, "noobai": 4550, "qwen_edit": 6450}
ANCHOR_X = 1650
ROW_Y = {1: 1000, 2: 2200, 3: 3400}
COLOR = {"dreamshaper": "#3f789e", "noobai": "#8a5a9e", "qwen_edit": "#b06634"}
SIZE = {
    "anchor_load": (340, 320),
    "sdxl_scale": (340, 150),
    "qwen_scale": (300, 80),
    "sdxl_ckpt": (340, 100),
    "sdxl_pos": (420, 170),
    "sdxl_neg": (420, 170),
    "qwen_unet": (340, 100),
    "qwen_clip": (340, 130),
    "qwen_vae": (340, 80),
    "qwen_lora": (340, 100),
    "qwen_ms": (340, 100),
    "qwen_cfg": (340, 100),
    "sdxl_enc": (300, 80),
    "sdxl_ks": (340, 330),
    "sdxl_dec": (260, 60),
    "sdxl_save": (420, 450),
    "qwen_enc": (300, 80),
    "qwen_ks": (340, 330),
    "qwen_dec": (260, 60),
    "qwen_save": (420, 450),
    "qwen_pos": (420, 220),
    "qwen_neg": (420, 220),
    "qwen_ref_pos": (420, 100),
    "qwen_ref_neg": (420, 100),
    "micro_ckpt": (340, 100),
    "micro_pos": (420, 190),
    "micro_neg": (420, 190),
    "micro_latent": (340, 160),
    "micro_ks": (340, 330),
    "micro_dec": (260, 60),
    "micro_save": (420, 500),
}


class Builder:
    def __init__(self):
        self.api = {}
        self.meta = {}

    def add(self, cls, inputs, role, model=None, anchor=None, title=None):
        nid = str(len(self.api) + 1)
        self.api[nid] = {"class_type": cls, "inputs": inputs}
        self.meta[nid] = {"role": role, "model": model, "anchor": anchor, "title": title}
        return nid


def sdxl_size(name):
    return Image.open(os.path.join(IMG, name + ".png")).size


def build_matrix():
    b = Builder()

    # anchors
    anchor_ids = {}
    for i, name, label in ANCHORS:
        anchor_ids[i] = b.add(
            "LoadImage", {"image": name + ".png"}, "anchor_load", anchor=i, title=label
        )

    # SDXL loaders + shared prompts
    sdxl = {}
    for short, label, ckpt in SDXL:
        ck = b.add("CheckpointLoaderSimple", {"ckpt_name": ckpt}, "sdxl_ckpt", model=short,
                   title=label)
        pos = b.add("CLIPTextEncode", {"text": TARGET, "clip": [ck, 1]}, "sdxl_pos",
                    model=short, title="positive prompt")
        neg = b.add("CLIPTextEncode", {"text": NEGATIVE, "clip": [ck, 1]}, "sdxl_neg",
                    model=short, title="negative prompt")
        sdxl[short] = (ck, pos, neg)

    # Qwen loader chain
    q_unet = b.add("UnetLoaderGGUF", {"unet_name": "qwen-image-edit-2511-Q4_K_M.gguf"},
                   "qwen_unet", model="qwen_edit", title="Qwen-Image-Edit 2511 (GGUF Q4)")
    q_clip = b.add("CLIPLoader", {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                                  "type": "qwen_image", "device": "default"},
                   "qwen_clip", model="qwen_edit")
    q_vae = b.add("VAELoader", {"vae_name": "qwen_image_vae.safetensors"},
                  "qwen_vae", model="qwen_edit")
    q_lora = b.add("LoraLoaderModelOnly", {
        "model": [q_unet, 0],
        "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "strength_model": 1.0,
    }, "qwen_lora", model="qwen_edit", title="Lightning 4-step LoRA")
    q_ms = b.add("ModelSamplingAuraFlow", {"model": [q_lora, 0], "shift": 3.1},
                 "qwen_ms", model="qwen_edit")
    q_cfg = b.add("CFGNorm", {"model": [q_ms, 0], "strength": 1.0, "pre_cfg": False},
                  "qwen_cfg", model="qwen_edit")

    for i, name, label in ANCHORS:
        w, h = sdxl_size(name)

        # shared per-anchor preprocessing
        sdxl_scale = b.add("ImageScale", {
            "image": [anchor_ids[i], 0], "upscale_method": "lanczos",
            "width": w, "height": h, "crop": "disabled",
        }, "sdxl_scale", anchor=i, title=f"resize to {w}×{h}")
        qwen_scale = b.add("FluxKontextImageScale", {"image": [anchor_ids[i], 0]},
                           "qwen_scale", anchor=i)
        qwen_enc = b.add("VAEEncode", {"pixels": [qwen_scale, 0], "vae": [q_vae, 0]},
                         "qwen_enc", model="qwen_edit", anchor=i)

        # Qwen reference conditioning is per anchor
        q_pos = b.add("TextEncodeQwenImageEditPlus", {
            "clip": [q_clip, 0], "prompt": QWEN_INSTRUCTION,
            "vae": [q_vae, 0], "image1": [qwen_scale, 0],
        }, "qwen_pos", model="qwen_edit", anchor=i, title="instruction")
        q_neg = b.add("TextEncodeQwenImageEditPlus", {
            "clip": [q_clip, 0], "prompt": "",
            "vae": [q_vae, 0], "image1": [qwen_scale, 0],
        }, "qwen_neg", model="qwen_edit", anchor=i, title="negative (unused at cfg 1)")
        q_ref_pos = b.add("FluxKontextMultiReferenceLatentMethod", {
            "conditioning": [q_pos, 0], "reference_latents_method": "index_timestep_zero",
        }, "qwen_ref_pos", model="qwen_edit", anchor=i)
        q_ref_neg = b.add("FluxKontextMultiReferenceLatentMethod", {
            "conditioning": [q_neg, 0], "reference_latents_method": "index_timestep_zero",
        }, "qwen_ref_neg", model="qwen_edit", anchor=i)

        for short, label, ckpt in SDXL:
            ck, pos, neg = sdxl[short]
            enc = b.add("VAEEncode", {"pixels": [sdxl_scale, 0], "vae": [ck, 2]},
                        "sdxl_enc", model=short, anchor=i)
            ks = b.add("KSampler", {
                "model": [ck, 0], "positive": [pos, 0], "negative": [neg, 0],
                "latent_image": [enc, 0], "seed": 4242, "steps": 30, "cfg": 6.0,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.75,
            }, "sdxl_ks", model=short, anchor=i)
            dec = b.add("VAEDecode", {"samples": [ks, 0], "vae": [ck, 2]},
                        "sdxl_dec", model=short, anchor=i)
            b.add("SaveImage", {
                "images": [dec, 0],
                "filename_prefix": f"matrix/{short}__{name}",
            }, "sdxl_save", model=short, anchor=i,
                title=f"{label} · anchor {i:02d}")

        q_ks = b.add("KSampler", {
            "model": [q_cfg, 0], "positive": [q_ref_pos, 0], "negative": [q_ref_neg, 0],
            "latent_image": [qwen_enc, 0], "seed": 4242, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
        }, "qwen_ks", model="qwen_edit", anchor=i)
        q_dec = b.add("VAEDecode", {"samples": [q_ks, 0], "vae": [q_vae, 0]},
                      "qwen_dec", model="qwen_edit", anchor=i)
        b.add("SaveImage", {
            "images": [q_dec, 0],
            "filename_prefix": f"matrix/qwen_edit__{name}",
        }, "qwen_save", model="qwen_edit", anchor=i,
            title=f"{QWEN[1]} · anchor {i:02d}")

    return b


def build_micro():
    b = Builder()
    ck = b.add("CheckpointLoaderSimple", {"ckpt_name": MICRO_CKPT},
               "micro_ckpt", model="micro", title="DreamShaper 8 LCM · 2.0 GiB")
    pos = b.add("CLIPTextEncode", {"text": MICRO_PROMPT, "clip": [ck, 1]},
                "micro_pos", model="micro", title="positive prompt")
    neg = b.add("CLIPTextEncode", {"text": MICRO_NEGATIVE, "clip": [ck, 1]},
                "micro_neg", model="micro", title="negative prompt")
    latent = b.add("EmptyLatentImage", {"width": 512, "height": 512, "batch_size": 1},
                   "micro_latent", model="micro", title="512 x 512")
    ks = b.add("KSampler", {
        "model": [ck, 0], "positive": [pos, 0], "negative": [neg, 0],
        "latent_image": [latent, 0], "seed": 4242, "steps": 4, "cfg": 2.0,
        "sampler_name": "lcm", "scheduler": "normal", "denoise": 1.0,
    }, "micro_ks", model="micro", title="4 steps · lcm · cfg 2.0")
    dec = b.add("VAEDecode", {"samples": [ks, 0], "vae": [ck, 2]},
                "micro_dec", model="micro")
    b.add("SaveImage", {
        "images": [dec, 0], "filename_prefix": "micro/micro_lcm",
    }, "micro_save", model="micro", title="fast live output")
    return b


def build_ladder():
    b = Builder()
    name = "anchor_1_surreal"
    w, h = sdxl_size(name)
    anchor = b.add("LoadImage", {"image": name + ".png"}, "anchor_load", anchor=1,
                   title="ANCHOR 01 · Surreal digital art")
    ck = b.add("CheckpointLoaderSimple", {"ckpt_name": "NoobAI-XL-v1.0.safetensors"},
               "sdxl_ckpt", model="noobai", title="NoobAI XL")
    pos = b.add("CLIPTextEncode", {"text": TARGET, "clip": [ck, 1]}, "sdxl_pos",
                model="noobai", title="positive prompt")
    neg = b.add("CLIPTextEncode", {"text": NEGATIVE, "clip": [ck, 1]}, "sdxl_neg",
                model="noobai", title="negative prompt")
    scale = b.add("ImageScale", {
        "image": [anchor, 0], "upscale_method": "lanczos",
        "width": w, "height": h, "crop": "disabled",
    }, "sdxl_scale", anchor=1, title=f"resize to {w}×{h}")

    for d in (0.35, 0.55, 0.75, 0.95):
        tag = str(d).replace(".", "")
        enc = b.add("VAEEncode", {"pixels": [scale, 0], "vae": [ck, 2]},
                    "sdxl_enc", model="noobai", anchor=1)
        ks = b.add("KSampler", {
            "model": [ck, 0], "positive": [pos, 0], "negative": [neg, 0],
            "latent_image": [enc, 0], "seed": 4242, "steps": 30, "cfg": 6.0,
            "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": d,
        }, "sdxl_ks", model="noobai", anchor=1)
        dec = b.add("VAEDecode", {"samples": [ks, 0], "vae": [ck, 2]},
                    "sdxl_dec", model="noobai", anchor=1)
        b.add("SaveImage", {
            "images": [dec, 0],
            "filename_prefix": f"ladder/noobai__{name}__denoise_{tag}",
        }, "sdxl_save", model="noobai", anchor=1, title=f"denoise {d}")
    return b


# --------------------------------------------------------------------------- UI

def layout_matrix(ui, meta):
    note_id = max(int(n["id"]) for n in ui["nodes"]) + 1
    for n in ui["nodes"]:
        m = meta[str(n["id"])]
        role = m["role"]
        w, h = SIZE[role]
        n["size"] = [w, h]
        if role == "anchor_load":
            n["pos"] = [ANCHOR_X, ROW_Y[m["anchor"]] + 40]
        elif role == "sdxl_scale":
            n["pos"] = [ANCHOR_X + 420, ROW_Y[m["anchor"]] + 40]
        elif role == "qwen_scale":
            n["pos"] = [ANCHOR_X + 420, ROW_Y[m["anchor"]] + 300]
        elif role in ("sdxl_ckpt", "sdxl_pos", "sdxl_neg"):
            x = COL_X[m["model"]]
            y = 50
            if role == "sdxl_pos":
                x, y = x + 420, 50
            elif role == "sdxl_neg":
                x, y = x + 420, 280
            n["pos"] = [x, y]
        elif role.startswith("qwen_") and role in (
            "qwen_unet", "qwen_clip", "qwen_vae", "qwen_lora", "qwen_ms", "qwen_cfg"
        ):
            x = COL_X["qwen_edit"]
            y = {"qwen_unet": 50, "qwen_clip": 250, "qwen_vae": 450,
                 "qwen_lora": 50, "qwen_ms": 300, "qwen_cfg": 500}[role]
            if role in ("qwen_lora", "qwen_ms", "qwen_cfg"):
                x += 420
            n["pos"] = [x, y]
        else:
            x = COL_X[m["model"]]
            y = ROW_Y[m["anchor"]]
            if role in ("sdxl_enc", "qwen_enc"):
                n["pos"] = [x, y]
            elif role in ("sdxl_ks", "qwen_ks"):
                n["pos"] = [x + 360, y]
            elif role in ("sdxl_dec", "qwen_dec"):
                n["pos"] = [x + 740, y]
            elif role in ("sdxl_save", "qwen_save"):
                n["pos"] = [x + 1050, y]
            elif role == "qwen_pos":
                n["pos"] = [x, y + 420]
            elif role == "qwen_neg":
                n["pos"] = [x, y + 700]
            elif role == "qwen_ref_pos":
                n["pos"] = [x + 420, y + 420]
            elif role == "qwen_ref_neg":
                n["pos"] = [x + 420, y + 700]
        if m.get("title"):
            n["title"] = m["title"]

    groups = []
    gid = 1
    for i, name, label in ANCHORS:
        groups.append({
            "id": gid, "title": f"ANCHOR {label}",
            "bounding": [ANCHOR_X - 50, ROW_Y[i] - 50, 900, 520],
            "color": "#6b5a20", "font_size": 22, "flags": {},
        })
        gid += 1
    for short, label, _ in SDXL:
        groups.append({
            "id": gid,
            "title": f"{label} — SDXL img2img · denoise 0.75 · 30 steps",
            "bounding": [COL_X[short] - 50, -50, 1650, 4600],
            "color": COLOR[short], "font_size": 24, "flags": {},
        })
        gid += 1
    groups.append({
        "id": gid,
        "title": "Qwen-Image-Edit 2511 — instruction edit · 4 steps · Lightning LoRA",
        "bounding": [COL_X["qwen_edit"] - 50, -50, 1650, 4600],
        "color": COLOR["qwen_edit"], "font_size": 24, "flags": {},
    })

    note = {
        "id": note_id, "type": "MarkdownNote",
        "pos": [0, 0], "size": [1550, 700], "flags": {}, "order": len(ui["nodes"]), "mode": 0,
        "inputs": [], "outputs": [], "title": "How to read this workflow",
        "properties": {"widget_ue_connectable": {}},
        "widgets_values": [(
            "# Anchor × Model — full 3 × 3 comparison\n\n"
            "Queue this graph to render **all nine combinations** at once. Outputs are "
            "saved to `outputs/matrix/`.\n\n"
            "**Columns** are diffusion models:\n"
            "- **DreamShaper XL** — SDXL img2img, denoise 0.75\n"
            "- **NoobAI XL** — SDXL img2img, denoise 0.75\n"
            "- **Qwen-Image-Edit 2511** — instruction edit, 4 steps, Lightning LoRA\n\n"
            "**Rows** are the three anchors. The SDXL models share one prompt per model; "
            "Qwen receives the same intent as an instruction and keeps the anchor "
            "structure.\n\n"
            "On an 8 GB GPU, mute a model column (select its group, press `Ctrl+M`) to run "
            "one model at a time, or use the single-model workflows 01 and 02."
        )],
    }
    ui["nodes"].append(note)
    ui["last_node_id"] = note_id
    ui["groups"] = groups
    ui["extra"] = {"ds": {"scale": 0.22, "offset": [40, 40]}}
    return ui


def layout_ladder(ui, meta):
    note_id = max(int(n["id"]) for n in ui["nodes"]) + 1
    for n in ui["nodes"]:
        m = meta[str(n["id"])]
        role = m["role"]
        w, h = SIZE[role]
        n["size"] = [w, h]
        if role == "anchor_load":
            n["pos"] = [0, 500]
        elif role == "sdxl_ckpt":
            n["pos"] = [420, 500]
        elif role == "sdxl_pos":
            n["pos"] = [840, 500]
        elif role == "sdxl_neg":
            n["pos"] = [840, 740]
        elif role == "sdxl_scale":
            n["pos"] = [0, 880]
        elif role in ("sdxl_enc", "sdxl_ks", "sdxl_dec", "sdxl_save"):
            # one column per denoise value, in node creation order
            pass
        if m.get("title"):
            n["title"] = m["title"]

    # columns for the four denoise branches, in creation order
    denoises = [0.35, 0.55, 0.75, 0.95]
    branches = {}
    for n in ui["nodes"]:
        m = meta[str(n["id"])]
        if m["role"] == "sdxl_save":
            branches.setdefault(m["title"], []).append(n["id"])

    for idx, d in enumerate(denoises):
        x = 1400 + idx * 1500
        y = 1200
        title = f"denoise {d}"
        ids = branches.get(title, [])
        # the branch nodes were created consecutively: enc, ks, dec, save
        # find them by following links instead of relying on creation order
        save_id = ids[0]
        save = next(n for n in ui["nodes"] if n["id"] == save_id)
        dec_id = save["inputs"][0]["link"]
        dec_link = next(l for l in ui["links"] if l[0] == dec_id)
        dec = next(n for n in ui["nodes"] if n["id"] == dec_link[1])
        ks_id = dec["inputs"][0]["link"]
        ks_link = next(l for l in ui["links"] if l[0] == ks_id)
        ks = next(n for n in ui["nodes"] if n["id"] == ks_link[1])
        enc_id = ks["inputs"][3]["link"]
        enc_link = next(l for l in ui["links"] if l[0] == enc_id)
        enc = next(n for n in ui["nodes"] if n["id"] == enc_link[1])
        enc["pos"] = [x, y]
        ks["pos"] = [x + 340, y]
        dec["pos"] = [x + 720, y]
        save["pos"] = [x + 1030, y]
        save["title"] = f"denoise {d}"

    groups = [{
        "id": 1, "title": "NoobAI XL — anchor 01 at four denoise levels",
        "bounding": [-50, 450, 7600, 1400], "color": "#8a5a9e",
        "font_size": 24, "flags": {},
    }]
    note = {
        "id": note_id, "type": "MarkdownNote",
        "pos": [0, 0], "size": [1400, 430], "flags": {}, "order": len(ui["nodes"]), "mode": 0,
        "inputs": [], "outputs": [], "title": "How to read this workflow",
        "properties": {"widget_ue_connectable": {}},
        "widgets_values": [(
            "# How strong is the anchor?\n\n"
            "The same anchor, prompt and seed through **NoobAI XL** at four denoise "
            "levels. Low denoise copies the anchor; high denoise lets the model invent.\n\n"
            "- `0.35` — nearly a copy\n"
            "- `0.55` — stylised, same scene\n"
            "- `0.75` — the model's prior takes over\n"
            "- `0.95` — the anchor is only a layout hint\n\n"
            "Outputs are saved to `outputs/ladder/`."
        )],
    }
    ui["nodes"].append(note)
    ui["last_node_id"] = note_id
    ui["groups"] = groups
    ui["extra"] = {"ds": {"scale": 0.25, "offset": [40, 40]}}
    return ui


def layout_micro(ui, meta):
    note_id = max(int(n["id"]) for n in ui["nodes"]) + 1
    positions = {
        "micro_ckpt": [40, 40],
        "micro_pos": [460, 40],
        "micro_neg": [460, 300],
        "micro_latent": [460, 560],
        "micro_ks": [920, 40],
        "micro_dec": [1300, 40],
        "micro_save": [1600, 40],
    }
    for n in ui["nodes"]:
        m = meta[str(n["id"])]
        n["size"] = list(SIZE[m["role"]])
        n["pos"] = list(positions[m["role"]])
        if m.get("title"):
            n["title"] = m["title"]

    note = {
        "id": note_id, "type": "MarkdownNote",
        "pos": [40, 820], "size": [2050, 330], "flags": {}, "order": len(ui["nodes"]),
        "mode": 0, "inputs": [], "outputs": [], "title": "Live demo: the 2 GB micro model",
        "properties": {"widget_ue_connectable": {}},
        "widgets_values": [(
            "# Micro LCM — fast text-to-image for live demos\n\n"
            "**DreamShaper 8 LCM** is a 2.0 GiB fp16 Stable Diffusion 1.5 checkpoint "
            "with LCM distillation baked in. It loads from the same `models/checkpoints` "
            "folder as the big models, but needs only **4 steps at 512 × 512**.\n\n"
            "- sampler `lcm`, scheduler `normal`, cfg `2.0`, 4 steps\n"
            "- one image takes roughly a second on the RTX 3070 Ti (first run includes "
            "the model load)\n"
            "- outputs are saved to `outputs/micro/`\n\n"
            "Use `scripts/run_micro.py` for a prompt grid with per-prompt timings, or "
            "queue this graph for a one-image live demo. Raise the batch size on "
            "`EmptyLatentImage` to show several variations per prompt."
        )],
    }
    ui["nodes"].append(note)
    ui["last_node_id"] = note_id
    ui["groups"] = [{
        "id": 1, "title": "MICRO LCM — DreamShaper 8 LCM · 4 steps · 512 px",
        "bounding": [0, 0, 2100, 780], "color": "#2f7d5a",
        "font_size": 24, "flags": {},
    }]
    ui["extra"] = {"ds": {"scale": 0.5, "offset": [40, 40]}}
    return ui


def save(builder, name, layout):
    os.makedirs(API, exist_ok=True)
    os.makedirs(UI, exist_ok=True)
    api_path = os.path.join(API, name + ".json")
    with open(api_path, "w") as f:
        json.dump(builder.api, f, indent=2)
    ui = convert(builder.api)
    ui = layout(ui, builder.meta)
    ui_path = os.path.join(UI, name + ".json")
    with open(ui_path, "w") as f:
        json.dump(ui, f, indent=2)
    print("wrote", api_path, "and", ui_path, f"({len(ui['nodes'])} nodes)")


def main():
    save(build_matrix(), "00_matrix_all", layout_matrix)
    save(build_ladder(), "03_denoise_ladder", layout_ladder)
    save(build_micro(), "04_micro_lcm", layout_micro)


if __name__ == "__main__":
    main()
