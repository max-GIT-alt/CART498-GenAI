"""Install the demo's UI workflows into the running ComfyUI Workflows sidebar.

ComfyUI stores user workflows under ``<user-directory>/default/workflows``. Copying
the UI JSON files there makes them appear in the sidebar after a page refresh.
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI = os.path.join(ROOT, "workflows", "ui")
DEST = os.path.join(ROOT, ".comfy-user", "default", "workflows")

WORKFLOWS = [
    ("00_matrix_all.json", "00 Matrix — 3 anchors x 3 models.json"),
    ("01_sdxl_img2img_process.json", "01 SDXL img2img + process GIF.json"),
    ("01a_sdxl_img2img_plain.json", "01a SDXL img2img (plain).json"),
    ("02_qwen_image_edit.json", "02 Qwen Image Edit (anchor demo).json"),
    ("03_denoise_ladder.json", "03 Denoise ladder — NoobAI.json"),
    ("04_micro_lcm.json", "04 Micro LCM — live demo (4 steps).json"),
]


def main():
    os.makedirs(DEST, exist_ok=True)
    for source_name, target_name in WORKFLOWS:
        source = os.path.join(UI, source_name)
        if not os.path.exists(source):
            print(f"skip missing {source}")
            continue
        shutil.copyfile(source, os.path.join(DEST, target_name))
        print(f"installed {target_name}")
    print(f"\nrefresh the ComfyUI page to see them in the Workflows sidebar.")


if __name__ == "__main__":
    main()
