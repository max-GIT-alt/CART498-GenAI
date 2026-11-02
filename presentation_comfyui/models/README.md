# Model weights

Only the **micro diffusion model** is packaged for download. Weights are not
stored in git; `scripts/fetch_models.py` writes the micro checkpoint into
`models/checkpoints/`, where `extra_model_paths.yaml` exposes it to ComfyUI.

```bash
python scripts/fetch_models.py            # 1.99 GiB
python scripts/fetch_models.py --verify   # also check SHA-256
python scripts/fetch_models.py --list
python scripts/fetch_models.py --force    # replace an existing file
```

| id | file | source | size |
|----|------|--------|------|
| `micro` | `checkpoints/DreamShaper8_LCM.safetensors` | [Lykon/dreamshaper-8-lcm](https://huggingface.co/Lykon/dreamshaper-8-lcm) | 1.99 GiB |

The micro model is CreativeML OpenRAIL-M. Review the source model card before
redistributing generated images or weights.

## Optional: use your local SDXL / Qwen models

The matrix, ladder, session and Qwen workflows in this repository still work
with the diffusion models from the original local flow. They are **not**
downloaded. If you already have them, symlink the files into these paths:

| local file | used by |
|------------|---------|
| `checkpoints/dreamshaper-xl.safetensors` | matrix, workflow 01 |
| `checkpoints/NoobAI-XL-v1.0.safetensors` | matrix, denoise ladder |
| `unet/qwen-image-edit-2511-Q4_K_M.gguf` | Qwen-Image-Edit graph |
| `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | Qwen-Image-Edit graph |
| `vae/qwen_image_vae.safetensors` | Qwen-Image-Edit graph |
| `loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | Qwen 4-step Lightning |

```bash
ln -s /path/to/dreamshaper-xl.safetensors models/checkpoints/
ln -s /path/to/NoobAI-XL-v1.0.safetensors models/checkpoints/
ln -s /path/to/qwen-image-edit-2511-Q4_K_M.gguf models/unet/
ln -s /path/to/qwen_2.5_vl_7b_fp8_scaled.safetensors models/text_encoders/
ln -s /path/to/qwen_image_vae.safetensors models/vae/
ln -s /path/to/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors models/loras/
```

`run_all.sh` checks for these files and skips the matrix/ladder automatically
when they are absent; the micro demo always runs. Every file is gitignored.

On Windows, symlinks may require Developer Mode. Copy the files into these
folders instead, or use `mklink /H` for a hard link.
