"""Custom nodes that capture the denoising process as a GIF + session history."""
import json
import os
import re
import uuid
from datetime import datetime

import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview

from . import session as session_lib


def _previewer(model):
    """Best available cheap latent previewer (TAESD if installed, else latent2rgb)."""
    try:
        previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)
        if previewer is not None:
            return previewer
    except Exception:
        pass
    try:
        latent_format = model.model.latent_format
        factors = getattr(latent_format, "latent_rgb_factors", None)
        if factors is not None:
            return latent_preview.Latent2RGBPreviewer(
                factors,
                getattr(latent_format, "latent_rgb_factors_bias", None),
                getattr(latent_format, "latent_rgb_factors_reshape", None),
            )
    except Exception:
        pass
    return None


def _tensor_to_pil(image):
    array = (image.clamp(0, 1).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _downscale(image, max_side):
    if max(image.size) <= max_side:
        return image.copy()
    out = image.copy()
    out.thumbnail((max_side, max_side), Image.LANCZOS)
    return out


class KSamplerFrames:
    """KSampler that captures the denoising trajectory, saves a GIF and a session entry."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frame_source": (["x0 (denoised prediction)", "x (current latent)"], {"default": "x0 (denoised prediction)"}),
                "frame_stride": ("INT", {"default": 2, "min": 1, "max": 10}),
                "max_side": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "fps": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "session_name": ("STRING", {"default": "session"}),
                "label": ("STRING", {"default": "SDXL img2img"}),
                "anchor_label": ("STRING", {"default": "anchor_1_surreal"}),
                "include_anchor": ("BOOLEAN", {"default": True}),
                "preview": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("final_image", "frames", "run_id", "gif_path", "final_path")
    FUNCTION = "sample"
    CATEGORY = "image/sampling"
    OUTPUT_NODE = False

    def sample(self, model, vae, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise, frame_source, frame_stride,
               max_side, fps, session_name, label, anchor_label, include_anchor, preview):
        latent_samples = latent_image["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(
            model, latent_samples,
            latent_image.get("downscale_ratio_spacial"),
            latent_image.get("downscale_ratio_temporal"),
        )
        batch_inds = latent_image.get("batch_index")
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        noise_mask = latent_image.get("noise_mask")

        stored = []
        previewer = _previewer(model) if preview else None
        pbar = comfy.utils.ProgressBar(steps)
        latent_format = model.model.latent_format
        use_x0 = frame_source.startswith("x0")

        def callback(step, x0, x, total_steps):
            comfy.model_management.throw_exception_if_processing_interrupted()
            x0_flat = x0.tensors[0] if getattr(x0, "is_nested", False) else x0
            x_flat = x.tensors[0] if getattr(x, "is_nested", False) else x
            source = x0_flat if use_x0 else x_flat
            if step % frame_stride == 0:
                stored.append((source.detach().to("cpu", copy=True), use_x0))
            if previewer is not None:
                try:
                    preview_image = previewer.decode_latent_to_preview(x0_flat)
                    pbar.update_absolute(step + 1, total_steps, ("JPEG", preview_image, max_side))
                    return
                except Exception:
                    pass
            pbar.update_absolute(step + 1, total_steps)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_samples,
            denoise=denoise, noise_mask=noise_mask,
            callback=callback, disable_pbar=disable_pbar, seed=seed,
        )
        comfy.model_management.throw_exception_if_processing_interrupted()

        latents = []
        if include_anchor:
            if getattr(latent_samples, "is_nested", False):
                anchor_latent = latent_samples.tensors[0]
            else:
                anchor_latent = latent_samples
            latents.append(anchor_latent.detach().to("cpu", copy=True))
        for tensor, is_x0 in stored:
            if is_x0:
                try:
                    tensor = latent_format.process_out(tensor)
                except Exception:
                    pass
            latents.append(tensor)
        final_samples = samples.tensors[0] if getattr(samples, "is_nested", False) else samples
        latents.append(final_samples.detach().to("cpu", copy=True))
        if not latents:
            latents = [final_samples.detach().to("cpu", copy=True)]

        with torch.no_grad():
            decoded = vae.decode(torch.cat(latents, dim=0))
        decoded = decoded.detach().to("cpu", copy=True)
        final_tensor = decoded[-1:].clamp(0, 1)

        pil_frames = []
        frame_tensors = []
        for index in range(decoded.shape[0]):
            frame = _downscale(_tensor_to_pil(decoded[index]), max_side)
            pil_frames.append(frame)
            frame_tensors.append(torch.from_numpy(np.array(frame)).float() / 255.0)
        frames_tensor = torch.stack(frame_tensors, dim=0)

        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex[:4]
        safe_session = re.sub(r"[^A-Za-z0-9._-]+", "_", session_name).strip("._") or "session"
        output_dir = folder_paths.get_output_directory()
        run_dir = os.path.join(output_dir, safe_session, run_id)
        os.makedirs(run_dir, exist_ok=True)

        final_path = os.path.join(run_dir, "final.png")
        _tensor_to_pil(final_tensor[0]).save(final_path)

        gif_path = os.path.join(run_dir, "process.gif")
        duration = max(1, int(round(1000.0 / max(fps, 0.1))))
        if len(pil_frames) == 1:
            pil_frames.append(pil_frames[0].copy())
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
            disposal=2,
        )

        entry = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "label": label,
            "anchor": anchor_label,
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "denoise": float(denoise),
            "sampler": sampler_name,
            "scheduler": scheduler,
            "frame_count": len(pil_frames),
            "fps": float(fps),
            "gif": f"{run_id}/process.gif",
            "final": f"{run_id}/final.png",
        }
        with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
        try:
            session_lib.append_run(output_dir, safe_session, entry)
        except Exception as exc:
            print(f"[ComfyUI-Session] gallery update failed: {exc}")

        return (final_tensor, frames_tensor, run_id, gif_path, final_path)


class ShowAnimatedGIF:
    """Output node that points the UI at a saved GIF and marks it animated."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"gif_path": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ()
    FUNCTION = "show"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def show(self, gif_path):
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        path = os.path.abspath(gif_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"GIF not found: {path}")
        relative = os.path.relpath(path, output_dir)
        if relative.startswith(".."):
            relative = os.path.basename(path)
        subfolder = os.path.dirname(relative).replace("\\", "/")
        filename = os.path.basename(relative)

        images = [{
            "filename": filename,
            "subfolder": subfolder,
            "type": "output",
        }]
        animated = [True]

        # The frontend only shows an animated <img> for outputs that are
        # marked animated AND include a .webp or .png filename. An animated
        # GIF on its own is treated as a <video>, which cannot play it, and
        # an unmarked GIF is drawn onto a canvas, which freezes the first
        # frame. Include the sibling final.png as a second image so the
        # output stays on the <img> path; the node preview uses the first
        # image, i.e. the GIF.
        final_path = os.path.join(os.path.dirname(path), "final.png")
        if not os.path.exists(final_path):
            # Fallback for a GIF that did not come from KSamplerFrames: use
            # its first frame as the static .png companion.
            try:
                final_path = os.path.join(os.path.dirname(path), "process_preview.png")
                if not os.path.exists(final_path):
                    with Image.open(path) as gif:
                        gif.seek(0)
                        gif.convert("RGB").save(final_path)
            except Exception:
                final_path = None
        if final_path and os.path.exists(final_path):
            images.append({
                "filename": os.path.basename(final_path),
                "subfolder": subfolder,
                "type": "output",
            })
            animated.append(False)

        return {
            "ui": {
                "images": images,
                "animated": animated,
            },
            "result": (),
        }
