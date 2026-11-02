from .nodes import KSamplerFrames, ShowAnimatedGIF

# Register /comfy-session routes so the gallery and its downloads are also
# available from the ComfyUI server, not only from the file:// copy on disk.
try:
    from . import routes  # noqa: F401
except Exception as exc:  # pragma: no cover - only when the server is unavailable
    print(f"[ComfyUI-Session] HTTP export routes unavailable: {exc}")

NODE_CLASS_MAPPINGS = {
    "KSamplerFrames": KSamplerFrames,
    "ShowAnimatedGIF": ShowAnimatedGIF,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerFrames": "KSampler (capture process)",
    "ShowAnimatedGIF": "Show animated GIF",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
