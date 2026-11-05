"""Check that a ComfyUI Python environment is complete, and what device it sees.

Shared by ``fetch_comfyui.py`` (to verify an install) and ``bootstrap.py`` (to
refuse to start ComfyUI from a half-installed venv). Standard library only.

The probe runs *inside* ComfyUI's interpreter:

    <comfyui-python> scripts/comfy_env.py --probe <comfyui-dir>

and prints one JSON object: missing core requirements plus the torch build and
the accelerator torch can see.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()


def core_requirements(comfyui: Path) -> list[str]:
    """Distribution names from requirements.txt above the non-essential block."""
    names = []
    path = Path(comfyui) / "requirements.txt"
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") and "non essential" in line.lower():
            break
        if not line or line.startswith(("#", "-")):
            continue
        name = re.split(r"[\s<>=!~;\[@]", line, maxsplit=1)[0]
        if name:
            names.append(name)
    return names


def _probe(comfyui: Path) -> dict:
    from importlib import metadata

    result = {"python": sys.executable, "version": sys.version.split()[0]}
    try:
        required = core_requirements(comfyui)
    except OSError as error:
        result["error"] = f"cannot read {comfyui / 'requirements.txt'}: {error}"
        return result
    missing = []
    for name in required:
        try:
            metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    result["missing"] = missing

    torch_info = {"installed": False}
    try:
        import torch  # noqa: PLC0415

        torch_info.update(installed=True, version=torch.__version__,
                          cuda_build=torch.version.cuda,
                          hip_build=getattr(torch.version, "hip", None))
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            torch_info["device"] = "rocm" if torch_info["hip_build"] else "cuda"
            torch_info["device_name"] = props.name
            torch_info["vram_gb"] = round(props.total_memory / 1024 ** 3, 1)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch_info["device"] = "mps"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch_info["device"] = "xpu"
        else:
            torch_info["device"] = "cpu"
    except Exception as error:  # broken wheels raise more than ImportError
        torch_info["error"] = f"{type(error).__name__}: {error}"
    result["torch"] = torch_info
    return result


def torch_build(info: dict) -> str | None:
    """Classify an installed torch build as cuda / rocm / xpu / cpu."""
    torch = info.get("torch", {})
    if not torch.get("installed"):
        return None
    if torch.get("hip_build"):
        return "rocm"
    if torch.get("cuda_build"):
        return "cuda"
    if "+xpu" in torch.get("version", ""):
        return "xpu"
    return "cpu"


def run_probe(python: str, comfyui: Path, timeout: int = 180) -> dict:
    """Run the probe under *python*; never raises for an unusable interpreter."""
    try:
        done = subprocess.run(
            [python, str(HERE), "--probe", str(comfyui)],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return {"error": f"interpreter not found: {python}"}
    except subprocess.TimeoutExpired:
        return {"error": f"{python} did not finish the environment check in {timeout}s"}
    lines = done.stdout.strip().splitlines()
    if done.returncode != 0 or not lines:
        tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        return {"error": f"{python} could not run the environment check: " + " | ".join(tail)}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"error": f"unexpected output from {python}: {lines[-1][:200]}"}


def describe(info: dict) -> str:
    torch = info.get("torch", {})
    if not torch.get("installed"):
        return "torch not installed"
    device = torch.get("device", "?")
    text = f"torch {torch.get('version')} · device: {device}"
    if torch.get("device_name"):
        text += f" ({torch['device_name']}, {torch.get('vram_gb')} GB)"
    return text


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--probe":
        print(json.dumps(_probe(Path(sys.argv[2]))))
    else:
        sys.exit("usage: comfy_env.py --probe <comfyui-dir>")
