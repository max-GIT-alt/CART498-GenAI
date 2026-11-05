#!/usr/bin/env python3
"""Fetch and set up ComfyUI on Windows, Linux, or macOS.

Uses only the Python standard library. Git is used when available; otherwise
the script downloads and extracts the corresponding GitHub ZIP archive. It then
creates ComfyUI's venv, installs the PyTorch build that matches this machine's
GPU (or the CPU build when there is none), installs ComfyUI's requirements.txt,
and verifies the result. Rerunning it repairs an interrupted install.
Model weights are fetched separately by this project's fetch_models.py.

    python3 scripts/fetch_comfyui.py                 # ~/ComfyUI, auto-detect GPU
    python3 scripts/fetch_comfyui.py --dry-run       # show the plan only
    python3 scripts/fetch_comfyui.py --torch cpu     # force the small CPU build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comfy_env  # noqa: E402

REPOSITORY = "https://github.com/comfyanonymous/ComfyUI.git"
ARCHIVE = "https://codeload.github.com/comfyanonymous/ComfyUI/zip/{ref}"
DEFAULT_DIRECTORY = Path.home() / "ComfyUI"
STAMP = ".comfyui-demo-ready.json"

# PyTorch indexes, as recommended in ComfyUI's README. On Linux, PyPI's default
# torch is the CUDA build (~3.5 GB of nvidia-* wheels) even on machines without
# an NVIDIA GPU; on Windows it is the CPU build even on NVIDIA machines. So the
# index is always chosen explicitly unless --torch default is given.
TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu130",
    "rocm": "https://download.pytorch.org/whl/rocm7.2",
    "xpu": "https://download.pytorch.org/whl/xpu",
    "cpu": "https://download.pytorch.org/whl/cpu",
}
# Older NVIDIA drivers cannot run CUDA 13 wheels.
CUDA_BY_DRIVER = [(580, "cu130"), (570, "cu128"), (0, "cu126")]
# Rough peak disk use (GiB) while installing: the downloaded wheels plus the
# unpacked venv. The CUDA build alone is ~3.5 GB of wheels and ~6.5 GB installed.
VENV_GIB = {"cuda": 10.0, "rocm": 18.0, "xpu": 7.0, "cpu": 3.0, "default": 10.0}
TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]


class FetchError(RuntimeError):
    """An expected, user-actionable setup error."""


# --------------------------------------------------------------------- fetch

def clone_with_git(destination: Path, ref: str) -> None:
    """Create a shallow checkout in *destination*."""
    command = ["git", "clone", "--depth", "1", "--branch", ref, REPOSITORY,
               str(destination)]
    environment = os.environ.copy()
    environment.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        subprocess.run(command, check=True, env=environment)
    except subprocess.CalledProcessError as error:
        raise FetchError(
            f"Git could not fetch ComfyUI ref {ref!r}. Check the ref and your "
            "network connection, or retry with --archive."
        ) from error


def validate_archive_member(name: str) -> None:
    """Reject archive entries that could escape the extraction directory."""
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise FetchError(f"Unsafe path in downloaded archive: {name!r}")


def download_archive(destination: Path, ref: str) -> None:
    """Download and extract a branch, tag, or commit archive."""
    archive_url = ARCHIVE.format(ref=urllib.parse.quote(ref, safe=""))
    archive_path = destination / "comfyui.zip"
    extraction_directory = destination / "extracted"

    print(f"Downloading {archive_url}")
    request = urllib.request.Request(
        archive_url, headers={"User-Agent": "comfyui-anchor-demo-fetcher/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with archive_path.open("wb") as output:
                shutil.copyfileobj(response, output)
    except (urllib.error.URLError, TimeoutError) as error:
        reason = getattr(error, "reason", error)
        raise FetchError(
            f"Could not download ComfyUI ref {ref!r}: {reason}. Check the ref "
            "and your network connection."
        ) from error

    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                validate_archive_member(member.filename)
            archive.extractall(extraction_directory)
    except zipfile.BadZipFile as error:
        raise FetchError("The downloaded ComfyUI archive is not a valid ZIP file.") from error

    if (extraction_directory / "main.py").is_file():
        checkout = extraction_directory
    else:
        roots = [entry for entry in extraction_directory.iterdir() if entry.is_dir()]
        if len(roots) != 1 or not (roots[0] / "main.py").is_file():
            raise FetchError("The downloaded archive does not contain a ComfyUI checkout.")
        checkout = roots[0]
    checkout.replace(destination / "checkout")


def fetch_source(destination: Path, ref: str, use_archive: bool) -> None:
    """Fetch ComfyUI without overwriting an unrelated directory."""
    if destination.exists() and not destination.is_dir():
        raise FetchError(f"Destination exists but is not a directory: {destination}")
    if destination.is_dir() and any(destination.iterdir()):
        raise FetchError(
            f"Destination is non-empty and is not a ComfyUI checkout: {destination}\n"
            "Choose another directory or move the existing contents aside."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.fetch-",
                                    dir=destination.parent))
    checkout = staging / "checkout"
    try:
        if use_archive or shutil.which("git") is None:
            if not use_archive:
                print("Git was not found; using the ZIP archive instead.")
            download_archive(staging, ref)
        else:
            clone_with_git(checkout, ref)
        if not (checkout / "main.py").is_file():
            raise FetchError("Fetch completed, but ComfyUI's main.py was not found.")
        if destination.is_dir():
            destination.rmdir()  # empty, checked above
        checkout.replace(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"\nComfyUI {ref!r} fetched to: {destination}")


# ------------------------------------------------------------ torch backend

def _nvidia_driver_major() -> Optional[int]:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.match(r"\s*(\d+)", out.stdout or "")
    return int(match.group(1)) if out.returncode == 0 and match else None


def detect_backend() -> tuple[str, str]:
    """Return (backend, reason) for this machine."""
    if sys.platform == "darwin":
        return "default", "macOS: PyPI's torch includes Apple-silicon (MPS) support"
    driver = _nvidia_driver_major()
    if driver is not None:
        return "cuda", f"NVIDIA driver {driver} found (nvidia-smi)"
    if sys.platform.startswith("linux") and (
            Path("/dev/kfd").exists() or shutil.which("rocminfo")):
        return "rocm", "AMD ROCm device found (/dev/kfd or rocminfo)"
    return "cpu", "no NVIDIA or AMD ROCm GPU detected"


def torch_index_for(backend: str) -> Optional[str]:
    if backend == "default":
        return None
    if backend == "cuda":
        driver = _nvidia_driver_major()
        if driver is None:  # forced without nvidia-smi: use the current default
            return TORCH_INDEX["cuda"]
        tag = next(tag for minimum, tag in CUDA_BY_DRIVER if driver >= minimum)
        return f"https://download.pytorch.org/whl/{tag}"
    return TORCH_INDEX[backend]


# -------------------------------------------------------------- pip running

DIAGNOSES = [
    (re.compile(r"Disk quota exceeded|Errno 122", re.I),
     "Disk quota exceeded. ComfyUI's venv needs about {need:.0f} GiB at {where}.\n"
     "  (A quota is not visible to the free-space check.) Free space in that\n"
     "  quota, delete ~/.cache/pip if an earlier install filled it, or install elsewhere:\n"
     "    python3 scripts/fetch_comfyui.py /path/with/space/ComfyUI\n"
     "  (on a GPU-less machine, --torch cpu needs only ~2 GiB)."),
    (re.compile(r"No space left on device|Errno 28", re.I),
     "The disk is full. ComfyUI's venv needs about {need:.0f} GiB at {where}."),
    (re.compile(r"NameResolutionError|Temporary failure in name resolution|"
                r"Failed to establish a new connection|ConnectionResetError|"
                r"Read timed out|IncompleteRead", re.I),
     "The network dropped during the download. Rerun the same command: the "
     "wheels downloaded so far are cached beside ComfyUI and are not fetched again."),
]


def run_pip(python: Path, args: list[str], env: dict, need: float, where: Path) -> None:
    """Run pip, echoing its output, and turn known failures into clear advice."""
    command = [str(python), "-m", "pip", *args, "--retries", "10", "--timeout", "60"]
    print("  $ " + " ".join(["pip", *args]))
    tail: list[str] = []
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        tail = (tail + [line])[-200:]
    if process.wait() == 0:
        return
    text = "".join(tail)
    for pattern, advice in DIAGNOSES:
        if pattern.search(text):
            raise FetchError(advice.format(need=need, where=where))
    raise FetchError(f"pip failed (exit code {process.returncode}); see the output above.")


def pip_environment(scratch: Path, keep_cache: bool) -> dict:
    """pip settings: temp files and download cache beside the venv.

    The cache lives in <ComfyUI>/.pip-scratch/cache instead of ~/.cache/pip, so it
    does not count twice against a home quota, is kept after a failed run (a
    rerun resumes instead of re-downloading GBs) and is deleted after success.
    """
    env = os.environ.copy()
    (scratch / "tmp").mkdir(parents=True, exist_ok=True)
    if not keep_cache:
        env["PIP_CACHE_DIR"] = str(scratch / "cache")
    for name in ("TMPDIR", "TEMP", "TMP"):  # not a small /tmp
        env[name] = str(scratch / "tmp")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    return env


def check_space(path: Path, need: float, allow_low: bool) -> None:
    free = shutil.disk_usage(path).free / 1024 ** 3
    if free >= need:
        return
    message = (f"Only {free:.1f} GiB free at {path}; ComfyUI's venv needs about "
               f"{need:.0f} GiB.")
    if allow_low:
        print(f"warning: {message} Continuing (--allow-low-space).")
        return
    raise FetchError(message + " Free some space, choose another directory, use "
                     "--torch cpu on a GPU-less machine, or pass --allow-low-space.")


def requirements_digest(checkout: Path) -> str:
    return hashlib.sha256((checkout / "requirements.txt").read_bytes()).hexdigest()


def venv_gib(backend: str) -> float:
    if backend == "default" and sys.platform == "darwin":
        return 4.0
    return VENV_GIB.get(backend, 10.0)


def venv_python(checkout: Path) -> Path:
    if os.name == "nt":
        return checkout / "venv" / "Scripts" / "python.exe"
    return checkout / "venv" / "bin" / "python"


# ----------------------------------------------------------------- install

def install_dependencies(checkout: Path, backend: str, torch_index: Optional[str],
                         keep_cache: bool, allow_low_space: bool) -> Path:
    """Create or repair ComfyUI's venv, install torch + requirements, verify."""
    requirements = checkout / "requirements.txt"
    if not requirements.is_file():
        raise FetchError(f"ComfyUI's requirements.txt was not found at: {requirements}")

    venv = checkout / "venv"
    python = venv_python(checkout)
    stamp = venv / STAMP
    need = venv_gib(backend)

    if stamp.is_file():
        stamp.unlink()  # re-verified below; a failed rerun must not look ready

    created = False
    if not python.is_file():
        if venv.exists() and (not venv.is_dir() or any(venv.iterdir())):
            raise FetchError(
                f"The virtual environment at {venv} has no Python interpreter. Move it "
                "aside or remove it, then rerun this script.")
        check_space(checkout, need, allow_low_space)
        print(f"\nCreating virtual environment: {venv}")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        except subprocess.CalledProcessError as error:
            raise FetchError(
                "Could not create the virtual environment. On Debian/Ubuntu, install "
                "the python3-venv package.") from error
        created = True
    else:
        print(f"\nReusing virtual environment: {venv}")

    scratch = checkout / ".pip-scratch"
    env = pip_environment(scratch, keep_cache)
    if (scratch / "cache").is_dir():
        print(f"Resuming with downloads cached by an earlier run in {scratch}")
    try:
        if created:
            print("\nUpgrading pip")
            run_pip(python, ["install", "--upgrade", "pip"], env, need, checkout)

        current = comfy_env.run_probe(str(python), checkout)
        have = comfy_env.torch_build(current)
        want = None if torch_index is None and backend == "default" else backend
        if have and (want is None or have == want or backend == "custom"):
            if backend == "custom":
                print(f"\nPyTorch already installed ({comfy_env.describe(current)}); "
                      "keeping it. Remove it first to reinstall from --torch-index.")
            else:
                print(f"\nPyTorch already installed: {comfy_env.describe(current)}")
        else:
            if have:
                print(f"\nInstalled PyTorch is a {have} build; replacing it with {want}.")
                run_pip(python, ["uninstall", "-y", *TORCH_PACKAGES], env, need, checkout)
            check_space(checkout, need, allow_low_space)
            args = ["install", *TORCH_PACKAGES]
            if torch_index:
                args += ["--index-url", torch_index]
            print(f"\nInstalling PyTorch ({backend}"
                  + (f", {torch_index}" if torch_index else ", PyPI") + ")")
            run_pip(python, args, env, need, checkout)

        print(f"\nInstalling ComfyUI dependencies from: {requirements}")
        # torch is already satisfied, so pip leaves the chosen build in place.
        run_pip(python, ["install", "-r", str(requirements)], env, need, checkout)
    finally:
        shutil.rmtree(scratch / "tmp", ignore_errors=True)
    shutil.rmtree(scratch, ignore_errors=True)  # success: drop the download cache

    print("\nVerifying the environment ...")
    info = comfy_env.run_probe(str(python), checkout)
    if info.get("error"):
        raise FetchError(f"Verification failed: {info['error']}")
    if info.get("missing"):
        raise FetchError("Verification failed; still missing: " + ", ".join(info["missing"]))
    if info["torch"].get("error"):
        raise FetchError(f"PyTorch is installed but does not import: {info['torch']['error']}")
    stamp.write_text(json.dumps({
        "backend": backend, "torch": info["torch"].get("version"),
        "requirements_sha256": requirements_digest(checkout),
    }, indent=1))
    print(f"  {comfy_env.describe(info)}")
    if backend in ("cuda", "rocm", "xpu") and info["torch"].get("device") == "cpu":
        print(f"  warning: a {backend} build was installed but torch sees no GPU; "
              "check the GPU driver. ComfyUI will fall back to the CPU.")
    return python


def fetch(destination: Path, ref: str, use_archive: bool, skip_deps: bool,
          backend: str, torch_index: Optional[str], keep_cache: bool,
          allow_low_space: bool, dry_run: bool) -> None:
    destination = destination.expanduser().resolve()
    present = destination.is_dir() and (destination / "main.py").is_file()

    reason = "set with --torch"
    if torch_index:
        backend, reason = "custom", "set with --torch-index"
    elif backend == "auto":
        backend, reason = detect_backend()
    if not torch_index:
        torch_index = torch_index_for(backend)

    print(f"ComfyUI:  {destination} ({'present' if present else 'will be fetched'})")
    print(f"PyTorch:  {backend} build — {reason}")
    print(f"          index: {torch_index or 'PyPI (default)'}")
    if not skip_deps:
        print(f"Disk:     about {venv_gib(backend):.0f} GiB for the venv")
    if dry_run:
        print("\n--dry-run: nothing changed.")
        return

    if not present:
        fetch_source(destination, ref, use_archive)
    else:
        print(f"ComfyUI is already present at: {destination}")
    if destination != DEFAULT_DIRECTORY.resolve():
        print("Set COMFYUI_DIR to this path before running the demo launcher.")
    if skip_deps:
        print("Dependency installation skipped (--skip-deps).")
        return

    python = install_dependencies(destination, backend, torch_index, keep_cache,
                                  allow_low_space)
    print("\nComfyUI and its core dependencies are ready.")
    print(f"Virtual environment: {python.parent.parent}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch ComfyUI, create its venv, install the PyTorch build for "
                    "this machine and ComfyUI's dependencies, then verify them. "
                    "Defaults to ~/ComfyUI, which the demo launcher detects. Rerun "
                    "to repair an interrupted install.")
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_DIRECTORY,
                        help="checkout directory (default: %(default)s)")
    parser.add_argument("--ref", default="master",
                        help="branch or tag to fetch (default: %(default)s)")
    parser.add_argument("--archive", action="store_true",
                        help="download a ZIP archive instead of using Git")
    deps = parser.add_mutually_exclusive_group()
    deps.add_argument("--skip-deps", action="store_true",
                      help="fetch source code only; do not create a venv or install packages")
    deps.add_argument("--torch", default="auto",
                      choices=["auto", "cuda", "rocm", "xpu", "cpu", "default"],
                      help="PyTorch build: auto-detect (default), a specific backend, "
                           "or 'default' for whatever PyPI serves")
    deps.add_argument("--torch-index", metavar="URL",
                      help="install torch/torchvision/torchaudio from this pip index")
    parser.add_argument("--keep-pip-cache", action="store_true",
                        help="use pip's normal cache (~/.cache/pip) instead of a "
                             "temporary one beside ComfyUI")
    parser.add_argument("--allow-low-space", action="store_true",
                        help="continue even if free disk space looks too small")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would be installed, change nothing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        fetch(args.directory, args.ref, args.archive, args.skip_deps, args.torch,
              args.torch_index, args.keep_pip_cache, args.allow_low_space, args.dry_run)
    except (FetchError, OSError) as error:
        print(f"\nerror: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nSetup cancelled. The venv may be incomplete; rerun the same command "
              "to finish it.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
