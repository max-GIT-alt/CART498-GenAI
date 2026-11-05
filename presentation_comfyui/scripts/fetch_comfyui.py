#!/usr/bin/env python3
"""Fetch and set up ComfyUI on Windows, Linux, or macOS.

Uses only the Python standard library. Git is used when available; otherwise
the script downloads and extracts the corresponding GitHub ZIP archive. By
default, it creates ComfyUI's venv and installs its complete requirements.txt.
Model weights are fetched separately by this project's fetch_models.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Optional

REPOSITORY = "https://github.com/comfyanonymous/ComfyUI.git"
ARCHIVE = "https://codeload.github.com/comfyanonymous/ComfyUI/zip/{ref}"
DEFAULT_DIRECTORY = Path.home() / "ComfyUI"


class FetchError(RuntimeError):
    """An expected, user-actionable setup error."""


def run_checked(command: list[str], failure_message: str) -> None:
    """Run a command and turn a non-zero exit into a concise setup error."""
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise FetchError(f"{failure_message} (exit code {error.returncode}).") from error


def clone_with_git(destination: Path, ref: str) -> None:
    """Create a shallow checkout in *destination*."""
    command = [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        ref,
        REPOSITORY,
        str(destination),
    ]
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
        archive_url,
        headers={"User-Agent": "comfyui-anchor-demo-fetcher/1.0"},
    )
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

    # Move the checkout beside the temporary files so the caller can move it
    # to its final destination using the same code path as a Git clone.
    checkout.replace(destination / "checkout")


def virtual_environment_python(checkout: Path) -> Path:
    """Return the venv interpreter path for the current operating system."""
    if os.name == "nt":
        return checkout / "venv" / "Scripts" / "python.exe"
    return checkout / "venv" / "bin" / "python"


def install_dependencies(checkout: Path, torch_index: Optional[str]) -> Path:
    """Create or reuse ComfyUI's venv and install all core requirements."""
    requirements = checkout / "requirements.txt"
    if not requirements.is_file():
        raise FetchError(f"ComfyUI's requirements.txt was not found at: {requirements}")

    venv = checkout / "venv"
    python = virtual_environment_python(checkout)

    if not python.is_file():
        if venv.exists() and (not venv.is_dir() or any(venv.iterdir())):
            raise FetchError(
                f"The virtual environment at {venv} is incomplete. Move it aside "
                "or remove it, then rerun this script."
            )
        print(f"\nCreating virtual environment: {venv}")
        try:
            run_checked(
                [sys.executable, "-m", "venv", str(venv)],
                "Could not create the virtual environment",
            )
        except FetchError as error:
            raise FetchError(
                f"{error} On Debian/Ubuntu, install the python3-venv package; "
                "on other systems, ensure your Python installation includes venv."
            ) from error

    print("\nUpgrading pip")
    run_checked(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        "Could not upgrade pip",
    )

    if torch_index:
        print(f"\nInstalling PyTorch from: {torch_index}")
        run_checked(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--index-url",
                torch_index,
                "torch",
                "torchvision",
            ],
            "Could not install PyTorch from the requested index",
        )

    print(f"\nInstalling ComfyUI dependencies from: {requirements}")
    run_checked(
        [str(python), "-m", "pip", "install", "-r", str(requirements)],
        "Could not install ComfyUI's dependencies",
    )
    return python


def fetch(
    destination: Path,
    ref: str,
    use_archive: bool,
    skip_deps: bool,
    torch_index: Optional[str],
) -> None:
    """Fetch and set up ComfyUI without overwriting an unrelated directory."""
    destination = destination.expanduser().resolve()
    already_present = destination.is_dir() and (destination / "main.py").is_file()

    if already_present:
        print(f"ComfyUI is already present at: {destination}")
    else:
        if destination.exists() and not destination.is_dir():
            raise FetchError(f"Destination exists but is not a directory: {destination}")
        if destination.is_dir() and any(destination.iterdir()):
            raise FetchError(
                f"Destination is non-empty and is not a ComfyUI checkout: {destination}\n"
                "Choose another directory or move the existing contents aside."
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.fetch-", dir=destination.parent)
        )
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

            # An empty destination is safe to replace; non-empty directories were
            # rejected above and are never removed.
            if destination.is_dir():
                destination.rmdir()
            checkout.replace(destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        print(f"\nComfyUI {ref!r} fetched to: {destination}")

    if destination != DEFAULT_DIRECTORY.resolve():
        print("Set COMFYUI_DIR to this path before running the demo launcher.")

    if skip_deps:
        print("Dependency installation skipped (--skip-deps).")
        return

    python = install_dependencies(destination, torch_index)
    print("\nComfyUI and its core dependencies are ready.")
    print(f"Virtual environment: {python.parent.parent}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch ComfyUI, create its venv, and install its core dependencies. "
            "Defaults to ~/ComfyUI, which the existing launcher detects."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_DIRECTORY,
        help="checkout directory (default: %(default)s)",
    )
    parser.add_argument(
        "--ref",
        default="master",
        help="branch or tag to fetch (default: %(default)s)",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="download a ZIP archive instead of using Git",
    )
    dependency_options = parser.add_mutually_exclusive_group()
    dependency_options.add_argument(
        "--skip-deps",
        action="store_true",
        help="fetch source code only; do not create a venv or install packages",
    )
    dependency_options.add_argument(
        "--torch-index",
        metavar="URL",
        help=(
            "install torch and torchvision from this pip index before installing "
            "the remaining ComfyUI requirements"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        fetch(args.directory, args.ref, args.archive, args.skip_deps, args.torch_index)
    except (FetchError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nSetup cancelled.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
