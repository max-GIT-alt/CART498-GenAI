"""Cross-platform launcher for the demo (Linux, macOS, Windows).

Examples
--------
    python scripts/bootstrap.py micro
    python scripts/bootstrap.py micro --prompt "a red fox in snow" --steps 6
    python scripts/bootstrap.py all --skip-matrix
    python scripts/bootstrap.py session --anchor 1 --model dreamshaper
    python scripts/bootstrap.py server

Set ``COMFYUI_DIR`` when ComfyUI is not in a standard location. Set
``COMFYUI_PYTHON`` to override the interpreter used to run ComfyUI and the
helper scripts. ``COMFYUI_PORT`` and ``COMFYUI_URL`` work as before.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "comfy-base"
LOG = ROOT / "comfyui.log"
PID = ROOT / "comfyui.pid"


def log(message):
    print(message, flush=True)


def comfyui_at(candidate):
    """Return the folder holding main.py, accepting a portable build's root.

    The Windows portable archive unpacks to ``ComfyUI_windows_portable`` with
    ``ComfyUI/main.py`` and ``python_embeded/python.exe`` as siblings. Users
    naturally point COMFYUI_DIR at the outer folder, so accept both levels.
    """
    for path in (candidate, candidate / "ComfyUI"):
        if (path / "main.py").is_file():
            return path.resolve()
    return None


def find_comfyui():
    configured = os.environ.get("COMFYUI_DIR", "").strip().strip('"')
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
        found = comfyui_at(Path(configured).expanduser())
        if found:
            return found
    if sys.platform.startswith("win"):
        home = Path.home()
        candidates += [
            Path(r"C:\ComfyUI"),
            Path(r"C:\ComfyUI_windows_portable"),
            home / "ComfyUI",
            home / "comfyui",
            home / "ComfyUI_windows_portable",
            home / "Documents" / "ComfyUI",
            home / "Documents" / "ComfyUI_windows_portable",
            home / "Desktop" / "ComfyUI_windows_portable",
            home / "Downloads" / "ComfyUI_windows_portable",
        ]
    elif sys.platform == "darwin":
        candidates += [
            Path.home() / "ComfyUI",
            Path.home() / "comfyui",
            Path.home() / "Documents" / "ComfyUI",
            Path("/Applications/ComfyUI"),
        ]
    else:
        candidates += [
            Path("/opt/comfyui"),
            Path.home() / "ComfyUI",
            Path.home() / "comfyui",
            Path.home() / "Documents" / "ComfyUI",
        ]
    for candidate in candidates:
        found = comfyui_at(candidate)
        if found:
            return found
    raise SystemExit(
        "ComfyUI not found. Set COMFYUI_DIR to the folder containing main.py.\n"
        "  Windows portable: C:\\ComfyUI_windows_portable\\ComfyUI\n"
        "  Linux/macOS:      export COMFYUI_DIR=~/ComfyUI\n"
        "  Windows cmd:      set COMFYUI_DIR=C:\\ComfyUI\n"
        "  PowerShell:       $env:COMFYUI_DIR = \"C:\\ComfyUI\""
    )


def find_python(comfyui):
    configured = os.environ.get("COMFYUI_PYTHON", "").strip().strip('"')
    if configured:
        return configured
    candidates = [
        comfyui / "venv" / "Scripts" / "python.exe",
        comfyui / "venv" / "bin" / "python",
        comfyui / ".venv" / "Scripts" / "python.exe",
        comfyui / ".venv" / "bin" / "python",
        comfyui / "python_embeded" / "python.exe",
        comfyui / "python_embeded" / "python",
        # Windows portable layout: ComfyUI_windows_portable/python_embeded
        # sits next to ComfyUI_windows_portable/ComfyUI.
        comfyui.parent / "python_embeded" / "python.exe",
        comfyui.parent / "python_embeded" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def server_url():
    configured = os.environ.get("COMFYUI_URL")
    if configured:
        return configured.rstrip("/")
    return f"http://127.0.0.1:{os.environ.get('COMFYUI_PORT', '8190')}"


def server_up(url):
    try:
        with urllib.request.urlopen(f"{url}/system_stats", timeout=2):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def node_available(url, name):
    try:
        with urllib.request.urlopen(f"{url}/object_info/{name}", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def replace_path(destination):
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    elif destination.is_dir():
        shutil.rmtree(destination)


def link_or_copy(source, destination):
    """Use a symlink when possible, copy when Windows forbids symlinks."""
    if not source.exists():
        return
    replace_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, destination, target_is_directory=source.is_dir())
        return
    except (OSError, NotImplementedError):
        pass
    if source.is_dir():
        shutil.copytree(
            source, destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    else:
        shutil.copy2(source, destination)


def prepare_base_dir(comfyui):
    (BASE_DIR / "custom_nodes").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "temp").mkdir(parents=True, exist_ok=True)
    (ROOT / ".comfy-user").mkdir(parents=True, exist_ok=True)

    link_or_copy(
        ROOT / "custom_nodes" / "ComfyUI-Session",
        BASE_DIR / "custom_nodes" / "ComfyUI-Session",
    )
    gguf = comfyui / "custom_nodes" / "ComfyUI-GGUF"
    if gguf.exists():
        link_or_copy(gguf, BASE_DIR / "custom_nodes" / "ComfyUI-GGUF")
    else:
        log("note: ComfyUI-GGUF not found; the optional Qwen graph will not load")
    websocket = comfyui / "custom_nodes" / "websocket_image_save.py"
    if websocket.is_file():
        link_or_copy(websocket, BASE_DIR / "custom_nodes" / "websocket_image_save.py")


def start_server(comfyui, python, url):
    port = os.environ.get("COMFYUI_PORT", "8190")
    prepare_base_dir(comfyui)
    db = (ROOT / ".comfy-user" / "comfyui.db").as_posix()
    command = [
        python, str(comfyui / "main.py"),
        "--base-directory", str(BASE_DIR),
        "--listen", "127.0.0.1", "--port", port,
        "--output-directory", str(ROOT / "outputs"),
        "--input-directory", str(ROOT / "images"),
        "--extra-model-paths-config", str(ROOT / "extra_model_paths.yaml"),
        "--user-directory", str(ROOT / ".comfy-user"),
        "--temp-directory", str(BASE_DIR / "temp"),
        "--database-url", f"sqlite:///{db}",
        "--preview-method", "auto", "--preview-size", "512", "--disable-auto-launch",
    ]
    if sys.platform != "darwin":
        command.append("--lowvram")
    extra = os.environ.get("COMFYUI_ARGS")
    if extra:
        import shlex
        command += shlex.split(extra, posix=(os.name != "nt"))

    log(f"starting ComfyUI on {url} ...")
    with open(LOG, "ab") as output:
        output.write(
            f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{subprocess.list2cmdline(command)}\n".encode("utf-8", "replace")
        )
        output.flush()
        kwargs = {"cwd": str(ROOT), "stdout": output, "stderr": subprocess.STDOUT}
        if sys.platform.startswith("win"):
            kwargs["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **kwargs)
    PID.write_text(str(process.pid))

    deadline = time.time() + 120
    while time.time() < deadline:
        if server_up(url):
            break
        if process.poll() is not None:
            break
        time.sleep(1)
    if not server_up(url):
        log(f"ComfyUI did not start; see {LOG}")
        if LOG.is_file():
            lines = LOG.read_text(errors="replace").splitlines()[-20:]
            log("\n".join(lines))
        log("check that COMFYUI_DIR points at the folder containing main.py and that")
        log("COMFYUI_PYTHON is ComfyUI's own python (venv or python_embeded).")
        log("If the log ends with 'No module named torch', the launcher is using the")
        log("wrong interpreter; set COMFYUI_PYTHON to the portable python.exe.")
        raise SystemExit(1)


def ensure_server(comfyui, python):
    url = server_url()
    if server_up(url):
        if not node_available(url, "KSamplerFrames"):
            log("warning: the running ComfyUI does not expose KSamplerFrames;")
            log("         the session workflow needs a server started by this launcher.")
        return url
    start_server(comfyui, python, url)
    return url


def run_script(python, name, args=()):
    script = ROOT / "scripts" / name
    command = [python, str(script), *args]
    env = os.environ.copy()
    env["COMFYUI_URL"] = server_url()
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


def cmd_micro(args, comfyui, python):
    run_script(python, "fetch_models.py")
    ensure_server(comfyui, python)
    run_script(python, "run_micro.py", args)


def cmd_session(args, comfyui, python, open_browser=True):
    ensure_server(comfyui, python)
    run_script(python, "run_session.py", args)
    gallery = ROOT / "outputs" / "session" / "session.html"
    log(f"session gallery: {gallery}")
    if open_browser:
        webbrowser.open(gallery.as_uri())


def cmd_all(args, comfyui, python, skip_matrix=False):
    skip_matrix = skip_matrix or os.environ.get("DEMO_SKIP_MATRIX") == "1"
    run_script(python, "fetch_models.py")
    ensure_server(comfyui, python)
    for script in ("build_workflows.py", "build_session_workflow.py", "install_workflows.py"):
        run_script(python, script)
    run_script(python, "prepare_anchors.py")
    run_script(python, "run_micro.py")
    log(f"micro demo: {ROOT / 'outputs' / '00_showcase_micro.png'}")

    optional = [
        ROOT / "models" / "checkpoints" / "dreamshaper-xl.safetensors",
        ROOT / "models" / "checkpoints" / "NoobAI-XL-v1.0.safetensors",
        ROOT / "models" / "unet" / "qwen-image-edit-2511-Q4_K_M.gguf",
        ROOT / "models" / "text_encoders" / "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        ROOT / "models" / "vae" / "qwen_image_vae.safetensors",
        ROOT / "models" / "loras" / "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    ]
    full = all(path.is_file() for path in optional) and not skip_matrix
    if full:
        run_script(python, "run_matrix.py")
        run_script(python, "build_gallery.py")
        log(f"done: open {ROOT / 'gallery.html'}")
    elif skip_matrix:
        log("matrix skipped (--skip-matrix)")
    else:
        log("matrix skipped: local SDXL/Qwen models not found in models/")
        log("add or symlink them as described in models/README.md, then rerun.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("micro", help="download the micro model and render the live grid")
    session_parser = sub.add_parser("session", help="run one process-capture session")
    session_parser.add_argument("--no-open", action="store_true",
                                help="do not open the session gallery in a browser")
    all_parser = sub.add_parser("all", help="micro demo, plus the matrix when local models exist")
    all_parser.add_argument("--skip-matrix", action="store_true",
                            help="run only the micro demo even if the big models are present")
    sub.add_parser("server", help="start/reuse ComfyUI without generating anything")
    args, passthrough = parser.parse_known_args(argv)

    comfyui = find_comfyui()
    python = find_python(comfyui)
    log(f"ComfyUI: {comfyui}")
    log(f"python:  {python}")
    if (not os.environ.get("COMFYUI_PYTHON")
            and os.path.normcase(os.path.abspath(python))
            == os.path.normcase(os.path.abspath(sys.executable))):
        log("note: no venv or python_embeded found in the ComfyUI folder; using the")
        log("      interpreter that runs the launcher. If ComfyUI reports")
        log("      'No module named torch', set COMFYUI_PYTHON to ComfyUI's python.exe.")

    if args.command == "micro":
        cmd_micro(passthrough, comfyui, python)
    elif args.command == "session":
        cmd_session(passthrough, comfyui, python, open_browser=not args.no_open)
    elif args.command == "all":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        cmd_all([], comfyui, python, skip_matrix=args.skip_matrix)
    elif args.command == "server":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        url = ensure_server(comfyui, python)
        log(f"ComfyUI ready: {url}")


if __name__ == "__main__":
    main()
