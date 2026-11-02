"""Minimal ComfyUI API client used by the anchor/transformation demo scripts."""
import json
import os
import time
import uuid

import requests

BASE = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8190").rstrip("/")


def queue(workflow, client_id=None):
    client_id = client_id or uuid.uuid4().hex
    r = requests.post(f"{BASE}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data)
    return data["prompt_id"]


def history(prompt_id):
    r = requests.get(f"{BASE}/history/{prompt_id}", timeout=30)
    r.raise_for_status()
    return r.json().get(prompt_id)


def wait(prompt_id, timeout=1800, poll=2.0):
    start = time.time()
    while time.time() - start < timeout:
        h = history(prompt_id)
        if h:
            status = h.get("status", {})
            if status.get("completed"):
                return h
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, indent=2)[:4000])
        time.sleep(poll)
    raise TimeoutError(f"prompt {prompt_id} timed out after {timeout}s")


def images(h):
    """Return [(filename, subfolder, type)] for all output images in a history entry."""
    out = []
    for node in h.get("outputs", {}).values():
        for im in node.get("images", []):
            out.append((im["filename"], im.get("subfolder", ""), im.get("type", "output")))
    return out


def run(workflow, timeout=1800):
    pid = queue(workflow)
    return wait(pid, timeout=timeout)


def download(filename, subfolder="", dest=None):
    r = requests.get(f"{BASE}/view", params={"filename": filename, "subfolder": subfolder, "type": "output"}, timeout=120)
    r.raise_for_status()
    if dest:
        with open(dest, "wb") as f:
            f.write(r.content)
        return dest
    return r.content
