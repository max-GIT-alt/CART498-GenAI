"""HTTP export routes for the session gallery.

With ComfyUI running, the gallery is available at::

    http://127.0.0.1:8190/comfy-session/session.html

Relative links in the page keep working there, so single files download through
the same routes as they do from the ``file://`` copy on disk.
"""
import json
import os

import folder_paths
from aiohttp import web
from server import PromptServer

from . import session as session_lib


def _session_base(name):
    output_dir = os.path.abspath(folder_paths.get_output_directory())
    base = os.path.abspath(os.path.join(output_dir, name))
    try:
        if os.path.commonpath([output_dir, base]) != output_dir:
            raise web.HTTPForbidden(text="session path outside the output directory")
    except ValueError:
        raise web.HTTPForbidden(text="invalid session path")
    return base


def _manifest_runs(base):
    manifest = os.path.join(base, "session.json")
    if not os.path.isfile(manifest):
        raise web.HTTPNotFound(text="session manifest not found")
    with open(manifest, encoding="utf-8") as f:
        return json.load(f).get("runs", [])


@PromptServer.instance.routes.get("/comfy-session")
async def session_index(request):
    name = request.query.get("session", "session")
    raise web.HTTPFound(f"/comfy-session/session.html?session={name}")


@PromptServer.instance.routes.get("/comfy-session/export.zip")
async def session_export(request):
    name = request.query.get("session", "session")
    base = _session_base(name)
    runs = _manifest_runs(base)
    archive, written = session_lib.build_export_zip(base, runs)
    if not written:
        raise web.HTTPNotFound(text="nothing to export yet")
    return web.FileResponse(
        archive,
        headers={"Content-Disposition": 'attachment; filename="session-export.zip"'},
    )


@PromptServer.instance.routes.get("/comfy-session/{path:.*}")
async def session_static(request):
    name = request.query.get("session", "session")
    base = _session_base(name)
    relative = request.match_info.get("path") or "session.html"
    path = os.path.abspath(os.path.join(base, relative))
    try:
        inside = os.path.commonpath([base, path]) == base
    except ValueError:
        inside = False
    if not inside or not os.path.isfile(path):
        raise web.HTTPNotFound(text="session file not found")
    return web.FileResponse(path)
