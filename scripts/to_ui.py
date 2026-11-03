"""Convert a ComfyUI API-format workflow into a UI-format workflow.

ComfyUI's API (`/prompt`) wants the flat {id: {class_type, inputs}} form, while the
editor's "Load" button wants the graph form (nodes/links/widgets_values). This
little converter lets the demo ship both without hand-authoring the graphs.

Usage: python to_ui.py <api_workflow.json> <out_ui.json>
"""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8190").rstrip("/")
SOCKET_TYPES = {
    "MODEL", "CLIP", "VAE", "IMAGE", "LATENT", "CONDITIONING", "MASK",
    "CONTROL_NET", "STYLE_MODEL", "CLIP_VISION", "UPSCALE_MODEL", "SAMPLER",
    "SIGMAS", "GUIDER", "NOISE", "AUDIO", "VIDEO",
}

_info_cache = {}


def node_info(cls):
    if cls not in _info_cache:
        with urllib.request.urlopen(f"{BASE}/object_info/{cls}") as r:
            _info_cache[cls] = json.load(r)[cls]
    return _info_cache[cls]


def is_widget(spec):
    t = spec[0]
    return not (isinstance(t, str) and t in SOCKET_TYPES)


def widget_specs(cls):
    info = node_info(cls)
    out = []
    for section in ("required", "optional"):
        for name, spec in info["input"].get(section, {}).items():
            opts = spec[1] if len(spec) > 1 else {}
            if is_widget(spec) and not opts.get("forceInput") and not (
                opts.get("advanced") and section == "optional"
            ):
                out.append((name, spec))
    return out


def widget_values(cls, inputs):
    """Build widgets_values in the order the ComfyUI frontend expects."""
    if cls == "KSampler":
        return [
            inputs.get("seed", 0), "randomize", inputs.get("steps", 20),
            inputs.get("cfg", 8.0), inputs.get("sampler_name", "euler"),
            inputs.get("scheduler", "normal"), inputs.get("denoise", 1.0),
        ]
    if cls == "LoadImage":
        return [inputs.get("image", ""), "image"]
    vals = []
    for name, spec in widget_specs(cls):
        value = inputs.get(name)
        is_link = (
            isinstance(value, list)
            and len(value) == 2
            and isinstance(value[0], str)
        )
        t = spec[0]
        opts = spec[1] if len(spec) > 1 else {}
        if name in inputs and not is_link:
            vals.append(value)
        else:
            if isinstance(t, list):
                vals.append(t[0])
            elif t == "COMBO":
                vals.append(opts.get("options", [None])[0])
            else:
                vals.append(opts.get("default"))
        # A seed widget with control_after_generate is followed by a second
        # widget (fixed / increment / decrement / randomize) in the UI.
        if opts.get("control_after_generate"):
            vals.append("fixed")
    return vals


def convert(api):
    ids = list(api.keys())
    info = {nid: node_info(api[nid]["class_type"]) for nid in ids}

    # ---- topological order (longest-path depth too) ----
    deps = {nid: [] for nid in ids}
    for nid, node in api.items():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and str(v[0]) in api:
                deps[nid].append(str(v[0]))
    depth, order, seen = {}, [], set()

    def visit(n, stack=()):
        if n in depth:
            return depth[n]
        d = 0
        for p in deps[n]:
            if p not in stack:
                d = max(d, visit(p, stack + (n,)) + 1)
        depth[n] = d
        return d

    for nid in ids:
        visit(nid)
    order = sorted(ids, key=lambda n: (depth[n], ids.index(n)))

    # ---- nodes ----
    nodes, links = [], []
    link_id = 0
    node_by_id = {}
    rows = {}
    for nid in order:
        cls = api[nid]["class_type"]
        inf = info[nid]
        inputs, outputs = [], []
        node = {
            "id": int(nid),
            "type": cls,
            "pos": [80 + depth[nid] * 430, 80 + rows.get(depth[nid], 0) * 210],
            "size": [360, 130 + 22 * max(0, len(widget_values(cls, api[nid]["inputs"])) - 1)],
            "flags": {},
            "order": order.index(nid),
            "mode": 0,
            "inputs": inputs,
            "outputs": outputs,
            "properties": {"Node name for S&R": cls},
            "widgets_values": widget_values(cls, api[nid]["inputs"]),
        }
        rows[depth[nid]] = rows.get(depth[nid], 0) + 1

        # linked socket inputs
        for name, v in api[nid]["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and str(v[0]) in api:
                src, slot = str(v[0]), v[1]
                link_id += 1
                t = info[src]["output"][slot]
                links.append([link_id, int(src), slot, int(nid), len(inputs), t])
                inputs.append({"name": name, "type": t, "link": link_id})

        # output sockets
        for slot, (t, nm) in enumerate(zip(inf["output"], inf["output_name"])):
            outputs.append({"name": nm, "type": t, "links": [], "slot_index": slot})

        node_by_id[int(nid)] = node
        nodes.append(node)

    # attach link ids to output sockets
    for lid, src, slot, _, _, _ in links:
        node_by_id[src]["outputs"][slot]["links"].append(lid)

    return {
        "id": "00000000-0000-0000-0000-000000000000",
        "revision": 0,
        "last_node_id": max(int(n) for n in ids),
        "last_link_id": link_id,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        api = json.load(f)
    ui = convert(api)
    with open(dst, "w") as f:
        json.dump(ui, f, indent=2)
    print("wrote", dst)


if __name__ == "__main__":
    main()
