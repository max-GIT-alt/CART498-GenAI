"""Session manifest, export archive and static gallery for the process-capture demo."""
import html
import json
import os
import re
import zipfile


EXPORT_NAME = "session-export.zip"


def _esc(value):
    return html.escape(str(value), quote=True)


def _slug(value, fallback="run"):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return (slug or fallback)[:80]


def _inside(base_dir, relative):
    """Resolve ``relative`` and return it only if it stays inside ``base_dir``."""
    if not relative:
        return None
    base_dir = os.path.abspath(base_dir)
    path = os.path.abspath(os.path.join(base_dir, relative))
    try:
        if os.path.commonpath([base_dir, path]) != base_dir:
            return None
    except ValueError:
        return None
    return path if os.path.isfile(path) else None


def build_export_zip(base_dir, runs, path=None):
    """Bundle every run's final image, process GIF and metadata into one archive."""
    path = path or os.path.join(base_dir, EXPORT_NAME)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    written = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for run in runs:
            run_id = str(run.get("run_id") or "")
            if not run_id:
                continue
            for key, default in (("final", "final.png"), ("gif", "process.gif")):
                source = _inside(base_dir, run.get(key) or f"{run_id}/{default}")
                if source:
                    arcname = "session/" + os.path.relpath(source, base_dir).replace(os.sep, "/")
                    archive.write(source, arcname)
                    written += 1
            meta = _inside(base_dir, f"{run_id}/meta.json")
            if meta:
                archive.write(meta, f"session/{run_id}/meta.json")
        manifest = _inside(base_dir, "session.json")
        if manifest:
            archive.write(manifest, "session/session.json")
    os.replace(tmp, path)
    return path, written


def _human_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _download_name(run, extension):
    label = _slug(run.get("label", "generation"), "generation")
    seed = _slug(run.get("seed", "seed"), "seed")
    return f"{label}-{seed}-{_slug(run.get('run_id'), 'run')}.{extension}"


def build_html(base_dir, runs):
    cards = []
    for run in runs:
        gif = run.get("gif", "")
        final = run.get("final", "")
        media = []
        if gif:
            # No loading="lazy": the GIF should start playing as soon as the
            # gallery opens, not only once it scrolls into view.
            media.append(
                f'<img src="{_esc(gif)}" alt="denoising process">'
            )
        if final:
            media.append(
                f'<img src="{_esc(final)}" alt="final image" loading="lazy">'
            )
        if not media:
            continue
        cols = "two" if len(media) > 1 else "one"
        downloads = []
        if final:
            downloads.append(
                f'<a class="download" href="{_esc(final)}" '
                f'download="{_esc(_download_name(run, "png"))}">Download PNG</a>'
            )
        if gif:
            downloads.append(
                f'<a class="download" href="{_esc(gif)}" '
                f'download="{_esc(_download_name(run, "gif"))}">Download GIF</a>'
            )
        meta_path = f"{_esc(run.get('run_id', ''))}/meta.json"
        downloads.append(
            f'<a class="download" href="{meta_path}" '
            f'download="{_esc(_download_name(run, "json"))}">Metadata</a>'
        )
        cards.append(
            f"""
      <article class="card">
        <div class="media {cols}">{''.join(media)}</div>
        <div class="meta">
          <strong>{_esc(run.get('label', 'generation'))}</strong>
          <div class="pills">
            <span class="pill">{_esc(run.get('anchor', 'anchor'))}</span>
            <span class="pill">seed {_esc(run.get('seed', '?'))}</span>
            <span class="pill">{_esc(run.get('steps', '?'))} steps</span>
            <span class="pill">denoise {_esc(run.get('denoise', '?'))}</span>
          </div>
          <div class="detail">{_esc(run.get('sampler', ''))} / {_esc(run.get('scheduler', ''))}
            · {_esc(run.get('frame_count', '?'))} frames · {_esc(run.get('fps', '?'))} fps</div>
          <div class="time">{_esc(run.get('timestamp', ''))}</div>
          <div class="downloads">{''.join(downloads)}</div>
        </div>
      </article>"""
        )

    body = (
        "\n".join(cards)
        if cards
        else '<div class="empty">No generations yet. Run the workflow once.</div>'
    )

    export_link = ""
    if runs:
        archive, written = build_export_zip(base_dir, runs)
        if written:
            export_link = (
                f'<a class="refresh" href="{EXPORT_NAME}" download>'
                f'Download all ({_human_size(os.path.getsize(archive))})</a>'
            )

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Session gallery — process GIFs</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    background: #0f1116;
    color: #e8eaf0;
  }}
  header {{ padding: 30px 34px 8px; }}
  h1 {{ margin: 0 0 6px; font-size: 25px; letter-spacing: -0.02em; }}
  .sub {{ color: #9aa3b2; font-size: 14px; line-height: 1.5; }}
  .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
  .refresh {{
    display: inline-block;
    margin-top: 0;
    padding: 7px 12px;
    border-radius: 8px;
    border: 1px solid #3a4150;
    background: #232838;
    color: #e8eaf0;
    font: inherit;
    font-size: 13px;
    text-decoration: none;
    cursor: pointer;
  }}
  .refresh:hover {{ background: #2c3245; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 18px;
    padding: 22px 34px 48px;
  }}
  .card {{
    background: #181b23;
    border: 1px solid #2a2f3a;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0,0,0,.25);
  }}
  .media {{ display: grid; gap: 1px; background: #2a2f3a; }}
  .media.two {{ grid-template-columns: 1fr 1fr; }}
  .media.one {{ grid-template-columns: 1fr; }}
  .media img {{ width: 100%; height: auto; display: block; background: #0a0b0e; }}
  .meta {{ padding: 13px 15px 15px; }}
  .meta strong {{ font-size: 14px; color: #fff; }}
  .pills {{ margin: 8px 0 6px; }}
  .pill {{
    display: inline-block;
    padding: 2px 8px;
    margin: 0 5px 5px 0;
    border-radius: 999px;
    background: #232838;
    color: #cfd6e4;
    font-size: 11px;
  }}
  .detail {{ color: #aeb7c6; font-size: 12px; line-height: 1.5; }}
  .time {{ color: #6f7888; font-size: 11px; margin-top: 4px; }}
  .downloads {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 11px; }}
  .download {{
    padding: 4px 9px;
    border-radius: 7px;
    border: 1px solid #3a4150;
    background: #202533;
    color: #dfe5f0;
    font-size: 11px;
    text-decoration: none;
  }}
  .download:hover {{ background: #2c3245; border-color: #4c5669; }}
  .empty {{ padding: 40px 34px; color: #9aa3b2; }}
</style>
</head>
<body>
<header>
  <h1>Session gallery</h1>
  <div class="sub">
    Every run from this session, newest first. The left image is the saved denoising
    GIF; the right image is the final full-resolution result. Download single files below,
    or grab everything as one archive. Press Refresh after a new run finishes to add it
    to the gallery.
  </div>
  <div class="toolbar">
    <button class="refresh" onclick="location.reload()">Refresh gallery</button>
    {export_link}
  </div>
</header>
<main class="grid">
{body}
</main>
</body>
</html>
"""
    with open(os.path.join(base_dir, "session.html"), "w", encoding="utf-8") as f:
        f.write(doc)


def append_run(output_dir, session_name, entry):
    base_dir = os.path.join(output_dir, session_name)
    os.makedirs(base_dir, exist_ok=True)
    manifest_path = os.path.join(base_dir, "session.json")
    data = {"runs": []}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"runs": []}
    runs = [r for r in data.get("runs", []) if r.get("run_id") != entry.get("run_id")]
    runs.append(entry)
    runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    data["runs"] = runs
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    build_html(base_dir, runs)
