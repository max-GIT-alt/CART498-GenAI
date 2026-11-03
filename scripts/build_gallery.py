"""Build the labelled contact sheets and the standalone HTML gallery."""
import html
import os
import zipfile

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images")
MAT = os.path.join(ROOT, "outputs", "matrix")
LAD = os.path.join(ROOT, "outputs", "ladder")
OUT = os.path.join(ROOT, "outputs")

ANCHORS = [
    ("anchor_1_surreal", "01 · Surreal digital art", "Chihuahua-foot colossus over a village and dome"),
    ("anchor_2_nature", "02 · Nature photograph", "Mossy stream, shallow depth of field"),
    ("anchor_3_collage", "03 · Mixed-media collage", "Einstein, Persian carpet, 1920 newspaper"),
]
COLS = [
    ("ANCHOR — input", "the image every model receives"),
    ("DreamShaper XL", "SDXL · img2img · denoise 0.75"),
    ("NoobAI XL", "SDXL anime · img2img · denoise 0.75"),
    ("Qwen-Image-Edit 2511", "20B DiT · instruction edit · 4 steps"),
]
FONT = "/usr/share/fonts/noto/NotoSans-Regular.ttf"
BOLD = "/usr/share/fonts/noto/NotoSans-Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def cover(im, w, h):
    im = im.convert("RGB")
    r = max(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)
    x = (im.width - w) // 2
    y = (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))


def fit(im, w, h):
    """Contain the image on a card-coloured canvas, preserving its aspect ratio."""
    im = im.convert("RGB")
    r = min(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), "#15151c")
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def matrix_sheet():
    cell, gap, margin = 400, 14, 28
    head = 168
    label_h = 36
    row_h = {}
    for key, _, _ in ANCHORS:
        w, h = Image.open(os.path.join(IMG, key + ".png")).size
        row_h[key] = round(cell * h / w)
    W = margin * 2 + cell * 4 + gap * 3
    H = head + sum(label_h + row_h[k] for k, _, _ in ANCHORS) + gap * 2 + margin
    sheet = Image.new("RGB", (W, H), "#0e0e12")
    d = ImageDraw.Draw(sheet)

    d.text((margin, 26), "ONE ANCHOR, THREE MODELS", font=font(34, True), fill="#ffffff")
    d.text(
        (margin, 74),
        "Same input image, same target concept. Only the diffusion model changes.",
        font=font(18), fill="#a7a7b7",
    )
    x0 = margin
    for i, (title, sub) in enumerate(COLS):
        x = x0 + i * (cell + gap)
        d.text((x, 118), title, font=font(19, True), fill="#ffffff" if i else "#ffd166")
        d.text((x, 143), sub, font=font(13), fill="#8d8da0")

    y = head
    for key, label, note in ANCHORS:
        d.text((margin, y + 4), label, font=font(17, True), fill="#ffffff")
        d.text((margin + 250, y + 7), note, font=font(13), fill="#8d8da0")
        y += label_h
        h = row_h[key]
        paths = [os.path.join(IMG, key + ".png")] + [
            os.path.join(MAT, f"{m}__{key}.png") for m in ("dreamshaper", "noobai", "qwen_edit")
        ]
        for i, p in enumerate(paths):
            x = x0 + i * (cell + gap)
            sheet.paste(cover(Image.open(p), cell, h), (x, y))
            d.rectangle([x, y, x + cell - 1, y + h - 1], outline="#26262f")
        y += h + gap

    sheet.save(os.path.join(OUT, "00_showcase_matrix.png"))
    return sheet


def ladder_sheet():
    cell, gap, margin, head = 400, 14, 28, 168
    items = [
        ("anchor", os.path.join(IMG, "anchor_1_surreal.png"), "ANCHOR", "the input"),
        ("035", os.path.join(LAD, "noobai__anchor_1_surreal__denoise_035.png"), "denoise 0.35", "anchor still dominant"),
        ("055", os.path.join(LAD, "noobai__anchor_1_surreal__denoise_055.png"), "denoise 0.55", "stylised, same scene"),
        ("075", os.path.join(LAD, "noobai__anchor_1_surreal__denoise_075.png"), "denoise 0.75", "model style takes over"),
        ("095", os.path.join(LAD, "noobai__anchor_1_surreal__denoise_095.png"), "denoise 0.95", "anchor only seeds layout"),
    ]
    W = margin * 2 + cell * 5 + gap * 4
    H = head + cell + margin
    sheet = Image.new("RGB", (W, H), "#0e0e12")
    d = ImageDraw.Draw(sheet)
    d.text((margin, 24), "HOW STRONG IS THE ANCHOR?  (NoobAI XL, img2img)", font=font(30, True), fill="#ffffff")
    d.text(
        (margin, 72),
        "Identical anchor, prompt and seed. Raising denoise hands more control to the model.",
        font=font(17), fill="#a7a7b7",
    )
    for i, (_, path, title, sub) in enumerate(items):
        x = margin + i * (cell + gap)
        sheet.paste(fit(Image.open(path), cell, cell), (x, head))
        d.rectangle([x, head, x + cell - 1, head + cell - 1], outline="#26262f")
        d.text((x, 116), title, font=font(18, True), fill="#ffd166" if i == 0 else "#ffffff")
        d.text((x, 141), sub, font=font(13), fill="#8d8da0")
    sheet.save(os.path.join(OUT, "00_showcase_denoise.png"))
    return sheet


def gallery_zip():
    """Bundle every image shown in gallery.html so the demo exports in one click."""
    items = []
    for key, _, _ in ANCHORS:
        items.append(os.path.join(IMG, key + ".png"))
        for model in ("dreamshaper", "noobai", "qwen_edit"):
            items.append(os.path.join(MAT, f"{model}__{key}.png"))
    for suffix in ("035", "055", "075", "095"):
        items.append(os.path.join(LAD, f"noobai__anchor_1_surreal__denoise_{suffix}.png"))
    for name in ("00_showcase_matrix.png", "00_showcase_denoise.png", "00_showcase_micro.png"):
        items.append(os.path.join(OUT, name))

    path = os.path.join(OUT, "gallery-images.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for source in items:
            if os.path.isfile(source):
                archive.write(source, os.path.relpath(source, ROOT))
    return path


def gallery_html():
    gallery_zip()
    rows = []
    for key, label, note in ANCHORS:
        aw, ah = Image.open(os.path.join(IMG, key + ".png")).size
        cells = [
            f'<figure class="cell anchor"><img src="images/{key}.png" alt="anchor">'
            f'<figcaption>ANCHOR<a class="dl" href="images/{key}.png" download>PNG</a></figcaption></figure>'
        ]
        for m, mlabel in (("dreamshaper", "DreamShaper XL"), ("noobai", "NoobAI XL"), ("qwen_edit", "Qwen-Image-Edit 2511")):
            cells.append(
                f'<figure class="cell" data-anchor="images/{key}.png"><img src="outputs/matrix/{m}__{key}.png" alt="{mlabel}">'
                f'<figcaption>{mlabel}<a class="dl" href="outputs/matrix/{m}__{key}.png" download>PNG</a></figcaption></figure>'
            )
        rows.append(
            f'<section class="row" style="--ar: {aw} / {ah}"><div class="rowhead"><h3>{html.escape(label)}</h3>'
            f'<p>{html.escape(note)}</p></div><div class="grid">{"".join(cells)}</div></section>'
        )

    ladder = [
        ("ANCHOR", "images/anchor_1_surreal.png"),
        ("denoise 0.35", "outputs/ladder/noobai__anchor_1_surreal__denoise_035.png"),
        ("denoise 0.55", "outputs/ladder/noobai__anchor_1_surreal__denoise_055.png"),
        ("denoise 0.75", "outputs/ladder/noobai__anchor_1_surreal__denoise_075.png"),
        ("denoise 0.95", "outputs/ladder/noobai__anchor_1_surreal__denoise_095.png"),
    ]
    ladder_html = "".join(
        f'<figure class="cell"><img src="{p}" alt="{t}">'
        f'<figcaption>{t}<a class="dl" href="{p}" download>PNG</a></figcaption></figure>' for t, p in ladder
    )

    micro_sheet = os.path.join(OUT, "00_showcase_micro.png")
    micro_html = ""
    if os.path.isfile(micro_sheet):
        micro_html = f"""
  <h2>A micro model for live demos</h2>
  <p class="sub">DreamShaper 8 LCM — a 2.0 GiB fp16 Stable Diffusion 1.5 checkpoint with LCM
  distillation. Four steps at 512 px, about a second per image once loaded. The times on the
  sheet are wall-clock seconds for each prompt on the RTX 3070 Ti.</p>
  <figure class="cell wide"><img src="outputs/00_showcase_micro.png" alt="micro LCM prompt grid">
  <figcaption>MICRO LCM · 4 STEPS · 512 × 512
  <a class="dl" href="outputs/00_showcase_micro.png" download>PNG</a></figcaption></figure>
"""

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anchor × Model — ComfyUI demo</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #0e0e12; color: #e8e8f0;
         font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  header {{ max-width: 1500px; margin: 0 auto; padding: 56px 28px 28px; }}
  h1 {{ margin: 0 0 10px; font-size: clamp(28px, 4vw, 46px); letter-spacing: -0.02em; }}
  h1 span {{ color: #ffd166; }}
  .lede {{ margin: 0; max-width: 850px; color: #a7a7b7; font-size: 18px; }}
  .pill {{ display: inline-block; margin-top: 18px; padding: 5px 12px; border: 1px solid #2c2c38;
           border-radius: 999px; color: #b9b9c9; font-size: 13px; }}
  button.pill {{ background: #1a1a22; color: #ffd166; border-color: #6b5a20; font: inherit;
                 font-size: 13px; cursor: pointer; }}
  button.pill:active {{ background: #2a2410; }}
  a.pill {{ color: #ffd166; border-color: #6b5a20; text-decoration: none; }}
  a.pill:hover {{ background: #1a1a22; }}
  main {{ max-width: 1500px; margin: 0 auto; padding: 8px 28px 80px; }}
  .row {{ margin: 34px 0 10px; }}
  .rowhead {{ display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin-bottom: 10px; }}
  .rowhead h3 {{ margin: 0; font-size: 20px; }}
  .rowhead p {{ margin: 0; color: #8d8da0; font-size: 14px; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
  .cell {{ margin: 0; position: relative; background: #15151c; border: 1px solid #23232d; border-radius: 10px;
           overflow: hidden; }}
  .cell img {{ display: block; width: 100%; aspect-ratio: var(--ar, 1); object-fit: cover; background: #15151c; }}
  .cell.wide img {{ aspect-ratio: auto; object-fit: contain; }}
  .cell.anchor {{ border-color: #6b5a20; }}
  .cell.anchor figcaption {{ color: #ffd166; }}
  figcaption {{ display: flex; align-items: center; justify-content: space-between; gap: 8px;
                padding: 8px 11px; font-size: 12px; letter-spacing: .04em; text-transform: uppercase;
                color: #9a9aad; border-top: 1px solid #23232d; }}
  .dl {{ color: #ffd166; text-decoration: none; font-size: 11px; letter-spacing: .04em; }}
  .dl:hover {{ text-decoration: underline; }}
  .models {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 6px 0 0; }}
  .models div {{ padding: 10px 12px; border: 1px solid #23232d; border-radius: 8px; font-size: 13px; color: #a7a7b7; }}
  .models b {{ display: block; color: #fff; font-size: 14px; margin-bottom: 2px; }}
  .ladder {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }}
  .notes {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 28px; margin-top: 18px; }}
  .notes ul {{ margin: 0; padding-left: 18px; color: #b9b9c9; }}
  .notes li {{ margin: 7px 0; }}
  code {{ background: #1a1a22; border: 1px solid #26262f; padding: 1px 6px; border-radius: 5px; font-size: 13px; }}
  h2 {{ margin: 54px 0 8px; font-size: 24px; }}
  .sub {{ margin: 0 0 18px; color: #8d8da0; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
  th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #23232d; }}
  th {{ color: #8d8da0; font-weight: 600; }}
  @media (max-width: 900px) {{
    .grid, .ladder, .models {{ grid-template-columns: repeat(2, 1fr); }}
    .notes {{ grid-template-columns: 1fr; }}
  }}
</style></head>
<body>
<header>
  <h1>One anchor. <span>Three models.</span> Three different worlds.</h1>
  <p class="lede">The same three images are used as anchors for three diffusion models with the
  same target concept. What survives — face, composition, palette, style — depends as much on the
  model as on the anchor.</p>
  <button id="peek" class="pill" type="button">Hold to reveal the anchor</button>
  <a class="pill" href="outputs/gallery-images.zip" download>Download all images</a>
  <span class="pill">ComfyUI 0.37 · RTX 3070 Ti 8 GB · SDXL img2img @ 0.75 · Qwen-Image-Edit 2511 (GGUF Q4, 4-step Lightning)</span>
  <span class="pill">+ micro LCM · 4 steps · 512 px · ~1 s/image</span>
</header>
<main>
  <div class="models">
    <div><b>Anchor</b>the input image</div>
    <div><b>DreamShaper XL</b>SDXL, photoreal / versatile</div>
    <div><b>NoobAI XL</b>SDXL, anime / illustration</div>
    <div><b>Qwen-Image-Edit 2511</b>20B DiT, instruction edit</div>
  </div>
  {"".join(rows)}

  <h2>Anchor strength: the denoise ladder</h2>
  <p class="sub">NoobAI XL, anchor 01, identical prompt and seed. Low denoise copies; high denoise invents.</p>
  <div class="ladder">{ladder_html}</div>
  {micro_html}

  <h2>What to look for</h2>
  <div class="notes">
    <ul>
      <li><b>Anchors are not neutral.</b> A photograph pushes the SDXL models toward realism; the anime
      anchor pushes them toward line art. The same model produces visibly different transformations
      depending on which image it is anchored to.</li>
      <li><b>Model identity dominates at denoise 0.75.</b> DreamShaper repaints the fisherman as a
      photoreal fantasy warrior; NoobAI repaints him as an anime character; Qwen-Image-Edit keeps the
      same man and restyles the light and palette.</li>
      <li><b>Structure is the first thing lost by img2img.</b> The 3D bookshop keeps its silhouette at
      low denoise, but at 0.75 both SDXL models replace the tree with figures and architecture.
      Qwen-Image-Edit preserves the layout because the instruction is an edit, not a re-roll.</li>
      <li><b>Denoise is the anchor dial.</b> 0.35 is almost a copy, 0.55 is a stylised version,
      0.75 lets the model's prior take over, 0.95 uses the anchor only as a loose layout hint.</li>
    </ul>
    <div>
      <table>
        <tr><th>Model</th><th>Type</th><th>Method</th></tr>
        <tr><td>DreamShaper XL</td><td>SDXL checkpoint</td><td>img2img, 30 steps, cfg 6, dpmpp_2m/karras, denoise 0.75</td></tr>
        <tr><td>NoobAI XL v1.0</td><td>SDXL anime</td><td>img2img, 30 steps, cfg 6, dpmpp_2m/karras, denoise 0.75</td></tr>
        <tr><td>Qwen-Image-Edit 2511</td><td>20B DiT (GGUF Q4_K_M)</td><td>instruction edit, 4 steps, cfg 1, euler/simple, Lightning LoRA</td></tr>
      </table>
      <p class="sub" style="margin-top:14px">Reproduce with <code>scripts/make_anchors.py</code> and
      <code>scripts/run_matrix.py</code>; open the graphs in <code>workflows/ui/</code>.</p>
    </div>
  </div>
</main>
<script>
  const peekBtn = document.getElementById("peek");
  const peekCells = [...document.querySelectorAll("figure[data-anchor]")];
  let peeking = false;
  function setPeek(on) {{
    if (on === peeking) return;
    peeking = on;
    for (const fig of peekCells) {{
      const img = fig.querySelector("img");
      if (!fig.dataset.result) fig.dataset.result = img.getAttribute("src");
      img.src = on ? fig.dataset.anchor : fig.dataset.result;
    }}
    peekBtn.textContent = on ? "Showing the anchors…" : "Hold to reveal the anchor";
  }}
  peekBtn.addEventListener("pointerdown", (e) => {{ e.preventDefault(); setPeek(true); }});
  for (const ev of ["pointerup", "pointerleave", "pointercancel", "blur"])
    peekBtn.addEventListener(ev, () => setPeek(false));
  window.addEventListener("keydown", (e) => {{ if (e.code === "Space") {{ e.preventDefault(); setPeek(true); }} }});
  window.addEventListener("keyup", (e) => {{ if (e.code === "Space") setPeek(false); }});
</script>
</body></html>
"""
    with open(os.path.join(ROOT, "gallery.html"), "w") as f:
        f.write(doc)


if __name__ == "__main__":
    matrix_sheet()
    ladder_sheet()
    gallery_html()
    print("wrote outputs/00_showcase_matrix.png, outputs/00_showcase_denoise.png, gallery.html")
