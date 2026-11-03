# Recorded demo — fallback for the live walkthrough

**File:** [`anchor-model-demo.mp4`](anchor-model-demo.mp4)  
**Length:** 1:46 · **Format:** 1920×1080, H.264, 30 fps · **Audio:** none — the
captions are burned into the video.

Use this if the live ComfyUI run fails. It shows the real UI on the demo machine
(Linux, RTX 3070 Ti), not a mock-up: workflow 04 is loaded, queued, and the 2 GB
LCM checkpoint is loaded on the first run. The result and the gallery are the
actual files from this repository.

Poster: [`poster.png`](poster.png)

## Chapters

| Time | Section |
|------|---------|
| 0:00 | Title — *Anchor × Model* |
| 0:07 | The experiment: three anchors, one prompt |
| 0:17 | The 3 × 3 matrix: DreamShaper XL, NoobAI XL, Qwen-Image-Edit |
| 0:29 | The denoise ladder (0.35 → 0.95) |
| 0:38 | **Live demo:** workflow 04, micro LCM, 4 steps at 512 px |
| 0:56 | `gallery.html` — every result and the anchor reveal |
| 1:19 | Process capture — one preview per sampling step |
| 1:27 | How to run it on Windows (`main.py`, `COMFYUI_DIR`, line endings) |
| 1:40 | Close |

## Narration script (optional)

The video is silent so it can be played anywhere. If you want to talk over it,
these lines match the chapters:

> **0:00** Anchor × Model. Three very different input images, three diffusion
> models, one target concept. The point is that an anchor has no fixed effect:
> what a model does with it depends on the model's prior, and on how much
> denoise you give it.
>
> **0:07** These are the three anchors: a surreal chihuahua-foot colossus, a
> mossy stream, and a collage with a readable newspaper headline.
>
> **0:17** Same prompt, same seed, three models. DreamShaper XL turns the
> colossus into a blonde warrior; NoobAI makes an anime traveller; Qwen-Image-Edit
> keeps the chihuahua and changes the light. At denoise 0.75 the model's prior
> usually wins.
>
> **0:29** The denoise ladder turns the anchor into a dial: 0.35 is almost a
> copy, 0.95 keeps only a loose layout hint.
>
> **0:38** For the live part I use the micro model: a 2 GB LCM checkpoint. Four
> steps at 512 pixels, about a second per image once it is warm. I load workflow
> 04 and queue it; the first run also loads the checkpoint.
>
> **0:56** Every result is collected in `gallery.html`, together with the anchors
> and PNG downloads.
>
> **1:19** Workflow 01 can also save a process GIF: one latent preview per
> sampling step, plus `final.png` and `meta.json`.
>
> **1:27** On Windows it is two commands: fetch the model, then `run_micro.cmd`.
> `main.py` belongs to ComfyUI, not this project — set `COMFYUI_DIR` to the
> ComfyUI folder that contains it.
>
> **1:40** Thanks — the README covers the matrix, the optional SDXL and Qwen
> models, and the Windows notes.

## Rebuilding the recording

The video was assembled from files in this repository:

- `workflows/ui/04_micro_lcm.json` — queued live in the ComfyUI UI
- `docs/screenshots/00_showcase_matrix.png`, `…_denoise.png`,
  `…_micro.png` — result slides
- `outputs/session/<run>/process.gif` — process-capture segment
- `gallery.html` — results walkthrough

It is a screen recording of the ComfyUI frontend in headless Chrome
(`Page.startScreencast`), with title and explainer cards rendered at 1920×1080 and
captions composited with `ffmpeg`. The recording script is not required to play
the demo.
