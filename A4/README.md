# The Readymade Machine

*Draft — CART assignment: International Art English generator*

A Flask web app that turns any ordinary object into a work of contemporary art. A visitor describes something from their pocket, bag or desk; the app writes the object's museum wall label (tombstone + extended label) with the OpenAI API, then generates an installation view of the object with the OpenAI Images API, and hangs the two side by side like a gallery wall.

- **Live app:** `https://<your-app>.onrender.com` *(add after deploying)*
- **Repository:** `https://github.com/<you>/readymade-machine` *(add after pushing)*

---

## 1. IAE features and the rules they became

Rule and Levine (2012) describe International Art English as a dialect built from a few recurring moves. I chose the ones that are easiest to spot and funniest to exaggerate, and turned each into an explicit, checkable rule in the system prompt (`prompts.py → LABEL_SYSTEM`).

| IAE feature (Rule & Levine) | How it shows up in IAE | Prompt rule |
|---|---|---|
| **Nominalisation** — adjectives and verbs turned into abstract nouns ("visuality", "the global") | Things stop *doing* and start *having qualities* | **Rule 2**: at least three coinages ("crumpledness", "foldedness"), one with a prefix (post-, para-, proto-, hyper-…) |
| **"Space" as an all-purpose noun** and stock connectors like "the ways in which" | Everything becomes a space; arguments are gestured at, not made | **Rule 3**: "space" twice, never for an actual place; "the ways in which" exactly once |
| **Doubling and lists** — "both … and", triads of abstractions, verbs like *interrogates / problematises* | A small object is credited with questioning something huge | **Rule 4**: one "both … and", one triad, one sentence where the work interrogates something far larger than itself |
| **Theoretical name-dropping** | Authority borrowed from a canon of thinkers | **Rule 5**: exactly one theorist and book — *both invented*. No real artists, critics or curators |
| *(Not IAE, but the joke's engine)* | — | **Rule 1**: stay precise about the object (colour, material, damage) so the gap between the thing and the language stays visible |
| Museum conventions | Deadpan tombstone format | **Rule 6**: absurdity only inside the slots (medium mixes materials + abstractions; provenance is a chain ending in acquisition; "Gift of…") |
| Assignment constraint | — | **Rule 7**: extended label 80–120 words, also **enforced in code**: if the count misses, `app.py` asks the model for one revision |

The label is returned as **structured JSON** (OpenAI structured outputs with a strict schema), so every tombstone field is guaranteed to exist and the template can lay it out reliably.

*TODO before submitting: replace the left column with my own reading notes and one short example sentence from the essay per feature (quoted briefly, with page/section).*

## 2. How the image prompt is derived from the label

The pipeline is a chain of three calls, and the visitor's raw description **never reaches the image model**:

```
description ──▶ [1] label (gpt-4.1, JSON) ──▶ [2] image prompt (gpt-4.1) ──▶ [3] image (gpt-image-1)
                                   │                       ▲
                                   └──── label JSON only ──┘
```

Step 2 (`prompts.py → IMAGE_SYSTEM`) receives only the label JSON and writes the prompt from it:

- **Object** — inferred from the title, medium and extended label. Abstract "materials" (*fatigue*, *longing*) are not drawn literally; they become lighting and mood.
- **Scale** — read literally from the `dimensions` field:
  - small cm measurements → object alone on a plinth or in a vitrine, a distant visitor for scale
  - monumental measurements (e.g. 600 × 400 cm) → the object fills the room, visitors are tiny
  - "Dimensions variable" → multiple copies or fragments spread through the space
- **Curatorial setting** — the model picks one of: white cube, museum vitrine, lone plinth under a spotlight, biennale pavilion, converted industrial hall.
- **Style constraints** — documentary installation photography, no legible text, no logos, no real artworks, no recognisable people.

The images API returns base64 (`b64_json`); `app.py` wraps it as a `data:image/png;base64,…` URI for the `<img>` tag, so nothing is written to disk. The generated image prompt is shown under **Registrar's notes** on the page, so you can see the chain working.

## 3. User guide

### Using the app
1. Open the site.
2. In **Object for consideration**, describe one real object as precisely as you can: what it is, its colour and material, any damage, where it was found. *Example: "a blue BIC pen with a chewed cap and no ink left, found in my jacket."*
3. Click **Submit to the collection**. Generation takes roughly 20–60 seconds (label, then image prompt, then image).
4. The work appears on the wall: the installation view on the left, the label beside it (tombstone on top, extended label below with its theorist citation).
5. Open **Registrar's notes** to see the word count and the image prompt that was written from the label.
6. Edit your description and submit again to accession a new work.

Tip: specific, slightly sad details (melted, expired, half-eaten, found under a radiator) produce the best labels.

### Running locally
```bash
python3 -m venv venv
source venv/bin/activate            # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
mv .env-bup .env                    # then put your OPENAI_API_KEY in .env
python3 app.py                      # open http://127.0.0.1:5000
```
Set `MOCK_MODE=1` in `.env` to test the layout with a canned label and no API calls.

### Deploying on Render
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app --timeout 240` (image generation can exceed the default 30 s)
- **Environment variable:** `OPENAI_API_KEY` — never commit it; `.env` is in `.gitignore`.
- Optional env vars: `TEXT_MODEL`, `IMAGE_MODEL`, `IMAGE_QUALITY` (`low` is faster and cheaper while testing).

### Files
| File | Purpose |
|---|---|
| `app.py` | Flask routes and the three-stage pipeline |
| `prompts.py` | Label system prompt (the rules), JSON schema, image-prompt system prompt |
| `templates/index.html` | Form, loading state, gallery wall layout |
| `static/style.css` | Gallery styling (responsive; label stacks under the image on phones) |

## 4. Reflection: why is a language model so fluent in this dialect?

*Draft — rewrite in my own voice.*

- **It's statistically dense.** Rule and Levine built their study from thousands of e-flux press releases: IAE exists in enormous, highly repetitive quantities online. A model trained on web text has seen the same moves (nominalisations, "space", "interrogates") thousands of times, in near-identical contexts.
- **It's form without a referent.** IAE often sounds meaningful while committing to little that can be checked. A language model produces plausible continuations without needing a world to check them against, so the dialect suits it: neither of them needs the object to be what the words say it is.
- **It's a genre of recombination.** The dialect is assembled from a small kit of interchangeable parts. That is close to what a language model does best: recombine familiar patterns into fluent new arrangements.
- **The failure point is the interesting part.** The model is *too* fluent: without Rule 1 it drifts away from the granola bar into pure abstraction. The machine needed rules to stay tied to the object, which suggests that the hard part of writing about art was never the dialect, it was looking.
- *Question to develop:* if a machine can produce the dialect on demand, does that empty it out, or show that it was always a set of institutional signals rather than a description?

## References
Rule, Alix, and David Levine. 2012. "International Art English." *Triple Canopy*, no. 16.
