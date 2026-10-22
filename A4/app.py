"""
The Readymade Machine
Turns an ordinary object into a work of contemporary art:
  description -> wall label (OpenAI text) -> image prompt (from the label) -> image
"""

import datetime
import json
import os
import re

from dotenv import load_dotenv
from flask import Flask, render_template, request
from openai import OpenAI

from prompts import IMAGE_SYSTEM, LABEL_SCHEMA, LABEL_SYSTEM

load_dotenv()  # reads OPENAI_API_KEY from .env locally; on Render it's an env var

app = Flask(__name__)

TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4.1")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")
IMAGE_QUALITY = os.getenv("IMAGE_QUALITY", "medium")  # low | medium | high
MOCK = os.getenv("MOCK_MODE") == "1"  # offline testing without spending credits

client = None if MOCK else OpenAI()  # picks up OPENAI_API_KEY from the environment

MAX_INPUT_CHARS = 400


# ---------------------------------------------------------------- helpers

def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text))


def generate_label(description: str) -> dict:
    """Stage 1: object description -> structured wall label."""
    system = LABEL_SYSTEM.format(year=datetime.date.today().year)
    messages = [
        {"role": "developer", "content": system},
        {"role": "user", "content": f"The object: {description}"},
    ]
    response = client.responses.create(
        model=TEXT_MODEL,
        input=messages,
        temperature=1.0,
        text={"format": {"type": "json_schema", "name": "wall_label",
                         "schema": LABEL_SCHEMA, "strict": True}},
    )
    label = json.loads(response.output_text)

    # Enforce the 80-120 word rule: one revision pass if the model misses it.
    n = word_count(label["extended_label"])
    if not 80 <= n <= 120:
        messages += [
            {"role": "assistant", "content": response.output_text},
            {"role": "user", "content": (
                f"The extended_label has {n} words. Rewrite it to between 90 and 110 "
                "words, keeping every rule. Return the full JSON again."
            )},
        ]
        response = client.responses.create(
            model=TEXT_MODEL,
            input=messages,
            temperature=0.8,
            text={"format": {"type": "json_schema", "name": "wall_label",
                             "schema": LABEL_SCHEMA, "strict": True}},
        )
        label = json.loads(response.output_text)

    return label


def generate_image_prompt(label: dict) -> str:
    """Stage 2: the label (NOT the raw description) -> image prompt."""
    response = client.responses.create(
        model=TEXT_MODEL,
        input=[
            {"role": "developer", "content": IMAGE_SYSTEM},
            {"role": "user", "content": json.dumps(label, ensure_ascii=False)},
        ],
        temperature=0.7,
    )
    return response.output_text.strip()


def generate_image(prompt: str) -> str:
    """Stage 3: image prompt -> data URI for an <img> tag."""
    result = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        quality=IMAGE_QUALITY,
    )
    b64 = result.data[0].b64_json  # gpt-image-1 always returns base64
    return f"data:image/png;base64,{b64}"


def mock_pipeline(description: str):
    """Canned output so the layout can be tested without an API key."""
    label = {
        "artist_name": "Odile Prax-Wennerström",
        "artist_bio": "b. 1994, lives and works between Verdun and an expired transit pass",
        "title": "Untitled (Residual Commute)",
        "year": "2026",
        "medium": "Paper, thermal ink, sweat, deferred arrival",
        "dimensions": "8.6 × 5.4 × 0.02 cm",
        "provenance": ("Found in a coat pocket, 2026; briefly the property of a laundromat; "
                       "private collection of the artist's other coat; acquired by the museum "
                       "after a tense negotiation"),
        "credit_line": "Gift of the artist's left pocket",
        "extended_label": (
            "Folded twice and softened by repetition, this faded receipt for a single "
            "bus fare operates in a space between transaction and testimony. Prax-Wennerström "
            "foregrounds its crumpledness as a post-economic archive, interrogating the ways in "
            "which mobility is both purchased and forgotten. Its thermal text, now a pale "
            "near-absence, opens a space of para-legibility where memory, labour and lint "
            "collapse into a single plane. Following Ilse Carramore's notion of the "
            "\"pocketed real\", the work troubles the boundary between the city and the body "
            "that carries it, proposing a temporality that is always arriving and never "
            "quite validated."
        ),
        "theorist_name": "Ilse Carramore",
        "theorist_book": "Pocket Ontologies: Notes on the Carried Real (2019)",
    }
    prompt = ("[mock] A tiny faded paper receipt on a white plinth under a single spotlight, "
              "white cube gallery, a distant visitor for scale.")
    image = None  # template shows a placeholder
    return label, prompt, image


# ---------------------------------------------------------------- routes

@app.route("/", methods=["GET", "POST"])
def index():
    description = ""
    label = image_prompt = image = error = None

    if request.method == "POST":
        description = request.form.get("description", "").strip()[:MAX_INPUT_CHARS]
        if not description:
            error = "Please describe an object before submitting it to the collection."
        else:
            try:
                if MOCK:
                    label, image_prompt, image = mock_pipeline(description)
                else:
                    label = generate_label(description)
                    image_prompt = generate_image_prompt(label)
                    image = generate_image(image_prompt)
            except Exception as e:  # show the error rather than a blank 500 page
                app.logger.exception("Pipeline failed")
                error = f"The acquisitions committee rejected this object ({type(e).__name__}: {e})"

    word_total = word_count(label["extended_label"]) if label else None
    return render_template(
        "index.html",
        description=description,
        label=label,
        image=image,
        image_prompt=image_prompt,
        error=error,
        word_total=word_total,
        max_chars=MAX_INPUT_CHARS,
    )


if __name__ == "__main__":
    app.run(debug=True)
