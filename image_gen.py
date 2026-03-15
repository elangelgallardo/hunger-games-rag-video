"""Image generation for the short-form video pipeline.

Flow:
  1. One Gemini call → generates one visual prompt per script segment
  2. One Imagen 4 call per prompt → saves image to disk

Images are generated at 9:16 aspect ratio (vertical, short-form video format).

Usage:
    python image_gen.py                    # uses script.txt + output/audio metadata
    python image_gen.py <segments_json>    # path to JSON produced by tts.py
"""

import json
import sys
from pathlib import Path

from google import genai
from google.genai import types

IMAGE_MODEL = "gemini-3.1-flash-image-preview"
PROMPT_MODEL = "gemini-3.1-pro-preview"

STYLE_SUFFIX = (
    "vertical 9:16 portrait format, cinematic, dramatic, dystopian Panem, "
    "epic lighting, high contrast, photorealistic, ultra-detailed, "
    "no text, no watermarks, no logos"
)


# ---------------------------------------------------------------------------
# Step 1 — generate image prompts via Gemini
# ---------------------------------------------------------------------------

def generate_prompts(topic: str, segments: list[dict]) -> list[str]:
    """Ask Gemini to produce one visual image prompt per segment.

    Returns a list of prompt strings in the same order as segments.
    """
    numbered = "\n".join(f"{i+1}. {s['text']}" for i, s in enumerate(segments))

    instruction = f"""You are a visual director creating imagery for a short-form video about The Hunger Games.

For each numbered narration segment below, write one image generation prompt that:
- Describes a single cinematic scene that visually represents what is being narrated
- Focuses on environment, mood, lighting, and composition
- Avoids referring to characters by their real names — describe them by role or appearance instead (e.g. "a dark-haired young woman in a black uniform", "an elderly ruler in a white suit")
- Is self-contained so each image makes sense on its own
- Does NOT mention text, logos, watermarks, or UI elements

Return ONLY a numbered list with exactly {len(segments)} lines, one prompt per line, no extra commentary.

Topic: {topic}

Segments:
{numbered}"""

    client = genai.Client()
    response = client.models.generate_content(model=PROMPT_MODEL, contents=instruction)
    raw = response.text.strip()

    # Parse numbered list → plain strings
    prompts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading "1. ", "2. " etc.
        if line[0].isdigit():
            line = line.split(".", 1)[-1].strip()
        prompts.append(line)

    if len(prompts) != len(segments):
        raise ValueError(
            f"Expected {len(segments)} prompts from Gemini, got {len(prompts)}.\nRaw output:\n{raw}"
        )

    return prompts


# ---------------------------------------------------------------------------
# Step 2 — generate images via Gemini image model
# ---------------------------------------------------------------------------

def generate_images(
    prompts: list[str],
    output_dir: Path,
    on_progress: callable = None,
) -> list[Path]:
    """Generate one PNG per prompt using the Gemini image model.

    Returns a list of saved image paths in prompt order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client()
    paths: list[Path] = []

    for i, prompt in enumerate(prompts):
        if on_progress:
            on_progress(i, len(prompts), prompt)

        full_prompt = f"{prompt}, {STYLE_SUFFIX}"

        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[full_prompt],
        )

        image_path = output_dir / f"image_{i:03d}.png"
        for part in response.parts:
            if part.inline_data is not None:
                part.as_image().save(str(image_path))
                break
        else:
            raise RuntimeError(f"No image returned for prompt {i}: {full_prompt[:80]}")

        paths.append(image_path)
        print(f"  [{i+1}/{len(prompts)}] {image_path.name} — {prompt[:70]}{'…' if len(prompt) > 70 else ''}")

    return paths


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def generate_visuals(
    topic: str,
    segments: list[dict],
    output_dir: Path,
    on_progress: callable = None,
) -> list[dict]:
    """Full image pipeline: prompts → images.

    Args:
        topic:       The video topic (used for prompt context).
        segments:    Segment list from tts.generate_tts() output.
        output_dir:  Folder to save images.
        on_progress: Optional callback(current, total, text).

    Returns:
        Enriched segment list — each dict gains:
          {"prompt": str, "image_path": Path}
    """
    print(f"Generating {len(segments)} image prompts…")
    prompts = generate_prompts(topic, segments)

    for i, (seg, prompt) in enumerate(zip(segments, prompts)):
        print(f"  [{i+1}] {prompt[:80]}{'…' if len(prompt) > 80 else ''}")

    print(f"\nGenerating {len(prompts)} images…")
    image_paths = generate_images(prompts, output_dir, on_progress=on_progress)

    results = []
    for seg, prompt, image_path in zip(segments, prompts, image_paths):
        results.append({**seg, "prompt": prompt, "image_path": image_path})

    return results


# ---------------------------------------------------------------------------
# CLI entry point — reads script.txt and tts output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    base = Path(__file__).parent

    # Load topic from script.txt first line (used as context for prompts)
    script_text = (base / "script.txt").read_text(encoding="utf-8").strip()
    topic = script_text.splitlines()[0][:80]  # first line as topic summary

    # Load segments — either from a JSON arg or rebuild from script split
    if len(sys.argv) > 1:
        segments = json.loads(Path(sys.argv[1]).read_text())
    else:
        # Fallback: load from tts output metadata if it exists
        metadata_path = base / "output" / "audio" / "segments.json"
        if metadata_path.exists():
            data = json.loads(metadata_path.read_text())
            segments = data["segments"]
        else:
            # Build segments from script directly (no audio timing)
            from tts import split_script
            segments = [{"index": i, "text": t} for i, t in enumerate(split_script(script_text))]

    output_dir = base / "output" / "images"
    results = generate_visuals(topic=topic, segments=segments, output_dir=output_dir)

    print(f"\nDone — {len(results)} images saved to {output_dir}/")
    for r in results:
        print(f"  [{r['index']}] {r['image_path'].name}")
