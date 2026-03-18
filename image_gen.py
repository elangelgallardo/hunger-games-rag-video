"""Image generation for the short-form video pipeline.

Flow:
  1. One Gemini call generates one visual prompt per script segment
  2. One image model call per prompt saves a PNG to disk

Images are generated at 9:16 aspect ratio (vertical, short-form video format).

Usage:
    python image_gen.py                    # uses script.txt + output/audio metadata
    python image_gen.py <segments_json>    # path to JSON produced by tts.py
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

from api_retry import retry_api_call
from pipeline_config import (
    API_TIMEOUT_MS,
    IMAGE_MODEL,
    IMAGE_STYLE_SUFFIX,
    MAX_RETRIES,
    PROMPT_MODEL,
)

log = logging.getLogger(__name__)


def _build_style_suffix(wiki_name: str = "") -> str:
    """Build a style suffix that includes the wiki universe for visual grounding."""
    if wiki_name:
        return f"{wiki_name} inspired, {IMAGE_STYLE_SUFFIX}"
    return IMAGE_STYLE_SUFFIX


# ---------------------------------------------------------------------------
# Step 1 — generate image prompts via Gemini
# ---------------------------------------------------------------------------

def generate_prompts(
    topic: str,
    segments: list[dict],
    images_per_segment: int = 1,
    wiki_name: str = "",
) -> list[str]:
    """Ask Gemini to produce visual image prompts for each segment.

    When *images_per_segment* > 1, each segment gets multiple distinct prompts
    showing different angles/moments of the same narration.

    Returns a flat list of prompt strings (len = segments * images_per_segment).
    """
    total_prompts = len(segments) * images_per_segment
    numbered = "\n".join(f"{i+1}. {s['text']}" for i, s in enumerate(segments))

    universe_line = f"All imagery must be visually grounded in the {wiki_name} universe." if wiki_name else ""

    if images_per_segment == 1:
        count_instruction = "write one image generation prompt per segment"
        format_instruction = (
            f"Return ONLY a numbered list with exactly {total_prompts} lines, "
            "one prompt per line, no extra commentary."
        )
    else:
        count_instruction = f"write exactly {images_per_segment} image generation prompts per segment"
        format_instruction = (
            f"Return ONLY a numbered list with exactly {total_prompts} lines total "
            f"({images_per_segment} prompts per segment, in order). "
            "For each segment, vary the angle, composition, or moment — "
            "show different aspects of the same scene. No extra commentary."
        )

    instruction = f"""You are a visual director creating imagery for a short-form video about: {topic}
{universe_line}

For each numbered narration segment below, {count_instruction} that:
- Describes a single cinematic scene that visually represents what is being narrated
- Maintains a consistent visual style and atmosphere grounded in the topic's universe throughout all prompts
- Focuses on environment, mood, lighting, and composition
- Avoids referring to characters by their real names — describe them by role or appearance instead (e.g. "a dark-haired young woman in a black uniform", "an elderly ruler in a white suit")
- Is self-contained so each image makes sense on its own
- Does NOT include style keywords like "cinematic" or "photorealistic" — those will be added separately
- Does NOT mention text, logos, watermarks, or UI elements

{format_instruction}

Topic: {topic}

Segments:
{numbered}"""

    client = genai.Client()
    log.info("Generating %d image prompts with %s", total_prompts, PROMPT_MODEL)
    t0 = time.time()
    response = client.models.generate_content(model=PROMPT_MODEL, contents=instruction)
    log.info("Prompt generation took %.1fs", time.time() - t0)
    raw = response.text.strip()
    log.info("Raw prompt response (%d chars):\n%s", len(raw), raw[:500])

    # Parse numbered list → plain strings
    def _parse_prompts(text: str) -> list[str]:
        prompts = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^\**\d+[\.\)]\**\s*", "", line).strip()
            if cleaned and len(cleaned) > 15:
                prompts.append(cleaned)
        return prompts

    prompts = _parse_prompts(raw)
    log.info("Parsed %d prompts (expected %d)", len(prompts), total_prompts)

    # Retry once if count is wrong
    if len(prompts) != total_prompts:
        log.warning("Prompt count mismatch (%d vs %d), retrying", len(prompts), total_prompts)
        print(f"  Warning: expected {total_prompts} prompts, got {len(prompts)}. Retrying…")
        t0 = time.time()
        response = client.models.generate_content(model=PROMPT_MODEL, contents=instruction)
        log.info("Prompt retry took %.1fs", time.time() - t0)
        raw = response.text.strip()
        log.info("Retry raw response (%d chars):\n%s", len(raw), raw[:500])
        prompts = _parse_prompts(raw)
        log.info("Retry parsed %d prompts", len(prompts))

    if len(prompts) != total_prompts:
        if len(prompts) > total_prompts:
            prompts = prompts[:total_prompts]
        elif len(prompts) > 0:
            print(f"  Warning: padding {total_prompts - len(prompts)} missing prompts by repeating last")
            while len(prompts) < total_prompts:
                prompts.append(prompts[-1])
        else:
            raise ValueError(f"Gemini returned no usable prompts.\nRaw output:\n{raw}")

    return prompts


# ---------------------------------------------------------------------------
# Step 2 — generate images via Gemini image model
# ---------------------------------------------------------------------------

def generate_images(
    prompts: list[str],
    output_dir: Path,
    on_progress: callable = None,
    wiki_name: str = "",
) -> list[Path]:
    """Generate one PNG per prompt using the Gemini image model.

    Skips images that already exist on disk (resume after failure).
    Retries transient API errors with disconnect-aware delays.
    Falls back to a placeholder if all retries fail.

    Returns a list of saved image paths in prompt order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client_ref = [genai.Client(http_options={"timeout": API_TIMEOUT_MS})]
    paths: list[Path] = []
    failed_count = 0
    style = _build_style_suffix(wiki_name)

    log.info("Starting image generation: %d prompts, model=%s, wiki=%s",
             len(prompts), IMAGE_MODEL, wiki_name or "(none)")

    for i, prompt in enumerate(prompts):
        image_path = output_dir / f"image_{i:03d}.png"

        # Resume: skip already-generated images
        if image_path.exists() and image_path.stat().st_size > 0:
            log.info("[%d/%d] %s exists (%d bytes), skipping",
                     i+1, len(prompts), image_path.name, image_path.stat().st_size)
            paths.append(image_path)
            if on_progress:
                on_progress(i, len(prompts), f"(cached) {prompt[:50]}")
            continue

        if on_progress:
            on_progress(i, len(prompts), prompt)

        full_prompt = f"{prompt}, {style}"
        log.info("[%d/%d] Requesting image: %s", i+1, len(prompts), full_prompt[:120])

        # Capture loop vars for closure
        _i, _full_prompt, _image_path = i, full_prompt, image_path

        def make_image_call(_i=_i, _full_prompt=_full_prompt, _image_path=_image_path):
            t0 = time.time()
            response = client_ref[0].models.generate_content(
                model=IMAGE_MODEL,
                contents=[_full_prompt],
            )
            elapsed = time.time() - t0
            n_parts = len(response.parts) if hasattr(response, "parts") and response.parts else 0
            log.info("[%d/%d] API responded in %.1fs, %d parts", _i+1, len(prompts), elapsed, n_parts)

            if not hasattr(response, "parts") or not response.parts:
                finish_info = "unknown"
                if hasattr(response, "candidates") and response.candidates:
                    finish_info = getattr(response.candidates[0], "finish_reason", "unknown")
                raise RuntimeError(f"No image data returned (finish_reason={finish_info})")

            for part in response.parts:
                if part.inline_data is not None:
                    part.as_image().save(str(_image_path))
                    log.info("[%d/%d] Saved -> %s", _i+1, len(prompts), _image_path.name)
                    return
            raise RuntimeError("Response parts had no inline image data")

        try:
            retry_api_call(
                make_image_call,
                description=f"Image [{i+1}/{len(prompts)}]",
                on_retry=lambda attempt, wait, err, _i=i: (
                    print(f"  [{_i+1}/{len(prompts)}] attempt {attempt+1} failed: {err}\n"
                          f"    retrying in {wait}s…"),
                    on_progress(i, len(prompts), f"Retry {attempt+1}/{MAX_RETRIES}…")
                    if on_progress else None,
                ),
                recreate_client=lambda: client_ref.__setitem__(
                    0, genai.Client(http_options={"timeout": API_TIMEOUT_MS})
                ),
            )
            paths.append(image_path)
            print(f"  [{i+1}/{len(prompts)}] {image_path.name} — "
                  f"{prompt[:70]}{'…' if len(prompt) > 70 else ''}")
        except Exception as e:
            failed_count += 1
            log.error("[%d/%d] FAILED after all retries: %s", i+1, len(prompts), e)
            print(f"  [{i+1}/{len(prompts)}] FAILED: {e}")
            print(f"    Creating placeholder for {image_path.name}")
            _create_placeholder(image_path)
            paths.append(image_path)
            if on_progress:
                on_progress(i, len(prompts), "Failed — using placeholder")

    log.info("Image generation complete: %d/%d succeeded, %d placeholders",
             len(prompts) - failed_count, len(prompts), failed_count)
    if failed_count:
        print(f"\n  Warning: {failed_count}/{len(prompts)} images failed and got placeholders")

    return paths


def _create_placeholder(path: Path):
    """Create a minimal dark 1080x1920 PNG placeholder (no PIL dependency)."""
    import struct, zlib
    width, height = 1080, 1920
    raw_row = b"\x00" + b"\x1a\x1d\x2e" * width
    raw_data = raw_row * height
    compressed = zlib.compress(raw_data)

    def _chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(_chunk(b"IDAT", compressed))
        f.write(_chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def generate_visuals(
    topic: str,
    segments: list[dict],
    output_dir: Path,
    on_progress: callable = None,
    images_per_segment: int = 1,
    wiki_name: str = "",
) -> list[dict]:
    """Full image pipeline: prompts → images.

    Args:
        topic:               The video topic (used for prompt context).
        segments:            Segment list from tts.generate_tts() output.
        output_dir:          Folder to save images.
        on_progress:         Optional callback(current, total, text).
        images_per_segment:  How many images to generate per segment (1-3).
        wiki_name:           Wiki universe name for visual grounding.

    Returns:
        Enriched segment list — each dict gains ``prompt`` and ``image_path``.
        When images_per_segment > 1, each original segment is split into
        sub-segments with evenly divided duration.
    """
    images_per_segment = max(1, min(images_per_segment, 3))
    total_images = len(segments) * images_per_segment

    # Cache prompts to disk so they survive failures
    prompts_cache = Path(output_dir) / "prompts.json"
    if prompts_cache.exists():
        cached = json.loads(prompts_cache.read_text(encoding="utf-8"))
        if (cached.get("topic") == topic
                and cached.get("images_per_segment") == images_per_segment
                and cached.get("num_segments") == len(segments)
                and cached.get("wiki_name", "") == wiki_name):
            prompts = cached["prompts"]
            print(f"Loaded {len(prompts)} cached image prompts")
        else:
            prompts = None
    else:
        prompts = None

    if prompts is None:
        print(f"Generating {total_images} image prompts ({images_per_segment} per segment)…")
        prompts = generate_prompts(topic, segments, images_per_segment=images_per_segment, wiki_name=wiki_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        prompts_cache.write_text(json.dumps({
            "topic": topic,
            "images_per_segment": images_per_segment,
            "num_segments": len(segments),
            "wiki_name": wiki_name,
            "prompts": prompts,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}] {prompt[:80]}{'…' if len(prompt) > 80 else ''}")

    print(f"\nGenerating {len(prompts)} images…")
    image_paths = generate_images(prompts, output_dir, on_progress=on_progress, wiki_name=wiki_name)

    results = []
    for seg_idx, seg in enumerate(segments):
        seg_prompts = prompts[seg_idx * images_per_segment:(seg_idx + 1) * images_per_segment]
        seg_images = image_paths[seg_idx * images_per_segment:(seg_idx + 1) * images_per_segment]

        if images_per_segment == 1:
            results.append({**seg, "prompt": seg_prompts[0], "image_path": seg_images[0]})
        else:
            sub_duration = round(seg["duration_seconds"] / images_per_segment, 3)
            for j, (prompt, img_path) in enumerate(zip(seg_prompts, seg_images)):
                sub_start = round(seg["start_seconds"] + j * sub_duration, 3)
                results.append({
                    **seg,
                    "index": len(results),
                    "prompt": prompt,
                    "image_path": img_path,
                    "start_seconds": sub_start,
                    "duration_seconds": sub_duration,
                    "_parent_segment": seg_idx,
                })

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    base = Path(__file__).parent

    script_text = (base / "script.txt").read_text(encoding="utf-8").strip()
    topic = script_text.splitlines()[0][:80]

    if len(sys.argv) > 1:
        segments = json.loads(Path(sys.argv[1]).read_text())
    else:
        metadata_path = base / "output" / "audio" / "segments.json"
        if metadata_path.exists():
            data = json.loads(metadata_path.read_text())
            segments = data["segments"]
        else:
            from tts import split_script
            segments = [{"index": i, "text": t} for i, t in enumerate(split_script(script_text))]

    output_dir = base / "output" / "images"
    results = generate_visuals(topic=topic, segments=segments, output_dir=output_dir)

    print(f"\nDone — {len(results)} images saved to {output_dir}/")
    for r in results:
        print(f"  [{r['index']}] {r['image_path'].name}")
