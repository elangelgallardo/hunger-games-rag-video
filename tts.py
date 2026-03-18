"""Google Gemini TTS — single-call audio generation.

The full script is sent in ONE API call to maintain consistent language and tone
across all segments.  The resulting audio is then split into per-segment WAVs
using silence-aware splitting (word-count proportional estimates snapped to the
nearest quiet region) so the downstream video pipeline stays unchanged.

Usage:
    python tts.py <output_dir>
"""

import json
import re
import struct
import sys
import time
import wave
from pathlib import Path

from google import genai
from google.genai import types

from api_retry import retry_api_call
from pipeline_config import (
    API_TIMEOUT_MS,
    TTS_MODEL,
    TTS_VOICE as DEFAULT_VOICE,
    TTS_SAMPLE_RATE as SAMPLE_RATE,
    TTS_CHANNELS as CHANNELS,
    TTS_SAMPLE_WIDTH as SAMPLE_WIDTH,
    TTS_MAX_WORDS_PER_SEGMENT as MAX_WORDS_PER_SEGMENT,
)


# ---------------------------------------------------------------------------
# Script splitting
# ---------------------------------------------------------------------------

def split_script(script: str, max_words: int = MAX_WORDS_PER_SEGMENT) -> list[str]:
    """Split a script into paragraph-level segments.

    Splits on blank lines first; further splits long paragraphs on sentence
    boundaries only if they exceed *max_words*.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", script) if p.strip()]

    segments: list[str] = []
    for para in paragraphs:
        if len(para.split()) <= max_words:
            segments.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            bucket: list[str] = []
            bucket_words = 0
            for sentence in sentences:
                n = len(sentence.split())
                if bucket_words + n > max_words and bucket:
                    segments.append(" ".join(bucket))
                    bucket = [sentence]
                    bucket_words = n
                else:
                    bucket.append(sentence)
                    bucket_words += n
            if bucket:
                segments.append(" ".join(bucket))

    return segments


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------

def _write_wav(path: Path, pcm: bytes) -> None:
    """Write raw 16-bit PCM data to a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


def wav_duration(path: Path) -> float:
    """Return the duration in seconds of a WAV file."""
    with wave.open(str(path), "r") as wf:
        return wf.getnframes() / wf.getframerate()


# ---------------------------------------------------------------------------
# Audio splitting — silence-aware split points
# ---------------------------------------------------------------------------

def _find_split_frames(pcm: bytes, segment_word_counts: list[int]) -> list[int]:
    """Find frame indices to split combined audio into segments.

    Uses word-count proportional estimates, snapped to nearby silence regions.
    Returns N-1 frame indices for N segments.
    """
    n = len(segment_word_counts)
    if n <= 1:
        return []

    total_words = sum(segment_word_counts)
    total_frames = len(pcm) // (SAMPLE_WIDTH * CHANNELS)

    # Proportional split points (frame indices)
    proportional = []
    cumulative = 0
    for wc in segment_word_counts[:-1]:
        cumulative += wc
        proportional.append(int(total_frames * cumulative / total_words))

    # Read all samples for energy analysis
    samples = struct.unpack(f"<{total_frames}h", pcm)

    # Snap each split to the quietest point within +/- 0.5 seconds
    window = int(SAMPLE_RATE * 0.03)   # 30ms energy window
    search = int(SAMPLE_RATE * 0.5)    # +/-0.5s search range
    step = window // 2                 # 15ms step

    snapped = []
    for frame in proportional:
        lo = max(0, frame - search)
        hi = min(total_frames - window, frame + search)
        best_frame = frame
        best_energy = float("inf")
        for f in range(lo, hi, step):
            energy = sum(s * s for s in samples[f : f + window])
            if energy < best_energy:
                best_energy = energy
                best_frame = f
        snapped.append(best_frame)

    # Ensure splits are monotonically increasing with minimum 100ms gap
    min_gap = int(SAMPLE_RATE * 0.1)
    result = []
    prev = 0
    for frame in snapped:
        frame = max(frame, prev + min_gap)
        frame = min(frame, total_frames - min_gap)
        result.append(frame)
        prev = frame

    return result


def _split_pcm(pcm: bytes, split_frames: list[int]) -> list[bytes]:
    """Split PCM data at the given frame indices."""
    bpf = SAMPLE_WIDTH * CHANNELS  # bytes per frame
    total = len(pcm)

    boundaries = [0] + [f * bpf for f in split_frames] + [total]
    return [pcm[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]


# ---------------------------------------------------------------------------
# Main TTS function
# ---------------------------------------------------------------------------

def generate_tts(
    script: str,
    output_dir: Path,
    voice: str = DEFAULT_VOICE,
    on_progress: callable = None,
    language: str = None,
) -> dict:
    """Generate TTS audio in a single API call, then split into segment WAVs.

    Sending the full script in one call prevents language drift and ensures
    consistent tone/pacing across all segments.

    Returns:
        {
          "segments": [
              {"index", "text", "audio_path", "start_seconds", "duration_seconds"},
              ...
          ],
          "total_duration": float,
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client_ref = [genai.Client(http_options={"timeout": API_TIMEOUT_MS})]
    segments_text = split_script(script)

    if on_progress:
        on_progress(0, len(segments_text), "Generating audio…")

    # Build full text — single call for consistent tone & language
    full_text = "\n\n".join(segments_text)

    print(f"  Generating TTS for {len(segments_text)} segments in one call "
          f"({len(full_text.split())} words)…")

    def make_tts_call():
        return client_ref[0].models.generate_content(
            model=TTS_MODEL,
            contents=full_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice,
                        )
                    )
                ),
            ),
        )

    response = retry_api_call(
        make_tts_call,
        description=f"TTS ({len(full_text.split())} words)",
        on_retry=lambda attempt, wait, err: print(
            f"  TTS error (attempt {attempt+1}): {err}\n    Retrying in {wait}s…"
        ),
        recreate_client=lambda: client_ref.__setitem__(
            0, genai.Client(http_options={"timeout": API_TIMEOUT_MS})
        ),
    )

    pcm = response.candidates[0].content.parts[0].inline_data.data

    # Save combined audio for reference
    full_wav = output_dir / "full_audio.wav"
    _write_wav(full_wav, pcm)
    total_duration = len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
    print(f"  Full audio: {total_duration:.1f}s -> {full_wav}")

    # Split into per-segment WAVs using silence-aware splitting
    word_counts = [len(seg.split()) for seg in segments_text]
    split_frames = _find_split_frames(pcm, word_counts)
    chunks = _split_pcm(pcm, split_frames)

    results: list[dict] = []
    cursor = 0.0

    for i, (text, chunk) in enumerate(zip(segments_text, chunks)):
        audio_path = output_dir / f"segment_{i:03d}.wav"
        _write_wav(audio_path, chunk)
        duration = len(chunk) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)

        results.append({
            "index": i,
            "text": text,
            "audio_path": str(audio_path),
            "start_seconds": round(cursor, 3),
            "duration_seconds": round(duration, 3),
        })
        cursor += duration

        if on_progress:
            on_progress(i + 1, len(segments_text), text)

        print(f"  [{i+1}/{len(segments_text)}] {duration:.1f}s — "
              f"{text[:70]}{'…' if len(text) > 70 else ''}")

    total = round(cursor, 3)
    print(f"\n{len(results)} segments, {total:.1f}s total -> {output_dir}/")

    result = {"segments": results, "total_duration": total}

    metadata_path = output_dir / "segments.json"
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Metadata saved -> {metadata_path}")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    script_file = Path(__file__).parent / "script.txt"
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "output" / "audio"

    script_text = script_file.read_text(encoding="utf-8")
    print(f"Loaded script: {script_file} ({len(script_text.split())} words)\n")

    result = generate_tts(script=script_text, output_dir=output_dir)
    for seg in result["segments"]:
        print(f"  [{seg['index']}] {seg['start_seconds']:.1f}s + {seg['duration_seconds']:.1f}s — {seg['text'][:60]}")
