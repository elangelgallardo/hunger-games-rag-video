"""Google Gemini TTS — generate one WAV per script segment.

The script is split at paragraph boundaries (up to MAX_WORDS_PER_SEGMENT words
each) and one API call is made per segment. This keeps the number of calls
small (typically 3-6 for a 1-3 min script) while staying within the TTS
payload limit. Each WAV has a measured duration, giving exact image timing.

Usage:
    python tts.py <output_dir> "<script text>"
"""

import re
import sys
import wave
from pathlib import Path

from google import genai
from google.genai import types

TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_VOICE = "Kore"
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM
MAX_WORDS_PER_SEGMENT = 100  # large paragraphs → few API calls


# ---------------------------------------------------------------------------
# Script splitting
# ---------------------------------------------------------------------------

def split_script(script: str, max_words: int = MAX_WORDS_PER_SEGMENT) -> list[str]:
    """Split a script into paragraph-level segments.

    Splits on blank lines first; further splits long paragraphs on sentence
    boundaries only if they exceed max_words.
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
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "r") as wf:
        return wf.getnframes() / wf.getframerate()


# ---------------------------------------------------------------------------
# Main TTS function
# ---------------------------------------------------------------------------

def generate_tts(
    script: str,
    output_dir: Path,
    voice: str = DEFAULT_VOICE,
    on_progress: callable = None,
) -> dict:
    """Generate one WAV per segment and return timing metadata.

    Returns:
        {
          "segments": [
              {
                "index":            int,
                "text":             str,
                "audio_path":       Path,
                "start_seconds":    float,
                "duration_seconds": float,
              },
              ...
          ],
          "total_duration": float,
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client()
    segments_text = split_script(script)
    results: list[dict] = []
    cursor = 0.0

    for i, text in enumerate(segments_text):
        if on_progress:
            on_progress(i, len(segments_text), text)

        response = client.models.generate_content(
            model=TTS_MODEL,
            contents=text,
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

        pcm = response.candidates[0].content.parts[0].inline_data.data
        audio_path = output_dir / f"segment_{i:03d}.wav"
        _write_wav(audio_path, pcm)
        duration = wav_duration(audio_path)

        results.append(
            {
                "index": i,
                "text": text,
                "audio_path": audio_path,
                "start_seconds": round(cursor, 3),
                "duration_seconds": round(duration, 3),
            }
        )
        cursor += duration

        print(f"  [{i+1}/{len(segments_text)}] {duration:.1f}s — {text[:70]}{'…' if len(text) > 70 else ''}")

    total = round(cursor, 3)
    print(f"\n{len(results)} segments, {total:.1f}s total → {output_dir}/")

    # Serialize paths to strings for JSON
    serializable = [
        {**r, "audio_path": str(r["audio_path"])} for r in results
    ]
    result = {"segments": serializable, "total_duration": total}

    import json
    metadata_path = output_dir / "segments.json"
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Metadata saved → {metadata_path}")

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
