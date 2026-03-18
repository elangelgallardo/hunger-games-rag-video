"""Generate word-level ASS subtitles from WAV audio using faster-whisper.

Pipeline:
  1. Transcribe each segment WAV with faster-whisper (word_timestamps=True)
  2. Write one ASS subtitle file per segment — one word per dialogue line

Usage:
    python subtitles.py                   # generate ASS files from segments.json
    python subtitles.py --preview         # print word timestamps, no files written
"""

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

from faster_whisper import WhisperModel

from pipeline_config import (
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE,
)


# ---------------------------------------------------------------------------
# ASS template — short-form video style (big, bold, centered)
# ---------------------------------------------------------------------------

ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Bebas Neue,120,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,2,0,1,4,3,5,10,10,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""


def _ass_ts(seconds: float) -> str:
    """Convert seconds to ASS timestamp  H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

_model = None


def _get_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model ({WHISPER_MODEL})…")
        _model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE,
                              compute_type=WHISPER_COMPUTE)
    return _model


def transcribe_words(wav_path: Path, language: str = None) -> list[dict]:
    """Return [{word, start, end}, …] for every word in a WAV file."""
    model = _get_model()
    segments, _ = model.transcribe(
        str(wav_path),
        word_timestamps=True,
        language=language,
    )

    words = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            text = w.word.strip()
            if text:
                words.append({"word": text, "start": w.start, "end": w.end})
    return words


# ---------------------------------------------------------------------------
# Spell-correction: align transcribed words against the original script
# ---------------------------------------------------------------------------

def _normalize(word: str) -> str:
    """Lowercase, strip punctuation — used only for comparison."""
    return re.sub(r"[^\w']", "", word.lower())


def correct_words(transcribed: list[dict], original_text: str) -> list[dict]:
    """Replace transcribed spellings with the original script's words.

    Uses SequenceMatcher to align the two word lists, then for every
    matched or replaced pair, keeps the *timing* from Whisper but the
    *text* from the script.  Extra words in either list are handled
    gracefully (kept / dropped as appropriate).
    """
    orig_words = original_text.split()

    trans_norm = [_normalize(w["word"]) for w in transcribed]
    orig_norm = [_normalize(w) for w in orig_words]

    matcher = SequenceMatcher(None, trans_norm, orig_norm, autojunk=False)

    corrected = [dict(w) for w in transcribed]  # deep-ish copy

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Perfect match — swap in original spelling (preserves caps & punctuation)
            for ti, oi in zip(range(i1, i2), range(j1, j2)):
                corrected[ti]["word"] = orig_words[oi]
        elif tag == "replace":
            # Whisper mis-heard — pair them 1:1 as far as possible
            pairs = min(i2 - i1, j2 - j1)
            for k in range(pairs):
                corrected[i1 + k]["word"] = orig_words[j1 + k]
        # "insert" (extra words in script) → nothing to do
        # "delete" (extra words in transcription) → keep Whisper's word

    return corrected


# ---------------------------------------------------------------------------
# ASS generation
# ---------------------------------------------------------------------------

def generate_ass(words: list[dict], output_path: Path) -> Path:
    """Write an ASS file with one Dialogue line per word (UPPER CASE)."""
    lines = [ASS_HEADER]

    for w in words:
        start = _ass_ts(w["start"])
        end = _ass_ts(w["end"])
        lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{w['word'].upper()}"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_subtitles(
    segments: list[dict],
    output_dir: Path,
    on_progress=None,
    language: str = None,
) -> list[Path]:
    """Transcribe all segment WAVs and write per-segment ASS files.

    Args:
        language: Whisper language code (e.g. "en", "es"). None = auto-detect.

    Returns a list of ASS file paths (same order / count as segments).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _get_model()  # warm up once

    ass_paths: list[Path] = []

    for i, seg in enumerate(segments):
        if on_progress:
            on_progress(i, len(segments), seg["text"])

        wav_path = Path(seg["audio_path"])
        ass_path = output_dir / f"segment_{i:03d}.ass"

        words = transcribe_words(wav_path, language=language)
        words = correct_words(words, seg["text"])
        generate_ass(words, ass_path)

        print(f"  [{i+1}/{len(segments)}] {len(words)} words — {seg['text'][:55]}…")
        ass_paths.append(ass_path)

    return ass_paths


def generate_subtitles_from_full_wav(
    full_wav: Path,
    output_dir: Path,
    num_segments: int,
    segment_duration: float,
    full_text: str,
    language: str = None,
) -> list[Path]:
    """Transcribe the full audio once and slice word timestamps into per-segment ASS files.

    Args:
        full_wav:         Path to the continuous audio file (full_audio.wav).
        output_dir:       Directory to write segment_NNN.ass files.
        num_segments:     Number of equal-duration image segments.
        segment_duration: Duration of each segment in seconds.
        full_text:        Complete script text used for spell-correction alignment.
        language:         Whisper language code. None = auto-detect.

    Returns a list of ASS file paths (one per segment).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _get_model()

    print(f"  Transcribing full audio ({full_wav.name})…")
    words = transcribe_words(full_wav, language=language)
    words = correct_words(words, full_text)
    print(f"  {len(words)} words transcribed, slicing into {num_segments} segments")

    ass_paths: list[Path] = []
    for i in range(num_segments):
        seg_start = i * segment_duration
        seg_end = seg_start + segment_duration
        ass_path = output_dir / f"segment_{i:03d}.ass"

        seg_words = [
            {
                "word": w["word"],
                "start": max(0.0, w["start"] - seg_start),
                "end": min(segment_duration, w["end"] - seg_start),
            }
            for w in words
            if w["end"] > seg_start and w["start"] < seg_end
        ]

        generate_ass(seg_words, ass_path)
        print(f"  [{i+1}/{num_segments}] {len(seg_words)} words (t={seg_start:.1f}s–{seg_end:.1f}s)")
        ass_paths.append(ass_path)

    return ass_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate word-level ASS subtitles")
    parser.add_argument("--preview", action="store_true",
                        help="Print word timestamps only, don't write files")
    args = parser.parse_args()

    base = Path(__file__).parent
    metadata_path = base / "output" / "audio" / "segments.json"

    if not metadata_path.exists():
        print(f"segments.json not found at {metadata_path}")
        sys.exit(1)

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    segments = data["segments"]

    if args.preview:
        for seg in segments:
            wav = Path(seg["audio_path"])
            print(f"\n--- {wav.name} ---")
            raw = transcribe_words(wav)
            fixed = correct_words(raw, seg["text"])
            for r, f in zip(raw, fixed):
                changed = " [fixed]" if r["word"] != f["word"] else ""
                print(f"  [{f['start']:6.2f} -> {f['end']:6.2f}]  {f['word']:<20s}"
                      f"  (was: {r['word']}){changed}" if changed else
                      f"  [{f['start']:6.2f} -> {f['end']:6.2f}]  {f['word']}")
    else:
        subs_dir = base / "output" / "subtitles"
        paths = generate_subtitles(segments, subs_dir)
        print(f"\n{len(paths)} subtitle files -> {subs_dir}/")
