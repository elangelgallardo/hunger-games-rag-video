"""Video assembly — stitches images + audio into a short-form video.

Uses FFmpeg directly (no moviepy frame loop) for maximum speed:
  - zoompan filter  → Ken Burns effect in native C
  - h264_nvenc/amf/qsv → GPU-accelerated encoding
  - concat demuxer  → lossless segment join (stream copy)

Pipeline:
  1. Load segments.json (from tts.py)
  2. Per segment: FFmpeg  image + audio → temp MP4  (Ken Burns + fade + GPU encode)
  3. FFmpeg concat demuxer → final.mp4  (stream copy, no re-encode)

Usage:
    python video_stitch.py                     # auto-detect hardware encoder
    python video_stitch.py --encoder nvenc     # NVIDIA
    python video_stitch.py --encoder amf       # AMD
    python video_stitch.py --encoder qsv       # Intel
    python video_stitch.py --encoder cpu       # software fallback
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from subtitles import generate_subtitles

OUTPUT_WIDTH  = 1080
OUTPUT_HEIGHT = 1920
FPS           = 30
FADE          = 0.4   # fade-in / fade-out duration per segment (seconds)

EFFECTS = ["zoom_in", "zoom_out", "pan_right", "pan_left"]

ENCODERS = {
    "nvenc": ["-c:v", "h264_nvenc", "-preset", "p4",       "-cq",      "23"],
    "amf":   ["-c:v", "h264_amf",   "-quality", "balanced"],
    "qsv":   ["-c:v", "h264_qsv",   "-preset",  "medium"],
    "cpu":   ["-c:v", "libx264",    "-preset",   "fast",   "-crf",     "23"],
}


# ---------------------------------------------------------------------------
# Hardware encoder auto-detection
# ---------------------------------------------------------------------------

def detect_encoder() -> str:
    for name, codec in [("nvenc", "h264_nvenc"), ("amf", "h264_amf"), ("qsv", "h264_qsv")]:
        try:
            r = subprocess.run(
                ["ffmpeg", "-f", "lavfi", "-i", "nullsrc=s=64x64",
                 "-frames:v", "1", "-c:v", codec, "-f", "null", "-"],
                capture_output=True, timeout=5,
            )
            if r.returncode == 0:
                print(f"Hardware encoder: {codec}")
                return name
        except Exception:
            continue
    print("No hardware encoder found — using libx264 (CPU)")
    return "cpu"


# ---------------------------------------------------------------------------
# Ken Burns filter — crop+scale driven by `t` (seconds) for smooth motion
# ---------------------------------------------------------------------------

def _ken_burns_filter(effect: str, duration: float) -> str:
    """Build an FFmpeg filter string for a smooth Ken Burns effect.

    Rules:
      - crop w/h are STATIC (evaluated once at init) — cannot use `t`.
      - crop x/y and scale w/h with eval=frame DO support `t`.

    Zoom effects: normalize to output size, then use scale:eval=frame to
    grow/shrink the image per-frame, then static crop centers it.

    Pan effects: pre-scale to 1.1×, then crop with a time-varying x.
    """
    d = max(duration, 0.001)
    W, H = OUTPUT_WIDTH, OUTPUT_HEIGHT
    # Even-dimension 1.1× size for panning
    PW = (int(W * 1.1) // 2) * 2   # 1188
    PH = (int(H * 1.1) // 2) * 2   # 2112
    fade_out = max(0.0, duration - FADE)

    if effect == "zoom_in":
        # Image grows over time → same fixed crop shows less → zoom-in feel
        # Clamp scale to max 1.15× so late frames don't over-zoom
        filters = [
            f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
            f"scale=w='iw*min(1+0.15*t/{d:.4f},1.15)':h='ih*min(1+0.15*t/{d:.4f},1.15)':eval=frame",
            f"crop={W}:{H}:x='(in_w-{W})/2':y='(in_h-{H})/2'",
        ]
    elif effect == "zoom_out":
        # Image shrinks over time → zoom-out feel
        # Clamp scale to min 1.0× so the image never goes smaller than output
        filters = [
            f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
            f"scale=w='iw*max(1.15-0.15*t/{d:.4f},1.0)':h='ih*max(1.15-0.15*t/{d:.4f},1.0)':eval=frame",
            f"crop={W}:{H}:x='(in_w-{W})/2':y='(in_h-{H})/2'",
        ]
    elif effect == "pan_right":
        # Static scale to 1.1×, crop x moves left→right — clamp to valid range
        filters = [
            f"scale={PW}:{PH}:force_original_aspect_ratio=increase,crop={PW}:{PH}",
            f"crop={W}:{H}:x='min((in_w-{W})*t/{d:.4f},in_w-{W})':y='(in_h-{H})/2'",
        ]
    else:  # pan_left
        filters = [
            f"scale={PW}:{PH}:force_original_aspect_ratio=increase,crop={PW}:{PH}",
            f"crop={W}:{H}:x='max((in_w-{W})*(1-t/{d:.4f}),0)':y='(in_h-{H})/2'",
        ]

    filters += [
        f"fade=t=in:st=0:d={FADE}",
        f"fade=t=out:st={fade_out:.3f}:d={FADE}",
        "format=yuv420p",
    ]
    return ",".join(filters)


# ---------------------------------------------------------------------------
# Path escaping for FFmpeg filter expressions on Windows
# ---------------------------------------------------------------------------

def _ffmpeg_filter_path(p: Path) -> str:
    """Escape a file path for use inside an FFmpeg filter argument.

    FFmpeg filter syntax uses ':' as a separator, so the drive letter
    colon (C:) must be escaped.  Backslashes are also converted to '/'.
    """
    s = str(p.absolute()).replace("\\", "/")
    s = s.replace(":", "\\:")
    return s


# ---------------------------------------------------------------------------
# Per-segment encode
# ---------------------------------------------------------------------------

def _encode_segment(
    image_path: Path,
    audio_path: Path,
    output_path: Path,
    duration: float,
    effect: str,
    encoder: str,
    ass_path: Path | None = None,
) -> None:
    vf = _ken_burns_filter(effect, duration)

    # Burn subtitles if an ASS file is provided
    if ass_path and ass_path.exists():
        escaped = _ffmpeg_filter_path(ass_path)
        # Insert the ass filter just before format=yuv420p
        vf = vf.replace(",format=yuv420p", f",ass='{escaped}',format=yuv420p")

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", str(FPS), "-i", str(image_path),
        "-i", str(audio_path),
        "-filter:v", vf,
        *ENCODERS[encoder],
        "-c:a", "aac", "-b:a", "192k",
        "-t", str(duration),
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError(f"FFmpeg failed encoding {output_path.name}")


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------

def _concat_segments(segment_paths: list[Path], output_path: Path) -> None:
    list_file = output_path.parent / "concat_list.txt"
    list_file.write_text(
        "\n".join(f"file '{p.absolute().as_posix()}'" for p in segment_paths),
        encoding="utf-8",
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)

    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("FFmpeg concat failed")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stitch_video(
    segments: list[dict],
    images_dir: Path,
    output_path: Path,
    encoder: str = "auto",
    on_progress=None,
) -> Path:
    if encoder == "auto":
        encoder = detect_encoder()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / "_segments_tmp"
    temp_dir.mkdir(exist_ok=True)

    # --- Generate word-level subtitles ---
    subs_dir = output_path.parent.parent / "subtitles"
    print("Generating word-level subtitles…")
    ass_paths = generate_subtitles(segments, subs_dir)

    # --- Encode each segment ---
    segment_paths = []

    for i, seg in enumerate(segments):
        if on_progress:
            on_progress(i, len(segments), seg["text"])

        image_path = images_dir / f"image_{i:03d}.png"
        audio_path = Path(seg["audio_path"])
        duration   = seg["duration_seconds"]
        effect     = EFFECTS[i % len(EFFECTS)]

        print(f"  [{i+1}/{len(segments)}] {effect.replace('_',' ')}  "
              f"{duration:.1f}s — {seg['text'][:55]}…")

        seg_out = temp_dir / f"seg_{i:03d}.mp4"
        _encode_segment(
            image_path, audio_path, seg_out, duration, effect, encoder,
            ass_path=ass_paths[i],
        )
        segment_paths.append(seg_out)

    print(f"\nJoining {len(segment_paths)} segments (stream copy)…")
    _concat_segments(segment_paths, output_path)

    for p in segment_paths:
        p.unlink(missing_ok=True)
    temp_dir.rmdir()

    print(f"Done → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Metadata recovery (used when segments.json is missing)
# ---------------------------------------------------------------------------

def rebuild_segments_json(audio_dir: Path, script_path: Path) -> Path:
    from tts import split_script, wav_duration

    wav_files = sorted(audio_dir.glob("segment_*.wav"))
    if not wav_files:
        print(f"No segment_*.wav files in {audio_dir}")
        sys.exit(1)

    segments_text = split_script(script_path.read_text(encoding="utf-8").strip())

    if len(segments_text) != len(wav_files):
        print(f"Warning: {len(wav_files)} WAVs / {len(segments_text)} script segments — aligning to WAV count")
        segments_text = (segments_text + [""] * len(wav_files))[: len(wav_files)]

    segments, cursor = [], 0.0
    for i, (wav, text) in enumerate(zip(wav_files, segments_text)):
        dur = wav_duration(wav)
        segments.append({
            "index": i, "text": text,
            "audio_path": str(wav),
            "start_seconds": round(cursor, 3),
            "duration_seconds": round(dur, 3),
        })
        cursor += dur
        print(f"  segment_{i:03d}.wav  {dur:.1f}s")

    result = {"segments": segments, "total_duration": round(cursor, 3)}
    out = audio_dir / "segments.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved → {out}  (total {cursor:.1f}s)\n")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch images + audio into a short-form video")
    parser.add_argument(
        "--encoder", choices=["auto", "nvenc", "amf", "qsv", "cpu"], default="auto",
        help="Video encoder (default: auto-detect hardware)",
    )
    args = parser.parse_args()

    base          = Path(__file__).parent
    metadata_path = base / "output" / "audio" / "segments.json"

    if not metadata_path.exists():
        print("segments.json not found — rebuilding from existing WAV files…")
        metadata_path = rebuild_segments_json(
            audio_dir=base / "output" / "audio",
            script_path=base / "script.txt",
        )

    data     = json.loads(metadata_path.read_text(encoding="utf-8"))
    segments = data["segments"]

    stitch_video(
        segments=segments,
        images_dir=base / "output" / "images",
        output_path=base / "output" / "video" / "final.mp4",
        encoder=args.encoder,
    )
