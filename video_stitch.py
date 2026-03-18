"""Video assembly — stitches images + audio into a short-form video.

Uses FFmpeg directly (no moviepy frame loop) for maximum speed:
  - zoompan filter  -- Ken Burns effect in native C
  - h264_nvenc/amf/qsv -- GPU-accelerated encoding
  - concat demuxer  -- lossless segment join (stream copy)

Pipeline:
  1. Load segments.json (from tts.py)
  2. Per segment: FFmpeg  image + audio -> temp MP4  (Ken Burns + fade + GPU encode)
  3. FFmpeg concat demuxer -> final.mp4  (stream copy, no re-encode)

Usage:
    python video_stitch.py                     # auto-detect hardware encoder
    python video_stitch.py --encoder nvenc     # NVIDIA
    python video_stitch.py --encoder amf       # AMD
    python video_stitch.py --encoder qsv       # Intel
    python video_stitch.py --encoder cpu       # software fallback
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from subtitles import generate_subtitles_from_full_wav
from pipeline_config import (
    VIDEO_WIDTH as OUTPUT_WIDTH,
    VIDEO_HEIGHT as OUTPUT_HEIGHT,
    VIDEO_FPS as FPS,
    VIDEO_FADE as FADE,
    VIDEO_EFFECTS as EFFECTS,
    VIDEO_PAN_FACTOR,
)

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

    Zoom effects use scale:eval=frame for per-frame interpolation, then
    a static centered crop.

    Pan effects use the zoompan filter which provides sub-pixel
    interpolation — the crop filter rounds to integer pixels and causes
    visible jitter on slow pans.
    """
    d = max(duration, 0.001)
    W, H = OUTPUT_WIDTH, OUTPUT_HEIGHT
    fade_out = max(0.0, duration - FADE)

    if effect == "zoom_in":
        filters = [
            f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
            f"scale=w='iw*min(1+0.15*t/{d:.4f},1.15)':h='ih*min(1+0.15*t/{d:.4f},1.15)':eval=frame",
            f"crop={W}:{H}:x='(in_w-{W})/2':y='(in_h-{H})/2'",
        ]
    elif effect == "zoom_out":
        filters = [
            f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
            f"scale=w='iw*max(1.15-0.15*t/{d:.4f},1.0)':h='ih*max(1.15-0.15*t/{d:.4f},1.0)':eval=frame",
            f"crop={W}:{H}:x='(in_w-{W})/2':y='(in_h-{H})/2'",
        ]
    elif effect in ("pan_right", "pan_left"):
        # Pre-scale then use zoompan for smooth sub-pixel panning.
        # zoompan with d=1 processes one input frame → one output frame,
        # using 'in' (input frame counter) for time progression.
        PAN_FACTOR = VIDEO_PAN_FACTOR
        PW = (int(W * PAN_FACTOR) // 2) * 2
        PH = (int(H * PAN_FACTOR) // 2) * 2
        total_frames = max(int(d * FPS), 1)
        z = PW / W  # zoom so visible window = W×H
        travel_x = PW - W
        center_y = (PH - H) // 2

        if effect == "pan_right":
            x_expr = f"min({travel_x}*in/{total_frames},{travel_x})"
        else:
            x_expr = f"max({travel_x}-{travel_x}*in/{total_frames},0)"

        filters = [
            f"scale={PW}:{PH}:force_original_aspect_ratio=increase,crop={PW}:{PH}",
            f"zoompan=z='{z:.4f}':x='{x_expr}':y='{center_y}':d=1:s={W}x{H}:fps={FPS}",
        ]
    else:
        filters = [
            f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
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
# Per-segment encode (video-only — no audio track)
# ---------------------------------------------------------------------------

def _encode_segment(
    image_path: Path,
    output_path: Path,
    duration: float,
    effect: str,
    encoder: str,
    ass_path: Path | None = None,
) -> None:
    """Encode one image as a video-only MP4 (no audio).

    Audio is added in a single pass at the end by _concat_video_mux_audio,
    which avoids per-segment AAC encoder-delay artifacts at transitions.
    """
    vf = _ken_burns_filter(effect, duration)

    if ass_path and ass_path.exists():
        escaped = _ffmpeg_filter_path(ass_path)
        vf = vf.replace(",format=yuv420p", f",ass='{escaped}',format=yuv420p")

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", str(FPS), "-i", str(image_path),
        "-filter:v", vf,
        *ENCODERS[encoder],
        "-an",            # no audio track
        "-t", str(duration),
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError(f"FFmpeg failed encoding {output_path.name}")


# ---------------------------------------------------------------------------
# Concat video segments + mux full audio in one pass
# ---------------------------------------------------------------------------

def _concat_video_mux_audio(
    segment_paths: list[Path], audio_path: Path, output_path: Path
) -> None:
    """Concat video-only segments and mux the continuous audio track.

    Encoding AAC once across the full audio avoids the encoder-delay
    glitches that occur when independently-encoded AAC segments are
    stream-copied together.
    """
    list_file = output_path.parent / "concat_list.txt"
    list_file.write_text(
        "\n".join(f"file '{p.absolute().as_posix()}'" for p in segment_paths),
        encoding="utf-8",
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)

    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("FFmpeg concat+mux failed")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stitch_video(
    segments: list[dict],
    images_dir: Path,
    output_path: Path,
    encoder: str = "auto",
    on_progress=None,
    visual_segments: list[dict] = None,  # kept for API compatibility, not used
    language: str = None,
) -> Path:
    if encoder == "auto":
        encoder = detect_encoder()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Clean up any leftover temp dir from a previous failed run
    temp_dir = output_path.parent / "_segments_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(exist_ok=True)
    # Remove old output so FFmpeg doesn't complain
    if output_path.exists():
        output_path.unlink()

    # --- Locate the continuous audio file produced by tts.py ---
    full_audio = Path(segments[0]["audio_path"]).parent / "full_audio.wav"
    if not full_audio.exists():
        raise FileNotFoundError(f"full_audio.wav not found at {full_audio}")

    from tts import wav_duration
    total_audio_duration = wav_duration(full_audio)

    # --- Count available images to compute uniform duration per image ---
    image_files = sorted(images_dir.glob("image_*.png"))
    num_images = len(image_files)
    if num_images == 0:
        raise RuntimeError(f"No image_*.png files found in {images_dir}")

    image_duration = total_audio_duration / num_images
    print(f"Full audio: {total_audio_duration:.2f}s / {num_images} images = {image_duration:.3f}s per image")

    # --- Generate subtitles from the full audio (one transcription pass) ---
    subs_dir = output_path.parent.parent / "subtitles"
    whisper_lang = language.split("-")[0] if language else None
    full_text = " ".join(seg["text"] for seg in segments)
    print(f"Generating word-level subtitles from full audio (lang={whisper_lang or 'auto'})…")
    ass_paths = generate_subtitles_from_full_wav(
        full_wav=full_audio,
        output_dir=subs_dir,
        num_segments=num_images,
        segment_duration=image_duration,
        full_text=full_text,
        language=whisper_lang,
    )

    # --- Encode each image as video-only (no audio) ---
    segment_paths = []
    total = num_images

    for i, image_path in enumerate(image_files):
        if on_progress:
            on_progress(i, total, f"image {i+1}/{total}")

        effect = EFFECTS[i % len(EFFECTS)]
        ass_path = ass_paths[i] if i < len(ass_paths) else None

        print(f"  [{i+1}/{total}] {effect.replace('_', ' ')}  {image_duration:.2f}s")

        seg_out = temp_dir / f"seg_{i:03d}.mp4"
        _encode_segment(image_path, seg_out, image_duration, effect, encoder,
                        ass_path=ass_path)
        segment_paths.append(seg_out)

    print(f"\nConcatenating {len(segment_paths)} video segments and muxing audio…")
    _concat_video_mux_audio(segment_paths, full_audio, output_path)

    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Done -> {output_path}")
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
    print(f"Saved -> {out}  (total {cursor:.1f}s)\n")
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
