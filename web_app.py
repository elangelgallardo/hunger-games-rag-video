"""Flask web app — polished step-by-step video pipeline.

Run:
    python web_app.py
    → http://localhost:5000
"""

import json
import os
import queue
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

# -- Pipeline imports (lazy to avoid slow startup) ---------------------------
BASE = Path(__file__).parent
OUTPUT = BASE / "output"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Shared progress state
# ---------------------------------------------------------------------------
_progress_q: queue.Queue = queue.Queue()
_state = {
    "step": 0,          # 0=idle 1=script 2=audio 3=images 4=subs 5=stitch
    "busy": False,
    "error": None,
    "script": "",
    "topic": "",
    "segments": [],      # from tts
    "images": [],        # relative paths
    "total_duration": 0,
}


def _push(event: str, data: dict | str = ""):
    """Push a server-sent event."""
    _progress_q.put((event, data))


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------
@app.route("/api/events")
def events():
    def stream():
        while True:
            try:
                event, data = _progress_q.get(timeout=30)
                payload = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
                yield f"event: {event}\ndata: {payload}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Serve output files (audio, images, video)
# ---------------------------------------------------------------------------
@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(str(OUTPUT), filename)


# ---------------------------------------------------------------------------
# Step 1 — Generate script via RAG
# ---------------------------------------------------------------------------
@app.route("/api/generate-script", methods=["POST"])
def api_generate_script():
    if _state["busy"]:
        return jsonify(error="Pipeline is busy"), 409

    topic = request.json.get("topic", "").strip()
    if not topic:
        return jsonify(error="Topic is required"), 400

    _state["busy"] = True
    _state["error"] = None
    _state["topic"] = topic

    def work():
        try:
            _push("status", {"step": 1, "message": "Loading search index…"})
            from rag_query import load_stores, get_ranker, hybrid_search, rerank, generate_answer

            collection, ids, docs, bm25 = load_stores()
            get_ranker()

            _push("status", {"step": 1, "message": "Searching…"})
            candidates = hybrid_search(topic, collection, ids, docs, bm25)

            _push("status", {"step": 1, "message": "Reranking…"})
            top_passages = rerank(topic, candidates)

            _push("status", {"step": 1, "message": "Generating script with Gemini…"})
            script = generate_answer(topic, top_passages)

            _state["script"] = script
            _state["step"] = 1
            _push("script_done", {"script": script})
        except Exception as e:
            _state["error"] = str(e)
            _push("error", {"message": str(e)})
        finally:
            _state["busy"] = False

    threading.Thread(target=work, daemon=True).start()
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# Step 1b — Save edited script
# ---------------------------------------------------------------------------
@app.route("/api/save-script", methods=["POST"])
def api_save_script():
    script = request.json.get("script", "").strip()
    if not script:
        return jsonify(error="Script is empty"), 400
    _state["script"] = script
    # Also persist to disk for the CLI tools
    (BASE / "script.txt").write_text(script, encoding="utf-8")
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# Step 2 — Generate audio (TTS)
# ---------------------------------------------------------------------------
@app.route("/api/generate-audio", methods=["POST"])
def api_generate_audio():
    if _state["busy"]:
        return jsonify(error="Pipeline is busy"), 409
    if not _state["script"]:
        return jsonify(error="Generate a script first"), 400

    _state["busy"] = True
    _state["error"] = None

    def work():
        try:
            from tts import generate_tts

            audio_dir = OUTPUT / "audio"

            def on_progress(i, total, text):
                _push("status", {
                    "step": 2,
                    "message": f"TTS segment {i+1}/{total}",
                    "progress": round((i / total) * 100),
                })

            _push("status", {"step": 2, "message": "Starting TTS…", "progress": 0})
            result = generate_tts(
                script=_state["script"],
                output_dir=audio_dir,
                on_progress=on_progress,
            )

            _state["segments"] = result["segments"]
            _state["total_duration"] = result["total_duration"]
            _state["step"] = 2
            _push("audio_done", {
                "segments": result["segments"],
                "total_duration": result["total_duration"],
            })
        except Exception as e:
            _state["error"] = str(e)
            _push("error", {"message": str(e)})
        finally:
            _state["busy"] = False

    threading.Thread(target=work, daemon=True).start()
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# Step 3 — Generate images
# ---------------------------------------------------------------------------
@app.route("/api/generate-images", methods=["POST"])
def api_generate_images():
    if _state["busy"]:
        return jsonify(error="Pipeline is busy"), 409
    if not _state["segments"]:
        return jsonify(error="Generate audio first"), 400

    _state["busy"] = True
    _state["error"] = None

    def work():
        try:
            from image_gen import generate_visuals

            images_dir = OUTPUT / "images"

            def on_progress(i, total, text):
                _push("status", {
                    "step": 3,
                    "message": f"Image {i+1}/{total}",
                    "progress": round((i / total) * 100),
                })

            _push("status", {"step": 3, "message": "Generating image prompts…", "progress": 0})
            results = generate_visuals(
                topic=_state["topic"],
                segments=_state["segments"],
                output_dir=images_dir,
                on_progress=on_progress,
            )

            image_paths = [f"images/{Path(r['image_path']).name}" for r in results]
            _state["images"] = image_paths
            _state["step"] = 3
            _push("images_done", {"images": image_paths})
        except Exception as e:
            _state["error"] = str(e)
            _push("error", {"message": str(e)})
        finally:
            _state["busy"] = False

    threading.Thread(target=work, daemon=True).start()
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# Step 4 — Stitch video (subtitles + encode + concat)
# ---------------------------------------------------------------------------
@app.route("/api/stitch-video", methods=["POST"])
def api_stitch_video():
    if _state["busy"]:
        return jsonify(error="Pipeline is busy"), 409
    if not _state["segments"] or not _state["images"]:
        return jsonify(error="Generate audio and images first"), 400

    encoder = request.json.get("encoder", "auto")
    _state["busy"] = True
    _state["error"] = None

    def work():
        try:
            from video_stitch import stitch_video

            _push("status", {"step": 4, "message": "Generating subtitles & encoding…", "progress": 0})

            def on_progress(i, total, text):
                _push("status", {
                    "step": 4,
                    "message": f"Encoding segment {i+1}/{total}",
                    "progress": round((i / total) * 100),
                })

            video_path = OUTPUT / "video" / "final.mp4"
            stitch_video(
                segments=_state["segments"],
                images_dir=OUTPUT / "images",
                output_path=video_path,
                encoder=encoder,
                on_progress=on_progress,
            )

            _state["step"] = 4
            _push("video_done", {"video": "video/final.mp4"})
        except Exception as e:
            _state["error"] = str(e)
            _push("error", {"message": str(e)})
        finally:
            _state["busy"] = False

    threading.Thread(target=work, daemon=True).start()
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# State endpoint (for page reload recovery)
# ---------------------------------------------------------------------------
@app.route("/api/state")
def api_state():
    return jsonify(
        step=_state["step"],
        busy=_state["busy"],
        error=_state["error"],
        topic=_state["topic"],
        script=_state["script"],
        segments=_state["segments"],
        images=_state["images"],
        total_duration=_state["total_duration"],
        has_video=(OUTPUT / "video" / "final.mp4").exists(),
    )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    print(f"\n  Video Pipeline → http://localhost:5000\n")
    app.run(debug=False, port=5000, threaded=True)
