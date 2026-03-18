"""Flask web app — step-by-step video pipeline with multi-wiki support.

Orchestrates the full pipeline:
  1. Generate script (factual or theory mode) via RAG
  2. Generate audio (TTS)
  3. Generate images
  4. Stitch video (FFmpeg)
  5. Upload to YouTube (optional)

Run:
    python web_app.py
    → http://localhost:5000
"""

import json
import logging
import os
import queue
import shutil
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Configure logging — image_gen logs go to console + file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)

# Allow OAuth over HTTP for local development
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

from flask import Flask, Response, jsonify, render_template, request, send_from_directory, session

from wiki_registry import list_wikis, get_wiki_paths, get_wiki_meta, delete_wiki, WIKIS_DIR

BASE = Path(__file__).parent
OUTPUT = BASE / "output"
SESSION_FILE = BASE / "session.json"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "rag-video-pipeline-local")

# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------
_DEFAULT_STATE = {
    "step": 0,
    "busy": False,
    "error": None,
    "script": "",
    "topic": "",
    "language": "es-MX",
    "mode": "factual",
    "segments": [],
    "images": [],
    "visual_segments": [],
    "total_duration": 0,
    "active_wiki": None,    # slug of the currently selected wiki
}

_subscribers: list[queue.Queue] = []
_subscribers_lock = threading.Lock()
_state: dict = {}


def _load_session():
    """Load persisted session from disk, falling back to defaults."""
    global _state
    _state = dict(_DEFAULT_STATE)
    if SESSION_FILE.exists():
        try:
            saved = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
            _state.update(saved)
        except Exception:
            pass
    # Always reset transient fields
    _state["busy"] = False
    _state["error"] = None


def _save_session():
    """Persist current state to disk (skip transient fields)."""
    to_save = {k: v for k, v in _state.items() if k not in ("busy", "error")}
    SESSION_FILE.write_text(json.dumps(to_save, indent=2, default=str), encoding="utf-8")


_load_session()


def _push(event: str, data: dict | str = ""):
    with _subscribers_lock:
        for q in _subscribers:
            q.put((event, data))


def _clean_dir(path: Path):
    """Remove and recreate a directory to start fresh."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------
@app.route("/api/events")
def events():
    q: queue.Queue = queue.Queue()
    with _subscribers_lock:
        _subscribers.append(q)

    def stream():
        try:
            while True:
                try:
                    event, data = q.get(timeout=15)
                    payload = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
                    yield f"event: {event}\ndata: {payload}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _subscribers_lock:
                try:
                    _subscribers.remove(q)
                except ValueError:
                    pass

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Serve output files
# ---------------------------------------------------------------------------
@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(str(OUTPUT), filename)


# ---------------------------------------------------------------------------
# Wiki management
# ---------------------------------------------------------------------------
@app.route("/api/wikis")
def api_wikis():
    return jsonify(wikis=list_wikis(), active=_state.get("active_wiki"))


@app.route("/api/set-wiki", methods=["POST"])
def api_set_wiki():
    if _state["busy"]:
        return jsonify(error="Pipeline is busy"), 409
    slug = request.json.get("slug", "").strip()
    if not slug:
        return jsonify(error="Slug is required"), 400
    try:
        get_wiki_meta(slug)
    except FileNotFoundError:
        return jsonify(error=f"Wiki '{slug}' not found"), 404

    # Reset pipeline state when switching wikis
    for k, v in _DEFAULT_STATE.items():
        if k not in ("busy", "error"):
            _state[k] = v if not isinstance(v, list) else list(v)
    _state["active_wiki"] = slug
    _save_session()
    # Clean all output directories so stale files don't interfere
    for subdir in ("audio", "images", "video", "subtitles"):
        _clean_dir(OUTPUT / subdir)
    return jsonify(ok=True, slug=slug)


@app.route("/api/delete-wiki", methods=["POST"])
def api_delete_wiki():
    if _state["busy"]:
        return jsonify(error="Pipeline is busy"), 409
    slug = request.args.get("slug") or (request.json or {}).get("slug", "")
    slug = slug.strip()
    if not slug:
        return jsonify(error="Slug is required"), 400
    try:
        delete_wiki(slug)
    except FileNotFoundError:
        return jsonify(error=f"Wiki '{slug}' not found"), 404
    # If the deleted wiki was active, reset state
    if _state.get("active_wiki") == slug:
        for k, v in _DEFAULT_STATE.items():
            if k not in ("busy", "error"):
                _state[k] = v if not isinstance(v, list) else list(v)
        _save_session()
    return jsonify(ok=True)


@app.route("/api/reset-session", methods=["POST"])
def api_reset_session():
    if _state["busy"]:
        return jsonify(error="Pipeline is busy"), 409
    wiki = _state.get("active_wiki")
    for k, v in _DEFAULT_STATE.items():
        _state[k] = v if not isinstance(v, list) else list(v)
    _state["active_wiki"] = wiki  # keep the wiki selection
    _save_session()
    # Clean all output directories so stale files don't interfere
    for subdir in ("audio", "images", "video", "subtitles"):
        _clean_dir(OUTPUT / subdir)
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# Helper: resolve active wiki paths
# ---------------------------------------------------------------------------
def _active_wiki_paths():
    slug = _state.get("active_wiki")
    if not slug:
        return None, None, None
    paths = get_wiki_paths(slug)
    meta = get_wiki_meta(slug)
    return paths, meta, slug


# ---------------------------------------------------------------------------
# Step 1 — Generate script via RAG
# ---------------------------------------------------------------------------
@app.route("/api/generate-script", methods=["POST"])
def api_generate_script():
    if _state["busy"]:
        return jsonify(error="Pipeline is busy"), 409
    if not _state.get("active_wiki"):
        return jsonify(error="Select a wiki first"), 400

    topic = request.json.get("topic", "").strip()
    language = request.json.get("language", "es-MX")
    mode = request.json.get("mode", "factual")
    if not topic:
        return jsonify(error="Topic is required"), 400

    _state["busy"] = True
    _state["error"] = None
    _state["topic"] = topic
    _state["language"] = language
    _state["mode"] = mode

    paths, meta, slug = _active_wiki_paths()

    def work():
        try:
            _push("status", {"step": 1, "message": "Loading search index…"})
            from rag_query import (
                load_stores, get_ranker, hybrid_search, rerank,
                generate_answer, load_link_graph, graph_search, generate_theory,
            )

            collection, ids, docs, metadatas, bm25 = load_stores(
                chromadb_dir=paths["chromadb_store"],
                collection_name=meta["collection"],
            )
            get_ranker()

            if mode == "theory":
                _push("status", {"step": 1, "message": "Loading link graph…"})
                link_graph = load_link_graph(paths["base"])

                _push("status", {"step": 1, "message": "Graph-expanded search…"})
                expanded = graph_search(topic, collection, ids, docs, metadatas, bm25, link_graph)

                _push("status", {"step": 1, "message": f"Reranking {len(expanded)} chunks…"})
                # Cap at 50 before reranking to avoid OOM in FlashRank
                if len(expanded) > 50:
                    from rank_bm25 import BM25Okapi as _BM25
                    _tok = [c["document"].lower().split() for c in expanded]
                    _bm = _BM25(_tok)
                    _scores = _bm.get_scores(topic.lower().split())
                    ranked_idx = sorted(range(len(expanded)), key=lambda i: _scores[i], reverse=True)[:50]
                    expanded = [expanded[i] for i in ranked_idx]
                top_passages = rerank(topic, expanded, top_n=15)

                _push("status", {"step": 1, "message": "Generating theory with thinking model…"})
                script = generate_theory(topic, top_passages, wiki_name=meta["name"], language=language)
            else:
                _push("status", {"step": 1, "message": "Searching…"})
                candidates = hybrid_search(topic, collection, ids, docs, metadatas, bm25)

                _push("status", {"step": 1, "message": "Reranking…"})
                top_passages = rerank(topic, candidates)

                _push("status", {"step": 1, "message": "Generating script with Gemini…"})
                script = generate_answer(topic, top_passages, wiki_name=meta["name"], language=language)

            _state["script"] = script
            _state["step"] = 1
            _save_session()
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
    (BASE / "script.txt").write_text(script, encoding="utf-8")
    _save_session()
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
            _clean_dir(audio_dir)

            def on_progress(i, total, text):
                if i == 0:
                    _push("status", {"step": 2, "message": "Generating audio…", "progress": 10})
                else:
                    _push("status", {
                        "step": 2,
                        "message": f"Splitting segment {i}/{total}",
                        "progress": round(10 + (i / total) * 90),
                    })

            _push("status", {"step": 2, "message": "Starting TTS…", "progress": 0})
            result = generate_tts(
                script=_state["script"],
                output_dir=audio_dir,
                on_progress=on_progress,
                language=_state.get("language"),
            )

            _state["segments"] = result["segments"]
            _state["total_duration"] = result["total_duration"]
            _state["step"] = 2
            _save_session()
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

    images_per_segment = request.json.get("images_per_segment", 1)
    _state["busy"] = True
    _state["error"] = None

    def work():
        try:
            from image_gen import generate_visuals
            images_dir = OUTPUT / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            def on_progress(i, total, text):
                _push("status", {
                    "step": 3,
                    "message": f"Image {i+1}/{total}",
                    "progress": round((i / total) * 100),
                })

            _push("status", {"step": 3, "message": "Generating image prompts…", "progress": 0})
            _, wiki_meta, _ = _active_wiki_paths()
            wiki_display_name = wiki_meta["name"] if wiki_meta else ""
            results = generate_visuals(
                topic=_state["topic"],
                segments=_state["segments"],
                output_dir=images_dir,
                on_progress=on_progress,
                images_per_segment=images_per_segment,
                wiki_name=wiki_display_name,
            )

            image_paths = [f"images/{Path(r['image_path']).name}" for r in results]
            _state["images"] = image_paths
            # Store expanded visual segments for video stitching
            _state["visual_segments"] = [
                {
                    "index": i,
                    "text": r["text"],
                    "image_path": str(r["image_path"]),
                    "audio_path": r.get("audio_path", ""),
                    "start_seconds": r["start_seconds"],
                    "duration_seconds": r["duration_seconds"],
                    "_parent_segment": r.get("_parent_segment"),
                }
                for i, r in enumerate(results)
            ]
            _state["step"] = 3
            _save_session()
            _push("images_done", {"images": image_paths})
        except Exception as e:
            _state["error"] = str(e)
            _push("error", {"message": str(e)})
        finally:
            _state["busy"] = False

    threading.Thread(target=work, daemon=True).start()
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# Step 4 — Stitch video
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
            _clean_dir(OUTPUT / "subtitles")
            _clean_dir(OUTPUT / "video")
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
                visual_segments=_state.get("visual_segments") or None,
                language=_state.get("language"),
            )

            _state["step"] = 4
            _save_session()
            _push("video_done", {"video": "video/final.mp4"})
        except Exception as e:
            _state["error"] = str(e)
            _push("error", {"message": str(e)})
        finally:
            _state["busy"] = False

    threading.Thread(target=work, daemon=True).start()
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# Step 5 — YouTube upload
# ---------------------------------------------------------------------------
def _yt_authenticated():
    try:
        from youtube_upload import is_authenticated
        return is_authenticated()
    except Exception:
        return False


@app.route("/api/youtube/auth")
def api_youtube_auth():
    from youtube_upload import get_flow, CLIENT_SECRETS_FILE
    if not CLIENT_SECRETS_FILE.exists():
        return jsonify(error="client_secrets.json not found. See setup instructions."), 400
    redirect_uri = request.url_root.rstrip("/") + "/api/youtube/callback"
    flow = get_flow(redirect_uri)
    authorization_url, state = flow.authorization_url(
        access_type="offline", prompt="consent",
    )
    session["oauth_state"] = state
    # Store PKCE code verifier so the callback can use it
    session["code_verifier"] = flow.code_verifier
    return jsonify(url=authorization_url)


@app.route("/api/youtube/callback")
def api_youtube_callback():
    from youtube_upload import get_flow, save_credentials
    redirect_uri = request.url_root.rstrip("/") + "/api/youtube/callback"
    flow = get_flow(redirect_uri)
    flow.code_verifier = session.pop("code_verifier", None)
    flow.fetch_token(authorization_response=request.url)
    save_credentials(flow.credentials)
    return (
        "<html><body><script>"
        "window.opener && window.opener.postMessage('youtube_auth_ok','*');"
        "window.close();"
        "</script><p>Authorized! You can close this window.</p></body></html>"
    )


@app.route("/api/youtube/generate-metadata", methods=["POST"])
def api_youtube_generate_metadata():
    if not _state["script"]:
        return jsonify(error="No script available"), 400
    try:
        from rag_query import generate_youtube_metadata
        _, meta, _ = _active_wiki_paths()
        wiki_name = meta["name"] if meta else ""
        result = generate_youtube_metadata(
            topic=_state["topic"],
            script=_state["script"],
            wiki_name=wiki_name,
            language=_state.get("language", "es-MX"),
        )
        return jsonify(ok=True, **result)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/youtube/upload", methods=["POST"])
def api_youtube_upload():
    if _state["busy"]:
        return jsonify(error="Pipeline is busy"), 409
    video_path = OUTPUT / "video" / "final.mp4"
    if not video_path.exists():
        return jsonify(error="No video found. Build a video first."), 400

    title = request.json.get("title", "").strip()
    description = request.json.get("description", "").strip()
    tags = request.json.get("tags", [])
    privacy = request.json.get("privacy", "private")

    if not title:
        return jsonify(error="Title is required"), 400

    _state["busy"] = True
    _state["error"] = None

    def work():
        try:
            from youtube_upload import upload_video

            def on_progress(pct):
                _push("status", {"step": 5, "message": f"Uploading… {pct}%", "progress": pct})

            _push("status", {"step": 5, "message": "Starting upload…", "progress": 0})
            result = upload_video(
                video_path=str(video_path),
                title=title,
                description=description,
                tags=tags,
                privacy=privacy,
                on_progress=on_progress,
            )
            video_id = result["id"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            _push("youtube_done", {"video_id": video_id, "url": video_url})
        except Exception as e:
            _state["error"] = str(e)
            _push("error", {"message": str(e)})
        finally:
            _state["busy"] = False

    threading.Thread(target=work, daemon=True).start()
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# Scraping pipeline
# ---------------------------------------------------------------------------
_scrape_state = {"busy": False, "error": None}


@app.route("/api/start-scrape", methods=["POST"])
def api_start_scrape():
    if _scrape_state["busy"]:
        return jsonify(error="A scrape is already running"), 409

    url = request.json.get("url", "").strip()
    name = request.json.get("name", "").strip()
    max_pages = request.json.get("max_pages", 0)

    if not url or not name:
        return jsonify(error="URL and name are required"), 400

    _scrape_state["busy"] = True
    _scrape_state["error"] = None

    def work():
        try:
            from scrape_pipeline import run_full_pipeline

            def on_progress(stage, current, total, msg):
                _push("scrape_status", {
                    "stage": stage,
                    "current": current,
                    "total": total,
                    "message": msg,
                })

            slug = run_full_pipeline(
                wiki_url=url,
                wiki_name=name,
                max_pages=max_pages,
                on_progress=on_progress,
            )
            _push("scrape_done", {"slug": slug, "name": name})
        except Exception as e:
            _scrape_state["error"] = str(e)
            _push("scrape_error", {"message": str(e)})
        finally:
            _scrape_state["busy"] = False

    threading.Thread(target=work, daemon=True).start()
    return jsonify(ok=True)


@app.route("/api/scrape-state")
def api_scrape_state():
    return jsonify(busy=_scrape_state["busy"], error=_scrape_state.get("error"))


@app.route("/api/cancel-scrape", methods=["POST"])
def api_cancel_scrape():
    _scrape_state["busy"] = False
    _scrape_state["error"] = None
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# State endpoint
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
        active_wiki=_state.get("active_wiki"),
        has_video=(OUTPUT / "video" / "final.mp4").exists(),
        youtube_authenticated=_yt_authenticated(),
        has_client_secrets=(BASE / "client_secrets.json").exists(),
    )


# ---------------------------------------------------------------------------
# Restart
# ---------------------------------------------------------------------------
@app.route("/api/restart", methods=["POST"])
def api_restart():
    if _state["busy"]:
        return jsonify(error="Pipeline is busy — wait for it to finish first"), 409

    def _do_restart():
        import time
        time.sleep(0.6)  # let the HTTP response reach the client first
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_do_restart, daemon=True).start()
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scrape")
def scrape_page():
    return render_template("scrape.html")


if __name__ == "__main__":
    import socket
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"\n  Video Pipeline -> http://localhost:5000")
    print(f"  LAN access    -> http://{local_ip}:5000\n")
    app.run(host="0.0.0.0", debug=False, port=5000, threaded=True)
