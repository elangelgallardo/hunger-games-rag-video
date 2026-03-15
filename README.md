# Hunger Games Wiki — RAG Video Pipeline

An end-to-end pipeline that scrapes a Fandom wiki, builds a RAG (Retrieval-Augmented Generation) search index, and turns any topic into a short-form video with narration, images, and word-level subtitles.

![Pipeline](https://img.shields.io/badge/pipeline-scrape%20→%20embed%20→%20search%20→%20script%20→%20video-e94560?style=flat-square)

## Pipeline overview

```
Fandom Wiki
    │
    ▼
┌──────────┐     ┌──────────┐     ┌─────────────┐     ┌────────────┐
│  Scrape  │ ──▶ │  Parse   │ ──▶ │  Chunk &    │ ──▶ │  Embed in  │
│ (httpx)  │     │(mwparser)│     │  Split      │     │  ChromaDB  │
└──────────┘     └──────────┘     └─────────────┘     └────────────┘
                                                             │
                                                             ▼
                                                     ┌──────────────┐
                                            Topic ──▶│ Hybrid Search│
                                                     │ Vector + BM25│
                                                     └──────┬───────┘
                                                            │
                                                            ▼
                                                     ┌──────────────┐
                                                     │  FlashRank   │
                                                     │  Reranker    │
                                                     └──────┬───────┘
                                                            │
                                                            ▼
                                                     ┌──────────────┐
                                                     │ Gemini LLM   │
                                                     │ Script Gen   │
                                                     └──────┬───────┘
                                                            │
                          ┌─────────────────────────────────┼──────────────────┐
                          ▼                                 ▼                  ▼
                   ┌─────────────┐                  ┌─────────────┐    ┌─────────────┐
                   │ Gemini TTS  │                  │ Gemini Image│    │ Whisper     │
                   │ Audio Gen   │                  │ Generation  │    │ Subtitles   │
                   └──────┬──────┘                  └──────┬──────┘    └──────┬──────┘
                          │                                │                  │
                          └────────────────┬───────────────┘──────────────────┘
                                           ▼
                                    ┌─────────────┐
                                    │ FFmpeg      │
                                    │ Ken Burns + │
                                    │ NVENC encode│
                                    └──────┬──────┘
                                           ▼
                                      final.mp4
```

## Features

- **Wiki scraping** — async bulk scraper using MediaWiki API (httpx, 50 pages/request)
- **Wikitext parsing** — structured extraction of infoboxes, sections, summaries, links (mwparserfromhell)
- **Chunking** — section-level, infobox, summary, and graph chunks
- **Embedding** — Ollama nomic-embed-text into ChromaDB (cosine similarity)
- **Hybrid search** — vector similarity + BM25, fused with Reciprocal Rank Fusion (k=60)
- **Cross-encoder reranking** — FlashRank ms-marco-MiniLM-L-12-v2
- **Script generation** — Gemini generates a 1–3 minute video script from retrieved context
- **Text-to-speech** — Gemini TTS (per-segment WAV generation)
- **Image generation** — Gemini generates cinematic scene prompts, then renders images
- **Word-level subtitles** — faster-whisper transcription, spell-corrected against the original script
- **Video assembly** — FFmpeg with Ken Burns effects (crop+scale), hardware-accelerated encoding (NVENC/AMF/QSV), per-word ASS subtitles burned in
- **Web UI** — Flask app with step-by-step wizard, real-time progress via SSE, audio/image/video preview

## Quick start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with `nomic-embed-text` pulled
- [FFmpeg](https://ffmpeg.org/) on PATH (with NVENC support for GPU encoding)
- A [Google AI Studio](https://aistudio.google.com/) API key

### Setup

```bash
git clone https://github.com/<your-username>/hunger-games-rag-video.git
cd hunger-games-rag-video

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Pull the Ollama embedding model

```bash
ollama pull nomic-embed-text
```

### 1. Scrape the wiki

```bash
python scrape_wiki.py https://thehungergames.fandom.com
# → raw/ (1208 pages)
```

### 2. Parse, chunk & embed

```bash
python parse_wiki.py          # → parsed/
python chunk_sections.py      # → chunks/
python chunk_infobox.py       # → infobox/
python chunk_summary.py       # → summary/
python chunk_graph.py         # → graph/
python embed_chunks.py        # → chromadb_store/
```

### 3. Generate a video (Web UI)

```bash
python web_app.py
# → open http://localhost:5000
```

The web app walks you through each step:
1. Enter a topic → RAG search → Gemini writes the script
2. Edit the script → generate TTS audio
3. Generate images
4. Pick encoder → stitch final video with subtitles

### 3b. Generate a video (CLI)

```bash
# Generate script
python rag_query.py --interactive

# Save script to script.txt, then:
python tts.py                          # → output/audio/
python image_gen.py                    # → output/images/
python video_stitch.py --encoder nvenc # → output/video/final.mp4
```

## Project structure

```
├── wiki_scraper.py      # Async bulk wiki scraper (httpx + MediaWiki API)
├── scrape_wiki.py       # CLI entry point for scraping
├── config.py            # Scraper configuration dataclass
├── parse_wiki.py        # Raw wikitext → structured JSON (mwparserfromhell)
├── chunk_sections.py    # Parsed → section chunks
├── chunk_infobox.py     # Parsed → infobox text chunks
├── chunk_summary.py     # Parsed → summary chunks
├── chunk_graph.py       # Parsed → link graph
├── embed_chunks.py      # Embed all chunks into ChromaDB via Ollama
├── rag_query.py         # Hybrid search + reranker + Gemini script generation
├── tts.py               # Gemini TTS — script → per-segment WAV files
├── image_gen.py         # Gemini image generation — segment prompts → PNGs
├── subtitles.py         # faster-whisper transcription + spell correction → ASS
├── video_stitch.py      # FFmpeg Ken Burns + NVENC + subtitle burn-in → MP4
├── web_app.py           # Flask web UI (step wizard + SSE progress)
├── rag_gui.py           # Legacy Tkinter GUI
├── templates/
│   └── index.html       # Web UI frontend (single-page app)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Configuration

| Variable | Where | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | `.env` | — | Google AI Studio API key (Gemini + TTS + Images) |
| `OLLAMA_URL` | `rag_query.py` | `localhost:11434` | Ollama server for embeddings |
| `EMBED_MODEL` | `rag_query.py` | `nomic-embed-text` | Embedding model |
| `GEMINI_MODEL` | `rag_query.py` | `gemini-3.1-pro-preview` | LLM for script generation |
| `TTS_MODEL` | `tts.py` | `gemini-2.5-flash-preview-tts` | TTS model |
| `IMAGE_MODEL` | `image_gen.py` | `gemini-3.1-flash-image-preview` | Image generation model |
| `TOP_CANDIDATES` | `rag_query.py` | `25` | Hybrid search candidates |
| `TOP_RERANKED` | `rag_query.py` | `5` | Passages sent to LLM |

## Video output specs

| Property | Value |
|---|---|
| Resolution | 1080 x 1920 (9:16 portrait) |
| Frame rate | 30 fps |
| Video codec | H.264 (NVENC/AMF/QSV/libx264) |
| Audio codec | AAC 192kbps |
| Subtitles | Burned-in ASS (word-by-word, uppercase) |
| Effects | Ken Burns (zoom in/out, pan L/R), fade transitions |

## License

MIT
