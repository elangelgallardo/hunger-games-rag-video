"""Centralized configuration for the RAG video pipeline.

All model names, URLs, and tuning parameters live here so they can be
changed in one place.  Environment variables override defaults where noted.
"""

import os

# ---------------------------------------------------------------------------
# Ollama (local embedding server)
# ---------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"

# ---------------------------------------------------------------------------
# Google Gemini models
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"          # factual scripts
GEMINI_THINKING_MODEL = "gemini-3.1-pro-preview"         # theory / speculation
IMAGE_MODEL = "gemini-2.5-flash-image"                   # image generation
PROMPT_MODEL = "gemini-3.1-pro-preview"                  # image prompt generation
TTS_MODEL = "gemini-2.5-flash-preview-tts"               # text-to-speech

# ---------------------------------------------------------------------------
# Whisper (subtitle generation)
# ---------------------------------------------------------------------------
WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"          # "cuda" for GPU
WHISPER_COMPUTE = "int8"        # "float16" for GPU

# ---------------------------------------------------------------------------
# RAG search parameters
# ---------------------------------------------------------------------------
TOP_CANDIDATES = 25             # hybrid search candidate pool
TOP_RERANKED = 5                # passages sent to LLM after reranking
RRF_K = 60                      # reciprocal rank fusion constant

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBED_MAX_WORDS = 1000          # max words per chunk before splitting
EMBED_BATCH_SIZE = 50           # chunks per Ollama batch request

# ---------------------------------------------------------------------------
# TTS / Audio
# ---------------------------------------------------------------------------
TTS_VOICE = "Charon"
TTS_SAMPLE_RATE = 24000
TTS_CHANNELS = 1
TTS_SAMPLE_WIDTH = 2            # 16-bit PCM
TTS_MAX_WORDS_PER_SEGMENT = 40  # ~8-12s per segment for good visual pacing

# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30
VIDEO_FADE = 0.4                # fade-in / fade-out duration (seconds)
VIDEO_EFFECTS = ["zoom_in", "zoom_out", "pan_right", "pan_left"]
VIDEO_PAN_FACTOR = 1.2          # scale factor for pan effects (1.2 = 20% extra)

# ---------------------------------------------------------------------------
# API retry settings (shared by TTS, image gen, etc.)
# ---------------------------------------------------------------------------
API_TIMEOUT_MS = 300_000                            # 5-minute client timeout
MAX_RETRIES = 7
RETRY_DELAYS_DISCONNECT = [2, 3, 5, 5, 10, 10, 15]  # server dropped — retry fast
RETRY_DELAYS_OTHER = [5, 10, 20, 40, 60, 60, 90]     # rate limit / overload — back off

# ---------------------------------------------------------------------------
# Image generation style
# ---------------------------------------------------------------------------
IMAGE_STYLE_SUFFIX = (
    "vertical 9:16 portrait format, cinematic, dramatic, "
    "epic lighting, high contrast, photorealistic, ultra-detailed, "
    "no text, no watermarks, no logos"
)
