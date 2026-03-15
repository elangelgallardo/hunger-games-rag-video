"""Embed all chunks into a local ChromaDB using Ollama nomic-embed-text.

Reads chunks from chunks/ and infobox/ folders, builds plain text with
Page/Section context, splits long texts, and stores embeddings in ChromaDB.

Usage:
    python embed_chunks.py              # embed everything
    python embed_chunks.py --debug      # print plain text, don't embed
    python embed_chunks.py --debug -n 5 # print first 5 texts only
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb
import httpx
from tqdm import tqdm

CHUNKS_DIR = Path(__file__).parent / "chunks"
INFOBOX_DIR = Path(__file__).parent / "infobox"
SUMMARY_DIR = Path(__file__).parent / "summary"
CHROMA_DIR = Path(__file__).parent / "chromadb_store"

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"

# nomic-embed-text has an 8192 token context window (~4000 words).
# We split at ~1500 words to leave headroom.
MAX_WORDS = 1000
BATCH_SIZE = 50


def build_text(chunk: dict) -> str:
    """Build plain text with context header for embedding."""
    page = chunk.get("page_title") or chunk.get("page", "")
    section = chunk.get("section", "")
    content = chunk.get("content", "")

    header = f"Page: {page}"
    if section:
        header += f"\nSection: {section}"

    return f"{header}\n{content}"


def split_text(text: str, chunk_id: str) -> list[tuple[str, str]]:
    """Split text into pieces under MAX_WORDS. Returns (id, text) pairs.

    Each piece keeps the header (Page/Section lines) for context.
    """
    lines = text.split("\n")

    # Extract header lines (Page: and Section:)
    header_lines = []
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith("Page:") or line.startswith("Section:"):
            header_lines.append(line)
            body_start = i + 1
        else:
            break

    header = "\n".join(header_lines)
    body = "\n".join(lines[body_start:])
    words = body.split()

    if len(words) <= MAX_WORDS:
        return [(chunk_id, text)]

    # Split body into pieces, prepend header to each
    pieces = []
    part = 0
    for i in range(0, len(words), MAX_WORDS):
        part_words = words[i:i + MAX_WORDS]
        part_text = f"{header}\n{' '.join(part_words)}"
        part_id = f"{chunk_id}_part{part}" if part > 0 else chunk_id
        pieces.append((part_id, part_text))
        part += 1

    return pieces


def get_embedding(text: str) -> list[float] | None:
    """Embed a single text. Returns None on failure."""
    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={"model": MODEL, "input": [text]},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except Exception as e:
        print(f"\n  Single embed failed ({len(text.split())} words): {e}")
        return None


def get_embeddings(texts: list[str]) -> list[list[float]] | None:
    """Embed a batch. Falls back to one-at-a-time on failure."""
    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={"model": MODEL, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]
    except Exception as e:
        print(f"\n  Batch failed ({len(texts)} texts): {e}")
        print("  Retrying individually...")
        return None


def load_all_chunks() -> list[tuple[str, str]]:
    """Load chunks from all source folders and return (id, text) pairs."""
    items = []

    # Section chunks
    for path in sorted(CHUNKS_DIR.glob("*.json")):
        chunk = json.loads(path.read_text(encoding="utf-8"))
        text = build_text(chunk)
        items.extend(split_text(text, chunk["chunk_id"]))

    # Infobox chunks
    for path in sorted(INFOBOX_DIR.glob("*.json")):
        chunk = json.loads(path.read_text(encoding="utf-8"))
        text = f"Page: {chunk.get('page', '')}\nSection: Infobox\n{chunk['content']}"
        items.extend(split_text(text, chunk["chunk_id"]))

    # Summary chunks
    for path in sorted(SUMMARY_DIR.glob("*.json")):
        chunk = json.loads(path.read_text(encoding="utf-8"))
        # Summary files don't have page_title, derive from chunk_id
        page = chunk["chunk_id"].replace("_summary", "").replace("_", " ").title()
        text = f"Page: {page}\nSection: Summary\n{chunk['content']}"
        items.extend(split_text(text, chunk["chunk_id"]))

    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Print texts without embedding")
    parser.add_argument("-n", type=int, default=0, help="Limit items in debug mode")
    args = parser.parse_args()

    print("Loading chunks...")
    items = load_all_chunks()
    print(f"Loaded {len(items)} embedding items")

    if args.debug:
        limit = args.n if args.n > 0 else len(items)
        for i, (cid, text) in enumerate(items[:limit]):
            print(f"\n{'='*60}")
            print(f"ID: {cid}")
            print(f"{'='*60}")
            print(text)
            print(f"[{len(text.split())} words]")
        print(f"\nTotal: {len(items)} items")
        return

    # Create ChromaDB
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="hunger_games_wiki",
        metadata={"hnsw:space": "cosine"},
    )

    # Check what's already stored to allow resuming
    existing = set(collection.get()["ids"]) if collection.count() > 0 else set()
    to_embed = [(cid, text) for cid, text in items if cid not in existing]

    if not to_embed:
        print("All items already embedded!")
        return

    if existing:
        print(f"Skipping {len(existing)} already embedded, {len(to_embed)} remaining")

    # Embed in batches
    errors = 0
    pbar = tqdm(total=len(to_embed), desc="Embedding")
    for i in range(0, len(to_embed), BATCH_SIZE):
        batch = to_embed[i:i + BATCH_SIZE]
        ids = [cid for cid, _ in batch]
        texts = [text for _, text in batch]

        embeddings = get_embeddings(texts)

        if embeddings is not None:
            collection.add(ids=ids, documents=texts, embeddings=embeddings)
        else:
            # Fallback: embed one at a time
            for cid, text in batch:
                emb = get_embedding(text)
                if emb is not None:
                    collection.add(ids=[cid], documents=[text], embeddings=[emb])
                else:
                    errors += 1
                    print(f"  Skipped: {cid}")

        pbar.update(len(batch))

    pbar.close()
    if errors:
        print(f"\nWarning: {errors} items failed to embed")
    print(f"Done! {collection.count()} total embeddings in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
