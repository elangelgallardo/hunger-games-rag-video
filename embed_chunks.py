"""Embed all chunks into a local ChromaDB using Ollama nomic-embed-text.

Reads chunks from chunks/ and infobox/ folders, builds plain text with
Page/Section context, splits long texts, and stores embeddings in ChromaDB.

Each chunk is stored with:
  - document: the raw content text only
  - metadata: {"page": ..., "section": ..., "type": ...}
  - embedding: computed from content + Page/Section context header

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

from pipeline_config import (
    OLLAMA_URL,
    EMBED_MODEL,
    EMBED_MAX_WORDS as MAX_WORDS,
    EMBED_BATCH_SIZE as BATCH_SIZE,
)

CHUNKS_DIR = Path(__file__).parent / "chunks"
INFOBOX_DIR = Path(__file__).parent / "infobox"
SUMMARY_DIR = Path(__file__).parent / "summary"
CHROMA_DIR = Path(__file__).parent / "chromadb_store"

_EMBED_URL = f"{OLLAMA_URL}/api/embed"


def _build_embed_text(page: str, section: str, content: str) -> str:
    """Build the text used for embedding (includes context header)."""
    header = f"Page: {page}"
    if section:
        header += f"\nSection: {section}"
    return f"{header}\n{content}"


def _split_content(content: str, chunk_id: str, page: str, section: str, chunk_type: str) -> list[tuple[str, str, dict]]:
    """Split content into pieces under MAX_WORDS.

    Returns (id, content, metadata) tuples.
    Each piece keeps the same page/section metadata.
    """
    words = content.split()

    if len(words) <= MAX_WORDS:
        meta = {"page": page, "section": section, "type": chunk_type}
        return [(chunk_id, content, meta)]

    pieces = []
    part = 0
    for i in range(0, len(words), MAX_WORDS):
        part_words = words[i:i + MAX_WORDS]
        part_content = " ".join(part_words)
        part_id = f"{chunk_id}_part{part}" if part > 0 else chunk_id
        meta = {"page": page, "section": section, "type": chunk_type}
        pieces.append((part_id, part_content, meta))
        part += 1

    return pieces


def get_embedding(text: str) -> list[float] | None:
    """Embed a single text. Returns None on failure."""
    try:
        resp = httpx.post(
            _EMBED_URL,
            json={"model": EMBED_MODEL, "input": [text]},
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
            _EMBED_URL,
            json={"model": EMBED_MODEL, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]
    except Exception as e:
        print(f"\n  Batch failed ({len(texts)} texts): {e}")
        print("  Retrying individually...")
        return None


def load_all_chunks(
    chunks_dir: Path = None,
    infobox_dir: Path = None,
    summary_dir: Path = None,
) -> list[tuple[str, str, dict]]:
    """Load chunks from all source folders and return (id, content, metadata) tuples."""
    chunks_dir = chunks_dir or CHUNKS_DIR
    infobox_dir = infobox_dir or INFOBOX_DIR
    summary_dir = summary_dir or SUMMARY_DIR

    items = []

    # Section chunks
    for path in sorted(chunks_dir.glob("*.json")):
        chunk = json.loads(path.read_text(encoding="utf-8"))
        page = chunk.get("page_title") or chunk.get("page", "")
        section = chunk.get("section", "")
        content = chunk.get("content", "")
        items.extend(_split_content(content, chunk["chunk_id"], page, section, "section"))

    # Infobox chunks
    for path in sorted(infobox_dir.glob("*.json")):
        chunk = json.loads(path.read_text(encoding="utf-8"))
        page = chunk.get("page", "")
        section = "Infobox"
        content = chunk["content"]
        items.extend(_split_content(content, chunk["chunk_id"], page, section, "infobox"))

    # Summary chunks
    for path in sorted(summary_dir.glob("*.json")):
        chunk = json.loads(path.read_text(encoding="utf-8"))
        # Use page_title if available (new format); fall back to lossy derivation
        page = chunk.get("page_title") or chunk["chunk_id"].replace("_summary", "").replace("_", " ").title()
        section = "Summary"
        content = chunk["content"]
        items.extend(_split_content(content, chunk["chunk_id"], page, section, "summary"))

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
        for i, (cid, content, meta) in enumerate(items[:limit]):
            print(f"\n{'='*60}")
            print(f"ID: {cid}")
            print(f"Meta: {meta}")
            print(f"{'='*60}")
            print(content)
            print(f"[{len(content.split())} words]")
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
    to_embed = [(cid, content, meta) for cid, content, meta in items if cid not in existing]

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
        ids = [cid for cid, _, _ in batch]
        contents = [content for _, content, _ in batch]
        metadatas = [meta for _, _, meta in batch]

        # Embed with context header for better semantic search
        embed_texts = [
            _build_embed_text(meta["page"], meta["section"], content)
            for content, meta in zip(contents, metadatas)
        ]

        embeddings = get_embeddings(embed_texts)

        if embeddings is not None:
            collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            # Fallback: embed one at a time
            for cid, content, meta in batch:
                embed_text = _build_embed_text(meta["page"], meta["section"], content)
                emb = get_embedding(embed_text)
                if emb is not None:
                    collection.add(
                        ids=[cid],
                        documents=[content],
                        metadatas=[meta],
                        embeddings=[emb],
                    )
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
