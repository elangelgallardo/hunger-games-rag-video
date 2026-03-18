"""Full scraping pipeline orchestrator for the web app.

Runs: scrape → parse → chunk (sections + infobox + summary + graph) → embed
All data goes into wikis/{slug}/.
"""

import asyncio
import json
import sys
from pathlib import Path

from wiki_registry import create_wiki, get_wiki_paths, get_wiki_meta


def run_full_pipeline(
    wiki_url: str,
    wiki_name: str,
    max_pages: int = 0,
    on_progress=None,
) -> str:
    """Run the complete scrape → parse → chunk → embed pipeline.

    Args:
        wiki_url:    Fandom wiki base URL (e.g. https://avatar.fandom.com)
        wiki_name:   Display name for the wiki
        max_pages:   0 = unlimited
        on_progress: callback(stage, current, total, message)

    Returns:
        The wiki slug.
    """
    def progress(stage, current, total, msg):
        if on_progress:
            on_progress(stage, current, total, msg)
        print(f"  [{stage}] {current}/{total} — {msg}")

    # ── 1. Create wiki folder structure ─────────────────────────────
    slug = create_wiki(wiki_name, wiki_url)
    paths = get_wiki_paths(slug)
    meta = get_wiki_meta(slug)

    # ── 2. Scrape ───────────────────────────────────────────────────
    progress("scraping", 0, 0, f"Scraping {wiki_url}…")
    from config import Config
    from wiki_scraper import run_scraper

    api_url = wiki_url.rstrip("/") + "/api.php"
    cfg = Config(wiki_api_url=api_url, raw_dir=paths["raw"], max_pages=max_pages)
    saved = asyncio.run(run_scraper(cfg))
    progress("scraping", len(saved), len(saved), f"{len(saved)} pages scraped")

    # ── 3. Parse ────────────────────────────────────────────────────
    from parse_wiki import parse_page

    raw_files = sorted(paths["raw"].glob("*.json"))
    parsed_dir = paths["parsed"]
    parsed_dir.mkdir(parents=True, exist_ok=True)

    parsed_count = 0
    for i, raw_path in enumerate(raw_files):
        progress("parsing", i + 1, len(raw_files), raw_path.name)
        try:
            result = parse_page(raw_path)
            out_path = parsed_dir / raw_path.name
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            parsed_count += 1
        except Exception as e:
            print(f"  Parse error on {raw_path.name}: {e}")

    progress("parsing", parsed_count, parsed_count, f"{parsed_count} pages parsed")

    # ── 4. Chunk: sections ──────────────────────────────────────────
    from chunk_sections import chunk_page

    chunks_dir = paths["chunks"]
    chunks_dir.mkdir(parents=True, exist_ok=True)

    parsed_files = sorted(parsed_dir.glob("*.json"))
    chunk_count = 0
    for i, pf in enumerate(parsed_files):
        progress("chunking", i + 1, len(parsed_files), f"sections: {pf.name}")
        try:
            chunks = chunk_page(pf)
            for c in chunks:
                fname = f"{c['chunk_id']}.json"
                (chunks_dir / fname).write_text(
                    json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                chunk_count += 1
        except Exception as e:
            print(f"  Chunk error on {pf.name}: {e}")

    # ── 5. Chunk: infoboxes ─────────────────────────────────────────
    from chunk_infobox import build_infobox_chunk

    infobox_dir = paths["infobox"]
    infobox_dir.mkdir(parents=True, exist_ok=True)

    infobox_count = 0
    for i, pf in enumerate(parsed_files):
        progress("chunking", i + 1, len(parsed_files), f"infobox: {pf.name}")
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
            chunk = build_infobox_chunk(data)
            if chunk:
                fname = f"{chunk['chunk_id']}.json"
                (infobox_dir / fname).write_text(
                    json.dumps(chunk, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                infobox_count += 1
        except Exception as e:
            print(f"  Infobox error on {pf.name}: {e}")

    # ── 6. Chunk: summaries ─────────────────────────────────────────
    from chunk_summary import build_summary_chunk

    summary_dir = paths["summary"]
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_count = 0
    for i, pf in enumerate(parsed_files):
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
            chunk = build_summary_chunk(data)
            if chunk:
                fname = f"{chunk['chunk_id']}.json"
                (summary_dir / fname).write_text(
                    json.dumps(chunk, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                summary_count += 1
        except Exception:
            pass

    # ── 7. Chunk: graph ─────────────────────────────────────────────
    from chunk_graph import build_graph_chunk

    graph_dir = paths["graph"]
    graph_dir.mkdir(parents=True, exist_ok=True)

    for pf in parsed_files:
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
            chunk = build_graph_chunk(data)
            if chunk and chunk.get("links"):
                fname = f"{chunk['node']}.json"
                (graph_dir / fname).write_text(
                    json.dumps(chunk, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        except Exception:
            pass

    # ── 7b. Build consolidated link graph ───────────────────────────
    link_graph = {}
    for gf in sorted(graph_dir.glob("*.json")):
        try:
            g = json.loads(gf.read_text(encoding="utf-8"))
            if g.get("node") and g.get("links"):
                link_graph[g["node"]] = g["links"]
        except Exception:
            pass

    graph_path = paths["base"] / "link_graph.json"
    graph_path.write_text(json.dumps(link_graph, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("chunking", 0, 0, f"Link graph: {len(link_graph)} pages")

    total_chunks = chunk_count + infobox_count + summary_count
    progress("chunking", total_chunks, total_chunks,
             f"{chunk_count} sections, {infobox_count} infoboxes, {summary_count} summaries")

    # ── 8. Embed into ChromaDB ──────────────────────────────────────
    from embed_chunks import load_all_chunks, get_embeddings, get_embedding, _build_embed_text, BATCH_SIZE

    progress("embedding", 0, 0, "Loading chunks for embedding…")

    items = load_all_chunks(
        chunks_dir=paths["chunks"],
        infobox_dir=paths["infobox"],
        summary_dir=paths["summary"],
    )

    import chromadb
    chromadb_dir = paths["chromadb_store"]
    chromadb_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chromadb_dir))
    collection = client.get_or_create_collection(
        name=meta["collection"],
        metadata={"hnsw:space": "cosine"},
    )

    existing = set(collection.get()["ids"]) if collection.count() > 0 else set()
    to_embed = [(cid, content, m) for cid, content, m in items if cid not in existing]

    progress("embedding", 0, len(to_embed), f"{len(to_embed)} items to embed")

    errors = 0
    for i in range(0, len(to_embed), BATCH_SIZE):
        batch = to_embed[i:i + BATCH_SIZE]
        ids = [cid for cid, _, _ in batch]
        contents = [content for _, content, _ in batch]
        metadatas = [m for _, _, m in batch]
        embed_texts = [
            _build_embed_text(m["page"], m["section"], content)
            for content, m in zip(contents, metadatas)
        ]

        progress("embedding", min(i + BATCH_SIZE, len(to_embed)), len(to_embed),
                 f"Embedding batch {i // BATCH_SIZE + 1}…")

        embeddings = get_embeddings(embed_texts)
        if embeddings is not None:
            collection.add(ids=ids, documents=contents, metadatas=metadatas, embeddings=embeddings)
        else:
            for cid, content, m in batch:
                embed_text = _build_embed_text(m["page"], m["section"], content)
                emb = get_embedding(embed_text)
                if emb is not None:
                    collection.add(ids=[cid], documents=[content], metadatas=[m], embeddings=[emb])
                else:
                    errors += 1

    doc_count = collection.count()
    progress("embedding", doc_count, doc_count, f"Done! {doc_count} embeddings stored")

    # Update meta with stats
    meta["doc_count"] = doc_count
    meta["raw_count"] = len(raw_files)
    meta["parsed_count"] = parsed_count
    (paths["meta"]).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return slug
