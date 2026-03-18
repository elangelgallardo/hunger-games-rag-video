"""RAG pipeline: hybrid search + reranker + LLM video script generation.

Flow:
  Topic → embedding → hybrid search (vector + BM25) →
  top candidates → FlashRank reranker → top passages → Gemini script

Supports two modes:
  - Factual: direct hybrid search → rerank → script
  - Theory:  graph-expanded search → rerank → thinking model → speculation script

Usage:
    python rag_query.py "Finnick Odair"
    python rag_query.py "Finnick Odair" --debug
    python rag_query.py --interactive
"""

import argparse
import json
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()
import httpx
import numpy as np
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi

from pipeline_config import (
    OLLAMA_URL,
    EMBED_MODEL,
    GEMINI_MODEL,
    GEMINI_THINKING_MODEL,
    TOP_CANDIDATES,
    TOP_RERANKED,
    RRF_K,
)

CHROMA_DIR = Path(__file__).parent / "chromadb_store"

_gemini = genai.Client()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

LANG_INSTRUCTIONS = {
    "es-MX": (
        "Write the ENTIRE script in Mexican Spanish (español mexicano). "
        "Use natural, conversational Mexican Spanish — not formal Castilian. "
        "Use local expressions and phrasing that feel authentic to a Mexican audience."
    ),
    "en": "Write the script in English.",
}


def _passage_text(passage: dict) -> str:
    """Format a passage with Page/Section headers for LLM context."""
    meta = passage.get("metadata", {})
    page = meta.get("page", "")
    section = meta.get("section", "")
    header = f"Page: {page}" if page else ""
    if section:
        header += f"\nSection: {section}" if header else f"Section: {section}"
    if header:
        return f"{header}\n{passage['document']}"
    return passage["document"]


# ---------------------------------------------------------------------------
# 1) Load ChromaDB + build BM25 index
# ---------------------------------------------------------------------------

def load_stores(chromadb_dir=None, collection_name=None):
    """Load ChromaDB collection and build a BM25 index over its documents.

    Returns:
        (collection, ids, docs, metadatas, bm25)
    """
    chromadb_dir = str(chromadb_dir or CHROMA_DIR)
    client = chromadb.PersistentClient(path=chromadb_dir)

    if collection_name is None:
        cols = client.list_collections()
        collection_name = cols[0].name if cols else "hunger_games_wiki"

    collection = client.get_collection(collection_name)

    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    docs = all_data["documents"]
    metadatas = all_data["metadatas"]

    tokenized = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)

    return collection, ids, docs, metadatas, bm25


def load_link_graph(wiki_base_path) -> dict:
    """Load the consolidated link graph for a wiki.

    Falls back to building from individual graph/*.json files if the
    consolidated file doesn't exist.
    """
    wiki_base_path = Path(wiki_base_path)
    graph_path = wiki_base_path / "link_graph.json"
    if graph_path.exists():
        return json.loads(graph_path.read_text(encoding="utf-8"))
    graph_dir = wiki_base_path / "graph"
    link_graph = {}
    if graph_dir.exists():
        for gf in graph_dir.glob("*.json"):
            data = json.loads(gf.read_text(encoding="utf-8"))
            if data.get("node") and data.get("links"):
                link_graph[data["node"]] = data["links"]
    return link_graph


# ---------------------------------------------------------------------------
# 2) Query embedding
# ---------------------------------------------------------------------------

def embed_query(query: str) -> list[float]:
    """Embed a query string via Ollama."""
    resp = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": [query]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


# ---------------------------------------------------------------------------
# 3) Hybrid search: vector + BM25 with reciprocal rank fusion
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    collection,
    ids: list[str],
    docs: list[str],
    metadatas: list[dict],
    bm25: BM25Okapi,
    top_n: int = TOP_CANDIDATES,
) -> list[dict]:
    """Combine vector similarity and BM25 scores via RRF, return top_n candidates."""

    # Vector search
    query_embedding = embed_query(query)
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    )
    vector_ids = vector_results["ids"][0]
    vector_docs = vector_results["documents"][0]
    vector_metas = vector_results["metadatas"][0]
    vector_scores = [1 - d for d in vector_results["distances"][0]]

    # BM25 search
    tokenized_query = query.lower().split()
    bm25_scores_all = bm25.get_scores(tokenized_query)
    bm25_top_idx = np.argsort(bm25_scores_all)[::-1][:top_n]
    bm25_ids = [ids[i] for i in bm25_top_idx]
    bm25_docs_top = [docs[i] for i in bm25_top_idx]
    bm25_metas_top = [metadatas[i] for i in bm25_top_idx]
    bm25_scores = [bm25_scores_all[i] for i in bm25_top_idx]

    # Normalize scores to [0, 1]
    def normalize(scores):
        mn, mx = min(scores), max(scores)
        if mx == mn:
            return [0.5] * len(scores)
        return [(s - mn) / (mx - mn) for s in scores]

    v_norm = normalize(vector_scores)
    b_norm = normalize(bm25_scores)

    # Merge via reciprocal rank fusion
    combined = {}
    for rank, (cid, doc, meta, score) in enumerate(zip(vector_ids, vector_docs, vector_metas, v_norm)):
        combined[cid] = {
            "id": cid,
            "document": doc,
            "metadata": meta or {},
            "vector_score": score,
            "bm25_score": 0.0,
            "rrf": 1 / (rank + RRF_K),
        }
    for rank, (cid, doc, meta, score) in enumerate(zip(bm25_ids, bm25_docs_top, bm25_metas_top, b_norm)):
        if cid in combined:
            combined[cid]["bm25_score"] = score
            combined[cid]["rrf"] += 1 / (rank + RRF_K)
        else:
            combined[cid] = {
                "id": cid,
                "document": doc,
                "metadata": meta or {},
                "vector_score": 0.0,
                "bm25_score": score,
                "rrf": 1 / (rank + RRF_K),
            }

    ranked = sorted(combined.values(), key=lambda x: x["rrf"], reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# 3b) Graph-expanded search
# ---------------------------------------------------------------------------

def graph_search(
    query: str,
    collection,
    ids: list[str],
    docs: list[str],
    metadatas: list[dict],
    bm25: BM25Okapi,
    link_graph: dict,
    hops: int = 1,
    max_neighbor_pages: int = 15,
) -> list[dict]:
    """Find seed pages via hybrid search, walk the link graph, retrieve chunks.

    To avoid blowing up on dense wikis:
    - Only keeps neighbors that exist in the link graph (real pages)
    - Caps neighbors per hop via vote-based ranking
    - Defaults to 1 hop (2 hops fans out too aggressively)
    """
    seed_candidates = hybrid_search(query, collection, ids, docs, metadatas, bm25, top_n=TOP_CANDIDATES)
    seed_reranked = rerank(query, seed_candidates, top_n=TOP_RERANKED)

    seed_pages = set()
    for c in seed_reranked:
        page = c.get("metadata", {}).get("page", "")
        if page:
            seed_pages.add(page)

    pages_in_db = {m.get("page", "") for m in metadatas}

    final_pages = set(seed_pages)
    frontier = set(seed_pages)
    for _ in range(hops):
        neighbor_votes: dict[str, int] = {}
        for page in frontier:
            for neighbor in link_graph.get(page, []):
                if neighbor not in final_pages and neighbor in pages_in_db:
                    neighbor_votes[neighbor] = neighbor_votes.get(neighbor, 0) + 1
        ranked_neighbors = sorted(neighbor_votes, key=neighbor_votes.get, reverse=True)
        new_pages = ranked_neighbors[:max_neighbor_pages]
        if not new_pages:
            break
        final_pages.update(new_pages)
        frontier = set(new_pages)

    expanded = []
    for i, meta in enumerate(metadatas):
        if meta.get("page", "") in final_pages:
            expanded.append({
                "id": ids[i],
                "document": docs[i],
                "metadata": meta,
            })

    return expanded


# ---------------------------------------------------------------------------
# 4) Reranker
# ---------------------------------------------------------------------------

_ranker = None


def get_ranker():
    """Return a cached FlashRank cross-encoder reranker."""
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".flashrank_cache")
    return _ranker


def rerank(query: str, candidates: list[dict], top_n: int = TOP_RERANKED) -> list[dict]:
    """Use FlashRank cross-encoder to rerank candidates."""
    ranker = get_ranker()

    passages = [{"id": c["id"], "text": c["document"]} for c in candidates]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)

    results.sort(key=lambda x: x["score"], reverse=True)

    reranked = []
    for r in results[:top_n]:
        orig = next(c for c in candidates if c["id"] == r["id"])
        orig["rerank_score"] = r["score"]
        reranked.append(orig)

    return reranked


# ---------------------------------------------------------------------------
# 5) LLM script generation — factual mode
# ---------------------------------------------------------------------------

def generate_answer(query: str, passages: list[dict], wiki_name: str = "The Hunger Games", language: str = "es-MX") -> str:
    """Generate a factual video script from the top passages."""
    context = "\n\n---\n\n".join(_passage_text(p) for p in passages)
    lang_line = LANG_INSTRUCTIONS.get(language, LANG_INSTRUCTIONS["en"])

    prompt = f"""Write a short voiceover narration about the following topic from {wiki_name}.
Use ONLY the information in the provided context. Aim for 150–300 words (1–2 minutes spoken).
{lang_line}

Write plain continuous prose — no headers, labels, or bullet points. Do not follow any fixed structure; let the content determine the shape of the text. If the context lacks enough information to cover the topic, say so briefly.

Context:
{context}

Topic: {query}"""

    response = _gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


# ---------------------------------------------------------------------------
# 5b) LLM script generation — theory / speculation mode
# ---------------------------------------------------------------------------

def generate_theory(query: str, passages: list[dict], wiki_name: str = "The Hunger Games", language: str = "es-MX") -> str:
    """Generate a theory/speculation video script using the thinking model."""
    context = "\n\n---\n\n".join(_passage_text(p) for p in passages)
    lang_line = LANG_INSTRUCTIONS.get(language, LANG_INSTRUCTIONS["en"])

    prompt = f"""Write a short speculative narration about the following topic from {wiki_name}.
Use the provided context as your evidence base. Aim for 150–300 words (1–2 minutes spoken).
{lang_line}

Connect details from the lore to build a theory or interpretation around the topic. Draw on specific evidence from the context; distinguish clearly between what is established and what is speculation. Write plain continuous prose — no headers, labels, or bullet points. Do not follow any fixed structure.

Context:
{context}

Topic: {query}"""

    response = _gemini.models.generate_content(
        model=GEMINI_THINKING_MODEL,
        contents=prompt,
    )
    return response.text


# ---------------------------------------------------------------------------
# 6) YouTube metadata generation
# ---------------------------------------------------------------------------

def generate_youtube_metadata(topic: str, script: str, wiki_name: str = "", language: str = "es-MX") -> dict:
    """Generate title, description, and tags for a YouTube video.

    Returns a dict with keys: title, description, tags.
    Falls back to basic metadata if the LLM response can't be parsed.
    """
    lang_instructions = {
        "es-MX": "Write the title, description, and tags in Mexican Spanish.",
        "en": "Write the title, description, and tags in English.",
    }
    lang_line = lang_instructions.get(language, lang_instructions["en"])

    prompt = f"""You are a YouTube content creator specializing in {wiki_name or 'pop culture'} videos.

Given the video topic and script below, generate YouTube metadata.

Requirements:
- {lang_line}
- Title: catchy, under 80 characters, optimized for clicks and search
- Description: 2-3 short paragraphs, include relevant keywords naturally, end with a call to action (like, subscribe, comment)
- Tags: 8-15 relevant tags for YouTube search, mix of broad and specific

Return your answer in EXACTLY this JSON format (no markdown, no code blocks):
{{"title": "...", "description": "...", "tags": ["tag1", "tag2", ...]}}

Topic: {topic}

Script:
{script}"""

    response = _gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    raw = response.text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "title": topic[:80],
            "description": script[:500],
            "tags": [wiki_name, topic] if wiki_name else [topic],
        }


# ---------------------------------------------------------------------------
# Interactive / CLI
# ---------------------------------------------------------------------------

def run_query(query: str, collection, ids, docs, metadatas, bm25, debug: bool = False):
    """Execute the full RAG pipeline for a single query (CLI use)."""
    candidates = hybrid_search(query, collection, ids, docs, metadatas, bm25)

    if debug:
        print(f"\n{'='*60}")
        print(f"HYBRID SEARCH — Top {len(candidates)} candidates")
        print(f"{'='*60}")
        for i, c in enumerate(candidates[:10], 1):
            preview = c["document"][:100].replace("\n", " ")
            print(f"  {i}. [{c['id']}] RRF={c['rrf']:.4f} V={c['vector_score']:.3f} B={c['bm25_score']:.3f}")
            print(f"     {preview}...")

    top_passages = rerank(query, candidates)

    if debug:
        print(f"\n{'='*60}")
        print(f"RERANKED — Top {len(top_passages)}")
        print(f"{'='*60}")
        for i, p in enumerate(top_passages, 1):
            preview = p["document"][:120].replace("\n", " ")
            print(f"  {i}. [{p['id']}] rerank={p['rerank_score']:.4f}")
            print(f"     {preview}...")

    if debug:
        print(f"\n{'='*60}")
        print("CONTEXT SENT TO LLM")
        print(f"{'='*60}")
        for i, p in enumerate(top_passages, 1):
            print(f"\n--- Passage {i} [{p['id']}] ---")
            print(p["document"][:500])
            if len(p["document"]) > 500:
                print(f"  ... ({len(p['document'])} chars total)")

    print(f"\n{'='*60}")
    print("VIDEO SCRIPT")
    print(f"{'='*60}\n")
    answer = generate_answer(query, top_passages)
    print(answer)
    return answer


def main():
    parser = argparse.ArgumentParser(description="RAG query — video script generation")
    parser.add_argument("query", nargs="?", help="Topic to generate a script about")
    parser.add_argument("--debug", action="store_true", help="Show search/rerank details")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    print("Loading stores...")
    collection, ids, docs, metadatas, bm25 = load_stores()
    print(f"Ready! {len(ids)} documents indexed.\n")

    get_ranker()

    if args.interactive:
        while True:
            try:
                query = input("\nTopic: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not query or query.lower() in ("quit", "exit", "q"):
                break
            run_query(query, collection, ids, docs, metadatas, bm25, debug=args.debug)
    elif args.query:
        run_query(args.query, collection, ids, docs, metadatas, bm25, debug=args.debug)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
