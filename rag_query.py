"""RAG pipeline: hybrid search + reranker + LLM video script generation.

Flow:
  Topic → embedding → hybrid search (vector + BM25) →
  top 25 candidates → FlashRank reranker → top 5 → short-form video script (1-3 min)

Usage:
    python rag_query.py "Finnick Odair"
    python rag_query.py "Finnick Odair" --debug
    python rag_query.py --interactive
"""

import argparse
import json
import os
import re
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

CHROMA_DIR = Path(__file__).parent / "chromadb_store"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
GEMINI_MODEL = "gemini-3.1-pro-preview"

_gemini = genai.Client()

TOP_CANDIDATES = 25
TOP_RERANKED = 5


# ---------------------------------------------------------------------------
# 1) Load ChromaDB + build BM25 index
# ---------------------------------------------------------------------------

def load_stores():
    """Load ChromaDB collection and build a BM25 index over its documents."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("hunger_games_wiki")

    # Pull all documents for BM25
    all_data = collection.get(include=["documents"])
    ids = all_data["ids"]
    docs = all_data["documents"]

    # Tokenize for BM25
    tokenized = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)

    return collection, ids, docs, bm25


# ---------------------------------------------------------------------------
# 2) Query embedding
# ---------------------------------------------------------------------------

def embed_query(query: str) -> list[float]:
    resp = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": [query]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


# ---------------------------------------------------------------------------
# 3) Hybrid search: vector + BM25
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    collection,
    ids: list[str],
    docs: list[str],
    bm25: BM25Okapi,
    top_n: int = TOP_CANDIDATES,
) -> list[dict]:
    """Combine vector similarity and BM25 scores, return top_n candidates."""

    # Vector search
    query_embedding = embed_query(query)
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["documents", "distances"],
    )
    vector_ids = vector_results["ids"][0]
    vector_docs = vector_results["documents"][0]
    # ChromaDB cosine distance: lower = more similar. Convert to score.
    vector_scores = [1 - d for d in vector_results["distances"][0]]

    # BM25 search
    tokenized_query = query.lower().split()
    bm25_scores_all = bm25.get_scores(tokenized_query)
    bm25_top_idx = np.argsort(bm25_scores_all)[::-1][:top_n]
    bm25_ids = [ids[i] for i in bm25_top_idx]
    bm25_docs_top = [docs[i] for i in bm25_top_idx]
    bm25_scores = [bm25_scores_all[i] for i in bm25_top_idx]

    # Normalize scores to [0, 1]
    def normalize(scores):
        mn, mx = min(scores), max(scores)
        if mx == mn:
            return [0.5] * len(scores)
        return [(s - mn) / (mx - mn) for s in scores]

    v_norm = normalize(vector_scores)
    b_norm = normalize(bm25_scores)

    # Merge into a single ranked list using reciprocal rank fusion
    combined = {}
    for rank, (cid, doc, score) in enumerate(zip(vector_ids, vector_docs, v_norm)):
        combined[cid] = {
            "id": cid,
            "document": doc,
            "vector_score": score,
            "bm25_score": 0.0,
            "rrf": 1 / (rank + 60),  # RRF constant k=60
        }
    for rank, (cid, doc, score) in enumerate(zip(bm25_ids, bm25_docs_top, b_norm)):
        if cid in combined:
            combined[cid]["bm25_score"] = score
            combined[cid]["rrf"] += 1 / (rank + 60)
        else:
            combined[cid] = {
                "id": cid,
                "document": doc,
                "vector_score": 0.0,
                "bm25_score": score,
                "rrf": 1 / (rank + 60),
            }

    # Sort by RRF score
    ranked = sorted(combined.values(), key=lambda x: x["rrf"], reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# 4) Reranker
# ---------------------------------------------------------------------------

_ranker = None

def get_ranker():
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

    # Sort by rerank score
    results.sort(key=lambda x: x["score"], reverse=True)

    reranked = []
    for r in results[:top_n]:
        # Find original candidate to keep metadata
        orig = next(c for c in candidates if c["id"] == r["id"])
        orig["rerank_score"] = r["score"]
        reranked.append(orig)

    return reranked


# ---------------------------------------------------------------------------
# 5) LLM answer generation
# ---------------------------------------------------------------------------

def generate_answer(query: str, passages: list[dict]) -> str:
    """Send the top passages + topic to Gemini and generate a video script."""
    context = "\n\n---\n\n".join(p["document"] for p in passages)

    prompt = f"""You are a scriptwriter for informative short-form videos about The Hunger Games universe.
Using ONLY the provided context, write an engaging video script that lasts 1 to 3 minutes when read aloud (roughly 150–450 words).

Script requirements:
- Open with a strong hook that grabs the viewer's attention in the first sentence
- Present the information in a clear, conversational, engaging tone suitable for voiceover
- Structure it with a beginning, middle, and end
- End with a memorable closing line or call to reflection
- Do NOT use section labels like "Hook:", "Intro:", "Outro:" — write it as flowing narration
- If the context doesn't contain enough information to answer the topic, say so briefly

Context:
{context}

Topic: {query}

Script:"""

    response = _gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_query(query: str, collection, ids, docs, bm25, debug: bool = False):
    """Execute the full RAG pipeline for a single query."""

    # Step 1: Hybrid search
    candidates = hybrid_search(query, collection, ids, docs, bm25)

    if debug:
        print(f"\n{'='*60}")
        print(f"HYBRID SEARCH — Top {len(candidates)} candidates")
        print(f"{'='*60}")
        for i, c in enumerate(candidates[:10], 1):
            preview = c["document"][:100].replace("\n", " ")
            print(f"  {i}. [{c['id']}] RRF={c['rrf']:.4f} V={c['vector_score']:.3f} B={c['bm25_score']:.3f}")
            print(f"     {preview}...")

    # Step 2: Rerank
    top_passages = rerank(query, candidates)

    if debug:
        print(f"\n{'='*60}")
        print(f"RERANKED — Top {len(top_passages)}")
        print(f"{'='*60}")
        for i, p in enumerate(top_passages, 1):
            preview = p["document"][:120].replace("\n", " ")
            print(f"  {i}. [{p['id']}] rerank={p['rerank_score']:.4f}")
            print(f"     {preview}...")

    # Step 3: Generate answer
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
    parser = argparse.ArgumentParser(description="RAG query over Hunger Games wiki")
    parser.add_argument("query", nargs="?", help="Question to ask")
    parser.add_argument("--debug", action="store_true", help="Show search/rerank details")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    print("Loading stores...")
    collection, ids, docs, bm25 = load_stores()
    print(f"Ready! {len(ids)} documents indexed.\n")

    # Warm up the reranker
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
            run_query(query, collection, ids, docs, bm25, debug=args.debug)
    elif args.query:
        run_query(args.query, collection, ids, docs, bm25, debug=args.debug)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
