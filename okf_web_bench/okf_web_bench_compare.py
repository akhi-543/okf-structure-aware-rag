from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

PACK_BUDGET = 2048
# The web bench serves the Brightmart bundle loaded by scripts/ingest_corpus.py.
ALLOWED_CORPORA = frozenset({"synthetic_retail_pilot"})
# Two retrieval paths, the ones the pilot evaluated:
# - flat: dense bge cosine over flat chunks (frontmatter, links, hierarchy removed)
# - structured: metadata-aware BM25 over each document's normalized frontmatter
#   (okf_rag.retrieve.brightmart_pilot_arms_v3.structured_rank), delivering the
#   ranked documents' structured chunks as evidence.
ARM_FLAT = "flat"
ARM_STRUCTURED = "structured"
ARMS = (ARM_FLAT, ARM_STRUCTURED)


def pack_context(chunks, tokenizer, budget):
    """Retain whole passages when possible, clipping only the final passage.

    Counts the fully serialized context, including path headers, and keeps
    precisely the delivered body so cited evidence is exactly what the model saw.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    context = ""
    parts = []
    for chunk in chunks:
        header = f"\nSource: {chunk['doc_path']}\n"
        body = chunk["text"]
        if len(tokenizer.encode(context + header, add_special_tokens=False)) >= budget:
            break
        if len(tokenizer.encode(context + header + body, add_special_tokens=False)) > budget:
            ids = tokenizer.encode(body, add_special_tokens=False)
            lo, hi = 0, len(ids)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                candidate = tokenizer.decode(ids[:mid], skip_special_tokens=True)
                if len(tokenizer.encode(context + header + candidate, add_special_tokens=False)) <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            body = tokenizer.decode(ids[:lo], skip_special_tokens=True)
        if body:
            context += header + body
            parts.append({**chunk, "text": body})
        if body != chunk["text"]:
            break
    return context, parts

_ZERO_METRICS = {
    "retrieve_ms": 0.0,
    "pack_ms": 0.0,
    "generate_ms": 0.0,
    "chunks": 0,
    "tokens_in": None,
    "clipped": False,
}


def _lane_error(arm: str, exc: Exception) -> dict:
    return {
        "mode": arm,
        "answer": None,
        "error": str(exc),
        "sources": [],
        "metrics": dict(_ZERO_METRICS),
    }


def _sources_from_retrieved(retrieved: list[dict]) -> list[dict]:
    return [
        {
            "citation": i,
            "doc_path": chunk["doc_path"],
            "chunk_id": chunk["chunk_id"],
            "heading_path": chunk.get("heading_path", []),
            "score": float(chunk["score"]),
            "truncated": False,
        }
        for i, chunk in enumerate(retrieved, 1)
    ]


def numbered_documentation(parts: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {part['doc_path']}\n{part['text']}"
        for i, part in enumerate(parts, 1)
    )


def run_lane(
    *,
    question: str,
    corpus: str,
    arm: str,
    top_k: int,
    tokenizer: Any,
    retrieve_fn: Callable[[str, str, str, int], list[dict]],
    generate_fn: Callable[[str, str], str | None] | None,
) -> dict:
    try:
        t0 = time.perf_counter()
        retrieved = retrieve_fn(corpus, arm, question, top_k)
        retrieve_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        return _lane_error(arm, exc)

    if tokenizer is None:
        return {
            "mode": arm,
            "answer": None,
            "error": "packing tokenizer unavailable",
            "sources": _sources_from_retrieved(retrieved),
            "metrics": {
                "retrieve_ms": retrieve_ms,
                "pack_ms": 0.0,
                "generate_ms": 0.0,
                "chunks": len(retrieved),
                "tokens_in": None,
                "clipped": False,
            },
        }

    try:
        original_text = {chunk["chunk_id"]: chunk["text"] for chunk in retrieved}

        t0 = time.perf_counter()
        packed_context, parts = pack_context(retrieved, tokenizer, PACK_BUDGET)
        pack_ms = (time.perf_counter() - t0) * 1000.0

        sources = []
        for i, part in enumerate(parts, 1):
            chunk_id = part["chunk_id"]
            orig = original_text.get(chunk_id, "")
            truncated = part["text"] != orig
            sources.append(
                {
                    "citation": i,
                    "doc_path": part["doc_path"],
                    "chunk_id": chunk_id,
                    "heading_path": part.get("heading_path", []),
                    "score": float(part["score"]),
                    "truncated": truncated,
                }
            )

        clipped = any(s["truncated"] for s in sources)
        tokens_in = len(tokenizer.encode(packed_context, add_special_tokens=False))

        answer: str | None = None
        error: str | None = None
        generate_ms = 0.0

        if generate_fn is None:
            error = "generator unavailable"
        else:
            try:
                t0 = time.perf_counter()
                answer = generate_fn(question, numbered_documentation(parts))
                generate_ms = (time.perf_counter() - t0) * 1000.0
                if answer is None:
                    error = "generator unavailable"
            except Exception as exc:
                answer = None
                error = str(exc)

        return {
            "mode": arm,
            "answer": answer,
            "error": error,
            "sources": sources,
            "metrics": {
                "retrieve_ms": retrieve_ms,
                "pack_ms": pack_ms,
                "generate_ms": generate_ms,
                "chunks": len(parts),
                "tokens_in": tokens_in,
                "clipped": clipped,
            },
        }
    except Exception as exc:
        return _lane_error(arm, exc)


def run_compare(
    *,
    question: str,
    corpus: str,
    lane_a: str,
    lane_b: str,
    top_k: int,
    tokenizer: Any,
    retrieve_fn: Callable[[str, str, str, int], list[dict]],
    generate_fn: Callable[[str, str], str | None] | None,
) -> dict:
    if corpus not in ALLOWED_CORPORA:
        raise ValueError(f"corpus must be one of {sorted(ALLOWED_CORPORA)}")
    if not question.strip():
        raise ValueError("question must not be empty")

    lane_kwargs = {
        "question": question,
        "corpus": corpus,
        "top_k": top_k,
        "tokenizer": tokenizer,
        "retrieve_fn": retrieve_fn,
        "generate_fn": generate_fn,
    }
    return {
        "question": question,
        "corpus": corpus,
        "lane_a": run_lane(arm=lane_a, **lane_kwargs),
        "lane_b": run_lane(arm=lane_b, **lane_kwargs),
    }
