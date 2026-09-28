"""Pure helpers for the Brightmart pilot v3 fixed structure arms.

Spec: docs/superpowers/specs/2026-09-24-brightmart-v3-design.md.
R2s: stemmed metadata BM25 surface. R1c: contextualised struct-chunk text.
R3d: document-level authored-graph expansion that returns up to k_docs (fewer only if the
ranked list is shorter).
No gold ever reaches these functions.
"""
from __future__ import annotations

import re

STRUCTURE_ARMS_V3 = ("R1c", "R2s", "R3d", "R4c")
_WORD = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize(text: str) -> list[str]:
    return [_stem(t) for t in _WORD.findall(str(text).lower().replace("_", " "))]


def _value(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, list):
        return " ".join(_value(x) for x in v)
    return str(v)


def metadata_doc_dict(path: str, title: str, description: str | None, frontmatter: dict | None,
                      okf_type: str, status: str) -> dict:
    """One `documents` row -> the input dict `metadata_pseudo_docs_v3` expects.

    Shared by the evaluation run and the web bench so both build the
    structured (R2s) index from identical fields."""
    fm = frontmatter or {}
    return {"path": path, "title": title, "description": description, "tags": fm.get("tags", []),
            "okf_type": okf_type, "status": status,
            "frontmatter": {k: v for k, v in fm.items() if k != "tags"}}


def metadata_pseudo_docs_v3(documents: list[dict]) -> list[dict]:
    out = []
    for d in documents:
        parts = [d["title"], d.get("description") or "", " ".join(d.get("tags") or []),
                 f"type {d.get('okf_type', '')}", f"status {d.get('status', '')}"]
        parts += [f"{k} {_value(v)}" for k, v in sorted((d.get("frontmatter") or {}).items())]
        out.append({"chunk_id": f"meta:{d['path']}", "doc_path": d["path"],
                    "text": " ".join(normalize(" ".join(parts)))})
    return out


def contextual_text(title: str, heading_path: list[str], text: str) -> str:
    head = " > ".join([title, *heading_path]) if heading_path else title
    return f"{head}\n{text}"


def undirected_neighbors(edge_rows: list[tuple[str, str]]) -> dict[str, set[str]]:
    nb: dict[str, set[str]] = {}
    for src, dst in edge_rows:
        if src == dst:
            continue
        nb.setdefault(src, set()).add(dst)
        nb.setdefault(dst, set()).add(src)
    return nb


def doc_graph_arm(ranked_docs: list[str], neighbors: dict[str, set[str]],
                  k_docs: int = 10, n_seeds: int = 5) -> list[str]:
    seeds = list(dict.fromkeys(ranked_docs))[:n_seeds]
    out = list(seeds)
    for seed in seeds:
        for nb in sorted(neighbors.get(seed, ())):
            if nb not in out:
                out.append(nb)
    for d in ranked_docs:
        if d not in out:
            out.append(d)
    return out[:k_docs]


def structured_rank(index, question: str, k_docs: int, k_candidates: int = 50) -> list[tuple[str, float]]:
    """R2s document ranking: the normalized question against a BM25 index built
    over `metadata_pseudo_docs_v3`. Returns (doc_path, score), best first."""
    ranked: dict[str, float] = {}
    for chunk_id, score in index.search(" ".join(normalize(question)), k_candidates):
        ranked.setdefault(chunk_id.split("meta:", 1)[1], float(score))
    return list(ranked.items())[:k_docs]
