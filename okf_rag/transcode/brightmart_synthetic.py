"""Brightmart synthetic retail corpus (doc_synthetic/corpus) -> OKF v0.2.

The corpus is already OKF-shaped markdown written by
`doc_synthetic/brightmart_render.py`: YAML frontmatter with `type, title,
description, tags, status, timestamp, parent` plus type-specific keys, and a
body whose links are bundle-absolute (`/regions/...md`).

Edges are authored only:
- `child` parent -> doc, from each doc's `parent` key;
- `prose` src -> dst, from body links.
The archive docs' `supersedes` key is deliberately NOT an edge: the authors
never linked an archive doc to the page it replaces, and turning the key into
an edge would hand the graph arm a shortcut to the distractor.

No git pin exists (the corpus lives in this repo); `corpus_sha256` is the
reproducibility record instead.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from .edge_weights import EDGE_WEIGHTS
from .links import ResolutionStats, extract_links, resolve
from .model import ConceptDoc, Edge

# `tags` stays in x_source too: the Postgres `frontmatter` column stores
# x_source only, and the R2 metadata arm reads tags from there.
OKF_CORE_KEYS = frozenset({"type", "title", "description", "status", "timestamp"})


def _split(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        raise ValueError("brightmart doc without frontmatter")
    _, fm, body = text.split("---\n", 2)
    return yaml.safe_load(fm) or {}, body


def _files(corpus_dir: Path) -> list[Path]:
    return sorted(Path(corpus_dir).rglob("*.md"), key=lambda p: p.relative_to(corpus_dir).as_posix())


def corpus_sha256(corpus_dir: Path) -> str:
    corpus_dir = Path(corpus_dir)
    h = hashlib.sha256()
    for p in _files(corpus_dir):
        h.update(p.relative_to(corpus_dir).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def transcode(corpus_dir: Path) -> tuple[list[ConceptDoc], list[Edge], ResolutionStats]:
    corpus_dir = Path(corpus_dir)
    parsed = []
    for p in _files(corpus_dir):
        fm, body = _split(p.read_text(encoding="utf-8"))
        parsed.append((p.relative_to(corpus_dir).as_posix(), fm, body))
    known = {path for path, _, _ in parsed}

    docs: list[ConceptDoc] = []
    edges: list[Edge] = []
    stats = ResolutionStats()
    for path, fm, body in parsed:
        ts = fm.get("timestamp")
        docs.append(ConceptDoc(
            path=path,
            okf_type=str(fm["type"]),
            title=str(fm["title"]),
            description=fm.get("description"),
            resource=None,
            tags=[str(t) for t in fm.get("tags") or []],
            timestamp=str(ts) if ts is not None else None,
            status=str(fm.get("status", "stable")),
            x_source={k: v for k, v in fm.items() if k not in OKF_CORE_KEYS},
            body=body,
        ))
        parent = fm.get("parent")
        if parent:
            dst_parent = resolve(path, parent, known)
            if dst_parent is None:
                stats.record_unresolved(path, parent)
            else:
                stats.resolved += 1
                edges.append(Edge(src=dst_parent, dst=path, kind="child", weight=EDGE_WEIGHTS["child"]))
        seen: set[str] = set()
        for href in extract_links(body):
            dst = resolve(path, href, known)
            if dst is None:
                stats.record_unresolved(path, href)
                continue
            stats.resolved += 1
            if dst != path and dst not in seen:
                seen.add(dst)
                edges.append(Edge(src=path, dst=dst, kind="prose", weight=EDGE_WEIGHTS["prose"]))
    if stats.unresolved:
        raise ValueError(f"unresolved links: {stats.unresolved_samples[:5]}")
    return docs, edges, stats
