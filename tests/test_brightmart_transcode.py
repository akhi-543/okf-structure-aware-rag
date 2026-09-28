"""Brightmart synthetic corpus -> OKF ConceptDocs + authored edges."""
from pathlib import Path

import pytest

from okf_rag.transcode.brightmart_synthetic import corpus_sha256, transcode

CORPUS = Path("doc_synthetic/corpus")


def test_doc_count_and_paths():
    docs, _, _ = transcode(CORPUS)
    assert len(docs) == 53
    paths = {d.path for d in docs}
    assert "brightmart-home.md" in paths
    assert "regions/northeast/store-s01-riverside.md" in paths
    assert all(not p.startswith("/") for p in paths)


def test_all_links_resolve():
    _, _, stats = transcode(CORPUS)
    assert stats.unresolved == 0
    assert stats.resolved > 0


def test_every_parent_yields_child_edge():
    docs, edges, _ = transcode(CORPUS)
    child = {(e.src, e.dst) for e in edges if e.kind == "child"}
    for d in docs:
        parent = d.x_source.get("parent")
        if parent:
            assert (parent.lstrip("/"), d.path) in child


def test_frontmatter_preserved_in_x_source():
    docs, _, _ = transcode(CORPUS)
    s01 = next(d for d in docs if d.path == "regions/northeast/store-s01-riverside.md")
    assert s01.okf_type == "store"
    assert s01.title == "Brightmart Riverside"
    assert s01.x_source["sq_ft"] == 182000
    assert s01.x_source["store_format"] == "supercenter"
    assert s01.status == "stable"
    assert "S01" in s01.x_source["tags"]


def test_supersedes_is_not_an_edge():
    docs, edges, _ = transcode(CORPUS)
    a1 = next(d for d in docs if d.path == "archive/archived-cedar-falls-profile-2022.md")
    target = a1.x_source["supersedes"].lstrip("/")
    assert not any(e.src == a1.path and e.dst == target for e in edges)


def test_prose_edges_have_prose_weight():
    _, edges, _ = transcode(CORPUS)
    prose = [e for e in edges if e.kind == "prose"]
    assert prose and all(e.weight == 0.6 for e in prose)
    assert not any(e.src == e.dst for e in edges)


def test_unresolved_link_raises(tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntype: page\ntitle: A\nstatus: stable\ntimestamp: '2024-01-01'\n---\n\n"
        "See [broken](/nope.md) for more.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unresolved"):
        transcode(tmp_path)


def test_deterministic_and_hash_stable():
    a = transcode(CORPUS)
    b = transcode(CORPUS)
    assert [d.path for d in a[0]] == [d.path for d in b[0]]
    assert [(e.src, e.dst, e.kind) for e in a[1]] == [(e.src, e.dst, e.kind) for e in b[1]]
    assert corpus_sha256(CORPUS) == corpus_sha256(CORPUS)
    assert len(corpus_sha256(CORPUS)) == 64
