from okf_web_bench.okf_web_bench_corpus_map import (
    texts_overlap,
    identical_chunk_pair,
    overlap_ids,
    document_row,
    map_document,
)


def test_identical_badge():
    a = [{"chunk_id": 1, "text": "same"}]
    b = [{"chunk_id": 2, "text": "same"}]
    assert identical_chunk_pair(a, b) is True
    assert identical_chunk_pair(a, a + a) is False


def test_overlap_shared_span():
    sel = {"chunk_id": 1, "text": "alpha bravo charlie"}
    others = [
        {"chunk_id": 10, "text": "bravo charlie delta"},
        {"chunk_id": 11, "text": "zzzzzzzz"},
    ]
    assert overlap_ids(sel, others) == [10]


def test_document_row_counts():
    row = document_row("p.md", "P", [{"text": "a"}, {"text": "b"}], [{"text": "a"}])
    assert row["n_flat"] == 2
    assert row["n_struct"] == 1
    assert row["identical"] is False


def test_overlap_short_text_substring():
    sel = {"chunk_id": 1, "text": "ab"}
    others = [{"chunk_id": 2, "text": "abc"}]
    assert overlap_ids(sel, others) == [2]


def test_map_keeps_edges_separate():
    doc = map_document(
        path="p.md",
        title="P",
        flat_chunks=[
            {
                "chunk_id": 1,
                "ord": 0,
                "text": "hello world",
                "heading_path": [],
                "n_tokens": 2,
            }
        ],
        struct_chunks=[
            {
                "chunk_id": 2,
                "ord": 0,
                "text": "hello world extra",
                "heading_path": ["H"],
                "n_tokens": 3,
            }
        ],
        authored_edges=[
            {"src_path": "p.md", "dst_path": "q.md", "edge_kind": "related"}
        ],
    )
    assert doc["edges"][0]["edge_kind"] == "related"
    assert "related" not in doc["flat"][0]["text"]
