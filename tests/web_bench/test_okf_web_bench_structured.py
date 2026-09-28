"""Structured lane of the web bench: metadata-aware ranking + evidence ordering."""
from okf_web_bench.okf_web_bench_models import BgeQueryEncoder
from okf_web_bench.okf_web_bench_store import WebBenchStore, order_structured_evidence


def test_order_structured_evidence_lead_sections_first():
    """Every ranked document's first section comes before any second section, in
    rank order; within a document, sections keep reading order."""
    rows = [
        (12, 2, "b.md", ["B", "Two"], "b2", 1),
        (11, 2, "b.md", [], "b1", 0),
        (13, 2, "b.md", ["B", "Three"], "b3", 2),
        (22, 1, "a.md", ["A", "Two"], "a2", 1),
        (21, 1, "a.md", ["A"], "a1", 0),
        (99, 9, "unranked.md", [], "x", 0),
    ]
    out = order_structured_evidence([("b.md", 3.5), ("a.md", 1.25)], rows)
    assert [e["text"] for e in out] == ["b1", "a1", "b2", "a2", "b3"]
    assert [e["score"] for e in out] == [3.5, 1.25, 3.5, 1.25, 3.5]
    assert out[2]["heading_path"] == ["B", "Two"] and out[0]["heading_path"] == []
    assert order_structured_evidence([("missing.md", 1.0)], rows) == []


class _FakeCursor:
    """Answers the two SELECTs the structured lane issues."""

    def __init__(self, docs, chunks):
        self._docs, self._chunks, self._rows = docs, chunks, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        assert sql.lstrip().upper().startswith("SELECT")
        if "FROM documents" in sql and "frontmatter" in sql:
            self._rows = list(self._docs)
        else:
            wanted = set(params[1])
            self._rows = [r for r in self._chunks if r[2] in wanted]

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, docs, chunks):
        self._docs, self._chunks = docs, chunks

    def cursor(self):
        return _FakeCursor(self._docs, self._chunks)

    def close(self):
        pass


def test_structured_retrieve_ranks_by_metadata_and_returns_struct_chunks():
    docs = [  # (path, title, description, frontmatter, okf_type, status)
        ("regions/west/store-s10.md", "Brightmart Cedar Park", "Store.",
         {"region": "West", "has_fuel": True, "tags": ["store"]}, "store", "stable"),
        ("regions/east/store-s02.md", "Brightmart Riverdale", "Store.",
         {"region": "Northeast", "has_fuel": False, "tags": ["store"]}, "store", "stable"),
        ("suppliers/supplier-x.md", "Kestrel Electronics", "Supplier.",
         {"country": "Taiwan", "tags": ["supplier"]}, "supplier", "stable"),
    ]
    chunks = [  # (chunk_id, doc_id, path, heading_path, text, ord)
        (1, 10, "regions/west/store-s10.md", [], "Cedar Park intro", 0),
        (2, 10, "regions/west/store-s10.md", ["Departments"], "Cedar Park departments", 1),
        (3, 20, "regions/east/store-s02.md", [], "Riverdale intro", 0),
        (4, 30, "suppliers/supplier-x.md", [], "Kestrel intro", 0),
    ]
    store = WebBenchStore(lambda: _FakeConn(docs, chunks), BgeQueryEncoder(available=False))
    out = store.retrieve("synthetic_retail_pilot", "structured", "Which West stores have fuel?", 2)
    assert out[0]["doc_path"] == "regions/west/store-s10.md"
    assert out[0]["text"] == "Cedar Park intro"  # lead section of the top-ranked document first
    assert "Cedar Park departments" in [e["text"] for e in out]
    assert {e["doc_path"] for e in out} <= {d[0] for d in docs}
    assert len({e["doc_path"] for e in out}) <= 2


def test_structured_retrieve_needs_no_encoder():
    """The structured lane never embeds the question, so it works with bge down."""
    store = WebBenchStore(lambda: _FakeConn([], []), BgeQueryEncoder(available=False))
    assert store.retrieve("synthetic_retail_pilot", "structured", "anything", 5) == []
