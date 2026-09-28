from okf_web_bench.okf_web_bench_compare import (
    PACK_BUDGET,
    run_compare,
    run_lane,
    numbered_documentation,
)


class CharTok:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids).decode("utf-8")


def test_pack_budget_is_2048():
    assert PACK_BUDGET == 2048


def test_numbered_documentation():
    text = numbered_documentation([
        {"doc_path": "a.md", "text": "A"},
        {"doc_path": "b.md", "text": "B"},
    ])
    assert text == "[1] a.md\nA\n\n[2] b.md\nB"


def test_run_lane_clips_and_cites():
    long_a = "AAAA" + ("B" * 4000)

    def retrieve_fn(corpus, arm, question, k):
        return [
            {
                "chunk_id": 1,
                "doc_id": 10,
                "doc_path": "doc.md",
                "text": long_a,
                "heading_path": ["H"],
                "score": 0.9,
            },
            {
                "chunk_id": 2,
                "doc_id": 11,
                "doc_path": "other.md",
                "text": "short",
                "heading_path": [],
                "score": 0.1,
            },
        ]

    out = run_lane(
        question="q",
        corpus="synthetic_retail_pilot",
        arm="flat",
        top_k=2,
        tokenizer=CharTok(),
        retrieve_fn=retrieve_fn,
        generate_fn=lambda q, d: "ans",
    )
    assert out["answer"] == "ans"
    assert out["error"] is None
    assert out["metrics"]["clipped"] is True
    assert out["sources"][0]["citation"] == 1
    assert out["sources"][0]["truncated"] is True
    assert out["metrics"]["tokens_in"] <= PACK_BUDGET
    assert all(s["citation"] == i for i, s in enumerate(out["sources"], 1))


def test_run_compare_isolates_lane_errors():
    def retrieve_fn(corpus, arm, question, k):
        if arm == "flat":
            raise RuntimeError("flat boom")
        return [
            {
                "chunk_id": 3,
                "doc_id": 1,
                "doc_path": "ok.md",
                "text": "hello",
                "heading_path": [],
                "score": 1.0,
            }
        ]

    out = run_compare(
        question="q",
        corpus="synthetic_retail_pilot",
        lane_a="flat",
        lane_b="structured",
        top_k=5,
        tokenizer=CharTok(),
        retrieve_fn=retrieve_fn,
        generate_fn=lambda q, d: "ok",
    )
    assert out["lane_a"]["error"] is not None
    assert out["lane_a"]["answer"] is None
    assert out["lane_b"]["answer"] == "ok"


def test_run_lane_isolates_generate_errors():
    def retrieve_fn(corpus, arm, question, k):
        return [
            {
                "chunk_id": 3,
                "doc_id": 1,
                "doc_path": "ok.md",
                "text": "hello",
                "heading_path": [],
                "score": 1.0,
            }
        ]

    def boom_generate(q, d):
        raise RuntimeError("generate boom")

    out = run_compare(
        question="q",
        corpus="synthetic_retail_pilot",
        lane_a="flat",
        lane_b="structured",
        top_k=1,
        tokenizer=CharTok(),
        retrieve_fn=retrieve_fn,
        generate_fn=boom_generate,
    )
    assert out["lane_a"]["error"] == "generate boom"
    assert out["lane_b"]["error"] == "generate boom"


def test_run_lane_isolates_pack_errors(monkeypatch):
    def retrieve_fn(corpus, arm, question, k):
        return [
            {
                "chunk_id": 1,
                "doc_id": 1,
                "doc_path": "ok.md",
                "text": "hello",
                "heading_path": [],
                "score": 1.0,
            }
        ]

    def bad_pack(*args, **kwargs):
        raise RuntimeError("pack boom")

    monkeypatch.setattr(
        "okf_web_bench.okf_web_bench_compare.pack_context", bad_pack
    )
    out = run_lane(
        question="q",
        corpus="synthetic_retail_pilot",
        arm="flat",
        top_k=1,
        tokenizer=CharTok(),
        retrieve_fn=retrieve_fn,
        generate_fn=lambda q, d: "ok",
    )
    assert out["error"] == "pack boom"


def test_run_lane_without_tokenizer_returns_unpacked():
    def retrieve_fn(corpus, arm, question, k):
        return [
            {
                "chunk_id": 3,
                "doc_id": 1,
                "doc_path": "ok.md",
                "text": "hello",
                "heading_path": [],
                "score": 1.0,
            }
        ]

    out = run_lane(
        question="q",
        corpus="synthetic_retail_pilot",
        arm="flat",
        top_k=1,
        tokenizer=None,
        retrieve_fn=retrieve_fn,
        generate_fn=None,
    )
    assert out["error"] == "packing tokenizer unavailable"
    assert out["sources"][0]["truncated"] is False
    assert out["metrics"]["tokens_in"] is None


def test_generator_down_still_returns_sources():
    def retrieve_fn(corpus, arm, question, k):
        return [
            {
                "chunk_id": 3,
                "doc_id": 1,
                "doc_path": "ok.md",
                "text": "hello",
                "heading_path": [],
                "score": 1.0,
            }
        ]

    out = run_lane(
        question="q",
        corpus="synthetic_retail_pilot",
        arm="structured",
        top_k=1,
        tokenizer=CharTok(),
        retrieve_fn=retrieve_fn,
        generate_fn=None,
    )
    assert out["answer"] is None
    assert out["error"] == "generator unavailable"
    assert out["sources"][0]["doc_path"] == "ok.md"


def test_generate_exception_keeps_sources():
    def retrieve_fn(corpus, arm, question, k):
        return [
            {
                "chunk_id": 3,
                "doc_id": 1,
                "doc_path": "ok.md",
                "text": "hello",
                "heading_path": [],
                "score": 1.0,
            }
        ]

    def generate_fn(question, documentation):
        raise RuntimeError("cuda oom")

    out = run_lane(
        question="q",
        corpus="synthetic_retail_pilot",
        arm="flat",
        top_k=1,
        tokenizer=CharTok(),
        retrieve_fn=retrieve_fn,
        generate_fn=generate_fn,
    )
    assert out["answer"] is None
    assert "cuda oom" in out["error"]
    assert out["sources"][0]["doc_path"] == "ok.md"


def test_rejects_unknown_corpus():
    import pytest

    with pytest.raises(ValueError):
        run_compare(
            question="q",
            corpus="cheat_cc0_98",
            lane_a="flat",
            lane_b="structured",
            top_k=1,
            tokenizer=CharTok(),
            retrieve_fn=lambda *a: [],
            generate_fn=None,
        )


class FakeStore:
    def retrieve(self, corpus, arm, question, k):
        return [
            {
                "chunk_id": 1,
                "doc_id": 1,
                "doc_path": "x.md",
                "text": "hello",
                "heading_path": [],
                "score": 0.5,
            }
        ]

    def available_arms(self):
        return ["flat", "structured"]

    def list_documents(self, corpus):
        return [
            {
                "path": "x.md",
                "title": "X",
                "n_flat": 1,
                "n_struct": 1,
                "identical": True,
            }
        ]

    def load_document_map(self, corpus, path):
        from okf_web_bench.okf_web_bench_corpus_map import map_document

        ch = {
            "chunk_id": 1,
            "ord": 0,
            "text": "hello",
            "heading_path": [],
            "n_tokens": 1,
        }
        return map_document(
            path=path,
            title="X",
            flat_chunks=[ch],
            struct_chunks=[ch],
            authored_edges=[],
        ) | {
            "overlap_ids": {"flat": {"1": [1]}, "struct": {"1": [1]}},
        }


def test_compare_503_when_postgres_down():
    from fastapi.testclient import TestClient

    from okf_web_bench.okf_web_bench_app import WebBenchRuntime, app, get_runtime

    rt = WebBenchRuntime(
        store=FakeStore(),
        encoder_available=True,
        qwen_available=False,
        tokenizer=CharTok(),
        generate_fn=None,
        cuda=False,
        cuda_index=None,
        postgres=False,
    )
    app.dependency_overrides[get_runtime] = lambda: rt
    client = TestClient(app)
    r = client.post(
        "/api/okf-web-bench/compare",
        json={
            "question": "q",
            "corpus": "synthetic_retail_pilot",
            "lane_a": "flat",
            "lane_b": "structured",
            "top_k": 5,
        },
    )
    assert r.status_code == 503
    assert r.json()["detail"] == "postgres unavailable"
    app.dependency_overrides.clear()


def test_compare_503_when_bge_down():
    from fastapi.testclient import TestClient

    from okf_web_bench.okf_web_bench_app import WebBenchRuntime, app, get_runtime

    rt = WebBenchRuntime(
        store=FakeStore(),
        encoder_available=False,
        qwen_available=False,
        tokenizer=CharTok(),
        generate_fn=None,
        cuda=False,
        cuda_index=None,
        postgres=True,
    )
    app.dependency_overrides[get_runtime] = lambda: rt
    client = TestClient(app)
    r = client.post(
        "/api/okf-web-bench/compare",
        json={
            "question": "q",
            "corpus": "synthetic_retail_pilot",
            "lane_a": "flat",
            "lane_b": "structured",
            "top_k": 5,
        },
    )
    assert r.status_code == 503
    assert r.json()["detail"] == "bge unavailable"
    app.dependency_overrides.clear()


def test_corpus_map_503_when_postgres_down():
    from fastapi.testclient import TestClient

    from okf_web_bench.okf_web_bench_app import WebBenchRuntime, app, get_runtime

    rt = WebBenchRuntime(
        store=FakeStore(),
        encoder_available=True,
        qwen_available=False,
        tokenizer=CharTok(),
        generate_fn=None,
        cuda=False,
        cuda_index=None,
        postgres=False,
    )
    app.dependency_overrides[get_runtime] = lambda: rt
    client = TestClient(app)
    r = client.get(
        "/api/okf-web-bench/corpus-map", params={"corpus": "synthetic_retail_pilot"}
    )
    assert r.status_code == 503
    r = client.get(
        "/api/okf-web-bench/corpus-map/doc",
        params={"corpus": "synthetic_retail_pilot", "path": "x.md"},
    )
    assert r.status_code == 503
    app.dependency_overrides.clear()


def test_health_and_compare_endpoints():
    from fastapi.testclient import TestClient

    from okf_web_bench.okf_web_bench_app import WebBenchRuntime, app, get_runtime

    rt = WebBenchRuntime(
        store=FakeStore(),
        encoder_available=True,
        qwen_available=False,
        tokenizer=CharTok(),
        generate_fn=None,
        cuda=False,
        cuda_index=None,
        postgres=True,
    )
    app.dependency_overrides[get_runtime] = lambda: rt
    client = TestClient(app)
    h = client.get("/api/okf-web-bench/health").json()
    assert h["postgres"] is True
    assert h["bge"] is True
    assert h["qwen"] is False
    assert h["device"] == "cpu"
    assert h["arms"] == ["flat", "structured"]
    arms = client.get("/api/okf-web-bench/arms").json()
    assert arms["arms"] == ["flat", "structured"]
    r = client.post(
        "/api/okf-web-bench/compare",
        json={
            "question": "q",
            "corpus": "synthetic_retail_pilot",
            "lane_a": "flat",
            "lane_b": "structured",
            "top_k": 5,
            "gpu": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["lane_a"]["error"] == "generator unavailable"
    assert body["lane_a"]["sources"][0]["doc_path"] == "x.md"
    docs = client.get(
        "/api/okf-web-bench/corpus-map", params={"corpus": "synthetic_retail_pilot"}
    ).json()
    assert docs["documents"][0]["identical"] is True
    app.dependency_overrides.clear()


def test_console_has_tabs_and_no_stub_fill():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    html = (
        root / "okf_web_bench/okf_web_bench_static/okf_web_bench_console.html"
    ).read_text(encoding="utf-8")
    js = (
        root / "okf_web_bench/okf_web_bench_static/okf_web_bench_console.js"
    ).read_text(encoding="utf-8")
    assert "Corpus map" in html
    assert 'id="modeA"' in html
    assert "Ask once, answer twice." in html
    assert "Qwen3-Embedding-0.6B" in html
    assert "superior" not in html.lower()
    assert "superior" not in js.lower()
    assert "/api/okf-web-bench/compare" in js
    assert "textsOverlap" not in js
    assert "populateLaneSelects" in js
    assert "setTimeout" not in js
    assert "fill(t" not in js
    assert js.count("function switchTab(") == 1
    assert "async function loadCorpusDocuments(" in js
    assert "function switchTab(tab) {\n  const corpus" not in js


def test_root_serves_console_html():
    from fastapi.testclient import TestClient

    from okf_web_bench.okf_web_bench_app import WebBenchRuntime, app, get_runtime

    rt = WebBenchRuntime(
        store=FakeStore(),
        encoder_available=True,
        qwen_available=False,
        tokenizer=CharTok(),
        generate_fn=None,
        cuda=False,
        cuda_index=None,
        postgres=True,
    )
    app.dependency_overrides[get_runtime] = lambda: rt
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "Corpus map" in r.text
    app.dependency_overrides.clear()
