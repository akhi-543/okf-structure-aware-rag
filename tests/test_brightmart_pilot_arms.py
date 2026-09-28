import inspect

from okf_rag.retrieve import brightmart_pilot_arms_v1 as m


def test_metadata_pseudo_docs_render_frontmatter():
    docs = [{"path": "a.md", "title": "Brightmart Riverside", "description": "Profile.",
             "tags": ["store", "S01"], "frontmatter": {"store_format": "supercenter", "has_pharmacy": True,
                                                       "departments": ["Grocery", "Bakery"]}}]
    out = m.metadata_pseudo_docs(docs)
    assert out == [{"chunk_id": "meta:a.md", "doc_path": "a.md", "text": out[0]["text"]}]
    t = out[0]["text"]
    for s in ("Brightmart Riverside", "Profile.", "store", "store_format: supercenter",
              "has_pharmacy: True", "departments: Grocery, Bakery"):
        assert s in t


def test_collapse_to_docs_first_occurrence_and_cut():
    assert m.collapse_to_docs(["a", "a", "b", "c", "b", "d"], 3) == ["a", "b", "c"]


def test_doc_metrics_keys_and_values():
    r = m.doc_metrics(["x", "g1", "g2"], {"g1", "g2"})
    assert set(r) == {"P@1", "P@5", "P@10", "R@5", "R@10", "MRR@10", "nDCG@10"}
    assert r["P@1"] == 0.0 and r["R@5"] == 1.0 and r["MRR@10"] == 0.5


def _scores(s1_delta, s34_delta, s34p_delta, n=20, arms=m.STRUCTURE_ARMS):
    """R0 fixed at 0.5 with small noise; each arm = R0 + delta (+ same noise)."""
    import random
    rnd = random.Random(1)
    out = {}
    for set_name, delta, size in (("S1_t", s1_delta, n), ("S34_t", s34_delta, 2 * n), ("S34_p", s34p_delta, 2 * n)):
        base = {f"q{i}": 0.5 + rnd.uniform(-0.05, 0.05) for i in range(size)}
        out[set_name] = {"R0": base}
        for a in arms:
            # arms without a delta equal R0 exactly (no noise), so "no effect" is deterministic
            out[set_name][a] = {q: v + delta[a] + rnd.uniform(-0.01, 0.01) if a in delta else v
                                for q, v in base.items()}
    return out


def test_verdict_all_pass():
    v = m.verdict(_scores({}, {"R2": 0.3}, {"R2": 0.25}))
    assert v["C1"]["pass"] and v["C2"]["pass"] and v["C3"]["pass"]
    assert v["C2"]["winning_arms"] == ["R2"]


def test_verdict_c1_fails_when_one_arm_far_ahead_on_s1():
    v = m.verdict(_scores({"R4": 0.3}, {"R2": 0.3}, {"R2": 0.3}))
    assert not v["C1"]["pass"]


def test_verdict_c2_fail_forces_c3_fail():
    v = m.verdict(_scores({}, {}, {"R2": 0.3}))
    assert not v["C2"]["pass"]
    assert v["C3"]["pass"] is False and v["C3"]["reason"] == "criterion 2 failed"


def test_verdict_c3_ratio_rule():
    v = m.verdict(_scores({}, {"R2": 0.3}, {"R2": 0.1}))
    assert v["C2"]["pass"] and not v["C3"]["pass"]


def test_verdict_records_thresholds():
    v = m.verdict(_scores({}, {"R2": 0.3}, {"R2": 0.3}))
    assert v["thresholds"] == m.THRESHOLDS


def test_no_gold_parameter_in_nonoracle_helpers():
    for fn in (m.metadata_pseudo_docs, m.collapse_to_docs):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"gold", "gold_sql", "qrels", "expected_doc_paths"}


def test_score_runs_builds_verdict_sets():
    from scripts.run_brightmart_pilot_v1 import score_runs
    qs = [{"query_id": "bm-s1-01-t", "stratum": "S1"}, {"query_id": "bm-s3-01-t", "stratum": "S3"},
          {"query_id": "bm-s3-01-p", "stratum": "S3"}, {"query_id": "bm-s5-01-t", "stratum": "S5"}]
    qrels = {"bm-s1-01-t": {"a"}, "bm-s3-01-t": {"a", "b"}, "bm-s3-01-p": {"a", "b"}}
    runs = {"R0": {q["query_id"]: ["a", "b"] for q in qs}}
    rows, scores = score_runs(runs, qs, qrels)
    assert {r["query_id"] for r in rows} == {"bm-s1-01-t", "bm-s3-01-t", "bm-s3-01-p"}  # S5 skipped
    assert scores["S1_t"]["R0"]["bm-s1-01-t"] == 1.0
    assert set(scores["S34_t"]["R0"]) == {"bm-s3-01-t"}
    assert set(scores["S34_p"]["R0"]) == {"bm-s3-01-p"}


def test_score_runs_paraphrase_pairing_uses_base_id():
    from scripts.run_brightmart_pilot_v1 import score_runs
    qs = [{"query_id": "bm-s4-02-t", "stratum": "S4"}, {"query_id": "bm-s4-02-p", "stratum": "S4"}]
    qrels = {q["query_id"]: {"x"} for q in qs}
    _, scores = score_runs({"R0": {q["query_id"]: ["x"] for q in qs}}, qs, qrels)
    assert list(scores["S34_t"]["R0"]) == ["bm-s4-02-t"]


def test_oracle_child_dir_regions():
    from scripts.run_brightmart_pilot_v1 import _oracle_child_dir
    assert _oracle_child_dir("regions/region-northeast.md") == "regions/northeast/"


def test_oracle_child_dir_departments():
    from scripts.run_brightmart_pilot_v1 import _oracle_child_dir
    assert _oracle_child_dir("departments/dept-home-garden.md") == "departments/home-garden/"


def test_oracle_child_dir_grocery():
    from scripts.run_brightmart_pilot_v1 import _oracle_child_dir
    assert _oracle_child_dir("departments/dept-grocery.md") == "departments/grocery/"


def test_build_manifest_from_embedding_manifest():
    from scripts.run_brightmart_pilot_v1 import build_manifest
    emb_manifest = {
        "model_id": "BAAI/bge-base-en-v1.5",
        "revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        "query_instruction_prefix": "Represent this sentence for searching relevant passages: ",
        "rows": {"emb_flat.parquet": 300, "emb_struct.parquet": 320, "queries.parquet": 200},
    }
    input_hashes = {
        "doc_synthetic/brightmart_questions.jsonl": "abc123",
        "doc_synthetic/brightmart_qrels.jsonl": "def456",
        "doc_synthetic/brightmart_gold_spans.jsonl": "ghi789",
        "results/brightmart-pilot-v1/embeddings/queries.parquet": "jkl012",
    }
    manifest = build_manifest(
        git_head="1234567890abcdef",
        n_docs=100,
        n_chunks_flat=250,
        n_chunks_struct=300,
        n_edges=150,
        emb_manifest=emb_manifest,
        input_hashes=input_hashes,
        arms=["R0", "R1", "R2", "R3a", "R4", "C_flat_hybrid", "C_hier_oracle"],
    )
    assert manifest["rows"]["n_queries"] == 200
    assert manifest["bundle"] == "synthetic_retail_pilot"
    assert manifest["embedding"]["model_id"] == "BAAI/bge-base-en-v1.5"
    assert manifest["embedding"]["revision"] == "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    assert manifest["embedding"]["query_instruction_prefix"] == "Represent this sentence for searching relevant passages: "
    assert manifest["inputs"] == input_hashes
    assert manifest["k_chunks"] == 50
    assert manifest["k_docs"] == 10
    assert manifest["graph"] == {"k": 10, "seed_depth": 5, "hops": 1, "expansion_budget": 5}
    assert manifest["rows"]["n_docs"] == 100
    assert manifest["rows"]["n_chunks_flat"] == 250
    assert manifest["rows"]["n_chunks_struct"] == 300
    assert manifest["rows"]["n_chunks"] == 550
    assert manifest["rows"]["n_edges"] == 150
    assert set(manifest["arms"]) == {"R0", "R1", "R2", "R3a", "R4", "C_flat_hybrid", "C_hier_oracle"}
    assert manifest["structure_arms"] == ["R1", "R2", "R3a", "R4"]
    assert manifest["oracle_arms"] == ["C_hier_oracle"]


def test_verdict_md_title_uses_out_dir_name():
    from scripts.run_brightmart_pilot_v1 import _verdict_md
    v = m.verdict(_scores({}, {"R2": 0.3}, {"R2": 0.3}))
    md = _verdict_md(v, "brightmart-pilot-v2")
    assert md.splitlines()[0] == "# Brightmart pilot - verdict (brightmart-pilot-v2)"


def test_to_float32_pgvector():
    from pgvector import Vector
    from scripts.run_brightmart_pilot_v1 import _to_float32
    v = Vector([0.1, 0.2, 0.3])
    result = _to_float32(v)
    assert result.dtype == __import__("numpy").float32
    assert result.shape == (3,)
    assert __import__("numpy").allclose(result, [0.1, 0.2, 0.3], atol=1e-6)


def test_to_float32_list():
    from scripts.run_brightmart_pilot_v1 import _to_float32
    result = _to_float32([0.1, 0.2, 0.3])
    assert result.dtype == __import__("numpy").float32
    assert result.shape == (3,)
    assert __import__("numpy").allclose(result, [0.1, 0.2, 0.3], atol=1e-6)


def test_to_float32_ndarray():
    from scripts.run_brightmart_pilot_v1 import _to_float32
    import numpy as np
    arr = np.array([0.1, 0.2, 0.3])
    result = _to_float32(arr)
    assert result.dtype == np.float32
    assert result.shape == (3,)
    assert np.allclose(result, [0.1, 0.2, 0.3], atol=1e-6)


def test_graph_arm_reaches_a_doc_r1_cuts_at_k_docs():
    """Regression for the R3a no-op bug: with GRAPH sized to the real-corpus
    settings (k=10, expansion_budget=5), a doc reachable by one authored edge
    from a top seed, but ranked past K_DOCS by R1, must surface in R3a."""
    from okf_rag.retrieve.brightmart_pilot_arms_v1 import collapse_to_docs
    from okf_rag.retrieve.control_arms_v4 import EdgeSet
    from scripts.run_brightmart_pilot_v1 import K_DOCS, _graph_arm

    struct_chunks = [{"chunk_id": f"c{i}", "doc_path": f"docs/d{i}.md"} for i in range(12)]
    r1_chunk_ids = [c["chunk_id"] for c in struct_chunks]
    r1_docs = collapse_to_docs([c["doc_path"] for c in struct_chunks], K_DOCS)
    edges = EdgeSet.from_triples("authored", [("docs/d0.md", "docs/d11.md")])

    r3a_docs = _graph_arm(r1_chunk_ids, struct_chunks, edges)

    assert "docs/d11.md" not in r1_docs  # R1 ranks it 12th, cut at K_DOCS=10
    assert "docs/d11.md" in r3a_docs  # graph arm reaches it via the authored edge
    assert r3a_docs != r1_docs


def test_graph_noop_guard():
    from scripts.run_brightmart_pilot_v1 import _graph_noop

    identical = {"R1": {"q1": ["a", "b"]}, "R3a": {"q1": ["a", "b"]}}
    different = {"R1": {"q1": ["a", "b"]}, "R3a": {"q1": ["b", "a"]}}
    assert _graph_noop(identical)
    assert not _graph_noop(different)
