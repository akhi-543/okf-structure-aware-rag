import numpy as np

from okf_rag.retrieve import brightmart_pilot_arms_v1 as v1
from okf_rag.retrieve import brightmart_pilot_arms_v3 as m


def test_normalize_plural_snake_and_case():
    assert m.normalize("Which Stores have_fuel? categories express hours has") == [
        "which", "store", "have", "fuel", "category", "express", "hour", "has"]


def test_pseudo_doc_renders_booleans_type_status():
    assert m.normalize("status") == ["statu"]
    docs = [{"path": "a.md", "title": "Brightmart Riverside", "description": "Profile.", "tags": ["store"],
             "okf_type": "store", "status": "stable",
             "frontmatter": {"has_fuel": True, "has_pharmacy": False, "opened_year": 2004,
                             "departments": ["Grocery", "Home & Garden"]}}]
    out = m.metadata_pseudo_docs_v3(docs)
    assert out[0]["chunk_id"] == "meta:a.md" and out[0]["doc_path"] == "a.md"
    t = out[0]["text"]
    for s in ("has fuel yes", "has pharmacy no", "opened year 2004", "type store", "status stable",
              "brightmart riverside", "home garden"):
        assert " ".join(m.normalize(s)) in t, s


def test_query_and_doc_normalized_the_same_way():
    assert " ".join(m.normalize("stores with fuel")) == "store with fuel"


def test_contextual_text():
    assert m.contextual_text("Grocery", ["Categories"], "- Dairy") == "Grocery > Categories\n- Dairy"
    assert m.contextual_text("Grocery", [], "Intro") == "Grocery\nIntro"


def test_undirected_neighbors():
    nb = m.undirected_neighbors([("a", "b"), ("c", "a")])
    assert nb["a"] == {"b", "c"} and nb["b"] == {"a"} and nb["c"] == {"a"}


def test_doc_graph_arm_returns_ten_and_reaches_neighbor():
    ranked = [f"d{i}" for i in range(20)]
    nb = m.undirected_neighbors([("d0", "d19"), ("d1", "d18")])
    out = m.doc_graph_arm(ranked, nb, k_docs=10, n_seeds=5)
    assert len(out) == 10
    assert out[:5] == ["d0", "d1", "d2", "d3", "d4"]
    assert out[5:7] == ["d19", "d18"]
    assert out[7:] == ["d5", "d6", "d7"]


def test_doc_graph_arm_no_duplicates_when_neighbor_is_a_seed():
    ranked = [f"d{i}" for i in range(12)]
    nb = m.undirected_neighbors([("d0", "d1")])
    out = m.doc_graph_arm(ranked, nb)
    assert len(out) == len(set(out)) == 10


def test_verdict_accepts_arms():
    base = {f"q{i}": 0.5 for i in range(20)}
    arms = m.STRUCTURE_ARMS_V3
    scores = {s: {"R0": dict(base), **{a: dict(base) for a in arms}} for s in ("S1_t", "S34_t", "S34_p")}
    v = v1.verdict(scores, arms=arms)
    assert set(v["C1"]["arms"]) == set(arms)
    assert v1.verdict({s: {"R0": dict(base), **{a: dict(base) for a in v1.STRUCTURE_ARMS}}
                       for s in ("S1_t", "S34_t", "S34_p")})["C1"]["arms"].keys() == set(v1.STRUCTURE_ARMS)


def _ctx():
    from okf_rag.retrieve.control_arms_v4 import BM25Index, EdgeSet
    from okf_rag.retrieve.brightmart_pilot_arms_v3 import metadata_pseudo_docs_v3, undirected_neighbors
    paths = [f"d{i}.md" for i in range(12)]
    rng = np.random.default_rng(0)
    def chunks():
        out = []
        for i, p in enumerate(paths):
            v = rng.normal(size=8).astype(np.float32)
            out.append({"chunk_id": str(i), "doc_path": p, "text": f"doc {i} text", "vector": v / np.linalg.norm(v)})
        return out
    flat, struct, ctx_struct = chunks(), chunks(), chunks()
    docs = [{"path": p, "title": f"Doc {i}", "description": "", "tags": [], "okf_type": "store",
             "status": "stable", "frontmatter": {"has_fuel": i % 2 == 0}} for i, p in enumerate(paths)]
    from scripts.run_brightmart_pilot_v1 import _precompute_dense_matrices
    edge_rows = [("d0.md", "d11.md")]
    return {
        "flat": flat, "struct": struct, "struct_ctx": ctx_struct,
        "matrices": _precompute_dense_matrices({"flat": flat, "struct": struct, "struct_ctx": ctx_struct}),
        "meta_v2": BM25Index.build([{"chunk_id": f"meta:{d['path']}", "doc_path": d["path"], "text": d["title"]} for d in docs]),
        "meta_v3": BM25Index.build(metadata_pseudo_docs_v3(docs)),
        "flat_bm25": BM25Index.build(flat),
        "edges": EdgeSet.from_triples("authored", edge_rows),
        "neighbors": undirected_neighbors(edge_rows),
    }


def test_v3_arms_shapes():
    from scripts.run_brightmart_pilot_v3 import v3_arms
    ctx = _ctx()
    q = ctx["struct_ctx"][0]["vector"]
    out = v3_arms("doc with fuel", q, ctx)
    assert set(out) == {"R0", "R1", "R2", "R3a", "R4", "C_flat_hybrid", "R1c", "R2s", "R3d", "R4c"}
    assert len(out["R3d"]) == 10 and len(set(out["R3d"])) == 10
    assert all(len(v) <= 10 for v in out.values())


def test_v3_arms_r3d_uses_graph():
    from scripts.run_brightmart_pilot_v3 import v3_arms
    ctx = _ctx()
    q = ctx["struct_ctx"][0]["vector"]  # makes d0 the top R1c doc
    out = v3_arms("x", q, ctx)
    assert out["R3d"][0] == "d0.md" and "d11.md" in out["R3d"][:6]


def test_ctx_vectors_fresh_matches_pairs():
    from scripts.run_brightmart_pilot_v3 import _ctx_vectors_fresh
    db_pairs = [(1, 100), (2, 100), (3, 101)]
    assert _ctx_vectors_fresh(db_pairs, db_pairs) is True
    assert _ctx_vectors_fresh(db_pairs, db_pairs[:-1]) is False
    assert _ctx_vectors_fresh(db_pairs, db_pairs + [(4, 102)]) is False
    assert _ctx_vectors_fresh(db_pairs, [(1, 100), (2, 100), (3, 999)]) is False


def test_annotate_v3_manifest_sets_v3_and_v2_arms():
    from scripts.run_brightmart_pilot_v3 import _annotate_v3_manifest
    manifest = {"structure_arms": ["R1", "R2", "R3a", "R4"], "other": "kept"}
    out = _annotate_v3_manifest(manifest)
    assert out is manifest
    assert out["structure_arms"] == ["R1c", "R2s", "R3d", "R4c"]
    assert out["reference_arms_v2"] == ["R1", "R2", "R3a", "R4"]
    assert out["other"] == "kept"


def test_metadata_doc_dict_moves_tags_out_of_frontmatter():
    d = m.metadata_doc_dict("a.md", "T", "D", {"tags": ["x"], "region": "West"}, "store", "stable")
    assert d == {"path": "a.md", "title": "T", "description": "D", "tags": ["x"], "okf_type": "store",
                 "status": "stable", "frontmatter": {"region": "West"}}
    assert m.metadata_doc_dict("b.md", "T", None, None, "store", "stable")["tags"] == []


def test_structured_rank_matches_previous_inline_ranking():
    from okf_rag.retrieve.brightmart_pilot_arms_v1 import collapse_to_docs
    from okf_rag.retrieve.control_arms_v4 import BM25Index
    docs = [m.metadata_doc_dict(f"d{i}.md", f"Store {i}", "", {"region": r, "has_fuel": i % 2 == 0},
                                "store", "stable")
            for i, r in enumerate(["West", "East", "West", "North", "West"])]
    index = BM25Index.build(m.metadata_pseudo_docs_v3(docs))
    q = "Which West stores have fuel?"
    old = collapse_to_docs([p.split("meta:", 1)[1] for p, _ in index.search(" ".join(m.normalize(q)), 50)], 3)
    new = m.structured_rank(index, q, 3, 50)
    assert [p for p, _ in new] == old
    assert all(isinstance(s, float) for _, s in new)
