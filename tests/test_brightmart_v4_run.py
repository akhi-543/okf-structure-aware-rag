"""Tests for the pure parts of scripts/run_brightmart_v4.py, the T8 review script and the
v4 ingest bundle guard. No Postgres, no models."""
from collections import defaultdict

import numpy as np
import pytest

from scripts.ingest_corpus import _resolve_bundle_name
from scripts.review_brightmart_v4 import kappas, sample_rows
from scripts.run_brightmart_v4 import NOPARENT, STANDARD, cells_from_rows, edge_agreement, oracle_rank, t7_passages


@pytest.mark.parametrize("corpus,bundle", [("brightmart_v4", "synthetic_retail_v4"),
                                           ("brightmart_v4_noparent", "synthetic_retail_v4_noparent")])
def test_v4_bundles_are_fixed(corpus, bundle):
    assert _resolve_bundle_name(corpus, bundle, False, False) == bundle
    with pytest.raises(ValueError):
        _resolve_bundle_name(corpus, "synthetic_retail_pilot", False, False)
    with pytest.raises(ValueError):
        _resolve_bundle_name(corpus, bundle, False, True)


def _row(bundle, arm, qid, stratum, cond="", v=0.5):
    return {"bundle": bundle, "arm": arm, "query_id": qid, "stratum": stratum,
            "set": "p" if qid.endswith("-p") else "t", "condition": cond, "nDCG@10": v}


def test_cells_route_rows_to_the_preregistered_cells():
    rows = [_row(STANDARD, "R0", "a-t", "S3", "numeric"), _row(STANDARD, "R0", "a-p", "S3", "numeric"),
            _row(STANDARD, "R0", "b-t", "S2"), _row(STANDARD, "R0", "c-t", "S1"),
            _row(NOPARENT, "R5p", "d-t", "S4"), _row(NOPARENT, "R5p", "e-t", "S3")]
    cells = cells_from_rows(rows)
    assert set(cells["S34_t"]["R0"]) == {"a-t"} and set(cells["S34_p"]["R0"]) == {"a-p"}
    assert set(cells["S3num_t"]["R0"]) == {"a-t"} and set(cells["S2_t"]["R0"]) == {"b-t"}
    assert set(cells["S1to4_t"]["R0"]) == {"a-t", "b-t", "c-t"}
    assert cells["S4_np_t"] == {"R5p": {"d-t": 0.5}}


def test_edge_agreement_is_undirected():
    authored = [("a", "b", "child"), ("b", "c", "prose")]
    extracted = [("b", "a", "prose"), ("a", "c", "prose"), ("c", "c", "prose")]
    got = edge_agreement(authored, extracted)
    assert got["shared_pairs"] == 1 and got["precision"] == 0.5 and got["recall"] == 0.5


def test_oracle_rank_lists_children_of_each_gold_parent():
    paths = ["regions/north/store-a.md", "regions/north/store-b.md", "regions/south/store-c.md", "regions/region-north.md"]
    spans = [{"path": ["brightmart-home.md", "regions/region-north.md"]}]
    assert oracle_rank(spans, paths) == ["regions/north/store-a.md", "regions/north/store-b.md"]
    assert oracle_rank([{"doc_path": "x"}], paths) is None


def test_t7_passages_orders_structured_evidence_lead_sections_first():
    struct = [{"doc_path": "d1", "chunk_id": 1, "heading_path": [], "text": "d1 lead", "ord": 0},
              {"doc_path": "d1", "chunk_id": 2, "heading_path": [], "text": "d1 more", "ord": 1},
              {"doc_path": "d2", "chunk_id": 3, "heading_path": [], "text": "d2 lead", "ord": 0}]
    idx = defaultdict(list)
    for i, c in enumerate(struct):
        idx[c["doc_path"]].append(i)
    flat = [{"doc_path": "d9", "chunk_id": 9, "text": "flat"}, {"doc_path": "d8", "chunk_id": 8, "text": "flat2"}]
    b = {"struct": struct, "struct_doc_idx": idx, "flat": flat, "mat_flat": np.array([[1.0, 0.0], [0.0, 1.0]])}
    assert [p["text"] for p in t7_passages(b, "R2s", ["d1", "d2"], np.array([1.0, 0.0]))] == ["d1 lead", "d2 lead", "d1 more"]
    assert [p["doc_path"] for p in t7_passages(b, "R0", [], np.array([0.0, 1.0]))] == ["d8", "d9"]


def test_review_sample_is_stratified_and_deterministic():
    qs = [{"query_id": f"bm4-h-s{s}-{i:02d}-{t}", "stratum": f"S{s}", "text": "q", "reference_answer": None,
           "source_paths": []} for s in range(1, 6) for i in range(1, 41) for t in "tp"]
    rows = sample_rows(qs, {})
    assert len(rows) == 40 and all(r["query_id"].endswith("-t") for r in rows)
    assert {r["stratum"] for r in rows} == {"S1", "S2", "S3", "S4", "S5"}
    assert rows == sample_rows(qs, {})


def test_review_kappas_pass_threshold():
    a = [{"query_id": str(i), "gold_ok": "y", "R0_correct": "y" if i % 2 else "n", "R2s_correct": "y" if i % 3 else "n"}
         for i in range(12)]
    assert kappas(a, a)["pass"]
    b = [{**r, "R0_correct": "n" if r["R0_correct"] == "y" else "y"} for r in a]
    assert not kappas(a, b)["pass"]


def test_minicheck_memory_budget_leaves_headroom_on_every_gpu():
    from scripts.llm_brightmart_v4 import max_memory_map
    t4 = int(14.56 * 2**30)
    assert max_memory_map([t4, t4], 5) == {0: "9GiB", 1: "9GiB"}   # 18 GiB >= 15.5 GiB of fp16 weights
    assert max_memory_map([int(22.0 * 2**30)], 5) == {0: "17GiB"}   # one A10G (g5)


def test_last_token_logits_matches_full_forward():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from scripts.llm_brightmart_v4 import last_token_logits
    torch.manual_seed(0)
    cfg = AutoConfig.for_model("llama", num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                               hidden_size=16, intermediate_size=32, vocab_size=50, max_position_embeddings=32)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    assert hasattr(model, "model") and hasattr(model, "lm_head")  # exercises the body + head path
    ids = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    with torch.no_grad():
        full = model(**ids).logits[0, -1].float()
    assert torch.allclose(last_token_logits(model, ids), full, atol=1e-5)
