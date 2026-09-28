"""Tests for the Brightmart v4 LLM prompt/parse helpers and T7/T8 answer scoring (no models)."""
import pytest

from okf_rag.eval import answer_scoring_v4 as S
from okf_rag.jobs import brightmart_v4_llm as L

TITLES = {"Northeast": "regions/region-northeast.md", "Brightmart Riverside": "regions/northeast/store-s001.md",
          "Grocery": "departments/dept-grocery.md"}


def test_parse_graph_output_maps_titles_and_salvages_truncation():
    raw = '{"parent": "Northeast", "mentions": ["Grocery", "Unknown Page", "Brightmart Riv'
    got = L.parse_graph_output(raw, "regions/northeast/store-s001.md", TITLES)
    assert {"src": "regions/region-northeast.md", "dst": "regions/northeast/store-s001.md", "kind": "child"} in got["edges"]
    assert {"src": "regions/northeast/store-s001.md", "dst": "departments/dept-grocery.md", "kind": "prose"} in got["edges"]
    assert got["unknown_titles"] == ["Unknown Page"]
    assert got["parse_ok"]


def test_parse_graph_output_never_links_a_page_to_itself():
    raw = '{"parent": null, "mentions": ["brightmart riverside"]}'
    assert L.parse_graph_output(raw, "regions/northeast/store-s001.md", TITLES)["edges"] == []


def _meta():
    return {
        "s1.md": {"title": "S1", "type": "store", "status": "stable",
                  "fm": {"region": "North", "store_format": "express", "opened_year": 2001, "has_fuel": True,
                         "tags": ["x"], "parent": "/r.md"}},
        "p1.md": {"title": "P1", "type": "promotion", "status": "stable",
                  "fm": {"formats": ["express"], "start": "2025-01-01", "end": "2025-01-31", "discount_pct": 10}},
    }


def test_filter_schema_and_prompt():
    schema = L.filter_schema(_meta())
    assert set(schema) == {"store", "promotion"}
    assert "tags" not in schema["store"] and "parent" not in schema["store"]
    assert schema["store"]["opened_year"] == "integer"
    msgs = L.filter_prompt("Which stores opened after 2000?", schema)
    assert "Which stores opened after 2000?" in msgs[-1]["content"]


def test_parse_filter_output_validates_and_coerces():
    kinds = L.schema_kinds(_meta())
    raw = ('Sure: {"type": "store", "filters": [{"field": "opened_year", "op": ">", "value": "2,000"}, '
           '{"field": "has_fuel", "op": "==", "value": "yes"}, {"field": "bogus", "op": "==", "value": 1}, '
           '{"field": "region", "op": "in", "value": "North"}, {"field": "sq_ft", "op": "~", "value": 3}]}')
    got = L.parse_filter_output(raw, kinds)
    assert got == {"type": "store", "filters": [
        {"field": "opened_year", "op": ">", "value": 2000},
        {"field": "has_fuel", "op": "==", "value": True},
        {"field": "region", "op": "in", "value": ["North"]}]}
    assert L.parse_filter_output("no json here", kinds) == {"type": None, "filters": []}
    assert L.parse_filter_output('{"type": "planet", "filters": []}', kinds)["type"] is None
    months = L.parse_filter_output('{"type": "promotion", "filters": [{"field": "x", "op": "month_overlap", "value": "12"}]}', kinds)
    assert months["filters"] == [{"field": "start", "op": "month_overlap", "value": 12}]


def test_answer_and_minicheck_prompts():
    msgs = L.answer_prompt("Q?", "[1] a.md\nfact")
    assert "NOT FOUND" in msgs[0]["content"] and "Q?" in msgs[1]["content"]
    assert L.minicheck_prompt("doc", "claim")[1]["content"] == "Document: doc\nClaim: claim"


@pytest.mark.parametrize("answer,reference,want", [
    ("It covers 182,000 square feet [1].", "182,000 square feet", 1.0),
    ("About 18,200 square feet.", "182,000 square feet", 0.0),
    ("You must spend $100.", "$100", 1.0),
    ("It runs from 2025-03-15 to 2025-04-15.", "2025-03-15 to 2025-04-15", 1.0),
    ("It starts 2025-03-15.", "2025-03-15 to 2025-04-15", 0.0),
    ("The discount is 20 percent.", "20%", 1.0),
    ("Q2 sales were $28,702.49.", "28,702.49", 1.0),
    ("NOT FOUND", "Albany", 0.0),
])
def test_fact_correct(answer, reference, want):
    assert S.fact_correct(answer, reference) == want


def test_set_f1_uses_aliases_and_penalizes_extras():
    cands = {"Brightmart Riverside": "Brightmart Riverside", "Riverside": "Brightmart Riverside",
             "Brightmart Riverdale": "Brightmart Riverdale", "Riverdale": "Brightmart Riverdale",
             "Brightmart Oak Hollow": "Brightmart Oak Hollow", "Oak Hollow": "Brightmart Oak Hollow"}
    gold = ["Brightmart Riverside", "Brightmart Oak Hollow"]
    assert S.set_f1("Riverside and Oak Hollow [1][2].", gold, cands) == pytest.approx(1.0)
    assert S.set_f1("Riverside, Riverdale.", gold, cands) == pytest.approx(0.5)
    assert S.set_f1("NOT FOUND", gold, cands) == 0.0


def test_faithful_and_claims():
    assert S.faithful(None, "NOT FOUND", 0.5) == 1.0
    assert S.faithful(0.7, "Yes [1].", 0.5) == 1.0
    assert S.faithful(0.2, "Yes.", 0.5) == 0.0
    assert S.claim_sentences("A is big [1]. B is small [2].") == ["A is big.", "B is small."]


def test_t7_verdict_pass_logic():
    rows = []
    for i in range(30):
        st = ["S3", "S4", "S5"][i % 3]
        for arm, good in (("R0", 0.0), ("R2s", 1.0)):
            rows.append({"query_id": f"q{i}", "arm": arm, "stratum": st, "correct": good,
                         "faithful": good, "abstained": good if st == "S5" else 0.0})
    rows += [{"query_id": "f1", "arm": a, "stratum": "S1", "correct": 1.0, "faithful": 1.0, "abstained": 0.0}
             for a in ("R0", "R2s")]
    v = S.t7_verdict(rows, 0.5, n_boot=500)
    assert v["pass"] and v["S34_correct_F1"]["delta"] == pytest.approx(1.0)
    flipped = [{**r, "correct": 1.0 - r["correct"]} if r["stratum"] in ("S3", "S4") else r for r in rows]
    assert not S.t7_verdict(flipped, 0.5, n_boot=500)["pass"]


def test_cohen_kappa():
    assert S.cohen_kappa(list("yyyynnnn"), list("yyyynnnn")) == pytest.approx(1.0)
    assert S.cohen_kappa(list("yyyynnnn"), list("yynnyynn")) == pytest.approx(0.0)
    assert S.cohen_kappa(list("yyyy"), list("yyyy")) == 1.0
    with pytest.raises(ValueError):
        S.cohen_kappa([], [])
