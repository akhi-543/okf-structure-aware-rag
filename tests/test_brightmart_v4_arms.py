"""Unit tests for okf_rag.retrieve.brightmart_v4_arms (pure; no DB, no models)."""
import pytest

from okf_rag.retrieve import brightmart_v4_arms as A


def _meta():
    stores = {
        "regions/north/store-a.md": {"region": "North", "store_format": "supercenter", "opened_year": 2001,
                                     "sq_ft": 180000, "has_pharmacy": True, "has_fuel": False, "store_status": "open"},
        "regions/north/store-b.md": {"region": "North", "store_format": "express", "opened_year": 2019,
                                     "sq_ft": 12000, "has_pharmacy": False, "has_fuel": True, "store_status": "open"},
        "regions/south/store-c.md": {"region": "South", "store_format": "express", "opened_year": 2010,
                                     "sq_ft": 15000, "has_pharmacy": True, "has_fuel": True, "store_status": "closed"},
    }
    meta = {p: {"title": f"Brightmart {p[-4].upper()}", "type": "store", "status": "stable", "fm": fm}
            for p, fm in stores.items()}
    meta["regions/region-north.md"] = {"title": "North", "type": "region", "status": "stable", "fm": {}}
    meta["regions/region-south.md"] = {"title": "South", "type": "region", "status": "stable", "fm": {}}
    meta["promotions/promo-x.md"] = {"title": "Promo X", "type": "promotion", "status": "stable",
                                     "fm": {"formats": ["supercenter"], "start": "2025-11-20", "end": "2025-12-05",
                                            "discount_pct": 25, "department": "Toys"}}
    meta["promotions/promo-y.md"] = {"title": "Promo Y", "type": "promotion", "status": "stable",
                                     "fm": {"formats": ["supercenter", "express"], "start": "2025-03-01",
                                            "end": "2025-03-10", "discount_pct": 10, "department": "Toys"}}
    meta["archive/old-a.md"] = {"title": "Brightmart A (2022 profile)", "type": "archive", "status": "deprecated",
                                "fm": dict(stores["regions/north/store-a.md"])}
    meta["brightmart-home.md"] = {"title": "Home", "type": "home", "status": "stable", "fm": {}}
    return meta


EDGES = [
    ("regions/region-north.md", "regions/north/store-a.md", "child"),
    ("regions/region-north.md", "regions/north/store-b.md", "child"),
    ("regions/region-south.md", "regions/south/store-c.md", "child"),
    ("regions/north/store-a.md", "promotions/promo-x.md", "prose"),
]


def test_metrics_include_recall50_and_f1():
    m = A.doc_metrics_v4(["a", "x", "b"], {"a", "b", "c", "d"})
    assert m["P@10"] == pytest.approx(0.2)
    assert m["R@10"] == pytest.approx(0.5)
    assert m["F1@10"] == pytest.approx(2 * 0.2 * 0.5 / 0.7)
    assert A.doc_metrics_v4(["x"], {"a"})["F1@10"] == 0.0


def test_heading_path_text_drops_page_title_only():
    assert A.heading_path_text("Store A", ["Store A", "Departments"], "body") == "Departments\nbody"
    assert A.heading_path_text("Store A", ["Store A"], "body") == "body"
    assert A.heading_path_text("Store A", [], "body") == "body"


@pytest.mark.parametrize("q,want", [
    ("Which stores belong to the North region?", "store"),
    ("Which Mountain region locations first opened later than 2012?", "store"),
    ("Which promotions run at supercenter stores?", "promotion"),
    ("What product categories does the Deli department cover?", "category"),
    ("Among our express stores, which ones operate without a pharmacy?", "store"),
    ("Where in the North region can customers fill a prescription?", "store"),
    ("Who is the regional manager responsible for Brightmart A?", None),
])
def test_target_type(q, want):
    assert A.target_type(q) == want


def test_title_matches_longest_first_and_excludes_home():
    meta = _meta()
    assert A.title_matches("Tell me about Brightmart A (2022 profile) please", meta)[0] == "archive/old-a.md"
    assert "brightmart-home.md" not in A.title_matches("home page", meta)


def test_weighted_expansion_adds_children_of_seeds():
    ranked = A.weighted_expansion([("regions/region-north.md", 4.0), ("promotions/promo-y.md", 0.4)], EDGES)
    assert ranked[:3] == ["regions/region-north.md", "regions/north/store-a.md", "regions/north/store-b.md"]
    assert A.weighted_expansion([], EDGES) == []


def test_parent_aware_returns_children_then_r2s():
    meta = _meta()
    children = A.children_by_parent(EDGES)
    r2s = ["regions/region-north.md", "promotions/promo-x.md"]
    out = A.parent_aware("Which stores belong to the North region?", {"regions/north/store-b.md": 2.0}, r2s, meta, children)
    assert out[:2] == ["regions/north/store-b.md", "regions/north/store-a.md"]
    assert out[2:] == r2s
    # not a list question for the parent's child type: plain R2s
    assert A.parent_aware("Who manages the North region?", {}, r2s, meta, children) == r2s


def test_parse_filters_numeric_boolean_categorical():
    meta = _meta()
    pf = A.parse_filters("Which North stores with a pharmacy opened before 2005?", meta)
    fields = {(f["field"], f["op"]) for f in pf["filters"]}
    assert pf["type"] == "store"
    assert {("region", "in"), ("has_pharmacy", "=="), ("opened_year", "<")} <= fields
    assert A.apply_filters(pf, meta) == ["regions/north/store-a.md"]  # the archived copy is type archive


def test_parse_filters_negation_between_and_squarefeet():
    meta = _meta()
    pf = A.parse_filters("Which stores are not supercenters and have neither a pharmacy nor a fuel station?", meta)
    assert {"field": "store_format", "op": "not_in", "value": ["supercenter"]} in pf["filters"]
    assert {"field": "has_pharmacy", "op": "==", "value": False} in pf["filters"]
    assert {"field": "has_fuel", "op": "==", "value": False} in pf["filters"]
    pf = A.parse_filters("Which stores are between 11,000 and 16,000 square feet?", meta)
    assert A.apply_filters(pf, meta) == ["regions/north/store-b.md", "regions/south/store-c.md"]
    pf = A.parse_filters("Which stores have been open since before 2015?", meta)
    assert all(f["field"] != "store_status" for f in pf["filters"])


def test_parse_filters_promotions_months_percent_only():
    meta = _meta()
    assert A.apply_filters(A.parse_filters("Which promotions run in December?", meta), meta) == ["promotions/promo-x.md"]
    assert A.apply_filters(A.parse_filters("Which promotions start in March?", meta), meta) == ["promotions/promo-y.md"]
    pf = A.parse_filters("Which promotions give 20% off or more?", meta)
    assert pf["filters"] == [{"field": "discount_pct", "op": ">=", "value": 20}]
    pf = A.parse_filters("Which promotions are limited to supercenter stores?", meta)
    assert A.apply_filters(pf, meta) == ["promotions/promo-x.md"]


def test_filter_rank_falls_back_to_r2s():
    meta = _meta()
    r2s = ["promotions/promo-y.md", "regions/north/store-a.md"]
    assert A.filter_rank({"type": "store", "filters": []}, meta, {}, r2s) == r2s
    assert A.filter_rank({"type": "store", "filters": [{"field": "sq_ft", "op": ">", "value": 10**7}]},
                         meta, {}, r2s) == r2s
    got = A.filter_rank({"type": "store", "filters": [{"field": "has_fuel", "op": "==", "value": True}]},
                        meta, {"regions/south/store-c.md": 1.0}, r2s)
    assert got[:2] == ["regions/south/store-c.md", "regions/north/store-b.md"]


def test_two_step_entity_then_neighbours_by_dense_score():
    meta = _meta()
    nb = A.undirected_adjacency(EDGES)
    got = A.two_step("What discount does Promo X give at Brightmart A?", meta, nb,
                     {"regions/region-north.md": 0.9}, ["fallback.md"])
    assert got[0] in ("promotions/promo-x.md", "regions/north/store-a.md")
    assert "regions/region-north.md" in got
    assert got[-1] == "fallback.md"
    assert A.two_step("nothing named here", meta, nb, {}, ["f.md"]) == ["f.md"]


def _cells(delta):
    base = {f"q{i}": 0.3 + 0.01 * (i % 5) for i in range(40)}
    better = {q: v + delta for q, v in base.items()}
    arms = {a: dict(better) for a in A.MAIN_FAMILY}
    arms["R0"] = dict(base)
    return {"S34_t": arms, "S34_p": arms, "S4_np_t": {"R0": base, "R2s": better, "R5p": better},
            "S3num_t": {"R0": base, "R2s": base, "R6f": {q: v + 2 * delta for q, v in base.items()}},
            "S2_t": arms, "S1to4_t": arms, "S1_t": arms}


def test_verdict_v4_passes_on_a_clear_effect_and_fails_on_none():
    v = A.verdict_v4(_cells(0.2))
    assert v["T1"]["pass"] and v["T2"]["pass"] and v["T3"]["pass"] and v["T4"]["pass"]
    assert all(r["kept"] for r in v["T5"].values())
    assert v["T6_authored_minus_extracted"] is None
    v = A.verdict_v4(_cells(0.0))
    assert not (v["T1"]["pass"] or v["T2"]["pass"] or v["T3"]["pass"] or v["T4"]["pass"])


def test_weighted_expansion_does_not_sum_hub_boosts():
    seeds = [("s1", 10.0), ("s2", 9.5), ("s3", 9.0), ("x", 8.0)]
    edges = [("s1", "hub", "prose"), ("s2", "hub", "prose"), ("s3", "hub", "prose")]
    assert A.weighted_expansion(seeds, edges).index("hub") > A.weighted_expansion(seeds, edges).index("x")


def test_choice_question_listing_every_value_is_not_a_filter():
    meta = _meta()
    pf = A.parse_filters("Which stores are in the North or South regions?", meta)
    assert pf["filters"] == []
