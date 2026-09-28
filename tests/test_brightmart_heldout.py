import json
import sys
from pathlib import Path

sys.path.insert(0, "doc_synthetic")
import brightmart_heldout_build as hb  # noqa: E402
import brightmart_seed  # noqa: E402
import brightmart_validate as bv  # noqa: E402


def _built():
    conn = brightmart_seed.connect()
    try:
        return hb.build(conn)
    finally:
        conn.close()


def test_counts_and_ids():
    qs, _, _ = _built()
    assert len(qs) == 100
    for n in range(1, 6):
        assert sum(q["stratum"] == f"S{n}" for q in qs) == 20
    assert all(q["query_id"].startswith("bm-h-s") and q["query_id"].endswith("-t") for q in qs)
    assert len({q["text"] for q in qs}) == 100


def test_no_overlap_with_v2():
    qs, _, _ = _built()
    v2 = [json.loads(l)["text"] for l in Path("doc_synthetic/brightmart_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert hb.v2_overlap([q["text"] for q in qs], v2) == []
    assert hb.v2_overlap(["Which Stores are  currently remodeling?"], ["Which stores are currently remodeling?"]) != []


def test_cross_checks_from_spec():
    qs, qrels, _ = _built()
    by_id = {q["query_id"]: q for q in qs}
    assert "176,000" in by_id["bm-h-s1-01-t"]["reference_answer"]
    assert "closed" in by_id["bm-h-s1-19-t"]["reference_answer"].lower()
    gold = {}
    for r in qrels:
        gold.setdefault(r["query_id"], set()).add(r["doc_path"])
    assert gold["bm-h-s3-01-t"] == {"regions/northeast/store-s02-riverdale.md", "regions/midwest/store-s08-lakeshore.md"}
    assert len(gold["bm-h-s4-12-t"]) == 6
    assert "bm-h-s5-01-t" not in gold


def test_s2_two_docs_of_different_types():
    qs, qrels, _ = _built()
    for q in qs:
        if q["stratum"] == "S2":
            docs = [r["doc_path"] for r in qrels if r["query_id"] == q["query_id"]]
            assert len(docs) == 2 and len(set(docs)) == 2  # type distinctness is enforced by validate_questions


def test_validator_accepts_heldout(tmp_path):
    qs, qrels, spans = _built()
    for name, rows in (("q.jsonl", qs), ("r.jsonl", qrels), ("s.jsonl", spans)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    errs = bv.validate_questions(str(tmp_path / "q.jsonl"), str(tmp_path / "r.jsonl"), str(tmp_path / "s.jsonl"),
                                 paraphrases_file=None, id_prefix="bm-h-s", expected_total=100)
    assert errs == [], errs[:5]


def test_validator_defaults_unchanged():
    assert bv.validate_questions() == []
