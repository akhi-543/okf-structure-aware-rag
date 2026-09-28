"""Tests for the Brightmart v4 fact database, renderer, question sets and validator checks.

Renders in memory; nothing here reads results/ or needs Postgres.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "doc_synthetic"))

import brightmart_v4_questions_build as B  # noqa: E402
import brightmart_v4_render as R  # noqa: E402
import brightmart_v4_seed as seed  # noqa: E402
import brightmart_v4_validate as V  # noqa: E402


@pytest.fixture(scope="module")
def world():
    conn = seed.connect()
    f = R.load_facts(conn)
    return conn, f, R.render(conn, "standard"), R.render(conn, "noparent")


def _fm(text):
    return yaml.safe_load(text.split("---\n", 2)[1])


def test_seed_counts_and_decoupled_year(world):
    conn, f, _, _ = world
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("region", "store", "department", "category", "supplier", "promotion", "policy", "archive_doc")}
    assert counts == {"region": 8, "store": 96, "department": 15, "category": 60, "supplier": 25,
                      "promotion": 20, "policy": 10, "archive_doc": 16}
    years = {fmt: [s[6] for s in f["stores"].values() if s[5] == fmt] for fmt in ("supercenter", "express")}
    assert min(years["express"]) < 2005 and max(years["supercenter"]) > 2015  # format no longer implies age


def test_pharmacy_department_iff_has_pharmacy(world):
    _, f, _, _ = world
    for sid, s in f["stores"].items():
        assert ("D05" in f["store_depts"][sid]) == bool(s[8])


def test_render_is_deterministic_and_variants_share_paths(world):
    conn, _, docs, docs_np = world
    assert len(docs) == 348 and set(docs) == set(docs_np)
    assert R.render(conn, "standard") == docs


def test_noparent_variant_removes_only_child_to_parent_naming(world):
    _, f, docs, docs_np = world
    store = R.store_path(f["stores"]["S001"])
    std, np_ = _fm(docs[store]), _fm(docs_np[store])
    assert std["region"] == "Northeast" and "region" not in np_
    assert np_["parent"] == std["parent"] and "[Back to " not in docs_np[store]
    cat = R.category_path(f["categories"]["C01"], f["depts"])
    assert "department" not in _fm(docs_np[cat]) and "This category is part of" not in docs_np[cat]
    assert docs_np[R.HOME_PATH] == docs[R.HOME_PATH]


def test_validator_checks_pass_on_fresh_render(world):
    _, f, docs, docs_np = world
    assert V.check_corpus("standard", docs, f) == []
    assert V.check_corpus("noparent", docs_np, f) == []
    assert V.check_filler(docs, f) == []
    assert V.check_name_substrings(f) == []


def test_validator_catches_a_leaked_region(world):
    _, f, _, docs_np = world
    store = R.store_path(f["stores"]["S001"])
    broken = dict(docs_np)
    broken[store] = broken[store].replace("store_id: S001", "region: Northeast\nstore_id: S001")
    assert any("still names its region" in e for e in V.check_corpus("noparent", broken, f))


def test_committed_question_sets_match_the_builder(world):
    conn, _, _, _ = world
    built = B.build(conn)
    for which in B.SETS:
        on_disk = [json.loads(x) for x in (ROOT / B.OUT[which]["questions"]).read_text(encoding="utf-8").splitlines()]
        assert sorted(q["query_id"] for q in on_disk) == sorted(q["query_id"] for q in built[which][0])
        assert {q["query_id"]: q["text"] for q in on_disk} == {q["query_id"]: q["text"] for q in built[which][0]}


def test_question_checks_pass_and_sets_are_disjoint(world):
    conn, _, docs, docs_np = world
    for which in B.SETS:
        assert V.check_questions(which, conn, docs, docs_np) == []
    dev = [json.loads(x) for x in (ROOT / B.OUT["dev"]["questions"]).read_text(encoding="utf-8").splitlines()]
    ho = [json.loads(x) for x in (ROOT / B.OUT["heldout"]["questions"]).read_text(encoding="utf-8").splitlines()]
    assert not {q["text"] for q in dev} & {q["text"] for q in ho}
    numeric = [q for q in ho if q["stratum"] == "S3" and q["query_id"].endswith("-t") and q["condition"] == "numeric"]
    assert len(numeric) == 20


def test_freeze_file_matches_committed_heldout(world):
    _, _, docs, docs_np = world
    frozen = json.loads(V.FREEZE_PATH.read_text(encoding="utf-8"))
    assert frozen["corpus_sha256"] == {"standard": V.corpus_sha256_of(docs), "noparent": V.corpus_sha256_of(docs_np)}
    assert frozen["heldout_files"] == {p: V.file_sha256(p) for p in V.HELDOUT_FILES}
