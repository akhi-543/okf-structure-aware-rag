"""Tests for the ingest CLI's guards and stats writer.

Lives at `tests/` because `scripts/ingest_corpus.py` is the pipeline driver,
not part of the `okf_rag.ingest` package. Nothing here connects to Postgres,
transcodes a corpus, or writes into `results/`.
"""

import argparse
from pathlib import Path

import pytest

from okf_rag.transcode.links import ResolutionStats
from scripts.ingest_corpus import _resolve_bundle_name, _stats_to_dict, _verify_pin, main, run

BRIGHTMART_BUNDLE = "synthetic_retail_pilot"


# --- Brightmart corpus: fixed scratch bundle, real tokens only ---


def test_brightmart_requires_its_scratch_bundle_name():
    with pytest.raises(ValueError, match="synthetic_retail_pilot"):
        _resolve_bundle_name("brightmart", None, False, False)
    with pytest.raises(ValueError, match="synthetic_retail_pilot"):
        _resolve_bundle_name("brightmart", "brightmart_smoke", False, False)


def test_brightmart_refuses_fast_tokens():
    with pytest.raises(ValueError, match="fast-tokens"):
        _resolve_bundle_name("brightmart", BRIGHTMART_BUNDLE, False, True)


def test_brightmart_bundle_accepted():
    assert _resolve_bundle_name("brightmart", BRIGHTMART_BUNDLE, False, False) == BRIGHTMART_BUNDLE


def test_brightmart_pin_check_records_content_hash(capsys):
    _verify_pin(Path("doc_synthetic/corpus"), "brightmart", allow_unpinned=False)
    assert "corpus_sha256" in capsys.readouterr().out


def test_run_refuses_before_doing_any_work(tmp_path):
    """The bundle guard is the first thing `run()` does, so an unsafe
    invocation cannot reach the transcoder or the database. The DSN points at
    an unroutable address so a removed guard cannot reach a real database."""
    args = argparse.Namespace(
        corpus="brightmart", repo_dir=str(tmp_path / "nope"),
        bundle_out=str(tmp_path / "bundle"), bundle_name=None, out_dir=str(tmp_path),
        dsn="postgresql://nobody:nobody@127.0.0.1:1/does-not-exist",
        allow_unpinned=False, fast_tokens=False, load_embeddings=None,
    )
    with pytest.raises(ValueError, match="synthetic_retail_pilot"):
        run(args)


# --- stats writer ---


def test_stats_to_dict_records_corpus_and_bundle():
    stats = ResolutionStats(resolved=10)
    stats.record_unresolved("src.md", "/missing.md")
    out = _stats_to_dict("brightmart", docs=[], edges=[], stats=stats, bundle=BRIGHTMART_BUNDLE)
    assert out["corpus"] == "brightmart"
    assert out["bundle"] == BRIGHTMART_BUNDLE
    assert out["unresolved"] == 1
    assert out["unresolved_in_scope"] + out["unresolved_out_of_scope"] == 1


# --- --load-embeddings must not silently ignore ingest arguments ---


def test_load_embeddings_with_corpus_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", [
        "ingest_corpus.py", "--load-embeddings", str(tmp_path / "e.parquet"), "--corpus", "brightmart",
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_load_embeddings_with_fast_tokens_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", [
        "ingest_corpus.py", "--load-embeddings", str(tmp_path / "e.parquet"), "--fast-tokens",
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_load_embeddings_alone_runs_that_mode(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr("scripts.ingest_corpus.run_load_embeddings",
                        lambda args: called.setdefault("parquet", args.load_embeddings))
    monkeypatch.setattr("sys.argv", ["ingest_corpus.py", "--load-embeddings", str(tmp_path / "e.parquet")])
    main()
    assert called["parquet"] == str(tmp_path / "e.parquet")


def test_cli_refuses_wrong_bundle_as_usage_error(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", [
        "ingest_corpus.py", "--corpus", "brightmart", "--repo-dir", str(tmp_path),
        "--bundle-out", str(tmp_path / "b"), "--bundle-name", "brightmart",
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
