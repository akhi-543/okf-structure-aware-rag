# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The full briefing (project, repository map, pipeline commands, hard rules) lives in `AGENTS.md`,
imported below. This file only adds what `AGENTS.md` leaves out.

@AGENTS.md

## Additions for Claude Code

**Tests.** Single test: `python -m pytest tests/test_brightmart_render.py::<test_name> -q`.
`pyproject.toml` sets `addopts = "-m 'not integration'"`, so anything marked
`@pytest.mark.integration` (needs Docker, network or models) is skipped unless you pass
`-m integration`. No linter or formatter is configured.

**Database.** Default DSN everywhere: `postgresql://okf:okf@localhost:5434/okf`
(`okf_rag/ingest/load_pg.py`). Override with `--dsn` on the run script, or `OKF_DSN` for the web
bench. Tables: `documents` (one row per page, `frontmatter` JSONB, `breadcrumb`, `depth`),
`chunks` (`policy` is `'flat'` or `'struct'` — the two paired representations share this table;
`embedding vector(768)`), `edges` (`edge_kind` child/curated/related/prose/sibling,
`provenance` authored/extracted), `queries`, `qrels`. `chunk_id` is BIGSERIAL, which is why a
re-ingest invalidates every embedding parquet (rule 6).

**Questions.** One JSONL row per question with `query_id`, `stratum`, `origin`, `text`,
`reference_answer`, `source_paths`. Ids: `bm-s<N>-<NN>-t|p` is development, `bm-h-s<N>-<NN>-t|p`
is held-out; `t` = template wording, `p` = paraphrase. Strata:
S1 direct fact (one document), S2 two-hop (two documents), S3 metadata set (SQL result set),
S4 hierarchy set (child documents plus the parent path). The verdict criteria C1–C3 are
defined over these strata in `brightmart_pilot_arms_v1.verdict`.

**Run outputs.** `--mode dev` writes `results/brightmart-pilot-v3-dev/` (overwritable, no
verdict). `--mode heldout` writes `results/brightmart-pilot-v3/` and refuses if its
`verdict.json` exists — a new held-out run needs a new versioned script/output directory, not a
deleted verdict.

**v4 run outputs.** `python -m scripts.run_brightmart_v4 --mode dev|smoke|heldout` writes
`results/brightmart-v4-dev/`, `-smoke/` (held-out code path on dev questions; use it to test changes)
and `results/brightmart-v4/` (one-shot; refuses on a freeze mismatch or an existing verdict). The v4
corpora live only under `results/corpus_v4*` (rendered, not committed), because their file names
repeat the pilot's and each other's. GPU steps (`scripts/llm_brightmart_v4.py`) run on Kaggle via
`notebooks/okf_v4_kaggle_full_run.ipynb`; locally, smoke them with `--model-override Qwen/Qwen3-0.6B
--out-dir results/<something>-smoke --limit N`.

**Web bench.** `OKF_WEB_BENCH_SKIP_QWEN=1` skips loading Qwen3-0.6B (retrieval lanes only, fast
startup). The bench serves only the `synthetic_retail_pilot` bundle.
