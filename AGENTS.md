# AGENTS.md — working context for OKF Final

Read this first. It is the complete briefing for an agent working in this repository.

## What this project is

A retrieval study: does authored document structure (frontmatter metadata, hierarchy, authored
links) make retrieval more accurate than conventional flat retrieval over the same prose? The two
representations are paired: identical prose, flat version stripped of frontmatter, links and
hierarchy.

**Current result (held-out, preregistered):** structure-aware retrieval over normalized document
metadata beats flat dense retrieval on structure-dependent questions (S3+S4): nDCG@10 +0.299
(95% CI +0.166 to +0.423, Holm-significant), and +0.312 on paraphrased questions. Full write-up:
`docs/okf_final_pipeline_and_findings.md`. Planned work: `docs/next_iteration_test_plan.md`.

**Next iteration (v4, held-out, preregistered in `docs/brightmart_v4_prereg.md`; results in
`docs/brightmart_v4_results.md`):** on a 348-document corpus, R2s still beats flat on S3+S4 (+0.291,
CI +0.202 to +0.380; T1 pass); parent-aware traversal wins when children do not name their parent
(S4 +0.735; T2 pass); frontmatter filters solve numeric/date conditions (+0.783; T3 pass); a two-step
entity → authored-link configuration keeps two-hop quality (T4 pass); link expansion is kept, heading-path
embeddings are not (T5); LLM-extracted links match authored ones at ~0.97 M index tokens (T6). T7 fails:
structure-aware contexts give more correct answers (F1 +0.146) but not more faithful ones (HHEM −0.100,
CI crosses 0). T8 was human-reviewed by the project author with no errors found (single reviewer, deviation D1); T10 (real corpora) is not implemented. Held-out artifacts are
archived in `docs/brightmart_v4_results/`; the run itself is `notebooks/okf_v4_kaggle_full_run.ipynb`.

## Repository map

| Path | What it is |
|---|---|
| `doc_synthetic/` | Brightmart synthetic corpus: fact DB (`brightmart_schema.sql`, `brightmart_seed.py`), renderer, 53 docs in `corpus/`, filler prose, question builders, question sets, validator |
| `okf_rag/transcode/` | markdown → OKF documents + authored edges (`brightmart_synthetic.py`) and OKF bundle writer/validator |
| `okf_rag/ingest/` | flatten, chunk (flat + struct), Postgres load, embedding load, `schema.sql` |
| `okf_rag/jobs/` | embedding job helpers, model pins (`model_lock.py` reads `configs/models.lock`) |
| `okf_rag/retrieve/` | retrieval configurations: `brightmart_pilot_arms_v3.py` (structure-aware helpers), `brightmart_pilot_arms_v1.py` (metrics + preregistered verdict), `control_arms_v4.py` (BM25, RRF, graph, oracle), `dense.py` |
| `okf_rag/eval/` | IR metrics, paired bootstrap, Holm |
| `scripts/` | `ingest_corpus.py`, `embed_brightmart_pilot_v1.py`, `embed_brightmart_ctx_v3.py`, `run_brightmart_pilot_v3.py` (+ helpers in `run_brightmart_pilot_v1.py`) |
| `okf_web_bench/` | FastAPI console comparing the flat and structured lanes, with optional Qwen3-0.6B answers |
| `tests/` | pytest suite (all green); `tests/web_bench/` for the console |
| `docs/` | findings and next-iteration plan |
| `docker/docker-compose.yml` | Postgres 16 + pgvector, project `okf_final`, host port **5434** |
| `results/` | generated run outputs (gitignored; regenerate with the pipeline) |
| `doc_synthetic/brightmart_v4_*.py` | v4 fact DB (`_seed`), renderer for the `standard` and `noparent` corpora (`_render`, writes `results/corpus_v4*`), question builder (`_questions_build`, dev + held-out JSONL), validator (`_validate`, `--freeze`) |
| `okf_rag/retrieve/brightmart_v4_arms.py` | v4 configurations (R3s, R5p, R6f, R7h, …), metrics, preregistered constants and verdict |
| `okf_rag/jobs/brightmart_v4_llm.py`, `okf_rag/eval/answer_scoring_v4.py` | v4 LLM prompts/parsers; T7/T8 answer scoring and Cohen's kappa |
| `scripts/*brightmart_v4.py` | v4 embeddings, evaluation run (`--mode dev/smoke/heldout`), GPU steps (`llm_…`), T8 review sheets |
| `notebooks/okf_v4_kaggle_full_run.ipynb` | end-to-end v4 run on Kaggle (Postgres + pgvector installed in the notebook) |

## Retrieval configurations (names used in code and outputs)

- **R0** — flat baseline: dense bge-base cosine over flat chunks.
- **R2s** — structure-aware metadata retrieval: BM25 over one metadata pseudo-document per page
  (title, description, tags, type, status, every frontmatter field; field names split into words,
  booleans as yes/no, plural stemming; same normalization on the question). Built by
  `metadata_doc_dict` → `metadata_pseudo_docs_v3` → `structured_rank`. This is the configuration
  carried forward and the one the web bench's `structured` lane uses.
- **R1c, R3d, R4c** — further structure-aware configurations under development (contextual
  structured-chunk embeddings, document-level authored-link expansion, rank fusion).
- **R1, R2, R3a, R4, C_flat_hybrid** — earlier-iteration configurations, still computed by the run
  script as a reference table.
- **C_hier_oracle** — hierarchy oracle given the gold parent; a capability ceiling, never in a verdict.
- **v4 only** (definitions in `docs/brightmart_v4_prereg.md`): **R1h** heading-path contextual chunks,
  **R3s/R3x** typed-weight link expansion seeded from R2s over authored/LLM-extracted edges, **R5p**
  parent-aware traversal, **R6f/R6l** rule-based/LLM query-to-filter, **R7f** RRF(R2s, R0), **R7h**
  two-step entity → authored links.

## Setup and commands

```bash
pip install -e ".[dev,db,ml,web]"
python -m pytest -q                                   # unit tests, no DB or models needed
docker compose -f docker/docker-compose.yml up -d     # Postgres on localhost:5434
```

Full pipeline (order matters):

```bash
python scripts/ingest_corpus.py --corpus brightmart --repo-dir doc_synthetic/corpus \
    --bundle-out results/bundle_synthetic_retail_pilot --bundle-name synthetic_retail_pilot
python scripts/embed_brightmart_pilot_v1.py
python scripts/ingest_corpus.py --load-embeddings results/brightmart-pilot-v1/embeddings/emb_flat.parquet
python scripts/ingest_corpus.py --load-embeddings results/brightmart-pilot-v1/embeddings/emb_struct.parquet
python scripts/embed_brightmart_ctx_v3.py
python -m scripts.run_brightmart_pilot_v3 --mode dev
python -m scripts.run_brightmart_pilot_v3 --mode heldout
uvicorn okf_web_bench.okf_web_bench_app:app --port 8000
```

v4 (next iteration; local CPU steps, then the Kaggle notebook for the GPU and held-out steps):

```bash
python doc_synthetic/brightmart_v4_render.py          # results/corpus_v4, results/corpus_v4_noparent
python doc_synthetic/brightmart_v4_validate.py        # OK: … (includes the freeze check)
python scripts/ingest_corpus.py --corpus brightmart_v4 --repo-dir results/corpus_v4 \
    --bundle-out results/bundle_synthetic_retail_v4 --bundle-name synthetic_retail_v4
python scripts/ingest_corpus.py --corpus brightmart_v4_noparent --repo-dir results/corpus_v4_noparent \
    --bundle-out results/bundle_synthetic_retail_v4_noparent --bundle-name synthetic_retail_v4_noparent
python scripts/embed_brightmart_v4.py --bundle synthetic_retail_v4        # and --bundle synthetic_retail_v4_noparent
python scripts/ingest_corpus.py --load-embeddings results/brightmart-v4/embeddings/<bundle>/emb_flat.parquet   # and emb_struct
python -m scripts.run_brightmart_v4 --mode dev        # development set; smoke = held-out code path on dev questions
```

Regenerate data (all deterministic; outputs are committed and reruns are byte-identical):
`python doc_synthetic/brightmart_render.py`, `brightmart_questions_build.py`,
`brightmart_heldout_build.py`; validate with `python doc_synthetic/brightmart_validate.py` and
`--heldout`.

## Rules (hard constraints — each one exists because breaking it cost real work)

1. **File names are unique across the repository.** Before creating a file, check the name with a
   search; if it exists anywhere, choose another (`test_ingest_pg_conn.py`, not a second
   `test_conn.py`). Do not rename existing files. A same-named fixture once overwrote real data.
2. **Identifiers that key stored data are never reused.** The corpus bundle is
   `synthetic_retail_pilot`; `scripts/ingest_corpus.py` refuses any other bundle name for the
   `brightmart` corpus. Scratch or test runs get obviously-scratch names.
3. **Gold is computed, never typed.** Answers, gold documents, spans and sets come from SQL plus the
   rendered documents. Both validators must pass after any data change.
4. **Held-out protocol.** Develop and tune only on the development set (`bm-s…` ids). Write and
   commit a new held-out set before running any new configuration on it. Never adjust code,
   thresholds or questions after looking at held-out results; a held-out run writes to a new,
   versioned output directory and the run script refuses to overwrite an existing verdict.
5. **Preregister thresholds.** Criteria (`THRESHOLDS` in `brightmart_pilot_arms_v1.py`) are fixed
   before a run; changing them means a new versioned run and a note explaining why.
6. **Pipeline order (A5).** ingest → embed → load embeddings → run. Any re-ingest reassigns chunk
   ids, so every embedding parquet must be regenerated. The run script refuses stale vectors.
7. **Read-only outside ingest.** Only `ingest_corpus.py` (ingest and `--load-embeddings`) writes to
   Postgres. Runs, embedding scripts and the web bench only SELECT.
8. **Determinism.** Seed `20260924` for data generation; bootstrap seed 7; embeddings on CPU with the
   pinned revision in `configs/models.lock`. Rendered files must equal the files on disk.
9. **LF line endings** everywhere (`.gitattributes`); the validator compares bytes.
10. **Windows import order.** In any process that uses both, import torch / sentence-transformers
    before pyarrow (`tests/conftest.py` does this for the test session).
11. **Run the evaluation as a module:** `python -m scripts.run_brightmart_pilot_v3 …` (it imports
    `scripts.run_brightmart_pilot_v1`). Run everything from the repository root.
12. **Scripts import this repository's code.** Every file in `scripts/` starts with a block that puts
    the repository root first on `sys.path`, because `python scripts/x.py` otherwise imports any
    pip-installed package also named `okf_rag` (the original OKFF checkout is installed that way on
    the development machine, and its database default is port 5433). Keep the block in new scripts;
    `tests/test_okf_final_script_bootstrap.py` enforces it.
13. **Git hygiene.** Stage files by explicit path. Do not run destructive git commands (`reset
    --hard`, `clean`, `stash`, force-push) without the owner's go-ahead.

## Verifying a change

- `python -m pytest -q` — all tests pass.
- Data changes: both validators print `OK: 53 docs, 200 questions` (counts change with the corpus).
- v4 data: `python doc_synthetic/brightmart_v4_validate.py` prints `OK: 348 docs x 2 variants, …`. It
  fails by design if the v4 corpora or held-out files differ from `configs/brightmart_v4_freeze.json`;
  a deliberate change needs a new versioned held-out set, not an edited freeze file.
- Retrieval changes: `--mode dev` first; compare with the previous dev summary; only then a new,
  preregistered held-out run.

## Environment notes

- Developed on Windows 11, Python 3.12, 16 GB RAM, no CUDA. Everything here runs on CPU; Qwen3-0.6B
  generation in the web bench is slow on CPU but works. Larger generators (e.g. Qwen3-4B for graph
  extraction) need a GPU (Kaggle).
- The study's origin repository (OKFF) holds the earlier real-corpus work and the complete run
  history; this repository starts from the current pipeline.
