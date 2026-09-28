# OKF Final — Structure-aware retrieval vs flat retrieval: pipeline and findings

Status: 2026-09-24. This repository holds the pipeline, corpus, question sets, evaluation and web
bench for the structure-aware retrieval study. Section 10 lists the planned next iteration; the
detailed test plan is in `docs/next_iteration_test_plan.md`.

---

## 1. Research question

Does authored document structure (frontmatter metadata, hierarchy, links) make retrieval more
accurate than conventional flat retrieval over the same prose?

The comparison is between paired representations of identical content:

- **Flat:** the same document prose with frontmatter, hierarchy and link targets removed, split
  into fixed-size chunks (the conventional RAG input).
- **Structured:** the same prose with headings, hierarchy, metadata and authored links kept and
  available to the retriever.

Because the prose is identical, any difference in retrieval quality comes from the structure.

## 2. Headline finding

**Structure-aware retrieval outperforms flat retrieval on structure-dependent questions. The result
holds on a frozen held-out question set and on paraphrased questions.**

On 40 held-out set-membership and hierarchy questions (S3+S4), structure-aware retrieval over
normalized document metadata improved nDCG@10 over flat dense retrieval by **+0.299**
(95% CI +0.166 to +0.423). The improvement is statistically significant after Holm correction
across the structure-aware configurations tested. On paraphrased versions of the same questions
the improvement was **+0.312** (95% CI +0.189 to +0.427).

The same direction appears on the development question set. An exploratory re-analysis that
excludes question types most exposed to corpus-construction effects (year ranges,
category-by-department) keeps the improvement at +0.24 to +0.27.

## 3. Corpus: the Brightmart synthetic retail wiki

A fictional retailer, so a language model cannot answer from memory. The wiki has 53 documents
under `doc_synthetic/corpus/`.

| Document type | Count | Place in the hierarchy |
|---|---|---|
| Home page | 1 | root |
| Regions | 4 | home → region |
| Stores | 12 | region → store |
| Departments | 8 | home → department |
| Categories | 10 | department → category |
| Suppliers | 6 | home → supplier |
| Promotions | 5 | home → promotion |
| Policies | 3 | home → policy |
| Archive / draft pages | 3 | home → archive |
| Database schema page | 1 | home → schema |

How it is built (fully deterministic, seed `20260924`):

1. **Fact database** — `doc_synthetic/brightmart_schema.sql` and `brightmart_seed.py` build an
   in-memory SQLite database: regions, stores, departments, categories, suppliers, weekly 2025
   sales, promotions, policies and archive records. It is the single source of truth.
2. **Rendering** — `brightmart_render.py` writes every document from the database. Every fact
   sentence, frontmatter field, `parent` link and body link comes from the database, with at
   least three phrasing variants per fact.
3. **Filler prose** — `brightmart_llm_prose/*.prose.txt` holds LLM-written paragraphs appended
   under `## Notes`. They carry no facts: the validator rejects digits and any other entity's name.
4. **Difficulty by design** — near-duplicate archive and draft pages carry stale values (for
   example an old store profile with a different floor area), and look-alike names
   (Riverside/Riverdale, Cedar Falls/Cedar Park, Greenfield Farms/Greenfield Home Supply) test
   precision.
5. **Validation** — `brightmart_validate.py` checks unique file names, frontmatter, links, parent
   chains, that the files on disk equal the render, and the filler-prose rules.

## 4. Questions and gold

Five question types (strata), 20 base questions each, each also written as a paraphrase, giving
200 records per question set.

| Stratum | Question type | Gold |
|---|---|---|
| S1 | Direct fact from one document | 1 document and the exact answer line |
| S2 | Two-hop (e.g. the country of the supplier of a category) | exactly 2 documents of different types |
| S3 | Metadata set (e.g. stores with a pharmacy but no fuel station) | the SQL result set |
| S4 | Hierarchy set (e.g. stores in the Northeast or West regions) | child documents plus the home → parent path |
| S5 | Unanswerable | a term that appears nowhere in the corpus |

Gold is always computed from SQL plus the rendered documents, never typed by hand. The validator
re-executes the S3/S4 SQL and compares it with the gold.

Two question sets:

- **Development set** — `brightmart_questions.jsonl` (ids `bm-s…`), used while building the
  retrieval configurations.
- **Held-out set** — `brightmart_heldout_questions.jsonl` (ids `bm-h-s…`). It was written and
  committed before any evaluated configuration was run on it, and shares no question text with
  the development set. The reported result uses this set only.

## 5. Pipeline

```
fact DB (SQLite) ──render──▶ 53 markdown docs ──transcode──▶ OKF documents + authored edges
        │                                                    │
        └──SQL──▶ questions + computed gold                  ▼
                                               Postgres + pgvector (bundle synthetic_retail_pilot)
                                               documents · chunks (flat + struct) · edges
                                                             │
                              CPU embeddings (bge-base-en-v1.5, pinned revision)
                                                             │
                   retrieval configurations ──▶ document-level metrics ──▶ preregistered verdict
                                                             │
                                                  web bench (FastAPI console)
```

### 5.1 Components

| Stage | Code |
|---|---|
| Fact database | `doc_synthetic/brightmart_schema.sql`, `brightmart_seed.py` |
| Render corpus | `doc_synthetic/brightmart_render.py` |
| Validate corpus and questions | `doc_synthetic/brightmart_validate.py` (`--heldout` for the held-out set) |
| Questions and gold | `brightmart_questions_build.py` (development), `brightmart_heldout_build.py` (held-out) |
| Transcode to OKF | `okf_rag/transcode/brightmart_synthetic.py` (`child` edges from `parent`, `prose` edges from body links) |
| Ingest | `scripts/ingest_corpus.py` → `okf_rag/ingest/` (flatten, chunk flat + struct, load Postgres) |
| Flat twin | `okf_rag/ingest/flatten.py` (removes frontmatter, links and hierarchy) with 512/50 fixed chunks |
| Structured chunks | `okf_rag/ingest/chunk.py` (heading-aligned; heading path and frontmatter kept as columns) |
| Embeddings | `scripts/embed_brightmart_pilot_v1.py` (flat/struct chunks, development queries), `scripts/embed_brightmart_ctx_v3.py` (contextual struct chunks, held-out queries) |
| Retrieval | `okf_rag/retrieve/brightmart_pilot_arms_v3.py`, `okf_rag/retrieve/control_arms_v4.py`, `okf_rag/retrieve/dense.py` |
| Evaluation and verdict | `scripts/run_brightmart_pilot_v3.py` (helpers in `run_brightmart_pilot_v1.py`), verdict logic in `okf_rag/retrieve/brightmart_pilot_arms_v1.py`, metrics in `okf_rag/eval/` |
| Web bench | `okf_web_bench/` (see §8) |

### 5.2 Commands

```bash
pip install -e ".[dev,db,ml,web]"
docker compose -f docker/docker-compose.yml up -d          # Postgres + pgvector on port 5434

python doc_synthetic/brightmart_render.py                   # corpus (committed; reruns are byte-identical)
python doc_synthetic/brightmart_questions_build.py          # development questions
python doc_synthetic/brightmart_heldout_build.py            # held-out questions
python doc_synthetic/brightmart_validate.py
python doc_synthetic/brightmart_validate.py --heldout

python scripts/ingest_corpus.py --corpus brightmart --repo-dir doc_synthetic/corpus \
    --bundle-out results/bundle_synthetic_retail_pilot --bundle-name synthetic_retail_pilot
python scripts/embed_brightmart_pilot_v1.py
python scripts/ingest_corpus.py --load-embeddings results/brightmart-pilot-v1/embeddings/emb_flat.parquet
python scripts/ingest_corpus.py --load-embeddings results/brightmart-pilot-v1/embeddings/emb_struct.parquet
python scripts/embed_brightmart_ctx_v3.py

python -m scripts.run_brightmart_pilot_v3 --mode dev        # development set, no verdict
python -m scripts.run_brightmart_pilot_v3 --mode heldout    # held-out verdict; refuses to overwrite one

uvicorn okf_web_bench.okf_web_bench_app:app --port 8000     # web bench at http://localhost:8000
python -m pytest -q
```

Order rule: ingest → embed → load embeddings → run. A re-ingest reassigns chunk ids and
invalidates every embedding file, so the embedding steps must be repeated after it.

### 5.3 Retrieval configurations

Every configuration ranks 50 chunks, collapses them to documents by first occurrence, and keeps the
top 10 documents.

| Name | What it uses | Mechanism |
|---|---|---|
| **Flat (R0)** | none — the baseline | dense cosine similarity (bge-base) over flat chunks |
| **Structure-aware (R2s)** | frontmatter metadata | BM25 over one metadata pseudo-document per page: title, description, tags, type, status and every frontmatter field. Field names are split into words, booleans written as yes/no, light plural stemming; the question goes through the same normalization. |

Three further structure-aware configurations were explored and are included in the multiple-test
correction: contextual structured-chunk embeddings, document-level expansion along authored links,
and rank fusion. Their development continues in the next iteration (§10). A hierarchy oracle,
handed the gold parent directory, gives a capability ceiling for hierarchy questions (0.92 nDCG@10)
and is never part of the verdict.

### 5.4 Preregistered criteria (fixed before the held-out run)

Primary metric nDCG@10. Paired bootstrap, 10,000 resamples, seed 7. Holm correction across the
structure-aware configurations.

- **C1 — flat stays competitive on direct facts (S1 templates):** for every structure-aware
  configuration, the CI lower bound of (flat − structure-aware) is above −0.05.
- **C2 — structure helps on structure-dependent questions (S3+S4 templates, 40 items):** at least
  one structure-aware configuration with a positive delta, CI lower bound above 0, Holm-significant.
- **C3 — the effect survives paraphrase (S3+S4 paraphrases):** a C2 winner keeps CI lower bound
  above 0 and at least 50% of its template delta.

## 6. Results (held-out question set)

| Criterion | Outcome |
|---|---|
| **C2** — structure helps | **Met: +0.299 [+0.166, +0.423], Holm-significant** |
| **C3** — survives paraphrase | **Met: +0.312 [+0.189, +0.427] on paraphrased questions** |
| C1 — flat competitive on direct facts | Formal margin not met: a structure-aware configuration answered one direct-fact question (`bm-h-s1-14-t`) better than flat. On direct facts both approaches are near ceiling. |

nDCG@10 means, held-out set (n = 20 per cell):

| Stratum | Flat (R0) template | Structure-aware (R2s) template | Flat (R0) paraphrase | Structure-aware (R2s) paraphrase |
|---|---|---|---|---|
| S1 direct fact | 0.963 | 0.963 | 0.945 | 0.963 |
| S2 two-hop | 0.957 | 0.807 | 0.936 | 0.795 |
| S3 metadata set | 0.610 | **0.745** | 0.499 | **0.709** |
| S4 hierarchy set | 0.418 | **0.880** | 0.437 | **0.851** |

How the result was reached: a first full run (v2) showed that structure only pays off when the
retriever can read it. Un-normalized field names hid boolean and year fields from keyword search.
The v3 configuration normalizes the metadata surface (field names as words, booleans as yes/no,
stemming), and that change produced the significant gain on unseen questions.

## 7. Scope of this result

- One synthetic corpus of 53 documents; 20 questions per stratum (40 pooled for the primary test).
- The held-out questions are new instances of the same question types as the development set.
- In the current corpus each child document also names its parent (a store names its region, a
  category its department), so hierarchy questions are answered from the child's own metadata.
  The next iteration separates these fields so hierarchy traversal is tested directly (§10).

## 8. Web bench

`okf_web_bench/` is a FastAPI console that runs one question through both retrieval paths and,
optionally, generates an answer from each with the same generator, so the only difference is how
the context was found.

- **Lanes:** `flat` (dense over flat chunks) and `structured` (the evaluated structure-aware
  ranking). The structured lane delivers the ranked documents' structured chunks with every
  document's lead section first, so the context budget reaches all ranked documents. It uses the
  same helpers as the evaluation run (`metadata_doc_dict`, `metadata_pseudo_docs_v3`,
  `structured_rank`).
- **Generator:** Qwen3-0.6B (runs on CPU; uses the GPU when present). Answers cite the delivered
  passages; context is packed to a 2,048-token budget.
- **Corpus map tab:** browse any document's flat vs structured chunks and its authored links.
- **Start:** Postgres up and the bundle ingested with embeddings loaded (§5.2), then
  `uvicorn okf_web_bench.okf_web_bench_app:app --port 8000`. `OKF_DSN` overrides the database
  address; `OKF_WEB_BENCH_SKIP_QWEN=1` starts it without the generator (retrieval only).
- **Read-only:** every query is a SELECT; the console never writes to the database.

## 9. Reproducibility

- The corpus content hash (sha256 over sorted path + bytes) is printed at ingest and stored in
  `results/transcode_stats_synthetic_retail_pilot.json`.
- Embedding model pinned in `configs/models.lock` (`BAAI/bge-base-en-v1.5`, revision
  `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`), CPU, normalized vectors; queries use the bge query
  instruction prefix.
- The held-out run writes the complete per-configuration verdict to
  `results/brightmart-pilot-v3/verdict.md`, `verdict.json`, `metrics_summary.csv` and
  `manifest.json` (input hashes, model revision, row counts). It refuses to overwrite an existing
  verdict.
- The full pipeline was re-run from scratch in this repository on a fresh database. The held-out
  verdict, every ranking and every metric file were byte-identical to the original run. On the
  development set all metrics were identical; one reference ranking differed only in the order of
  two tied documents (ties are broken by database chunk id, which a fresh ingest reassigns).
  A later fix (2026-09-27) made BM25 scoring independent of Python's hash seed: it had summed term
  scores in `set` order, so near-tied documents could also swap between processes.
- Line endings are LF everywhere (`.gitattributes`), because the validator compares rendered bytes.

## 10. Future work (next iteration)

The next iteration extends the study along five lines. The concrete test plan, with acceptance
criteria, is in `docs/next_iteration_test_plan.md`; its held-out results (T1–T9) are in
`docs/brightmart_v4_results.md`.

1. **Scale** — grow the corpus to 300–400 documents from a regenerated fact database, with larger
   answer sets and metrics suited to them (recall@k with larger k, set-level F1).
2. **Harder structure tests** — separate parent names from child metadata so hierarchy questions
   require traversal; decouple correlated attributes (store format and opening year); add new
   question types and harder direct-fact items; a second, independently written held-out set.
3. **Richer structure-aware retrieval** — a query-to-filter component for numeric and date
   conditions; fusion of structured and dense signals for multi-hop questions; continued work on
   contextual structured chunks and link expansion; LLM-extracted graphs compared with authored
   links; cross-encoder reranking; a Postgres-native field-weighted metadata index.
4. **Answer quality** — answer generation over both paths with faithfulness scoring (HHEM-2.1,
   MiniCheck, RAGAS), abstention on unanswerable questions, and human review of a sample with
   inter-annotator agreement.
5. **Efficiency and real data** — index build time, index size, query latency and context tokens;
   then the real documentation corpora (github/docs, home-assistant.io) with mined real user
   questions.
