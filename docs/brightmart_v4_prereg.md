# Brightmart v4 — preregistration (next iteration, tests T1–T9)

Status: 2026-09-27, written after the corpus and both question sets were built and the held-out
set was frozen (`configs/brightmart_v4_freeze.json`), and before any retrieval configuration was
implemented or run. Test definitions follow `docs/next_iteration_test_plan.md`. The code constants
in `okf_rag/retrieve/brightmart_v4_arms.py` (`THRESHOLDS_V4`, `MAIN_FAMILY`, `T2_FAMILY`, …) must
equal this document. Changing any of them after a run requires a new versioned output directory
and a note here explaining why.

## Data

| Item | Value |
|---|---|
| Fact database | `doc_synthetic/brightmart_v4_seed.py` (seed 20260924), pilot schema |
| Corpus, standard (T1, T3–T7) | 348 documents, `results/corpus_v4`, bundle `synthetic_retail_v4`, sha256 `83559da2…8cc2ac` |
| Corpus, noparent (T2) | same paths and prose; stores carry no `region`, categories no `department`; bundle `synthetic_retail_v4_noparent`, sha256 `9781aa0a…6eb058a9` |
| Development set | `doc_synthetic/brightmart_v4_dev_*.jsonl`, ids `bm4-s…`, 200 base items + 200 paraphrases |
| Held-out set | `doc_synthetic/brightmart_v4_heldout_*.jsonl`, ids `bm4-h-s…`, 200 base items + 200 paraphrases, frozen 2026-09-27 |

Documents: 1 home, 8 regions, 96 stores, 96 store sales reports, 15 departments, 60 categories,
25 suppliers, 20 promotions, 10 policies, 16 archive/draft near-duplicates, 1 schema page. Store
format and opening year are drawn independently (T3).

Strata per set (40 base items each): S1 direct fact (includes stores with archived stale
profiles, promotions with stale drafts, policies with stale versions, look-alike names); S2 two-hop;
S3 metadata set (20 `categorical`, 20 `numeric` numeric/date conditions); S4 hierarchy set;
S5 unanswerable. Gold is computed from SQL and the rendered documents
(`brightmart_v4_validate.py` re-executes it).

## Configurations

Every configuration returns up to 50 ranked documents. Query vectors and chunk vectors come from
the pinned `BAAI/bge-base-en-v1.5` on CPU. No configuration receives gold.

| Name | Test | Mechanism |
|---|---|---|
| R0 | baseline | dense cosine over flat chunks, collapsed to documents by first occurrence |
| R2s | T1 | BM25 over one normalized metadata pseudo-document per page (pilot v3 configuration, unchanged) |
| R1h | T5 | dense over structured chunks embedded as `heading path (without the page title) + text` |
| R3s | T5 | R2s seeds (top 5, scores normalized by the maximum) expanded along authored edges: `score(d) = s(d) + 0.25 · max w(kind) · s(seed)` over undirected edges, `w` = `EDGE_WEIGHTS` (child 1.0, prose 0.6); see Amendment 1 |
| R3x | T6 | R3s with the LLM-extracted edge set instead of the authored one |
| R5p | T2 | parent-aware traversal: parents (documents with child edges, not the home page) whose full title appears in the question and whose children have the question's target type; return their children via authored child edges ordered by R2s score, then the R2s ranking; otherwise R2s |
| R6f | T3 | rule-based query-to-filter: target type plus numeric, date, boolean and categorical conditions parsed from the question into frontmatter filters; matching documents first (R2s order), then the R2s ranking; R2s when nothing is parsed |
| R6l | T3 (exploratory) | as R6f with filters produced by the pinned Qwen3-4B (`query_filter` role, greedy) |
| R7f | T4 | reciprocal rank fusion of R2s and R0; `rrf_k` chosen on the development templates S1–S4 from {1, 3, 10, 30, 60} by the highest mean nDCG@10 (ties: larger k), stored before any held-out run |
| R7h | T4 | two-step: documents whose title appears in the question (longest match first, at most 3), then their authored neighbours ordered by dense similarity to the question, then R7f |
| C_hier_oracle | ceiling | hierarchy oracle handed the gold parent directory; never in a verdict |

## Statistics (all tests)

Primary metric nDCG@10; recall@50, set-level F1@10, P@10 and MRR@10 reported alongside. Paired
bootstrap, 10,000 resamples, seed 7; Holm correction at α = 0.05 within each family below. Held-out
templates are the confirmatory set; paraphrases test robustness.

## Pass criteria

- **T1 — scale (standard corpus, S3+S4 templates, n = 80).** Main family = {R2s, R1h, R3s, R5p, R6f,
  R7f, R7h}, each minus R0. Pass iff R2s − R0 has delta > 0, CI lower bound > 0 and is
  Holm-significant in the main family, **and** on S3+S4 paraphrases its CI lower bound > 0 with
  delta ≥ 0.5 × the template delta. Recall@50 and F1@10 deltas reported with CIs.
- **T2 — hierarchy without parent names (noparent corpus, S4 templates, n = 40).** Family = {R5p, R2s}
  minus R0. Pass iff R5p − R0 has CI lower bound > 0 and is Holm-significant. The oracle ceiling is
  reported alongside.
- **T3 — numeric and date conditions (standard corpus, S3 `numeric` templates, n = 20).** Family =
  {R6f − R0, R6f − R2s}. Pass iff both have CI lower bound > 0 and are Holm-significant. R6l is
  reported with the same comparisons, outside the verdict.
- **T4 — multi-hop fusion.** For V ∈ {R7f, R7h}: S2 templates CI lower bound of (V − R0) > −0.05,
  **and** V − R0 on S3+S4 templates has CI lower bound > 0 and is Holm-significant in the main family.
  Pass iff at least one V meets both.
- **T5 — contextual chunks and link expansion.** For R1h and R3s, report the S3+S4 template delta vs
  R0 with CI; a variant is kept iff its CI lower bound > 0.
- **T6 — extracted vs authored structure.** Report R3s − R3x (authored minus extracted) on S3+S4
  templates and on all S1–S4 templates with CIs, plus extraction cost (prompt and completion
  tokens, wall time), extracted edge count and document-pair precision/recall against the
  authored edges. No threshold.
- **T7 — answer quality (held-out templates).** Contexts from R0 (top 10 flat chunks) and R2s
  (structured chunks of the top 10 documents, lead sections first), packed to 2,048 tokens;
  generator `answer_generator` pin, greedy, 256 new tokens, identical prompt. Correctness: S1/S2
  normalized containment of the reference answer; S3/S4 answer-set F1 over entity names of the
  target type. Faithfulness: HHEM-2.1 score ≥ 0.5 against the delivered context (an abstention
  counts as faithful); MiniCheck-7B support reported as a secondary judge. Abstention: the answer
  contains `NOT FOUND`. Pass iff on S3+S4, (R2s − R0) answer F1 has CI lower bound > 0 **and**
  (R2s − R0) faithful rate has CI lower bound > 0, **and** on S5 the abstention rate of R2s is not
  lower than R0's (point estimate).
- **T8 — human review.** `scripts/review_brightmart_v4.py` draws a 20% stratified sample of the
  held-out templates (8 per stratum) with gold and both answers; two reviewers label
  independently. Pass iff Cohen's κ ≥ 0.7 on every judgment column. A gold error found triggers a
  versioned correction and rerun.
- **T9 — efficiency.** Per configuration: index build time, index size, query latency p50/p95,
  context tokens per answer (T7), index-time LLM tokens (T6; zero for authored structure).
  Reported as a table; no threshold.
- **T10 — real corpora.** Not part of this iteration (requires pinned downloads, mined forum
  questions and hand-labelled gold).

## Known limitations fixed in advance

- The questions, the paraphrases and the rule-based parser (R6f) are written by the same author.
  The held-out wording was fixed and frozen before R6f was written, but the author had seen it. The
  LLM-assisted filter (R6l) and the paraphrase criterion are the checks against wording overfit.
- One synthetic corpus; region names follow real US geography, so a dense model may partly infer
  regions from city and state names (this works against the structure-aware configurations on T2).

## Amendment 1 — development-set changes before any held-out run (2026-09-27)

Made after development runs only; the held-out set had not been run by any configuration.

1. **R3s/R3x aggregation.** The first development run showed hub pages (departments linked from every
   store, the home page) collecting a boost from each seed and outranking the seeds' own hits
   (R3s S3+S4 nDCG@10 0.23 vs R2s 0.59). The boost is now the best single seed's,
   `score(d) = s(d) + λ · max over seeds of w(kind) · s(seed)`, and λ was chosen from {0.25, 0.5} by the
   same development objective as `rrf_k` (mean nDCG@10 over S1–S4 templates): **λ = 0.25**. Other
   variants tried on the development set (child edges only, directed, 3 seeds) scored lower on that
   objective.
2. **R6f choice questions.** A categorical condition that names every value of its field
   ("in the food, health or general group") is a choice between answers, not a filter, and is dropped.
3. **R6f parser development.** The noun-phrase head rule for the target type, the `where` → store rule,
   "open since" (not a status), "only / limited to" for list fields and "rather than" as negation were
   added while reading development-set misses.
4. **R7f tuning result.** `rrf_k = 30` (development means: 1 → 0.678, 3 → 0.692, 10 → 0.725,
   30 → 0.735, 60 → 0.735), frozen in `configs/brightmart_v4_tuning.json`.
5. **Determinism fix (shared code).** `BM25Index.search` summed term scores in `set` order, which follows
   the per-process string hash seed; near-tied documents could swap between runs. Terms are now
   summed in sorted order (`tests/test_bm25_hashseed_determinism.py`). This affects R2s and every
   configuration built on it, including the pilot's; development runs are now byte-identical across
   hash seeds.
6. **HHEM loading.** HHEM-2.1-Open is loaded as `T5ForTokenClassification` with its pinned weights and
   prompt instead of its remote code (which fails on transformers 5); it reproduces the model card's
   reference scores exactly (`tests/test_hhem_reimplementation.py`, integration).

## Amendment 2 — after the held-out run (operational, no criterion changed)

On Kaggle (2x T4) the secondary judge MiniCheck-7B ran out of GPU memory, so the first T7 report
has HHEM only. The loader now leaves 5 GiB per GPU free for activations, and
`llm_brightmart_v4.py judge --minicheck-only` adds MiniCheck to the recorded answers and HHEM
scores in a new output directory. The preregistered T7 verdict uses HHEM and is unaffected.
The checkpoint's config selects eager attention (a 32 x L x L fp32 score matrix per layer, which still ran
out of memory); MiniCheck now runs with the SDPA attention path shipped in the same modeling code and
computes next-token logits for the last position only. Both are the same computation up to fp16 rounding.
The MiniCheck resume ran on AWS (1x A10G) on 2026-09-28: 299 answers, 402 s, every HHEM score unchanged.
Results: `docs/brightmart_v4_results.md`.

## Deviation D1 — T8 reviewed by one human reviewer (2026-09-28)

The preregistered 20% sample (40 held-out templates, 80 generated answers) was reviewed by one
human reviewer, the project author, instead of two independent reviewers. The review found no
errors in the gold data or the answer judgements and agreed with the automatic T7 scoring on all
80 judgements. With a single reviewer, Cohen's kappa between reviewers could not be computed, so
the two-reviewer part of the T8 criterion is not met.
