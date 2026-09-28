# Brightmart v4 — held-out results (next iteration, tests T1–T9)

Status: 2026-09-28. Confirmatory run of the study preregistered in `docs/brightmart_v4_prereg.md`
(with Amendments 1 and 2). Retrieval, graph extraction, LLM filters, answers and HHEM ran on Kaggle
(2x T4) with `notebooks/okf_v4_kaggle_full_run.ipynb`; the secondary MiniCheck judge ran on AWS
(1x A10G) with `llm_brightmart_v4.py judge --minicheck-only`. Archived outputs (verdicts, manifests,
per-query metrics, T7 scores): `docs/brightmart_v4_results/`. The complete Kaggle and AWS outputs,
including rankings, answers, extracted edges and the T8 sheets, are unpacked locally in
`results/kaggle_v4_run/` (gitignored; the zips are `results/okf_v4_all_outputs.zip` and
`results/t7-minicheck.zip`).

## Verdicts

| Test | Question | Outcome |
|---|---|---|
| **T1** scale (348 docs) | Does structure-aware metadata retrieval (R2s) still beat flat retrieval on S3+S4? | **PASS** — +0.291 nDCG@10, 95% CI [+0.202, +0.380], Holm-significant; paraphrases +0.204 [+0.091, +0.316] (70% of the template gain) |
| **T2** hierarchy without parent names | With the child→parent naming removed, does parent-aware traversal (R5p) beat flat on S4? | **PASS** — +0.735 [+0.640, +0.822], Holm-significant (R2s +0.299 [+0.185, +0.416]) |
| **T3** numeric and date conditions | Does query-to-filter (R6f) beat flat and R2s on numeric/date S3 items? | **PASS** — vs flat +0.783 [+0.678, +0.877]; vs R2s +0.765 [+0.654, +0.862]; both Holm-significant |
| **T4** multi-hop fusion | Keep S2 within 0.05 of flat while keeping the S3+S4 gain? | **PASS** via two-step R7h — S2 +0.108 [+0.058, +0.159], S3+S4 +0.167 [+0.092, +0.237]. RRF (R7f) fails on S2 (CI low −0.096) |
| **T5** contextual chunks, link expansion | Is each variant's S3+S4 CI lower bound above 0? | R3s **kept** (+0.253 [+0.167, +0.338]); R1h **not kept** (−0.019 [−0.081, +0.041]) |
| **T6** extracted vs authored links | Authored − extracted, same expansion (R3s vs R3x) | No difference: S3+S4 −0.001 [−0.018, +0.017]; S1–S4 +0.001 [−0.008, +0.011]. Extraction cost 968,467 LLM tokens (1,984 s on 2x T4) |
| **T7** answer quality | Are structure-aware answers more correct **and** more often faithful on S3+S4, with S5 abstention not lower? | **FAIL** — correctness improves (answer F1 +0.146 [+0.074, +0.219]) but faithfulness does not (HHEM −0.100 [−0.212, +0.013]; MiniCheck −0.100 [−0.237, +0.037]); S5 abstention 0.900 vs 0.875 |
| **T8** human review | Cohen's κ ≥ 0.7 between two reviewers | **Completed by human review** — the project author found no errors in the 40 sampled gold items or the 80 answer judgements, and the automatic T7 scoring agreed with the review on all 80 (κ = 1.0). One reviewer instead of two (deviation D1) |
| **T9** efficiency | Cost table | Reported below |
| T10 real corpora | — | Not part of this iteration |

Primary metric nDCG@10, paired bootstrap 10,000 resamples (seed 7), Holm α = 0.05 within the
preregistered families; held-out templates are confirmatory, n = 80 for S3+S4, 40 per stratum.

## Retrieval, held-out set (nDCG@10 means)

Standard corpus. t = template wording, p = paraphrase. S3num is the 20-item numeric/date subset of S3.

| Stratum | R0 flat | R2s | R1h | R3s | R3x | R5p | R6f | R6l | R7f | R7h |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 t | **0.960** | 0.761 | **0.963** | 0.783 | 0.775 | 0.761 | 0.761 | 0.710 | 0.932 | 0.950 |
| S1 p | **0.957** | 0.746 | 0.897 | 0.769 | 0.759 | 0.746 | 0.746 | 0.708 | 0.913 | 0.747 |
| S2 t | 0.845 | 0.661 | 0.820 | 0.798 | 0.799 | 0.661 | 0.661 | 0.565 | 0.811 | **0.954** |
| S2 p | 0.801 | 0.649 | 0.696 | 0.784 | 0.778 | 0.649 | 0.649 | 0.581 | 0.817 | **0.923** |
| S3 t | 0.251 | 0.448 | 0.260 | 0.418 | 0.410 | 0.471 | **1.000** | 0.929 | 0.463 | 0.434 |
| S3 p | 0.240 | 0.344 | 0.240 | 0.326 | 0.341 | 0.389 | **0.964** | 0.954 | 0.389 | 0.388 |
| S3num t | 0.217 | 0.235 | 0.216 | 0.189 | 0.216 | 0.235 | **1.000** | 0.990 | 0.335 | 0.335 |
| S3num p | 0.209 | 0.170 | 0.223 | 0.139 | 0.164 | 0.170 | **1.000** | **1.000** | 0.245 | 0.241 |
| S4 t | 0.281 | 0.667 | 0.235 | 0.619 | 0.628 | 0.861 | 0.975 | **0.988** | 0.634 | 0.432 |
| S4 p | 0.264 | 0.568 | 0.162 | 0.574 | 0.574 | 0.871 | **0.975** | 0.950 | 0.587 | 0.392 |

No-parent corpus (T2), S4: flat 0.143 (t) / 0.130 (p), R2s 0.442 / 0.353, R3s 0.540 / 0.562,
**R5p 0.878 / 0.886**, R6f 0.661 / 0.589. The hierarchy oracle (all children of the gold parents,
ignoring extra conditions) reaches 0.792 on both corpora; R5p and R6f exceed it because many S4
items add a condition (pharmacy, format, opening year) that the oracle ignores.

Set-size metrics, S3/S4 templates (recall@50 / F1@10): flat 0.63 / 0.18 and 0.81 / 0.22; R2s 0.84 /
0.35 and 0.98 / 0.57; R5p 0.84 / 0.37 and 0.98 / 0.67; R6f 1.00 / 0.68 and 0.98 / 0.74.

Direct facts (S1 templates, vs flat): every metadata-only configuration loses (R2s −0.199
[−0.257, −0.140]); the ones that keep dense evidence do not (R7f −0.028 [−0.062, +0.000], R7h −0.010
[−0.075, +0.049], R1h +0.003).

## What the results say

1. **The structure-aware advantage holds at 6.6x the pilot corpus** (T1), with an effect of the same
   size as the pilot's (+0.291 here, +0.299 in the pilot) and a 70% paraphrase retention.
2. **Authored hierarchy works even when children do not name their parent** (T2): following the
   authored parent link lifts S4 from 0.14 to 0.88, while flat retrieval collapses (0.28 → 0.14)
   once the child pages stop naming their region.
3. **Numeric and date conditions need filters, not ranking** (T3): lexical metadata ranking is no
   better than flat on these items (0.235 vs 0.217); turning the conditions into frontmatter filters
   reaches 1.000. The LLM-assisted filter, which does not share the question author's knowledge of
   the wording, reaches 0.990 (templates) and 1.000 (paraphrases), so the rule-based result is not
   only an artifact of the author writing both the questions and the rules.
4. **Two-hop questions need a hop, not a fusion** (T4): RRF of metadata and dense rankings keeps the
   S3+S4 gain but costs S2; resolving the named entity and following its authored links raises S2
   above flat (0.954 vs 0.845) while keeping a significant S3+S4 gain.
5. **Link expansion helps, heading-path embeddings do not** (T5). Expansion needed the development
   fix in Amendment 1 (one seed's boost, not a sum over seeds).
6. **LLM-extracted links were as good as authored links for this configuration** (T6), at a cost of
   about 0.97 M LLM tokens to index 348 pages. The extracted graph is noisier (precision 0.30, recall
   0.60 against the authored document pairs) but the expansion is seeded from R2s and the weights are
   small, so the noise barely moves the ranking. Authored structure is free at index time.
7. **Better retrieval gives more correct answers but not more faithful ones** (T7, FAIL). See below.

## T7 in detail

| S3+S4 held-out templates (n = 80) | Flat context (R0) | Structure-aware context (R2s) | Δ (95% CI) |
|---|---|---|---|
| Answer-set F1 | 0.518 | 0.663 | +0.146 [+0.074, +0.219] |
| Faithful, HHEM ≥ 0.5 (primary) | 0.863 | 0.762 | −0.100 [−0.212, +0.013] |
| Faithful, MiniCheck (secondary) | 0.650 | 0.550 | −0.100 [−0.237, +0.037] |
| S1+S2 correct (n = 80) | 0.800 | 0.725 | −0.075 [−0.200, +0.050] |
| S5 abstention (n = 40) | 0.875 | 0.900 | +0.025 [−0.050, +0.100] |
| Mean context tokens | 1,560 | 1,788 | |

Generator Qwen3-4B-Instruct-2507 (greedy, 256 tokens), 2,048-token context budget. The two judges
agree on the direction and size of the faithfulness difference; neither is significant.

**Exploratory (post hoc, not part of the verdict).** The faithfulness gap is not explained by longer
answers: R2s answers name fewer entities (6.1 vs 7.4) and far fewer wrong ones (0.43 vs 1.78 per
answer), and longer lists are judged *more* faithful for both contexts. It concentrates in questions
qualified by a region:

| Question type | Faithful R0 | Faithful R2s | F1 R0 | F1 R2s | Context names the region |
|---|---|---|---|---|---|
| Region-qualified (n = 39) | 0.90 | 0.74 | 0.59 | 0.81 | R0 1.00, R2s 0.87 |
| Attribute-only (n = 41) | 0.83 | 0.78 | 0.45 | 0.53 | — |

R2s selects stores through their metadata (`region`, `has_pharmacy`) but delivers their body text,
whose lead sections do not state that metadata; flat retrieval tends to deliver the region page,
whose store list states the relation. The answer is then right but not verifiable from the text the
generator was given. A context that carries the matched metadata of each delivered document is the
obvious next test (it has to be preregistered and run on a new held-out set).

## T8 human review

The project author reviewed the preregistered 20% sample (8 held-out templates per stratum, both
answers each) against rules fixed before judging: *gold_ok* — the gold answer and documents answer
the question as worded; *correct* — S1/S2 state the gold fact (the group, not only the department,
for group questions), S3/S4 name every gold entity and no wrong one, S5 decline.

- **Gold:** 40/40 confirmed correct (every S3/S4 gold is also re-executed from SQL by the validator).
- **Answer judgements:** no errors found. Correct answers for flat/structure-aware — S1 8/8 vs 6/8,
  S2 4/8 vs 5/8, S3 0/8 vs 0/8, S4 1/8 vs 3/8, S5 8/8 vs 8/8. Typical failures: set answers that are
  incomplete or cut off at the 256-token limit (both contexts), two-hop answers that stop at the
  department instead of its group (mostly flat), and one structure-aware answer taken from an
  archived page's stale value.
- **Automatic scoring check:** the automatic T7 correctness (S3/S4 binarized at an exact answer set)
  matched the human review on all 80 judgements, κ = 1.0 per column
  (`python scripts/review_brightmart_v4.py kappa` on the two archived sheets).

The preregistration specified two independent reviewers; with one reviewer, inter-reviewer κ was not
computed (deviation D1). Sheets: `docs/brightmart_v4_results/brightmart_v4_t8_human_review.csv` and
`…_t8_automatic_sheet.csv`.

## T9 efficiency (Kaggle, held-out run, 400 queries per corpus)

| Configuration | Index-time model cost | Index size | Query latency p50 / p95 (ms) | Query-time model cost |
|---|---|---|---|---|
| R0 flat dense | 384 chunk embeddings, 131 s CPU | 1.18 MB vectors | 0.30 / 0.38 | 1 query embedding |
| R1h heading-path dense | 1,003 chunk embeddings, 108 s CPU | 3.08 MB vectors | 0.63 / 0.77 | 1 query embedding |
| R2s metadata BM25 | none | 9,716 postings | 1.28 / 1.95 | none |
| R3s authored link expansion | none (authored edges) | 9,716 postings + 2,888 edges | 3.87 / 4.41 | none |
| R3x extracted link expansion | 968,467 LLM tokens, 1,984 s on 2x T4 | 9,716 postings + 3,101 edges | 4.24 / 4.92 | none |
| R5p parent-aware | none | postings + edges | 2.61 / 3.18 | none |
| R6f rule filter | none | postings + field types | 0.54 / 1.12 | none |
| R6l LLM filter | none | postings + field types | 0.22 / 0.39 after the filter | ~533 LLM tokens and ~0.9 s per question (T4, batch 16) |
| R7f RRF | as R0 | vectors + postings | 0.12 / 0.15 after R0 and R2s | 1 query embedding |
| R7h two-step | as R0 + structured chunks | 4.26 MB vectors + postings + edges | 4.71 / 5.48 | 1 query embedding |

Latencies are in-process ranking times on CPU with precomputed query vectors; they exclude query
embedding (about 20 ms per question on the Kaggle CPU, batches of 32) and, for R6l, the LLM call. Answer
generation used 1,560 (flat) and 1,788 (structure-aware) context tokens per answer on average.

## Reproducibility

- The held-out run used the frozen corpora (sha256 `83559da2…` standard, `9781aa0a…` noparent), the
  frozen held-out files and the frozen `rrf_k = 30`; the run manifest records all three and matches
  `configs/brightmart_v4_freeze.json` and `configs/brightmart_v4_tuning.json`.
- The Kaggle development run reproduced the local one: 5,161 of 5,200 rankings identical and every
  summary metric identical to four decimals; the 39 differing rankings are near-ties in the two
  configurations that score structured-chunk vectors (R1h, R7h), whose CPU embeddings differ in
  the last bits between machines.
- The MiniCheck resume kept every HHEM score of the scored run unchanged.

## Limitations

- One synthetic corpus and one author for questions, paraphrases and the rule parser; the LLM filter
  (R6l ≈ R6f) is the check on the last point.
- Metadata-only configurations lose about 0.2 nDCG@10 on direct facts; a deployable system needs
  the dense evidence as well (R7h keeps both).
- T7 uses automatic judges; T8 was reviewed by one human reviewer, not two (deviation D1).

## Next steps

1. **Optional second reviewer:** an independent reviewer fills in the untouched sheet
   `results/kaggle_v4_run/brightmart-v4/t8/review_B.csv`; then
   `python scripts/review_brightmart_v4.py kappa --a docs/brightmart_v4_results/brightmart_v4_t8_human_review.csv --b …`
   measures inter-reviewer agreement (pass: every κ ≥ 0.7).
2. **Next iteration (new preregistration, new held-out set):** metadata-carrying contexts for
   answer generation (the T7 mechanism above); R7h + R6f combined as the default pipeline; T10 on
   real documentation corpora.
