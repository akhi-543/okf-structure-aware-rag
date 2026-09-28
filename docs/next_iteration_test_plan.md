# Next iteration — test plan

Status: 2026-09-28, T1–T9 run as the Brightmart v4 study (`docs/brightmart_v4_prereg.md`, results in
`docs/brightmart_v4_results.md`): T1–T4 pass, T5 keeps link expansion only, T6 reported, T7 fails,
T8 human-reviewed by the project author (no errors found), T9 reported; T10 not implemented. Originally written 2026-09-24. Builds on the held-out result in
`docs/okf_final_pipeline_and_findings.md` (structure-aware retrieval +0.299 nDCG@10 over flat on
S3+S4, significant, robust to paraphrase). Each block below is one test with a pass criterion fixed
before it runs. Keep the protocol from the pilot: freeze questions before running any new
configuration on them, compute gold from SQL, and never tune on a held-out set.

## Ground rules for every test

- **Preregister first.** Write the thresholds into the test's spec before the first run; changing
  them afterwards requires a new versioned output directory.
- **Two question sets.** A development set for building, and a held-out set written and committed
  before any configuration is run on it. Also add a paraphrase for every item.
- **Statistics.** Paired bootstrap (10,000 resamples) with Holm correction across the
  configurations compared; report effect sizes with 95% CIs, not only pass/fail.
- **Power.** Size question sets for the effect you want to detect. With n = 40 the CI half-width
  was about ±0.13; detecting +0.10 needs about 80–100 items per pooled comparison.
- **Reproducibility.** Every run records the corpus hash, model revisions, input file hashes and
  row counts; runs are deterministic end to end.

## T1 — Scale to 300–400 documents

- **Goal:** show the structure-aware advantage holds at realistic corpus size.
- **Build:** extend the fact database (about 8 regions, 80–100 stores, 15 departments, 60
  categories, 25 suppliers, 20 promotions, 10 policies, plus archive/draft pages), re-render, and
  regenerate both question sets (at least 40 base items per stratum).
- **Metrics:** nDCG@10 plus recall@50 and set-level F1, because answer sets at this size often
  exceed 10 documents.
- **Pass:** structure-aware − flat on pooled S3+S4 has CI lower bound > 0 (Holm-significant) on
  the held-out set, and survives paraphrase (C3 rule unchanged).

## T2 — Hierarchy traversal without parent names in child metadata

- **Goal:** test hierarchy questions that can only be answered by following the structure.
- **Build:** a corpus variant where stores do not carry `region` and categories do not carry
  `department` in their own frontmatter; the relation exists only as the authored `parent` link
  and the parent page's child list.
- **Configurations:** a parent-aware structured configuration (resolve the parent from the question,
  then return its children via authored edges) against flat.
- **Pass:** S4 improvement over flat with CI lower bound > 0; report the hierarchy oracle ceiling
  alongside.

## T3 — Numeric and date conditions

- **Goal:** answer "opened after 2015", "larger than 100,000 sq ft", "runs in December" from
  structured fields rather than text matching.
- **Build:** decouple store format from opening year in the fact data; add a query-to-filter
  component (rule-based first, then LLM-assisted) that turns numeric/date conditions into
  frontmatter filters combined with the metadata ranking.
- **Pass:** on numeric/date S3 items, the filter configuration beats both flat and the current
  structure-aware ranking with CI lower bound > 0.

## T4 — Multi-hop (S2) fusion

- **Goal:** keep flat-level quality on two-hop questions while retaining the structure-aware gain
  on S3/S4.
- **Build:** fusion of structure-aware and dense rankings (RRF with k tuned on the development
  set only), and a two-step variant (resolve entity → follow authored link).
- **Pass:** S2 not worse than flat by more than 0.05 (CI lower bound of fusion − flat > −0.05) and
  S3+S4 gain still significant.

## T5 — Contextual structured chunks and link expansion

- **Goal:** make the other structure-aware configurations competitive.
- **Build:** contextual chunk embeddings that prefix only the section heading path (not the page
  title, so parent pages don't crowd out children); link expansion weighted by edge type and seeded
  from the structure-aware ranking instead of dense.
- **Pass:** each variant's S3+S4 delta vs flat reported with CI; a variant is kept if its CI lower
  bound > 0.

## T6 — Extracted vs authored structure

- **Goal:** the original project question: do LLM-extracted links match authored links?
- **Build:** extract a graph with the pinned Qwen3-4B (GPU), measure index-time tokens, and run the
  identical link-expansion configuration on authored vs extracted edges.
- **Pass:** report the authored − extracted difference with CI, plus extraction cost.

## T7 — Answer quality and faithfulness

- **Goal:** show better retrieval turns into better answers.
- **Build:** generate answers for both paths with the same generator and budget (the web bench
  already does this), score with HHEM-2.1 and MiniCheck, and report abstention and false-answer
  rates on S5.
- **Pass:** structure-aware answers more often faithful and correct on S3/S4 (paired test, CI
  lower bound > 0); S5 abstention not lower than flat.

## T8 — Human review

- **Goal:** confirm the gold and the answer judgments.
- **Build:** two reviewers independently check a 20% stratified sample of questions, gold sets and
  generated answers.
- **Pass:** Cohen's κ ≥ 0.7; any gold error found triggers a versioned correction and a rerun.

## T9 — Efficiency

- **Measure for each configuration:** index build time, index size, query latency p50/p95, context
  tokens per answer, index-time LLM tokens (zero for authored structure).
- **Report:** a cost table next to the quality table.

## T10 — Real corpora

- **Goal:** move from the synthetic companion corpus to real documentation.
- **Build:** ingest github/docs and home-assistant.io (pinned commits), mine real user questions
  from their community forums, label gold documents by hand, and run the configurations that
  passed T1–T5.
- **Pass:** the structure-aware advantage on real questions, with the same criteria.

## Suggested order

T1 → T2 → T3 → T4 (corpus and questions first), then T5 and T6 (configurations), then T7 → T8
(answers and review), with T9 measured throughout and T10 last.
