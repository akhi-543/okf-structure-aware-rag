# Brightmart Synthetic Retail Benchmark

## Purpose

A synthetic retail corpus for testing whether authored document structure improves retrieval over
flat retrieval of the same prose. The study tests three preregistered hypotheses:

1. Flat retrieval (dense embeddings over flat chunks) remains competitive on direct-fact questions (S1).
2. Structure helps on structure-dependent questions (S3 metadata sets, S4 hierarchy sets).
3. The effect survives natural-language paraphrases of the questions.

Corpus ID (Postgres bundle): `synthetic_retail_pilot`

Results, pipeline and next steps: `docs/okf_final_pipeline_and_findings.md`.

## What is scripted vs. LLM-written

**Scripted (deterministic, validated):**
- Database schema and all fact data (stores, regions, departments, categories, suppliers,
  promotions, policies, weekly sales, archive records)
- Fact sentences rendered from templates (at least 3 variants per fact, chosen deterministically)
- Frontmatter, `parent` links and body links (bundle-absolute paths)
- Questions' gold: SQL queries, entity-to-document mapping, answer spans taken from the rendered
  body
- S5 unanswerable questions: absent terms verified absent from the whole corpus

**LLM-written:**
- `## Notes` filler prose (validated: no digits, no other entity's name, 60–160 words)
- Paraphrases (one per base question; no 4-word sequence shared with the template; same answer
  and gold)

**Validator guarantees** (`brightmart_validate.py`, and `--heldout` for the held-out set):
- 200 records per question set: 100 templates (20 per stratum S1–S5) + 100 paraphrases
- Paraphrases share no 4-word sequence with their template; paraphrase gold equals template gold
- S5 absent terms truly absent (Notes included) and without gold documents
- S1/S2 spans are real body lines (not frontmatter, not empty) and contain the answer; the S2
  answer is in the second-hop document
- S3/S4 gold equals the re-executed SQL; S4 records carry the home → parent path
- Every document's parent resolves; the files on disk equal the render byte for byte

## File map

```
doc_synthetic/
├── corpus/                               # 53 rendered markdown docs
│   ├── brightmart-home.md
│   ├── regions/                          # 4 region + 12 store docs
│   ├── departments/                      # 8 department + 10 category docs
│   ├── suppliers/                        # 6 supplier docs
│   ├── promotions/                       # 5 promotion docs
│   ├── policies/                         # 3 policy docs
│   ├── archive/                          # 3 archive/draft docs with stale values (distractors)
│   └── schema/                           # 1 database schema doc
├── brightmart_llm_prose/                 # 53 .prose.txt files (Notes filler)
├── brightmart_schema.sql                 # fact database DDL
├── brightmart_seed.py                    # fact data + connect() (in-memory SQLite)
├── brightmart_render.py                  # database → corpus
├── brightmart_validate.py                # corpus + question validation
├── brightmart_questions_build.py         # development questions and gold
├── brightmart_paraphrases.json           # development paraphrases (100)
├── brightmart_questions.jsonl            # development questions (200)
├── brightmart_qrels.jsonl                # development relevance judgments (356)
├── brightmart_gold_spans.jsonl           # development spans, SQL, paths, absent terms (542)
├── brightmart_heldout_build.py           # held-out questions and gold
├── brightmart_heldout_paraphrases.json   # held-out paraphrases (100)
├── brightmart_heldout_questions.jsonl    # held-out questions (200)
├── brightmart_heldout_qrels.jsonl        # held-out relevance judgments (336)
└── brightmart_heldout_gold_spans.jsonl   # held-out spans, SQL, paths, absent terms (524)
```

## Regeneration commands

Deterministic, seed `20260924`; repeated runs are byte-identical.

```bash
python doc_synthetic/brightmart_render.py              # corpus
python doc_synthetic/brightmart_questions_build.py     # development set (adds -p records from the paraphrases file)
python doc_synthetic/brightmart_heldout_build.py       # held-out set; refuses any text shared with the development set
python doc_synthetic/brightmart_validate.py
python doc_synthetic/brightmart_validate.py --heldout
python -m pytest tests/test_brightmart_questions.py tests/test_brightmart_heldout.py -q
```

## Question strata (20 base + 20 paraphrase each, per set)

- **S1** direct fact (store size, opening year, city, department manager, quarterly sales,
  promotion discount and dates, policy values, supplier country)
- **S2** two hops (category → supplier country; store → regional manager; category → promotion
  discount; department → category supplier; category → department group)
- **S3** metadata sets (stores by pharmacy, fuel, format, status, region, opening year, size;
  promotions by format or discount; categories by supplier)
- **S4** hierarchy sets (regions or departments → children, including multi-parent questions)
- **S5** unanswerable (absent entities and absent attributes)

## Distractors

- Archive/draft pages with stale values: an older Cedar Falls store profile (150,000 sq ft vs
  171,000 current), a 2026 draft of the Spring Garden Days promotion (30% vs 20%), and an older
  Returns Policy (60-day vs 30-day window)
- Look-alike names: Cedar Falls / Cedar Park, Riverside / Riverdale, Greenfield Farms Co-op /
  Greenfield Home Supply
- Status variety: stores that are remodeling or closed (Bayou Gate closed after week 30 of 2025)

## Retrieval experiment

Ingest, embeddings, retrieval configurations and the verdict are documented in
`docs/okf_final_pipeline_and_findings.md`.
