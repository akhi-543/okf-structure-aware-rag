# OKF Final

Structure-aware vs flat retrieval on the Brightmart synthetic retail corpus.

**Result:** structure-aware retrieval over normalized document metadata improves nDCG@10 by
+0.299 over flat retrieval on structure-dependent questions (held-out, Holm-significant), and by
+0.312 on paraphrased questions.

- Pipeline, results and future work: [`docs/okf_final_pipeline_and_findings.md`](docs/okf_final_pipeline_and_findings.md)
- Next-iteration test plan: [`docs/next_iteration_test_plan.md`](docs/next_iteration_test_plan.md)
- Next iteration (v4): results [`docs/brightmart_v4_results.md`](docs/brightmart_v4_results.md) —
  T1–T5 pass at 348 documents, T6 authored ≈ extracted links, T7 fails (answers more correct, not more
  faithful), T8 human-reviewed with no errors found; preregistration [`docs/brightmart_v4_prereg.md`](docs/brightmart_v4_prereg.md);
  full run on Kaggle: [`notebooks/okf_v4_kaggle_full_run.ipynb`](notebooks/okf_v4_kaggle_full_run.ipynb)
- Agent / contributor briefing: [`AGENTS.md`](AGENTS.md)
- Corpus and question generators: [`doc_synthetic/`](doc_synthetic/)
- Web bench (flat vs structured, side by side): [`okf_web_bench/`](okf_web_bench/)

Quick start:

```bash
pip install -e ".[dev,db,ml,web]"
python -m pytest -q
docker compose -f docker/docker-compose.yml up -d   # Postgres + pgvector on port 5434
```

The full run order is in section 5.2 of the findings document.

## Author

Akhilesh Teja Goli
