"""T8 human review for Brightmart v4: stratified sample sheet and Cohen's kappa.

    sample   20% stratified sample of the held-out templates (8 per stratum), with gold
             and both generated answers (from `llm_brightmart_v4.py answer`), written as
             one CSV per reviewer with empty judgment columns.
    kappa    Cohen's kappa per judgment column between two filled sheets; pass iff every
             kappa >= 0.7 (docs/brightmart_v4_prereg.md, T8).

Judgment columns take y or n: gold_ok (the gold answer and documents are right),
R0_correct and R2s_correct (the generated answer is correct and supported).

Usage:
    python scripts/review_brightmart_v4.py sample
    python scripts/review_brightmart_v4.py kappa --a results/brightmart-v4/t8/review_A.csv --b results/brightmart-v4/t8/review_B.csv
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Run this repository's packages even if another checkout providing `okf_rag` is
# pip-installed: `python scripts/x.py` puts scripts/, not the repo root, on sys.path.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from okf_rag.eval.answer_scoring_v4 import cohen_kappa

SEED = 20260924
PER_STRATUM = 8
KAPPA_MIN = 0.7
JUDGMENTS = ("gold_ok", "R0_correct", "R2s_correct")
FIELDS = ["query_id", "stratum", "question", "gold_answer", "gold_docs", "answer_R0", "answer_R2s", *JUDGMENTS, "notes"]
OUT = Path("results/brightmart-v4/t8")


def sample_rows(questions: list[dict], answers: dict[tuple[str, str], str], per_stratum: int = PER_STRATUM,
                seed: int = SEED) -> list[dict]:
    """Deterministic stratified sample of template questions."""
    by = defaultdict(list)
    for q in questions:
        if q["query_id"].endswith("-t"):
            by[q["stratum"]].append(q)
    rng = random.Random(seed)
    rows = []
    for st in sorted(by):
        for q in sorted(rng.sample(sorted(by[st], key=lambda x: x["query_id"]), per_stratum), key=lambda x: x["query_id"]):
            rows.append({"query_id": q["query_id"], "stratum": st, "question": q["text"],
                         "gold_answer": q["reference_answer"] or "(unanswerable)",
                         "gold_docs": " | ".join(q["source_paths"]),
                         "answer_R0": answers.get((q["query_id"], "R0"), ""),
                         "answer_R2s": answers.get((q["query_id"], "R2s"), ""),
                         **{j: "" for j in JUDGMENTS}, "notes": ""})
    return rows


def kappas(a_rows: list[dict], b_rows: list[dict]) -> dict:
    b_by = {r["query_id"]: r for r in b_rows}
    shared = [r for r in a_rows if r["query_id"] in b_by]
    out = {}
    for j in JUDGMENTS:
        pairs = [(r[j].strip().lower(), b_by[r["query_id"]][j].strip().lower()) for r in shared
                 if r[j].strip() and b_by[r["query_id"]][j].strip()]
        out[j] = {"n": len(pairs), "kappa": cohen_kappa([x for x, _ in pairs], [y for _, y in pairs]) if pairs else None}
    out["pass"] = all(v["kappa"] is not None and v["kappa"] >= KAPPA_MIN for k, v in out.items() if k in JUDGMENTS)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="T8 review sample and agreement.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--answers", default="results/brightmart-v4/t7/answers_heldout.jsonl")
    s.add_argument("--out-dir", default=str(OUT))
    k = sub.add_parser("kappa")
    k.add_argument("--a", required=True)
    k.add_argument("--b", required=True)
    args = ap.parse_args()

    if args.cmd == "sample":
        qs = [json.loads(x) for x in Path("doc_synthetic/brightmart_v4_heldout_questions.jsonl")
              .read_text(encoding="utf-8").splitlines() if x.strip()]
        answers = {}
        if Path(args.answers).exists():
            for line in Path(args.answers).read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                answers[(r["query_id"], r["arm"])] = r["answer"]
        rows = sample_rows(qs, answers)
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for reviewer in ("A", "B"):
            with (out / f"review_{reviewer}.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
        print(f"wrote {len(rows)} items to {out}/review_A.csv and review_B.csv")
    else:
        read = lambda p: list(csv.DictReader(Path(p).open(encoding="utf-8")))
        res = kappas(read(args.a), read(args.b))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
