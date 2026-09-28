"""Answer scoring for Brightmart v4 T7 (answer quality) and T8 (human review).

Pure functions: no models, no files. Definitions are preregistered in
docs/brightmart_v4_prereg.md:
- abstention: the answer contains NOT FOUND;
- S1/S2 correctness: the reference answer's core (number, date range, name) occurs
  in the answer as a whole token (thousands separators ignored);
- S3/S4 correctness: F1 between the entities of the target type the answer names and
  the gold entities;
- faithful: HHEM score >= threshold, or an abstention.
"""
from __future__ import annotations

import re

from okf_rag.eval.stats import paired_bootstrap

ABSTAIN = "NOT FOUND"
_UNITS = re.compile(r"\s*(square feet|days|day|cents|cent|hours|hour)$")
_CITE = re.compile(r"\s*\[\d+\]")


def abstained(answer: str | None) -> bool:
    return answer is None or ABSTAIN in answer.upper()


def _norm(text: str) -> str:
    text = text.lower().replace("–", "-")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"\s*percent\b", "%", text)
    return " ".join(text.split())


def _has(text: str, needle: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(needle)}(?![\w])", text) is not None


def reference_cores(reference: str) -> list[str]:
    """Normalized pieces that must all occur: units dropped, '$' and '%' kept apart."""
    ref = _norm(reference)
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})", ref)
    if m:
        return [m.group(1), m.group(2)]
    ref = _UNITS.sub("", ref).lstrip("$").rstrip("%")
    return [ref]


def fact_correct(answer: str | None, reference: str) -> float:
    if abstained(answer):
        return 0.0
    text = _norm(answer)
    return float(all(_has(text, core) for core in reference_cores(reference)))


def mentioned(answer: str, candidates: dict[str, str]) -> set[str]:
    """Canonical names whose surface form (alias -> canonical) occurs in the answer."""
    text = _norm(answer)
    return {canon for alias, canon in candidates.items() if _has(text, _norm(alias))}


def set_f1(answer: str | None, gold: list[str], candidates: dict[str, str]) -> float:
    if abstained(answer) or not gold:
        return 0.0
    got = mentioned(answer, candidates)
    hit = len(got & set(gold))
    if not hit:
        return 0.0
    p, r = hit / len(got), hit / len(gold)
    return 2 * p * r / (p + r)


def strip_citations(answer: str) -> str:
    return " ".join(_CITE.sub("", answer).split())


def claim_sentences(answer: str) -> list[str]:
    text = strip_citations(answer)
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return parts or ([text] if text else [])


def faithful(hhem_score: float | None, answer: str | None, threshold: float) -> float:
    if abstained(answer):
        return 1.0
    return float(hhem_score is not None and hhem_score >= threshold)


def t7_verdict(rows: list[dict], threshold: float, n_boot: int = 10000, seed: int = 7) -> dict:
    """rows: one per (query_id, arm) with stratum, correct, faithful, abstained.

    Pass iff on S3+S4 the (R2s - R0) correctness and faithfulness deltas both have CI
    lower bound > 0 and on S5 R2s abstains at least as often as R0."""
    by = {(r["query_id"], r["arm"]): r for r in rows}
    qids = sorted({r["query_id"] for r in rows})

    def pair(strata, key):
        qs = [q for q in qids if (q, "R0") in by and by[(q, "R0")]["stratum"] in strata and (q, "R2s") in by]
        if not qs:
            return None
        res = paired_bootstrap([by[(q, "R0")][key] for q in qs], [by[(q, "R2s")][key] for q in qs],
                               n_boot=n_boot, seed=seed)
        return res | {"n": len(qs), "R0_mean": sum(by[(q, "R0")][key] for q in qs) / len(qs),
                      "R2s_mean": sum(by[(q, "R2s")][key] for q in qs) / len(qs)}

    s34_correct = pair(("S3", "S4"), "correct")
    s34_faithful = pair(("S3", "S4"), "faithful")
    s12_correct = pair(("S1", "S2"), "correct")
    s5 = pair(("S5",), "abstained")
    return {
        "threshold_hhem": threshold,
        "S34_correct_F1": s34_correct,
        "S34_faithful": s34_faithful,
        "S12_correct": s12_correct,
        "S5_abstention": s5,
        "pass": bool(s34_correct and s34_faithful and s5 and s34_correct["ci_low"] > 0
                     and s34_faithful["ci_low"] > 0 and s5["delta"] >= 0),
    }


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters' labels on the same items."""
    if len(a) != len(b) or not a:
        raise ValueError("two equally long, non-empty label lists required")
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
