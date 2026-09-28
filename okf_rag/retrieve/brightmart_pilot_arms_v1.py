"""Pure helpers for the Brightmart pilot run (spec 2026-09-24 phase 2).

Nothing here touches Postgres, models, or files. The verdict thresholds are
preregistered in THRESHOLDS and copied into every verdict; changing them
after a run requires a new versioned output directory.
"""
from __future__ import annotations

from okf_rag.eval.ir_metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from okf_rag.eval.stats import holm, paired_bootstrap

STRUCTURE_ARMS = ("R1", "R2", "R3a", "R4")
THRESHOLDS = {
    "metric": "nDCG@10",
    "n_boot": 10000,
    "seed": 7,
    "c1_margin": -0.05,
    "c2_alpha": 0.05,
    "c3_ratio": 0.5,
}


def _fmt(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def metadata_pseudo_docs(documents: list[dict]) -> list[dict]:
    """One lexical pseudo-document per doc: title, description, tags and
    every frontmatter key rendered as `key: value`. This is the R2 metadata
    surface; it carries only authored metadata, never gold."""
    out = []
    for d in documents:
        lines = [d["title"], d.get("description") or "", " ".join(d.get("tags") or [])]
        lines += [f"{k}: {_fmt(v)}" for k, v in sorted((d.get("frontmatter") or {}).items())]
        out.append({"chunk_id": f"meta:{d['path']}", "doc_path": d["path"],
                    "text": "\n".join(x for x in lines if x)})
    return out


def collapse_to_docs(ranked_doc_paths_by_chunk: list[str], k_docs: int) -> list[str]:
    return list(dict.fromkeys(ranked_doc_paths_by_chunk))[:k_docs]


def doc_metrics(ranked: list[str], gold: set[str]) -> dict[str, float]:
    return {
        "P@1": precision_at_k(ranked, gold, 1),
        "P@5": precision_at_k(ranked, gold, 5),
        "P@10": precision_at_k(ranked, gold, 10),
        "R@5": recall_at_k(ranked, gold, 5),
        "R@10": recall_at_k(ranked, gold, 10),
        "MRR@10": mrr_at_k(ranked, gold, 10),
        "nDCG@10": ndcg_at_k(ranked, gold, 10),
    }


def _pair(scores: dict[str, dict[str, float]], a: str, b: str) -> dict:
    qids = sorted(scores[a])
    return paired_bootstrap([scores[a][q] for q in qids], [scores[b][q] for q in qids],
                            n_boot=THRESHOLDS["n_boot"], seed=THRESHOLDS["seed"])


def verdict(scores: dict[str, dict[str, dict[str, float]]], arms=STRUCTURE_ARMS) -> dict:
    """Apply the three preregistered criteria.

    C1 (S1 templates): for every structure arm A, CI low of (R0 - A) > c1_margin.
    C2 (S3+S4 templates): paired (A - R0) per arm, Holm over the four p-values;
       pass iff >=1 arm with delta > 0, CI low > 0 and Holm-significant.
    C3 (S3+S4 paraphrases): for each C2 winner, CI low of (A - R0) > 0 and
       delta_p >= c3_ratio * delta_t; pass iff >=1 winner survives. FAIL if C2 failed.
    """
    c1 = {a: _pair(scores["S1_t"], a, "R0") for a in arms}  # b - a = R0 - A
    c1_pass = all(r["ci_low"] > THRESHOLDS["c1_margin"] for r in c1.values())

    c2 = {a: _pair(scores["S34_t"], "R0", a) for a in arms}  # A - R0
    sig = holm({a: r["p"] for a, r in c2.items()}, alpha=THRESHOLDS["c2_alpha"])
    winners = [a for a in arms if c2[a]["delta"] > 0 and c2[a]["ci_low"] > 0 and sig[a]]

    if not winners:
        c3 = {"pass": False, "reason": "criterion 2 failed", "arms": {}}
    else:
        c3_arms = {}
        for a in winners:
            r = _pair(scores["S34_p"], "R0", a)
            ok = r["ci_low"] > 0 and r["delta"] >= THRESHOLDS["c3_ratio"] * c2[a]["delta"]
            c3_arms[a] = {**r, "template_delta": c2[a]["delta"], "pass": ok}
        c3 = {"pass": any(x["pass"] for x in c3_arms.values()), "reason": "", "arms": c3_arms}

    return {
        "thresholds": dict(THRESHOLDS),
        "C1": {"pass": c1_pass, "arms": c1},
        "C2": {"pass": bool(winners), "arms": {a: {**r, "holm_significant": sig[a]} for a, r in c2.items()},
               "winning_arms": winners},
        "C3": c3,
    }
