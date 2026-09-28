"""Rank-based IR metrics for retrieval evaluation per stratum.

Computes precision, recall, MRR, and nDCG@k for ranked results against gold
evidence. Per-stratum policy implements constraints D7 (S4 cutoffs), D8
(chunk-level metrics for S1 only), and D13 (S5 abstention).

This module scores ranked lists using fixed k denominators; it differs from
evidence_coverage_v3.set_retrieval_metrics, which scores unique-document sets.

Every metric function deduplicates its ranked list on entry, preserving first
occurrence (`list(dict.fromkeys(ranked))`), matching the convention in
evidence_coverage_v3 (line 91). Callers do not need to pre-deduplicate: a
ranking with repeated ids - e.g. one entry per retrieved chunk, with several
chunks sharing a document path, which is what scripts/run_inference_v4.py
produces - scores exactly as its deduplicated form would.
"""

import math


def precision_at_k(ranked, gold, k):
    """Precision@k: fraction of top k that are in gold.

    Raises ValueError if k is not a positive integer.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    ranked = list(dict.fromkeys(ranked))
    top = ranked[:k]
    return sum(1 for d in top if d in gold) / k


def recall_at_k(ranked, gold, k):
    """Recall@k: fraction of gold documents retrieved in top k.

    Raises ValueError if k is not a positive integer.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not gold:
        return 0.0
    ranked = list(dict.fromkeys(ranked))
    return sum(1 for d in ranked[:k] if d in gold) / len(gold)


def mrr_at_k(ranked, gold, k):
    """Mean Reciprocal Rank@k: 1/rank of first hit, or 0 if no hit.

    Raises ValueError if k is not a positive integer.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    ranked = list(dict.fromkeys(ranked))
    for i, d in enumerate(ranked[:k], start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, gold, k):
    """Normalized DCG@k with binary gains.

    Raises ValueError if k is not a positive integer.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    ranked = list(dict.fromkeys(ranked))
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


STRATUM_POLICY = {
    "s1": {"ks": [1, 5, 10], "chunk_level": True, "ir": True},
    "s2": {"ks": [1, 5, 10], "ir": True},
    "s3": {"ks": [5, 10, 20], "ir": True},
    "s4": {"ks": [20, 50], "ir": True},  # D7: subtree gold too large for @10
    "s5": {"ir": False},  # D13: abstention metrics only
    "mined": {"ks": [1, 5, 10], "ir": True},
}


def evaluate_query(ranked, gold, stratum) -> dict:
    """Score ranked results against gold per stratum policy.

    Args:
        ranked: list of doc ids (any hashable type) in best-first order
        gold: set of doc ids (any hashable type) that are correct
        stratum: S1-S5 or s1-s5; case-insensitive

    Returns:
        dict of metric names to floats (P@k, R@k, nDCG@k, MRR@10);
        empty dict if stratum has ir: False.

    Raises:
        ValueError if stratum is unknown.
    """
    # Normalize stratum case to lowercase
    stratum_key = stratum.lower()

    # Raise if unknown
    if stratum_key not in STRATUM_POLICY:
        known = ", ".join(sorted(STRATUM_POLICY.keys()))
        raise ValueError(f"Unknown stratum '{stratum}'. Known strata: {known}")

    policy = STRATUM_POLICY[stratum_key]
    if not policy.get("ir"):
        return {}

    out = {}
    for k in policy["ks"]:
        out[f"P@{k}"] = precision_at_k(ranked, gold, k)
        out[f"R@{k}"] = recall_at_k(ranked, gold, k)
        out[f"nDCG@{k}"] = ndcg_at_k(ranked, gold, k)
    out["MRR@10"] = mrr_at_k(ranked, gold, 10)
    return out
