"""Paired bootstrap vs the R0 control; Holm-Bonferroni across strata (spec s11).

Produces confidence intervals and p-values for per-stratum arm differences.
Each arm is compared against R0 within a stratum; strata are corrected for
multiple testing using the step-down Holm-Bonferroni procedure.
bootstrap_mean_ci() gives the analogous percentile-bootstrap CI for the mean
of a single sample, used for rows (e.g. the baseline arm) that have no paired
comparison to report a delta or p-value for.

This module does not claim to be a general statistics library. It implements
the exact resampling and multiple-testing protocol required by the evaluation.
"""

import numpy as np


def paired_bootstrap(a, b, n_boot: int = 10000, seed: int = 7) -> dict:
    """Paired bootstrap CI and p-value for mean difference b - a.

    Resamples paired observations with replacement and computes the
    distribution of mean differences. Confidence interval bounds are
    the 2.5 and 97.5 percentiles of the bootstrap distribution.
    P-value is two-sided: min(P(boot <= 0), P(boot >= 0)) * 2, capped at 1.0.
    This counts a bootstrap resample mean of exactly 0 in both tails, which
    is conservative; for the discrete-valued metrics used here (P@k over a
    small k takes only a handful of distinct values), exact-zero bootstrap
    means are common, so this matters in practice, not just in theory.

    Args:
        a: array-like of numeric values (first arm, e.g., R0 baseline)
        b: array-like of numeric values (second arm, same length as a)
        n_boot: positive integer, number of bootstrap resamples (default 10000)
        seed: integer random seed for reproducibility (default 7)

    Returns:
        dict with keys:
            - "delta": float, mean difference (b - a)
            - "ci_low": float, lower CI bound (2.5th percentile)
            - "ci_high": float, upper CI bound (97.5th percentile)
            - "p": float, two-sided p-value [0, 1]

    Raises:
        ValueError if a and b have different lengths, are empty, or n_boot <= 0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same length: got {a.shape} and {b.shape}")
    if len(a) == 0:
        raise ValueError("a and b cannot be empty")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")

    diff = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)

    delta = float(diff.mean())
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))

    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    p = min(float(p), 1.0)

    return {
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p": p,
    }


def bootstrap_mean_ci(values, n_boot: int = 10000, seed: int = 7) -> dict:
    """Percentile bootstrap CI for the mean of a single sample.

    Resamples values with replacement and computes the distribution of
    resample means. Confidence interval bounds are the 2.5 and 97.5
    percentiles of that distribution.

    Args:
        values: array-like of numeric values
        n_boot: positive integer, number of bootstrap resamples (default 10000)
        seed: integer random seed for reproducibility (default 7)

    Returns:
        dict with keys:
            - "ci_low": float, lower CI bound (2.5th percentile)
            - "ci_high": float, upper CI bound (97.5th percentile)

    Raises:
        ValueError if values is empty, or n_boot <= 0.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        raise ValueError("values cannot be empty")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)

    return {
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    """Step-down Holm-Bonferroni multiple testing correction.

    Tests are ordered by p-value (ascending). The i-th smallest p-value
    is compared against alpha / (m - i), where m is the number of tests.
    Once a test fails, all remaining tests are marked non-significant.

    Args:
        pvals: dict mapping test name (str) to p-value (float in [0, 1])
        alpha: significance level (default 0.05)

    Returns:
        dict mapping test name to boolean (True if significant after correction,
        False otherwise). Keys match input pvals.
    """
    if not pvals:
        return {}

    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    blocked = False

    for i, (name, p) in enumerate(items):
        if blocked or p > alpha / (m - i):
            blocked = True
            out[name] = False
        else:
            out[name] = True

    return out
