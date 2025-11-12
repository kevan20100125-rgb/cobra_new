# cobra/quantize/pct/policy.py
"""
percentile_policy.py

Implements distribution-based best-percentile selection for clipping.
Used in cobra.quantize.pct.collectors to automatically choose the optimal
percentile when no manual percentile override is provided. Pair this with
`cobra.quantize.pct.range_calc.compute_clip_range` to translate the chosen
percentile into concrete (lo, hi) clip intervals.

References:
- Robust scaling using IQR-to-MAD conversion (S = 0.7413 * (p75 - p25))
- Tail distance (z-scores) and tail growth ratios (g-values)
"""

from typing import Dict, Any, Optional
import torch


def _robust_scale(p25: float, p75: float) -> float:
    """Robust scale S based on IQR → MAD conversion."""
    return max(0.7413 * (p75 - p25), 1e-6)


def select_best_percentile(stats: Dict[str, Any],
                           eps: float = 1e-6,
                           tau: float = 3.0,
                           r: float = 1.5) -> float:
    """
    Decide best percentile from distribution percentiles.

    stats: dict with keys { "percentiles": { "p25.0":..., "p50.0":..., ... } }
    Returns: best_p (float)
    """

    P = stats.get("percentiles", {})
    required = ["p25.0", "p50.0", "p75.0", "p90.0",
                "p99.0", "p99.9", "p99.99", "p99.999"]
    for k in required:
        if k not in P:
            raise ValueError(f"Missing percentile {k} in stats")

    S = _robust_scale(P["p25.0"], P["p75.0"])

    z99 = (P["p99.0"] - P["p50.0"]) / S
    z999 = (P["p99.9"] - P["p50.0"]) / S
    z9999 = (P["p99.99"] - P["p50.0"]) / S
    z99999 = (P["p99.999"] - P["p50.0"]) / S

    d99 = P["p99.0"] - P["p90.0"]
    d999 = P["p99.9"] - P["p99.0"]
    d9999 = P["p99.99"] - P["p99.9"]
    d99999 = P["p99.999"] - P["p99.99"]

    g999 = (d999) / (d99 + eps)
    g9999 = (d9999) / (d999 + eps)
    g99999 = (d99999) / (d9999 + eps)

    # Rule-based decision tree
    if (z999 <= 6) and (g999 <= r) and (g9999 <= r):
        p_base = 99.0
    elif g999 >= tau:
        p_base = 99.0
    elif g9999 >= tau:
        p_base = 99.9
    elif g99999 >= tau:
        p_base = 99.99
    elif (g999 <= r) and (g9999 <= r) and (g99999 <= r) and (z99999 >= 12):
        p_base = 99.999
    elif z9999 <= 9:
        p_base = 99.99
    else:
        p_base = 99.9

    return p_base


def decide_percentile(stats: Dict[str, Any],
                      override: Optional[float] = None) -> float:
    """
    Wrapper that chooses override if provided, otherwise use best-percentile.
    """
    if override is not None:
        return override
    return select_best_percentile(stats)


__all__ = ["select_best_percentile", "decide_percentile"]
