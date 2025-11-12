# cobra/pipeline/pct_schema.py
"""
Percentile-clipping schema utilities.

Provides:
- normalize_stage(name) -> canonical stage name in {"vision.dino","vision.siglip","llm","projector"}
- validate_observer_payload(d) -> raises AssertionError on schema violation
- validate_export_file(stats) -> raises AssertionError on schema violation
- compute_affine_params(x_min, x_max, bits, signed) -> (scale: float, zero_point: int)

Design notes:
- "Percentile keys" are those starting with 'p' (e.g., "p99.9", "p99_99"). At least one must exist.
- Targets must be normalized to the canonical four-class vocabulary to keep collect/apply aligned.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, Mapping, Optional, Tuple, TypedDict

ALLOWED_TARGETS = {"vision.dino", "vision.siglip", "llm", "projector"}

# Accept both dot and underscore decimals, e.g., p99.9 or p99_9
_PCT_KEY_RE = re.compile(r"^p\d+(?:[._]\d+)?$", re.IGNORECASE)


def _clean_name(s: str) -> str:
    """Basic cleanup before rule mapping."""
    s = s.strip().lower()
    # Normalize path-like separators to dots
    s = s.replace("/", ".")
    s = re.sub(r"\s+", "", s)
    # Collapse duplicate dots
    s = re.sub(r"\.{2,}", ".", s)
    # Drop very common trailing suffixes that add noise
    s = re.sub(r"\.(out|featurizer|features|head|mlp)$", "", s)
    return s


def normalize_stage(name: str) -> str:
    """
    Map arbitrary module/alias/path to one of:
      {"vision.dino","vision.siglip","llm","projector"}

    Rules (priority order):
      1) Contains "dino"  -> "vision.dino"
      2) Contains "siglip"-> "vision.siglip"
      3) Projector keywords -> "projector"
      4) LLM keywords ("mamba","llm","decoder","language") or generic "backbone"
         that does not already match dino/siglip/projector -> "llm"
    Raises:
      ValueError if no rule applies.
    """
    s = _clean_name(name)

    if "dino" in s:
        return "vision.dino"
    if "siglip" in s:
        return "vision.siglip"

    # projector variants frequently seen in VLM codebases
    projector_hits = ("projector", "mm_projector", "proj.")
    if any(k in s for k in projector_hits) or s.endswith(".proj"):
        return "projector"

    # LLM side; guard against vision collisions already handled above
    if any(k in s for k in ("mamba", "llm", "decoder", "language")):
        return "llm"

    # Heuristic: a generic "backbone" that isn't already vision.* heuristics goes to llm
    if ("backbone" in s) and not any(k in s for k in ("dino", "siglip", "projector", "mm_projector", "proj.")):
        return "llm"

    raise ValueError(f"normalize_stage: unable to classify '{name}' into {sorted(ALLOWED_TARGETS)}")


def _require_keys(d: Mapping, keys: Iterable[str], ctx: str) -> None:
    missing = [k for k in keys if k not in d]
    assert not missing, f"{ctx}: missing keys {missing}; found {sorted(d.keys())}"


def _has_percentile_keys(d: Mapping) -> bool:
    return any(isinstance(k, str) and _PCT_KEY_RE.match(k) for k in d.keys())


def _expect_number(x, name: str, ctx: str) -> None:
    assert isinstance(x, (int, float)), f"{ctx}: '{name}' must be number, got {type(x).__name__}"


def validate_observer_payload(d: Mapping) -> None:
    """
    Validate a single target's observer payload.

    Required keys:
      - "mode": str
      - "target": one of ALLOWED_TARGETS (normalized)
      - "module": str (free-form descriptor)
      - "numel": int >= 0
      - "min": float
      - "max": float
      - at least one percentile key starting with 'p' (e.g., 'p99.9', 'p99_99')

    Raises AssertionError with actionable message if invalid.
    """
    ctx = "validate_observer_payload"
    _require_keys(d, ("mode", "target", "module", "numel", "min", "max"), ctx)

    assert isinstance(d["mode"], str), f"{ctx}: 'mode' must be str"
    assert isinstance(d["module"], str), f"{ctx}: 'module' must be str"
    _expect_number(d["numel"], "numel", ctx)
    _expect_number(d["min"], "min", ctx)
    _expect_number(d["max"], "max", ctx)

    # Numeric sanity
    numel = int(d["numel"])
    assert numel >= 0, f"{ctx}: 'numel' must be >= 0"
    x_min = float(d["min"])
    x_max = float(d["max"])
    assert math.isfinite(x_min) and math.isfinite(x_max), f"{ctx}: min/max must be finite"
    assert x_max >= x_min, f"{ctx}: max ({x_max}) must be >= min ({x_min})"

    # Target normalization check
    target = str(d["target"])
    assert target in ALLOWED_TARGETS, f"{ctx}: 'target' must be one of {sorted(ALLOWED_TARGETS)}, got '{target}'"

    # At least one percentile entry
    assert _has_percentile_keys(d), f"{ctx}: no percentile keys found (expected keys like 'p99.9' or 'p99_99')"

    # Optional: if present, ensure common percentile keys are numeric
    for k, v in d.items():
        if isinstance(k, str) and _PCT_KEY_RE.match(k):
            _expect_number(v, k, ctx)


def validate_export_file(stats: Mapping) -> None:
    """
    Validate the full exported stats object loaded from torch.load(...).

    Required top-level keys:
      - "config": Mapping (free-form)
      - "targets": Iterable[str] of normalized stage names
      - "observers": Mapping[str -> Mapping] where each value passes validate_observer_payload()

    Also checks:
      - listed targets are subset of ALLOWED_TARGETS
      - each observer's 'target' is consistent with its normalized form
    """
    ctx = "validate_export_file"
    _require_keys(stats, ("config", "targets", "observers"), ctx)

    # targets
    targets = stats["targets"]
    assert isinstance(targets, (list, tuple, set)), f"{ctx}: 'targets' must be a list/tuple/set"
    for t in targets:
        assert isinstance(t, str), f"{ctx}: target entries must be str"
        assert t in ALLOWED_TARGETS, f"{ctx}: unknown target '{t}', allowed={sorted(ALLOWED_TARGETS)}"

    # observers
    observers = stats["observers"]
    assert isinstance(observers, Mapping), f"{ctx}: 'observers' must be a mapping"
    assert len(observers) > 0, f"{ctx}: 'observers' is empty"

    for key, payload in observers.items():
        assert isinstance(key, str), f"{ctx}: observer key must be str"
        assert isinstance(payload, Mapping), f"{ctx}: observer payload for '{key}' must be a mapping"
        validate_observer_payload(payload)

        # Consistency: payload.target should already be normalized; if not, normalize and compare
        t_payload = str(payload.get("target", ""))
        try:
            t_norm = normalize_stage(t_payload)
        except Exception as e:
            raise AssertionError(f"{ctx}: payload.target invalid for '{key}': {e}")
        assert t_payload == t_norm, f"{ctx}: payload.target='{t_payload}' is not normalized ('{t_norm}')"
        # If observer map uses target names as keys, allow mismatch but encourage alignment
        if key in ALLOWED_TARGETS:
            assert key == t_payload, f"{ctx}: observer key '{key}' != payload.target '{t_payload}'"


def _qmin_qmax(bits: int, signed: bool) -> Tuple[int, int]:
    assert isinstance(bits, int) and bits > 0, "bits must be positive int"
    if signed:
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
    else:
        qmin = 0
        qmax = (1 << bits) - 1
    return qmin, qmax


def compute_affine_params(x_min: float, x_max: float, bits: int, signed: bool) -> Tuple[float, int]:
    """
    Compute (scale, zero_point) for uniform affine quantization.

    scale = (x_max - x_min) / (qmax - qmin)   with epsilon safeguard
    zero  = round(qmin - x_min / scale), clamped to [qmin, qmax]

    Edge cases:
      - If x_max == x_min: use scale = 1.0 and map zero_point to mid-range.
    """
    _expect_number(x_min, "x_min", "compute_affine_params")
    _expect_number(x_max, "x_max", "compute_affine_params")
    assert x_max >= x_min, "compute_affine_params: x_max must be >= x_min"

    qmin, qmax = _qmin_qmax(bits, signed)

    if x_max == x_min:
        # Degenerate: avoid zero scale. Choose a safe default.
        scale = 1.0
        zero = int(max(min(-x_min, qmax), qmin))
        return float(scale), zero

    rng = float(x_max - x_min)
    denom = float(qmax - qmin)
    # Epsilon to avoid zero division if denom==0 (bits==1 signed is still fine; denom>=1)
    eps = 1e-12
    scale = max(rng / max(denom, eps), eps)

    # Standard asymmetric zero-point
    zero_fp = qmin - (x_min / scale)
    zero = int(round(zero_fp))
    zero = max(min(zero, qmax), qmin)

    # Final guards
    assert math.isfinite(scale) and scale > 0.0, "compute_affine_params: scale must be finite and > 0"
    return float(scale), int(zero)


class CalibrationEntry(TypedDict, total=False):
    scale: float | list | Tuple
    zero_point: Optional[float | list | Tuple]
    clip: Tuple[float, float]
    percentile: float
