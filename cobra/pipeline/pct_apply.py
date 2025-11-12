# cobra/pipeline/pct_apply.py
"""
pct_apply.py

Read percentile activation statistics (.pt) and compute final clipping
configuration for the four canonical modules:
    "vision.siglip", "vision.dino", "llm", "projector"

Outputs:
  - JSON file (default: outputs/percentile_overrides.json)
    {
      "vision.siglip": {"percentile": 99.9, "lo": -3.15, "hi": 3.14},
      "vision.dino":   {"percentile": 99.99, "lo": -2.86, "hi": 2.85},
      "llm":           {"percentile": 99.9, "lo": -1.42, "hi": 1.40},
      "projector":     {"percentile": 99.999, "lo": -4.80, "hi": 4.78}
    }

This JSON can then be consumed by quantizer initialization or clipping modules
before low-bit conversion.

Usage:
    python -m cobra.pipeline.pct_apply --stats outputs/percentile_stats.pt --out outputs/percentile_overrides.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from cobra.models.load import load as cobra_load
from cobra.pipeline.pct_schema import (
    validate_export_file,
    normalize_stage,
    compute_affine_params,
)
from cobra.quantize.pct import decide_percentile, compute_clip_range
from cobra.quantize.pct.calibrator import build_table_from_file
from quantize.quantizer import UniformAffineQuantizer, apply_calibration_table


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

PreferredPercentiles = ("p99.99", "p99.9", "p99.0")


def _normalize_percentiles_map(percentiles: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in percentiles.items():
        if value is None:
            continue
        name = str(key).replace("_", ".")
        if not name.startswith("p"):
            name = "p" + name
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _entry_to_stats(entry: Mapping[str, Any]) -> Dict[str, Any]:
    percentiles = _normalize_percentiles_map(entry.get("percentiles", {}) if isinstance(entry, Mapping) else {})
    for key, value in entry.items():
        if not isinstance(key, str) or not key.lower().startswith("p"):
            continue
        name = key.replace("_", ".")
        if not name.startswith("p"):
            name = "p" + name
        if name in percentiles:
            continue
        try:
            percentiles[name] = float(value)
        except (TypeError, ValueError):
            continue
    return {
        "percentiles": percentiles,
        "min": float(entry.get("min", 0.0) or 0.0),
        "max": float(entry.get("max", 0.0) or 0.0),
    }


def _convert_legacy_summary(data: Mapping[str, Any]) -> Dict[str, Any]:
    observers: Dict[str, Dict[str, Any]] = {}
    targets: List[str] = []
    meta = data.get("meta", {})
    for key, value in data.items():
        if key == "meta":
            continue
        if not isinstance(value, Mapping):
            continue
        try:
            stage = normalize_stage(key)
        except Exception:
            stage = key
        if stage in observers:
            continue
        entry = {
            "mode": "legacy",
            "percent": 0.0,
            "numel": int(value.get("numel", 0) or 0),
            "target": stage,
            "module": key,
            "min": float(value.get("min", 0.0) or 0.0),
            "max": float(value.get("max", 0.0) or 0.0),
            "percentiles": value.get("percentiles", {}),
        }
        observers[stage] = entry
        targets.append(stage)
    summary = {"config": meta, "targets": targets, "observers": observers}
    try:
        validate_export_file(summary)
    except AssertionError as exc:
        logging.warning("Legacy summary validation warning: %s", exc)
    return summary


def _load_summary_data(summary: Union[str, Mapping[str, Any]]):
    if isinstance(summary, str):
        if not os.path.exists(summary):
            raise FileNotFoundError(f"stats file not found: {summary}")
        data = torch.load(summary, map_location="cpu")
    else:
        data = summary
    if not isinstance(data, Mapping):
        raise ValueError("Invalid stats file format")
    if "observers" in data:
        validate_export_file(data)  # raises on mismatch
        return data
    return _convert_legacy_summary(data)


def _print_best_percentiles(observers: Mapping[str, Mapping[str, Any]]) -> None:
    for target, entry in observers.items():
        percentiles = _normalize_percentiles_map(entry.get("percentiles", {}))
        for candidate in PreferredPercentiles:
            if candidate in percentiles:
                pct_value = percentiles[candidate]
                try:
                    pct_number = float(candidate[1:].replace("_", "."))
                except ValueError:
                    pct_number = 0.0
                magnitude = abs(float(pct_value))
                hi = magnitude
                lo = -magnitude
                print(f"[BestPercentile] {target}: {pct_number:.3f} applied:true")
                print(f"[BestPercentile] hi={hi:.6g} lo={lo:.6g}")
                break
        else:
            print(f"[BestPercentile] {target}: unavailable applied:false")


def _guess_stage_from_module_name(name: str) -> Optional[str]:
    lname = name.lower()
    if "dino" in lname:
        return "vision.dino"
    if "siglip" in lname:
        return "vision.siglip"
    if "projector" in lname or "mm_projector" in lname:
        return "projector"
    if any(token in lname for token in ("llm", "mamba", "decoder", "language", "backbone")):
        return "llm"
    return None


def _gather_quantizers(model: nn.Module) -> List[Tuple[str, UniformAffineQuantizer, Optional[str], str]]:
    quantizers: List[Tuple[str, UniformAffineQuantizer, Optional[str], str]] = []
    for name, module in model.named_modules():
        if isinstance(module, UniformAffineQuantizer):
            stage = _guess_stage_from_module_name(name)
            qtype = "weight" if "weight_quantizer" in name else "act"
            quantizers.append((name, module, stage, qtype))
    return quantizers


def _apply_clips_to_quantizers(
    model: nn.Module,
    clip_table: Mapping[str, Mapping[str, float]],
) -> Tuple[int, int, int, int]:
    quantizers = _gather_quantizers(model)
    if not quantizers:
        logging.warning("[Warning] No quantizers found. Per-quantizer export will be empty.")
        return 0, 0, 0, 0

    applied = skipped = weight_applied = act_applied = 0
    for name, quantizer, stage, qtype in quantizers:
        if stage is None or stage not in clip_table:
            skipped += 1
            continue
        entry = clip_table[stage]
        lo = entry.get("lo")
        hi = entry.get("hi")
        if lo is None or hi is None:
            skipped += 1
            continue
        try:
            scale, zero = compute_affine_params(float(lo), float(hi), bits=quantizer.n_bits, signed=True)
            quantizer.set_clip(float(lo), float(hi), bucket=stage)
            quantizer.recompute_params_from_clip()
            device = (
                quantizer.scale.device
                if isinstance(getattr(quantizer, "scale", None), torch.Tensor)
                else torch.device("cpu")
            )
            quantizer.scale = torch.tensor(scale, dtype=torch.float32, device=device)
            if quantizer.disable_zero_point:
                quantizer.round_zero_point = None
            else:
                quantizer.round_zero_point = torch.tensor(int(zero), dtype=torch.int32, device=device)
            applied += 1
            if qtype == "weight":
                weight_applied += 1
            else:
                act_applied += 1
        except Exception as exc:
            logging.warning("Failed to apply clip to %s (%s): %s", name, stage, exc)
            skipped += 1
    return applied, skipped, weight_applied, act_applied


# ------------------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------------------

def apply_percentile_policy(
    summary: Union[str, Mapping[str, Any]],
    out_json: str,
    *,
    percentile_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Produce final clip table (percentile, lo, hi) from a percentile summary.
    """
    summary_data = _load_summary_data(summary)
    observers = summary_data.get("observers", {})

    out: Dict[str, Any] = {}
    for key, entry in observers.items():
        try:
            stats = _entry_to_stats(entry)
            p = decide_percentile(stats, override=percentile_override)
            lo, hi = compute_clip_range(stats, p)
            out[key] = {
                "percentile": float(p),
                "lo": float(lo),
                "hi": float(hi),
            }
            logging.info(f"[{key}] best_p={p:.3f}, lo={lo:.3f}, hi={hi:.3f}")
        except Exception as e:
            logging.warning(f"[{key}] failed: {e}")

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logging.info(f"[export] saved -> {out_json}")
    return out


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def _default_overrides_path(summary_path: str) -> str:
    base, ext = os.path.splitext(summary_path)
    if not base:
        base = summary_path
    suffix = "_overrides.json"
    if ext:
        return base + suffix
    return summary_path + suffix


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Overwatch PCT Apply — convert summary to clip ranges")
    ap.add_argument("--model", type=str, required=True, help="Model alias or checkpoint path understood by cobra.models.load")
    ap.add_argument("--summary", type=str, default="outputs/percentile_summary.pt", help="Path to percentile summary (.pt)")
    ap.add_argument("--device", type=str, default="cuda", help="Device for sanity checks (cuda or cpu)")
    ap.add_argument(
        "--best-percentile",
        type=str,
        default="none",
        choices=("none", "apply"),
        help="If set to 'apply', print preferred percentiles (99.99/99.9/99.0) per target",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)5s | %(message)s",
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = cobra_load(args.model)
    model.to(device)
    logging.info("Loaded model '%s' on %s for percentile application", args.model, device)

    summary_data = _load_summary_data(args.summary)
    if args.best_percentile == "apply":
        _print_best_percentiles(summary_data.get("observers", {}))

    out_json = _default_overrides_path(args.summary)
    clip_table = apply_percentile_policy(
        summary=summary_data,
        out_json=out_json,
        percentile_override=None,
    )

    applied, skipped, weight_applied, act_applied = _apply_clips_to_quantizers(model, clip_table)
    logging.info(
        "[PctApply] applied=%d skipped=%d weight_q=%d act_q=%d",
        applied,
        skipped,
        weight_applied,
        act_applied,
    )
    logging.info("Percentile overrides exported -> %s", out_json)


def run_calibration_only(
    model: nn.Module,
    pct_summary_path: str,
    bits: int,
    signed: bool,
    best_percent: Optional[Dict[str, float]] = None,
) -> int:
    """
    Build calibration table from percentile summary and apply to model quantizers.
    """
    table = build_table_from_file(
        pct_summary_path,
        default_bits=bits,
        signed=signed,
        best_percent_map=best_percent,
    )
    return apply_calibration_table(model, table)


if __name__ == "__main__":
    main()
