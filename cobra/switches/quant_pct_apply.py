# cobra/switches/quant_pct_apply.py
# -*- coding: utf-8 -*-
"""
Switch utility for applying percentile-based clipping overrides to a model.

This script reads a percentile summary (.pt) exported from
`cobra/pipeline/pct_collect.py` or `cobra/pipeline/pct_apply.py`
and writes the clipping boundaries (lo/hi) into each quantizer or clipper
found in the model.

Supported modules:
    - vision.siglip
    - vision.dino
    - llm
    - projector

Typical usage
-------------
>>> from cobra.models.load import load
>>> from cobra.switches.quant_pct_apply import enable
>>> model = load("cobra+3b")
>>> total = enable(model, "outputs/percentile_summary.pt")
>>> print(f"Applied {total} percentile overrides.")
"""

from __future__ import annotations
import torch
from torch import nn
from typing import Any, Dict, Optional
from cobra.integration.wrap_replace import wrap_model
from cobra.quantize.wrap.policy import WrapPolicy
from cobra.quantize.percentile_io import load_summary


def _apply_to_clipper(clipper: Optional[nn.Module], info: Dict[str, Any]) -> int:
    """
    Assign lo/hi from percentile summary to a clipper module.
    Returns 1 if successfully applied, else 0.
    """
    if clipper is None or not isinstance(info, dict):
        return 0
    lo, hi = info.get("lo"), info.get("hi")
    if lo is None or hi is None:
        return 0
    try:
        clipper.lo.fill_(float(lo))
        clipper.hi.fill_(float(hi))
        clipper.calibrated = True
        return 1
    except Exception:
        return 0


@torch.no_grad()
def enable(model: nn.Module, summary_path: str) -> int:
    """
    Enable percentile clipping for all major modules.

    Parameters
    ----------
    model : nn.Module
        Target model instance (cobra multimodal model).
    summary_path : str
        Path to `percentile_summary.pt` file exported from pct_collect/pct_apply.

    Returns
    -------
    int
        Number of clip modules successfully updated.
    """
    stats = load_summary(summary_path)
    applied = 0

    # Vision backbones
    vb = getattr(model, "vision_backbone", None)
    if vb is not None:
        applied += _apply_to_clipper(getattr(vb, "_clip_siglip", None), stats.get("vision.siglip", {}))
        applied += _apply_to_clipper(getattr(vb, "_clip_dino", None), stats.get("vision.dino", {}))

    # LLM backbone
    llm = getattr(model, "llm_backbone", None)
    if llm is not None:
        applied += _apply_to_clipper(getattr(llm, "_clip_l", None), stats.get("llm", {}))

    # Projector / encoder output
    applied += _apply_to_clipper(getattr(model, "_clip_enc", None), stats.get("projector", {}))

    print(f"[QuantPctApply] Applied {applied} percentile clip ranges from {summary_path}")
    return applied


def dry_run(summary_path: str) -> None:
    """
    Print the contents of the percentile summary file for inspection.
    """
    stats = load_summary(summary_path)
    print(f"=== Percentile Summary ({summary_path}) ===")
    for key, entry in stats.items():
        if key == "meta":
            continue
        lo, hi, p = entry.get("lo"), entry.get("hi"), entry.get("best_percentile")
        print(f"{key:<15} p={p:<7} lo={lo:<8.3f} hi={hi:<8.3f}")
    print("==========================================")


if __name__ == "__main__":
    import argparse
    from cobra.models.load import load

    parser = argparse.ArgumentParser(description="Apply percentile clipping summary to model")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--summary", type=str, default="outputs/percentile_summary.pt")
    parser.add_argument("--dry", action="store_true", help="Dry-run: print summary only")
    parser.add_argument("--calibrate-only", action="store_true", help="Only apply calibration table without wrapping/clipping")

    args = parser.parse_args()
    if args.dry:
        dry_run(args.summary)
    else:
        model = load(args.model)
        if args.calibrate_only:
            from cobra.pipeline.pct_apply import run_calibration_only

            count = run_calibration_only(model, args.summary, bits=8, signed=True)
            print(f"[QuantPctCalibrate] Applied {count} calibration entries")
            return
        if not any(hasattr(m, "quant_meta") for m in model.modules()):
            wrap_model(model, WrapPolicy(), dry_run=False, strict=True, capture_snapshot=False)
        enable(model, args.summary)
