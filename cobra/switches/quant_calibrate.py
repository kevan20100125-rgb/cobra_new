# cobra/switches/quant_calibrate.py
# -*- coding: utf-8 -*-
"""
CLI entry for percentile-based calibration step in the quantization pipeline.
Implements the "校正" stage (Calibration) following the sequence:
    百分位裁剪觀測 → 匯出/套用 → 包裝替換 → 校正 → 旋轉 → 量化最終化

This script:
  1. Loads a model checkpoint or registry name.
  2. Reads percentile summary (.pt) from pct_collect/export.
  3. Builds a calibration table (per-quantizer scale, zero_point, clip range).
  4. Optionally applies the table to the model.
  5. Prints or saves the summary.

SmoothQuant logic is completely removed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Mapping

import torch
from torch import nn

from cobra.models.load import load_model
from cobra.models.materialize import materialize_model
from cobra.quantize.percentile_io import read_percentile_summary, write_calibration_table
from cobra.quantize.pct.calibrator import (
    PercentileCalibrator,
    CalibratorConfig,
    build_table_from_file,
    summarize_table,
)
from cobra.quantize.quantizer import apply_calibration_table
from cobra.quantize.pct.policy import get_optimal_percent_map
from cobra.util.torch_utils import get_logger


def _split_table_key(key: str) -> Optional[Tuple[str, str]]:
    parts = key.split(".")
    if len(parts) < 3:
        return None
    role = parts[-2]
    module_path = ".".join(parts[:-2])
    return module_path, role


def _gather_alignable_quantizers(model: nn.Module, table: Mapping[str, Mapping[str, Any]]) -> Dict[str, nn.Module]:
    modules = dict(model.named_modules())
    alignable: Dict[str, nn.Module] = {}
    for key in table.keys():
        split = _split_table_key(key)
        if not split:
            continue
        module_path, role = split
        module = modules.get(module_path)
        if module is None:
            continue
        quant = getattr(module, role, None)
        if quant is None:
            continue
        alignable[key] = quant
    return alignable


def _tensor_allclose(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5, rtol: float = 1e-3) -> bool:
    if a is None or b is None:
        return False
    a_f = a.detach().float().view(-1)
    b_f = b.detach().to(dtype=a_f.dtype, device=a_f.device).float().view(-1)
    return torch.allclose(a_f, b_f, atol=atol, rtol=rtol)


def _quantizer_matches_entry(quantizer: Any, entry: Mapping[str, Any]) -> bool:
    scale_expected = entry.get("scale")
    zero_expected = entry.get("zero_point", entry.get("zero"))
    if scale_expected is None or not hasattr(quantizer, "scale"):
        return False
    quant_scale = getattr(quantizer, "scale", None)
    if not isinstance(quant_scale, torch.Tensor):
        return False
    scale_tensor = torch.as_tensor(scale_expected, dtype=quant_scale.dtype, device=quant_scale.device)
    if not _tensor_allclose(quant_scale, scale_tensor):
        return False
    if getattr(quantizer, "disable_zero_point", False):
        return True
    quant_zero = getattr(quantizer, "round_zero_point", None)
    if zero_expected is None:
        return False
    if not isinstance(quant_zero, torch.Tensor):
        return False
    zero_tensor = torch.as_tensor(zero_expected, dtype=quant_zero.dtype, device=quant_zero.device)
    return _tensor_allclose(quant_zero, zero_tensor, atol=1, rtol=0)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cobra Quantization Calibration (Percentile-based)")
    p.add_argument("--model", type=str, required=True,
                   help="Model identifier (local path or HF hub id)")
    p.add_argument("--pct-summary", type=str, required=True,
                   help="Path to percentile_stats.pt generated from pct_collect")
    p.add_argument("--bits", type=int, default=8, help="Quantization bitwidth (default: 8)")
    p.add_argument("--signed", action="store_true", help="Use signed quantization range")
    p.add_argument("--best-percentile", type=str, choices=["off", "apply"], default="off",
                   help="Use best percentile map from conf/pct.py or stage defaults")
    p.add_argument("--targets", type=str, default=None,
                   help="Comma-separated list of stage targets (e.g. vision_backbone,llm_backbone,projector)")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not modify model; only print calibration summary")
    p.add_argument("--save-table", type=str, default=None,
                   help="Optional path to save generated calibration table as .pt file")
    return p


# ------------------------------------------------------------------------------
# Core
# ------------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    log = get_logger("quant_calibrate")

    log.info(f"[Step] Calibration with targets: {args.targets or '(all)'}")
    log.info(f"[Config] bits={args.bits}, signed={args.signed}, best-percentile={args.best_percentile}")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    try:
        model: nn.Module = load_model(args.model)
        model = materialize_model(model)
    except Exception as e:
        log.error(f"[Error] Failed to load model: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Read percentile summary
    # ------------------------------------------------------------------
    summary = read_percentile_summary(args.pct_summary)
    if not summary or "observers" not in summary:
        log.error(f"[Error] Invalid or empty percentile summary: {args.pct_summary}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Build calibration table
    # ------------------------------------------------------------------
    stage_whitelist: Optional[Tuple[str, ...]] = None
    if args.targets:
        stage_whitelist = tuple(x.strip() for x in args.targets.split(",") if x.strip())

    best_percent_map: Optional[Dict[str, float]] = None
    if args.best_percentile == "apply":
        best_percent_map = get_optimal_percent_map(stages=stage_whitelist)
        log.info("[BestPercentile] map:")
        if best_percent_map:
            for k, v in sorted(best_percent_map.items()):
                log.info(f"  {k}: {v:.3f}")
        else:
            log.info("  (no overrides)")

    cfg = CalibratorConfig(
        default_bits=args.bits,
        signed=args.signed,
        default_percentile=99.9,
        best_percent_map=best_percent_map,
        stage_whitelist=stage_whitelist,
    )

    calibrator = PercentileCalibrator(summary, cfg)
    table = calibrator.build_table()

    # ------------------------------------------------------------------
    # 4. Print or apply
    # ------------------------------------------------------------------
    print(summarize_table(table, top_k=30))

    if args.save_table:
        save_path = Path(args.save_table)
        write_calibration_table(save_path, table)
        log.info(f"[Save] calibration_table written to: {save_path}")

    if args.dry_run:
        log.info("[DryRun] No parameters applied.")
        return

    # Apply calibration
    alignable = _gather_alignable_quantizers(model, table)
    alignable_count = len(alignable)
    try:
        apply_calibration_table(model, table)
    except Exception as e:
        log.error(f"[Error] applying calibration table: {e}")
        sys.exit(1)

    missing_keys = [
        key for key, quant in alignable.items()
        if not _quantizer_matches_entry(quant, table.get(key, {}))
    ]
    successful = alignable_count - len(missing_keys)

    # ------------------------------------------------------------------
    # 5. Summary statistics
    # ------------------------------------------------------------------
    total_entries = len(table)
    miss_ratio = (len(missing_keys) / max(alignable_count, 1)) if alignable_count else 0.0

    log.info(f"[Stats] observed_entries={total_entries}")
    log.info(f"[Stats] alignable_quantizers={alignable_count}")
    log.info(f"[Stats] applied_entries={successful}")
    log.info(f"[Stats] missing_ratio={miss_ratio:.2%}")

    if miss_ratio > 0.5 and missing_keys:
        log.warning("[Warning] More than 50%% of alignable calibration entries failed to apply.")
        log.warning("[Warning] Missing calibration keys (up to 20 shown):")
        for k in sorted(missing_keys)[:20]:
            log.warning(f"  - {k}")

    log.info("[Done] Calibration step completed.")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
