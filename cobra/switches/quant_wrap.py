"""
CLI utility for previewing or applying quantized wrapper replacements.

Usage example:
    python -m cobra.switches.quant_wrap \
        --model cobra+3b \
        --stages vision.dino,vision.siglip,projector,llm \
        --include ".*(qkv|proj|fc|conv).*" \
        --exclude ".*(embed|norm|rmsnorm).*" \
        --dry-run \
        --manifest outputs/reports/wrap_manifest.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Optional, Tuple, Union

from cobra.models.load import load as cobra_load
from cobra.quantize.wrap.manifest import save_manifest, summarize_manifest, load_manifest
from cobra.quantize.wrap.policy import WrapPolicy
from cobra.integration.wrap_replace import wrap_model, unwrap_model
from cobra.quantize.utils import FinalizeSpec, finalize_model
from cobra.quantize.percentile_io import export_finalize_bundle


def _parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Quantized wrapper replacement CLI")
    parser.add_argument("--model", type=str, required=True, help="Model alias or checkpoint path understood by cobra.models.load")
    parser.add_argument(
        "--stages",
        type=str,
        default="vision.dino,vision.siglip,projector,llm",
        help="Comma-separated canonical stages to enable",
    )
    parser.add_argument(
        "--include",
        type=str,
        default=".*(qkv|proj|fc|conv).*",
        help="Comma-separated regex patterns to include (default matches qkv/proj/fc/conv)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=".*(embed|norm|rmsnorm).*",
        help="Comma-separated regex patterns to exclude (default skips embed/norm variants)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview replacements without mutating the model")
    parser.add_argument("--ensure-before-calib", action="store_true", help="Force replacement if used before percentile calibration")
    parser.add_argument(
        "--manifest",
        type=str,
        default="outputs/reports/wrap_manifest.json",
        help="Path to save manifest (ignored in dry-run)",
    )
    parser.add_argument("--undo", type=str, default=None, help="Undo replacements using the specified manifest path")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def _print_preview(manifest: dict, limit: int = 50) -> None:
    items = manifest.get("items", [])
    if not items:
        print("[WrapPreview] No eligible modules found.")
        return
    print(f"[WrapPreview] First {min(limit, len(items))} planned replacements (of {len(items)} total):")
    for entry in items[:limit]:
        path = entry.get("path")
        from_type = entry.get("from")
        to_type = entry.get("to")
        stage = entry.get("stage")
        print(f"  {path} [{stage}] : {from_type} -> {to_type}")
    if len(items) > limit:
        print(f"  ... ({len(items) - limit} more)")


def _print_summary(manifest: dict) -> None:
    summary = manifest.get("summary", {})
    stage_counts = summary.get("by_stage", {})
    type_counts = summary.get("by_type", {})

    print("[WrapSummary] By Stage")
    if stage_counts:
        for stage, count in sorted(stage_counts.items()):
            print(f"  {stage:15s} : {count:6d}")
    else:
        print("  <none>")

    print("\n[WrapSummary] By Type")
    if type_counts:
        for key, count in sorted(type_counts.items()):
            print(f"  {key:25s} : {count:6d}")
    else:
        print("  <none>")

    skipped = summary.get("skipped", [])
    if skipped:
        print(f"\n[WrapSummary] Skipped entries (first {min(10, len(skipped))})")
        for entry in skipped[:10]:
            print(f"  {entry}")
        if len(skipped) > 10:
            print(f"  ... ({len(skipped) - 10} more)")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    model = cobra_load(args.model)

    if args.undo:
        manifest = load_manifest(args.undo)
        unwrap_model(model, manifest, strict=True)
        print(f"[WrapUndo] restored {len(manifest.get('items', []))} entries from {args.undo}")
        print(summarize_manifest(manifest))
        return

    stages = tuple(_parse_csv(args.stages)) or None
    include = _parse_csv(args.include)
    exclude = _parse_csv(args.exclude)

    policy = WrapPolicy(stages=stages, include=include, exclude=exclude)
    print(policy.summarize())

    manifest = wrap_model(
        model,
        policy,
        dry_run=args.dry_run,
        strict=True,
        capture_snapshot=not args.dry_run,
    )

    if args.dry_run and not args.ensure_before_calib:
        _print_preview(manifest)
    else:
        save_manifest(manifest, args.manifest)
        print(f"[WrapManifest] Saved to {args.manifest}")

    summary_text = summarize_manifest(manifest)
    _print_summary(manifest)

    summary_path = Path(args.manifest).with_suffix(".txt")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print(f"[WrapSummary] Saved summary to {summary_path}")


def _parse_prefix_tuple(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    if not value:
        return None
    entries = tuple(item.strip() for item in value.split(",") if item.strip())
    return entries or None


def parse_finalize_args() -> argparse.Namespace:
    examples = """
Example 1: Finalize with best-percentile overrides and packing
  python -m cobra.switches.quant_wrap finalize --model cobra+3b --dataset textvqa --split val[:256] \\
    --best-percent outputs/pct_collect_best/best_percentile_map.pt \\
    --wbits 8 --abits 8 --symmetric --per-channel --pack-linear --pack-conv \\
    --outdir outputs/finalize/w8a8

Example 2: 4-bit weights / 8-bit activations with verification
  python -m cobra.switches.quant_wrap finalize --model cobra+3b --dataset textvqa --split val[:512] \\
    --wbits 4 --abits 8 --symmetric --per-channel --pack-linear --pack-conv \\
    --verify-batches 10 --atol 2e-2 --rtol 2e-1 \\
    --outdir outputs/finalize/w4a8
"""
    parser = argparse.ArgumentParser(
        "Finalize quantized model and export bundle",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--best-percent", type=str, default=None)
    parser.add_argument("--wbits", type=int, default=8)
    parser.add_argument("--abits", type=int, default=8)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--rounding", type=str, choices=("nearest", "stochastic"), default="nearest")
    parser.add_argument("--pack-linear", action="store_true")
    parser.add_argument("--pack-conv", action="store_true")
    parser.add_argument("--allowlist", type=str, default=None)
    parser.add_argument("--denylist", type=str, default=None)
    parser.add_argument("--verify-batches", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-1)
    parser.add_argument("--outdir", type=str, default="outputs/finalize")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main_finalize() -> None:
    args = parse_finalize_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    model = cobra_load(args.model)

    spec = FinalizeSpec(
        weight_bits=args.wbits,
        act_bits=args.abits,
        symmetric=args.symmetric,
        per_channel=args.per_channel,
        rounding=args.rounding,
        pack_linear=args.pack_linear,
        pack_conv=args.pack_conv,
        allowlist=_parse_prefix_tuple(args.allowlist),
        denylist=_parse_prefix_tuple(args.denylist),
        verify_batches=args.verify_batches,
        atol=args.atol,
        rtol=args.rtol,
    )

    if args.dry_run:
        eligible = []
        for name, module in model.named_modules():
            if hasattr(module, "finalized"):
                eligible.append(name)
        print(f"[FinalizeDryRun] Would finalize {len(eligible)} modules under spec {spec}")
        for name in eligible[:50]:
            print(f"  - {name}")
        if len(eligible) > 50:
            print(f"  ... ({len(eligible) - 50} more)")
        return

    result = finalize_model(model, spec, best_percent_path=args.best_percent)
    manifest = result.get("manifest", {})
    bundle_paths = export_finalize_bundle(model, args.outdir, manifest)
    print("[Finalize] Completed with counts:", result.get("counts"))
    print("[Finalize] Bundle written to:", bundle_paths)
