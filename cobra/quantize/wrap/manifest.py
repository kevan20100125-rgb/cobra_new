# cobra/quantize/wrap/manifest.py
"""
Manifest utilities for quantized wrapper replacement.

This file provides lightweight save/load/report helpers for
the wrap replacement phase, bridging to calibration and rotation.

Manifest structure (see wrap_replace.wrap_model):
{
  "version": 1,
  "items": [ ... ],
  "summary": { "by_stage": {...}, "by_type": {...}, "skipped": [...] }
}

All functions are dependency-light, safe for import anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from cobra.quantize.utils import is_quant_eligible

# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------


def save_manifest(manifest: Dict[str, Any], path: str | Path) -> None:
    """
    Save manifest to a JSON file with pretty indentation.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[Manifest] Saved {len(manifest.get('items', []))} entries to {p}")


def load_manifest(path: str | Path) -> Dict[str, Any]:
    """
    Load manifest JSON from disk.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest malformed: expected dict, got {type(data)}")
    return data


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def summarize_manifest(manifest: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary table from manifest.
    """
    summary = manifest.get("summary", {})
    by_stage = summary.get("by_stage", {})
    by_type = summary.get("by_type", {})
    skipped = summary.get("skipped", [])

    lines: List[str] = []
    lines.append(f"[Manifest Summary] Total entries = {len(manifest.get('items', []))}")

    if by_stage:
        lines.append("\n[By Stage]")
        for k, v in sorted(by_stage.items()):
            lines.append(f"  {k:15s} : {v:>6d}")

    if by_type:
        lines.append("\n[By Type]")
        for k, v in sorted(by_type.items()):
            lines.append(f"  {k:25s} : {v:>6d}")

    if skipped:
        lines.append("\n[Skipped]")
        for s in skipped[:50]:
            lines.append(f"  {s}")
        if len(skipped) > 50:
            lines.append(f"  ... ({len(skipped) - 50} more)")

    return "\n".join(lines)


def summarize_finalize(model: nn.Module) -> Dict[str, Any]:
    """
    Summarize finalized quant modules (counts, bitwidths, non-finalized list).
    """
    total = 0
    finalized = 0
    bitwidths: Dict[str, int] = {}
    missing: List[str] = []

    for name, module in model.named_modules():
        if not is_quant_eligible(module):
            continue
        total += 1
        if getattr(module, "finalized", False):
            finalized += 1
            w_bits = getattr(getattr(module, "weight_quantizer", None), "n_bits", None)
            if w_bits is not None:
                key = f"weight_{w_bits}"
                bitwidths[key] = bitwidths.get(key, 0) + 1
            a_bits = getattr(getattr(module, "act_quantizer", None), "n_bits", None)
            if a_bits is not None:
                key = f"act_{a_bits}"
                bitwidths[key] = bitwidths.get(key, 0) + 1
        else:
            missing.append(name)

    return {
        "total_quant_modules": total,
        "finalized": finalized,
        "bitwidth_counts": bitwidths,
        "pending": missing,
    }


# ---------------------------------------------------------------------------
# Merge & compare utilities
# ---------------------------------------------------------------------------


def merge_manifests(manifests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple manifests (e.g., from multiple GPUs or stages)
    into a single unified manifest.
    """
    merged_items: List[Dict[str, Any]] = []
    merged_stage: Dict[str, int] = {}
    merged_type: Dict[str, int] = {}
    merged_skipped: List[str] = []

    for m in manifests:
        merged_items.extend(m.get("items", []))
        s = m.get("summary", {})
        for k, v in s.get("by_stage", {}).items():
            merged_stage[k] = merged_stage.get(k, 0) + v
        for k, v in s.get("by_type", {}).items():
            merged_type[k] = merged_type.get(k, 0) + v
        merged_skipped.extend(s.get("skipped", []))

    return {
        "version": 1,
        "items": merged_items,
        "summary": {
            "by_stage": merged_stage,
            "by_type": merged_type,
            "skipped": merged_skipped,
        },
    }


def diff_manifests(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """
    Compare two manifests and return textual differences by path/type.
    """
    items_a = {it["path"]: it for it in a.get("items", [])}
    items_b = {it["path"]: it for it in b.get("items", [])}

    diffs: List[str] = []

    all_paths = sorted(set(items_a.keys()) | set(items_b.keys()))
    for p in all_paths:
        if p not in items_a:
            diffs.append(f"[+]{p} added in B ({items_b[p]['to']})")
            continue
        if p not in items_b:
            diffs.append(f"[-]{p} removed in B (was {items_a[p]['to']})")
            continue
        if items_a[p].get("to") != items_b[p].get("to"):
            diffs.append(f"[Δ]{p} type changed: {items_a[p]['to']} -> {items_b[p]['to']}")
    return diffs


# ---------------------------------------------------------------------------
# CLI summary printer
# ---------------------------------------------------------------------------


def print_manifest_summary(path: str | Path) -> None:
    """
    Load and print summary for quick inspection.
    """
    manifest = load_manifest(path)
    print(summarize_manifest(manifest))


__all__ = [
    "save_manifest",
    "load_manifest",
    "summarize_manifest",
    "merge_manifests",
    "diff_manifests",
    "print_manifest_summary",
]
