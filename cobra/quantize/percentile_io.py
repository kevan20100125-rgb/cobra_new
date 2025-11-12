# cobra/quantize/percentile_io.py
# -*- coding: utf-8 -*-
"""
Utilities for exporting and applying percentile-based clipping artifacts.

This module standardizes:
1) Canonical target names for activation buckets
   -> "vision.siglip", "vision.dino", "llm", "projector"
2) I/O helpers for module-level summary (percentile_summary.pt)
   and per-quantizer overrides (percentile_overrides.pt)

File formats (torch.save):

percentile_summary.pt
{
  "vision.siglip": {
    "min": float,
    "max": float,
    "percentiles": {"p25.0": float, "p50.0": float, ...},
    "best_percentile": float,    # e.g., 99.9
    "lo": Optional[float],       # chosen clipping lower bound
    "hi": Optional[float]        # chosen clipping upper bound
  },
  "vision.dino": {...},
  "llm": {...},
  "projector": {...},
  "meta": {
    "order": [99.0, 99.9, 99.99, 99.999],
    "eps": 1e-6, "tau": 3.0, "r": 1.5,
    "created_at": "...",
  }
}

percentile_overrides.pt
{
  "<module_path>.<role>.<idx>": {          # e.g., "llm_backbone.layers.0.mixer.in_proj.act_quantizer.0"
    "percentile": Optional[float],         # if present but lo/hi missing, front-end will compute from stats
    "lo": Optional[float],
    "hi": Optional[float],
    "scale": Optional[float],              # optional precomputed
    "zero": Optional[float]                # optional precomputed
  },
  ...
}

All functions here are pure utilities; no model imports.
"""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple, Union, Sequence

import torch
from torch import nn

# -------------------------
# Canonical target handling
# -------------------------

CANON_TARGETS = ("vision.siglip", "vision.dino", "llm", "projector")


def normalize_target(name: str) -> str:
    """
    Map arbitrary module/bucket names to one of the four canonical targets.

    Heuristics use lowercase substring checks and common aliases.
    Fallback: if any canonical token appears, return it; else default to "projector".
    """
    if not isinstance(name, str):
        return "projector"
    n = name.lower()

    # Direct hints
    if "siglip" in n:
        return "vision.siglip"
    if "dino" in n or "dinov2" in n:
        return "vision.dino"
    if "llm" in n or "mamba" in n:
        return "llm"
    if "projector" in n or "mm.out" in n or "encoder_out" in n or "proj" in n:
        return "projector"

    # Fallback: any canonical token present?
    for c in CANON_TARGETS:
        token = c.split(".")[-1]
        if token in n:
            return c

    # Last resort
    return "projector"


def normalize_target_name(name: str) -> str:
    """
    Normalize arbitrary path-like names by trimming prefixes and separators.
    """
    if not isinstance(name, str):
        return ""
    cleaned = name.strip()
    if "::" in cleaned:
        cleaned = cleaned.split("::")[-1]
    cleaned = cleaned.replace("\\", "/")
    cleaned = cleaned.replace("//", "/")
    cleaned = cleaned.strip("/ ")
    cleaned = cleaned.replace("/", ".")
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    return cleaned


def normalize_target_name(name: str) -> str:
    """
    Normalize arbitrary path-like names by trimming prefixes and separators.
    """
    if not isinstance(name, str):
        return ""
    cleaned = name.strip()
    if "::" in cleaned:
        cleaned = cleaned.split("::")[-1]
    cleaned = cleaned.replace("\\", "/")
    cleaned = cleaned.replace("//", "/")
    cleaned = cleaned.strip("/ ")
    cleaned = cleaned.replace("/", ".")
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    return cleaned


# -------------------------
# Safe I/O helpers
# -------------------------

def _ensure_dir(path: str) -> None:
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _to_builtin(obj: Any) -> Any:
    """
    Recursively convert Tensors/ndarrays to Python scalars/lists for JSON export if needed.
    Remains no-op for torch.save usage but useful for debugging and fallbacks.
    """
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def _coerce_float(x: Any) -> Optional[float]:
    try:
        if isinstance(x, torch.Tensor):
            return float(x.item())
        return float(x)
    except Exception:
        return None


def _normalize_percentile_key(name: str) -> str:
    key = str(name).strip().lower()
    key = key.replace("::", ".").replace("/", ".").replace("_", ".")
    if not key.startswith("p"):
        key = "p" + key
    return key


def _normalize_observers(data: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    observers: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        if "smoothquant" in str(key).lower():
            continue
        if not isinstance(value, Mapping):
            continue
        raw = dict(value)
        target_name = normalize_target_name(value.get("target", key))
        percentiles_raw = value.get("percentiles", {})
        percentiles: Dict[str, float] = {}
        if isinstance(percentiles_raw, Mapping):
            for pk, pv in percentiles_raw.items():
                fv = _coerce_float(pv)
                if fv is None:
                    continue
                percentiles[_normalize_percentile_key(pk)] = float(fv)
        entry: Dict[str, Any] = {
            "_raw": raw,
            "target": target_name,
            "module": value.get("module", target_name),
            "mode": value.get("mode", "collect"),
            "numel": int(_coerce_float(value.get("numel", 0)) or 0),
            "min": float(_coerce_float(value.get("min", 0.0)) or 0.0),
            "max": float(_coerce_float(value.get("max", 0.0)) or 0.0),
            "percentiles": percentiles,
        }
        best = _coerce_float(value.get("best_percentile"))
        if best is not None:
            entry["best_percentile"] = float(best)
        lo = _coerce_float(value.get("lo"))
        hi = _coerce_float(value.get("hi"))
        if lo is not None:
            entry["lo"] = float(lo)
        if hi is not None:
            entry["hi"] = float(hi)
        observers[target_name] = entry
    return observers


# -------------------------
# Summary (.pt) I/O
# -------------------------

def _validate_summary_struct(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Soft-validate and coerce a summary dict to expected schema.
    Missing fields are filled with defaults. Extra fields are preserved.
    """
    out: Dict[str, Any] = {}

    # Pass-through meta
    if "meta" in summary and isinstance(summary["meta"], Mapping):
        out["meta"] = dict(summary["meta"])

    for key in CANON_TARGETS:
        src = summary.get(key, {})
        if not isinstance(src, Mapping):
            src = {}
        dst: Dict[str, Any] = {}

        # Basic scalars
        dst["min"] = _coerce_float(src.get("min", 0.0)) or 0.0
        dst["max"] = _coerce_float(src.get("max", 0.0)) or 0.0

        # Percentiles dict
        psrc = src.get("percentiles", {})
        pdst: Dict[str, float] = {}
        if isinstance(psrc, Mapping):
            for k, v in psrc.items():
                # normalize keys like "p99.9" / "p99_9"
                kk = str(k).replace("_", ".")
                if not kk.startswith("p"):
                    kk = "p" + kk
                val = _coerce_float(v)
                if val is not None:
                    pdst[kk] = val
        dst["percentiles"] = pdst

        # Best percentile, lo/hi
        bp = _coerce_float(src.get("best_percentile", 99.9)) or 99.9
        dst["best_percentile"] = bp

        lo = src.get("lo", None)
        hi = src.get("hi", None)
        lo_f = _coerce_float(lo) if lo is not None else None
        hi_f = _coerce_float(hi) if hi is not None else None
        dst["lo"] = lo_f
        dst["hi"] = hi_f

        # Copy any extra keys transparently
        for k, v in src.items():
            if k not in dst:
                dst[k] = v

        out[key] = dst

    return out


def save_summary(path: str, summary: Mapping[str, Any]) -> None:
    """
    Save module-level percentile summary (.pt).
    Ensures canonical structure and directory existence.
    """
    _ensure_dir(path)
    clean = _validate_summary_struct(summary)
    torch.save(clean, path)


def load_summary(path: str) -> Dict[str, Any]:
    """
    Load module-level percentile summary (.pt).
    Returns a dict with canonical keys and soft-validated structure.
    """
    summary = read_percentile_summary(path)
    legacy: Dict[str, Any] = {}
    for target, entry in summary.get("observers", {}).items():
        payload: Dict[str, Any] = {
            "min": entry.get("min", 0.0),
            "max": entry.get("max", 0.0),
            "percentiles": entry.get("percentiles", {}),
            "best_percentile": entry.get("best_percentile"),
            "lo": entry.get("lo"),
            "hi": entry.get("hi"),
        }
        legacy[target] = payload
    legacy["meta"] = summary.get("config", {})
    return _validate_summary_struct(legacy)


# -------------------------
# Overrides (.pt) I/O
# -------------------------

def _validate_overrides_struct(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Soft-validate per-quantizer overrides dict.

    Structure:
      key: "<module_path>.<role>.<idx>"
      val: {"percentile": Optional[float], "lo": Optional[float], "hi": Optional[float],
            "scale": Optional[float], "zero": Optional[float]}
    """
    out: Dict[str, Any] = {}
    for k, v in overrides.items():
        if not isinstance(k, str):
            # skip invalid key
            continue
        if not isinstance(v, Mapping):
            # skip invalid payload
            continue

        payload: Dict[str, Optional[float]] = {}
        for field in ("percentile", "lo", "hi", "scale", "zero"):
            fv = v.get(field, None)
            payload[field] = _coerce_float(fv) if fv is not None else None

        out[k] = payload
    return out


def save_overrides(path: str, overrides: Mapping[str, Any]) -> None:
    """
    Save per-quantizer overrides (.pt).
    """
    _ensure_dir(path)
    clean = _validate_overrides_struct(overrides)
    torch.save(clean, path)


def load_overrides(path: str) -> Dict[str, Any]:
    """
    Load per-quantizer overrides (.pt).
    """
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, Mapping):
        raise TypeError(f"Invalid overrides object at {path}: expected Mapping, got {type(obj)}")
    return _validate_overrides_struct(obj)


# -------------------------
# Helpers for debugging/JSON
# -------------------------

def export_summary_json(path: str, summary: Mapping[str, Any]) -> None:
    """
    Optional: dump summary as JSON (human-readable).
    """
    _ensure_dir(path)
    clean = _validate_summary_struct(summary)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(clean), f, ensure_ascii=False, indent=2)


def export_overrides_json(path: str, overrides: Mapping[str, Any]) -> None:
    """
    Optional: dump overrides as JSON (human-readable).
    """
    _ensure_dir(path)
    clean = _validate_overrides_struct(overrides)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(clean), f, ensure_ascii=False, indent=2)


def read_percentile_summary(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and normalize percentile summary into canonical schema.
    """
    p = Path(path)
    data = torch.load(p, map_location="cpu")
    if not isinstance(data, Mapping):
        raise TypeError(f"Invalid summary object at {p}: expected Mapping, got {type(data)}")
    if "observers" in data and isinstance(data["observers"], Mapping):
        observers = _normalize_observers(data["observers"])
        config = data.get("config") or data.get("meta") or {}
    else:
        legacy = {
            k: v
            for k, v in data.items()
            if isinstance(v, Mapping) and k not in ("config", "meta", "targets")
        }
        observers = _normalize_observers(legacy)
        config = data.get("config") or data.get("meta") or {}
    summary = {
        "config": config,
        "targets": list(observers.keys()),
        "observers": observers,
    }
    return summary


def load_best_percent_map(path: Union[str, Path]) -> Dict[str, Dict[str, float]]:
    """
    Read pct_collect_best (or similar) files mapping stage/prefix -> percentile metadata.

    Returns {name: {"best_percentile": float, "hi": float, "lo": float}}; empty dict if file missing.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = _load_structured_file(p)
    if not isinstance(data, Mapping):
        raise TypeError(f"Invalid best-percentile map at {path}: expected Mapping, got {type(data)}")

    if "items" in data and isinstance(data["items"], list):
        entries = {item.get("name", f"item_{idx}"): item for idx, item in enumerate(data["items"])}
    else:
        entries = dict(data)

    result: Dict[str, Dict[str, float]] = {}
    for key, value in entries.items():
        if not isinstance(value, Mapping):
            continue
        best = _coerce_float(value.get("best_percentile"))
        hi = _coerce_float(value.get("hi"))
        lo = _coerce_float(value.get("lo"))
        if best is None and hi is None and lo is None:
            continue
        result[str(key)] = {
            "best_percentile": float(best) if best is not None else 0.0,
            "hi": float(hi) if hi is not None else 0.0,
            "lo": float(lo) if lo is not None else 0.0,
        }
    return result


def _load_structured_file(path: Union[str, Path]) -> Any:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".json", ".jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return torch.load(p, map_location="cpu")


def _find_layer_blob(data: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    preferred_keys = (
        "layers",
        "layer_stats",
        "layer_summaries",
        "activation_summaries",
        "activations",
        "modules",
        "module_stats",
        "per_layer",
    )
    for key in preferred_keys:
        blob = data.get(key)
        if isinstance(blob, Mapping) and blob:
            return blob
    for key in ("data", "payload", "summary"):
        nested = data.get(key)
        if isinstance(nested, Mapping):
            found = _find_layer_blob(nested)
            if found:
                return found
    return None


def load_activation_summaries(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load per-layer activation summaries from pct_collect outputs (stage-level or detailed).

    Supports torch.save (.pt/.pth) and JSON exports. When only stage-level best percentile data
    is present (e.g., vision.siglip → {best_percentile, hi, lo}), returns an empty dict instead
    of raising so downstream rotation steps can fall back gracefully.
    """
    obj = _load_structured_file(path)
    if not isinstance(obj, Mapping):
        raise TypeError(f"Activation summary at {path} must be a Mapping, got {type(obj)}")

    blob = _find_layer_blob(obj)
    if blob is None:
        candidates: Dict[str, Any] = {}
        for key, value in obj.items():
            if not isinstance(value, Mapping):
                continue
            name = normalize_target_name(str(key))
            if name in CANON_TARGETS:
                continue
            if any(k in value for k in ("percentiles", "min", "max")):
                candidates[str(key)] = dict(value)
        return candidates

    summaries: Dict[str, Any] = {}
    for key, value in blob.items():
        if not isinstance(value, Mapping):
            continue
        name = normalize_target_name(str(key))
        if name in CANON_TARGETS and not "." in str(key):
            # Stage-level bucket; skip to avoid treating it as layer stats.
            continue
        summaries[str(key)] = dict(value)
    return summaries


def save_rotation_manifest(path: Union[str, Path], manifest: Dict[str, Any]) -> None:
    """
    Persist a rotation manifest (Hadamard/KLT plans) as JSON or torch.save depending on suffix.
    """
    p = Path(path)
    _ensure_dir(str(p))
    suffix = p.suffix.lower()
    if suffix in (".json", ".jsonl"):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(_to_builtin(manifest), f, ensure_ascii=False, indent=2)
    else:
        torch.save(manifest, p)


def write_calibration_table(path: Union[str, Path], table: Mapping[str, Mapping[str, Any]]) -> None:
    """
    Persist a per-module calibration table via torch.save.
    """
    payload = {
        "version": 1,
        "calibration_table": dict(table),
    }
    torch.save(payload, Path(path))


def save_quant_manifest(path: Union[str, Path], manifest: Dict[str, Any]) -> None:
    """
    Save a quantization manifest (finalization metadata) in JSON or torch format based on extension.
    """
    p = Path(path)
    _ensure_dir(str(p))
    suffix = p.suffix.lower()
    if suffix in (".json", ".jsonl"):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(_to_builtin(manifest), f, ensure_ascii=False, indent=2)
    else:
        torch.save(manifest, p)


def export_finalize_bundle(
    model: nn.Module,
    outdir: Union[str, Path],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Export finalized quant artifacts (packed weights, quant params, manifest).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    weights: Dict[str, Any] = {}
    params: Dict[str, Any] = {}

    for name, module in model.named_modules():
        if not hasattr(module, "finalized") or not getattr(module, "finalized"):
            continue
        weight_entry: Dict[str, Any] = {}
        params_entry: Dict[str, Any] = {}

        w_qint = getattr(module, "w_qint", None)
        if isinstance(w_qint, torch.Tensor):
            weight_entry["tensor"] = w_qint.detach().cpu()
            weight_entry["pack_meta"] = getattr(module, "pack_meta", {})
        weight_fp32_cache = getattr(module, "weight_fp32_cache", None)
        if isinstance(weight_fp32_cache, torch.Tensor):
            weight_entry["fp32_backup"] = weight_fp32_cache.detach().cpu()
        if weight_entry:
            weights[name] = weight_entry

        weight_quantizer = getattr(module, "weight_quantizer", None)
        act_quantizer = getattr(module, "act_quantizer", None)
        if weight_quantizer is not None:
            params_entry["weight_scale"] = getattr(weight_quantizer, "scale", torch.tensor([])).detach().cpu()
            zero_w = getattr(weight_quantizer, "round_zero_point", None)
            if isinstance(zero_w, torch.Tensor):
                params_entry["weight_zero"] = zero_w.detach().cpu()
            params_entry["weight_bits"] = getattr(weight_quantizer, "n_bits", None)
            params_entry["weight_symmetric"] = getattr(weight_quantizer, "symmetric", True)
            params_entry["weight_qmin"] = getattr(weight_quantizer, "qmin", None)
            params_entry["weight_qmax"] = getattr(weight_quantizer, "qmax", None)
        if act_quantizer is not None:
            params_entry["act_scale"] = getattr(act_quantizer, "scale", torch.tensor([])).detach().cpu()
            zero_a = getattr(act_quantizer, "round_zero_point", None)
            if isinstance(zero_a, torch.Tensor):
                params_entry["act_zero"] = zero_a.detach().cpu()
            params_entry["act_bits"] = getattr(act_quantizer, "n_bits", None)
            params_entry["act_symmetric"] = getattr(act_quantizer, "symmetric", True)
        if params_entry:
            params[name] = params_entry

    weights_path = outdir / "weights.pt"
    params_path = outdir / "params.pt"
    torch.save(weights, weights_path)
    torch.save(params, params_path)
    manifest_path = outdir / "manifest.json"
    save_quant_manifest(manifest_path, manifest)

    return {
        "weights": str(weights_path),
        "params": str(params_path),
        "manifest": str(manifest_path),
    }


__all__ = [
    "CANON_TARGETS",
    "normalize_target",
    "normalize_target_name",
    "save_summary",
    "load_summary",
    "save_overrides",
    "load_overrides",
    "export_summary_json",
    "export_overrides_json",
    "read_percentile_summary",
    "load_activation_summaries",
    "save_rotation_manifest",
    "load_best_percent_map",
    "save_quant_manifest",
]
