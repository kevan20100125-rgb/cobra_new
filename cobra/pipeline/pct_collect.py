# cobra/pipeline/pct_collect.py
"""
Collect activation percentiles for four canonical modules and export stats.

Targets (fixed keys):
  - "vision.siglip" : SigLIP visual backbone feature output (pre-projector)
  - "vision.dino"   : DINO/DINOv2 visual backbone feature output (pre-projector)
  - "llm"           : LLM token embedding output (post-embedding / pre-first-block)
  - "projector"     : Multimodal projector output (encoder output fed to LLM)

This script intentionally lives outside training/materialize.py so you can run
calibration as a small, independent preprocessing step.

Outputs:
  - <out>.pt (torch.save dict): {
        "<bucket>": {
            "min": float, "max": float, "numel": int,
            "percentiles": {"p25.0":..., "p50.0":..., ...}
        }, ...
        "meta": {
            "created_at": str,
            "targets": [...],
            "images_dir": str|None,
            "texts_path": str|None,
            "num_batches": int,
            "batch_size": int,
        }
    }
Optionally:
  - --export-best-json <path>  # will also compute best percentile & (lo,hi) per bucket
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from cobra.models.load import load as cobra_load
from cobra.pipeline.pct_schema import normalize_stage, validate_export_file

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    Image = None  # type: ignore

# PCT toolkit (you created these modules)
from cobra.quantize.pct import (
    PercentileAccumulator,
    TARGETS,
    resolve_hooks,
    decide_percentile,
    compute_clip_range,
)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _module_qualname(root: nn.Module, module: nn.Module) -> str:
    for name, mod in root.named_modules():
        if mod is module:
            return name or module.__class__.__name__
    return module.__class__.__name__


def _infer_percent(percentiles: Dict[str, float]) -> float:
    best = None
    for key in percentiles.keys():
        if not key.startswith("p"):
            continue
        try:
            value = float(key[1:].replace("_", "."))
        except ValueError:
            continue
        if best is None or value > best:
            best = value
    return best if best is not None else 0.0


def _summarize_target(
    target: str,
    module_name: str,
    stats: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    stats = stats or {}
    percentiles_raw = stats.get("percentiles", {}) or {}
    percentiles = {
        k if k.startswith("p") else f"p{k}": float(v)
        for k, v in percentiles_raw.items()
        if isinstance(v, (int, float))
    }
    entry: Dict[str, any] = {
        "mode": "collect",
        "percent": _infer_percent(percentiles),
        "numel": int(stats.get("numel", 0) or 0),
        "target": target,
        "module": module_name,
        "min": float(stats.get("min", 0.0) or 0.0),
        "max": float(stats.get("max", 0.0) or 0.0),
        "percentiles": percentiles,
    }
    for key in ("p99.0", "p99.9", "p99.99"):
        if key in percentiles:
            entry[key] = percentiles[key]
    return entry


def _format_pct_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{value:.4f}"
    except Exception:
        return str(value)

# ------------------------------------------------------------------------------
# Minimal image I/O and batching
# ------------------------------------------------------------------------------

def _find_images(images_dir: Optional[str]) -> List[str]:
    if not images_dir:
        return []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(images_dir, ext)))
        files.extend(glob.glob(os.path.join(images_dir, "**", ext), recursive=True))
    return sorted(list(dict.fromkeys(files)))


def _load_image_as_tensor(path: str, device: torch.device, size: int = 384) -> torch.Tensor:
    """
    Minimal preprocessing: RGB, resize shorter side to `size`, center-crop square, to [0,1] float tensor.
    NOTE: This may not match exact backbone normalization. For percentile stats it's often sufficient.
    """
    if Image is None:
        raise RuntimeError("PIL not available. Please install pillow.")
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if min(w, h) != size:
        # scale shortest side to `size`
        if w < h:
            new_w, new_h = size, int(h * (size / w))
        else:
            new_w, new_h = int(w * (size / h)), size
        img = img.resize((new_w, new_h))
    # center crop to size x size
    w, h = img.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    img = img.crop((left, top, left + size, top + size))
    # to tensor [C,H,W], float32 in [0,1]
    t = torch.from_numpy(
        (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
         .view(size, size, 3)
         .permute(2, 0, 1)
         .to(torch.uint8)
         .numpy())
    ).float() / 255.0
    return t.to(device)


def _make_image_batch(paths: List[str], device: torch.device, size: int = 384) -> torch.Tensor:
    imgs = [_load_image_as_tensor(p, device=device, size=size) for p in paths]
    return torch.stack(imgs, dim=0) if imgs else torch.empty(0, 3, size, size, device=device)


def _prepare_text_batch(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    *,
    random_text: bool = True,
    vocab_high: int = 32000,
) -> torch.Tensor:
    """
    Produce integer token ids to drive LLM embeddings.
    Default is random ints in [1, vocab_high). This avoids requiring a tokenizer.
    You can later replace this with real tokenization.
    """
    if random_text:
        return torch.randint(1, vocab_high, (batch_size, seq_len), device=device, dtype=torch.long)
    # fallback: zeros (UNK/pad-like)
    return torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)


# ------------------------------------------------------------------------------
# Model loading (best-effort for cobra)
# ------------------------------------------------------------------------------

def _load_model(model_id: str, device: torch.device) -> nn.Module:
    """Load Cobra model via the canonical loader."""
    model = cobra_load(model_id)
    return model.to(device)


def _forward_once(model: nn.Module, images: torch.Tensor, input_ids: torch.Tensor) -> None:
    """
    Run a forward pass to trigger hooks. We try several calling conventions to be robust.
    """
    model.eval()
    with torch.no_grad():
        # Try VLM 'generate' API
        try:
            # Some cobra variants accept dicts or keyword arguments.
            _ = model.generate(images=images, input_ids=input_ids, max_new_tokens=1)
            return
        except Exception:
            pass
        # Try callable forward with kwargs
        try:
            _ = model(images=images, input_ids=input_ids)
            return
        except Exception:
            pass
        # Try minimal: forward images only (vision hooks) then text only (llm hooks)
        try:
            _ = model(images=images)
        except Exception:
            pass
        try:
            _ = model(input_ids=input_ids)
        except Exception:
            pass


# ------------------------------------------------------------------------------
# Main collection logic
# ------------------------------------------------------------------------------

def collect_percentiles(
    model: nn.Module,
    images_dir: Optional[str],
    batches: int,
    batch_size: int,
    device: torch.device,
    export_pt: str,
    *,
    img_size: int = 384,
    text_len: int = 16,
    random_text: bool = True,
    targets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Register hooks at the canonical targets, run a few batches, and export stats.
    """
    raw_targets = targets or TARGETS
    normalized_targets: List[str] = []
    for name in raw_targets:
        try:
            stage = normalize_stage(name)
        except Exception as exc:
            logging.warning("Skipping unknown percentile target '%s': %s", name, exc)
            continue
        if stage not in normalized_targets:
            normalized_targets.append(stage)
    if not normalized_targets:
        normalized_targets = list(TARGETS)
    target_set = set(normalized_targets)

    acc = PercentileAccumulator(device=str(device))

    mods = resolve_hooks(model)
    handles = []
    module_name_map: Dict[str, str] = {}
    for key, mod in mods.items():
        try:
            stage = normalize_stage(key)
        except Exception:
            continue
        if stage not in target_set:
            continue

        module_label = _module_qualname(model, mod)
        module_name_map.setdefault(stage, module_label)

        def _hook(_m, _inp, out, _stage=stage):
            try:
                acc.record_activation(out, bucket=_stage)
            except Exception as e:
                logging.warning(f"[hook:{_stage}] record_activation failed: {e}")

        h = mod.register_forward_hook(_hook)
        handles.append(h)
        logging.info(f"[hook] attached -> {stage} @ {module_label}")

    # Build image file list
    paths = _find_images(images_dir)
    if images_dir and not paths:
        logging.warning(f"No images found under: {images_dir}")

    # Iterate
    n = max(1, batches)
    for bi in range(n):
        # mini-batch image paths
        start = (bi * batch_size) % max(1, len(paths))
        batch_paths = paths[start:start + batch_size] if paths else []
        if batch_paths and len(batch_paths) < batch_size and len(paths) >= batch_size:
            # wrap-around
            wrap = batch_size - len(batch_paths)
            batch_paths += paths[:wrap]

        imgs = _make_image_batch(batch_paths, device=device, size=img_size) if batch_paths else torch.zeros((batch_size, 3, img_size, img_size), device=device)
        toks = _prepare_text_batch(batch_size, text_len, device=device, random_text=random_text)

        _forward_once(model, imgs, toks)
        logging.info(f"[run] batch {bi+1}/{n} collected; acc keys={list(acc.state_dict().keys())}")

        # if all buckets collected at least once, we can stop early
        if all(k in acc.state_dict() for k in target_set):
            break

    for handle in handles:
        handle.remove()

    raw_stats = acc.state_dict()
    observers: Dict[str, Dict[str, Any]] = {}
    for stage in normalized_targets:
        entry = _summarize_target(stage, module_name_map.get(stage, stage), raw_stats.get(stage))
        observers[stage] = entry
        logging.info(
            "[PctCollect] %s samples=%d p99=%s p99.9=%s p99.99=%s min=%.4f max=%.4f",
            stage,
            entry["numel"],
            _format_pct_value(entry.get("p99.0")),
            _format_pct_value(entry.get("p99.9")),
            _format_pct_value(entry.get("p99.99")),
            entry["min"],
            entry["max"],
        )

    payload: Dict[str, Any] = {
        "config": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "images_dir": images_dir,
            "num_batches": bi + 1,
            "batch_size": batch_size,
            "img_size": img_size,
            "text_len": text_len,
            "random_text": random_text,
        },
        "targets": normalized_targets,
        "observers": observers,
    }

    validate_export_file(payload)
    os.makedirs(os.path.dirname(export_pt) or ".", exist_ok=True)
    torch.save(payload, export_pt)
    logging.info(f"[export] saved percentile stats -> {export_pt}")
    return payload


def maybe_export_best_json(
    payload: Dict[str, Any],
    export_json: str,
    *,
    percentile_override: Optional[float] = None,
) -> None:
    """
    Optional: compute best percentile + (lo,hi) for each bucket and export JSON.
    """
    observers = payload.get("observers", {}) if isinstance(payload, dict) else {}
    out: Dict[str, dict] = {}
    for key, entry in observers.items():
        stats = {
            "percentiles": entry.get("percentiles", {}),
            "min": entry.get("min", 0.0),
            "max": entry.get("max", 0.0),
        }
        try:
            p = decide_percentile(stats, override=percentile_override)
            lo, hi = compute_clip_range(stats, p)
            out[key] = {"percentile": float(p), "lo": float(lo), "hi": float(hi)}
        except Exception as exc:
            logging.warning("Failed to export percentile for %s: %s", key, exc)
    os.makedirs(os.path.dirname(export_json) or ".", exist_ok=True)
    with open(export_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logging.info(f"[export] saved best-percentile clip table -> {export_json}")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Overwatch PCT Collect — gather activation percentiles")
    ap.add_argument("--model", type=str, required=True, help="Model alias or checkpoint path understood by cobra.models.load")
    ap.add_argument("--data", type=str, required=True, help="Directory containing calibration images")
    ap.add_argument("--batches", type=int, default=8, help="Number of calibration mini-batches to run")
    ap.add_argument("--export", type=str, default="outputs/percentile_summary.pt", help="Path to export percentile summary (.pt)")
    ap.add_argument("--device", type=str, default="cuda", help="Device to run calibration on (cuda or cpu)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)5s | %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = _load_model(args.model, device=device)

    payload = collect_percentiles(
        model=model,
        images_dir=args.data,
        batches=args.batches,
        batch_size=4,
        device=device,
        export_pt=args.export,
        img_size=384,
        text_len=16,
        random_text=True,
        targets=TARGETS,
    )

    logging.info(
        "[summary] exported percentile stats for %s -> %s",
        ", ".join(payload.get("targets", [])),
        args.export,
    )


if __name__ == "__main__":
    main()
