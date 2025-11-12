# cobra/quantize/pct/range_calc.py
"""
Core math utilities for percentile-based calibration.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

_MAX_LAYER_ACT_SAMPLES = 65536


def _coerce_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _normalize_percentiles(stats: Mapping[str, Any]) -> Dict[str, float]:
    percentiles = stats.get("percentiles", {})
    out: Dict[str, float] = {}
    if not isinstance(percentiles, Mapping):
        return out
    for key, value in percentiles.items():
        numeric = _coerce_float(value)
        if numeric is None:
            continue
        name = str(key).strip().lower()
        name = name.replace("::", ".").replace("/", ".").replace("_", ".")
        if not name.startswith("p"):
            name = "p" + name
        out[name] = float(numeric)
    return out


def compute_symmetric_clip(
    stats: Mapping[str, Any],
    key_hi: str,
    key_lo: str,
) -> Tuple[float, float]:
    percentiles = _normalize_percentiles(stats)
    hi_key = key_hi.replace("_", ".")
    if not hi_key.startswith("p"):
        hi_key = "p" + hi_key
    lo_key = key_lo.replace("_", ".")
    if not lo_key.startswith("p"):
        lo_key = "p" + lo_key

    hi = percentiles.get(hi_key)
    lo = percentiles.get(lo_key)

    if hi is None and hi_key.replace(".", "_") in percentiles:
        hi = percentiles[hi_key.replace(".", "_")]
    if lo is None and lo_key.replace(".", "_") in percentiles:
        lo = percentiles[lo_key.replace(".", "_")]

    if hi is None and lo is not None:
        hi = abs(float(lo))
    if lo is None and hi is not None:
        lo = -abs(float(hi))

    if hi is None or lo is None:
        min_v = float(_coerce_float(stats.get("min")) or 0.0)
        max_v = float(_coerce_float(stats.get("max")) or 0.0)
        max_abs = max(abs(min_v), abs(max_v))
        return -max_abs, max_abs

    return float(lo), float(hi)


def choose_percentile(stats: Mapping[str, Any], default_p: float) -> float:
    best = _coerce_float(stats.get("best_percentile"))
    if best is not None:
        return float(best)
    percentiles = _normalize_percentiles(stats)
    if percentiles:
        def _score(name: str) -> float:
            try:
                return float(name[1:].replace("_", "."))
            except ValueError:
                return float("-inf")
        best_name = max(percentiles.keys(), key=_score)
        return float(best_name[1:].replace("_", "."))
    return float(default_p)


def range_to_affine(
    min_v: float,
    max_v: float,
    bits: int,
    signed: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if bits <= 0:
        raise ValueError("range_to_affine: bits must be positive")
    if signed:
        qmax = 2 ** (bits - 1) - 1
        max_abs = max(abs(min_v), abs(max_v))
        scale = max(max_abs / max(qmax, 1), torch.finfo(torch.float32).eps)
        return (
            torch.tensor([scale], dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
        )
    qrange = max((2**bits) - 1, 1)
    diff = max(max_v - min_v, torch.finfo(torch.float32).eps)
    scale = max(diff / qrange, torch.finfo(torch.float32).eps)
    return torch.tensor([scale], dtype=torch.float32), None


def fuse_observer_payload(ob: Mapping[str, Any], percentile: float) -> Tuple[float, float]:
    hi_key = f"p{percentile}"
    lo_key = f"p{max(0.0, 100.0 - percentile)}"
    return compute_symmetric_clip(ob, hi_key, lo_key)


def compute_clip_range(
    stats: Dict[str, Any],
    chosen_percentile: float,
    *,
    symmetric: bool = True,
) -> Tuple[float, float]:
    percentiles = _normalize_percentiles(stats)
    key = f"p{chosen_percentile}".replace("_", ".")
    hi_val = percentiles.get(key)
    if hi_val is None:
        raise ValueError(f"compute_clip_range: percentile '{key}' not found in stats")
    mag = abs(float(hi_val))
    if symmetric:
        return -mag, mag
    min_val = float(_coerce_float(stats.get("min")) or -mag)
    lo = min(min_val, -mag)
    hi = mag
    return lo, hi


def apply_best_percent_overrides(
    module_name: str,
    quantizer: Any,
    best_map: Mapping[str, Mapping[str, float]],
) -> None:
    """
    Apply longest-prefix best-percentile overrides (lo/hi) to a quantizer before finalization.
    """
    if not best_map or quantizer is None:
        return
    if not isinstance(best_map, Mapping):
        return

    name = (module_name or "").strip().lower()
    if not name:
        return

    matched: Optional[Mapping[str, float]] = None
    match_len = -1
    for prefix, payload in best_map.items():
        if not isinstance(prefix, str) or not isinstance(payload, Mapping):
            continue
        prefix_norm = prefix.strip().lower()
        if not prefix_norm:
            continue
        if not name.startswith(prefix_norm):
            continue
        if payload.get("hi") is None or payload.get("lo") is None:
            continue
        if len(prefix_norm) > match_len:
            matched = payload
            match_len = len(prefix_norm)

    if not matched:
        return

    hi = matched.get("hi")
    lo = matched.get("lo")
    if hi is None or lo is None:
        return

    set_clip = getattr(quantizer, "set_clip", None)
    symmetric = getattr(quantizer, "symmetric", True)
    if callable(set_clip):
        set_clip(lo, hi, symmetric=symmetric)
    elif hasattr(quantizer, "observer") and quantizer.observer is not None:
        quantizer.observer.set_manual_range(lo, hi)


def _move_to_device(sample: Any, device: torch.device) -> Any:
    if torch.is_tensor(sample):
        return sample.to(device)
    if isinstance(sample, dict):
        return {k: _move_to_device(v, device) for k, v in sample.items()}
    if isinstance(sample, list):
        return [_move_to_device(v, device) for v in sample]
    if isinstance(sample, tuple):
        return tuple(_move_to_device(v, device) for v in sample)
    return sample


class _ActivationBuffer:
    def __init__(self, capacity: int = _MAX_LAYER_ACT_SAMPLES) -> None:
        self.capacity = max(int(capacity), 1)
        self._tensor: Optional[torch.Tensor] = None

    def add(self, chunk: torch.Tensor) -> None:
        if chunk is None or chunk.numel() == 0:
            return
        data = chunk.detach().to(device="cpu", dtype=torch.float32)
        if data.shape[0] > self.capacity:
            idx = torch.randperm(data.shape[0])[: self.capacity]
            data = data[idx]
        if self._tensor is None:
            self._tensor = data.contiguous()
            return
        combined = torch.cat([self._tensor, data], dim=0)
        if combined.shape[0] > self.capacity:
            idx = torch.randperm(combined.shape[0])[: self.capacity]
            combined = combined[idx]
        self._tensor = combined.contiguous()

    def tensor(self) -> Optional[torch.Tensor]:
        return self._tensor


def _flatten_module_input(module: nn.Module, tensor: torch.Tensor) -> Optional[torch.Tensor]:
    if not isinstance(tensor, torch.Tensor):
        return None
    if tensor.dim() < 2:
        return None
    if isinstance(module, nn.Linear):
        features = tensor.shape[-1]
        return tensor.reshape(-1, features)
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        b, c = tensor.shape[:2]
        flat = tensor.reshape(b, c, -1).permute(0, 2, 1).reshape(-1, c)
        return flat
    return None


def sample_layer_activations(
    model: nn.Module,
    dataloader: Iterable[Any],
    targets: Sequence[str],
    max_batches: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Collect flattened [N, C] activation samples for selected Linear/Conv modules.

    Hooks capture module inputs (pre-activation) during an evaluation pass over at most `max_batches`
    mini-batches. Batches from `dataloader` must be ready for direct forwarding into `model`
    (either tensors, tuples/lists of args, or dicts of kwargs). Returned samples live on CPU.
    """
    if max_batches <= 0 or not targets:
        return {}

    modules = dict(model.named_modules())
    selected: Dict[str, nn.Module] = {}
    for name in dict.fromkeys(targets):
        module = modules.get(name)
        if module is None:
            continue
        if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            continue
        selected[name] = module
    if not selected:
        return {}

    buffers = {name: _ActivationBuffer() for name in selected}
    handles = []

    def _hook_factory(name: str):
        def _hook(mod: nn.Module, inputs):
            if not inputs:
                return
            tensor = inputs[0]
            if isinstance(tensor, (list, tuple)):
                tensor = tensor[0]
            if not isinstance(tensor, torch.Tensor):
                return
            flat = _flatten_module_input(mod, tensor)
            if flat is None:
                return
            buffers[name].add(flat)
        return _hook

    for name, module in selected.items():
        handles.append(module.register_forward_pre_hook(_hook_factory(name)))

    was_training = model.training
    target_device = torch.device(device)
    model.eval()
    model.to(target_device)

    try:
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if idx >= max_batches:
                    break
                moved = _move_to_device(batch, target_device)
                if isinstance(moved, dict):
                    model(**moved)
                elif isinstance(moved, (list, tuple)):
                    model(*moved)
                else:
                    model(moved)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    return {
        name: buf.tensor()
        for name, buf in buffers.items()
        if buf.tensor() is not None
    }


__all__ = [
    "compute_clip_range",
    "compute_symmetric_clip",
    "choose_percentile",
    "range_to_affine",
    "fuse_observer_payload",
    "apply_best_percent_overrides",
    "sample_layer_activations",
]
