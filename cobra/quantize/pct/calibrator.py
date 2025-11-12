# cobra/quantize/pct/calibrator.py
# -*- coding: utf-8 -*-
"""
Percentile-based calibration for quantized modules.

Reads a percentile summary (produced during "百分位裁剪觀測 → 匯出/套用"),
selects a percentile per stage or per entry, converts observed ranges to
affine quantization parameters, and emits a per-quantizer calibration table.

No SmoothQuant logic is included.

Output table key format:
    "<module_path>.<role>.<index>"
where:
    - module_path: dotted module name from model.named_modules()
    - role: "weight_quantizer" or "act_quantizer" (extensible)
    - index: integer index if the quantizer is a list/batched, else 0
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

from cobra.quantize.percentile_io import (
    read_percentile_summary,
    normalize_target_name,
    save_rotation_manifest,
    load_best_percent_map,
)
from cobra.quantize.pct.range_calc import (
    compute_symmetric_clip,
    choose_percentile,
    range_to_affine,
    fuse_observer_payload,
    sample_layer_activations,
    apply_best_percent_overrides,
)
from cobra.quantize.wrap.registry import get_rotation_targets
from cobra.quantize.wrap.policy import RotationPolicy
from cobra.quantize.utils import (
    bake_rotation_linear,
    bake_rotation_convnd,
    FinalizeSpec,
    is_quant_eligible,
    finalize_quant_params,
)
from cobra.quantize.hadamard_utils import hadamard_rotation
from cobra.quantize.get_klt_matrix import klt_rotation_from_acts

# ---- Public types ----------------------------------------------------------------

CalibrationEntry = Dict[str, Any]
CalibrationTable = Dict[str, CalibrationEntry]
SummaryLike = Mapping[str, Any]


# ---- Helpers ---------------------------------------------------------------------

_SUPPORTED_ROLES: Tuple[str, ...] = ("weight_quantizer", "act_quantizer", "weight", "act")

def _is_supported_role(role: str) -> bool:
    r = role.lower()
    return r in _SUPPORTED_ROLES or r.endswith("_quantizer")


def _key(module_path: str, role: str, index: Optional[int]) -> str:
    idx = 0 if index is None else int(index)
    return f"{module_path}.{role}.{idx}"


def _guess_role_from_key(k: str) -> Optional[str]:
    # try common tails
    parts = k.split(".")
    for cand in ("weight_quantizer", "act_quantizer", "weight", "act"):
        if parts[-1] == cand:
            return cand
    # try substring
    for cand in ("weight_quantizer", "act_quantizer", "weight", "act"):
        if cand in k:
            return cand
    return None


def _extract_module_role(k: str) -> Tuple[str, Optional[str]]:
    """Split observer key into (module_path, role_or_None)."""
    role = _guess_role_from_key(k)
    if role is None:
        return k, None
    pos = k.rfind("." + role)
    if pos == -1:
        return k, role
    return k[:pos], role


def _iter_observer_items(summary: SummaryLike) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Yield (observer_key, payload) from summary['observers'] with robustness:
    - flat dict: {"a.b.weight_quantizer": {...}, "a.b.act_quantizer": {...}}
    - nested dict: {"a.b": {"weight_quantizer": {...}, "act_quantizer": {...}}}
    """
    observers = summary.get("observers", {})
    if not isinstance(observers, Mapping):
        return
    for k, v in observers.items():
        if isinstance(v, Mapping) and any(subk in v for subk in _SUPPORTED_ROLES):
            # nested
            for subk, subv in v.items():
                if isinstance(subv, Mapping):
                    yield f"{k}.{subk}", dict(subv)
        elif isinstance(v, Mapping):
            # flat
            yield k, dict(v)


def _select_percentile_for_module(
    module_path: str,
    stage_percent: Optional[Mapping[str, float]],
    fallback: float,
) -> float:
    """
    Pick percentile for a module by longest-prefix stage match, else fallback.
    stage_percent maps stage-name or prefix → percentile (e.g. {'vision_backbone.dino': 99.9})
    """
    if not stage_percent:
        return fallback
    best_len, best_p = -1, None
    for stage, p in stage_percent.items():
        if not stage:
            continue
        # normalize dots/colons in stage name
        st = normalize_target_name(stage)
        if module_path.startswith(st):
            if len(st) > best_len:
                best_len, best_p = len(st), p
    return float(best_p) if best_p is not None else fallback


def _payload_to_clip(
    payload: Mapping[str, Any],
    percentile: float,
) -> Tuple[float, float]:
    """
    Derive (lo, hi) from observer payload with percentile override.
    Falls back to symmetric range if exact percentile keys are missing.
    """
    # try exact percentile in payload via helper
    try:
        lo, hi = fuse_observer_payload(payload, percentile)
        if lo < hi:
            return float(lo), float(hi)
    except Exception:
        pass

    # fallback to min/max symmetric
    lo2, hi2 = compute_symmetric_clip(payload, key_hi="max", key_lo="min")
    return float(lo2), float(hi2)


_ROTATION_ACT_CAPACITY = 65536


class _RotationActivationBuffer:
    def __init__(self, capacity: int = _ROTATION_ACT_CAPACITY) -> None:
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
        merged = torch.cat([self._tensor, data], dim=0)
        if merged.shape[0] > self.capacity:
            idx = torch.randperm(merged.shape[0])[: self.capacity]
            merged = merged[idx]
        self._tensor = merged.contiguous()

    def tensor(self) -> Optional[torch.Tensor]:
        return self._tensor


def _move_batch_to_device(sample: Any, device: torch.device) -> Any:
    if torch.is_tensor(sample):
        return sample.to(device)
    if isinstance(sample, dict):
        return {k: _move_batch_to_device(v, device) for k, v in sample.items()}
    if isinstance(sample, list):
        return [_move_batch_to_device(v, device) for v in sample]
    if isinstance(sample, tuple):
        return tuple(_move_batch_to_device(v, device) for v in sample)
    return sample


def _flatten_feature_tensor(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    if tensor is None or not torch.is_tensor(tensor):
        return None
    if tensor.dim() < 2:
        return None
    if tensor.dim() == 2:
        return tensor
    batch = tensor.shape[0]
    channels = tensor.shape[1]
    flat = tensor.reshape(batch, channels, -1).transpose(1, 2).reshape(-1, channels)
    return flat


def _extract_first_tensor(obj: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            tensor = _extract_first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(obj, Mapping):
        for item in obj.values():
            tensor = _extract_first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _collect_module_outputs(
    model: nn.Module,
    dataloader: Iterable[Any],
    targets: Sequence[str],
    max_batches: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if not targets or max_batches <= 0:
        return {}

    buffers = {name: _RotationActivationBuffer() for name in targets}
    handles: List[RemovableHandle] = []
    module_lookup = dict(model.named_modules())

    def _hook_factory(name: str):
        def _hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any):
            tensor = _extract_first_tensor(output)
            flat = _flatten_feature_tensor(tensor) if tensor is not None else None
            if flat is not None:
                buffers[name].add(flat)
        return _hook

    for name in targets:
        module = module_lookup.get(name)
        if module is None:
            continue
        handles.append(module.register_forward_hook(_hook_factory(name)))

    was_training = model.training
    model.eval()
    model.to(device)

    try:
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if idx >= max_batches:
                    break
                moved = _move_batch_to_device(batch, device)
                if isinstance(moved, dict):
                    model(**moved)
                elif isinstance(moved, (list, tuple)):
                    model(*moved)
                else:
                    model(moved)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    return {
        name: buf.tensor()
        for name, buf in buffers.items()
        if buf.tensor() is not None
    }


def _module_io_dims(module: nn.Module) -> Tuple[int, int]:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise ValueError(f"Module {module} lacks weight for rotation.")
    shape = tuple(weight.shape)
    if len(shape) < 2:
        raise ValueError(f"Unexpected weight shape {shape} for rotation.")
    out_dim = int(shape[0])
    in_dim = int(shape[1])
    if len(shape) > 2:
        for dim in shape[2:]:
            in_dim *= int(dim)
    return in_dim, out_dim


def _align_rotation(rotation: Optional[torch.Tensor], reference: torch.Tensor) -> Optional[torch.Tensor]:
    if rotation is None:
        return None
    return rotation.to(device=reference.device, dtype=reference.dtype)


def _apply_input_rotation(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor @ rotation
    batch = tensor.shape[0]
    channels = tensor.shape[1]
    flat = tensor.reshape(batch, channels, -1).transpose(1, 2).reshape(-1, channels)
    rotated = flat @ rotation
    return rotated.reshape(batch, -1, channels).transpose(1, 2).reshape(tensor.shape)


def _apply_output_rotation(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor @ rotation
    batch = tensor.shape[0]
    channels = tensor.shape[1]
    flat = tensor.reshape(batch, channels, -1).transpose(1, 2).reshape(-1, channels)
    rotated = flat @ rotation
    return rotated.reshape(batch, -1, channels).transpose(1, 2).reshape(tensor.shape)


def _install_injection_hooks(
    module: nn.Module,
    rotation_in: Optional[torch.Tensor],
    rotation_out: Optional[torch.Tensor],
) -> Tuple[List[int], List[RemovableHandle]]:
    hook_ids: List[int] = []
    handles: List[RemovableHandle] = []

    if rotation_in is not None:
        def _pre_hook(_module: nn.Module, inputs: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
            if not inputs:
                return None
            tensor = inputs[0]
            if not torch.is_tensor(tensor):
                return None
            rotated = _apply_input_rotation(tensor, rotation_in)
            return (rotated, *inputs[1:])

        handle = module.register_forward_pre_hook(_pre_hook)
        hook_ids.append(handle.id)
        handles.append(handle)

    if rotation_out is not None:
        def _post_hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
            if not torch.is_tensor(output):
                return output
            return _apply_output_rotation(output, rotation_out)

        handle = module.register_forward_hook(_post_hook)
        hook_ids.append(handle.id)
        handles.append(handle)

    return hook_ids, handles


# ---- Calibrator ------------------------------------------------------------------

@dataclass
class CalibratorConfig:
    default_bits: int = 8
    signed: bool = True
    default_percentile: float = 99.9
    # Optional stage→percentile overrides. Keys should match normalize_target_name().
    best_percent_map: Optional[Mapping[str, float]] = None
    # Optional stage whitelist. If provided, only modules under these stages are calibrated.
    stage_whitelist: Optional[Tuple[str, ...]] = None
    # Role whitelist. Defaults to weight+act.
    role_whitelist: Tuple[str, ...] = _SUPPORTED_ROLES


class PercentileCalibrator:
    """
    Build a calibration table from percentile summary.

    Usage:
        summary = read_percentile_summary(path)
        calib = PercentileCalibrator(summary, CalibratorConfig(...))
        table = calib.build_table()
    """

    def __init__(
        self,
        summary: SummaryLike,
        config: Optional[CalibratorConfig] = None,
    ) -> None:
        if not isinstance(summary, Mapping):
            raise TypeError("summary must be a Mapping")
        self.summary: SummaryLike = summary
        self.cfg = config or CalibratorConfig()

        # Pre-normalize whitelist stages for quick prefix filtering
        if self.cfg.stage_whitelist:
            self._wl = tuple(normalize_target_name(s) for s in self.cfg.stage_whitelist)
        else:
            self._wl = None

        # Normalize best_percent_map keys
        if self.cfg.best_percent_map:
            self._best_map = {
                normalize_target_name(k): float(v)
                for k, v in self.cfg.best_percent_map.items()
            }
        else:
            self._best_map = None

    # ---- Core API ---------------------------------------------------------------

    def build_table(self) -> CalibrationTable:
        table: CalibrationTable = {}
        observers = list(_iter_observer_items(self.summary))
        for obs_key, payload in observers:
            module_path, role = _extract_module_role(obs_key)
            if not role:
                # If summary omitted role suffix, skip silently.
                continue
            if not _is_supported_role(role):
                continue
            if self._wl and not any(module_path.startswith(p) for p in self._wl):
                continue
            if not self._role_allowed(role):
                continue

            entry = self.calibrate_entry(module_path, role, index=None, payload=payload)
            if entry is None:
                continue
            table[_key(module_path, role, index=None)] = entry
        return table

    def calibrate_entry(
        self,
        module_path: str,
        role: str,
        index: Optional[int],
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Optional[CalibrationEntry]:
        """
        Compute calibration for single (module_path, role, index).
        `payload` can be supplied directly to bypass summary lookup.
        """
        if not _is_supported_role(role):
            return None
        if self._wl and not any(module_path.startswith(p) for p in self._wl):
            return None
        if not self._role_allowed(role):
            return None

        ob = payload or self._lookup_payload(module_path, role)
        if not ob:
            return None
        # stage filter
        stage = self._infer_stage(module_path)
        if self._wl and stage not in self._wl:
            return None

        # choose percentile
        p_from_stage = _select_percentile_for_module(
            module_path, self._best_map, self.cfg.default_percentile
        )
        p_eff = choose_percentile(ob, p_from_stage)

        # compute clip
        lo, hi = _payload_to_clip(ob, p_eff)
        if not torch.isfinite(torch.tensor([lo, hi])).all():
            return None
        if hi <= lo:
            return None

        # range -> affine
        scale, zero = range_to_affine(lo, hi, bits=self.cfg.default_bits, signed=self.cfg.signed)

        entry: CalibrationEntry = {
            "scale": scale.detach().clone(),
            "zero_point": None if zero is None else zero.detach().clone(),
            "clip": (float(lo), float(hi)),
            "percentile": float(p_eff),
            "role": role,
            "module": module_path,
            "index": 0 if index is None else int(index),
            "bits": int(self.cfg.default_bits),
            "signed": bool(self.cfg.signed),
        }
        return entry

    # ---- Internals -------------------------------------------------------------

    def _role_allowed(self, role: str) -> bool:
        if not self.cfg.role_whitelist:
            return True
        r = role.lower()
        wl = tuple(x.lower() for x in self.cfg.role_whitelist)
        return r in wl or r.endswith("_quantizer") and any(r.startswith(x[:-1]) for x in wl)

    def _lookup_payload(self, module_path: str, role: str) -> Optional[Dict[str, Any]]:
        """Find observer payload for a given module_path and role."""
        # pass 1: exact key
        key1 = f"{module_path}.{role}"
        observers = self.summary.get("observers", {})
        ob = observers.get(key1)
        if isinstance(ob, Mapping):
            return dict(ob)

        # pass 2: nested
        ob2 = observers.get(module_path)
        if isinstance(ob2, Mapping):
            cand = ob2.get(role)
            if isinstance(cand, Mapping):
                return dict(cand)

        # pass 3: linear scan fallback
        for k, v in _iter_observer_items(self.summary):
            if not isinstance(v, Mapping):
                continue
            m, r = _extract_module_role(k)
            if m == module_path and (r == role or r is None and role in k):
                return dict(v)
        return None

    def _infer_stage(self, module_path: str) -> Optional[str]:
        from cobra.quantize.pct.targets import TARGET_PREFIXES

        path = module_path.lower()
        for stage, prefixes in TARGET_PREFIXES.items():
            for prefix in prefixes:
                if path.startswith(prefix):
                    return stage
        return None


# ---- Rotation controller ---------------------------------------------------------

def estimate_and_apply_rotation(
    model: nn.Module,
    dataloader: DataLoader,
    policy: RotationPolicy,
    method: Literal["bake", "inject"],
    klt_batches: int,
    manifest_out: Optional[Union[str, Path]] = None,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Estimate rotations offline and apply them by baking weights or injecting runtime hooks.
    """

    if not isinstance(policy, RotationPolicy):
        raise TypeError("policy must be a RotationPolicy instance.")
    rotation_method = method or policy.method
    if rotation_method not in ("bake", "inject"):
        raise ValueError(f"Unsupported rotation method: {rotation_method}")

    if device is None:
        try:
            first_param = next(model.parameters())
            inferred = first_param.device
        except StopIteration:
            inferred = torch.device("cpu")
        device = inferred
    else:
        device = device if isinstance(device, torch.device) else torch.device(device)

    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    spec = policy.spec
    axis = spec.axis
    rotate_in = axis in ("in", "io")
    rotate_out = axis in ("out", "io")
    spec_name = spec.name.lower()
    if spec_name not in ("klt", "hadamard"):
        raise ValueError(f"Unsupported rotation spec: {spec.name}")

    start_time = time.time()

    candidates = get_rotation_targets(model, None)
    targets = [(name, module) for name, module in candidates if policy.should_rotate(name, module)]

    manifest: Dict[str, Any] = {
        "version": 1,
        "spec": asdict(spec),
        "method": rotation_method,
        "items": [],
    }
    stats: Dict[str, Any] = {
        "rotation": spec.name,
        "axis": spec.axis,
        "method": rotation_method,
        "total_candidates": len(candidates),
        "policy_eligible": len(targets),
        "rotated": 0,
        "skipped": 0,
        "skipped_details": [],
    }
    handles: List[RemovableHandle] = []

    if not targets:
        stats["duration_sec"] = time.time() - start_time
        if manifest_out:
            save_rotation_manifest(manifest_out, manifest)
        return {"manifest": manifest, "stats": stats, "handles": handles}

    acts_in: Dict[str, torch.Tensor] = {}
    acts_out: Dict[str, torch.Tensor] = {}
    max_batches = max(1, int(klt_batches) if klt_batches else 1)
    target_names = [name for name, _ in targets]

    if spec_name == "klt":
        if dataloader is None:
            raise ValueError("KLT rotation requires a dataloader for activations.")
        acts_in = sample_layer_activations(model, dataloader, target_names, max_batches, device)
        if rotate_out:
            acts_out = _collect_module_outputs(model, dataloader, target_names, max_batches, device)

    for name, module in targets:
        entry: Dict[str, Any] = {
            "path": name,
            "rotation": spec.name,
            "axis": spec.axis,
            "method": rotation_method,
        }

        def _record_skip(reason: str) -> None:
            stats["skipped"] += 1
            stats.setdefault("skipped_details", []).append({"path": name, "reason": reason})

        try:
            in_dim, out_dim = _module_io_dims(module)
        except ValueError as exc:
            _record_skip(str(exc))
            continue

        entry["dims"] = {"in": in_dim, "out": out_dim}
        weight = getattr(module, "weight", None)
        if weight is None:
            _record_skip("module_has_no_weight")
            continue
        entry["weight_shape"] = tuple(weight.shape)

        R_in: Optional[torch.Tensor] = None
        R_out: Optional[torch.Tensor] = None

        if spec_name == "klt":
            acts_in_tensor = acts_in.get(name)
            if acts_in_tensor is None:
                if rotate_in:
                    _record_skip("missing_input_activations")
                    continue
                acts_in_tensor = torch.zeros((1, max(1, in_dim)), dtype=torch.float32)
            acts_out_tensor = acts_out.get(name) if rotate_out else None
            if rotate_out and acts_out_tensor is None:
                _record_skip("missing_output_activations")
                continue
            try:
                R_in, R_out = klt_rotation_from_acts(acts_in_tensor, acts_out_tensor, spec)
            except Exception as exc:
                _record_skip(f"klt_failure: {exc}")
                continue
        else:
            block = spec.block_size or max(8, min(256, max(in_dim, out_dim)))
            try:
                had_in, had_out = hadamard_rotation(
                    in_dim,
                    out_dim if rotate_out else None,
                    block_size=block,
                    dtype=torch.float32,
                    device=device,
                )
            except Exception as exc:
                _record_skip(f"hadamard_failure: {exc}")
                continue
            R_in = had_in if rotate_in else None
            R_out = had_out if rotate_out else None

        if rotate_in and R_in is None:
            _record_skip("missing_R_in")
            continue
        if rotate_out and R_out is None:
            _record_skip("missing_R_out")
            continue

        R_in_prepared = _align_rotation(R_in, weight) if R_in is not None else None
        R_out_prepared = _align_rotation(R_out, weight) if R_out is not None else None

        if rotation_method == "bake":
            try:
                if isinstance(module, nn.Linear):
                    bake_rotation_linear(module, R_in_prepared, R_out_prepared)
                elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    bake_rotation_convnd(module, R_in_prepared, R_out_prepared)
                else:
                    _record_skip("unsupported_module")
                    continue
            except Exception as exc:
                _record_skip(f"bake_failed: {exc}")
                continue
            entry["hooks"] = []
            manifest["items"].append(entry)
            stats["rotated"] += 1
            continue

        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and rotate_in:
            _record_skip("conv_injection_in_axis_not_supported")
            continue
        if R_in_prepared is None and R_out_prepared is None:
            _record_skip("empty_rotation_matrices")
            continue
        try:
            hook_ids, new_handles = _install_injection_hooks(module, R_in_prepared, R_out_prepared)
        except Exception as exc:
            _record_skip(f"inject_failed: {exc}")
            continue

        entry["hooks"] = hook_ids
        manifest["items"].append(entry)
        handles.extend(new_handles)
        stats["rotated"] += 1

    stats["duration_sec"] = time.time() - start_time
    stats["handles_installed"] = len(handles)

    if manifest_out:
        save_rotation_manifest(manifest_out, manifest)

    return {
        "manifest": manifest,
        "stats": stats,
        "handles": handles,
    }


def finalize_model(
    model: nn.Module,
    spec: FinalizeSpec,
    best_percent_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Freeze quantizers, apply best-percentile overrides, and pack weights according to FinalizeSpec.
    """
    best_map = load_best_percent_map(best_percent_path) if best_percent_path else {}
    manifest_items: List[Dict[str, Any]] = []
    counts = {"eligible": 0, "finalized": 0, "skipped": 0}

    deny = tuple((spec.denylist or ()))
    allow = tuple((spec.allowlist or ()))

    for name, module in model.named_modules():
        if not is_quant_eligible(module):
            continue
        counts["eligible"] += 1
        canonical_name = name or module.__class__.__name__
        lower_name = canonical_name.lower()

        if deny and any(lower_name.startswith(d.lower()) for d in deny):
            counts["skipped"] += 1
            continue
        if allow and not any(lower_name.startswith(a.lower()) for a in allow):
            counts["skipped"] += 1
            continue

        weight_q = getattr(module, "weight_quantizer", None)
        act_q = getattr(module, "act_quantizer", None)

        if weight_q is not None:
            apply_best_percent_overrides(canonical_name, weight_q, best_map)
            finalize_quant_params(weight_q, spec.symmetric, spec.per_channel)
        if act_q is not None:
            finalize_quant_params(act_q, spec.symmetric, False)

        if hasattr(module, "finalize"):
            module.finalize(spec)

        item = {
            "name": canonical_name,
            "module_type": module.__class__.__name__,
            "weight_bits": getattr(weight_q, "n_bits", None),
            "act_bits": getattr(act_q, "n_bits", None),
            "symmetric": spec.symmetric,
            "per_channel": spec.per_channel,
            "pack_meta": getattr(module, "pack_meta", {}),
            "weight_scale_shape": tuple(getattr(weight_q, "scale", torch.tensor([])).shape) if weight_q else (),
            "weight_zero_shape": tuple(getattr(weight_q, "round_zero_point", torch.tensor([])).shape) if weight_q and getattr(weight_q, "round_zero_point", None) is not None else (),
            "act_scale_shape": tuple(getattr(act_q, "scale", torch.tensor([])).shape) if act_q else (),
            "act_zero_shape": tuple(getattr(act_q, "round_zero_point", torch.tensor([])).shape) if act_q and getattr(act_q, "round_zero_point", None) is not None else (),
        }
        manifest_items.append(item)
        counts["finalized"] += 1

    return {
        "status": "ok",
        "counts": counts,
        "manifest": {
            "version": 1,
            "items": manifest_items,
            "spec": asdict(spec),
        },
    }


# ---- Convenience factory ---------------------------------------------------------

def build_table_from_file(
    pct_summary_path: Union[str, Path],
    default_bits: int = 8,
    signed: bool = True,
    best_percent_map: Optional[Mapping[str, float]] = None,
    stage_whitelist: Optional[Tuple[str, ...]] = None,
    default_percentile: float = 99.9,
) -> CalibrationTable:
    """
    One-shot convenience: read summary, build calibration table.
    """
    summary = read_percentile_summary(pct_summary_path)
    cfg = CalibratorConfig(
        default_bits=default_bits,
        signed=signed,
        default_percentile=default_percentile,
        best_percent_map=best_percent_map,
        stage_whitelist=stage_whitelist,
    )
    calibrator = PercentileCalibrator(summary, cfg)
    return calibrator.build_table()


# ---- Debug printing --------------------------------------------------------------

def summarize_table(table: Mapping[str, Mapping[str, Any]], top_k: int = 20) -> str:
    """
    Return a short human-readable summary of the calibration table.
    """
    n = len(table)
    if n == 0:
        return "[Calib] empty table"
    lines = [f"[Calib] entries={n} (showing up to {min(top_k, n)})"]
    k = 0
    for name, e in table.items():
        if k >= top_k:
            break
        lo, hi = e.get("clip", (None, None))
        p = e.get("percentile", None)
        bits = e.get("bits", None)
        lines.append(f"  - {name}: p={p}, clip=({lo:.6g},{hi:.6g}), bits={bits}")
        k += 1
    return "\n".join(lines)
