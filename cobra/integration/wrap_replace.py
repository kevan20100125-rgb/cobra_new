# cobra/integration/wrap_replace.py
"""
Quantized wrapper replacement utilities.

Goals
-----
- Replace FP modules with quantized wrappers according to a WrapPolicy.
- Support dry-run preview and real in-place replacement.
- Produce a manifest for downstream stages (calib/rotation/apply).
- Optional rollback using snapshots captured before replacement.

Public API
---------
wrap_model(model, policy, dry_run=False, strict=True, capture_snapshot=True) -> dict
unwrap_model(model, manifest: dict, strict=True) -> None

Manifest schema (returned by wrap_model)
---------------------------------------
{
  "version": 1,
  "items": [
    {
      "path": "vision_backbone.dino_featurizer.blocks.0.attn.qkv",
      "stage": "vision.dino",
      "from": "torch.nn.modules.linear.Linear",
      "to":   "cobra.quantize.int_linear.QuantLinear",
      "params_from": 589824,
      "params_to":   589824,
      "snapshot": {                     # present only if capture_snapshot=True and dry_run=False
        "fqcn": "torch.nn.modules.linear.Linear",
        "state_dict": {...}             # CPU tensors
      }
    },
    ...
  ],
  "summary": {
    "by_stage": {"vision.dino": 123, "vision.siglip": 45, "projector": 67, "llm": 89},
    "by_type":  {"Linear→QuantLinear": 250, "Conv2d→QuantConv2d": 74},
    "skipped":  ["projector.ln", "llm_backbone.embed_tokens", ...]  # first N only
  }
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from cobra.quantize.wrap.registry import get_wrapper, is_supported
from cobra.quantize.wrap.policy import WrapPolicy, normalize_stage_name
from cobra.quantize.wrap.utils import has_rotation_injection, force_bake_if_injected
from cobra.pipeline.pct_schema import normalize_stage as pct_normalize_stage

log = logging.getLogger("wrap_replace")


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def _fully_qualified_name(obj: Any) -> str:
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__name__}"


def _iter_with_parents(model: nn.Module) -> Iterable[Tuple[nn.Module, str, nn.Module, str]]:
    """
    Yield (parent, name, module, path) for every leaf in module tree.
    """
    root_path = ""

    def _walk(parent: nn.Module, prefix: str) -> Iterable[Tuple[nn.Module, str, nn.Module, str]]:
        for name, child in parent.named_children():
            path = f"{prefix}.{name}" if prefix else name
            # only replace leaves or direct supported modules; recurse otherwise
            if any(child.named_children()):
                yield from _walk(child, path)
            else:
                yield parent, name, child, path

    yield from _walk(model, root_path)


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters(recurse=True))


def _cpu_state_dict(m: nn.Module) -> Dict[str, torch.Tensor]:
    # minimal CPU snapshot
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}


# ---------------------------------------------------------------------------
# Core replacement
# ---------------------------------------------------------------------------


def wrap_model(
    model: nn.Module,
    policy: WrapPolicy,
    dry_run: bool = False,
    strict: bool = True,
    capture_snapshot: bool = True,
    skip_if_already_wrapped: bool = True,
    max_skipped_report: int = 50,
) -> Dict[str, Any]:
    """
    Replace eligible modules in-place with quantized wrappers.

    Parameters
    ----------
    model : nn.Module
        Model to transform in-place.
    policy : WrapPolicy
        Inclusion/exclusion and stage filtering rules.
    dry_run : bool
        If True, do not mutate the model; only return a manifest preview.
    strict : bool
        If True, raise on shape/type inconsistencies.
    capture_snapshot : bool
        If True and not dry_run, store original FP state for rollback.
    skip_if_already_wrapped : bool
        If True, do not re-wrap modules that already look quantized.
    max_skipped_report : int
        Cap the number of skipped entries recorded in manifest.summary.skipped.

    Returns
    -------
    manifest : dict
        See schema in module docstring.
    """
    items: List[Dict[str, Any]] = []
    by_stage: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    skipped: List[str] = []

    for parent, name, module, path in _iter_with_parents(model):
        stage = normalize_stage_name(path)
        try:
            stage_key = pct_normalize_stage(path)
        except Exception:
            stage_key = stage

        # Eligibility by policy
        if not policy.allows(path, module):
            if len(skipped) < max_skipped_report:
                skipped.append(path)
            continue

        # Avoid re-wrapping quantized modules
        if skip_if_already_wrapped and not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # if not a standard FP layer, check if supported anyway
            if not is_supported(module):
                if len(skipped) < max_skipped_report:
                    skipped.append(path)
                continue

        wrap_fn = get_wrapper(module)
        if wrap_fn is None:
            if len(skipped) < max_skipped_report:
                skipped.append(path)
            continue

        from_fqcn = _fully_qualified_name(module)
        params_from = _count_params(module)

        record: Dict[str, Any] = {
            "path": path,
            "stage": stage,
            "stage_key": stage_key,
            "from": from_fqcn,
            "params_from": params_from,
        }

        if not dry_run and has_rotation_injection(module):
            baked = force_bake_if_injected(module)
            if baked:
                log.info(f"[wrap_replace] Baked injected rotation at {path} before wrapping.")
            else:
                log.info(f"[wrap_replace] Detected rotation hooks at {path} but failed to bake them; continuing.")

        if dry_run:
            # Probe the destination type by instantiating on a clone? avoid heavy ops.
            # We can infer "to" from wrapper's qualname by temporarily building a tiny module,
            # but that would require device/dtype. Instead, call and discard with a safe try on CPU.
            try:
                probe = wrap_fn(module)
                record["to"] = _fully_qualified_name(probe)
                record["params_to"] = _count_params(probe)
                # no mutation in dry-run: do not install probe
                del probe
            except Exception:
                record["to"] = "<unknown>"
                record["params_to"] = params_from
            items.append(record)
            by_stage[stage] = by_stage.get(stage, 0) + 1
            key = f"{record['from'].split('.')[-1]}→{record['to'].split('.')[-1]}"
            by_type[key] = by_type.get(key, 0) + 1
            continue

        # Real replacement
        try:
            wrapped = wrap_fn(module)
        except Exception as e:
            if strict:
                raise
            if len(skipped) < max_skipped_report:
                skipped.append(f"{path} [wrap-failed: {e}]")
            continue

        # Basic shape sanity (Linear / Conv)
        if strict:
            if isinstance(module, nn.Linear):
                if not hasattr(wrapped, "weight") or wrapped.weight.shape != module.weight.shape:
                    raise RuntimeError(f"[wrap] shape mismatch at {path}: {wrapped.weight.shape} vs {module.weight.shape}")
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if not hasattr(wrapped, "weight") or wrapped.weight.shape != module.weight.shape:
                    raise RuntimeError(f"[wrap] shape mismatch at {path}: {wrapped.weight.shape} vs {module.weight.shape}")

        # Annotate provenance for later stages
        try:
            setattr(wrapped, "__wrapped_from__", type(module).__name__)
            setattr(wrapped, "__wrapped_path__", path)
            meta = {"stage": stage, "stage_key": stage_key, "path": path, "observer": None, "clip": None}
            setattr(wrapped, "quant_meta", meta)
        except Exception:
            pass

        # Install
        parent._modules[name] = wrapped  # in-place swap

        params_to = _count_params(wrapped)
        to_fqcn = _fully_qualified_name(wrapped)

        if capture_snapshot:
            state_cpu = _cpu_state_dict(module)
            param_dtype = None
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                param_dtype = str(module.weight.dtype)
            record["snapshot"] = {
                "fqcn": from_fqcn,
                "state_dict": state_cpu,
                "param_dtype": param_dtype,
            }

        record["to"] = to_fqcn
        record["params_to"] = params_to
        items.append(record)

        by_stage[stage_key] = by_stage.get(stage_key, 0) + 1
        key = f"{from_fqcn.split('.')[-1]}→{to_fqcn.split('.')[-1]}"
        by_type[key] = by_type.get(key, 0) + 1

    manifest = {
        "version": 1,
        "items": items,
        "summary": {
            "by_stage": by_stage,
            "by_type": by_type,
            "skipped": skipped,
        },
    }
    return manifest


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def _resolve_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name:
        return None
    token = name.split(".")[-1]
    return getattr(torch, token, None)


def _apply_dtype(module: nn.Module, dtype: Optional[torch.dtype]) -> None:
    if dtype is None:
        return
    for param in module.parameters(recurse=True):
        param.data = param.data.to(dtype=dtype)
    for buf in module.buffers(recurse=True):
        buf.data = buf.data.to(dtype=dtype)


def _rebuild_module(fqcn: str, state_dict: Dict[str, torch.Tensor], param_dtype: Optional[str] = None) -> nn.Module:
    """
    Instantiate a module by FQCN and load the provided state_dict.
    Only supports simple nn.Linear/ConvNd which have shape encoded in weights.
    """
    parts = fqcn.split(".")
    if len(parts) < 2:
        raise ValueError(f"Invalid fqcn: {fqcn}")
    # Resolve class
    mod_name = ".".join(parts[:-1])
    cls_name = parts[-1]
    mod = __import__(mod_name, fromlist=[cls_name])
    cls = getattr(mod, cls_name)

    # Minimal constructors from state
    if cls is nn.Linear:
        w = state_dict["weight"]
        bias = "bias" in state_dict and state_dict["bias"] is not None
        m = nn.Linear(w.shape[1], w.shape[0], bias=bias)
    elif cls in (nn.Conv1d, nn.Conv2d, nn.Conv3d):
        w = state_dict["weight"]
        out_c, in_c_per_group, *ks = w.shape
        # Heuristic reconstruction assuming groups=1 and stride=1, padding=0, dilation=1
        # Users can rewire manually if needed.
        dim = 1 if isinstance(w, torch.Tensor) and w.dim() == 3 else 2 if w.dim() == 4 else 3
        if dim == 1:
            m = nn.Conv1d(in_c_per_group, out_c, kernel_size=ks, bias=("bias" in state_dict))
        elif dim == 2:
            m = nn.Conv2d(in_c_per_group, out_c, kernel_size=ks, bias=("bias" in state_dict))
        else:
            m = nn.Conv3d(in_c_per_group, out_c, kernel_size=ks, bias=("bias" in state_dict))
    else:
        # Fallback: try default ctor then load
        m = cls()

    m.load_state_dict(state_dict)
    dtype = _resolve_dtype(param_dtype)
    _apply_dtype(m, dtype)
    return m


def unwrap_model(
    model: nn.Module,
    manifest: Dict[str, Any],
    strict: bool = True,
) -> None:
    """
    Roll back replacements using snapshots included in the manifest.

    Notes
    -----
    - Only entries that contain a "snapshot" will be restored.
    - Paths not found are ignored unless strict=True.
    """
    items: List[Dict[str, Any]] = manifest.get("items", [])
    # Build fast lookup by path
    restore_by_path = {it["path"]: it for it in items if "snapshot" in it}

    # Walk and restore
    for parent, name, module, path in _iter_with_parents(model):
        if path not in restore_by_path:
            continue
        snap = restore_by_path[path]["snapshot"]
        try:
            restored = _rebuild_module(snap["fqcn"], snap["state_dict"], snap.get("param_dtype"))
        except Exception as e:
            if strict:
                raise RuntimeError(f"[unwrap] failed to rebuild {path}: {e}") from e
            else:
                continue
        parent._modules[name] = restored  # in-place restore


__all__ = [
    "wrap_model",
    "unwrap_model",
]
