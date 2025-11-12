# cobra/quantize/wrap/utils.py
"""
Utility functions for safe parameter transfer and inspection between
floating-point and quantized modules.

Used internally by wrap_replace.py during wrapper substitution.

All helpers are standalone and free of side effects on global state.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Dict, List
import torch
from torch import nn

from cobra.quantize.utils import bake_rotation_linear, bake_rotation_convnd, ConvNd


# ---------------------------------------------------------------------------
# Core transfer helpers
# ---------------------------------------------------------------------------


def _copy_tensor(dst: torch.Tensor, src: torch.Tensor) -> None:
    if dst.shape != src.shape:
        raise RuntimeError(f"Tensor shape mismatch: {dst.shape} vs {src.shape}")
    dst.data.copy_(src.detach().to(dst.device, dst.dtype))


def transfer_linear(src: nn.Linear, dst: nn.Module, strict: bool = True) -> None:
    """
    Copy parameters from FP Linear to QuantLinear.
    """
    if not hasattr(dst, "weight"):
        raise AttributeError("Destination module has no weight attribute")

    try:
        _copy_tensor(dst.weight, src.weight)
    except Exception as e:
        if strict:
            raise
        else:
            print(f"[transfer_linear] warning: weight copy failed: {e}")

    if hasattr(dst, "bias"):
        if src.bias is None:
            with torch.no_grad():
                dst.bias.zero_()
        else:
            try:
                _copy_tensor(dst.bias, src.bias)
            except Exception as e:
                if strict:
                    raise
                else:
                    print(f"[transfer_linear] warning: bias copy failed: {e}")

    # provenance tag
    dst.__wrapped_from__ = "Linear"
    dst.__wrapped_shape__ = tuple(src.weight.shape)


def transfer_conv(src: nn.modules.conv._ConvNd, dst: nn.Module, strict: bool = True) -> None:
    """
    Copy parameters from FP ConvNd to QuantConvNd.
    """
    if not hasattr(dst, "weight"):
        raise AttributeError("Destination module has no weight attribute")

    try:
        _copy_tensor(dst.weight, src.weight)
    except Exception as e:
        if strict:
            raise
        else:
            print(f"[transfer_conv] warning: weight copy failed: {e}")

    if hasattr(dst, "bias"):
        if src.bias is None:
            with torch.no_grad():
                dst.bias.zero_()
        else:
            try:
                _copy_tensor(dst.bias, src.bias)
            except Exception as e:
                if strict:
                    raise
                else:
                    print(f"[transfer_conv] warning: bias copy failed: {e}")

    dst.__wrapped_from__ = type(src).__name__
    dst.__wrapped_shape__ = tuple(src.weight.shape)


# ---------------------------------------------------------------------------
# State inspection helpers
# ---------------------------------------------------------------------------


def describe_module(m: nn.Module) -> str:
    """
    Return short textual description of module type, shape, and device.
    """
    cls_name = type(m).__name__
    params = sum(p.numel() for p in m.parameters(recurse=True))
    device = next(m.parameters(), torch.empty(0)).device if any(m.parameters()) else "cpu"
    return f"{cls_name}(params={params}, device={device})"


def safe_to_cpu_state(m: nn.Module) -> dict[str, torch.Tensor]:
    """
    Detach and move all tensors in state_dict to CPU.
    """
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def verify_shape_compat(src: nn.Module, dst: nn.Module) -> bool:
    """
    Verify shape compatibility between FP and quantized modules.
    """
    if hasattr(src, "weight") and hasattr(dst, "weight"):
        if src.weight.shape != dst.weight.shape:
            return False
    if hasattr(src, "bias") and hasattr(dst, "bias"):
        if src.bias is not None and dst.bias is not None:
            if src.bias.shape != dst.bias.shape:
                return False
    return True


# ---------------------------------------------------------------------------
# Debug print
# ---------------------------------------------------------------------------


def print_transfer_summary(src: nn.Module, dst: nn.Module) -> None:
    """
    Print concise summary of a parameter transfer.
    """
    print(f"[Transfer] {_short_type(src)} -> {_short_type(dst)}")
    if hasattr(src, "weight") and hasattr(dst, "weight"):
        print(f"  weight: {tuple(src.weight.shape)} -> {tuple(dst.weight.shape)}")
    if hasattr(src, "bias") and hasattr(dst, "bias"):
        if src.bias is not None and dst.bias is not None:
            print(f"  bias: {tuple(src.bias.shape)} -> {tuple(dst.bias.shape)}")


def _short_type(m: Any) -> str:
    return type(m).__name__


__all__ = [
    "transfer_linear",
    "transfer_conv",
    "verify_shape_compat",
    "describe_module",
    "safe_to_cpu_state",
    "print_transfer_summary",
]


# ---------------------------------------------------------------------------
# Calibration markers
# ---------------------------------------------------------------------------


def mark_calibrated(mod: nn.Module) -> None:
    setattr(mod, "_pct_calibrated", True)


def is_calibrated(mod: nn.Module) -> bool:
    return bool(getattr(mod, "_pct_calibrated", False))


__all__ += ["mark_calibrated", "is_calibrated"]


_ROTATION_HOOK_NAME_IN = "rotation_in"
_ROTATION_HOOK_NAME_OUT = "rotation_out"


def _extract_rotation_tensor(hook: Any, var_name: str) -> Optional[torch.Tensor]:
    code = getattr(hook, "__code__", None)
    closure = getattr(hook, "__closure__", None)
    if code is None or closure is None:
        return None
    freevars = code.co_freevars
    if var_name not in freevars:
        return None
    idx = freevars.index(var_name)
    if idx >= len(closure):
        return None
    tensor = closure[idx].cell_contents
    if isinstance(tensor, torch.Tensor):
        return tensor
    return None


def _collect_rotation_hooks(
    module: nn.Module,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[int], List[int]]:
    rot_in: Optional[torch.Tensor] = None
    rot_out: Optional[torch.Tensor] = None
    pre_ids: List[int] = []
    post_ids: List[int] = []

    pre_hooks: Dict[int, Any] = getattr(module, "_forward_pre_hooks", {})  # type: ignore[assignment]
    for hid, hook in list(pre_hooks.items()):
        tensor = _extract_rotation_tensor(hook, _ROTATION_HOOK_NAME_IN)
        if tensor is not None:
            rot_in = tensor
            pre_ids.append(hid)

    post_hooks: Dict[int, Any] = getattr(module, "_forward_hooks", {})  # type: ignore[assignment]
    for hid, hook in list(post_hooks.items()):
        tensor = _extract_rotation_tensor(hook, _ROTATION_HOOK_NAME_OUT)
        if tensor is not None:
            rot_out = tensor
            post_ids.append(hid)

    return rot_in, rot_out, pre_ids, post_ids


def has_rotation_injection(module: nn.Module) -> bool:
    """
    Best-effort detection for runtime rotation hooks installed via rotation controller.
    """
    rot_in, rot_out, pre_ids, post_ids = _collect_rotation_hooks(module)
    return bool((rot_in is not None and pre_ids) or (rot_out is not None and post_ids))


def force_bake_if_injected(module: nn.Module) -> bool:
    """
    If the module carries runtime rotation hooks, bake them into the weight and remove hooks.
    """
    rot_in, rot_out, pre_ids, post_ids = _collect_rotation_hooks(module)
    if rot_in is None and rot_out is None:
        return False

    if isinstance(module, nn.Linear):
        bake_rotation_linear(module, rot_in, rot_out)
    elif isinstance(module, ConvNd):
        bake_rotation_convnd(module, rot_in, rot_out)
    else:
        return False

    pre_hooks: Dict[int, Any] = getattr(module, "_forward_pre_hooks", {})  # type: ignore[assignment]
    for hid in pre_ids:
        pre_hooks.pop(hid, None)

    post_hooks: Dict[int, Any] = getattr(module, "_forward_hooks", {})  # type: ignore[assignment]
    for hid in post_ids:
        post_hooks.pop(hid, None)

    return True


__all__ += ["has_rotation_injection", "force_bake_if_injected"]
