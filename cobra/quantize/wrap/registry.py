# cobra/quantize/wrap/registry.py
"""
Wrapper registry mapping FP modules to quantized wrappers.

Public API:
    - register_wrapper(fp_cls, creator)
    - get_wrapper(module) -> Optional[Callable[[nn.Module], nn.Module]]
    - is_supported(module) -> bool

The returned callable from `get_wrapper` must:
    - Accept the FP module instance (src) and return a quantized module (dst).
    - Preserve shapes and basic hyperparameters.
    - Copy weights/bias to the destination.
    - Leave quantization parameters (scale/zero_point) uninitialized for later calibration.

This file is dependency-light and safe to import anywhere.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Type, Any, Sequence, List, Tuple

import torch
from torch import nn
from .utils import mark_calibrated, is_calibrated
from ..utils import eligible_module

# --- Optional quant backends -------------------------------------------------

_QLinear = None
_QConv1d = None
_QConv2d = None
_QConv3d = None

try:
    from cobra.quantize.int_linear import QuantLinear as _QLinear  # type: ignore
except Exception:
    _QLinear = None

try:
    from cobra.quantize.int_conv import (  # type: ignore
        QuantConv1d as _QConv1d,
        QuantConv2d as _QConv2d,
        QuantConv3d as _QConv3d,
    )
except Exception:
    _QConv1d = _QConv2d = _QConv3d = None


# --- Helpers -----------------------------------------------------------------


def _to_device_dtype(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if t is None:
        return t
    return t.detach().to(device=ref.device, dtype=ref.dtype)


def _copy_linear(src: nn.Linear, dst: nn.Module) -> None:
    # weight
    if hasattr(dst, "weight") and src.weight is not None:
        dst.weight.data.copy_(_to_device_dtype(src.weight, dst.weight))
    # bias
    if hasattr(dst, "bias"):
        if src.bias is None and getattr(dst, "bias", None) is not None:
            # remove bias if dst created with bias but src has none
            with torch.no_grad():
                dst.bias.zero_()
        elif src.bias is not None and getattr(dst, "bias", None) is not None:
            dst.bias.data.copy_(_to_device_dtype(src.bias, dst.bias))


def _copy_conv(src: nn.modules.conv._ConvNd, dst: nn.Module) -> None:
    if hasattr(dst, "weight") and src.weight is not None:
        dst.weight.data.copy_(_to_device_dtype(src.weight, dst.weight))
    if hasattr(dst, "bias"):
        if src.bias is None and getattr(dst, "bias", None) is not None:
            with torch.no_grad():
                dst.bias.zero_()
        elif src.bias is not None and getattr(dst, "bias", None) is not None:
            dst.bias.data.copy_(_to_device_dtype(src.bias, dst.bias))


def _annotate(dst: nn.Module, src: nn.Module, module_path: Optional[str] = None) -> None:
    # Light provenance breadcrumbs for later stages (calib/rotation/export)
    try:
        setattr(dst, "__wrapped_from__", src.__class__.__name__)
        if module_path is not None:
            setattr(dst, "__wrapped_path__", module_path)
    except Exception:
        pass


# --- Creators ----------------------------------------------------------------


def _ensure_calib_interface(dst: nn.Module) -> None:
    if hasattr(dst, "load_calib"):
        return

    def load_calib(calib: Mapping[str, Any]) -> None:
        applied = False
        for role in ("weight_quantizer", "act_quantizer"):
            quant = getattr(dst, role, None)
            if quant is None:
                continue
            payload = calib.get(role) if isinstance(calib, Mapping) else None
            if not isinstance(payload, Mapping):
                continue
            scale = payload.get("scale")
            zero = payload.get("zero")
            if scale is not None and hasattr(quant, "scale"):
                quant.scale = torch.tensor(scale, dtype=torch.float32, device=getattr(quant.scale, "device", "cpu"))
                applied = True
            if zero is not None and not quant.disable_zero_point and hasattr(quant, "round_zero_point"):
                quant.round_zero_point = torch.tensor(int(zero), dtype=torch.int32, device=getattr(quant.round_zero_point, "device", "cpu"))
                applied = True
        if applied:
            mark_calibrated(dst)

    setattr(dst, "load_calib", load_calib)


def _copy_calibration_state(src: nn.Module, dst: nn.Module) -> bool:
    copied = False
    for role in ("weight_quantizer", "act_quantizer"):
        src_quant = getattr(src, role, None)
        dst_quant = getattr(dst, role, None)
        if src_quant is None or dst_quant is None:
            continue
        if hasattr(src_quant, "scale") and hasattr(dst_quant, "scale"):
            src_scale = getattr(src_quant, "scale", None)
            dst_scale = getattr(dst_quant, "scale", None)
            if isinstance(src_scale, torch.Tensor) and isinstance(dst_scale, torch.Tensor):
                dst_quant.scale = src_scale.detach().clone().to(dst_scale.device)
                copied = True
        if hasattr(src_quant, "round_zero_point") and hasattr(dst_quant, "round_zero_point"):
            src_zero = getattr(src_quant, "round_zero_point", None)
            dst_zero = getattr(dst_quant, "round_zero_point", None)
            if isinstance(src_zero, torch.Tensor) and isinstance(dst_zero, torch.Tensor):
                dst_quant.round_zero_point = src_zero.detach().clone().to(dst_zero.device)
                copied = True
    if copied:
        mark_calibrated(dst)
    return copied


def _make_linear_creator() -> Optional[Callable[[nn.Linear], nn.Module]]:
    if _QLinear is None:
        return None

    def _creator(src: nn.Linear, module_path: Optional[str] = None) -> nn.Module:
        dst = _QLinear(
            in_features=src.in_features,
            out_features=src.out_features,
            bias=src.bias is not None,
        )
        # match device/dtype
        dst.to(device=src.weight.device, dtype=src.weight.dtype)
        _copy_linear(src, dst)
        _annotate(dst, src, module_path)
        _ensure_calib_interface(dst)
        if is_calibrated(src):
            if not _copy_calibration_state(src, dst):
                mark_calibrated(dst)
        return dst

    return _creator


def _make_conv_creator(dim: int) -> Optional[Callable[[nn.modules.conv._ConvNd], nn.Module]]:
    klass = {1: _QConv1d, 2: _QConv2d, 3: _QConv3d}.get(dim)
    if klass is None:
        return None

    def _creator(src: nn.modules.conv._ConvNd, module_path: Optional[str] = None) -> nn.Module:
        kwargs = dict(
            in_channels=src.in_channels,
            out_channels=src.out_channels,
            kernel_size=src.kernel_size,
            stride=src.stride,
            padding=src.padding,
            dilation=src.dilation,
            groups=src.groups,
            bias=src.bias is not None,
            padding_mode=getattr(src, "padding_mode", "zeros"),
        )
        dst = klass(**kwargs)  # type: ignore[operator]
        dst.to(device=src.weight.device, dtype=src.weight.dtype)
        _copy_conv(src, dst)
        _annotate(dst, src, module_path)
        _ensure_calib_interface(dst)
        if is_calibrated(src):
            if not _copy_calibration_state(src, dst):
                mark_calibrated(dst)
        return dst

    return _creator


# --- Registry core -----------------------------------------------------------

Creator = Callable[[nn.Module], nn.Module]

# Internal mapping from FP class to a creator factory
_WRITABLE_CREATORS: Dict[Type[nn.Module], Callable[..., Optional[Creator]]] = {}

# Concrete creators (resolved now) from FP class to a callable wrapper
_WRAP_CREATORS: Dict[Type[nn.Module], Creator] = {}


def _bootstrap_defaults() -> None:
    # Map FP -> factory
    _WRITABLE_CREATORS.clear()
    _WRITABLE_CREATORS[nn.Linear] = _make_linear_creator
    _WRITABLE_CREATORS[nn.Conv1d] = lambda: _make_conv_creator(1)
    _WRITABLE_CREATORS[nn.Conv2d] = lambda: _make_conv_creator(2)
    _WRITABLE_CREATORS[nn.Conv3d] = lambda: _make_conv_creator(3)

    # Resolve available creators based on backend availability
    _WRAP_CREATORS.clear()
    for fp_cls, factory in _WRITABLE_CREATORS.items():
        try:
            creator = factory()
        except Exception:
            creator = None
        if creator is not None:
            # Bind without path; wrap_replace can pass module_path via functools.partial
            def _bind(c: Creator) -> Creator:
                def _wrapped(src: nn.Module) -> nn.Module:
                    # creator may accept (src, module_path) or (src)
                    try:
                        return c(src)  # type: ignore[misc]
                    except TypeError:
                        # If creator expects (src, module_path), we pass None for registry default
                        return c(src, None)  # type: ignore[misc]
                return _wrapped

            _WRAP_CREATORS[fp_cls] = _bind(creator)  # type: ignore[arg-type]


_bootstrap_defaults()


# --- Public API --------------------------------------------------------------


def register_wrapper(
    fp_cls: Type[nn.Module],
    creator: Callable[..., nn.Module],
) -> None:
    """
    Register or override a wrapper.

    `creator` must accept at least (src_module) and return a wrapped module.
    It may optionally accept `module_path` as a second positional arg.
    """
    if not isinstance(fp_cls, type) or not issubclass(fp_cls, nn.Module):
        raise TypeError("fp_cls must be subclass of nn.Module")

    # Normalize to a no-path callable
    def _wrapped(src: nn.Module) -> nn.Module:
        try:
            return creator(src)
        except TypeError:
            return creator(src, None)

    _WRAP_CREATORS[fp_cls] = _wrapped


def get_wrapper(module: nn.Module) -> Optional[Callable[[nn.Module], nn.Module]]:
    """
    Return a callable that wraps `module` into its quantized counterpart.
    If the exact type is not found, try base classes (Linear/ConvNd).
    """
    mtype = type(module)
    # Fast path
    if mtype in _WRAP_CREATORS:
        return _WRAP_CREATORS[mtype]

    # Fallbacks: ConvNd family
    if isinstance(module, nn.Conv1d) and nn.Conv1d in _WRAP_CREATORS:
        return _WRAP_CREATORS[nn.Conv1d]
    if isinstance(module, nn.Conv2d) and nn.Conv2d in _WRAP_CREATORS:
        return _WRAP_CREATORS[nn.Conv2d]
    if isinstance(module, nn.Conv3d) and nn.Conv3d in _WRAP_CREATORS:
        return _WRAP_CREATORS[nn.Conv3d]
    if isinstance(module, nn.Linear) and nn.Linear in _WRAP_CREATORS:
        return _WRAP_CREATORS[nn.Linear]

    return None


def is_supported(module: nn.Module) -> bool:
    """True if a quantized wrapper is available for this module."""
    return get_wrapper(module) is not None


# Export minimal surface
__all__ = [
    "register_wrapper",
    "get_wrapper",
    "is_supported",
    "get_rotation_targets",
]


def get_rotation_targets(
    model: nn.Module,
    user_targets: Optional[Sequence[str]] = None,
) -> List[Tuple[str, nn.Module]]:
    """
    Enumerate Linear/Conv modules eligible for rotation planning, filtered by optional prefixes.
    """
    prefixes: Tuple[str, ...] = tuple(
        p.strip() for p in (user_targets or []) if isinstance(p, str) and p.strip()
    )
    targets: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if not eligible_module(module):
            continue
        if prefixes and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        targets.append((name, module))
    return targets
