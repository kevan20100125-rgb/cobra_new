from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Literal, Optional, Tuple, Union, Sequence, Dict, Any, Mapping, TYPE_CHECKING, Type

import copy
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:  # pragma: no cover
    from .int_conv import QuantConv1d, QuantConv2d, QuantConv3d
    from .int_linear import QuantLinear
    from .int_matmul import QuantMatMul
from .normalized_modules import flatten_conv_weight, restore_conv_weight
from .quantizer import UniformAffineQuantizer

class NoHookContext:
    def __init__(self, module):
        self.module = module
        self.hooks = []

    def __enter__(self):
        # 保存hooks
        for hook_id in list(self.module._forward_hooks.keys()):
            self.hooks.append((hook_id, self.module._forward_hooks.pop(hook_id)))

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复hooks
        for hook_id, hook in self.hooks:
            self.module._forward_hooks[hook_id] = hook

class Logger(object):
    def __init__(self, folder="logs"):
        # 获取当前时间并格式化为字符串
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 创建日志文件夹（如果不存在）
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # 定义日志文件名
        filename = os.path.join(folder, f"log_{current_time}.txt")
        
        # 打开日志文件
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()


def set_seed(seed):
    torch.manual_seed(seed)  # 设置 CPU 上的随机数种子
    torch.cuda.manual_seed(seed)  # 设置当前 GPU 上的随机数种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 上的随机数种子（如果有多个 GPU）
    np.random.seed(seed)  # 设置 NumPy 的随机数种子
    random.seed(seed)  # 设置 Python 自带的随机数种子

    # 如果使用了 CuDNN 后端
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 确保卷积算法的选择是确定的
def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


@dataclass
class RotationSpec:
    """Describe a rotation applied before/after quantization; independent from smoothing-based quant flows."""

    name: Literal["hadamard", "klt"]
    scope: Literal["per_tensor", "per_channel"]
    axis: Literal["in", "out", "io"]
    block_size: Optional[int] = None
    whiten: bool = False  # Only meaningful for KLT
    eps: float = 1e-6


@dataclass
class FinalizeSpec:
    """
    Parameters governing final quantization export (bits, packing, verification).
    """

    weight_bits: int
    act_bits: int
    symmetric: bool = True
    per_channel: bool = True
    rounding: Literal["nearest", "stochastic"] = "nearest"
    pack_linear: bool = True
    pack_conv: bool = True
    export_scales: bool = True
    denylist: Tuple[str, ...] = ()
    allowlist: Optional[Tuple[str, ...]] = None
    verify_batches: int = 0
    atol: float = 1e-2
    rtol: float = 1e-1


def _validate_rotation_matrix(matrix: torch.Tensor, target_dim: int, name: str) -> None:
    if matrix.dim() != 2:
        raise ValueError(f"{name} must be 2D, got {matrix.dim()}D.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got {matrix.shape}.")
    if matrix.shape[0] != target_dim:
        raise ValueError(
            f"{name} has size {matrix.shape[0]}, expected {target_dim} to match the weight."
        )


def apply_rotation(
    weight: torch.Tensor, R_in: Optional[torch.Tensor], R_out: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Rotate a weight tensor as R_out^T @ W @ R_in around quantization steps.

    The rotation is performed prior to weight quantization (e.g., calibration) and can be baked
    afterward without relying on any smoothing-based quantization tricks.
    """
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor.")
    if R_in is None and R_out is None:
        return weight

    if weight.dim() not in (1, 2, 3, 4, 5):
        raise ValueError(f"Unsupported weight with {weight.dim()} dims.")

    flattened: torch.Tensor
    shape_meta: Optional[Tuple[int, ...]] = None
    squeezed_row = False

    if weight.dim() >= 3:
        kernel_nd = weight.dim() - 2
        flattened, shape_meta = flatten_conv_weight(weight, kernel_nd)
    elif weight.dim() == 1:
        flattened = weight.unsqueeze(0)
        squeezed_row = True
    else:
        flattened = weight

    rotated = flattened
    if R_in is not None:
        _validate_rotation_matrix(R_in, rotated.shape[1], "R_in")
        rotated = rotated @ R_in
    if R_out is not None:
        _validate_rotation_matrix(R_out, rotated.shape[0], "R_out")
        rotated = R_out.transpose(-1, -2) @ rotated

    if shape_meta is not None:
        rotated = restore_conv_weight(rotated, shape_meta)
    elif squeezed_row:
        rotated = rotated.squeeze(0)
    return rotated


def bake_rotation_linear(
    m: nn.Linear, R_in: Optional[torch.Tensor], R_out: Optional[torch.Tensor]
) -> None:
    if not isinstance(m, nn.Linear):
        raise TypeError("bake_rotation_linear expects an nn.Linear module.")
    if R_in is None and R_out is None:
        return

    snap = snapshot_weights((m,))
    try:
        rotated = apply_rotation(m.weight.detach(), R_in, R_out)
        with torch.no_grad():
            m.weight.copy_(rotated)
    except Exception:
        restore_weights(m, snap)
        raise


ConvNd = Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]


def bake_rotation_convnd(
    m: ConvNd, R_in: Optional[torch.Tensor], R_out: Optional[torch.Tensor]
) -> None:
    if not isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        raise TypeError("bake_rotation_convnd expects a Conv1d/2d/3d module.")
    if R_in is None and R_out is None:
        return

    snap = snapshot_weights((m,))
    try:
        rotated = apply_rotation(m.weight.detach(), R_in, R_out)
        with torch.no_grad():
            m.weight.copy_(rotated)
    except Exception:
        restore_weights(m, snap)
        raise


def eligible_module(m: nn.Module) -> bool:
    """Return True when the module supports weight rotations (Linear/Conv only)."""
    return isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))


@lru_cache(None)
def _quant_conv_types() -> Tuple[Type[nn.Module], ...]:
    from cobra.quantize.int_conv import QuantConv1d, QuantConv2d, QuantConv3d

    return QuantConv1d, QuantConv2d, QuantConv3d


@lru_cache(None)
def _quant_linear_type() -> Type[nn.Module]:
    from cobra.quantize.int_linear import QuantLinear

    return QuantLinear


@lru_cache(None)
def _quant_matmul_type() -> Type[nn.Module]:
    from cobra.quantize.int_matmul import QuantMatMul

    return QuantMatMul


def is_quant_eligible(m: nn.Module) -> bool:
    """Return True if module is a quantized Linear/Conv/MatMul wrapper."""
    quant_types = _quant_conv_types() + (_quant_linear_type(), _quant_matmul_type())
    return isinstance(m, quant_types)


def snapshot_weights(modules: Sequence[nn.Module]) -> Dict[str, torch.Tensor]:
    """
    Capture detached weight clones for a set of modules to support rollback.
    """
    snap: Dict[str, torch.Tensor] = {}
    for module in modules:
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            snap[str(id(module))] = weight.detach().clone()
    return snap


def restore_weights(model: nn.Module, snapshot: Dict[str, torch.Tensor]) -> None:
    """
    Restore weights from a snapshot created via snapshot_weights.
    """
    if not snapshot:
        return
    for module in model.modules():
        key = str(id(module))
        if key not in snapshot:
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            with torch.no_grad():
                weight.copy_(snapshot[key].to(weight.device, weight.dtype))


def estimate_and_apply_rotation(*args, **kwargs):
    """
    Lazy import wrapper so callers can trigger the rotation controller via cobra.quantize.utils.
    """
    from cobra.quantize.pct.calibrator import estimate_and_apply_rotation as _rotation_impl

    return _rotation_impl(*args, **kwargs)


def qualname(model: nn.Module, module: nn.Module) -> str:
    """
    Resolve the dotted path of `module` within `model`.
    """
    for name, mod in model.named_modules():
        if mod is module:
            return name or module.__class__.__name__
    raise ValueError("Module not found in the provided model.")


def finalize_model(*args, **kwargs):
    """
    Convenience proxy so callers can import finalize_model from quantize.utils.
    """
    from cobra.quantize.pct.calibrator import finalize_model as _impl

    return _impl(*args, **kwargs)


def safer_unwrap_weight(module: nn.Module) -> Optional[torch.Tensor]:
    """
    Return a cloned FP32 weight if the module has not been packed; otherwise None.
    """
    if hasattr(module, "w_qint") and getattr(module, "w_qint") is not None:
        return None
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight.detach().clone()
    return None


def _move_sample_to_device(sample: Any, device: torch.device) -> Any:
    if torch.is_tensor(sample):
        return sample.to(device)
    if isinstance(sample, dict):
        return {k: _move_sample_to_device(v, device) for k, v in sample.items()}
    if isinstance(sample, (list, tuple)):
        arr = [_move_sample_to_device(v, device) for v in sample]
        return type(sample)(arr)
    return sample


def _tensorize_output(output: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(output):
        return output.detach().float()
    if isinstance(output, (list, tuple)):
        tensors = [_tensorize_output(o) for o in output]
        tensors = [t for t in tensors if t is not None]
        if not tensors:
            return None
        return torch.cat([t.reshape(-1) for t in tensors], dim=0)
    if isinstance(output, Mapping):
        tensors = [_tensorize_output(v) for v in output.values()]
        tensors = [t for t in tensors if t is not None]
        if not tensors:
            return None
        return torch.cat([t.reshape(-1) for t in tensors], dim=0)
    return None


def quick_verify(
    model: nn.Module,
    dataloader: Optional[Any],
    batches: int,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    """
    Compare finalized model outputs against a float reference copy for a few batches.
    """
    if dataloader is None or batches <= 0:
        return {"status": "skipped", "reason": "no_dataloader"}

    device = next(model.parameters(), torch.empty(0)).device
    model.eval()
    reference = copy.deepcopy(model)
    reference.eval()

    for mod in reference.modules():
        if hasattr(mod, "finalized") and getattr(mod, "finalized"):
            setattr(mod, "finalized", False)
            if hasattr(mod, "use_weight_quant"):
                mod.use_weight_quant = False
            if hasattr(mod, "use_act_quant"):
                mod.use_act_quant = False
            cache = getattr(mod, "weight_fp32_cache", None)
            if isinstance(cache, torch.Tensor):
                with torch.no_grad():
                    mod.weight.copy_(cache.to(mod.weight.device, mod.weight.dtype))

    quant_buffers: Dict[str, torch.Tensor] = {}
    ref_buffers: Dict[str, torch.Tensor] = {}

    hooks_quant = []
    hooks_ref = []

    for name, module in model.named_modules():
        if not is_quant_eligible(module):
            continue

        def _hook_factory(buf: Dict[str, torch.Tensor], key: str):
            def _hook(_module, _inputs, output):
                tensor = _tensorize_output(output)
                if tensor is not None:
                    buf[key] = tensor
            return _hook

        if name in dict(reference.named_modules()):
            hooks_quant.append(module.register_forward_hook(_hook_factory(quant_buffers, name)))
            hooks_ref.append(reference.get_submodule(name).register_forward_hook(_hook_factory(ref_buffers, name)))

    stats: Dict[str, Any] = {
        "status": "ok",
        "batches": 0,
        "max_mse": 0.0,
        "max_mae": 0.0,
        "max_rel": 0.0,
        "violations": [],
    }

    try:
        with torch.no_grad():
            for idx, sample in enumerate(dataloader):
                if idx >= batches:
                    break
                stats["batches"] += 1
                moved = _move_sample_to_device(sample, device)
                quant_buffers.clear()
                ref_buffers.clear()
                quant_out = model(moved)
                ref_out = reference(moved)
                quant_tensor = _tensorize_output(quant_out)
                ref_tensor = _tensorize_output(ref_out)
                if quant_tensor is None or ref_tensor is None:
                    continue
                diff = quant_tensor - ref_tensor.to(quant_tensor.device)
                mse = torch.mean(diff.pow(2)).item()
                mae = torch.mean(diff.abs()).item()
                denom = torch.mean(ref_tensor.abs()).item() + 1e-6
                rel = (mae / denom) if denom > 0 else mae
                stats["max_mse"] = max(stats["max_mse"], mse)
                stats["max_mae"] = max(stats["max_mae"], mae)
                stats["max_rel"] = max(stats["max_rel"], rel)
                if mae > atol or rel > rtol:
                    layer_errors = []
                    for name, q_out in quant_buffers.items():
                        ref_out_layer = ref_buffers.get(name)
                        if ref_out_layer is None:
                            continue
                        layer_diff = q_out - ref_out_layer.to(q_out.device)
                        layer_mae = torch.mean(layer_diff.abs()).item()
                        layer_errors.append((layer_mae, name))
                    layer_errors.sort(reverse=True)
                    stats["violations"] = layer_errors[:10]
                    stats["status"] = "fail"
                    break
    finally:
        for h in hooks_quant:
            h.remove()
        for h in hooks_ref:
            h.remove()

    return stats


__all__ = [
    "RotationSpec",
    "FinalizeSpec",
    "apply_rotation",
    "bake_rotation_linear",
    "bake_rotation_convnd",
    "eligible_module",
    "snapshot_weights",
    "restore_weights",
    "estimate_and_apply_rotation",
    "qualname",
    "is_quant_eligible",
    "finalize_model",
]



def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    quant_types = _quant_conv_types() + (_quant_linear_type(), _quant_matmul_type())
    for name,m in self.named_modules():
        if isinstance(m, quant_types):
            m.set_quant_state(weight_quant, act_quant)

def set_static_quant(self, static_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    for m in self.modules():
        if isinstance(m, UniformAffineQuantizer):
            m.is_dynamic_quant = not static_quant

def set_static_quant_weight(self, static_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    for name, m in self.named_modules():
        if "weight" in name:
            if isinstance(m, UniformAffineQuantizer):
                m.is_dynamic_quant = not static_quant

def set_observing(self, observing: bool = True):
    self.use_observing = observing
    for name, m in self.named_modules():
        if isinstance(m, (UniformAffineQuantizer)):
           m.is_observing = observing

