from re import U
import json
import logging
import math
import os
import pdb
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from .observers.hist_observers import KLObserver, MSEObserver, PercentileObserver
from .observers.minmax_observers import MinMaxObserver

LOGGER = logging.getLogger(__name__)

CLIPMIN = 1e-5

class ClampSte(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,min_,max_):
        return x.clamp(min_,max_)
    
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output.clone(),None,None

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=False,
        rescale=False,
        rescale_limit=False,
        has_batch_dim = False,
        is_weight=False,
        observe="minmax",
        percent = 0.999999,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        if self.disable_zero_point or self.symmetric:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        self.rescale = rescale # for channel-rescale
        self.rescale_limit = rescale_limit
        self._clip_override: Optional[Tuple[float, float]] = None
        self._pct_bucket: Optional[str] = None
        self._external_percentile: Optional[float] = None

        init_value = 4.0  # inti value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(shape[0] * math.ceil(shape[1] / group_size))
                self.deficiency = shape[-1] % group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric  # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
        
        if rescale:
            if rescale_limit:
                self.rescale_param = nn.Parameter(torch.zeros(dim1,1) )
            else:
                self.rescale_param = nn.Parameter(torch.ones(dim1,1) )

        self.sigmoid = nn.Sigmoid()

        self.enable = True #Percentile Clipping 開關
        self.group_size = group_size
        
        self.has_batch_dim = has_batch_dim
        self.is_observing = False
        self.is_dynamic_quant = True
        granularity = 'dim{}'.format(per_channel_axes[0]) if len(per_channel_axes) > 0 else 'tensor'
        
        if observe == "percentile":
            self.observer = PercentileObserver(percent=0.999999,granularity=granularity)
        else:
            self.observer = MinMaxObserver(granularity=granularity)
 
        self.observered = False
        
        self.is_weight = is_weight

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1

    def set_clip(
        self,
        lo: float,
        hi: float,
        *,
        symmetric: Optional[bool] = None,
        bucket: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Inject externally computed clipping bounds and freeze observer stats.
        """
        if lo is None or hi is None:
            raise ValueError("lo/hi must be provided for clip overrides")
        lo_val = float(lo)
        hi_val = float(hi)
        use_symmetric = self.symmetric if symmetric is None else symmetric
        if use_symmetric:
            magnitude = max(abs(lo_val), abs(hi_val))
            lo_val, hi_val = -magnitude, magnitude

        if self.observer is None:
            self.observer = MinMaxObserver(granularity="tensor")
        if hasattr(self.observer, "set_manual_range"):
            self.observer.set_manual_range(lo_val, hi_val)

        self._clip_override = (lo_val, hi_val)
        self._pct_bucket = bucket
        self._external_percentile = None
        self.is_dynamic_quant = False
        self.is_observing = False
        self.observered = False
        return self._clip_override

    def set_percentile_clip(
        self,
        lo: float,
        hi: float,
        *,
        symmetric: bool = True,
        bucket: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Backwards-compatible alias for set_clip.
        """
        return self.set_clip(lo, hi, symmetric=symmetric, bucket=bucket)

    def set_percentile(self, percentile: float) -> None:
        """
        Record a desired percentile so upstream observers can honor it later.
        """
        self._external_percentile = float(percentile)

    def recompute_params_from_clip(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Recompute scale/zero-point from the currently stored manual clip range.
        """
        if self._clip_override is None:
            raise RuntimeError("No clip override available to recompute params")
        xmin = torch.tensor([self._clip_override[0]], dtype=torch.float32)
        xmax = torch.tensor([self._clip_override[1]], dtype=torch.float32)
        if self.symmetric or self.disable_zero_point:
            self.symmetric_cal_scale(xmin, xmax)
        else:
            self.assymmetric_cal_scale(xmin, xmax)
        return self.scale, self.round_zero_point

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros(
                (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)

        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, : -self.deficiency]

        if self.rescale:
            rescale_param = self.rescale_param
            if self.rescale_limit:
                rescale_param = 0.5 + F.sigmoid(rescale_param)
            if len(rescale_param.shape) == 2 and len(x_dequant.shape)==3:
                rescale_param = rescale_param.unsqueeze(-1)
            x_dequant = x_dequant*rescale_param.to(x_dequant.device)
        return x_dequant

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)
        
        if self.is_weight:#权重量化，没有observe过程
            if True:#not self.is_dynamic_quant:
                if  self.is_observing:
                    return x
                if self.observer is not None:
                    self.observer.update(x)
                    xmin,xmax = self.observer.cal_min_max()
                    self.assymmetric_cal_scale(xmin,xmax)
                    self.scale = self.expand_scale_shape_2_x(x, self.scale)
                    self.round_zero_point = self.expand_scale_shape_2_x(x, self.round_zero_point)
                    self.observer = None
                x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
                return x_dequant.type_as(x)
            # else:
            #     if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            #         self.per_token_dynamic_calibration(x)
            #     else:
            #         self.dynamic_per_tensor_calibration(x)
            #     x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
            #     return x_dequant
        else:#激活量化
            if not self.is_dynamic_quant:
                if self.is_observing:
                    self.observer.update(x)
                    return x.type_as(x)
                else:
                    if not self.observered:
                        xmin,xmax = self.observer.cal_min_max()
                        self.assymmetric_cal_scale(xmin,xmax)
                        self.scale = self.expand_scale_shape_2_x(x, self.scale)
                        self.round_zero_point = self.expand_scale_shape_2_x(x, self.round_zero_point)
                        self.observered = True
                        self.observer = None
                    x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
                    return x_dequant.type_as(x)
                    
            else:
                if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
                    self.per_token_dynamic_calibration(x)
                else:
                    self.dynamic_per_tensor_calibration(x)

                x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
                return x_dequant.type_as(x)

    def expand_scale_shape_2_x(self, x, scale):
        if self.per_channel_axes:
            dim=self.per_channel_axes[0]
            for i in range(len(x.shape)):
                if i != dim:
                    scale = scale.unsqueeze(i)
        return scale

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1, self.group_size)
            else:
                pad_zeros = torch.zeros(
                    (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
                )
                x = torch.cat((x, pad_zeros), dim=1)
                x = x.reshape(-1, self.group_size)
        if self.dynamic_method == "per_channel":
            if len(self.per_channel_axes):
                assert len(self.per_channel_axes) == 1,"must be one"
                reduce_shape = list(range(x.dim()))
                reduce_shape.remove(self.per_channel_axes[0])
            else:
                reduce_shape = list(range(x.dim()-1))
        else:
            reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax = x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor) * xmax
            xmin = self.sigmoid(self.lowbound_factor) * xmin
        self.xmin_tmp = xmin.detach()
        self.xmax_tmp = xmax.detach()
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (self.n_bits - 1) - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(self.scale)
        else:
            dynamic_range = xmax - xmin
            scale = dynamic_range / (2**self.n_bits - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
    
    def MaxMin_except_first_dim(self,tensor,func):
        # 获取张量的维度数
        dims = list(range(1, tensor.dim()))
        # 逐步在每个维度上取最大值
        for dim in dims:
            tensor, _ = func(tensor, dim=dim, keepdim=True)
        return tensor
    
    def dynamic_per_tensor_calibration(self,x):
        if not self.has_batch_dim:
            xmin = x.min()
            xmax = x.max()
        else:
            shape = [1] * len(x.shape)
            shape[0] = -1
            xmin = self.MaxMin_except_first_dim(x,torch.min).view(shape)
            xmax = self.MaxMin_except_first_dim(x,torch.max).view(shape)
        if self.symmetric or self.disable_zero_point:
            self.symmetric_cal_scale(xmin,xmax)
        else:
            self.assymmetric_cal_scale(xmin,xmax)

    def symmetric_cal_scale(self,xmin,xmax):
        abs_max = torch.max(xmax.abs(), xmin.abs())
        scale = abs_max / (2 ** (self.n_bits - 1) - 1)
        self.scale = scale.clamp(min=CLIPMIN, max=1e4)
        self.round_zero_point = None
        
    def assymmetric_cal_scale(self,xmin,xmax):
        dynamic_range = xmax - xmin
        scale = dynamic_range / (2**self.n_bits - 1)
        self.scale = scale.clamp(min=CLIPMIN, max=1e4)
        zero_point = -(xmin) / (self.scale)
        self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
    
    def normal_quantize(self, x, scales: torch.Tensor, mig_cof: torch.Tensor):
        s = (scales / mig_cof).max()
        s = s / (2**self.n_bits - 1)
        self.scale = s
        # only support symmetric quantization
        self.round_zero_point = None
        
    def scale_frexp(self):
        k = 16
        m = (self.scale*(2**k)).round()
        self.scale = m*(2**(-k))
        
        return self.scale

    def register_scales_and_zeros(self):
        self.register_buffer("scales", self.scale)
        self.register_buffer("zeros", self.round_zero_point)
        del self.scale
        del self.round_zero_point
        
    def quant2int(self, x):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)
        if self.deficiency > 0:
            pad_zeros = torch.zeros(
                (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / self.scale)
        if self.round_zero_point is not None:
            x_int = x_int.add(self.round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        
        if self.group_size:
            x_int = x_int.reshape(dim1, dim2)
        return x_int
    
    def dequant(self, x_int):
        if self.group_size:
            assert len(x_int.shape) == 2, "only support linear layer now"
            dim1, dim2 = x_int.shape
            x_int = x_int.reshape(-1, self.group_size)
            
        x_dequant = x_int
        if self.round_zero_point is not None:
            x_dequant = x_dequant.sub(self.round_zero_point)
        x_dequant = x_dequant.mul(self.scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, : -self.deficiency]

        if self.rescale:
            rescale_param = self.rescale_param
            if self.rescale_limit:
                rescale_param = F.sigmoid(rescale_param) + 0.5
            x_dequant = x_dequant*self.rescale_param
        return x_dequant



class ActQuantizer(nn.Module):
    def __init__(self):
        self.register_parameter("scale",torch.ones(1))
        self.register_buffer("calibed_enabled",torch.tensor([0],dtype=torch.uint8))
    
    # @property
    # def calib
    
    def forward(self,x):
        pass


if __name__ == "__main__":
    cfg = {"dynamic_method":"per_tensor","n_bits":8,"symmetric":True}
    weight = torch.randn(100,100)
    quantizer = UniformAffineQuantizer(**cfg)
    weight_quant = quantizer.forward(weight)
    diff = weight-weight_quant
    print(diff.sum())


def load_pct_override_table(path: str) -> Dict[str, Tuple[float, float]]:
    """
    Load percentile clip overrides JSON and return mapping bucket -> (lo, hi).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"override file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    table: Dict[str, Tuple[float, float]] = {}
    for bucket, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        if "lo" not in payload or "hi" not in payload:
            continue
        lo = float(payload["lo"])
        hi = float(payload["hi"])
        table[bucket] = (lo, hi)
    return table


def _iter_quantizers(
    quantizers: Union["UniformAffineQuantizer", Sequence["UniformAffineQuantizer"]],
) -> Iterable["UniformAffineQuantizer"]:
    if isinstance(quantizers, UniformAffineQuantizer):
        yield quantizers
        return
    if isinstance(quantizers, Sequence) and not isinstance(quantizers, (str, bytes)):
        for item in quantizers:
            if not isinstance(item, UniformAffineQuantizer):
                raise TypeError("quantizer collection contains non-quantizer objects")
            yield item
        return
    raise TypeError("quantizers must be UniformAffineQuantizer or a sequence of them")


def apply_pct_overrides_to_quantizers(
    bucket_quantizers: Mapping[str, Union["UniformAffineQuantizer", Sequence["UniformAffineQuantizer"]]],
    overrides_json: str,
    *,
    symmetric: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Given a mapping from bucket name to activation quantizer(s), load overrides
    JSON and push the (lo, hi) clip bounds into each quantizer.
    """
    if not bucket_quantizers:
        return {}
    override_table = load_pct_override_table(overrides_json)
    log = logger or LOGGER
    applied: Dict[str, Tuple[float, float]] = {}
    for bucket, quantizers in bucket_quantizers.items():
        clip = override_table.get(bucket)
        if clip is None:
            log.warning("pct overrides missing bucket '%s' in %s", bucket, overrides_json)
            continue
        for quantizer in _iter_quantizers(quantizers):
            quantizer.set_percentile_clip(clip[0], clip[1], symmetric=symmetric, bucket=bucket)
        applied[bucket] = clip
    unused = set(override_table.keys()) - set(bucket_quantizers.keys())
    if unused:
        log.warning("pct overrides unused buckets: %s", ", ".join(sorted(unused)))
    return applied


def _resolve_min_max_from_quantizer(q: "UniformAffineQuantizer") -> Tuple[torch.Tensor, torch.Tensor]:
    if getattr(q, "_clip_override", None) is not None:
        lo, hi = q._clip_override
        return torch.as_tensor(lo, dtype=torch.float32), torch.as_tensor(hi, dtype=torch.float32)
    observer = getattr(q, "observer", None)
    if observer is not None and observer.min_val.numel() > 0 and observer.max_val.numel() > 0:
        return observer.min_val.detach().clone(), observer.max_val.detach().clone()
    raise ValueError("finalize_quant_params: quantizer lacks clip range statistics.")


def finalize_quant_params(q: "UniformAffineQuantizer", symmetric: bool, per_channel: bool) -> None:
    """
    Ensure quantizer has frozen scale/zero based on collected stats or injected overrides.
    """
    if not isinstance(q, UniformAffineQuantizer):
        raise TypeError("finalize_quant_params expects a UniformAffineQuantizer instance.")

    if getattr(q, "frozen", False):
        return

    scale_tensor = q.scale if isinstance(q.scale, torch.Tensor) else None
    zero_tensor = q.round_zero_point if isinstance(q.round_zero_point, torch.Tensor) else None

    if scale_tensor is None or scale_tensor.numel() == 0:
        lo, hi = _resolve_min_max_from_quantizer(q)
        lo = lo.to(dtype=torch.float32)
        hi = hi.to(dtype=torch.float32)
        if not per_channel:
            lo = lo.min().view(1)
            hi = hi.max().view(1)
        absmax = torch.max(lo.abs(), hi.abs())
        if symmetric or getattr(q, "disable_zero_point", False):
            denom = float((2 ** (q.n_bits - 1)) - 1)
            scale_tensor = (absmax / max(denom, 1.0)).clamp(min=CLIPMIN)
            zero_tensor = None
        else:
            denom = float(q.qmax - q.qmin)
            dynamic = hi - lo
            scale_tensor = (dynamic / max(denom, 1.0)).clamp(min=CLIPMIN)
            zero_tensor = torch.round(q.qmin - lo / scale_tensor).clamp(q.qmin, q.qmax)
    else:
        scale_tensor = scale_tensor.detach().clone().to(dtype=torch.float32)
        if zero_tensor is not None:
            zero_tensor = zero_tensor.detach().clone().to(dtype=torch.float32)

    if torch.any(~torch.isfinite(scale_tensor)) or torch.any(scale_tensor <= 0):
        raise ValueError("finalize_quant_params: invalid scale detected.")
    expected_channels = getattr(q, "_expected_channels", None)
    if per_channel and expected_channels is not None and scale_tensor.numel() not in (1, expected_channels):
        raise ValueError(
            f"finalize_quant_params: per-channel scale mismatch ({scale_tensor.numel()} vs expected {expected_channels}). "
            "Please re-run calibration or disable per-channel."
        )

    q.scale = scale_tensor
    q.round_zero_point = None if symmetric or getattr(q, "disable_zero_point", False) else zero_tensor
    q.is_dynamic_quant = False
    q.is_observing = False
    q.observer = None
    q.observered = True
    q.frozen = True


def _coerce_number(val: Any) -> Optional[float]:
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            val = val.item()
        else:
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def apply_external_clip(qobj: Any, payload: Mapping[str, Any]) -> bool:
    """
    Apply percentile overrides to a quantizer instance.
    """
    if qobj is None or not isinstance(payload, Mapping):
        return False
    lo = _coerce_number(payload.get("lo"))
    hi = _coerce_number(payload.get("hi"))
    if lo is not None and hi is not None:
        if hasattr(qobj, "set_clip") and hasattr(qobj, "recompute_params_from_clip"):
            qobj.set_clip(lo, hi)
            qobj.recompute_params_from_clip()
            return True
        return False

    percentile = _coerce_number(payload.get("percentile"))
    if percentile is not None and hasattr(qobj, "set_percentile"):
        qobj.set_percentile(percentile)
        return True
    return False


def attach_overrides_if_any(
    module: Any,
    overrides: Optional[Mapping[str, Mapping[str, Any]]],
    *,
    prefix: Optional[str] = None,
    roles: Tuple[str, ...] = ("weight_quantizer", "act_quantizer"),
) -> int:
    """
    Iterate through common quantizer roles on a module and apply overrides
    when matching keys exist.
    """
    if not overrides:
        return 0
    key_prefix = prefix or getattr(module, "_qualified_name", None) or getattr(module, "qualified_name", None)
    if not key_prefix:
        return 0

    applied = 0
    for role in roles:
        quant_ref = getattr(module, role, None)
        if quant_ref is None:
            continue
        if isinstance(quant_ref, (list, tuple)):
            iterator = enumerate(quant_ref)
        else:
            iterator = [(0, quant_ref)]
        for idx, quantizer in iterator:
            if quantizer is None:
                continue
            key = f"{key_prefix}.{role}.{idx}"
            payload = overrides.get(key)
            if payload and apply_external_clip(quantizer, payload):
                applied += 1
    return applied


def _apply_calib_to_quantizer(quantizer: Any, params: Mapping[str, Any]) -> bool:
    scale = params.get("scale")
    zero = params.get("zero")
    if zero is None:
        zero = params.get("zero_point")
    if scale is None and zero is None:
        return False
    if hasattr(quantizer, "set_calibrated_params"):
        try:
            quantizer.set_calibrated_params(scale, zero)
            return True
        except Exception:
            pass
    success = False
    if scale is not None and hasattr(quantizer, "scale"):
        try:
            tensor = torch.as_tensor(scale, dtype=getattr(quantizer.scale, "dtype", torch.float32))
            device = getattr(quantizer.scale, "device", torch.device("cpu"))
            quantizer.scale = tensor.to(device=device)
            success = True
        except Exception:
            pass
    if (
        zero is not None
        and not getattr(quantizer, "disable_zero_point", False)
        and hasattr(quantizer, "round_zero_point")
    ):
        try:
            tensor = torch.as_tensor(zero, dtype=torch.int32)
            device = getattr(quantizer.round_zero_point, "device", torch.device("cpu"))
            quantizer.round_zero_point = tensor.to(device=device)
            success = True
        except Exception:
            pass
    return success


def apply_calibration_table(
    model: nn.Module,
    table: Mapping[str, Mapping[str, Any]],
) -> int:
    """
    Apply calibration parameters to quantized wrappers.
    The table keys follow `<module_path>.<role>.<index>` convention.
    """
    if not isinstance(table, Mapping):
        return 0
    module_map = {name: module for name, module in model.named_modules()}
    applied = 0
    for key, entry in table.items():
        if not isinstance(entry, Mapping):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        role = parts[-2]
        module_path = ".".join(parts[:-2])
        module = module_map.get(module_path)
        if module is None:
            continue
        quantizer = getattr(module, role, None)
        if quantizer is None:
            continue
        if _apply_calib_to_quantizer(quantizer, entry):
            applied += 1
    return applied
