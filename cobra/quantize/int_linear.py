from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .percentile_io import load_overrides
from .quantizer import UniformAffineQuantizer, attach_overrides_if_any, finalize_quant_params
from .utils import FinalizeSpec
from .int_others import pack_4bit_tensor, unpack_4bit_tensor


def _clamp_pct_activation(x: torch.Tensor, quantizer: Optional[UniformAffineQuantizer]) -> torch.Tensor:
    """
    Clamp activation tensor to precomputed percentile range before quantization.
    Prefers explicit per-tensor overrides (tensor._pct_clip_range), otherwise
    falls back to the quantizer's configured override when buckets match.
    """
    if quantizer is None or not isinstance(x, torch.Tensor):
        return x

    tensor_bucket = getattr(x, "_pct_bucket", None)
    tensor_clip = getattr(x, "_pct_clip_range", None)

    q_clip = getattr(quantizer, "_clip_override", None)
    q_bucket = getattr(quantizer, "_pct_bucket", None)

    clip: Optional[Tuple[float, float]] = None
    if isinstance(tensor_clip, (tuple, list)) and len(tensor_clip) == 2:
        clip = (float(tensor_clip[0]), float(tensor_clip[1]))
    elif isinstance(tensor_clip, dict) and "lo" in tensor_clip and "hi" in tensor_clip:
        clip = (float(tensor_clip["lo"]), float(tensor_clip["hi"]))
    elif q_clip is not None:
        if q_bucket is None or tensor_bucket is None or tensor_bucket == q_bucket:
            clip = (float(q_clip[0]), float(q_clip[1]))
            tensor_bucket = tensor_bucket or q_bucket

    if clip is None or clip[0] is None or clip[1] is None:
        return x

    clamped = x.clamp(clip[0], clip[1])
    if tensor_bucket is not None:
        setattr(clamped, "_pct_bucket", tensor_bucket)
    setattr(clamped, "_pct_clip_range", clip)
    return clamped






def _normalize_overrides_input(
    overrides: Optional[Union[str, Dict[str, Dict[str, float]]]]
) -> Optional[Dict[str, Dict[str, float]]]:
    if overrides is None:
        return None
    if isinstance(overrides, str):
        return load_overrides(overrides)
    return overrides


class QuantLinear(nn.Linear):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {"dynamic_method":"per_tensor"},
        act_quant_params: dict = {"dynamic_method":"per_tensor"},
        disable_input_quant=False,
        observe = "minmax",
        *,
        overrides: Optional[Union[str, Dict[str, Dict[str, float]]]] = None,
        qualified_name: Optional[str] = None,
    ):
        super().__init__(org_module.in_features,org_module.out_features)
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.weight=org_module.weight
        if org_module.bias is not None:
            self.bias=org_module.bias
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self._expected_weight_shape = tuple(org_module.weight.shape)
        self.weight_quantizer._expected_channels = self.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape,is_weight=True,observe=observe)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params,has_batch_dim=True,observe=observe)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        
        self.weight_quantized = False
        self.finalized = False
        self.pack_meta: Dict[str, Any] = {}
        self.w_qint: Optional[torch.Tensor] = None
        self._expected_weight_shape = tuple(org_module.weight.shape)
        self.weight_fp32_cache: Optional[torch.Tensor] = None
        self._qualified_name = (
            qualified_name
            or getattr(org_module, "_qualified_name", None)
            or getattr(org_module, "qualified_name", None)
            or f"{org_module.__class__.__name__}"
        )

        overrides_dict = _normalize_overrides_input(overrides)
        if overrides_dict:
            attach_overrides_if_any(self, overrides_dict)

    
    
    def forward(self, input: torch.Tensor):
        if input.shape[-1] != self.in_features:
            name = getattr(self, "_qualified_name", self.__class__.__name__)
            raise RuntimeError(
                f"QuantLinear[{name}] expected last-dim {self.in_features}, got {input.shape[-1]} "
                "after rotation/injection. Please ensure rotations preserve feature size."
            )
        if tuple(self.weight.shape) != self._expected_weight_shape:
            name = getattr(self, "_qualified_name", self.__class__.__name__)
            raise RuntimeError(
                f"QuantLinear[{name}] weight shape changed from {self._expected_weight_shape} "
                f"to {tuple(self.weight.shape)}; cannot quantize safely."
            )

        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            if self.weight_quantizer.is_observing:
                weight = self.weight
            elif not self.weight_quantized:
                self.weight = torch.nn.Parameter(self.weight_quantizer(self.weight))
                weight = self.weight
                self.weight_quantized = True
            else:
                weight = self.weight
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = _clamp_pct_activation(input, self.act_quantizer)
            input = self.act_quantizer(input)
        
        if bias is not None:bias = bias.to(weight)
        out = self.fwd_func(
                input.to(weight), weight, bias, **self.fwd_kwargs)


        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def pack_weight(self) -> None:
        if self.weight_quantizer.scale is None or self.weight_quantizer.scale.numel() == 0:
            raise RuntimeError("pack_weight requires finalized weight quantizer.")
        q = self.weight_quantizer
        if q.n_bits not in (4, 8):
            return
        if q.scale.numel() not in (1, self.weight.shape[0]):
            raise ValueError("Weight quantizer scale shape mismatch for packing.")
        qint = q.quant2int(self.weight.detach()).round()
        if torch.any(~torch.isfinite(qint)):
            raise ValueError("pack_weight encountered NaN/Inf in quantized weights.")
        flat = qint.view(-1)
        if q.n_bits == 8:
            dtype = torch.int8 if q.qmin < 0 else torch.uint8
            data = flat.to(dtype)
            zero_offset = 0
        else:  # 4-bit
            zero_offset = -q.qmin if q.qmin < 0 else 0
            values = (flat + zero_offset).to(torch.int16)
            if torch.any(values < 0) or torch.any(values > 15):
                raise ValueError("4-bit weights exceed nibble range after offsetting.")
            data = pack_4bit_tensor(values)
        self.w_qint = data.contiguous()
        self.pack_meta = {
            "n_bits": q.n_bits,
            "weight_shape": tuple(self.weight.shape),
            "packed_dtype": str(self.w_qint.dtype),
            "zero_offset": zero_offset,
        }

    def finalize(self, spec: FinalizeSpec) -> None:
        if self.finalized:
            return
        finalize_quant_params(self.weight_quantizer, spec.symmetric, spec.per_channel)
        if self.act_quantizer is not None:
            finalize_quant_params(self.act_quantizer, spec.symmetric, False)
        if spec.pack_linear:
            self.weight_fp32_cache = self.weight.detach().clone()
            self.pack_weight()
        self.finalized = True
