from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer, finalize_quant_params
from .utils import FinalizeSpec
from .int_others import pack_4bit_tensor, unpack_4bit_tensor


class QuantMatMul(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict = {"dynamic_method":"per_tensor"},
        x2_quant_params: dict = {"dynamic_method":"per_tensor"},
        disable_act_quant=False,
        observe = "minmax",
        matmul_func=torch.matmul,
    ):
        super().__init__()
        # de-activate the quantized forward default
        self.use_act_quant = False
        # initialize quantizer
        self.i_cluster_counts = None
        self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params,has_batch_dim=True,observe=observe)
        self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params,has_batch_dim=True,observe=observe)
        self.matmul_func = matmul_func

        self.disable_act_quant = disable_act_quant
        self.finalized = False
        self.w_qint: Optional[torch.Tensor] = None
        self.pack_meta: Dict[str, Any] = {}
        self.weight_fp32_cache: Optional[torch.Tensor] = None


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def quant_x1(self, x1):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
        return x1

    def quant_x2(self, x2):
        if self.use_act_quant:
            x2 = self.x2_quantizer(x2)
        return x2

    def forward(self, x1, x2):
        if getattr(self, "finalized", False) and self.w_qint is not None:
            return self._int_forward(x1)
        if hasattr(self,"pertoken"):
            B,L,ED,N = x1.shape
            x1 = x1.reshape(B,L*ED,N)
            x1 = self.quant_x1(x1)
            x1 = x1.reshape(B,L,ED,N)
            x2 = self.quant_x2(x2)
            out = self.matmul_func(x1, x2)
            pass
        else:
            x1 = self.quant_x1(x1)
            x2 = self.quant_x2(x2)
            out = self.matmul_func(x1, x2)
        return out

    def _int_forward(self, x1: torch.Tensor) -> torch.Tensor:
        weight_int = self._unpack_weight()
        if weight_int is None:
            raise RuntimeError("QuantMatMul finalized without packed weight.")
        if self.x1_quantizer.scale.numel() != 1 or self.x2_quantizer.scale.numel() != 1:
            raise NotImplementedError("Per-channel matmul finalization is not supported yet.")
        x1_q = self.x1_quantizer.quant2int(x1)
        x1_int = x1_q.to(torch.int32)
        zp1 = getattr(self.x1_quantizer, "round_zero_point", None)
        if zp1 is not None:
            x1_int = x1_int - zp1.to(x1_int.dtype)
        w_int = weight_int.to(torch.int32)
        zp2 = getattr(self.x2_quantizer, "round_zero_point", None)
        if zp2 is not None:
            w_int = w_int - zp2.to(w_int.dtype)
        out_int = self.matmul_func(x1_int, w_int)
        scale = (self.x1_quantizer.scale.view(1) * self.x2_quantizer.scale.view(1)).to(out_int.device, torch.float32)
        return out_int.to(torch.float32) * scale

    def _unpack_weight(self) -> Optional[torch.Tensor]:
        if self.w_qint is None:
            return None
        n_bits = self.pack_meta.get("n_bits", 8)
        shape = self.pack_meta.get("weight_shape")
        if shape is None:
            return None
        if n_bits == 8:
            dtype = torch.int8 if self.x2_quantizer.qmin < 0 else torch.uint8
            data = self.w_qint.to(dtype).view(shape)
        else:
            total = 1
            for dim in shape:
                total *= dim
            offset = int(self.pack_meta.get("zero_offset", 0))
            unpacked = unpack_4bit_tensor(self.w_qint, total).to(torch.int32)
            data = (unpacked - offset).view(shape)
        return data

    def finalize(self, spec: FinalizeSpec, weight: Optional[torch.Tensor] = None) -> None:
        if self.finalized:
            return
        finalize_quant_params(self.x1_quantizer, spec.symmetric, False)
        finalize_quant_params(self.x2_quantizer, spec.symmetric, False)
        if weight is not None:
            self._pack_weight(weight)
        self.finalized = True

    def _pack_weight(self, weight: torch.Tensor) -> None:
        q = self.x2_quantizer
        qint = q.quant2int(weight.detach()).round()
        flat = qint.view(-1)
        if torch.any(~torch.isfinite(flat)):
            raise ValueError("QuantMatMul pack_weight encountered invalid values.")
        if q.n_bits == 8:
            dtype = torch.int8 if q.qmin < 0 else torch.uint8
            data = flat.to(dtype)
            zero_offset = 0
        else:
            zero_offset = -q.qmin if q.qmin < 0 else 0
            values = (flat + zero_offset).to(torch.int16)
            if torch.any(values < 0) or torch.any(values > 15):
                raise ValueError("4-bit matmul weights exceed nibble range.")
            data = pack_4bit_tensor(values)
        self.w_qint = data.contiguous()
        self.pack_meta = {
            "n_bits": q.n_bits,
            "weight_shape": tuple(weight.shape),
            "packed_dtype": str(self.w_qint.dtype),
            "zero_offset": zero_offset,
        }
        self.weight_fp32_cache = weight.detach().clone()
