# cobra/quantize/wrap/policy.py
"""
Policy system for deciding which modules to wrap with quantized counterparts.

This layer defines:
    - WrapPolicy: determines inclusion/exclusion of modules by path pattern and stage.
    - normalize_stage_name: maps module path to canonical stage keys
        ("vision.dino", "vision.siglip", "projector", "llm").

Typical usage:
    policy = WrapPolicy(
        stages=("vision.dino", "vision.siglip", "projector", "llm"),
        include=[".*(qkv|proj|fc|conv).*"],
        exclude=[".*(embedding|norm|rmsnorm|ln|pos_embed).*"],
    )
    if policy.allows(path, module):
        wrap = get_wrapper(module)
        ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Pattern, Sequence, Literal

from torch import nn

from cobra.pipeline.pct_schema import normalize_stage as pct_normalize_stage
from cobra.quantize.utils import RotationSpec, eligible_module

# ---------------------------------------------------------------------------
# Stage normalization
# ---------------------------------------------------------------------------


def normalize_stage_name(module_path: str) -> str:
    """
    Convert module path to canonical stage name.
    """
    try:
        return pct_normalize_stage(module_path)
    except Exception:
        p = module_path.lower()
        if "vision_backbone.dino" in p or "dino" in p:
            return "vision.dino"
        if "vision_backbone.siglip" in p or "siglip" in p:
            return "vision.siglip"
        if p.startswith("projector") or ".projector" in p or "mm_projector" in p:
            return "projector"
        if "llm_backbone" in p or p.startswith("llm") or "mamba" in p:
            return "llm"
        return "projector"


_CANONICAL_STAGES = ("vision.dino", "vision.siglip", "projector", "llm")


# ---------------------------------------------------------------------------
# Policy core
# ---------------------------------------------------------------------------


class WrapPolicy:
    """
    Determines which modules are eligible for quantized wrapping.

    Parameters
    ----------
    stages : Sequence[str] or None
        List of canonical stage names to allow. None means all.
    include : list[str]
        Regex or glob patterns for modules to include.
    exclude : list[str]
        Regex or glob patterns for modules to exclude (highest priority).

    Methods
    -------
    allows(module_path, module) -> bool
        Return True if this module should be wrapped.
    """

    def __init__(
        self,
        stages: Optional[Sequence[str]] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> None:
        self.stages = tuple(stages) if stages else _CANONICAL_STAGES
        self.include = self._compile_patterns(include or [".*"])
        self.exclude = self._compile_patterns(
            exclude
            or [
                ".*(embedding|embed_tokens|norm|rmsnorm|layernorm|ln|pos_embed).*",
            ]
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compile_patterns(pats: List[str]) -> List[Pattern]:
        out = []
        for p in pats:
            try:
                out.append(re.compile(p))
            except re.error:
                # fallback: escape bad regex
                out.append(re.compile(re.escape(p)))
        return out

    def _match_any(self, patterns: List[Pattern], s: str) -> bool:
        for pat in patterns:
            if pat.search(s):
                return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allows(self, module_path: str, module: nn.Module) -> bool:
        """
        Determine if the given module should be wrapped.

        Precedence:
            1. Stage filtering
            2. Exclude pattern
            3. Include pattern
        """
        stage = normalize_stage_name(module_path)
        if stage not in self.stages:
            return False
        # Exclusion first
        if self._match_any(self.exclude, module_path):
            return False
        # Inclusion second
        if not self._match_any(self.include, module_path):
            return False
        # Exclude leaf types we never quantize
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return False
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            return False
        return True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summarize(self) -> str:
        lines = [
            f"[WrapPolicy] stages={','.join(self.stages)}",
            f"[WrapPolicy] include={len(self.include)} regexes, exclude={len(self.exclude)} regexes",
        ]
        for i, pat in enumerate(self.include, 1):
            lines.append(f"  include[{i}] = {pat.pattern}")
        for i, pat in enumerate(self.exclude, 1):
            lines.append(f"  exclude[{i}] = {pat.pattern}")
        return "\n".join(lines)


@dataclass
class RotationPolicy:
    """
    Decide whether a module should receive a rotation (Hadamard/KLT) using RotationSpec metadata.
    """

    spec: RotationSpec
    method: Literal["bake", "inject"] = "bake"
    allow_conv: bool = True
    allow_linear: bool = True

    def should_rotate(self, name: str, module: nn.Module) -> bool:
        if not eligible_module(module):
            return False
        if isinstance(module, nn.Linear) and not self.allow_linear:
            return False
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and not self.allow_conv:
            return False
        lowered = name.lower()
        if "ssm" in lowered and not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return False
        # LayerNorm/Embedding already rejected via eligible_module but keep guardrails explicit
        if isinstance(module, (nn.LayerNorm, nn.Embedding, nn.EmbeddingBag)):
            return False
        return True


__all__ = ["WrapPolicy", "RotationPolicy", "normalize_stage_name"]
