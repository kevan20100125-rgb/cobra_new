# cobra/quantize/pct/targets.py
"""
Resolve four canonical activation-collection targets for percentile clipping:

Targets (fixed keys):
    - "vision.siglip" : SigLIP visual backbone feature output (pre-projector)
    - "vision.dino"   : DINO/DINOv2 visual backbone feature output (pre-projector)
    - "llm"           : LLM token embedding output (post-embedding / pre-first-block)
    - "projector"     : Multimodal projector output (encoder output fed to LLM)

This module provides a best-effort, name-based resolver that navigates the
Cobra model tree and returns the modules on which you can register forward hooks.
It does NOT modify the model; it only finds hook points.

Usage:
    from cobra.quantize.pct.targets import TARGETS, resolve_hooks
    mods = resolve_hooks(model)
    handles = []
    for key, mod in mods.items():
        def _hook(_m, _inp, out, _key=key):
            accumulator.record_activation(out, bucket=_key)
        handles.append(mod.register_forward_hook(_hook))

    # ... run a few calibration batches ...
    for h in handles: h.remove()
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Mapping, Sequence

from torch import nn

# Canonical four-target list
TARGETS: List[str] = ["vision.siglip", "vision.dino", "llm", "projector"]
TARGET_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "vision.siglip": (
        "vision_backbone.siglip",
        "models.backbones.vision.siglip_vit",
    ),
    "vision.dino": (
        "vision_backbone.dino",
        "models.backbones.vision.dinov2_vit",
        "models.backbones.vision.dinosiglip_vit",
    ),
    "projector": (
        "projector",
        "models.vlms.projector",
        "projector.out",
    ),
    "llm": (
        "llm_backbone",
        "models.backbones.llm",
        "models.mamba.modeling_mamba",
    ),
}


# ----------------------------- helpers --------------------------------- #

def _norm(s: str) -> str:
    return s.replace("/", ".").lower()


def _iter_named_modules(root: nn.Module) -> List[Tuple[str, nn.Module]]:
    # torch's named_modules includes root as ('', root); filter it out
    return [(n, m) for n, m in root.named_modules() if n]


def _filter_by_keywords(
    pairs: List[Tuple[str, nn.Module]],
    include: List[str],
    exclude: Optional[List[str]] = None,
) -> List[Tuple[str, nn.Module]]:
    inc = [k.lower() for k in include]
    exc = [k.lower() for k in (exclude or [])]
    out: List[Tuple[str, nn.Module]] = []
    for n, m in pairs:
        nl = _norm(n)
        if all(k in nl for k in inc) and not any(k in nl for k in exc):
            out.append((n, m))
    return out


def _prefer_modules(
    pairs: List[Tuple[str, nn.Module]],
    prefer_types: Tuple[type, ...],
) -> List[Tuple[str, nn.Module]]:
    """Stable-filter keeping only modules whose type is in prefer_types when possible."""
    typed = [(n, m) for n, m in pairs if isinstance(m, prefer_types)]
    return typed if typed else pairs


def _pick_last_by_depth(pairs: List[Tuple[str, nn.Module]]) -> Optional[Tuple[str, nn.Module]]:
    """Pick the deepest (longest dotted name) module."""
    if not pairs:
        return None
    return max(pairs, key=lambda kv: kv[0].count("."))


def _find_embedding_module(root: nn.Module) -> Optional[Tuple[str, nn.Module]]:
    """Try to find token embedding for LLM."""
    pairs = _iter_named_modules(root)
    # common names
    candidates = _filter_by_keywords(
        pairs,
        include=["embed", "token"],
        exclude=["vision", "proj", "projector"],
    )
    # also accept plain nn.Embedding anywhere under llm
    if not candidates:
        candidates = [(n, m) for n, m in pairs if isinstance(m, nn.Embedding)]
    # prefer Embedding / Linear used as input-embed
    candidates = _prefer_modules(candidates, (nn.Embedding, nn.Linear))
    return _pick_last_by_depth(candidates)


def _find_projector_module(root: nn.Module) -> Optional[Tuple[str, nn.Module]]:
    """Find the multimodal projector module output."""
    pairs = _iter_named_modules(root)
    candidates = _filter_by_keywords(pairs, include=["projector"])
    # prefer MLP-ish or Linear / Sequential blocks
    candidates = _prefer_modules(candidates, (nn.Sequential, nn.Linear))
    return _pick_last_by_depth(candidates)


def _find_visual_branch(root: nn.Module, key: str) -> Optional[Tuple[str, nn.Module]]:
    """
    Find a visual backbone branch by keyword:
        key == "siglip" or "dino"
    Returns the deepest module under that branch to hook.
    """
    pairs = _iter_named_modules(root)
    # Names in Cobra often carry the backbone id: "siglip", "dinov2", "dino"
    inc = ["vision", key]
    # avoid projector and text modules
    exc = ["projector", "llm", "language", "embed", "token"]
    candidates = _filter_by_keywords(pairs, include=inc, exclude=exc)

    # Prefer end-of-encoder blocks: LayerNorm / Linear / Sequential often sit at outputs
    candidates = _prefer_modules(candidates, (nn.LayerNorm, nn.Linear, nn.Sequential))
    return _pick_last_by_depth(candidates)


def _resolver_for_key(model: nn.Module, key: str) -> Optional[Tuple[str, nn.Module]]:
    """Internal: dispatch heuristics for a canonical target key."""
    pairs = _iter_named_modules(model)
    key = key.lower()

    if key == "vision.siglip":
        sig = _find_visual_branch(model, key="siglip")
        if sig is None:
            sig = _pick_last_by_depth(_filter_by_keywords(pairs, include=["siglip"]))
        return sig

    if key == "vision.dino":
        dino = _find_visual_branch(model, key="dinov2")
        if dino is None:
            dino = _find_visual_branch(model, key="dino")
        if dino is None:
            dino = _pick_last_by_depth(_filter_by_keywords(pairs, include=["dino"]))
        return dino

    if key == "llm":
        llm = _find_embedding_module(model)
        if llm is None:
            llm_pairs = _filter_by_keywords(pairs, include=["llm"])
            llm_embed = [(n, m) for n, m in llm_pairs if isinstance(m, (nn.Embedding, nn.Linear))]
            llm = _pick_last_by_depth(llm_embed) if llm_embed else None
        return llm

    if key == "projector":
        proj = _find_projector_module(model)
        if proj is None:
            proj = _pick_last_by_depth(_filter_by_keywords(pairs, include=["projector"]))
        return proj

    return None


# --------------------------- public resolver ---------------------------- #

def find_node(model: nn.Module, key: str) -> Tuple[str, nn.Module]:
    """
    Locate a module that matches the canonical percentile target described by `key`.

    The resolver uses both name-based heuristics (keywords inside the dotted module
    path) and preferred module types (LayerNorm/Linear/etc.) to choose a stable hook
    point near the end of each branch. Returns the dotted name and module so callers
    can inspect or hook as needed.
    """
    node = _resolver_for_key(model, key)
    if node is None:
        raise RuntimeError(
            f"find_node: cannot locate target '{key}'. "
            "Ensure the model exposes matching module names and types."
        )
    return node


def get_finalize_default_targets(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Conservative defaults for finalization: vision/projector/LLM mixer projections.
    """
    targets: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        lname = name.lower()
        if isinstance(module, (nn.LayerNorm, nn.Embedding, nn.EmbeddingBag)):
            continue
        if lname.startswith("vision") or lname.startswith("projector"):
            targets.append((name, module))
            continue
        if "llm_backbone" in lname and ".mixer." in lname and (
            lname.endswith(".in_proj") or lname.endswith(".out_proj")
        ):
            targets.append((name, module))
    return targets


def _is_rotation_linear(module: nn.Module) -> bool:
    return isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))


def get_rotate_default_targets(
    model: nn.Module,
    extra_prefixes: Optional[Sequence[str]] = None,
) -> List[Tuple[str, nn.Module]]:
    """
    Enumerate conservative default rotation targets:
      - Any module under `vision*` or `projector*` that is Linear/Conv.
      - LLM mixer projections (`llm_backbone.*.mixer.*.(in_proj|out_proj)`).
      - LayerNorm/Embedding modules are excluded even if the name matches.
    """
    prefixes = tuple(extra_prefixes or ())
    results: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        lname = name.lower()
        if isinstance(module, (nn.LayerNorm, nn.Embedding, nn.EmbeddingBag)):
            continue
        include = False
        if lname.startswith("vision") or lname.startswith("projector"):
            include = _is_rotation_linear(module)
        elif "llm_backbone" in lname and ".mixer." in lname and (
            lname.endswith(".in_proj") or lname.endswith(".out_proj")
        ):
            include = isinstance(module, nn.Linear)
        elif prefixes and any(lname.startswith(p.lower()) for p in prefixes):
            include = _is_rotation_linear(module)
        if include:
            results.append((name, module))
    return results


def resolve_hooks(model: nn.Module) -> Dict[str, nn.Module]:
    """
    Resolve modules to attach forward hooks for the canonical four targets.

    Returns:
        dict:
           {
             "vision.siglip":  <nn.Module>,
             "vision.dino":    <nn.Module>,
             "llm":            <nn.Module>,
             "projector":      <nn.Module>,
           }

    Raises:
        RuntimeError if any target cannot be resolved with clear guidance.
    """
    resolved: Dict[str, nn.Module] = {}
    for key in TARGETS:
        _name, module = find_node(model, key)
        resolved[key] = module
    return resolved


__all__ = ["TARGETS", "find_node", "resolve_hooks"]
