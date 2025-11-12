"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Union

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.utils.hooks import RemovableHandle

from cobra.quantize.pct import PercentileAccumulator, TARGETS, resolve_hooks
from cobra.pipeline.pct_schema import normalize_stage
from cobra.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from cobra.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from cobra.models.vlms import CobraVLM
from cobra.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = "han1997/cobra"


# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `cobra.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path], hf_token: Optional[str] = None, cache_dir: Optional[Union[str, Path]] = None
):
    """Loads a pretrained CobraVLM from either local disk or the HuggingFace Hub."""
    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for `config.json` and pretrained checkpoint
        assert (config_json := run_dir / "config.json").exists(), f"Missing `config.json` for `{run_dir = }`"
        assert (checkpoint_pt := run_dir / "checkpoints" / "latest-checkpoint.pt").exists(), "Missing checkpoint!"
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `cobra.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
        checkpoint_pt = hf_hub_download(
            repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
        )

    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=True,
    )

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint; Freezing Weights 🥶")
    vlm = CobraVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
    )

    return vlm


_SUBSAMPLE_LIMIT = 2_000_000


def attach_pct_observers(
    model: nn.Module,
    acc: PercentileAccumulator,
    targets: Optional[Iterable[str]] = None,
) -> List[RemovableHandle]:
    """
    Attach forward hooks to canonical percentile targets and stream activations
    into `acc`. Returns the list of hook handles so callers can remove them.
    """

    def _extract_tensor(obj: object) -> Optional[torch.Tensor]:
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (list, tuple)):
            for item in obj:
                t = _extract_tensor(item)
                if t is not None:
                    return t
        if isinstance(obj, dict):
            for item in obj.values():
                t = _extract_tensor(item)
                if t is not None:
                    return t
        return None

    raw_targets = list(targets or TARGETS)
    selected: List[str] = []
    for name in raw_targets:
        try:
            stage = normalize_stage(name)
        except Exception as exc:
            overwatch.warning(f"[pct] skipping unknown target '{name}': {exc}")
            continue
        if stage not in selected:
            selected.append(stage)
    if not selected:
        selected = list(TARGETS)

    resolved_raw = resolve_hooks(model)
    resolved: Dict[str, nn.Module] = {}
    for key, module in resolved_raw.items():
        try:
            stage = normalize_stage(key)
        except Exception:
            continue
        resolved[stage] = module
    missing = [k for k in selected if k not in resolved]
    if missing:
        raise RuntimeError(f"attach_pct_observers: unresolved targets {missing}")

    handles: List[RemovableHandle] = []
    for key in selected:
        module = resolved[key]

        def _hook(_module: nn.Module, _inputs, output, *, bucket: str = key) -> None:
            tensor = _extract_tensor(output)
            if tensor is None:
                overwatch.warning(f"[pct] hook '{bucket}' produced no tensor output; skipping")
                return
            tensor = tensor.detach()
            flat = tensor.reshape(-1)
            if flat.numel() > _SUBSAMPLE_LIMIT:
                idx = torch.randint(0, flat.numel(), (_SUBSAMPLE_LIMIT,), device=flat.device)
                tensor = flat[idx]
            else:
                tensor = flat

            try:
                acc.record_activation(tensor, bucket=bucket)
            except Exception as exc:
                overwatch.warning(f"[pct] record_activation failed for '{bucket}': {exc}")

        handle = module.register_forward_hook(_hook)
        handles.append(handle)
        overwatch.info(f"[pct] attached observer -> {key} ({module.__class__.__name__})")

    return handles
