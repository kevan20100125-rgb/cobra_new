"""
CLI utility for estimating and applying rotations (Hadamard / KLT) in the quantization pipeline.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from cobra.models.load import load_model
from cobra.models.materialize import materialize_model
from cobra.quantize.utils import RotationSpec, estimate_and_apply_rotation
from cobra.quantize.wrap.policy import RotationPolicy
from cobra.quantize.wrap.registry import get_rotation_targets
from cobra.util.torch_utils import get_logger


def _parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _module_io_dims(module: torch.nn.Module) -> Optional[Tuple[int, int]]:
    weight = getattr(module, "weight", None)
    if weight is None or not torch.is_tensor(weight):
        return None
    shape = tuple(weight.shape)
    if len(shape) < 2:
        return None
    out_dim = int(shape[0])
    in_dim = 1
    for dim in shape[1:]:
        in_dim *= int(dim)
    return in_dim, out_dim


class _DummyRotationDataset(Dataset):
    def __init__(self, length: int = 32, seq_len: int = 16, vocab: int = 32000, image_size: int = 224, seed: int = 0) -> None:
        super().__init__()
        self.length = max(1, int(length))
        self.seq_len = seq_len
        self.vocab = vocab
        self.image_size = image_size
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        pixel_values = torch.randn(3, self.image_size, self.image_size, generator=self.generator)
        input_ids = torch.randint(0, self.vocab, (self.seq_len,), generator=self.generator)
        return {"pixel_values": pixel_values, "input_ids": input_ids}


def _call_factory(factory: Any, kwargs: Dict[str, Any]) -> Any:
    sig = inspect.signature(factory)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return factory(**supported)


def _normalize_loader(obj: Any, batch_size: int, num_workers: int) -> DataLoader:
    if isinstance(obj, DataLoader):
        return obj
    if isinstance(obj, Dataset):
        return DataLoader(obj, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    raise TypeError(f"Factory must return a DataLoader or Dataset, got {type(obj)}")


def _build_dataloader(
    dataset_spec: Optional[str],
    split: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> Optional[DataLoader]:
    if not dataset_spec or dataset_spec.lower() in {"", "none"}:
        return None
    spec = dataset_spec.strip()
    if spec.lower() == "dummy":
        dataset = _DummyRotationDataset(length=batch_size * 2, seed=seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    if ":" in spec:
        module_name, attr = spec.split(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, attr)
        try:
            obj = _call_factory(
                factory,
                {
                    "split": split,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "seed": seed,
                    "device": device,
                },
            )
        except TypeError:
            obj = _call_factory(
                factory,
                {
                    "split": split,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "seed": seed,
                },
            )
        return _normalize_loader(obj, batch_size, num_workers)
    path = Path(spec)
    if path.is_file():
        obj = torch.load(path, map_location="cpu")
        return _normalize_loader(obj, batch_size, num_workers)
    raise ValueError(
        f"Unsupported dataset spec '{dataset_spec}'. Use 'none', 'dummy', "
        "a module path like 'pkg.module:create_loader', or a torch.save'd dataset file."
    )


def _print_dry_run(targets: Sequence[Tuple[str, torch.nn.Module]]) -> None:
    if not targets:
        print("[QuantRotate] No eligible modules matched the provided prefixes.")
        return
    width = max(len(name) for name, _ in targets)
    print(f"[QuantRotate] Previewing {len(targets)} rotation candidates:")
    for name, module in targets:
        dims = _module_io_dims(module)
        shape = tuple(getattr(getattr(module, "weight", None), "shape", ()))
        if dims is None:
            print(f"  {name:<{width}}  weight={shape}  (skipped: missing weight)")
            continue
        print(f"  {name:<{width}}  weight={shape}  in={dims[0]}  out={dims[1]}")


def build_argparser() -> argparse.ArgumentParser:
    examples = """
Example 1: Hadamard rotation with baking
  python -m cobra.switches.quant_rotate --model cobra+3b --dataset textvqa --split train[:512] \\
    --targets vision,projector,llm_backbone --rotation hadamard --axis io --scope per_channel \\
    --block-size 64 --method bake --manifest-out outputs/rotate/rot_manifest.json

Example 2: KLT rotation injection (estimate 8 batches)
  python -m cobra.switches.quant_rotate --model cobra+3b --dataset textvqa --split train[:1024] \\
    --targets vision,projector --rotation klt --axis in --scope per_channel \\
    --klt-batches 8 --method inject --manifest-out outputs/rotate/rot_manifest.pt
"""
    parser = argparse.ArgumentParser(
        description="Cobra Quantization Rotation Stage",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="Model identifier understood by cobra.models.load")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset loader spec. Use 'none' for no data, 'dummy' for synthetic random batches, "
                        "or 'module:factory' to call a custom function that returns a DataLoader/Dataset.")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split hint passed to the loader factory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the calibration dataloader")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of worker processes for the dataloader")
    parser.add_argument("--targets", type=str, default=None,
                        help="Comma-separated module prefixes to rotate (e.g., vision_backbone,llm_backbone)")
    parser.add_argument("--rotation", type=str, choices=("hadamard", "klt"), default="hadamard")
    parser.add_argument("--axis", type=str, choices=("in", "out", "io"), default="io")
    parser.add_argument("--scope", type=str, choices=("per_tensor", "per_channel"), default="per_channel")
    parser.add_argument("--block-size", type=int, default=64, help="Block size for Hadamard rotations")
    parser.add_argument("--klt-batches", type=int, default=8, help="Number of batches to sample for KLT estimation")
    parser.add_argument("--method", type=str, choices=("bake", "inject"), default="bake")
    parser.add_argument("--manifest-out", type=str, default="outputs/rotation_manifest.json",
                        help="Path to save rotation manifest (.json or .pt)")
    parser.add_argument("--dry-run", action="store_true", help="List eligible modules and exit without modifying weights")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling / dataloader factories")
    parser.add_argument("--device", type=str, default=None, help="Device for running rotation estimation (default: auto)")
    parser.add_argument("--whiten", action="store_true", help="Enable mean subtraction before computing KLT rotations")
    parser.add_argument("--eps", type=float, default=1e-6, help="Stability epsilon for covariance eigendecomposition")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging verbosity (DEBUG, INFO, ...)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    log = get_logger("quant_rotate")
    log.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    log.info(f"[Step] Loading model {args.model}")
    model = load_model(args.model)
    model = materialize_model(model)

    prefixes = _parse_csv(args.targets)
    targets = get_rotation_targets(model, prefixes or None)
    spec = RotationSpec(
        name=args.rotation,
        scope=args.scope,
        axis=args.axis,
        block_size=args.block_size if args.rotation == "hadamard" else None,
        whiten=args.whiten,
        eps=args.eps,
    )
    policy = RotationPolicy(spec=spec, method=args.method)

    if args.dry_run:
        _print_dry_run(targets)
        return

    dataloader = _build_dataloader(
        args.dataset,
        args.split,
        args.batch_size,
        args.num_workers,
        args.seed,
        device,
    )
    if dataloader is None:
        log.info("[Data] No dataloader provided; Hadamard rotations will not sample activations.")
    else:
        log.info(f"[Data] Using dataloader from spec '{args.dataset}' with batch_size={args.batch_size}")
    if args.rotation == "klt" and dataloader is None:
        raise ValueError("KLT rotation requires a dataset/dataloader. Please supply --dataset.")

    manifest_path = Path(args.manifest_out) if args.manifest_out else None
    result = estimate_and_apply_rotation(
        model=model,
        dataloader=dataloader,
        policy=policy,
        method=args.method,
        klt_batches=args.klt_batches,
        manifest_out=manifest_path,
        seed=args.seed,
        device=device,
    )

    stats = result.get("stats", {})
    manifest = result.get("manifest", {})
    handles = result.get("handles", [])
    rotated = stats.get("rotated", 0)
    skipped = stats.get("skipped", 0)
    duration = stats.get("duration_sec", 0.0)
    print(
        "[QuantRotate] Rotation complete | "
        f"method={stats.get('method')} axis={stats.get('axis')} "
        f"rotated={rotated} skipped={skipped} duration={duration:.2f}s"
    )
    skipped_details = stats.get("skipped_details") or []
    if skipped_details:
        print("[QuantRotate] Skipped modules (first 10):")
        for entry in skipped_details[:10]:
            print(f"  - {entry.get('path')}: {entry.get('reason')}")
        if len(skipped_details) > 10:
            print(f"  ... ({len(skipped_details) - 10} more)")
    if manifest_path:
        print(f"[QuantRotate] Manifest saved to {manifest_path}")
    else:
        print("[QuantRotate] Manifest saving skipped (no --manifest-out provided)")
    print(f"[QuantRotate] Manifest items recorded: {len(manifest.get('items', []))}")
    if handles:
        print(f"[QuantRotate] Active injection hooks: {len(handles)} (keep references alive during inference)")


if __name__ == "__main__":
    main()
