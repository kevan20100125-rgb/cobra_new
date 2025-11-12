# cobra/quantize/pct/observers.py
"""
PercentileAccumulator — lightweight activation statistics collector.

This module is used during offline calibration to collect activation
distributions for the four primary modules:
    "vision.siglip", "vision.dino", "llm", "projector"

It records basic statistics:
    - min, max
    - percentiles (p25, p50, p75, p90, p99, p99.9, p99.99, p99.999)
    - number of elements (numel)

These are later used by cobra.quantize.pct.policy to decide the best
percentile for clipping each module.
"""

import torch
from typing import Dict, Any


class PercentileAccumulator:
    """
    Collects activation statistics for percentile clipping calibration.
    It keeps only aggregated percentiles, no raw samples.
    """

    _ORDER = [25.0, 50.0, 75.0, 90.0, 99.0, 99.9, 99.99, 99.999]

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        # mapping: bucket -> dict of stats
        self._stats: Dict[str, Dict[str, Any]] = {}

    @torch.no_grad()
    def record_activation(self, x: torch.Tensor, bucket: str) -> None:
        """
        Record statistics for one activation tensor under the given bucket name.
        """
        if x is None:
            return
        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        flat = x.detach().to(self.device, non_blocking=True).float().reshape(-1)
        numel = flat.numel()
        if numel == 0:
            return

        # subsample to avoid OOM if activation is huge
        if numel > 2_000_000:
            idx = torch.randint(0, numel, (2_000_000,), device=flat.device)
            flat = flat[idx]

        stats = {
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
            "numel": int(numel),
        }

        qs = torch.tensor([p / 100 for p in self._ORDER], device=flat.device)
        vals = torch.quantile(flat, qs)
        percentiles = {f"p{p}": float(v.item()) for p, v in zip(self._ORDER, vals)}
        stats["percentiles"] = percentiles

        self._stats[bucket] = stats

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Return all collected statistics.
        """
        return self._stats

    def clear(self) -> None:
        """Erase all stored statistics."""
        self._stats.clear()

    def __len__(self) -> int:
        return len(self._stats)

    def __repr__(self) -> str:
        return f"PercentileAccumulator(keys={list(self._stats.keys())})"


__all__ = ["PercentileAccumulator"]
