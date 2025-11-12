# cobra/quantize/pct/__init__.py
"""
Percentile Clipping (PCT) toolkit.

Public API:
- PercentileAccumulator:   lightweight collector for activation percentiles
- select_best_percentile:  distribution-based best-p selection (z/g rules)
- decide_percentile:       override-aware selector (uses best-p if override None)
- TARGETS:                 canonical 4-module target keys
- resolve_hooks(model):    locate hook functions for the 4 targets
- compute_clip_range:      turn chosen percentile into (lo, hi)

Typical usage (collect → decide → apply):
    from cobra.quantize.pct import (
        PercentileAccumulator, TARGETS, resolve_hooks,
        decide_percentile, compute_clip_range
    )

    acc = PercentileAccumulator()
    hooks = resolve_hooks(model)  # dict[str, Callable[[Tensor], None]]

    # register forward hooks that push tensors into the accumulator
    handles = []
    for key, fn in hooks.items():
        def _hook(_m, _inp, out, *, _key=key):
            acc.record_activation(out, bucket=_key)
        handles.append(hooks[key].__self__.register_forward_hook(_hook))  # type: ignore

    # ... run a few calibration batches ...

    stats = acc.state_dict()  # {bucket: {min,max,percentiles,numel}}
    results = {}
    for key, s in stats.items():
        p = decide_percentile(s, override=None)
        lo, hi = compute_clip_range(s, p)
        results[key] = {"percentile": p, "lo": float(lo), "hi": float(hi)}

    # cleanup
    for h in handles: h.remove()
"""

from .observers import PercentileAccumulator
from .policy import select_best_percentile, decide_percentile
from .targets import TARGETS, resolve_hooks
from .range_calc import compute_clip_range

__all__ = [
    "PercentileAccumulator",
    "select_best_percentile",
    "decide_percentile",
    "TARGETS",
    "resolve_hooks",
    "compute_clip_range",
]

__version__ = "0.1.0"
