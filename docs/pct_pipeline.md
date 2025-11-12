# Percentile Clipping (PCT) Pipeline

This document describes Cobra's activation percentile collection pipeline and how to bridge the collected clip ranges into the quantization stack.

## Overview

The pipeline has three high-level phases:

1. **Collect** activation statistics on the four canonical buckets with `cobra.pipeline.pct_collect`.
2. **Decide** the best percentile per bucket and emit a JSON of `(percentile, lo, hi)` pairs with `cobra.pipeline.pct_apply`.
3. **Inject** the resulting ranges into activation quantizers before fake/real quant either by calling `apply_pct_overrides_to_quantizers` or by manually using the quantizer `set_percentile_clip` API.

All helpers assume the four canonical bucket names:

| Bucket          | Description                                                                   |
|-----------------|-------------------------------------------------------------------------------|
| `vision.siglip` | SigLIP visual branch output (pre-projector, SigLIP-specific backbone)        |
| `vision.dino`   | DINO / DINOv2 visual branch output (pre-projector)                           |
| `llm`           | LLM token embedding output (post embedding, pre first block)                 |
| `projector`     | Multimodal projector output (activations that feed into the LLM backbone)    |

If you need fewer than four hooks you can pass a subset via `--targets` during collection.

## CLI Workflow

```bash
# 1) Collect percentile stats (.pt)
python -m cobra.pipeline.pct_collect \
  --model cobra+3b \
  --device cuda \
  --images /path/to/calibration/images \
  --batches 8 \
  --batch-size 4 \
  --out outputs/percentile_stats.pt

# 2) Convert stats to JSON clip table
python -m cobra.pipeline.pct_apply \
  --stats outputs/percentile_stats.pt \
  --out outputs/percentile_overrides.json
```

Key options:

- `--targets`: Comma-separated bucket names. Defaults to the four canonical keys above.
- `--percentile-override`: Force a single percentile for every bucket (skips the rule-based selector). Leave unset to use the heuristics in `cobra.quantize.pct.policy`.
- `--export-best-json`: Optional direct export during collection so you receive both `.pt` and `.json` without running `pct_apply` separately.

### Hooking and Injection

- During collection you can rely on `resolve_hooks(model)` or the higher-level `attach_pct_observers(model, accumulator)` helper (see `cobra.models.load`) to register forward hooks. If a bucket cannot be found you can call `find_node(model, "vision.siglip")` to inspect what the resolver is trying to match.
- After `pct_apply`, map bucket names to activation quantizers and call:

```python
from cobra.quantize.quantizer import apply_pct_overrides_to_quantizers

bucket_to_quant = {
    "vision.siglip": siglip_act_quantizer,
    "vision.dino": dino_act_quantizer,
    "llm": llm_act_quantizer,
    "projector": projector_act_quantizer,
}
apply_pct_overrides_to_quantizers(bucket_to_quant, "outputs/percentile_overrides.json")
```

Each quantizer receives the `(lo, hi)` overrides via `set_percentile_clip`, and activation wrappers (`QuantLinear`, `QuantConv*`, etc.) will clamp tensors tagged with `_pct_bucket` before fake/real quantization.

## Recommended Settings & Safeguards

- **Batch count**: 4–8 mini-batches with batch size 4 is typically sufficient; going beyond 16 mini-batches rarely changes percentiles but increases calibration time.
- **Random text**: Leave `--no-random-text` disabled unless you supply real token sequences. Random token IDs ensure the embedding target receives non-zero variance.
- **Random subsampling**: Hook helpers automatically flatten outputs and randomly sample up to `2_000_000` elements to avoid OOM. This preserves percentile fidelity while preventing large ViT activations from exhausting memory.
- **I/O Paths**: default `outputs/percentile_stats.pt` and `outputs/percentile_overrides.json`. Override via `--out`/`--stats`/`--export-best-json` or the `PctConfig` dataclass (see `cobra/conf/pct.py`).

## Common Errors & Remedies

| Symptom | Cause | Fix |
|---------|-------|-----|
| `resolve_hooks: cannot locate ...` | Module naming differs from heuristics. | Use `find_node(model, bucket)` to inspect candidates or pass `--targets` to skip missing buckets. Custom models may need adapter modules named with `vision.*`, `llm`, or `projector` keywords. |
| `S=0` or `select_best_percentile` complaining about zero scale | Activation slice is constant or too few samples; `_robust_scale` guards with `1e-6` but extreme sparsity can still propagate ZeroDivision warnings. | Collect more batches or ensure the hook taps a tensor before ReLU clamps everything to zero. |
| `Missing percentile p99.999` | Accumulator never recorded the bucket or the tensor contained NaNs causing quantile computation to skip. | Re-run collection and confirm hooks fire (log output shows `[hook] attached -> ...`). Verify tensors are finite. |
| Clip override conflicts | You manually call `set_percentile_clip` and also provide overrides through JSON. | Last writer wins. Prefer the JSON flow and avoid per-quantizer overrides unless debugging. |

## Additional Tips

- When running the CLI on large models, keep the calibration batches short and rely on the built-in subsampling rather than raising the batch size.
- If you need to inspect stats, `torch.load(outputs/percentile_stats.pt)` returns a dict `{bucket: {"min": ..., "max": ..., "percentiles": {...}}}` which you can pretty-print before running `pct_apply`.
- Automation scripts can parse the JSON and log `% coverage` per bucket to verify expected ranges before quantizing weights/activations.
