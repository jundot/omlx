# MTP draft acceptance ~0% on mixed-convention Qwen3.6 MXFP4 checkpoints

## Summary

OMLX's native Multi-Token Prediction (MTP) speculative decoding produced **0%
draft acceptance** (no speedup; MTP effectively disabled) on certain
pre-quantized Qwen3.6 checkpoints — specifically MXFP4 bundles whose MTP-head
RMSNorm weights are stored in a **mixed** convention. The backbone and the
`mtp.norm` tensor were already in MLX's `+1` RMSNorm convention, while the
per-layer MTP head norms were still in raw-HuggingFace convention.

Affected model in this report: `Qwen3.6-27B-MXFP4-MTP` (JANGQ MXFP4 bundle).

## Symptom

Server MTP stats logged ~0% acceptance with all cycles falling back to full
verification:

```
MTP[0] finish=length tokens=150 cycles=148 accept=0/148 (0.0%)
  emits[init=2,draft=0,bonus=0,verify=148]
  timing[backbone=11097.9ms mtp=709.5ms ...]
```

Throughput sat at the no-MTP baseline (~11.5 tok/s) despite `mtp_enabled=true`.

## Root cause

`TextModel.sanitize` (in `omlx/patches/mlx_lm_mtp/qwen35_model.py`, and the
mirror in `omlx/patches/mlx_vlm_mtp/qwen35_vlm_model.py`) decides whether to
shift RMSNorm weights by `+1` using a single global flag derived solely from
the **backbone**:

```python
has_unsanitized_conv1d = any(
    "conv1d.weight" in k and getattr(v, "shape", (1,))[-1] != 1
    for k, v in weights.items()
)
should_shift_norm_weights = has_unsanitized_conv1d   # <-- global, backbone-only
...
if should_shift_norm_weights and any(k.endswith(sfx) for sfx in norm_keys):
    if v.ndim == 1:
        weights[k] = v + 1.0
```

For a checkpoint whose backbone is already converted (conv1d already sanitized),
`should_shift_norm_weights` is `False`, so **no** norm — backbone or MTP head —
is shifted. That is correct for the backbone. But this particular checkpoint
ships its per-layer MTP head norms in raw-HF convention (mean ~ 0), so they are
left unshifted and the head's RMSNorm multiplies activations by ~0. The MTP head
then emits near-flat logits, the verifier rejects every draft token, and
acceptance is 0%.

Measured pre-shift means for this model's MTP norms (a single head, layer 0):

| MTP norm tensor                              | pre-shift mean | convention   |
|----------------------------------------------|---------------:|--------------|
| `mtp.norm.weight`                            |        +1.2740 | MLX (+1)     |
| `mtp.layers.0.self_attn.q_norm.weight`       |        +0.7550 | MLX (+1)     |
| `mtp.layers.0.self_attn.k_norm.weight`       |        +0.7427 | MLX (+1)     |
| `mtp.layers.0.input_layernorm.weight`        |        +0.0406 | raw-HF       |
| `mtp.layers.0.post_attention_layernorm.weight`|       +0.2108 | raw-HF       |
| `mtp.pre_fc_norm_embedding.weight`           |        -0.4400 | raw-HF       |
| `mtp.pre_fc_norm_hidden.weight`              |        -0.1711 | raw-HF       |

The conventions are **mixed within the same head**, so neither a global
"shift all" nor a global "shift none" flag is correct.

## Fix

Decide the `+1` shift **per-key** for MTP norms, from each weight's own
magnitude. Raw-HF RMSNorm weights center near 0; MLX-shifted weights center
near 1. Threshold at 0.5:

```python
import mlx.core as _mx

def _mtp_norm_is_raw_hf(_w):
    try:
        return float(_mx.mean(_w.astype(_mx.float32)).item()) < 0.5
    except Exception:
        return False
...
if v.ndim == 1 and any(k.endswith(sfx) for sfx in norm_keys):
    if "mtp." in k:
        if _mtp_norm_is_raw_hf(v):
            weights[k] = v + 1.0
    elif should_shift_norm_weights:
        weights[k] = v + 1.0
```

Backbone norms keep the existing conv1d-derived behavior (no regression for
already-correct checkpoints). MTP head norms are evaluated individually, so a
raw-HF head norm gets `+1` even when a sibling head norm (e.g. `mtp.norm`) is
already shifted. The same change is applied to the mlx-vlm sanitize mirror,
where MTP keys are additionally prefixed `language_model.mtp.*` (hence the
`"mtp." in key` substring test rather than `startswith`).

The `< 0.5` threshold is robust: a correctly-converted MLX RMSNorm weight has
mean near 1.0 (it is a learned scale around unity), well clear of a raw-HF
weight centered near 0. Already-shifted norms are never double-shifted.

## Verification

Per-key decision against the real checkpoint (standalone):

```
mtp.layers.0.input_layernorm.weight          pre=+0.0406 raw_hf=True  -> +1 -> +1.0406  OK
mtp.layers.0.post_attention_layernorm.weight pre=+0.2108 raw_hf=True  -> +1 -> +1.2108  OK
mtp.layers.0.self_attn.k_norm.weight         pre=+0.7427 raw_hf=False ->  0 -> +0.7427  OK
mtp.layers.0.self_attn.q_norm.weight         pre=+0.7550 raw_hf=False ->  0 -> +0.7550  OK
mtp.norm.weight                              pre=+1.2740 raw_hf=False ->  0 -> +1.2740  OK
mtp.pre_fc_norm_embedding.weight             pre=-0.4400 raw_hf=True  -> +1 -> +0.5600  OK
mtp.pre_fc_norm_hidden.weight                pre=-0.1711 raw_hf=True  -> +1 -> +0.8289  OK
RESULT: ALL MTP NORMS IN +1 CONVENTION POST-FIX
```

Live server, 3 requests of 150 tokens each, `Qwen3.6-27B-MXFP4-MTP`:

| Metric             | Before fix       | After fix         |
|--------------------|------------------|-------------------|
| MTP draft accept   | 0/148 (0.0%)     | 73/75 (97.3%)     |
| Throughput         | ~11.5 tok/s      | ~21.4 tok/s       |
| Speedup            | 1.0x (baseline)  | ~1.86x            |

Results were stable across all three requests after the fix.

## Files changed

- `omlx/patches/mlx_lm_mtp/qwen35_model.py` — `TextModel.sanitize`
- `omlx/patches/mlx_vlm_mtp/qwen35_vlm_model.py` — VLM `sanitize` mirror

## Notes / scope

- No behavior change for checkpoints that were already correct: backbone norms
  still follow the conv1d signal, and MTP norms already in `+1` convention are
  left untouched (mean ≥ 0.5).
- The threshold approach assumes MTP RMSNorm scales are not legitimately
  centered below 0.5 in the MLX convention. This holds for the Qwen3.6 family
  (learned scales around unity). If a future checkpoint violates this, a more
  explicit convention marker in the checkpoint metadata would be preferable.
