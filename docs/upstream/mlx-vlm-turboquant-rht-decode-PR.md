# Upstream PR: fix TurboQuant L=1 value kernels under RHT (mlx-vlm)

**Target:** `Blaizzy/mlx-vlm` `main` (verified against `fea81522`; same on `f96138e` / `v0.5.0`).
**File:** `mlx_vlm/turboquant.py` — `_TurboQuantMSECodec`.

## Summary

`_TurboQuantMSECodec.weighted_sum` and `weighted_sum_stats_from_scores` call the
L=1 value-reconstruction Metal kernels (`_metal_mse_weighted_sum`,
`_metal_mse_weighted_sum_sum_from_scores`) **without** the `if not self.use_rht`
guard that the sibling `weighted_sum_from_scores` already has. Those kernels
finish with `matmul(weighted_rot, rotation)`, which is the inverse only for a
plain rotation. The codec defaults to `use_rht=True` (randomized Hadamard
transform), whose inverse is `_rht_inverse(.; signs)`. So under RHT the kernels
apply the wrong inverse transform and return essentially uncorrelated output.

## Impact

- Single-query **decode attention** through the slow/masked path is corrupt.
- Reproduction error is ~140% (of signal magnitude) at every bit depth (2–8),
  i.e. not a precision issue — a wrong-transform issue.
- Latent because the common decode path uses the fused `_fused_mse_decode_kernel`
  (mask is `None`/`"causal"`); the bug only shows when an array mask forces the
  slow path — e.g. continuous-batching decode with per-request left-padding.

## Root cause

`weighted_sum_from_scores` is guarded:
```python
if not self.use_rht:
    fast_output = _metal_mse_weighted_sum_from_scores(...)
    ...
```
but `weighted_sum` and `weighted_sum_stats_from_scores` are not. The non-fused
fallback paths (einsum + `self._rotate_inverse(...)`) are correct for both RHT
and plain rotation.

## Fix

Add `not self.use_rht and` to the two L=1 guards (see the patch). Under RHT this
takes the correct einsum/`_rotate_inverse` fallback; with a plain rotation
(`use_rht=False`) the kernels still run.

## Verification

```
# _TurboQuantMSECodec, 8-bit, single-query decode through the masked path
array-mask decode error:  before = 140.0%   after = 1.2%
```
End-to-end on `mlx-community/Llama-3.2-1B-Instruct-4bit`, continuous-batching
decode (B>1, left-padded) produces coherent output after the fix; before it is
garbage.

## Notes / suggested follow-up

- A proper fix in the kernels themselves (apply the RHT inverse instead of
  `matmul(., rotation)`) would let RHT use the fast path; this PR takes the
  conservative route (fall back to the correct math) matching the existing
  `weighted_sum_from_scores` behavior.
- Related: the fused single-token quantize kernel (`_try_fused_kv_quantize` /
  `_fused_kv_quantize_kernel`, the T=1 path) had an analogous decode-time
  defect that was fixed on `main` (`fea81522`); this PR addresses the remaining
  value-kernel/RHT case.
