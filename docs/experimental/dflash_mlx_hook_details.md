# DFlash-MLX Hook Details

Date: 2026-04-22

Notes on how dflash-mlx inserts hooks into a model when it loads. 

dflash-mlx installs three types of hooks at the **class level** on model layers when `load_target_bundle()` is called. These hooks wrap attention and projection layers to enable speculative decoding, but they also affect any code path that uses the same model — including fallback engines.

## Hook Types

### 1. Split Full Attention Hook (`_install_split_full_attention_hook`)

Wraps `self_attn.__call__` with a custom split attention path. When `_dflash_split_sdpa_enabled=True`, the hook runs its own manual SDPA computation (separate q/k/v projection, rope, attention) instead of the original optimized path.

**Impact on fallback engine:** Adds Python wrapper overhead and runs a less-optimized attention path even during prefill. Causes ~25% TG TPS regression and ~30-40% slower prefill throughput on the fallback path.

**Mechanism:**
```python
# In dflash_mlx.runtime.py line 652
cls.__call__ = split_call  # wraps original __call__

# split_call checks _dflash_split_sdpa_enabled flag
# When True: runs custom attention path (manual q/k/v proj, rope, SDPA)
# When False: calls original_call (unhooked path)
```

### 2. Speculative Linear Cache Hook (`_install_speculative_linear_cache_hook`)

Wraps `linear_attn.__call__` with speculative logic. Checks if cache is `RecurrentRollbackCache` and armed; otherwise calls original.

**Impact on fallback engine:** Minimal — the cache won't be `RecurrentRollbackCache` in BatchedEngine, so it already calls `original_call`. But the hook wrapper still adds a Python call frame.

**Mechanism:**
```python
# In dflash_mlx.runtime.py line 381-400
def speculative_call(self, x, cache=None):
    if isinstance(cache, RecurrentRollbackCache) and cache.is_armed:
        # Run speculative path with tape replay
        return _speculative_linear_attn(self, x, cache)
    return original_call(x, cache)  # fallback to original
```

### 3. Exact Small Proj Hooks (`_install_exact_small_proj_hooks`)

Wraps `in_proj_b` and `in_proj_a` with `_ExactSmallProjPad` class. Pads short sequences to a minimum length (`pad_m=16`) before projection, ensuring the draft model's small-sequence assumptions hold.

**Impact on fallback engine:** Changes weight shape behavior — the wrapped layer adds padding logic that isn't needed for normal inference.

**Mechanism:**
```python
# In dflash_mlx.runtime.py line 274-303
class _ExactSmallProjPad(nn.Module):
    def __init__(self, linear: nn.Module, *, pad_m: int = 16):
        self.linear = linear  # stores original wrapped layer
        self.pad_m = pad_m
    
    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3 and x.shape[1] < self.pad_m:
            # Pad short sequences before projection
            pad = mx.zeros((batch_size, self.pad_m - seq_len, hidden_dim))
            out = self.linear(mx.concatenate([x, pad], axis=1))
            return out[:, :seq_len, :]
        return self.linear(x)  # no padding needed
```

## Hook Lifecycle

Hooks are installed once when `load_target_bundle()` is called in dflash-mlx's runtime. They persist on the model class for the lifetime of the process.

```
load_target_bundle()
    │
    ├─ _install_split_full_attention_hook(self_attn)
    ├─ _install_speculative_linear_cache_hook(linear_attn)
    └─ _install_exact_small_proj_hooks(linear_attn)
```

The hooks are **class-level**, meaning they affect all instances of the attention class — including any fallback engine that shares the same model object.
