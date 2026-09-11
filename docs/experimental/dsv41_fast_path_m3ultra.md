# DeepSeek-V4.1-Flash fast path on Apple M3 Ultra

Opt-in decode/prefill stack for `DeepSeek-V4.1-Flash` on Mac Studio (M3 Ultra).
Enable with `OMLX_DSV41_FAST=1`.

## What changed

- **Fast model path** (`omlx/patches/deepseek_v41/deepseek_v41_model.py` via
  `fast_path.py`): SwitchGLU MoE, wsdpa / native sparse prefill, deferred mHC,
  CSA2 shared runtime, decode-oriented indexer Metal helpers.
- **Engram modes**: `OMLX_DSV41_ENGRAM=stub|mmap|full` (default `mmap` when
  fast path is on). Optional hot-row cache for mmap/SSD gathers.
- **Quant load fix**: V4.1 fp8 weights use V4.1 `make_quantization_config`
  (engram.wkv mxfp8) so mmap load no longer expects missing affine biases.
- **Batch decode plumbing**: fresh pools per `make_cache`, shared-pool identity
  when converting to `BatchPoolingCache`, append-only updates for index /
  ratio-1 pools, array RoPE/indexer offsets.

## How to run (M3 Ultra)

```bash
source ~/omlx-venv313/bin/activate
export PYTHONPATH=/path/to/omlx
export OMLX_DSV41_FAST=1
export OMLX_DSV41_ENGRAM=stub   # or mmap + OMLX_DSV41_ENGRAM_DIR=...

python - <<'PY'
import mlx.core as mx
from omlx.patches.deepseek_v41.fast_path import apply_fast_path
apply_fast_path()
from mlx_lm import load, generate

mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
model, tok = load("/path/to/DeepSeek-V4.1-Flash")
print(generate(model, tok, prompt="The capital of France is", max_tokens=64))
PY
```

**Required for speed:** `mx.set_wired_limit(...)` to the device
`max_recommended_working_set_size`. Without it, decode collapses to ~0.4 tok/s
from memory thrash on a ~300 GB working set.

## M3 Ultra — before vs after

Hardware: Mac Studio, Apple M3 Ultra, mlx **0.32.2**, weights on Thunderbolt
SSD (`DeepSeek-V4.1-Flash`). Wired limit set in all timed runs below.

| Config | Decode (tok/s) | Prefill ~2k (tok/s) | Notes |
| --- | ---: | ---: | --- |
| **Before** stock patch (`language.py`, no `OMLX_DSV41_FAST`) | ~15.5 | ~337 | Prompt ~1105, gen 15; stock wall clock |
| **After** `OMLX_DSV41_FAST=1` + Engram **stub** | **~26.5–27.2** | **~490–500** | L=2048 prefill med ~499 |
| **After** fast + Engram **mmap** + hot cache | **~25.5** | **~481** | Hot cache ~25 GB/layer, ~85–87% hit |

Relative to stock on the same machine:

- Decode: **~1.7×** (15.5 → ~27 tok/s)
- Prefill: **~1.45×** (337 → ~490 tok/s)

Stub vs mmap decode gap is ~2–4% with the hot cache warm; prefill gap ~2%.

### Batched decode (fast path, stub, equal-length prompts)

After BatchPoolingCache fixes (fresh pools, shared identity, append-only index
pools):

| Batch B | Aggregate tok/s | ms/step | ms/tok |
| ---: | ---: | ---: | ---: |
| 1 | 26.8 | 37.4 | 37.4 |
| 2 | 44.4 | 45.1 | 22.5 |
| 4 | 62.0 | 64.5 | 16.1 |
| 8 | 79.5 | 100.6 | 12.6 |

~3× aggregate throughput at B=8 vs B=1. Batch vs independent B=1 is not yet
bit-exact (bf16 / path differences); twin rows inside one batch match.

## Env reference

| Variable | Meaning |
| --- | --- |
| `OMLX_DSV41_FAST=1` | Enable fast path |
| `OMLX_DSV41_ENGRAM=stub / mmap / full` | Engram backend |
| `OMLX_DSV41_ENGRAM_DIR` | mmap/SSD table directory |
| `OMLX_DSV41_ENGRAM_CACHE_GB` | Hot-row RAM budget (mmap) |

## Tests

```bash
pytest tests/test_dsv41_fast_path_unit.py \
       tests/test_dsv41_batch_pooling_append.py \
       tests/test_deepseek_v41_engram_hotcache.py -q
```
