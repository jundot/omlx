# DeepSeek-V4.1 (CSA2) omlx patch

Loads `deepseek-ai/DeepSeek-V4.1-Flash` (`model_type=deepseek_v41`) into omlx / mlx-lm on Apple Silicon.

## Engram hot-row cache

The quantized RAM working-set cache lives in **`engram_cache.py`**.

| Piece | Where |
| --- | --- |
| Cache class | `QuantHotRowCache` in `engram_cache.py` |
| Bound on tables | `MmapEngramTable.hot_cache` in `engram.py` |
| Budget | `OMLX_DSV41_ENGRAM_CACHE_GB` (default `50`) |
| Policy | `OMLX_DSV41_ENGRAM_CACHE_POLICY` (`lru` \| `clock`) |
| Stats | `engram_cache_stats(model)` |
| FP8 helpers | `engram_fp8.py` |
| SSD tables | `OMLX_DSV41_ENGRAM_DIR` + `OMLX_DSV41_ENGRAM=mmap` |

Full tables stay on SSD via memmap; only touched rows enter the hot cache as FP8+scale (~264 B/row).


## Architecture (exactness)

Source of truth: HuggingFace `inference/model.py` + `DeepSeek_V41_Tech_Report.pdf`.

| Topic | V4.1 behavior |
|--------|----------------|
| Attention | Unified CSA2: window + optional compress top-k |
| Sharing | `SharedAttentionRuntime` via `kv_source_layer_ids` / `index_source_layer_ids` |
| compress_ratios | `{0,1,2}` only — **no** V4 `{4,128}`, **no APE**, **no ratio-4 overlap** |
| Indexer | `wk` from compressor latent; `index_n_heads=32`; `select_candidate_blocks` |
| mHC | **Deferred**: attn uses prior FFN `pre_mix`; FFN uses this attn `pre` |
| Norm | `rms_norm_eps=1e-20` |
| MoE gate | `sqrtsoftplus`; bias for select; scores for weights; `bias_vl` for VL |
| Engram | Layers `[1,14]` (stub tables by default — see below) |
| DSpark | Config fields wired (`dspark_target_layer_ids=[37,38,39]`); MTP forward TODO |
| Vision | Phase-2 stub: raises if `images=` passed |

## Gate collision

**Never** use `model_type.startswith("deepseek_v4")` — it matches `deepseek_v41`.
Use `omlx.patches.deepseek_v41.predicates`:

- `is_deepseek_v4` — V4 / V4-Flash only
- `is_deepseek_v41` — V4.1 only
- `is_deepseek_v4_family` — either (shared tooling)

## Quant notes

- Weight block: 32×32
- Window KV: fp8 @ 32
- Compress KV: fp4 @ 16 / E4M3 scales
- Index: fp4 @ 32
- Experts: mxfp4; shared / attn: mxfp8 (`make_quantization_config`)

## Engram memory

Full tables are ~200GB FP8 (layers 1 and 14). Modes:

```bash
export OMLX_DSV41_ENGRAM=stub   # default: no real tables
export OMLX_DSV41_ENGRAM=mmap   # SSD memmap + optional hot-row cache
# export OMLX_DSV41_ENGRAM=full  # resident embeddings (not for 512GB hosts)
export OMLX_DSV41_ENGRAM_DIR=~/llm/DeepSeek-V4.1-Flash-engram
export OMLX_DSV41_ENGRAM_CACHE_GB=50   # see engram_cache.py
```

Hot-row cache details: **`engram_cache.py`** (section above).

## How to load (once weights are local)

```python
from omlx.patches.deepseek_v41 import apply_deepseek_v41_patch
apply_deepseek_v41_patch()

from mlx_lm import load
model, tokenizer = load("/Volumes/TB5/llm/DeepSeek-V4.1-Flash")  # or HF id after convert
```

Via omlx server: set model path / HF id with `model_type=deepseek_v41` in `config.json`;
`maybe_apply_pre_load_patches` applies this package automatically.

Converted MLX weights (after official `convert.py` / omlx oq) should sanitize through
`Model.sanitize` (embed/head/hc_*/expert stack remaps).

## Tests

```bash
cd /Users/homecorpstudio/third-party/omlx
/Users/homecorpstudio/omlx-venv313/bin/pytest tests/test_deepseek_v41.py -q
```

## Reused from V4 (do not break V4)

- `PoolingCache` / generate patch / cache handlers
- `hyper_connection` Sinkhorn kernels + `HyperHead`
- `SwitchGLU` (MXFP4)
- `wsdpa_attention` prefill paths
- `decode_consistency.matmul`

## Known gaps

1. Vision ViT/aligner not implemented (explicit error).
2. DSpark speculative loop / `mtp.*` modules not executed (config retained).
3. Engram hash state + full mmap tables optional.
4. Chat template: use shipped tokenizer_config / encoding until a dedicated
   `chat_template_v41` is ported (V4 DSML template is a starting point for tools).
5. Native DSA indexer kernels tuned for H=64; V4.1 uses H=32 — MLX fallback path.
