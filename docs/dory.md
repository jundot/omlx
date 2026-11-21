# NVIDIA Dory text models

The text engine supports local MLX Dory checkpoints with the architecture in
mlx-lm [PR 1988](https://github.com/ml-explore/mlx-lm/pull/1988), head
`6fb10761eb7fd6814210a0122e6f4d3b4037eee1`. Place an already prepared MLX
checkpoint under the server's model directory and select it normally; for
example, `omlx serve --model-dir /path/to/local/models`. This change does not
provide weights or convert an original checkpoint. Check the model's own
license before use or distribution.

`config.json` selects `model_type: "dory"`, `n_recurrent_loops` and
`recurrent_kv_cache_mode: "per_loop"`. The architecture defaults to two loops
when the field is absent; four loops can be selected in a separately prepared
model configuration using the same shared weights. This is a load-time model
configuration, not a per-request switch. For the 72-layer Dory layout, two and
four loops have 56 and 96 logical attention cache slots respectively. These
counts do not establish prefix-cache persistence, batching or long-context
correctness. The PR's `last_loop` mode is not validated or recommended here.

The implementation in `omlx/patches/dory/dory_model.py` is the unchanged
MIT-licensed upstream model file (SHA-256
`897a5d79d13b49258b8d0ee4311ff0c374f596ac97f33f4899ed88742859586a`);
the upstream MIT notice is in `omlx/patches/dory/LICENSE`. oMLX registers it
under the mlx-lm architecture namespace before normal text-model loading. It
retains the pinned mlx-lm dependency. A Dory-only tokenizer load fallback handles
the known Transformers `layer_types` validation failure for local tokenizer
files; other errors propagate. It preserves the mlx-lm tokenizer wrapper,
model EOS IDs and template handling without changing dependency source files.

The model implementation and fallback require the existing MLX, mlx-lm,
Transformers and tokenizers dependencies; no new package is required. This
branch has not been tested with real weights or a running server. Prefix-cache
reuse/persistence, streaming, concurrent requests, long context, numerical
parity and four-loop serving need separate real-model verification.
