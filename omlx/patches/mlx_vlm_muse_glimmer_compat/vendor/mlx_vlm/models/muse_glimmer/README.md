# Vendored mlx-vlm muse_glimmer (Meta Muse Glimmer 30B)

Source: Blaizzy/mlx-vlm PR #1838 head (model package + prompt_utils
registration; includes the CenteredRMSNorm FP32 operation-order fix from
commits edfb0ef1 + 6242d295), plus upstream #1848 (reasoning config,
`_prepare_mlp_input`/`_finish_mlp` norm fusions, FP32 `qk_scale_factor`
math, plain `nn.Embedding` + separate weightless `embed_norm`) synced in
2026-08-16 (vendor commit 3a82f856). Vendored because these PRs are newer
than oMLX's mlx-vlm pin (`78b96eb`).

The CenteredRMSNorm implementation is duplicated in dflash-mlx's
`dflash_mlx/models/muse_glimmer.py`; the two must stay numerically
identical or DFlash verify logits drift from serving logits
(tests/test_dflash_muse_glimmer.py guards this). The FP32 qk_scale math
is mirrored in the dflash-mlx fork (commit cc57617) for the same reason.

## oMLX deltas against the PR head (marked with `oMLX:` comments)

1. `language.py` — `initialize_rope` is imported from
   `mlx_lm.models.rope_utils` instead of `..rope_utils`: the pinned
   mlx-vlm rope_utils only carries MRoPE machinery. The mlx-lm signature
   matches the upstream call exactly (this is also why upstream's
   `implementation="eager"` is NOT adopted — mlx-lm's initialize_rope
   has no such parameter).
2. `muse_glimmer.py` — public `encode_image()` alias for `_encode_image`
   so oMLX's vision feature cache can precompute/persist image features
   (`engine/vlm.py::_compute_vision_features` strategy 1).

Upstream #1848 replaced the PR #1839 NormedEmbedding quantization
workaround with the cleaner design we now carry: `embed_tokens` is a
plain `nn.Embedding` and `embed_norm` is a separate weightless module, so
quantizing the embedding can never drop the norm. `quant_predicate` was
removed along with the wrapper.

`../activations.py` is the same shared shim the inkling vendor carries
(the pin has no `mlx_vlm.models.activations`); the two copies must stay
byte-identical since whichever is imported first wins in `sys.modules`.

## Pin-bump checklist

When bumping the mlx-vlm pin past the upstream merge of PR #1838, the
upstream module wins automatically (vendor path is searched last). Before
deleting this vendor package, verify upstream carries:

- [ ] the #1848 embed design (`nn.Embedding` + separate `embed_norm`);
      if absent, quantized checkpoints (oQ and community) load with
      silently broken logits (this is what the PR #1839 wrapper guarded)
- [ ] an `encode_image` public method (or update
      `_compute_vision_features` probing); if absent, the vision feature
      cache silently no-ops for muse_glimmer
- [ ] `initialize_rope` importable at the new pin (upstream imports it
      from `..rope_utils`, which needs a newer mlx-vlm than `78b96eb`)
- [ ] `prompt_utils.MODEL_CONFIG["muse_glimmer"]` registered upstream
- [ ] a real-model smoke test (text + image + quantized load)
