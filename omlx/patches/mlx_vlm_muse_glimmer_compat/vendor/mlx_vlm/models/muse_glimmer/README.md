# Vendored mlx-vlm muse_glimmer (Meta Muse Glimmer 30B)

Source: Blaizzy/mlx-vlm PR #1838 head (model package + prompt_utils
registration), plus the quantization fix from PR #1839. Vendored because
both PRs are newer than oMLX's mlx-vlm pin (`78b96eb`).

## oMLX deltas against the PR head (marked with `oMLX:` comments)

1. `language.py` — `initialize_rope` is imported from
   `mlx_lm.models.rope_utils` instead of `..rope_utils`: the pinned
   mlx-vlm rope_utils only carries MRoPE machinery. The mlx-lm signature
   matches the upstream call exactly.
2. `language.py` — `QuantizedNormedEmbedding` + `NormedEmbedding.to_quantized`
   ported from PR #1839 (applied semantically; the PR's diff context
   predates the PR #1838 head). Without it, quantizing `embed_tokens`
   silently drops the weightless `embed_norm` and logits collapse.
3. `muse_glimmer.py` — public `encode_image()` alias for `_encode_image`
   so oMLX's vision feature cache can precompute/persist image features
   (`engine/vlm.py::_compute_vision_features` strategy 1).

`../activations.py` is the same shared shim the inkling vendor carries
(the pin has no `mlx_vlm.models.activations`); the two copies must stay
byte-identical since whichever is imported first wins in `sys.modules`.

## Pin-bump checklist

When bumping the mlx-vlm pin past the upstream merge of PR #1838, the
upstream module wins automatically (vendor path is searched last). Before
deleting this vendor package, verify upstream carries:

- [ ] the PR #1839 fix (`QuantizedNormedEmbedding`); if absent, quantized
      checkpoints (oQ and community) load with silently broken logits
- [ ] an `encode_image` public method (or update
      `_compute_vision_features` probing); if absent, the vision feature
      cache silently no-ops for muse_glimmer
- [ ] `initialize_rope` importable at the new pin (upstream imports it
      from `..rope_utils`, which needs a newer mlx-vlm than `78b96eb`)
- [ ] `prompt_utils.MODEL_CONFIG["muse_glimmer"]` registered upstream
- [ ] a real-model smoke test (text + image + quantized load)
