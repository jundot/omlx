# oMLX mlx-vlm muse_spark (Muse Spark Open)

oMLX implementation, not a copy of an upstream PR. It supports text and image
input. Video and audio input raise an error.

- The text tower is `omlx.patches.muse_spark.muse_spark_model`, the same module
  that serves the LLM path as `mlx_lm.models.muse_spark`. Keep one copy so both
  paths produce the same logits.
- The vision tower and the image processor are the vendored Muse Glimmer ones
  (`../muse_glimmer`). Muse Spark Open differs in one place: patch `(h, w)` reads the
  position table at `(w, h)` (`vision.py`).
- ModelOpt NVFP4 experts load packed. Their per-expert `weight_scale_2` tensors
  become `switch_mlp.{gate,up,down}_proj.global_scale` (the amax, as in mlx-lm)
  and stay FP32.

## When mlx-vlm gains muse_spark upstream

The vendor path is searched first, so this package keeps winning. Before
removing it, check that upstream:

- [ ] matches the transformers reference at FP32 on real weights (text logits
      and image features), including the transposed position-table lookup
- [ ] loads ModelOpt NVFP4 experts without folding `weight_scale_2` into the
      E4M3 block scales (the scales are about 1e-5 and underflow E4M3)
- [ ] exposes `encode_image` for the oMLX vision feature cache
- [ ] registers `prompt_utils.MODEL_CONFIG["muse_spark"]`
