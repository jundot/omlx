# Vendored GLM-5.3-Flash support

This package is based on the GLM-5 Next model implementation merged by
[`Blaizzy/mlx-vlm#2030`](https://github.com/Blaizzy/mlx-vlm/pull/2030) at
`fa27a9a692770c39fdf57b9a985fad084a90aec2`. It is vendored because oMLX's
pinned mlx-vlm release predates that model family.

The text and vision paths also carry the non-speculative correctness fixes from
[`Blaizzy/mlx-vlm#2044`](https://github.com/Blaizzy/mlx-vlm/pull/2044), plus
oMLX runtime integration:

- pooled indexer caches that support continuous batching and cache lifecycle
  operations;
- GLM-5.2 DSA, sparse MLA, MoE weighted-sum, and gated-delta kernels where the
  GLM-5.3 tensor contracts match;
- shared affine prefill QMM kernels for supported Q2/Q4/Q5/Q6/Q8 projections;
- a torch-free NumPy/Pillow image processor compatible with the official
  checkpoint metadata and oMLX's pinned Transformers release.

Lightning MTP is intentionally not included in this compatibility layer. The
base model drops `mtp.*` tensors during sanitization; GLM-5.3 Lightning MTP can
be added independently after its recurrent-state rollback and batched cache
semantics are validated.
