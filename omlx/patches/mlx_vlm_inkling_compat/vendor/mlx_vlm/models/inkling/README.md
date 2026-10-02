# Vendored: mlx_vlm/models/inkling

Source: mlx-vlm PR #1756 head (`pcuenca/mlx-vlm@73e4cb6878691062fc55c1f969f0eedaa6ca3caf`),
which extends the base inkling package from PR #1637 (`Blaizzy/mlx-vlm@03f3603`)
for Inkling Small. MIT licensed like the rest of mlx-vlm.

Byte-identical to the PR head except for changes marked with `OMLX:`
comments in `language.py` (batched right-padded prefill support:
conv_mask wiring, lengths-aware conv state writes, per-sequence key
masking and log-tau positions, per-layer cache.advance; an explicit
error for configs missing `dense_intermediate_size`; MTP verify
rollback conv-input stashes, with an opt-in rolling window
`_omlx_stash_rows` for the MTP head-block caches).

The Lightning MTP runtime (`patches/mlx_vlm_mtp/inkling_vlm_runtime.py`)
deliberately does NOT follow upstream mlx-vlm's
`speculative/drafters/inkling_mtp` drafter: that reference only folds
block 0 and snapshot-restores per round, leaving depth blocks 1..7 with
empty caches (measured per-depth agreement 0.68/0.45/0.19/0.00). It
follows vLLM's Inkling MTP semantics instead (vllm-project/vllm#48768:
per-depth stateful caches, uniform-window chained passes, double-normed
embedding input), measured 0.87/0.76/0.67/0.61 with the same weights.

`processing_inkling.py` is NOT from mlx-vlm: it is a torch-free numpy/PIL
port of transformers 5.14's Inkling processors (Apache-2.0), needed
because oMLX pins transformers <5.13. Audio (dMel) extraction is not
implemented yet (text+vision first).

The sibling shims `../mlp.py`, `../switch_layers.py`, `../activations.py`
are byte-identical copies from the same PR head; they do not exist at the
oMLX mlx-vlm pin (`78b96eb`). The `__path__`-append install searches the
real package first, so a future pin bump that ships any of these upstream
automatically retires the vendored copy.

Retire this whole package when the mlx-vlm pin advances past PR #1756's
merge (and transformers reaches 5.14+ for the processors).
