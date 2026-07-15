# SPDX-License-Identifier: Apache-2.0
"""Hidden-state tap wrapper for the Bonsai 27B Qwen3.5 hybrid target model.

The Bonsai 27B model is a Qwen3.5 VLM (loaded via mlx-vlm) with a hybrid
SSM + full-attention architecture (48 Gated Delta Net layers + 16 full-attention).

mlx-dspark's generic ``_run_mlxlm`` tap explicitly rejects hybrid models:
it uses a single causal mask and assumes all layers are full-attention.
For Bonsai we instead use the **native mlx-vlm capture mechanism**:

    LanguageModel.__call__(inputs=ids, cache=cache, capture_layer_ids=tap)

which returns ``LanguageModelOutput.hidden_states`` (a list of [B, L, H] tensors
at each tapped layer). This is the correct path because:
  1. The model handles SSM / full-attention masks internally per-layer.
  2. The capture hook exists in mlx-vlm's Qwen3.5 implementation.
  3. The verify path activates target_verify=True for the GDN layers
     (correct — DSpark verify rounds set this flag anyway).

BonsaiTarget exposes the same interface as mlx-dspark's Target:
  make_cache()                           → list of per-layer cache entries
  run(ids, cache, tap) → (logits, fused) → forward with hidden-state capture
  plain(ids, cache)    → logits          → no-capture decode step
"""

from __future__ import annotations

import mlx.core as mx


class BonsaiTarget:
    """Hidden-state tap and cache wrapper for the Bonsai 27B Qwen3.5 VLM target.

    Parameters
    ----------
    model:
        The loaded mlx-vlm Qwen3.5 model (has ``model.language_model``).
    tokenizer:
        The model's tokenizer (stored for caller convenience).
    """

    def __init__(self, model, tokenizer):
        if not hasattr(model, "language_model"):
            raise ValueError(
                "BonsaiTarget requires an mlx-vlm model with a .language_model attribute. "
                "Make sure the model is loaded via mlx-vlm (e.g. mlx_vlm.load)."
            )
        self.model = model
        self.tokenizer = tokenizer
        self._lm = model.language_model

    # -----------------------------------------------------------------------
    # Cache
    # -----------------------------------------------------------------------

    def make_cache(self):
        """Per-request KV + SSM cache list (one entry per model layer)."""
        return self._lm.make_cache()

    # -----------------------------------------------------------------------
    # Forward with hidden-state tap
    # -----------------------------------------------------------------------

    def run(self, ids: mx.array, cache, tap: list[int]):
        """ids [1, L] → (logits [1, L, V], fused_hidden [1, L, n_tap*H]).

        Calls ``language_model(capture_layer_ids=tap)`` which activates the
        native mlx-vlm hidden_sink capture. The SSM and full-attention layers
        are both handled correctly with their respective masks.
        """
        out = self._lm(inputs=ids, cache=cache, capture_layer_ids=tap)
        # out.hidden_states is a list of [1, L, H] tensors, one per tapped layer
        fused = mx.concatenate(out.hidden_states, axis=-1)  # [1, L, n_tap*H]
        return out.logits, fused

    # -----------------------------------------------------------------------
    # Plain forward (no capture)
    # -----------------------------------------------------------------------

    def plain(self, ids: mx.array, cache) -> mx.array:
        """Forward without hidden-state capture — used for the greedy baseline."""
        return self._lm(inputs=ids, cache=cache).logits

    # -----------------------------------------------------------------------
    # Compatibility shim for mlx-dspark generate functions
    # -----------------------------------------------------------------------

    @property
    def is_vlm(self) -> bool:
        return True

    def verify_tap(self) -> None:
        """No-op: native mlx-vlm capture hook needs no external verification."""
        pass
