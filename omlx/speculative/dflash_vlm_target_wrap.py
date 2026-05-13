"""mlx_vlm -> dflash_mlx Gemma4TargetOps adapter wrapper.

Non-destructive proxy view: wraps an mlx_vlm-loaded Gemma 4 model and
exposes the mlx_lm-shaped surface that ``dflash_mlx.engine.target_gemma4.
Gemma4TargetOps`` consumes.

Spike validation (tmp_spike/spike{1,2,3}_*.py @ 2026-05-11) confirmed:
  - DecoderLayer.__call__ signature is identical between mlx_vlm and mlx_lm
    Gemma 4 implementations.
  - install_verify_linears only touches QuantizedLinear modules; the vlm
    vision projector (plain nn.Linear) is left alone.
  - make_cache(enable_speculative_linear_cache=True) ignores the kwarg for
    Gemma 4 (it just calls ``wrapper.make_cache()``).

After cross-reading both source trees the surface drift is:

  1. ``lm.args.X``                          -> ``lm.config.X``           (rename)
  2. ``lm.args.num_kv_shared_layers``       -> covered by (1)
  3. ``inner._get_per_layer_inputs``        SIGNATURE MISMATCH:
        mlx_lm:   (input_ids, input_embeddings=None)   two args
        mlx_vlm:  (input_ids)                          one arg
     This is not just a rename - we adapt by dropping the embeddings arg.
     CAVEAT: dflash's input_ids=None / embeddings-only path (nearest-vocab
     reconstruction in mlx_lm) is unsupported here. Multimodal-embedding-
     fed forward will break with NotImplementedError - documented as a
     known limitation; only matters if Gemma4TargetOps is invoked with
     image-tower embeddings directly (Path A target=text only, so safe).
  4. ``inner._project_per_layer_inputs``    -> ``inner.project_per_layer_inputs``
     (pure rename - signatures match)
  5. ``lm.final_logit_softcapping``         attribute already exists on
     mlx_vlm LanguageModel; also reachable via args view (1).
  6. ``Gemma4TargetOps.supports_model``     returns False on vlm models
     (model_type == "gemma4", inner.embed_tokens does not exist on the
     vlm Model -> language_model.model.embed_tokens does). We bypass by
     constructing ``Gemma4TargetOps()`` directly instead of routing through
     ``resolve_target_ops`` dispatch.

Use:
    wrapped = DFlashVLMTargetWrapper(vlm_model)
    ops = Gemma4TargetOps()                       # skip dispatch
    ops.install_speculative_hooks(wrapped)        # no-op for Gemma 4
    cache = ops.make_cache(wrapped, enable_speculative_linear_cache=True)
"""
from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx


class _ArgsView:
    """Expose ``lm.args.X`` reading underneath from ``lm.config``."""

    __slots__ = ("_config",)

    def __init__(self, config: Any) -> None:
        object.__setattr__(self, "_config", config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._config, name)


class _InnerView:
    """Non-destructive proxy for ``LanguageModel.model`` (the Gemma4TextModel).

    Bridges the private-name variants and adapts the
    ``_get_per_layer_inputs`` signature to the mlx_lm two-arg form.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def _get_per_layer_inputs(
        self,
        input_ids: Optional[mx.array],
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        # mlx_vlm only knows how to derive per-layer inputs from input_ids.
        # The mlx_lm nearest-vocab reconstruction (input_ids=None path) is
        # not implemented in mlx_vlm; refuse rather than silently miscompute.
        if input_ids is None:
            raise NotImplementedError(
                "DFlashVLMTargetWrapper: mlx_vlm Gemma 4 cannot derive "
                "per-layer inputs from input_embeddings alone. Provide "
                "input_ids (text-only target path)."
            )
        return self._inner.get_per_layer_inputs(input_ids)

    @property
    def _project_per_layer_inputs(self):
        # Signature already matches (input_embeddings, per_layer_inputs=None).
        return self._inner.project_per_layer_inputs

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _LangModelView:
    """Non-destructive proxy for ``LanguageModel``; exposes ``args``."""

    __slots__ = ("_lm", "_inner_view", "_args_view")

    def __init__(self, lm: Any) -> None:
        object.__setattr__(self, "_lm", lm)
        object.__setattr__(self, "_inner_view", _InnerView(lm.model))
        object.__setattr__(self, "_args_view", _ArgsView(lm.config))

    @property
    def args(self) -> Any:
        return self._args_view

    @property
    def model(self) -> Any:
        return self._inner_view

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lm, name)


class DFlashVLMTargetWrapper:
    """Wrap an mlx_vlm Gemma 4 model so it presents an mlx_lm-shaped surface
    to ``dflash_mlx.engine.target_gemma4.Gemma4TargetOps``.
    """

    __slots__ = ("_vlm", "_lm_view")

    def __init__(self, vlm_model: Any) -> None:
        object.__setattr__(self, "_vlm", vlm_model)
        object.__setattr__(
            self, "_lm_view", _LangModelView(vlm_model.language_model)
        )

    @property
    def language_model(self) -> Any:
        return self._lm_view

    def __getattr__(self, name: str) -> Any:
        return getattr(self._vlm, name)
