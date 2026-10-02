# SPDX-License-Identifier: Apache-2.0
"""``mlx_lm.models.minimax_m3_vl`` — the shape mlx-lm expects, oMLX's model.

Registered into ``sys.modules`` under the mlx-lm name so
``importlib.import_module(f"mlx_lm.models.{model_type}")`` in
``mlx_lm.utils._get_classes`` resolves it. Nothing here reimplements MiniMax:
the layers, attention, MoE routing and KV cache all come from the vendored
mlx-vlm implementation oMLX already uses to serve this model on one Mac. This
only adapts the calling convention.

Two differences have to be bridged.

**Return type.** mlx-lm models return a logits array; the mlx-vlm model returns
a ``LanguageModelOutput``. The wrapper unwraps it.

**Layer access.** mlx-lm reaches ``model.layers`` to size the KV cache and to
report the stage, while the mlx-vlm tree keeps them at
``model.language_model.model.layers``. The wrapper forwards, so a pipeline
stage can be read without the caller knowing which tree it is holding.

That forwarding must be a *property*, not a stored reference. An MLX
``Module`` is a dict, and a list assigned to an attribute is stored in that
dict — so ``self.layers = inner...layers`` gave the wrapper its own entry
pointing at the complete list object. ``pipeline()`` rebinds the *inner*
model's ``layers`` to a shorter list; the wrapper's dict entry still held the
original, so every layer stayed reachable from ``model.parameters()`` and
mlx-lm loaded the whole model on every rank — measured at 1.00x the full model
on a rank that should have held half, ~225 GiB per Mac on MiniMax-M3. A
property owns no dict entry and cannot go stale. The same pattern is what the
vendored ``mlx_vlm.models.minimax_m3.Model`` uses.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.mlx_vlm_minimax_m3_compat import (
    apply_mlx_vlm_minimax_m3_compat_patch,
)

# The vendored implementation is the model; import it through the same compat
# patch the single-node path uses, so both serve identical weights.
apply_mlx_vlm_minimax_m3_compat_patch()

# Absolute, not relative: this file is registered as ``mlx_lm.models.minimax_m3_vl``
# with ``__package__ = "mlx_lm.models"``, so a relative import would look for
# ``mlx_lm.models.pipeline_patch`` and fail.
from omlx.patches.minimax_m3_mlx_lm.pipeline_patch import (  # noqa: E402
    apply_minimax_m3_pipeline_patch,
)

# Must run before any MiniMaxM3Model is constructed: it installs the pipelined
# forward and the pipeline() contract mlx-lm gates on.
apply_minimax_m3_pipeline_patch()

from mlx_vlm.models.minimax_m3 import Model as _VendoredModel  # noqa: E402
from mlx_vlm.models.minimax_m3 import ModelConfig as _VendoredConfig  # noqa: E402

# mlx-lm names the config class ModelArgs. ``from_dict`` already folds
# ``text_config`` up to the top level, which is where MiniMax keeps its
# language-model dimensions.
ModelArgs = _VendoredConfig

# Pipelining is installed by patching the vendored class, so a source scan of
# this file would not find it. Declared here for the planner's capability
# check — see omlx/cluster/planner.py:_declares_pipeline.
SUPPORTS_PIPELINE = True
# The transformer class lives in the vendored mlx-vlm module, so the cluster
# capability probe cannot discover its marked method by scanning this wrapper
# module. Declare the same explicit contract here, adjacent to the adapter
# whose ``model`` property exposes that transformer to mlx-lm.
HONORS_PIPELINE_ASSIGNMENT = True


class Model(nn.Module):
    """MiniMax-M3, presented the way mlx-lm expects to hold a model."""

    # The distributed generation loop may ask non-coordinator ranks to advance
    # their local stage and KV cache without constructing vocabulary logits.
    # Keep this an explicit adapter contract: forwarding ``skip_logits`` through
    # an arbitrary model would otherwise be an unsafe, version-dependent guess.
    _omlx_supports_rank_zero_logits = True

    def __init__(self, args: Any) -> None:
        super().__init__()
        self.args = args
        self.model_type = getattr(args, "model_type", "minimax_m3_vl")
        self.inner = _VendoredModel(args)

    # -- what mlx-lm calls ---------------------------------------------------

    @property
    def layers(self) -> Any:
        """The list the vendored tree is executing, right now.

        Deliberately read-only and deliberately not stored: see the module
        docstring. ``pipeline()`` replaces the inner list object, so anything
        that captured the old one keeps the whole model alive. mlx-lm never
        assigns here — it calls ``model.model.pipeline(group)`` and the
        assignment happens inside, on the inner model (``mlx_lm/utils.py``
        ``sharded_load``). An assignment that did arrive raises rather than
        silently splitting the wrapper from the tree that runs.
        """

        return self.inner.language_model.model.layers

    def __call__(
        self,
        inputs: mx.array,
        cache: Any = None,
        mask: Any = None,
        skip_logits: bool = False,
        **kwargs: Any,
    ) -> mx.array:
        out = self.inner(
            inputs,
            mask=mask,
            cache=cache,
            skip_logits=skip_logits,
            **kwargs,
        )
        # LanguageModelOutput in the mlx-vlm tree; a bare array if that ever
        # changes. Both are accepted rather than assuming one.
        return getattr(out, "logits", out)

    @property
    def _omlx_output_vocab_size(self) -> int:
        """Vocabulary width for rank-local placeholder log probabilities."""

        return int(self.args.vocab_size)

    @property
    def language_model(self) -> Any:
        return self.inner.language_model

    @property
    def model(self) -> Any:
        """The transformer itself.

        mlx-lm decides a model can be pipelined with
        ``hasattr(model, "model") and hasattr(model.model, "pipeline")``, so
        this must expose the class carrying ``pipeline()`` — not the mlx-vlm
        wrapper above it.
        """

        return self.inner.language_model.model

    @property
    def head_dim(self) -> Any:
        return getattr(self.args, "head_dim", None)

    @property
    def n_kv_heads(self) -> Any:
        return getattr(self.args, "num_key_value_heads", None)

    def sanitize(self, weights: dict) -> dict:
        """Weight-name fixes, delegated — this file must not own that logic."""

        sanitized = self.inner.sanitize(weights)
        # The vendored tree lives under ``inner.``; mlx-lm loads against this
        # module's own parameter names, so re-root what sanitize produced.
        return {f"inner.{key}": value for key, value in sanitized.items()}

    def make_cache(self) -> Any:
        maker = getattr(self.inner, "make_cache", None)
        if maker is not None:
            return maker()
        language = getattr(self.inner, "language_model", None)
        maker = getattr(language, "make_cache", None)
        return maker() if maker is not None else None
