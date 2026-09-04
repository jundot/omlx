# SPDX-License-Identifier: Apache-2.0
"""Ling's trained per-layer SwiGLU clamp for ``bailing_hybrid``.

Ling-3.0-flash is trained with a clamped SwiGLU on its late layers and ships
the limits in ``config.json``::

    expert_swiglu_limit_list        layers 35-41 = 4
    share_expert_swiglu_limit_list  layers 34-39 = 5, layers 40-41 = 7

inclusionAI's vLLM port implements this (``_get_layer_swiglu_limit`` +
``SwigluStepAndMul``); the HF transformers reference does not, and neither
does the ``bailing_hybrid`` module mlx-lm resolves. Running unclamped costs
real accuracy — measured on Ling-3.0-flash (HumanEval, 164 problems, greedy,
thinking off): **88.41% clamped vs 71.34% unclamped**, +17.07pp for 28
problems, at no measurable runtime cost (the clamp is two elementwise ops).

Two code paths need covering, because the live ``mlx_lm.models.bailing_hybrid``
can come from either place:

* oMLX's vendored copy in this package, which implements the clamp natively —
  ``ensure_swiglu_clamp`` detects that and does nothing;
* an mlx-lm build that already ships ``bailing_hybrid``, in which case the
  clamp is installed onto the live classes here.

Semantics match vLLM exactly::

    silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_LIMIT_ATTR = "_omlx_swiglu_limit"


def layer_swiglu_limit(limit_list: Any, layer_idx: int) -> float | None:
    """Per-layer limit, or None when the list is absent, short, or zero."""
    if not limit_list or layer_idx >= len(limit_list):
        return None
    limit = limit_list[layer_idx]
    if limit in (None, 0):
        return None
    return float(limit)


def _install(module: Any) -> None:
    """Define the clamped activation and teach ``MLP`` to honour a limit."""
    import mlx.core as mx
    import mlx.nn as nn

    if not hasattr(module, "clamped_swiglu"):

        def clamped_swiglu(gate, x, limit):
            return mx.minimum(nn.silu(gate), limit) * mx.clip(x, -limit, limit)

        module.clamped_swiglu = clamped_swiglu

    if not hasattr(module, "ClampedSwiGLU"):

        class ClampedSwiGLU(nn.Module):
            """SwitchGLU-signature activation: ``__call__(x_up, x_gate)``.

            mlx-lm's ``SwitchGLU`` calls ``activation(x_up, x_gate)`` and its
            stock ``SwiGLU`` returns ``swiglu(gate, x)``, i.e. silu is applied
            to the *gate*. The clamp must match that or it lands on the wrong
            tensor.
            """

            def __init__(self, limit: float):
                super().__init__()
                self.limit = float(limit)

            def __call__(self, x, gate):
                return module.clamped_swiglu(gate, x, self.limit)

        module.ClampedSwiGLU = ClampedSwiGLU

    mlp_cls = module.MLP
    if not getattr(mlp_cls.__dict__.get("__call__"), "_omlx_clamp", False):
        stock_swiglu = module.swiglu

        def patched_call(self, x):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            limit = getattr(self, _LIMIT_ATTR, None)
            if limit:
                return self.down_proj(module.clamped_swiglu(gate, up, limit))
            return self.down_proj(stock_swiglu(gate, up))

        patched_call._omlx_clamp = True
        mlp_cls.__call__ = patched_call


def bind_limits(module: Any, model: Any, config: Any) -> int:
    """Bind per-layer limits onto a constructed model. Returns paths clamped."""
    expert_list = getattr(config, "expert_swiglu_limit_list", None)
    shared_list = getattr(config, "share_expert_swiglu_limit_list", None)
    if not expert_list and not shared_list:
        return 0

    clamped = 0
    for idx, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        switch_mlp = getattr(mlp, "switch_mlp", None)
        if switch_mlp is not None:
            limit = layer_swiglu_limit(expert_list, idx)
            if limit:
                switch_mlp.activation = module.ClampedSwiGLU(limit)
                clamped += 1
            shared = getattr(mlp, "shared_experts", None)
            if shared is not None:
                limit = layer_swiglu_limit(shared_list, idx)
                if limit:
                    setattr(shared, _LIMIT_ATTR, limit)
                    clamped += 1
    return clamped


def ensure_swiglu_clamp(module: Any) -> bool:
    """Make ``module`` apply Ling's clamp, whichever build it came from.

    Returns True when the clamp had to be installed (i.e. the module did not
    already implement it natively).
    """
    if getattr(module, "_omlx_swiglu_clamp_native", False):
        return False
    # The vendored copy implements the clamp in-source; nothing to do.
    if hasattr(module, "ClampedSwiGLU") and hasattr(module, "layer_swiglu_limit"):
        module._omlx_swiglu_clamp_native = True
        return False

    if getattr(module, "_omlx_swiglu_clamp_installed", False):
        return False

    _install(module)

    args_cls = module.ModelArgs
    if not getattr(args_cls, "_omlx_clamp_args", False):
        original_from_dict = args_cls.from_dict.__func__

        def patched_from_dict(cls, params):
            args = original_from_dict(cls, params)
            # BaseModelArgs drops undeclared keys, so carry these by hand.
            args.expert_swiglu_limit_list = params.get("expert_swiglu_limit_list")
            args.share_expert_swiglu_limit_list = params.get(
                "share_expert_swiglu_limit_list"
            )
            return args

        args_cls.from_dict = classmethod(patched_from_dict)
        args_cls._omlx_clamp_args = True

    model_cls = module.Model
    if not getattr(model_cls, "_omlx_clamp_init", False):
        original_init = model_cls.__init__

        def patched_init(self, config):
            original_init(self, config)
            clamped = bind_limits(module, self, config)
            if clamped:
                logger.info(
                    "Ling SwiGLU clamp applied to %d expert paths "
                    "(model is trained with it; mlx-lm's bailing_hybrid "
                    "does not implement it)",
                    clamped,
                )

        model_cls.__init__ = patched_init
        model_cls._omlx_clamp_init = True

    module._omlx_swiglu_clamp_installed = True
    return True


__all__ = [
    "bind_limits",
    "ensure_swiglu_clamp",
    "layer_swiglu_limit",
]
