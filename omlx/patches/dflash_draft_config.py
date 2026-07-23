# SPDX-License-Identifier: Apache-2.0
"""Normalize transformers 5.x-nested draft configs for dflash-mlx (issue #2317).

``dflash_mlx.runtime.loading.load_draft_bundle`` -> ``mlx_lm.utils.load_model``
-> ``DFlashDraftModelArgs.from_dict(config)``. ``from_dict`` filters the raw
config dict down to the dataclass's annotated fields and calls
``cls(**filtered)``. ``rope_theta: float`` and ``block_size: int`` are
required positional fields with no defaults.

z-lab retrained their DFlash draft checkpoints in June 2026 with
transformers 5.7-style configs: ``rope_theta`` moved from the config root
into a nested ``rope_parameters`` object, and ``block_size`` moved from the
root into the existing ``dflash_config`` object. Neither key survives
``from_dict``'s field filter at its new nested location, so construction
fails with
``TypeError: DFlashDraftModelArgs.__init__() missing 2 required positional
arguments: 'rope_theta' and 'block_size'``, and omlx's engine_pool logs
"DFlash start failed ... Falling back to vlm engine" instead of running the
speculative draft. Upstream dflash-mlx (pin 9ca0028, v0.1.10) has no fix.

This module hoists the nested values back to the config root before
``DFlashDraftModelArgs.from_dict`` runs, so both the new (nested) and old
(root-level) config layouts construct successfully:
  - ``rope_parameters.rope_theta`` -> root ``rope_theta`` (transformers 5.x
    folds ``rope_theta`` into ``rope_parameters``, the successor of the
    older ``rope_scaling`` dict).
  - ``rope_parameters`` with a non-"default" ``rope_type``/``type`` -> root
    ``rope_scaling`` (minus ``rope_theta``), since the draft model consumes
    yarn-style scaling via ``mlx_lm``'s ``initialize_rope(...,
    scaling_config=args.rope_scaling)``.
  - ``dflash_config.block_size`` -> root ``block_size``.

Hoisting is fill-only per key: a root-level key that is set (non-null)
always wins over its nested counterpart, and an explicit ``null`` counts
as missing (transformers 4.x-derived configs commonly serialize
``"rope_scaling": null``). Old-layout configs pass through unchanged. One
deliberate behavior change for hybrid configs: a config that sets root
``rope_theta``/``block_size`` but still carries a non-"default"
``rope_parameters`` and no root ``rope_scaling`` (e.g. a checkpoint
hand-patched per the issue #2317 workaround) now gains the scaling its
config declares, where it previously loaded with unscaled RoPE. A config
that is missing a key at both the root and its nested location still
fails loudly in ``from_dict`` exactly as it does today — this module only
widens which configs are recognized, it never invents values.
"""

from __future__ import annotations

import logging
from typing import Any

from dflash_mlx.model import DFlashDraftModelArgs

logger = logging.getLogger(__name__)


def normalize_dflash_draft_config(config: dict[str, Any]) -> dict[str, Any]:
    """Hoist transformers 5.x-nested draft config keys back to the root.

    Pure function: never mutates ``config``. Returns a shallow copy with
    missing root-level ``rope_theta`` / ``rope_scaling`` / ``block_size``
    filled in from their transformers 5.x nested locations
    (``rope_parameters`` and ``dflash_config`` respectively). A root key
    that is set (non-null) is never overwritten; an explicit ``null``
    counts as missing.
    """
    result = dict(config)
    hoisted: list[str] = []

    rope_parameters = config.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        if (
            result.get("rope_theta") is None
            and rope_parameters.get("rope_theta") is not None
        ):
            result["rope_theta"] = rope_parameters["rope_theta"]
            hoisted.append("rope_theta<-rope_parameters")
        if result.get("rope_scaling") is None:
            # Mirror initialize_rope's key precedence exactly ("type" first,
            # then "rope_type") so the hoist decision can never disagree
            # with how mlx_lm reads the synthesized dict.
            rope_type = (
                rope_parameters.get("type")
                or rope_parameters.get("rope_type")
                or "default"
            )
            if rope_type != "default":
                result["rope_scaling"] = {
                    key: value
                    for key, value in rope_parameters.items()
                    if key != "rope_theta"
                }
                hoisted.append("rope_scaling<-rope_parameters")

    dflash_config = config.get("dflash_config")
    if (
        isinstance(dflash_config, dict)
        and result.get("block_size") is None
        and dflash_config.get("block_size") is not None
    ):
        result["block_size"] = dflash_config["block_size"]
        hoisted.append("block_size<-dflash_config")

    if hoisted:
        logger.info(
            "DFlash draft config: hoisted %s to config root (issue #2317)",
            ", ".join(hoisted),
        )

    return result


def install_dflash_draft_config_normalizer() -> None:
    """Wrap ``DFlashDraftModelArgs.from_dict`` to normalize nested configs.

    Passes the incoming params dict through
    ``normalize_dflash_draft_config`` before the original ``from_dict``
    runs, so transformers 5.x-nested draft configs (``rope_theta`` under
    ``rope_parameters``, ``block_size`` under ``dflash_config``) construct
    successfully instead of raising a missing-positional-argument
    ``TypeError`` (issue #2317).

    Idempotent — only wraps once per process.
    """
    if getattr(
        DFlashDraftModelArgs, "_omlx_draft_config_normalizer_installed", False
    ):
        return

    original = DFlashDraftModelArgs.from_dict.__func__

    def wrapped_from_dict(
        cls: type[DFlashDraftModelArgs], params: dict[str, Any]
    ) -> DFlashDraftModelArgs:
        return original(cls, normalize_dflash_draft_config(params))

    DFlashDraftModelArgs.from_dict = classmethod(wrapped_from_dict)
    DFlashDraftModelArgs._omlx_draft_config_normalizer_installed = True
    logger.info("DFlash draft config normalizer installed (issue #2317)")
