# SPDX-License-Identifier: Apache-2.0
"""Let mlx-vlm load a text-only Gemma 4 checkpoint.

mlx-vlm's ``gemma4_unified`` model is written to run without vision or audio::

    if config.vision_config is not None:
        self.vision_embedder = VisionEmbedder(config.vision_config)
        ...
    else:
        self.vision_embedder = None

and its forward path guards on the same attributes. That branch is
unreachable from a checkpoint, for three separate reasons in three places:

1. ``load_model`` runs ``config.setdefault("vision_config", {})`` (and the
   same for audio), so a checkpoint that omits the section still gets one.
2. ``ModelConfig.from_dict`` converts that ``{}`` into a fully-defaulted
   ``VisionConfig``, because the field's default is a factory rather than
   ``None`` despite being typed ``Optional``.
3. ``update_module_configs`` then re-applies it from the raw dict, so
   correcting the dataclass afterwards is not enough.

The result is that a text-only Gemma 4 checkpoint builds a patch embedder and
two projections it has no weights for, and ``load_weights(strict=True)`` fails
on those tensors. oMLX then drops the VLM engine and serves the model through
mlx-lm instead.

Simply setting the sections to ``None`` does not work either: ``load_model``
reads ``config.get("vision_config", {}).get("skip_vision", False)``, which
assumes a dict.

So this marks the absent modalities in the config and intercepts both
inflation points, leaving the dict shape mlx-vlm's own plumbing expects. The
marker is set by the loader that inspected the shards -- see
``omlx.engine.vlm``, which decides from weight prefixes rather than from the
config, so an orphaned section and a missing one are handled the same way.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Key the shard inspection writes into the raw config dict. Unknown keys are
# dropped by ``BaseModelConfig.from_dict``, so carrying it there is harmless.
ABSENT_MARKER = "_omlx_absent_modalities"

_applied = False


def mark_absent(config: dict, modality: str) -> None:
    """Record that ``modality`` has no weights in this checkpoint."""
    absent = set(config.get(ABSENT_MARKER) or ())
    absent.add(modality)
    config[ABSENT_MARKER] = sorted(absent)


def _absent(config: Any) -> set[str]:
    if isinstance(config, dict):
        return set(config.get(ABSENT_MARKER) or ())
    return set()


def apply() -> bool:
    """Patch the two config-inflation points. Idempotent."""
    global _applied
    if _applied:
        return True

    try:
        import mlx_vlm.utils as vlm_utils
        from mlx_vlm.models.gemma4_unified import config as unified_config
    except Exception as exc:  # pragma: no cover - depends on mlx-vlm layout
        logger.debug("mlx-vlm gemma4 text-only patch unavailable: %s", exc)
        return False

    original_update = vlm_utils.update_module_configs

    def update_module_configs(model_config, model_class, config, modules):
        """Skip the modules whose weights the checkpoint does not carry."""
        absent = _absent(config)
        if absent:
            modules = [m for m in modules if m not in absent]
        return original_update(model_config, model_class, config, modules)

    vlm_utils.update_module_configs = update_module_configs

    original_from_dict = unified_config.ModelConfig.from_dict

    @classmethod
    def from_dict(cls, params):
        built = original_from_dict.__func__(cls, params)
        for modality in _absent(params):
            attr = f"{modality}_config"
            if hasattr(built, attr):
                setattr(built, attr, None)
        return built

    unified_config.ModelConfig.from_dict = from_dict

    _applied = True
    logger.info("mlx-vlm Gemma 4 text-only load patch applied")
    return True
