# SPDX-License-Identifier: Apache-2.0
"""DFlash target loader for the validated mixed ModelOpt Qwen3.8 checkpoint.

The exact-weight bridge in :mod:`qwen38_modelopt_mixed` loads the published
``unsloth/Qwen3.8-27B-NVFP4`` checkpoint through mlx-vlm. DFlash, however,
loads its target through ``dflash_mlx.runtime.loading``, whose module-level
``load`` symbol points directly at ``mlx_lm.utils.load``. That bypasses the
bridge and lets mlx-lm's generic compressed-tensors path see the source
``weight_packed`` / global-scale sidecars, so strict loading fails before
DFlash can install its target hooks.

This module provides the corresponding text-only mlx-lm route. It reuses the
same strict config gate, exact packed-weight transform and custom quantized
linear carrier as the VLM bridge, while leaving DFlash's own target-ops,
rollback hooks and verification setup untouched.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

from . import qwen38_modelopt_mixed as mixed

logger = logging.getLogger(__name__)

_DFLASH_LOADER_MARKER = "_omlx_qwen38_modelopt_loader"


def _local_supported_config(model_ref: str | Path) -> dict[str, Any] | None:
    """Return the validated local config, or ``None`` for every other target.

    oMLX resolves library entries to local checkpoint directories before the
    DFlash engine starts. Keeping this probe local-only is deliberate: merely
    installing the compatibility wrapper must not trigger a Hub download for
    unrelated DFlash targets.
    """

    try:
        model_path = Path(model_ref).expanduser()
    except TypeError:
        return None
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return config if mixed.is_supported_config(config) else None


def is_supported_model_ref(model_ref: str | Path) -> bool:
    """Whether *model_ref* is the validated local Qwen3.8 ModelOpt target."""

    return _local_supported_config(model_ref) is not None


def load_lm(
    path_or_hf_repo: str | Path,
    tokenizer_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    adapter_path: str | None = None,
    lazy: bool = False,
    return_config: bool = False,
    revision: str | None = None,
    trust_remote_code: bool = False,
):
    """Load the validated checkpoint through mlx-lm without re-quantizing it.

    The signature mirrors ``mlx_lm.utils.load`` because DFlash keeps a direct
    reference to that function. The target remains text-only, exactly like the
    normal mlx-lm Qwen3.8 route: vision tensors and the target checkpoint's MTP
    subtree are sanitized away before strict loading.
    """

    import mlx_lm.utils as lm_utils
    from mlx_lm.models import qwen3_5

    # Resolve once so the config gate and the actual loader inspect the same
    # checkpoint revision. _download is part of the pinned mlx-lm loader path;
    # when *path_or_hf_repo* is already local this is just a Path conversion.
    model_path = lm_utils._download(str(path_or_hf_repo), revision=revision)
    source_config = lm_utils.load_config(model_path)
    reported_config = copy.deepcopy(source_config)
    if model_config:
        reported_config.update(copy.deepcopy(model_config))
    rules = mixed._rules_from_config(reported_config)

    base_model_class = qwen3_5.Model
    model_args_class = qwen3_5.ModelArgs

    class ExactModelOptModel(base_model_class):
        def __init__(self, args):
            super().__init__(args)
            count = mixed._replace_modelopt_modules(self, rules)
            logger.info("Bound %d exact-weight ModelOpt DFlash target modules", count)

        def sanitize(self, weights):
            # First let mlx-lm perform the normal Qwen3.8 text-only namespace,
            # MTP and norm/conv sanitization. Then map the surviving source
            # packed tensors into the exact MLX NVFP4/MXFP8 carriers.
            sanitized = super().sanitize(weights)
            return mixed.transform_weights_exact(sanitized)

    def get_model_classes(_config):
        return ExactModelOptModel, model_args_class

    # mlx-lm treats every compressed-tensors checkpoint as affine 4-bit by
    # default. Suppress only that generic conversion; ExactModelOptModel has
    # already installed the validated mixed NVFP4/MXFP8 module layout and its
    # sanitize method preserves the source codes/scales without re-quantizing.
    internal_model_config = copy.deepcopy(model_config) if model_config else {}
    internal_model_config["quantization"] = None
    internal_model_config["quantization_config"] = None
    internal_model_config["quantize_activations"] = False

    logger.info("Using exact-weight mixed ModelOpt DFlash loader for %s", model_path)
    model, _loaded_config = lm_utils.load_model(
        model_path,
        lazy=lazy,
        model_config=internal_model_config,
        get_model_classes=get_model_classes,
        trust_remote_code=trust_remote_code,
    )
    if adapter_path is not None:
        model = lm_utils.load_adapters(model, adapter_path)
        model.eval()

    tokenizer = lm_utils.load_tokenizer(
        model_path,
        tokenizer_config,
        eos_token_ids=reported_config.get("eos_token_id", None),
    )
    if return_config:
        return model, tokenizer, reported_config
    return model, tokenizer


def install_dflash_modelopt_loader() -> bool:
    """Route only the validated Qwen3.8 ModelOpt target through ``load_lm``.

    ``dflash_mlx.runtime.loading.load_target_bundle`` looks up its module-level
    ``load`` global at call time, so replacing that one symbol is enough to
    preserve all of dflash-mlx's own target resolution and speculative hook
    installation. The wrapper is process-global but narrowly gated and
    idempotent; every unrelated model delegates byte-for-byte to the loader
    that was present when this wrapper was installed.
    """

    try:
        from dflash_mlx.runtime import loading as dflash_loading
    except ImportError:
        return False

    current_load = dflash_loading.load
    if getattr(current_load, _DFLASH_LOADER_MARKER, False):
        return False

    original_load = current_load

    def load_with_modelopt_target(path_or_hf_repo, *args, **kwargs):
        if is_supported_model_ref(path_or_hf_repo):
            return load_lm(path_or_hf_repo, *args, **kwargs)
        return original_load(path_or_hf_repo, *args, **kwargs)

    setattr(load_with_modelopt_target, _DFLASH_LOADER_MARKER, True)
    setattr(load_with_modelopt_target, "_omlx_original_load", original_load)
    dflash_loading.load = load_with_modelopt_target
    logger.debug("Qwen3.8 ModelOpt DFlash target loader installed")
    return True
