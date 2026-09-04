# SPDX-License-Identifier: Apache-2.0
"""Local-checkpoint fallback for mlx-lm's ``sharded_load``.

``sharded_load`` gates on mapping every constructed parameter name through
``model.safetensors.index.json`` before downloading, but an architecture whose
``sanitize()`` fuses weights at load time (glm_moe_dsa's DSA indexer fusing
``wk`` and ``weights_proj`` into ``wk_weights_proj``, deepseek_v4 stacking its
expert biases) constructs names the index cannot contain, so the gate raises
"Pipeline loading is only supported for MLX converted models" for a checkpoint
the normal loader handles fine. The gate only exists to pick which files to
download; for a local directory there is nothing to download, so on that
specific failure this fallback repeats the load-and-shard tail on the local
path instead, and ``load_model`` performs the sanitize fusion exactly as it
does for a single process.

KEEP: re-verify against ``mlx_lm.utils.sharded_load`` on every pin bump; the
fallback mirrors its tail (load, shard, eval, synchronise).
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _local_sharded_load(
    repo: Any,
    pipeline_group: Any = None,
    tensor_group: Any = None,
    return_config: bool = False,
    *,
    tokenizer_config: dict[str, Any] | None = None,
    trust_remote_code: bool = False,
) -> Any:
    import mlx.core as mx
    from mlx_lm.utils import load_model, load_tokenizer

    model_path = Path(repo)
    model, config = load_model(
        model_path, lazy=True, strict=False, trust_remote_code=trust_remote_code
    )
    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config or {"trust_remote_code": True},
        eos_token_ids=config.get("eos_token_id", None),
    )
    if tensor_group is not None:
        model.shard(tensor_group)
    if pipeline_group is not None:
        model.model.pipeline(pipeline_group)
    # Lazy load plus a post-slice eval materialises only this rank's shard.
    mx.eval(model.parameters())
    # Synchronise processes to avoid timeout, as the original does.
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    if return_config:
        return model, tokenizer, config
    return model, tokenizer


def _wrap(original: Any) -> Any:
    @functools.wraps(original)
    def sharded_load_with_local_fallback(repo: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original(repo, *args, **kwargs)
        except ValueError as error:
            if "MLX converted models" not in str(error) or not Path(repo).is_dir():
                raise
            logger.info(
                "sharded_load rejected local checkpoint %s (sanitize-fused "
                "parameter names); loading the shard from the local path",
                repo,
            )
            return _local_sharded_load(repo, *args, **kwargs)

    sharded_load_with_local_fallback._omlx_local_fallback = True  # type: ignore[attr-defined]
    return sharded_load_with_local_fallback


def install_local_sharded_load_fallback() -> bool:
    """Wrap ``sharded_load`` in ``mlx_lm.utils`` and its server alias. Idempotent."""

    try:
        import mlx_lm.server as server_module
        import mlx_lm.utils as utils_module
    except ImportError:
        logger.debug("mlx_lm not importable - sharded_load fallback skipped")
        return False

    original = utils_module.sharded_load
    if getattr(original, "_omlx_local_fallback", False):
        return False
    wrapped = _wrap(original)
    utils_module.sharded_load = wrapped
    # The server binds the name at import time, so rebind it there as well.
    if getattr(server_module, "sharded_load", None) is original:
        server_module.sharded_load = wrapped
    logger.info("sharded_load local-checkpoint fallback installed")
    return True
