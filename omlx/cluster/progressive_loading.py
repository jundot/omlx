# SPDX-License-Identifier: Apache-2.0
"""Progressive distributed model loading for the pinned MLX-LM server."""

from __future__ import annotations

import re
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from .tensor_strategies import apply_tensor_strategy, supports_model_type

_LAYER = re.compile(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)(?:\.|$)")
ProgressCallback = Callable[[dict[str, Any]], None]


def _layer_index(path: str) -> int | None:
    match = _LAYER.search(path)
    return int(match.group(1)) if match else None


def _eval_values(mx: Any, values: list[Any]) -> None:
    if values:
        mx.eval(*values)


def materialize_parameters_progressively(
    parameters: Any,
    *,
    mx_module: Any,
    tree_flatten: Callable[[Any], list[tuple[str, Any]]],
    progress: ProgressCallback | None = None,
) -> tuple[int, ...]:
    """Materialize fixed weights once and transformer weights one layer at a time."""

    fixed: list[Any] = []
    layers: dict[int, list[Any]] = {}
    for path, value in tree_flatten(parameters):
        index = _layer_index(path)
        if index is None:
            fixed.append(value)
        else:
            layers.setdefault(index, []).append(value)

    ordered = tuple(sorted(layers))
    if progress is not None:
        progress(
            {
                "phase": "materializing_fixed",
                "fixed_tensors": len(fixed),
                "layers_loaded": 0,
                "layers_total": len(ordered),
            }
        )
    _eval_values(mx_module, fixed)
    mx_module.clear_cache()
    for loaded, index in enumerate(ordered, start=1):
        _eval_values(mx_module, layers[index])
        mx_module.clear_cache()
        if progress is not None:
            progress(
                {
                    "phase": "materializing_layers",
                    "layer": index,
                    "layers_loaded": loaded,
                    "layers_total": len(ordered),
                }
            )
    return ordered


def progressive_sharded_load(
    repo: Any,
    pipeline_group: Any = None,
    tensor_group: Any = None,
    return_config: bool = False,
    *,
    tokenizer_config: dict[str, Any] | None = None,
    trust_remote_code: bool = False,
    progress: ProgressCallback | None = None,
    utils_module: Any = None,
    mx_module: Any = None,
) -> Any:
    """Pinned ``mlx_lm.utils.sharded_load`` with bounded materialization.

    The discovery/download behavior intentionally mirrors MLX-LM. The only
    semantic differences are progressive parameter evaluation and the explicit
    tensor strategy registry.
    """

    if utils_module is None:
        from mlx_lm import utils as utils_module
    if mx_module is None:
        import mlx.core as mx_module

    model_path = utils_module._download(
        repo,
        allow_patterns=[
            "*.json",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "tiktoken.model",
            "*.txt",
            "*.jsonl",
            "*.jinja",
        ],
    )
    from ..utils.model_loading import ensure_model_code_trusted

    metadata_config = utils_module.load_config(model_path)
    ensure_model_code_trusted(
        metadata_config,
        model_path=model_path,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = utils_module.load_tokenizer(
        model_path,
        tokenizer_config or {"trust_remote_code": trust_remote_code},
        eos_token_ids=metadata_config.get("eos_token_id", None),
    )
    model, config = utils_module.load_model(
        model_path,
        lazy=True,
        strict=False,
        trust_remote_code=trust_remote_code,
    )
    has_pipeline = hasattr(model, "model") and hasattr(model.model, "pipeline")
    has_native_tensor = callable(getattr(model, "shard", None))
    model_type = str(
        getattr(
            model,
            "model_type",
            getattr(getattr(model, "args", None), "model_type", ""),
        )
    )
    has_tensor = supports_model_type(
        model_type,
        native_shard=has_native_tensor,
    )
    if pipeline_group is not None and not has_pipeline:
        raise ValueError(
            "The model does not support pipelining but a pipeline_group was provided"
        )
    if tensor_group is not None and not has_tensor:
        raise ValueError(
            "The model does not support tensor parallelism but a tensor_group "
            "was provided"
        )
    if not has_pipeline and not has_tensor and tensor_group is None:
        raise ValueError("The model does not support any sharding")
    if pipeline_group is not None and tensor_group is not None:
        raise ValueError(
            "progressive loading does not combine pipeline and tensor groups"
        )
    if pipeline_group is tensor_group is None:
        group = mx_module.distributed.init()
        if has_tensor:
            tensor_group = group
        elif has_pipeline:
            pipeline_group = group

    if pipeline_group is not None:
        model.model.pipeline(pipeline_group)
        with utils_module.open(
            model_path / "model.safetensors.index.json",
            "r",
        ) as stream:
            weight_index = utils_module.json.load(stream)["weight_map"]
        local_files: set[str] = set()
        for key, _ in utils_module.tree_flatten(model.parameters()):
            file_name = weight_index.get(key)
            if file_name is None:
                raise ValueError(
                    "Pipeline loading is only supported for MLX converted models."
                )
            local_files.add(file_name)
        utils_module._download(repo, allow_patterns=local_files)
    else:
        utils_module._download(repo)

    model, _ = utils_module.load_model(
        model_path,
        lazy=True,
        strict=False,
        trust_remote_code=trust_remote_code,
    )
    if pipeline_group is not None:
        model.model.pipeline(pipeline_group)
        materialize_parameters_progressively(
            model.parameters(),
            mx_module=mx_module,
            tree_flatten=utils_module.tree_flatten,
            progress=progress,
        )
    elif tensor_group is not None:
        # Fixed embeddings/head weights are replicated. Materialize them first;
        # each strategy then materializes, shards, evaluates and releases one
        # transformer layer before touching the next.
        flat = utils_module.tree_flatten(model.parameters())
        fixed = [value for path, value in flat if _layer_index(path) is None]
        layer_count = len(
            {
                index
                for path, _value in flat
                if (index := _layer_index(path)) is not None
            }
        )
        if progress is not None:
            progress(
                {
                    "phase": "materializing_fixed",
                    "fixed_tensors": len(fixed),
                    "layers_loaded": 0,
                    "layers_total": layer_count,
                }
            )
        _eval_values(mx_module, fixed)
        mx_module.clear_cache()
        strategy = apply_tensor_strategy(
            model,
            tensor_group,
            mx_module=mx_module,
            progress=progress,
        )
        # A native strategy may also replace a replicated embedding or output
        # head outside its layer loop. Re-flatten after sharding so those new
        # arrays, rather than stale pre-shard references, are materialized.
        sharded_fixed = [
            value
            for path, value in utils_module.tree_flatten(model.parameters())
            if _layer_index(path) is None
        ]
        _eval_values(mx_module, sharded_fixed)
        mx_module.clear_cache()
        if progress is not None:
            progress({"phase": "tensor_ready", "strategy": strategy})

    # CPU synchronization avoids Metal watchdog timeouts while other ranks
    # finish their last layer.
    mx_module.eval(
        mx_module.distributed.all_sum(
            mx_module.array(1.0),
            stream=mx_module.cpu,
        )
    )
    if return_config:
        return model, tokenizer, config
    return model, tokenizer


@contextmanager
def install_progressive_loader(
    server_module: Any,
    *,
    progress: ProgressCallback | None = None,
) -> Any:
    """Install the loader only around ``ModelProvider.load_default()``."""

    original = server_module.sharded_load

    def load(*args: Any, **kwargs: Any) -> Any:
        return progressive_sharded_load(*args, **kwargs, progress=progress)

    server_module.sharded_load = load
    try:
        yield
    finally:
        server_module.sharded_load = original


__all__ = [
    "install_progressive_loader",
    "materialize_parameters_progressively",
    "progressive_sharded_load",
]
