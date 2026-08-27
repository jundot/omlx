# SPDX-License-Identifier: Apache-2.0
"""Progressive distributed model loading for the pinned MLX-LM server."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from .tensor_strategies import apply_tensor_strategy, supports_model_type

_LAYER = re.compile(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)(?:\.|$)")
_LM_HEAD = re.compile(r"(?:^|\.)lm_head(?:\.|$)")
_MTP = re.compile(r"(?:^|\.)mtp(?:\.|$)")
ProgressCallback = Callable[[dict[str, Any]], None]


def _ds4_shard_native_enabled() -> bool:
    """Explicit rollback gate; unknown values preserve the lazy loader."""

    return os.environ.get("OMLX_DSV4_SHARD_NATIVE_LOAD", "0").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


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

    # DeepSeek-V4 has an explicit structure-first adapter for complete local
    # checkpoints.  It builds/quantizes the empty module tree, applies the TP
    # wrappers to those placeholders, then reads only this rank's exact
    # safetensors ranges.  Try it before either call to ``load_model``: even a
    # lazy full-checkpoint load constructs one graph for every full expert bank
    # and defeats the purpose of shard-native loading.  Remote identifiers,
    # partial checkpoints and unsupported tensor metadata return ``None``
    # without changing the ordinary loader below.
    if (
        _ds4_shard_native_enabled()
        and tensor_group is not None
        and pipeline_group is None
    ):
        from .ds4_shard_native_adapter import try_deepseek_v4_rank_local_load

        native = try_deepseek_v4_rank_local_load(
            repo,
            model_path,
            metadata_config,
            tensor_group,
            utils_module=utils_module,
            mx_module=mx_module,
            progress=progress,
        )
        if native is not None:
            model, config = native
            if return_config:
                return model, tokenizer, config
            return model, tokenizer

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
    if tensor_group is not None:
        # Tensor loading repeatedly materializes one full layer and replaces
        # it with a local shard. MLX's default free-buffer cache can retain the
        # discarded full-layer allocations: on a 128 GB rank we observed
        # 48.4 GB active weights plus ~34 GB cached buffers at layer 22, with
        # no free pages left for layer 23. Disable only the *free* cache during
        # this one-shot transformation; active arrays remain untouched, and
        # the runtime cache limit is restored before inference begins.
        set_cache_limit = getattr(mx_module, "set_cache_limit", None)
        previous_cache_limit = (
            int(set_cache_limit(0)) if callable(set_cache_limit) else None
        )
        mx_module.clear_cache()
        # Materialize replicated fixed weights first. Keep a standalone output
        # head lazy until the tensor strategy has had a chance to row-shard it;
        # otherwise a rank briefly allocates the full head before retaining
        # only its local vocabulary rows.
        flat = utils_module.tree_flatten(model.parameters())
        fixed = [
            value
            for path, value in flat
            if _layer_index(path) is None
            and _LM_HEAD.search(path) is None
            and _MTP.search(path) is None
        ]
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
        # ``flat`` still pins every pre-shard layer array. The strategy below
        # swaps module parameters for sharded slices, but an array this
        # snapshot keeps alive can never be released: each materialized layer
        # would stay resident in full alongside its shard, so a TP=2 rank
        # grows toward ~1.5x the whole model instead of its half (observed: a
        # rank killed at 163 GiB against an 84 GiB plan). Drop the snapshot
        # — and the pre-shard fixed list — before any sharding begins.
        del flat, fixed
        # A pipeline model keeps global layer numbering by padding layers
        # before this stage with None. Tensor strategies intentionally reject
        # such a list because pure TP must never shard an incomplete model by
        # accident. For an explicitly signed hybrid graph, temporarily expose
        # only this stage's concrete layers, shard them inside its TP subgroup,
        # then restore the global None-padded view for cache/range validation.
        pipeline_owner = getattr(model, "model", None)
        pipeline_layers = (
            getattr(pipeline_owner, "layers", None)
            if pipeline_group is not None
            else None
        )
        original_pipeline_layers = (
            list(pipeline_layers) if isinstance(pipeline_layers, list) else None
        )
        active_positions = (
            [
                index
                for index, layer in enumerate(original_pipeline_layers)
                if layer is not None
            ]
            if original_pipeline_layers is not None
            else []
        )
        if original_pipeline_layers is not None:
            if not active_positions:
                raise RuntimeError("hybrid pipeline stage contains no layers")
            pipeline_owner.layers = [
                original_pipeline_layers[index] for index in active_positions
            ]
        try:
            strategy = apply_tensor_strategy(
                model,
                tensor_group,
                mx_module=mx_module,
                progress=progress,
            )
        finally:
            if original_pipeline_layers is not None:
                sharded_layers = list(pipeline_owner.layers)
                if len(sharded_layers) != len(active_positions):
                    raise RuntimeError(
                        "tensor strategy changed the hybrid stage layer count"
                    )
                restored = list(original_pipeline_layers)
                for index, layer in zip(active_positions, sharded_layers):
                    restored[index] = layer
                pipeline_owner.layers = restored
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
        if previous_cache_limit is not None:
            set_cache_limit(previous_cache_limit)
        if progress is not None:
            progress({"phase": "tensor_ready", "strategy": strategy})
    elif pipeline_group is not None:
        materialize_parameters_progressively(
            model.parameters(),
            mx_module=mx_module,
            tree_flatten=utils_module.tree_flatten,
            progress=progress,
        )

    # Do not add a final collective here. Every tensor above has already been
    # materialized locally, and the launcher withholds the endpoint until it
    # has received ``rank_ready`` from every member. A redundant CPU-stream
    # all-reduce at peak wired-memory pressure can lose a JACCL completion and
    # tear down an otherwise complete load. Rank markers are the load barrier;
    # the first inference request cannot arrive before all of them are ready.
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
