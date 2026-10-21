# SPDX-License-Identifier: MIT
"""Stream original or converted V4.1 weights; Engram is resident or mmap-backed."""

import fcntl
import json
import logging
from contextlib import ExitStack, closing
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.generate import wired_limit
from transformers import PreTrainedTokenizerFast

from .config import ModelConfig
from .convert import iter_source_weights, source_engram_tables
from .model import Model
from .processing import Processor
from .quantization import QuantizedProjection
from .storage import DiskEngramEmbedding, EngramPrefetch, TensorFile, decode_array

logger = logging.getLogger(__name__)


def _load_shard(path):
    # The model retains a full Metal copy. Avoid filling the unified file cache
    # with a second copy of hundreds of GiB of ordinary checkpoint weights.
    with Path(path).open("rb") as reader:
        nocache = getattr(fcntl, "F_NOCACHE", None)
        if nocache is not None:
            fcntl.fcntl(reader.fileno(), nocache, 1)
        values = mx.load(reader)
        # File-object reads must finish before the descriptor is closed.
        mx.eval(values)
    # CPU file reads alone do not submit GPU work. Establish Metal residency
    # while shards accumulate, before loading another large host-visible buffer.
    # Only a bounded slice is reduced; the checkpoint tensors remain unchanged.
    value = next((value for value in values.values() if value.size), None)
    if value is not None:
        mx.eval(mx.sum(value.reshape(-1)[:32].astype(mx.float32)))
        mx.synchronize()
    return values


# A tensor-parallel rank splits these projections: rows hold whole heads,
# output groups or expert neurons, and columns their partial sums. Everything
# else, Engram included, is replicated.
_TENSOR_ROWS = ("attn.wq_b", "attn.wo_a", "experts.w1", "experts.w3")
_TENSOR_COLUMNS = ("attn.wo_b", "experts.w2")


def tensor_slice(key, value, rank, size):
    """This rank's part of checkpoint tensor ``key``."""
    module = key.rpartition(".")[0]
    if key.endswith("attn.attn_sink"):
        axis = 0
    elif module.endswith(_TENSOR_ROWS):
        axis = -2
    elif module.endswith(_TENSOR_COLUMNS):
        axis = -1
    else:
        return value
    return mx.contiguous(mx.split(value, size, axis=axis)[rank])


def set_module(model, path, module):
    parent = model
    parts = path.split(".")
    for part in parts[:-1]:
        parent = (
            parent[int(part)] if isinstance(parent, list) else getattr(parent, part)
        )
    if isinstance(parent, list):
        parent[int(parts[-1])] = module
    else:
        setattr(parent, parts[-1], module)


def load(
    path,
    *,
    engram_ssd_offload=False,
    preserve_mtp=None,
    moe_expert_offload_resident_fraction=None,
    ced_prefill=False,
    stage=None,
    tensor=None,
):
    """Load a converted or source checkpoint.

    ``stage`` makes a cluster rank: it is called with the language model to
    keep one pipeline stage, and only that stage's text weights are read.
    ``tensor`` (a distributed group) makes a tensor-parallel rank that reads
    only its slice of the split projections.
    """
    path = Path(path)
    if (path / "conversion.inprogress.json").exists():
        raise ValueError("DeepSeek V4.1 checkpoint conversion is still in progress")
    raw = json.loads((path / "config.json").read_text())
    mapping = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    format_spec = raw.get("omlx_deepseek_v41")
    if format_spec is not None and format_spec.get("version") != 1:
        raise ValueError("Unsupported DeepSeek V4.1 converted checkpoint version")
    source_checkpoint = format_spec is None
    if source_checkpoint:
        if raw.get("model_type") != "deepseek_v41":
            raise ValueError("Expected a DeepSeek V4.1 checkpoint")
        format_spec = {"engram_tables": source_engram_tables(mapping, raw)}
    config = ModelConfig.from_dict(raw)
    if stage is not None or tensor is not None:
        if source_checkpoint:
            raise ValueError(
                "DeepSeek V4.1 cluster ranks load converted checkpoints only"
            )
        # Cluster ranks serve the text model without draft heads.
        config.preserve_mtp = False
        config.vision_n_layers = 0
    if tensor is not None:
        config = config.tensor_shard(tensor.size())
    if source_checkpoint and preserve_mtp is None:
        from ..mlx_lm_mtp import is_mtp_active

        preserve_mtp = is_mtp_active()
    if preserve_mtp is not None:
        if not source_checkpoint and preserve_mtp != config.preserve_mtp:
            raise ValueError("Converted checkpoint MTP layout cannot be overridden")
        config.preserve_mtp = bool(preserve_mtp)
    if moe_expert_offload_resident_fraction is not None:
        from ..mlx_lm_mtp import is_mtp_active

        # Drop unused draft weights to keep the offload memory saving.
        if not is_mtp_active():
            config.preserve_mtp = False
    if ced_prefill:
        if config.ced_layout_supported():
            config.ced_prefill = True
            logger.info("DeepSeek V4.1 CED prefill enabled: decoder tail %d", config.window_size)
        else:
            config.ced_prefill = False
            logger.warning(
                "DeepSeek V4.1 CED prefill requested but layer layout is "
                "unsupported; falling back to full decoder prefill"
            )
    model = Model(config)
    held = None
    if stage is not None:
        stage(model.language_model)
    if stage is not None or tensor is not None:
        held = {name for name, _ in tree_flatten(model.parameters())}
    if tensor is not None:
        for block in model.language_model.layers:
            block._tensor_group = tensor

    def in_scope(key):
        module, _, name = key.rpartition(".")
        # A packed projection stores scales and biases beside its weight.
        return (
            held is None
            or key in held
            or (name in ("scales", "biases") and f"{module}.weight" in held)
        )

    if config.engram_layer_ids:
        logger.info(
            "DeepSeek V4.1 Engram mode: %s",
            "SSD offload" if engram_ssd_offload else "host RAM",
        )
    memory_scope = ExitStack()
    offload = None
    try:
        if not engram_ssd_offload and config.engram_layer_ids:
            # Include packed Engram storage in the normal MLX residency budget
            # during loading, before BatchGenerator takes ownership.
            memory_scope.enter_context(wired_limit(model))
        engram_keys = set()
        for module, spec in format_spec.get("engram_tables", {}).items():
            # Converted tables may live in their own file under another key.
            if not in_scope(f"{module}.weight"):
                continue
            for key_field, file_field in (
                ("weight_key", "weight_file"),
                ("scale_key", "scale_file"),
                ("bias_key", "scale_file"),
            ):
                key = spec.get(key_field)
                if key is None:
                    continue
                engram_keys.add(key)
                if format_spec.get("engram_in_index") and mapping.get(key) != (
                    spec.get(file_field) or spec["weight_file"]
                ):
                    raise ValueError(
                        f"Missing or misplaced indexed Engram tensor: {key}"
                    )
            set_module(
                model,
                module,
                DiskEngramEmbedding(
                    path / spec["weight_file"],
                    spec["weight_key"],
                    spec.get("scale_key"),
                    path / spec["scale_file"] if spec.get("scale_file") else None,
                    bias_key=spec.get("bias_key"),
                    bits=spec.get("bits"),
                    group_size=spec.get("group_size", 32),
                ),
            )
        if format_spec.get("engram_tables"):
            # Keep the GPU submission boundaries in both modes. Resident tables
            # are excluded by submit(), so RAM mode schedules no disk reads.
            model.language_model._engram_prefetch = EngramPrefetch()
        expected_shapes = {
            name: value.shape for name, value in tree_flatten(model.parameters())
        }

        if moe_expert_offload_resident_fraction is not None:
            from .moe_offload import ExpertOffloadPlan, OffloadedExpert

            offload = ExpertOffloadPlan(
                path,
                raw,
                mapping,
                config,
                moe_expert_offload_resident_fraction,
                mtp_resident=config.preserve_mtp,
            )
            model._moe_offload_plan = offload
            for layer_id, layer in enumerate(model.language_model.layers):
                prefix = f"language_model.layers.{layer_id}.ffn.experts"
                layer.ffn.experts = OffloadedExpert(layer.ffn.experts, offload, prefix)
            logger.info(
                "DeepSeek V4.1 MoE offload: %d/%d experts resident per layer "
                "(%.2f GiB -> %.2f GiB expert weights)%s",
                offload.capacity,
                offload.count,
                offload.full_bytes / 1024**3,
                offload.resident_bytes / 1024**3,
                ", DSpark draft head resident" if config.preserve_mtp else "",
            )
        offload_keys = set(offload.excluded_keys) if offload is not None else set()
        if offload is not None and not source_checkpoint and not config.preserve_mtp:
            offload_keys.update(
                k for k in mapping if k.startswith("language_model.mtp.")
            )

        def batches():
            if source_checkpoint:
                yield from iter_source_weights(
                    path,
                    raw,
                    {k: v for k, v in mapping.items() if k not in offload_keys},
                    preserve_mtp=config.preserve_mtp,
                )
            else:
                for filename in dict.fromkeys(mapping.values()):
                    keys = {key for key, file in mapping.items() if file == filename}
                    normal_keys = {
                        key
                        for key in keys - engram_keys - offload_keys
                        if in_scope(key)
                    }
                    if not normal_keys:
                        continue
                    if keys - normal_keys:
                        # Shared shards must not pull SSD-offloaded tables onto
                        # the GPU when loading their ordinary projections, nor
                        # another stage's layers onto this rank.
                        reader = TensorFile(path / filename)
                        try:
                            values = {
                                key: decode_array(*reader.read(key))
                                for key in normal_keys
                            }
                        finally:
                            reader.close()
                    else:
                        values = _load_shard(path / filename)
                    if tensor is not None:
                        values = {
                            key: tensor_slice(key, value, tensor.rank(), tensor.size())
                            for key, value in values.items()
                        }
                    for key in values:
                        if mapping.get(key) != filename:
                            raise ValueError(f"Unexpected or misplaced weight: {key}")
                    yield values, {
                        name: spec
                        for name, spec in format_spec.get(
                            "quantized_modules", {}
                        ).items()
                        if name + ".weight" in values
                    }

        seen, replaced, all_quantized = set(), set(), {}
        loaded_bytes = 0
        with closing(batches()) as stream:
            for number, (values, quantized) in enumerate(stream):
                if seen.intersection(values):
                    raise ValueError("Duplicate V4.1 target tensors")
                all_quantized.update(quantized)
                for name, spec in quantized.items():
                    if name + ".weight" in values:
                        if name + ".scales" not in values:
                            raise ValueError(
                                f"Quantized projection must be stored in one shard: {name}"
                            )
                        logical = expected_shapes[name + ".weight"]
                        packed = (*logical[:-1], logical[-1] * spec["bits"] // 32)
                        scales = (
                            *logical[:-1],
                            logical[-1] // spec.get("group_size", 32),
                        )
                        if (
                            values[name + ".weight"].shape != packed
                            or values[name + ".scales"].shape != scales
                        ):
                            raise ValueError(f"Invalid packed projection shape: {name}")
                        biases = values.get(name + ".biases")
                        if spec["mode"] == "affine" and (
                            biases is None or biases.shape != scales
                        ):
                            raise ValueError(
                                f"Missing or invalid affine biases: {name}"
                            )
                        if spec["mode"] != "affine" and biases is not None:
                            raise ValueError(f"Unexpected MXFP biases: {name}")
                        set_module(
                            model,
                            name,
                            QuantizedProjection(
                                values[name + ".weight"],
                                values[name + ".scales"],
                                biases=biases,
                                **spec,
                            ),
                        )
                        replaced.add(name)
                for key, value in values.items():
                    module_name = key.rsplit(".", 1)[0]
                    if module_name not in quantized and (
                        key not in expected_shapes
                        or value.shape != expected_shapes[key]
                    ):
                        raise ValueError(f"Unexpected target tensor shape: {key}")
                model.load_weights(list(values.items()), strict=False)
                mx.eval(values)
                if tensor is not None:
                    # Release the whole tensors this rank sliced its part from.
                    mx.clear_cache()
                seen.update(values)
                loaded_bytes += sum(value.nbytes for value in values.values())
                if number % 25 == 0:
                    logger.info(
                        "DeepSeek V4.1 loaded %d tensor groups (%.1f GiB tensors, %.1f GiB active, %.1f GiB cached)",
                        number + 1,
                        loaded_bytes / 1024**3,
                        mx.get_active_memory() / 1024**3,
                        mx.get_cache_memory() / 1024**3,
                    )
        expected = {name for name, _ in tree_flatten(model.parameters())}
        scope_keys = {key for key in mapping if in_scope(key)}
        if (
            seen != expected
            or replaced != set(all_quantized)
            or (
                not source_checkpoint
                and (
                    seen != scope_keys - engram_keys - offload_keys
                    or replaced
                    != {
                        name
                        for name in format_spec.get("quantized_modules", {})
                        if name + ".weight" not in offload_keys
                        and in_scope(name + ".weight")
                    }
                )
            )
        ):
            raise ValueError(
                f"Incomplete V4.1 checkpoint; missing={sorted(expected-seen)[:8]}, extra={sorted(seen-expected)[:8]}"
            )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
        from .tool_parser import parse_tool_call, tool_call_end, tool_call_start

        tokenizer.has_tool_calling = True
        tokenizer.tool_call_start = tool_call_start
        tokenizer.tool_call_end = tool_call_end
        tokenizer.tool_parser = parse_tool_call
        processor = Processor(tokenizer, model.config)
        if model.config.engram_layer_ids:
            model.language_model.set_tokenizer(tokenizer)
        # Load the backbone before reserving RAM for large Engram tables.
        if not engram_ssd_offload:
            for layer in model.language_model.pipeline_layers:
                if "engram" in layer and isinstance(
                    layer.engram.embed, DiskEngramEmbedding
                ):
                    layer.engram.embed.make_resident()
        mx.eval(model.parameters())
        model.eval()
        return model, processor
    except BaseException:
        model.close()
        raise
    finally:
        memory_scope.close()
