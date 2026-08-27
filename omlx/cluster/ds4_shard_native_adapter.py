# SPDX-License-Identifier: Apache-2.0
"""Structure-first, rank-local DeepSeek-V4 tensor loading.

This adapter is intentionally narrower than the universal progressive loader:
it accepts only a complete local DeepSeek-V4 safetensors checkpoint and an
already-created TP group.  Every other path returns ``None`` before production
selects it, leaving MLX-LM's ordinary lazy loader as the atomic fallback.
"""

from __future__ import annotations

import gc
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .shard_native_loading import (
    LocalSafetensors,
    TensorDescriptor,
    TensorPartition,
    deepseek_v4_partition,
    validate_quantized_partition,
)

ProgressCallback = Any
_LAYER = re.compile(r"^(?:model\.)?layers\.(\d+)(?:\.|$)")
_MTP = re.compile(r"^mtp\.(\d+)(?:\.|$)")
_ROUTED_EXPERT = re.compile(
    r"^(?:model\.)?(?:layers\.\d+|mtp\.\d+(?:\.block)?)\.ffn\.experts\.\d+\."
)
_VOCAB = re.compile(
    r"^(?:head|lm_head)\.(?:weight|scales|biases|bias)$|"
    r"^mtp\.\d+\.markov_head\.markov_w2\.(?:weight|scales|biases|bias)$"
)
_PRESENCE_SENTINEL = "mtp.__omlx_rank_local_presence__"


class DS4NativeQualificationError(ValueError):
    """A safe reason to use the unchanged lazy loader instead."""


@dataclass(frozen=True)
class _FileSnapshot:
    filename: str
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class DS4NativeTensor:
    name: str
    sanitized_name: str
    descriptor: TensorDescriptor
    partition: TensorPartition
    source_bytes: int
    local_bytes: int
    group: str


@dataclass(frozen=True)
class DS4NativeLoadPlan:
    checkpoint: LocalSafetensors
    rank: int
    world_size: int
    tensors: tuple[DS4NativeTensor, ...]
    files: tuple[_FileSnapshot, ...]
    source_bytes: int
    local_bytes: int
    non_moe_weights: tuple[int, ...]
    moe_weights: tuple[int, ...]


def _emit(progress: ProgressCallback | None, phase: str, **payload: Any) -> None:
    if progress is not None:
        progress({"phase": phase, **payload})


def _local_repo(repo: Any, model_path: Any) -> Path | None:
    if not isinstance(repo, (str, os.PathLike)):
        return None
    raw = os.fspath(repo)
    if "://" in raw:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_dir():
        return None
    candidate = candidate.resolve()
    try:
        resolved_model = Path(model_path).expanduser().resolve()
    except (OSError, TypeError, ValueError):
        return None
    return candidate if candidate == resolved_model else None


def _positive_vector(name: str, raw: str, size: int) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise DS4NativeQualificationError(f"{name} is not an integer vector") from exc
    if len(values) != size or any(not 1 <= value <= 4096 for value in values):
        raise DS4NativeQualificationError(
            f"{name} must contain one positive value per TP rank"
        )
    return values


def _normalized_slice_weights(values: tuple[int, ...]) -> tuple[int, ...]:
    if len(set(values)) == 1:
        return (1,) * len(values)
    return values


def _partition_vectors(args: Any, size: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    units = int(args.num_attention_heads) // int(args.o_groups)
    outer_raw = os.environ.get("OMLX_TP_SHARD_WEIGHTS", "").strip()
    if outer_raw:
        outer_signed = _positive_vector("OMLX_TP_SHARD_WEIGHTS", outer_raw, size)
        if len(set(outer_signed)) > 1 and sum(outer_signed) != units:
            raise DS4NativeQualificationError(
                "unequal DS4 TP weights do not sum to heads per output group"
            )
        outer = _normalized_slice_weights(outer_signed)
    else:
        outer_signed = ()
        outer = (1,) * size

    non_moe_raw = os.environ.get("OMLX_TP_NON_MOE_SHARD_WEIGHTS", "").strip()
    if non_moe_raw:
        non_moe_signed = _positive_vector(
            "OMLX_TP_NON_MOE_SHARD_WEIGHTS", non_moe_raw, size
        )
        if sum(non_moe_signed) != units:
            raise DS4NativeQualificationError(
                "DS4 non-MoE TP weights do not sum to heads per output group"
            )
        non_moe = _normalized_slice_weights(non_moe_signed)
    else:
        non_moe_signed = outer_signed
        non_moe = outer

    moe_raw = os.environ.get("OMLX_TP_MOE_SHARD_WEIGHTS", "").strip()
    if moe_raw:
        if not outer_signed and not non_moe_raw:
            raise DS4NativeQualificationError(
                "a routed-MoE override requires a signed outer/non-MoE plan"
            )
        moe_signed = _positive_vector("OMLX_TP_MOE_SHARD_WEIGHTS", moe_raw, size)
        if sum(moe_signed) != units:
            raise DS4NativeQualificationError(
                "DS4 routed-MoE TP weights do not sum to the signed plan"
            )
        moe = _normalized_slice_weights(moe_signed)
    else:
        moe_signed = outer_signed
        moe = outer

    intermediate = int(getattr(args, "moe_intermediate_size", 0) or 0)
    for label, signed in (("non-MoE", non_moe_signed), ("routed-MoE", moe_signed)):
        if not signed:
            continue
        total = sum(signed)
        if intermediate <= 0 or intermediate % total:
            raise DS4NativeQualificationError(
                f"DS4 {label} split does not divide the expert width"
            )
        boundaries = [sum(signed[:rank]) for rank in range(1, size)]
        if any(intermediate * boundary % total for boundary in boundaries):
            raise DS4NativeQualificationError(
                f"DS4 {label} split has a fractional expert boundary"
            )
        if any(intermediate * value // total % 32 for value in signed):
            raise DS4NativeQualificationError(
                f"DS4 {label} split cuts a 32-value MXFP group"
            )
    return non_moe, moe


def _raw_group(name: str, *, n_layers: int, keep_mtp: bool) -> str:
    if match := _LAYER.match(name):
        index = int(match.group(1))
        return f"layer:{index}" if index < n_layers else "ignored"
    if match := _MTP.match(name):
        return f"mtp:{int(match.group(1))}" if keep_mtp else "ignored"
    return "fixed"


def _partition_for(
    name: str,
    *,
    rank: int,
    size: int,
    non_moe: tuple[int, ...],
    moe: tuple[int, ...],
    vocab_sharded: bool,
    o_groups: int,
) -> TensorPartition:
    if _VOCAB.fullmatch(name) and not vocab_sharded:
        return TensorPartition()
    weights = moe if _ROUTED_EXPERT.match(name) else non_moe
    return deepseek_v4_partition(
        name,
        rank=rank,
        shard_weights=weights,
        world_size=size,
        quant_boundary=1,
        o_groups=o_groups,
    )


def _snapshot(
    checkpoint: LocalSafetensors, filenames: set[str]
) -> tuple[_FileSnapshot, ...]:
    result = []
    for filename in sorted(filenames):
        stat = (checkpoint.root / filename).stat()
        result.append(
            _FileSnapshot(
                filename=filename,
                device=int(stat.st_dev),
                inode=int(stat.st_ino),
                size=int(stat.st_size),
                modified_ns=int(stat.st_mtime_ns),
                changed_ns=int(stat.st_ctime_ns),
            )
        )
    return tuple(result)


def _verify_snapshot(plan: DS4NativeLoadPlan) -> None:
    current = _snapshot(plan.checkpoint, {item.filename for item in plan.files})
    if current != plan.files:
        raise RuntimeError(
            "local safetensors checkpoint changed during rank-local load"
        )


def _sanitized_parameter_path(name: str, *, dspark: bool) -> str:
    """Mirror DeepSeek-V4 ``sanitize`` key remapping without tensor data."""

    top = {
        "embed.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "head.weight": "lm_head.weight",
        "hc_head_fn": "model.hc_head.fn",
        "hc_head_base": "model.hc_head.base",
        "hc_head_scale": "model.hc_head.scale",
    }
    path = top.get(name, name)
    if path.startswith("layers."):
        path = "model." + path
    if path.startswith("mtp."):
        parts = path.split(".", 2)
        if len(parts) == 3:
            rest = parts[2]
            if not dspark and rest.startswith(
                (
                    "attn.",
                    "ffn.",
                    "attn_norm.",
                    "ffn_norm.",
                    "hc_attn_",
                    "hc_ffn_",
                    "hc_attn.",
                    "hc_ffn.",
                )
            ):
                path = f"mtp.{parts[1]}.block.{rest}"
            for parameter in ("fn", "base", "scale"):
                if rest == f"hc_head_{parameter}":
                    path = f"mtp.{parts[1]}.hc_head.{parameter}"
    path = path.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
    for sublayer in ("attn", "ffn"):
        for parameter in ("fn", "base", "scale"):
            path = path.replace(
                f".hc_{sublayer}_{parameter}",
                f".{sublayer}_hc.{parameter}",
            )
    path = path.replace(".hc_attn.", ".attn_hc.")
    path = path.replace(".hc_ffn.", ".ffn_hc.")
    path = path.replace(".shared_experts.w1.", ".shared_experts.gate_proj.")
    path = path.replace(".shared_experts.w2.", ".shared_experts.down_proj.")
    path = path.replace(".shared_experts.w3.", ".shared_experts.up_proj.")
    path = re.sub(
        r"\.ffn\.experts\.\d+\.w1(?=\.|$)",
        ".ffn.switch_mlp.gate_proj",
        path,
    )
    path = re.sub(
        r"\.ffn\.experts\.\d+\.w2(?=\.|$)",
        ".ffn.switch_mlp.down_proj",
        path,
    )
    path = re.sub(
        r"\.ffn\.experts\.\d+\.w3(?=\.|$)",
        ".ffn.switch_mlp.up_proj",
        path,
    )
    return path


def _sanitized_module_path(name: str, *, dspark: bool) -> str:
    """Map a raw ``*.scales`` stem to the module path used by sanitize()."""

    parameter = _sanitized_parameter_path(name + ".weight", dspark=dspark)
    return parameter[: -len(".weight")]


def _qualified_native_config(config: dict[str, Any]) -> dict[str, Any]:
    """Apply the same DS4 construction gates as the production lazy loader."""

    result = dict(config)
    if "quantization_config" not in result:
        text_config = result.get("text_config", {})
        if isinstance(text_config, dict) and "quantization_config" in text_config:
            result["quantization_config"] = text_config["quantization_config"]

    from ..patches.deepseek_v4.utils_patch import (
        _native_ratio128_attention_enabled,
    )

    result["use_native_ratio128_attention"] = bool(
        result.get("use_native_ratio128_attention", True)
    ) and _native_ratio128_attention_enabled(result)

    # The production loader applies this post-load, data-dependent transform.
    # Falling back is the only exact way to preserve it in the first native
    # cut; constructing a quantized placeholder would not match the BF16 raw
    # head stored on disk.
    if os.environ.get("OMLX_DSV4_LMHEAD_Q8", "0") == "1":
        raise DS4NativeQualificationError(
            "OMLX_DSV4_LMHEAD_Q8 uses the production lazy loader"
        )
    return result


def _quantized_paths(
    checkpoint: LocalSafetensors,
    *,
    dspark: bool,
) -> frozenset[str]:
    return frozenset(
        _sanitized_module_path(name[: -len(".scales")], dspark=dspark)
        for name in checkpoint.tensor_names
        if name.endswith(".scales")
    )


def _quantization_for_path(config: dict[str, Any], path: str) -> dict[str, Any] | None:
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        return None
    override = quantization.get(path, True)
    if override is False:
        return None
    result = {
        "group_size": quantization.get("group_size"),
        "bits": quantization.get("bits"),
        "mode": quantization.get("mode", "affine"),
    }
    if isinstance(override, dict):
        result.update(override)
    if not isinstance(result["group_size"], int) or not isinstance(result["bits"], int):
        return None
    return result


def _validate_raw_key_provenance(
    entries: list[DS4NativeTensor],
    *,
    expected_experts: int,
) -> None:
    """Reject ambiguous sanitizer aliases before reading checkpoint bytes.

    Routed expert banks are the sole intentional many-to-one transform: one
    raw tensor per expert is stacked into a single ``SwitchGLU`` parameter.
    Every other active raw key must own a unique sanitized destination.
    """

    destinations: dict[str, list[str]] = {}
    for entry in entries:
        if entry.group != "ignored":
            destinations.setdefault(entry.sanitized_name, []).append(entry.name)

    expert_source = re.compile(
        r"^(?:model\.)?(?:layers\.\d+|mtp\.\d+(?:\.block)?)\.ffn\.experts\."
        r"(\d+)\.w[123]\.(?:weight|scales|biases)$"
    )
    for destination, sources in destinations.items():
        if len(sources) == 1:
            continue
        matches = [expert_source.fullmatch(source) for source in sources]
        expert_ids = {int(match.group(1)) for match in matches if match is not None}
        if (
            all(match is not None for match in matches)
            and len(sources) == expected_experts
            and expert_ids == set(range(expected_experts))
        ):
            continue
        examples = ", ".join(repr(source) for source in sorted(sources)[:3])
        raise DS4NativeQualificationError(
            "ambiguous DS4 sanitizer aliases target "
            f"{destination!r} ({len(sources)} raw keys; {examples})"
        )


def _validate_structure_key_coverage(
    model: Any,
    entries: list[DS4NativeTensor],
) -> None:
    """Prove active checkpoint destinations equal the native parameter tree."""

    from mlx.utils import tree_flatten

    checkpoint_keys = {
        entry.sanitized_name for entry in entries if entry.group != "ignored"
    }
    model_keys = {name for name, _value in tree_flatten(model.parameters())}
    if checkpoint_keys == model_keys:
        return
    missing = sorted(model_keys - checkpoint_keys)
    extra = sorted(checkpoint_keys - model_keys)
    details = []
    if missing:
        details.append(f"model_unfilled={missing[:3]!r}")
    if extra:
        details.append(f"checkpoint_unconsumed={extra[:3]!r}")
    raise DS4NativeQualificationError(
        "DS4 checkpoint keys do not exactly cover the native model structure ("
        + "; ".join(details)
        + ")"
    )


def build_deepseek_v4_native_plan(
    model_path: str | Path,
    config: dict[str, Any],
    model: Any,
    group: Any,
    *,
    vocab_sharded: bool,
) -> DS4NativeLoadPlan:
    """Build the complete descriptor-only plan before reading tensor bytes."""

    model_type = str(config.get("model_type", ""))
    if not model_type.startswith("deepseek_v4") or config.get("model_file"):
        raise DS4NativeQualificationError("checkpoint is not built-in DeepSeek-V4")
    checkpoint = LocalSafetensors(model_path)
    names = set(checkpoint.tensor_names)
    if any(
        name.endswith(".scale") and name[: -len(".scale")] + ".weight" in names
        for name in names
    ):
        # Raw HF FP4 scale expansion changes tensor shape in sanitize(); keep
        # this first production cut on already-converted MLX checkpoints.
        raise DS4NativeQualificationError(
            "raw singular-scale checkpoints use lazy fallback"
        )
    checkpoint.validate_complete()
    rank = int(group.rank())
    size = int(group.size())
    if size < 2 or not 0 <= rank < size:
        raise DS4NativeQualificationError(
            "rank-local loading requires a valid TP group"
        )
    args = getattr(model, "args", None)
    if args is None:
        raise DS4NativeQualificationError(
            "DeepSeek-V4 model has no validated arguments"
        )
    non_moe, moe = _partition_vectors(args, size)
    keep_mtp = hasattr(model, "mtp")
    n_layers = int(args.num_hidden_layers)
    dspark = bool(getattr(model, "_omlx_dspark_decode_enabled", False))

    entries = []
    filenames: set[str] = set()
    by_name: dict[str, DS4NativeTensor] = {}
    for name in checkpoint.tensor_names:
        descriptor = checkpoint.descriptor(name)
        group_name = _raw_group(name, n_layers=n_layers, keep_mtp=keep_mtp)
        partition = (
            TensorPartition()
            if group_name == "ignored"
            else _partition_for(
                name,
                rank=rank,
                size=size,
                non_moe=non_moe,
                moe=moe,
                vocab_sharded=vocab_sharded,
                o_groups=int(args.o_groups),
            )
        )
        local_shape = partition.local_shape(descriptor.shape)
        itemsize = max(1, (descriptor.data_stop - descriptor.data_start))
        elements = 1
        for dimension in descriptor.shape:
            elements *= dimension
        local_elements = 1
        for dimension in local_shape:
            local_elements *= dimension
        local_bytes = (
            0
            if group_name == "ignored"
            else itemsize * local_elements // max(elements, 1)
        )
        entry = DS4NativeTensor(
            name=name,
            sanitized_name=_sanitized_parameter_path(name, dspark=dspark),
            descriptor=descriptor,
            partition=partition,
            source_bytes=descriptor.data_stop - descriptor.data_start,
            local_bytes=local_bytes,
            group=group_name,
        )
        entries.append(entry)
        by_name[name] = entry
        filenames.add(descriptor.filename)

    expected_experts = int(getattr(args, "n_routed_experts", 0) or 0)
    _validate_raw_key_provenance(entries, expected_experts=expected_experts)
    _validate_structure_key_coverage(model, entries)

    # ``sanitize`` stacks every routed bank once expert zero is present.  Prove
    # the whole bank and one common metadata shape now, before any tensor bytes
    # are read or a partially populated layer can replace model parameters.
    expert_pattern = re.compile(
        r"^((?:model\.)?(?:layers\.\d+|mtp\.\d+(?:\.block)?)\.ffn\.experts\.)"
        r"(\d+)\.(w[123])\.(weight|scales|biases)$"
    )
    banks: dict[tuple[str, str, str], dict[int, TensorDescriptor]] = {}
    for name, entry in by_name.items():
        if match := expert_pattern.fullmatch(name):
            bank = (match.group(1), match.group(3), match.group(4))
            banks.setdefault(bank, {})[int(match.group(2))] = entry.descriptor
    for (prefix, projection, suffix), descriptors in banks.items():
        if set(descriptors) != set(range(expected_experts)):
            raise DS4NativeQualificationError(
                f"incomplete routed expert bank {prefix}{projection}.{suffix}"
            )
        shapes = {(item.dtype, item.shape) for item in descriptors.values()}
        if len(shapes) != 1:
            raise DS4NativeQualificationError(
                f"inconsistent routed expert bank {prefix}{projection}.{suffix}"
            )

    for name, entry in by_name.items():
        if entry.group == "ignored" or not name.endswith(".scales"):
            continue
        stem = name[: -len(".scales")]
        weight = by_name.get(stem + ".weight")
        if weight is None or weight.group != entry.group:
            raise DS4NativeQualificationError(
                f"quantized tensor {stem!r} has no weight"
            )
        path = _sanitized_module_path(stem, dspark=dspark)
        quantization = _quantization_for_path(config, path)
        if quantization is None:
            raise DS4NativeQualificationError(
                f"quantized tensor {path!r} has no explicit DS4 quantization"
            )
        bias_entry = by_name.get(stem + ".biases")
        validate_quantized_partition(
            weight.descriptor,
            entry.descriptor,
            weight.partition,
            bits=int(quantization["bits"]),
            group_size=int(quantization["group_size"]),
            biases=bias_entry.descriptor if bias_entry is not None else None,
        )
        if entry.partition != weight.partition or (
            bias_entry is not None and bias_entry.partition != weight.partition
        ):
            raise DS4NativeQualificationError(
                f"quantized tensor {path!r} does not share one TP boundary"
            )

    source_bytes = sum(item.source_bytes for item in entries if item.group != "ignored")
    local_bytes = sum(item.local_bytes for item in entries)
    return DS4NativeLoadPlan(
        checkpoint=checkpoint,
        rank=rank,
        world_size=size,
        tensors=tuple(entries),
        files=_snapshot(checkpoint, filenames),
        source_bytes=source_bytes,
        local_bytes=local_bytes,
        non_moe_weights=non_moe,
        moe_weights=moe,
    )


def _construct_model(
    config: dict[str, Any],
    checkpoint: LocalSafetensors,
    *,
    utils_module: Any,
) -> Any:
    if config.get("model_file") is not None:
        raise DS4NativeQualificationError("custom model code uses lazy fallback")
    get_classes = getattr(utils_module, "_get_classes", None)
    if not callable(get_classes):
        raise DS4NativeQualificationError("MLX-LM class resolver is unavailable")
    model_class, args_class = get_classes(config=config)
    args = args_class.from_dict(config)
    model = model_class(args)

    quantization = config.get("quantization")
    if quantization is not None:
        if not isinstance(quantization, dict):
            raise DS4NativeQualificationError(
                "unsupported DeepSeek-V4 quantization config"
            )
        group_size = quantization.get("group_size")
        bits = quantization.get("bits")
        if not isinstance(group_size, int) or not isinstance(bits, int):
            raise DS4NativeQualificationError(
                "incomplete DeepSeek-V4 quantization config"
            )
        try:
            import mlx.nn as nn
        except ImportError as exc:  # pragma: no cover - production dependency
            raise DS4NativeQualificationError("MLX NN is unavailable") from exc
        dspark = bool(getattr(model, "_omlx_dspark_decode_enabled", False))
        quantized = _quantized_paths(checkpoint, dspark=dspark)

        def predicate(path: str, module: Any) -> bool | dict[str, Any]:
            override = quantization.get(path)
            if override is not None:
                return override
            if not hasattr(module, "to_quantized"):
                return False
            return path in quantized

        nn.quantize(
            model,
            group_size=group_size,
            bits=bits,
            mode=quantization.get("mode", "affine"),
            class_predicate=predicate,
        )
    elif config.get("quantization_config"):
        raise DS4NativeQualificationError("legacy quantization uses lazy fallback")
    if config.get("quantize_activations", False):
        raise DS4NativeQualificationError("activation quantization uses lazy fallback")
    model.eval()
    return model


def _apply_structure_sharding(
    model: Any,
    group: Any,
    mx_module: Any,
    progress: ProgressCallback | None,
) -> None:
    if not str(getattr(model, "model_type", "")).startswith("deepseek_v4"):
        raise RuntimeError("DS4 structure adapter received another architecture")
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, list) or not layers:
        raise RuntimeError("DS4 structure adapter cannot locate transformer layers")
    model.shard(group)
    original = list(layers)
    auxiliary = list(
        getattr(model, "_omlx_tensor_auxiliary_modules", lambda: ())() or ()
    )
    try:
        for module in auxiliary:
            backbone.layers = [getattr(module, "block", module)]
            model.shard(group)
    finally:
        backbone.layers = original

    # These wrappers only create lazy slices of zero placeholders.  Local
    # checkpoint values replace them before any eval, so no full head or expert
    # bank is ever made resident.
    from .tensor_strategies import _shard_auxiliary_vocab_heads, _shard_output_head

    if _shard_output_head(model, group, mx_module, progress):
        _shard_auxiliary_vocab_heads(model, group, mx_module, progress)
    model._omlx_shard_native_loading = True
    _emit(
        progress,
        "tensor_native_structure",
        strategy="deepseek_v4_rank_local",
        rank=int(group.rank()),
        ranks=int(group.size()),
        layers_total=len(original),
        auxiliary_modules=len(auxiliary),
        vocab_sharded=bool(getattr(model, "_omlx_vocab_parallel_head", False)),
    )


def _ordered_groups(
    plan: DS4NativeLoadPlan,
) -> tuple[tuple[str, list[DS4NativeTensor]], ...]:
    grouped: dict[str, list[DS4NativeTensor]] = {}
    for entry in plan.tensors:
        if entry.group != "ignored":
            grouped.setdefault(entry.group, []).append(entry)

    def key(item: tuple[str, list[DS4NativeTensor]]) -> tuple[int, int]:
        name = item[0]
        if name.startswith("mtp:"):
            return (0, int(name.split(":", 1)[1]))
        if name == "fixed":
            return (1, 0)
        return (2, int(name.split(":", 1)[1]))

    return tuple(sorted(grouped.items(), key=key))


def _load_plan(
    model: Any,
    plan: DS4NativeLoadPlan,
    *,
    mx_module: Any,
    tree_flatten: Any,
    progress: ProgressCallback | None,
) -> None:
    _verify_snapshot(plan)
    loaded_bytes = 0
    loaded_tensors = 0
    loaded_keys: set[str] = set()
    ignored_keys: list[str] = []
    groups = _ordered_groups(plan)
    has_mtp_weights = any(name.startswith("mtp:") for name, _ in groups)

    for group_index, (group_name, entries) in enumerate(groups, start=1):
        raw: dict[str, Any] = {}
        for entry in entries:
            raw[entry.name] = plan.checkpoint.load_partition(
                entry.name,
                entry.partition,
                mx_module=mx_module,
                expected_descriptor=entry.descriptor,
                evaluate=False,
            )
            loaded_bytes += entry.local_bytes
            loaded_tensors += 1
            if loaded_tensors % 256 == 0:
                _emit(
                    progress,
                    "tensor_native_reading",
                    strategy="deepseek_v4_rank_local",
                    rank=plan.rank,
                    tensors_loaded=loaded_tensors,
                    tensors_total=sum(item.group != "ignored" for item in plan.tensors),
                    bytes_loaded=loaded_bytes,
                    bytes_total=plan.local_bytes,
                    load_group=group_name,
                )

        sentinel = bool(
            hasattr(model, "mtp")
            and has_mtp_weights
            and not group_name.startswith("mtp:")
        )
        if sentinel:
            raw[_PRESENCE_SENTINEL] = mx_module.array(0, dtype=mx_module.uint8)
        sanitized = model.sanitize(raw) if hasattr(model, "sanitize") else raw
        sanitized.pop(_PRESENCE_SENTINEL, None)
        del raw

        expected_sanitized = {entry.sanitized_name for entry in entries}
        actual_sanitized = set(sanitized)
        if actual_sanitized != expected_sanitized:
            missing = sorted(expected_sanitized - actual_sanitized)
            extra = sorted(actual_sanitized - expected_sanitized)
            details = []
            if missing:
                details.append(f"dropped={missing[:3]!r}")
            if extra:
                details.append(f"unexpected={extra[:3]!r}")
            raise RuntimeError(
                "rank-local DS4 sanitizer did not consume every active raw key "
                f"exactly once for {group_name!r} (" + "; ".join(details) + ")"
            )

        current = {
            name: tuple(value.shape) for name, value in tree_flatten(model.parameters())
        }
        accepted = []
        for name, value in sanitized.items():
            expected = current.get(name)
            if expected is None:
                ignored_keys.append(name)
                continue
            if tuple(value.shape) != expected:
                raise RuntimeError(
                    f"rank-local DS4 tensor {name!r} has shape {tuple(value.shape)}; "
                    f"structure requires {expected}"
                )
            accepted.append((name, value))
            loaded_keys.add(name)
        if accepted:
            model.load_weights(accepted, strict=False)
            mx_module.eval(*(value for _name, value in accepted))
        synchronize = getattr(mx_module, "synchronize", None)
        if callable(synchronize):
            synchronize()
        del accepted, sanitized, current
        gc.collect()
        mx_module.clear_cache()

        layer = None
        if group_name.startswith("layer:"):
            layer = int(group_name.split(":", 1)[1])
        _emit(
            progress,
            "tensor_native_loading",
            strategy="deepseek_v4_rank_local",
            rank=plan.rank,
            load_group=group_name,
            layer=layer,
            groups_loaded=group_index,
            groups_total=len(groups),
            layers_loaded=(layer + 1) if layer is not None else 0,
            layers_total=int(getattr(model.args, "num_hidden_layers", 0)),
            bytes_loaded=loaded_bytes,
            bytes_total=plan.local_bytes,
        )

    _verify_snapshot(plan)
    remaining = {
        name
        for name, _value in tree_flatten(model.parameters())
        if name not in loaded_keys
    }
    if ignored_keys:
        examples = ", ".join(repr(name) for name in sorted(ignored_keys)[:3])
        raise RuntimeError(
            "rank-local DS4 sanitizer produced parameters outside the native "
            f"model structure ({len(ignored_keys)}; {examples})"
        )
    if remaining:
        examples = ", ".join(repr(name) for name in sorted(remaining)[:3])
        raise RuntimeError(
            "rank-local DS4 checkpoint did not fill the complete native model "
            f"structure ({len(remaining)}; {examples})"
        )
    _emit(
        progress,
        "tensor_native_ready",
        strategy="deepseek_v4_rank_local",
        rank=plan.rank,
        ranks=plan.world_size,
        source_bytes=plan.source_bytes,
        local_bytes=plan.local_bytes,
        tensors_loaded=loaded_tensors,
        sanitized_keys_loaded=len(loaded_keys),
        checkpoint_keys_consumed=loaded_tensors,
        checkpoint_keys_ignored=sum(entry.group == "ignored" for entry in plan.tensors),
        model_keys_unfilled=0,
    )


def try_deepseek_v4_rank_local_load(
    repo: Any,
    model_path: Any,
    config: dict[str, Any],
    group: Any,
    *,
    utils_module: Any,
    mx_module: Any,
    progress: ProgressCallback | None = None,
) -> tuple[Any, dict[str, Any]] | None:
    """Return a loaded DS4 model, or ``None`` before selecting this path."""

    local = _local_repo(repo, model_path)
    if local is None or not str(config.get("model_type", "")).startswith("deepseek_v4"):
        return None
    try:
        config = _qualified_native_config(config)
        checkpoint = LocalSafetensors(local)
        model = _construct_model(config, checkpoint, utils_module=utils_module)
        # Determine the exact head policy from the same production wrapper
        # before the descriptor plan chooses replicated vs equal vocab rows.
        _apply_structure_sharding(model, group, mx_module, progress)
        plan = build_deepseek_v4_native_plan(
            local,
            config,
            model,
            group,
            vocab_sharded=bool(getattr(model, "_omlx_vocab_parallel_head", False)),
        )
    except (DS4NativeQualificationError, OSError, ValueError) as exc:
        # Qualification owns only a fresh, still-unpublished module tree.
        # Discard it wholesale and let the caller invoke the unchanged lazy
        # loader; no rank-visible model or checkpoint bytes have been updated.
        gc.collect()
        mx_module.clear_cache()
        _emit(
            progress,
            "tensor_native_fallback",
            strategy="deepseek_v4_rank_local",
            reason=str(exc),
        )
        return None

    _emit(
        progress,
        "tensor_native_qualified",
        strategy="deepseek_v4_rank_local",
        rank=plan.rank,
        ranks=plan.world_size,
        source_bytes=plan.source_bytes,
        local_bytes=plan.local_bytes,
        tensors_total=sum(item.group != "ignored" for item in plan.tensors),
        model_identity=plan.checkpoint.model_identity,
        non_moe_weights=list(plan.non_moe_weights),
        moe_weights=list(plan.moe_weights),
    )
    _load_plan(
        model,
        plan,
        mx_module=mx_module,
        tree_flatten=utils_module.tree_flatten,
        progress=progress,
    )
    return model, config


__all__ = [
    "DS4NativeLoadPlan",
    "DS4NativeQualificationError",
    "DS4NativeTensor",
    "build_deepseek_v4_native_plan",
    "try_deepseek_v4_rank_local_load",
]
