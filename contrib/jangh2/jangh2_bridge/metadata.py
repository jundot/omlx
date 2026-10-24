"""Checkpoint validation and exact namespace translation (no MLX dependency)."""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Mapping

# Native jang_tools kernels use these cubic coefficients. Checkpoint levels must
# agree before using those kernels; accepting a different codebook is unsafe.
CUBIC = {2: (0.893, 0.05065), 3: (0.47125, 0.0116), 4: (0.2405, 0.0021)}
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
log = logging.getLogger("jangh2_bridge")
_PLE = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.(?:shards\.\d+|shard_\d+)\.(?:weight|scales|biases)$")


def ple_layer_indices(layer_ids: Any, num_hidden_layers: int | None = None) -> tuple[int, ...]:
    """Convert Qwen4's one-based config IDs to zero-based decoder indices.

    Keep list order: the list ordinal is ple_layer_index and participates in
    n-gram prime selection/hashing. Tensor paths and constructor layer_idx are
    already zero-based and must not be passed to this conversion a second time.
    """
    if (not isinstance(layer_ids, (list, tuple)) or not layer_ids
            or any(type(value) is not int or value < 1 for value in layer_ids)):
        raise ValueError('ple_layer_ids must be a nonempty list of positive one-based integers')
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError('Duplicate one-based PLE layer IDs are not supported')
    if num_hidden_layers is not None:
        if type(num_hidden_layers) is not int or num_hidden_layers < 1:
            raise ValueError('num_hidden_layers must be a positive integer')
        if any(value > num_hidden_layers for value in layer_ids):
            raise ValueError('A one-based PLE layer ID exceeds num_hidden_layers')
    return tuple(value - 1 for value in layer_ids)


def canonical(key: str) -> str:
    """Translate only known Qwen4 checkpoint roots to the VLM model namespace."""
    for prefix in ("model.language_model.visual.", "model.visual.", "visual."):
        if key.startswith(prefix):
            key = "vision_tower." + key[len(prefix):]
            break
    else:
        for prefix in ("model.language_model.", "language_model.model.", "language_model.", "model."):
            if not key.startswith(prefix):
                continue
            tail = key[len(prefix):]
            if tail.split(".", 1)[0] in ("layers", "embed_tokens", "norm", "hyper_connection_mixer"):
                key = "language_model.model." + tail
            elif (tail == "lm_head" or tail.startswith("lm_head.")):
                key = "language_model." + tail
            break
        if (key == "lm_head" or key.startswith("lm_head.")):
            key = "language_model." + key
    # Some Qwen4 checkpoints store the final (non-layer) hyper-connection
    # mixer directly at the root. The architecture nests it in
    # language_model.model; treat only this known root as an alias.
    if key.startswith("hyper_connection_mixer."):
        key = "language_model.model." + key
    key = key.replace(".ple.ngram_embedding.", ".ple.ple_embedding.ngram_embedding.")
    if key.endswith(".ple.ngram_embedding"):
        key = key[:-len(".ple.ngram_embedding")] + ".ple.ple_embedding.ngram_embedding"
    return re.sub(r"\.ngram_embedding\.shard_(\d+)(?=\.|$)", r".ngram_embedding.shards.\1", key)


def canonical_map(items: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    original: dict[str, str] = {}
    for key, value in items.items():
        target = canonical(key)
        if target in result and result[target] is not value:
            raise ValueError(f"Ambiguous tensor aliases: {original[target]!r} and {key!r} -> {target!r}")
        result[target] = value
        original[target] = key
    return result



# Qwen4-Exp vendor model.sanitize() retains these stock PLE tensor names,
# but MLX-VLM's live module tree uses a nested conv1d/ple_embedding layout.
# Restrict the adjustment to exact per-decoder-layer parameter leaves; the
# full strict parameter/shape audit remains responsible for any other changes.
_PLE_RUNTIME_NAME = re.compile(
    r"^(language_model\.model\.layers\.\d+\.ple\.)"
    r"(conv1d_weight|layer_multipliers|ngram_heads_offsets|ngram_heads_vocab_sizes)$"
)
_PLE_CONV1D_LIVE = re.compile(
    r"^language_model\.model\.layers\.\d+\.ple\.conv1d\.weight$"
)
_PLE_RUNTIME_LEAVES = {
    "conv1d_weight": "conv1d.weight",
    "layer_multipliers": "ple_embedding.layer_multipliers",
    "ngram_heads_offsets": "ple_embedding.ngram_heads_offsets",
    "ngram_heads_vocab_sizes": "ple_embedding.ngram_heads_vocab_sizes",
}


def reconcile_ple_module_parameters(
    weights: Mapping[str, Any], model_parameter_names: Any
) -> tuple[dict[str, Any], int]:
    """Resolve the four verified PLE namespace differences without losing tensors.

    `weights` must already have passed the architecture's sanitize() routine.
    The aliases are applied only when the exact target exists in the *live*
    model parameter tree, not by guessing new names or silently dropping keys.
    """
    expected = set(model_parameter_names)
    result: dict[str, Any] = {}
    source_by_target: dict[str, str] = {}
    resolved = 0
    for source, value in weights.items():
        target = source
        match = _PLE_RUNTIME_NAME.fullmatch(source)
        if match:
            alternate = match.group(1) + _PLE_RUNTIME_LEAVES[match.group(2)]
            if alternate not in expected:
                raise ValueError(
                    f"Unrecognised PLE runtime parameter: {source!r}; "
                    f"expected alias {alternate!r} is absent from model parameters"
                )
            target = alternate
        if target in result:
            raise ValueError(
                f"Ambiguous PLE parameter alias: {source_by_target[target]!r} "
                f"and {source!r} both map to {target!r}"
            )
        # The Qwen4 checkpoint stores the one-input-channel, four-tap PLE
        # convolution as [out, kernel].  MLX Conv1d expects [out, kernel,
        # in_channels/groups] = [out, kernel, 1].  Introduce ONLY that
        # singleton channel axis.  Any other mismatch remains subject to
        # the full strict audit; never transpose, drop or invent values.
        if _PLE_CONV1D_LIVE.fullmatch(target) and isinstance(model_parameter_names, Mapping):
            live = model_parameter_names.get(target)
            desired = tuple(live.shape) if live is not None else None
            actual = tuple(value.shape)
            if (desired is not None and len(desired) == 3
                    and len(actual) == 2 and desired[-1] == 1
                    and actual == desired[:2]):
                value = value.reshape(*desired)
                log.info(
                    'JANGH2: reshaped PLE Conv1d kernel %s from %s to %s '
                    '(singleton input-channel axis; values unchanged)',
                    target, actual, desired,
                )
        result[target] = value
        source_by_target[target] = source
        resolved += target != source
    return result, resolved


def normalize_quantization(config: Mapping[str, Any]) -> dict[str, Any]:
    q = config.get("quantization")
    if not isinstance(q, dict):
        raise ValueError("Expected a per-module 'quantization' object in config.json")
    result: dict[str, Any] = {}
    for key, value in q.items():
        target = canonical(key) if isinstance(value, dict) else key
        if target in result and result[target] != value:
            raise ValueError(f"Conflicting quantization aliases for {target}")
        result[target] = dict(value) if isinstance(value, dict) else value
    # A second config may use different namespaces, but it must not contradict
    # the effective config. Do not silently choose one in that case.
    other = config.get("quantization_config")
    if isinstance(other, dict):
        for key, value in other.items():
            target = canonical(key) if isinstance(value, dict) else key
            if target in result and result[target] != value:
                raise ValueError(f"quantization_config disagrees with quantization: {key}")
    return result


def expert_specs(q: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for name, spec in q.items():
        if not isinstance(spec, dict) or spec.get("mode") != "jangtq2":
            continue
        parent, _, projection = name.rpartition(".")
        if not parent.endswith(".switch_mlp") or projection not in PROJECTIONS:
            raise ValueError(f"Unsupported JANGH2 module path: {name}")
        groups.setdefault(parent, {})[projection] = dict(spec)
    if not groups:
        raise ValueError("No JANGH2 expert triplets found in quantization config")
    for name, specs in groups.items():
        if set(specs) != set(PROJECTIONS):
            raise ValueError(f"Incomplete gate/up/down triplet at {name}: {sorted(specs)}")
        for projection, spec in specs.items():
            if spec.get("bits") not in CUBIC:
                raise ValueError(f"Unsupported bit width at {name}.{projection}: {spec.get('bits')}")
            if spec.get("rotation") not in ("hadamard32", "none"):
                raise ValueError(f"Unsupported or missing rotation at {name}.{projection}")
            unknown = set(spec) - {"mode", "bits", "rotation", "group_size"}
            if unknown:
                raise ValueError(f"Unknown JANGH2 metadata at {name}.{projection}: {sorted(unknown)}")
        if specs["gate_proj"]["rotation"] != specs["up_proj"]["rotation"]:
            raise ValueError(f"Different gate/up rotations are unsupported at {name}")
    return groups


def validate_config(config: Mapping[str, Any]) -> dict[int, tuple[float, ...]]:
    if config.get("model_type") != "qwen4_exp":
        raise ValueError("This bridge is scoped to model_type=qwen4_exp")
    fmt = config.get("jangtq", {})
    for key, expected in (("version", 2), ("packing", "lsb-bitstream"),
                          ("scale_dtype", "float16"), ("codebook_family", "odd-cubic")):
        if fmt.get(key) != expected:
            raise ValueError(f"Unsupported jangtq.{key}: {fmt.get(key)!r}; expected {expected!r}")
    if fmt.get("rotation") not in ("none", "hadamard32"):
        raise ValueError("Unsupported jangtq.rotation")
    groups = expert_specs(normalize_quantization(config))
    used = {spec["bits"] for group in groups.values() for spec in group.values()}
    levels: dict[int, tuple[float, ...]] = {}
    for bits in used:
        cb = fmt.get("codebooks", {}).get(str(bits), {})
        values = cb.get("levels", [])
        if len(values) != 1 << bits or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
            raise ValueError(f"Missing, non-finite or wrong-sized {bits}-bit codebook")
        alpha, beta = CUBIC[bits]
        expected = [(i - ((1 << bits) - 1) / 2) *
                    (alpha + beta * (i - ((1 << bits) - 1) / 2) ** 2)
                    for i in range(1 << bits)]
        if any(abs(a - b) > 2e-6 for a, b in zip(values, expected)):
            raise ValueError(f"{bits}-bit codebook differs from the supported JANGH2 odd-cubic codebook")
        levels[bits] = tuple(float(x) for x in values)
    return levels


def is_ple_storage(key: str) -> bool:
    return bool(_PLE.search(canonical(key)))


def is_mtp(key: str) -> bool:
    return key.startswith(("mtp.", "model.mtp.", "model.language_model.mtp.",
                           "language_model.mtp.", "language_model.model.mtp."))


def choose_ple_prefix(requested: str, weight_map: Mapping[str, Any]) -> str:
    target = canonical(requested + ".shards.0.weight")
    candidates = []
    for key in weight_map:
        if canonical(key) == target:
            match = re.search(r"\.(?:shards\.0|shard_0)\.weight$", key)
            if match:
                candidates.append(key[:match.start()])
    if len(candidates) != 1:
        raise ValueError(f"Expected one physical PLE shard-0 prefix for {requested}; found {candidates}")
    return candidates[0]


def check_tensor_pair(packed: Any, scales: Any, experts: int, out_dims: int,
                      in_dims: int, bits: int) -> None:
    if bits not in CUBIC or in_dims % 32:
        raise ValueError(f"Unsupported expert shape/bit width: K={in_dims}, bits={bits}")
    if tuple(packed.shape) != (experts, out_dims, in_dims * bits // 32):
        raise ValueError(f"Packed tensor shape {packed.shape} != {(experts, out_dims, in_dims * bits // 32)}")
    if tuple(scales.shape) != (experts, out_dims):
        raise ValueError(f"Scale tensor shape {scales.shape} != {(experts, out_dims)}")
    if str(packed.dtype).split('.')[-1] != 'uint32' or str(scales.dtype).split('.')[-1] != 'float16':
        raise ValueError(f"Expected uint32 packed / float16 scales; got {packed.dtype} / {scales.dtype}")


def safe_shard_path(root: Path, filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError(f"Unsafe checkpoint shard path: {filename!r}")
    if relative.suffix != '.safetensors':
        raise ValueError(f"Not a safetensors shard: {filename!r}")
    # Keep legitimate Hugging Face cache symlinks usable. No index-controlled
    # absolute paths or '..' traversal are permitted.
    return root / relative
