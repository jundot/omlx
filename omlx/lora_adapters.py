"""LoRA adapters applied on top of a loaded model.

The adapter stays separate from the base weights: each targeted projection is
wrapped so it returns base(x) + (x @ A) @ B. Merging the update into quantized
weights would round most of it away, so the base module is never changed.

Two on-disk formats are read:
- Hugging Face PEFT: adapter_config.json (peft_type LORA) + adapter_model.safetensors
- mlx-lm / mlx-vlm: adapter_config.json with lora_parameters + adapters.safetensors

Only projections inside the decoder layers can be targeted. Embeddings, the
LM head, vision towers and MoE experts are refused with a reason.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .oqh import runtime as oqh_runtime

CONFIG_FILE = "adapter_config.json"
PEFT_WEIGHTS = "adapter_model.safetensors"
MLX_WEIGHTS = "adapters.safetensors"

_PEFT_KEY = re.compile(
    r"^(?:base_model\.model\.)?(.+)\.lora_([AB])(?:\.default)?\.weight$"
)
_MLX_KEY = re.compile(r"^(.+)\.lora_([ab])$")
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")
# Containers whose "layers.N" is not the language-model decoder stack.
_NON_DECODER = re.compile(r"vision|visual|audio|image|encoder|projector|mtp")

# PEFT options that change the forward beyond base(x) + scale * B(A(x)).
_PEFT_UNSUPPORTED = {
    "use_dora": "DoRA adapters",
    "modules_to_save": "adapters that also replace whole modules (modules_to_save)",
    "trainable_token_indices": "adapters that train token embeddings",
    "target_parameters": "adapters that target raw parameters such as MoE experts",
    "layer_replication": "adapters with layer replication",
    "use_qalora": "QA-LoRA adapters",
    "alora_invocation_tokens": "activated LoRA (aLoRA) adapters",
    "arrow_config": "Arrow multi-adapter routing",
    "lora_bias": "adapters with a LoRA bias",
    "fan_in_fan_out": "fan-in/fan-out (Conv1D) adapters",
    "loftq_config": "LoftQ adapters, which need their own quantized base,",
}
# Unconverted PiSSA, OLoRA and CorDA adapters are trained on a residual base.
_RESIDUAL_INITS = ("pissa", "olora", "corda")
# Router and expert projections: decode paths read their weights directly.
_MOE_PROJECTIONS = ("gate", "router", "shared_expert_gate")
_MAX_HEADER = 100 << 20


class AdapterError(ValueError):
    """The adapter cannot be applied; the message says why."""


@dataclass(frozen=True)
class AdapterInfo:
    """What the admin UI shows about an adapter folder."""

    path: str
    format: str  # "peft" or "mlx"
    base_model: str | None
    rank: int
    targets: tuple[str, ...]  # Projection names, e.g. ("down_proj", "q_proj")
    num_layers: int  # Highest decoder layer it touches + 1
    dims: tuple[int, ...]  # Input and output widths of the targeted projections
    size: int  # Weight file bytes on disk
    resident_bytes: int  # Bytes once loaded in a 16-bit activation dtype
    problem: str | None  # Why oMLX cannot apply it; None when it can

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "base_model": self.base_model,
            "rank": self.rank,
            "targets": list(self.targets),
            "num_layers": self.num_layers,
            "size": self.size,
            "problem": self.problem,
        }


@dataclass(frozen=True)
class _Spec:
    format: str
    weights: Path
    config: dict


class LoRALinear(nn.Module):
    """base(x) + x @ lora_a.T @ lora_b, with the LoRA scale folded into lora_b.

    lora_a is [rank, in] and lora_b is [rank, out], both row-contiguous: the
    layouts the small decode and verify matmuls read fastest.
    """

    def __init__(self, base: nn.Module, lora_a: mx.array, lora_b: mx.array):
        super().__init__()
        self.base = base
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        # One kernel for the up-projection and the add: decode is launch-bound.
        out = mx.addmm(y, x @ self.lora_a.T, self.lora_b)
        return out if out.dtype == y.dtype else out.astype(y.dtype)


def is_adapter_dir(path: Path) -> bool:
    return (path / CONFIG_FILE).is_file()


def _read_spec(path: Path) -> _Spec:
    try:
        config = json.loads((path / CONFIG_FILE).read_text())
    except (OSError, ValueError) as e:
        raise AdapterError(f"Cannot read {CONFIG_FILE}: {e}") from e
    if not isinstance(config, dict):
        raise AdapterError(f"{CONFIG_FILE} is not a JSON object")

    if "peft_type" in config:
        peft_type = str(config.get("peft_type") or "")
        if peft_type.upper() != "LORA":
            raise AdapterError(f"PEFT type {peft_type} is not supported, only LoRA")
        for key, label in _PEFT_UNSUPPORTED.items():
            if config.get(key):
                raise AdapterError(f"{label} are not supported")
        init = config.get("init_lora_weights")
        if isinstance(init, str) and init.lower().startswith(_RESIDUAL_INITS):
            raise AdapterError(
                f"{init} adapters need the residual base they were trained on; "
                "convert the adapter to plain LoRA first"
            )
        if config.get("bias") not in (None, "none"):
            raise AdapterError("adapters that train biases are not supported")
        if not isinstance(config.get("lora_alpha"), (int, float)):
            raise AdapterError("lora_alpha is missing from the adapter config")
        weights = path / PEFT_WEIGHTS
        fmt = "peft"
    else:
        fine_tune_type = config.get("fine_tune_type", "lora")
        if fine_tune_type == "dora":
            raise AdapterError("DoRA adapters are not supported")
        if fine_tune_type != "lora":
            raise AdapterError(f"fine_tune_type {fine_tune_type} is not a LoRA adapter")
        params = config.get("lora_parameters")
        if not isinstance(params, dict) or not isinstance(
            params.get("scale"), (int, float)
        ):
            raise AdapterError(
                "lora_parameters.scale is missing from the adapter config"
            )
        weights = path / MLX_WEIGHTS
        fmt = "mlx"
    if not weights.is_file():
        raise AdapterError(f"{weights.name} is missing")
    return _Spec(format=fmt, weights=weights, config=config)


def _header(path: Path) -> bytes:
    try:
        with open(path, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            if length > _MAX_HEADER:
                raise ValueError(f"header length {length}")
            return f.read(length)
    except (OSError, ValueError, struct.error) as e:
        raise AdapterError(f"Cannot read {path.name}: {e}") from e


def _header_shapes(path: Path) -> dict[str, list[int]]:
    try:
        header = json.loads(_header(path))
        header.pop("__metadata__", None)
        return {key: list(value["shape"]) for key, value in header.items()}
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise AdapterError(f"{path.name} has a malformed safetensors header") from e


def _split_key(spec: _Spec, key: str) -> tuple[str, str] | None:
    """Return (module path, "a" | "b"), or None for a tensor that is not A or B."""
    match = (_PEFT_KEY if spec.format == "peft" else _MLX_KEY).match(key)
    if match is None:
        return None
    return match.group(1), match.group(2).lower()


def _a_in_rank(spec: _Spec, shape) -> tuple[int, int]:
    # PEFT stores A as [rank, in]; mlx stores lora_a as [in, rank].
    return (shape[1], shape[0]) if spec.format == "peft" else (shape[0], shape[1])


def _b_out_rank(spec: _Spec, shape) -> tuple[int, int]:
    # PEFT stores B as [out, rank]; mlx stores lora_b as [rank, out].
    return (shape[0], shape[1]) if spec.format == "peft" else (shape[1], shape[0])


def _decoder_location(module_path: str) -> tuple[int, str] | None:
    match = _LAYER.search(module_path)
    if match is None or _NON_DECODER.search(module_path[: match.start()]):
        return None
    return int(match.group(1)), match.group(2)


def _pairs(spec: _Spec, shapes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Group A and B by module path as {path: {"a": (key, shape), "b": ...}}.

    Raises for anything that cannot be applied.
    """
    pairs: dict[str, dict[str, Any]] = {}
    foreign = []
    for key, shape in shapes.items():
        split = _split_key(spec, key)
        if split is None or len(shape) != 2:
            foreign.append(key)
            continue
        pairs.setdefault(split[0], {})[split[1]] = (key, shape)
    if foreign:
        raise AdapterError(
            "the adapter holds tensors oMLX does not apply: " + ", ".join(foreign[:3])
        )
    if not pairs:
        raise AdapterError("the adapter holds no LoRA tensors")
    outside = []
    for module_path, pair in pairs.items():
        if set(pair) != {"a", "b"}:
            raise AdapterError(f"{module_path} is missing its LoRA A or B tensor")
        if _a_in_rank(spec, pair["a"][1])[1] != _b_out_rank(spec, pair["b"][1])[1]:
            raise AdapterError(f"{module_path} has LoRA A and B of different ranks")
        if _decoder_location(module_path) is None:
            outside.append(module_path)
    if outside:
        raise AdapterError(
            "only projections inside the decoder layers can be adapted; "
            f"the adapter also targets {outside[0]}"
        )
    return pairs


def read_adapter_info(path: str | Path) -> AdapterInfo | None:
    """Describe an adapter folder; None when it is not one."""
    path = Path(path)
    if not is_adapter_dir(path):
        return None
    fmt, base_model, problem = "peft", None, None
    rank = num_layers = size = resident = 0
    targets: set[str] = set()
    dims: set[int] = set()
    try:
        raw = json.loads((path / CONFIG_FILE).read_text())
        if isinstance(raw, dict):
            fmt = "peft" if "peft_type" in raw else "mlx"
            base_model = raw.get("base_model_name_or_path") or raw.get("model")
        spec = _read_spec(path)
        size = spec.weights.stat().st_size
        pairs = _pairs(spec, _header_shapes(spec.weights))
        for module_path, pair in pairs.items():
            in_dims, r = _a_in_rank(spec, pair["a"][1])
            out_dims, _ = _b_out_rank(spec, pair["b"][1])
            layer, name = _decoder_location(module_path)
            rank = max(rank, r)
            num_layers = max(num_layers, layer + 1)
            targets.add(name.rsplit(".", 1)[-1])
            dims.update((in_dims, out_dims))
            resident += 2 * r * (in_dims + out_dims)
    except (AdapterError, OSError, ValueError) as e:
        problem = str(e)
    return AdapterInfo(
        path=str(path),
        format=fmt,
        base_model=base_model if isinstance(base_model, str) else None,
        rank=rank,
        targets=tuple(sorted(targets)),
        num_layers=num_layers,
        dims=tuple(sorted(dims)),
        size=size,
        resident_bytes=resident,
        problem=problem,
    )


@functools.lru_cache(maxsize=64)
def _cached_info(path: str, stamps: tuple) -> AdapterInfo | None:
    return read_adapter_info(path)


def adapter_info(path: str | Path) -> AdapterInfo | None:
    """read_adapter_info, cached until the adapter config or weights change."""
    path = Path(path).expanduser()
    stamps = []
    for name in (CONFIG_FILE, PEFT_WEIGHTS, MLX_WEIGHTS):
        try:
            stat = (path / name).stat()
        except OSError:
            if name == CONFIG_FILE:
                return None
            continue
        stamps.append((name, stat.st_mtime_ns, stat.st_size))
    return _cached_info(str(path), tuple(stamps))


def _text_config(config: dict) -> dict:
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def compatibility_problem(info: AdapterInfo, model_config: dict) -> str | None:
    """Cheap check against a model config before loading; load checks every shape."""
    if info.problem:
        return info.problem
    text = _text_config(model_config)
    layers = text.get("num_hidden_layers") or text.get("n_layer")
    hidden = text.get("hidden_size")
    if not isinstance(layers, int) or not isinstance(hidden, int):
        return "the model config does not describe decoder layers"
    if info.num_layers > layers:
        return (
            f"the adapter targets {info.num_layers} decoder layers; "
            f"the model has {layers}"
        )
    if hidden not in info.dims:
        return f"no adapted projection reads or writes the model width {hidden}"
    return None


def _scale_for(spec: _Spec, module_path: str, rank: int) -> float:
    if spec.format == "mlx":
        return float(spec.config["lora_parameters"]["scale"])
    alpha = float(spec.config["lora_alpha"])
    # PEFT matches pattern keys against the module name, or a dotted suffix of it.
    for pattern, value in (spec.config.get("alpha_pattern") or {}).items():
        if re.match(rf"(.*\.)?({pattern})$", module_path):
            alpha = float(value)
            break
    denominator = math.sqrt(rank) if spec.config.get("use_rslora") else rank
    return alpha / denominator


def _decoder_layers(model: nn.Module) -> list[nn.Module]:
    language_model = getattr(model, "language_model", None)
    owner = model if language_model is None else language_model
    layers = getattr(owner, "layers", None)
    if not isinstance(layers, (list, tuple)) or not layers:
        raise AdapterError("this model does not expose its decoder layers")
    return list(layers)


def _linear_dims(module: nn.Module) -> tuple[int, int] | None:
    if isinstance(module, nn.QuantizedLinear):
        return module.weight.shape[1] * 32 // module.bits, module.weight.shape[0]
    if isinstance(module, nn.Linear):
        return module.weight.shape[1], module.weight.shape[0]
    in_dims = getattr(module, "input_dims", None)
    out_dims = getattr(module, "output_dims", None)
    if isinstance(in_dims, int) and isinstance(out_dims, int):
        return in_dims, out_dims
    return None


def _activation_dtype(module: nn.Module) -> mx.Dtype:
    floats = (mx.bfloat16, mx.float16, mx.float32)
    # Quantized scales, oQh row scales, then dense weights follow the activations.
    for name in ("scales", "rscale", "weight"):
        value = module.get(name)
        if isinstance(value, mx.array) and value.dtype in floats:
            return value.dtype
    for _, value in tree_flatten(module.parameters()):
        if value.dtype in floats:
            return value.dtype
    return mx.bfloat16


def _probe_out_dims(module: nn.Module, in_dims: int, dtype: mx.Dtype) -> int:
    """Output width of a projection type without dimension attributes."""
    try:
        out = module(mx.zeros((1, in_dims), dtype=dtype))
        mx.eval(out)
    except Exception as e:
        raise AdapterError(f"the projection does not take {in_dims} inputs: {e}") from e
    return out.shape[-1]


def _target(layer: nn.Module, name: str) -> nn.Module:
    module = dict(layer.named_modules()).get(name)
    if module is None:
        raise AdapterError(f"the model has no projection {name}")
    if isinstance(module, LoRALinear):
        raise AdapterError(f"{name} already has an adapter")
    if "Switch" in type(module).__name__ or name.rsplit(".", 1)[-1] in _MOE_PROJECTIONS:
        raise AdapterError(f"{name} is a MoE router or expert, which is not supported")
    if module.children() and not hasattr(module, "input_dims"):
        raise AdapterError(f"{name} is not a linear projection")
    return module


def fingerprint(path: str | Path) -> str:
    """Hash of the adapter config, the weights header, size and modification time.

    It keys the SSD cache, so a replaced weights file gets a new one without
    reading every tensor a second time.
    """
    spec = _read_spec(Path(path).expanduser())
    stat = spec.weights.stat()
    digest = hashlib.blake2b(digest_size=8)
    digest.update(json.dumps(spec.config, sort_keys=True).encode())
    digest.update(_header(spec.weights))
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def apply_adapter(model: nn.Module, path: str | Path) -> int:
    """Wrap the projections the adapter targets; return how many were wrapped.

    Every target is checked before any module is replaced, so a refused
    adapter leaves the model unchanged.
    """
    path = Path(path).expanduser()
    if not path.is_dir():
        raise AdapterError(f"LoRA adapter folder not found: {path}")
    spec = _read_spec(path)
    weights = mx.load(str(spec.weights))
    pairs = _pairs(spec, {key: list(value.shape) for key, value in weights.items()})
    layers = _decoder_layers(model)

    plan = []
    for module_path, pair in pairs.items():
        index, name = _decoder_location(module_path)
        if index >= len(layers):
            raise AdapterError(
                f"the adapter targets layer {index}; the model has {len(layers)}"
            )
        base = _target(layers[index], name)
        # A as [rank, in] and B as [rank, out], whatever the file stores.
        a, b = weights[pair["a"][0]], weights[pair["b"][0]]
        if spec.format == "peft":
            b = b.T
        else:
            a = a.T
        dtype = _activation_dtype(base)
        dims = _linear_dims(base) or (
            a.shape[1],
            _probe_out_dims(base, a.shape[1], dtype),
        )
        if dims != (a.shape[1], b.shape[1]):
            raise AdapterError(
                f"layers.{index}.{name} is {dims[0]} -> {dims[1]}; the adapter "
                f"expects {a.shape[1]} -> {b.shape[1]}"
            )
        scale = _scale_for(spec, module_path, a.shape[0])
        plan.append(
            (
                layers[index],
                name,
                base,
                mx.contiguous(a.astype(dtype)),
                mx.contiguous((b * scale).astype(dtype)),
            )
        )

    signature = fingerprint(path)
    for layer, name, base, a, b in plan:
        wrapped = LoRALinear(base, a, b)
        # Plain attribute: read by the SSD cache numerics revision.
        object.__setattr__(wrapped, "_omlx_lora_signature", signature)
        layer.update_modules(tree_unflatten([(name, wrapped)]))
    # oQh fused MLPs call matmul_rotated on their projections, which a wrapper lacks.
    oqh_runtime.release_wrapped(model)
    mx.eval([(entry[3], entry[4]) for entry in plan])
    return len(plan)
