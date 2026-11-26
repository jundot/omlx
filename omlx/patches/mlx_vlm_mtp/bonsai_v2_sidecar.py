# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import ModuleType

import mlx.core as mx
import mlx.nn as nn
import safetensors

from ..mlx_lm_mtp import get_mtp_depth, is_mtp_active, is_mtp_depth_fixed
from ..prism_hadamard_verify import PRISM_HADAMARD_MODEL_TYPE
from ..prism_hadamard_verify import install as install_prism_hadamard_verify
from . import qwen35_vlm_runtime

q35_lang: ModuleType | None
try:
    from mlx_vlm.models.qwen3_5 import language as _q35_lang

    q35_lang = _q35_lang
except ImportError:
    q35_lang = None

BONSAI_V2_SIDECAR_REL_PATH = Path("mtp") / "mtp-v2-head.safetensors"
BONSAI_V2_QUANT_BITS = 4
BONSAI_V2_QUANT_GROUP_SIZE = 32

_GIT_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"

LINEAR_STEM_MAP: dict[str, str] = {
    "nextn.eh_proj": "fc",
    "attn_q": "layers.0.self_attn.q_proj",
    "attn_k": "layers.0.self_attn.k_proj",
    "attn_v": "layers.0.self_attn.v_proj",
    "attn_output": "layers.0.self_attn.o_proj",
    "ffn_gate": "layers.0.mlp.gate_proj",
    "ffn_up": "layers.0.mlp.up_proj",
    "ffn_down": "layers.0.mlp.down_proj",
}

NORM_STEM_MAP: dict[str, str] = {
    "nextn.enorm": "pre_fc_norm_embedding",
    "nextn.hnorm": "pre_fc_norm_hidden",
    "nextn.shared_head_norm": "norm",
    "attn_norm": "layers.0.input_layernorm",
    "post_attention_norm": "layers.0.post_attention_layernorm",
    "attn_q_norm": "layers.0.self_attn.q_norm",
    "attn_k_norm": "layers.0.self_attn.k_norm",
}

EXPECTED_SIDECAR_KEYS: frozenset[str] = frozenset(
    [
        f"{stem}.{suffix}"
        for stem in LINEAR_STEM_MAP
        for suffix in ("weight", "scales", "biases")
    ]
    + [f"{stem}.weight" for stem in NORM_STEM_MAP]
)


def bonsai_v2_sidecar_entry_exists(model_path: str | Path) -> bool:
    """Return True when mtp/mtp-v2-head.safetensors exists or is a symlink."""
    if not str(model_path):
        return False
    sidecar_path = Path(model_path).expanduser() / BONSAI_V2_SIDECAR_REL_PATH
    return sidecar_path.exists() or sidecar_path.is_symlink()


def _parse_positive_int(
    text_cfg: dict[str, object], key: str
) -> tuple[int | None, str]:
    value = text_cfg.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None, f"{key} must be a positive integer"
    return value, ""


def _expected_linear_dims(
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, tuple[int, int]]:
    return {
        "nextn.eh_proj": (hidden_size, 2 * hidden_size),
        "attn_q": (num_heads * head_dim * 2, hidden_size),
        "attn_k": (num_kv_heads * head_dim, hidden_size),
        "attn_v": (num_kv_heads * head_dim, hidden_size),
        "attn_output": (hidden_size, num_heads * head_dim),
        "ffn_gate": (intermediate_size, hidden_size),
        "ffn_up": (intermediate_size, hidden_size),
        "ffn_down": (hidden_size, intermediate_size),
    }


def _read_config_geometry(
    root: Path,
) -> tuple[tuple[int, int, int, int, int] | None, str]:
    config_path = root / "config.json"
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"Invalid mtp/mtp-v2-head.safetensors config: {exc}"

    if not isinstance(raw_config, dict):
        return (
            None,
            "Invalid mtp/mtp-v2-head.safetensors config: config.json must be a JSON object",
        )
    if raw_config.get("model_type") != PRISM_HADAMARD_MODEL_TYPE:
        return (
            None,
            f"Invalid mtp/mtp-v2-head.safetensors config: model_type must be {PRISM_HADAMARD_MODEL_TYPE!r}",
        )

    raw_text_cfg = raw_config.get("text_config")
    text_cfg = (
        raw_text_cfg if isinstance(raw_text_cfg, dict) and raw_text_cfg else raw_config
    )

    parsed: dict[str, int] = {}
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    ):
        value, err = _parse_positive_int(text_cfg, key)
        if value is None:
            return None, f"Invalid mtp/mtp-v2-head.safetensors config: {err}"
        parsed[key] = value

    hidden_size = parsed["hidden_size"]
    intermediate_size = parsed["intermediate_size"]
    num_heads = parsed["num_attention_heads"]
    num_kv_heads = parsed["num_key_value_heads"]
    head_dim = parsed["head_dim"]

    if hidden_size % BONSAI_V2_QUANT_GROUP_SIZE != 0:
        return (
            None,
            "Invalid mtp/mtp-v2-head.safetensors config: hidden_size must be divisible by 32",
        )
    if intermediate_size % BONSAI_V2_QUANT_GROUP_SIZE != 0:
        return (
            None,
            "Invalid mtp/mtp-v2-head.safetensors config: intermediate_size must be divisible by 32",
        )
    if (num_heads * head_dim) % BONSAI_V2_QUANT_GROUP_SIZE != 0:
        return (
            None,
            "Invalid mtp/mtp-v2-head.safetensors config: num_attention_heads * head_dim must be divisible by 32",
        )

    return (hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim), ""


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(len(_GIT_LFS_PREFIX)) == _GIT_LFS_PREFIX
    except OSError:
        return False


def validate_bonsai_v2_sidecar(
    model_path: str | Path,
) -> tuple[Path | None, str]:
    """Validate mtp/mtp-v2-head.safetensors header, metadata, dtypes, and shapes."""
    if not str(model_path):
        return None, "model_path missing"

    root = Path(model_path).expanduser()
    if not bonsai_v2_sidecar_entry_exists(root):
        return (
            None,
            "Ternary Bonsai checkpoint has no bundled mtp/mtp-v2-head.safetensors sidecar",
        )

    sidecar_path = root / BONSAI_V2_SIDECAR_REL_PATH
    if not sidecar_path.is_file():
        return (
            None,
            "Invalid mtp/mtp-v2-head.safetensors path: sidecar must be a readable regular file",
        )

    geometry, config_error = _read_config_geometry(root)
    if geometry is None:
        return None, config_error
    hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim = geometry
    expected_linear = _expected_linear_dims(
        hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim
    )

    try:
        with safetensors.safe_open(str(sidecar_path), framework="numpy") as handle:
            metadata = handle.metadata() or {}
            if "bits" in metadata and metadata["bits"] != str(BONSAI_V2_QUANT_BITS):
                return (
                    None,
                    f"Invalid mtp/mtp-v2-head.safetensors schema: expected bits='4', got {metadata['bits']!r}",
                )
            if "group_size" in metadata and metadata["group_size"] != str(
                BONSAI_V2_QUANT_GROUP_SIZE
            ):
                return (
                    None,
                    "Invalid mtp/mtp-v2-head.safetensors schema: "
                    f"expected group_size='32', got {metadata['group_size']!r}",
                )

            actual_keys = set(handle.keys())
            if actual_keys != EXPECTED_SIDECAR_KEYS:
                missing = sorted(EXPECTED_SIDECAR_KEYS - actual_keys)
                extra = sorted(actual_keys - EXPECTED_SIDECAR_KEYS)
                return (
                    None,
                    "Invalid mtp/mtp-v2-head.safetensors schema: "
                    f"tensor key mismatch (missing={missing[:3]}, extra={extra[:3]})",
                )

            for stem, (rows, width) in expected_linear.items():
                expected_specs = (
                    (f"{stem}.weight", "U32", [rows, width // 8]),
                    (f"{stem}.scales", "F16", [rows, width // 32]),
                    (f"{stem}.biases", "F16", [rows, width // 32]),
                )
                for key, expected_dtype, expected_shape in expected_specs:
                    tensor_slice = handle.get_slice(key)
                    actual_dtype = tensor_slice.get_dtype()
                    actual_shape = list(tensor_slice.get_shape())
                    if actual_dtype != expected_dtype:
                        return (
                            None,
                            f"Invalid mtp/mtp-v2-head.safetensors schema: "
                            f"{key} expected dtype {expected_dtype}, got {actual_dtype}",
                        )
                    if actual_shape != expected_shape:
                        return (
                            None,
                            f"Invalid mtp/mtp-v2-head.safetensors schema: "
                            f"{key} expected shape {expected_shape}, got {actual_shape}",
                        )

            for stem in NORM_STEM_MAP:
                key = f"{stem}.weight"
                tensor_slice = handle.get_slice(key)
                actual_dtype = tensor_slice.get_dtype()
                actual_shape = list(tensor_slice.get_shape())
                expected_shape = (
                    [head_dim]
                    if stem in ("attn_q_norm", "attn_k_norm")
                    else [hidden_size]
                )
                if actual_dtype not in ("F32", "F16"):
                    return (
                        None,
                        f"Invalid mtp/mtp-v2-head.safetensors schema: "
                        f"{key} expected dtype F32 or F16, got {actual_dtype}",
                    )
                if actual_shape != expected_shape:
                    return (
                        None,
                        f"Invalid mtp/mtp-v2-head.safetensors schema: "
                        f"{key} expected shape {expected_shape}, got {actual_shape}",
                    )
    except (OSError, safetensors.SafetensorError) as exc:
        if _is_git_lfs_pointer(sidecar_path):
            return (
                None,
                "Invalid mtp/mtp-v2-head.safetensors header: file is an unpulled Git LFS pointer",
            )
        return None, f"Invalid mtp/mtp-v2-head.safetensors header: {exc}"

    return sidecar_path, ""


def _resolve_parent_module(root: nn.Module, dotted_path: str) -> tuple[object, str]:
    parts = dotted_path.split(".")
    current: object = root
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current, parts[-1]


def bind_bonsai_v2_mtp_sidecar(model: nn.Module, model_path: str | Path) -> bool:
    """Attach and populate a 4-bit group-32 MTPModule onto a loaded Bonsai v2 VLM."""
    if not bonsai_v2_sidecar_entry_exists(model_path):
        return False

    sidecar_path, reason = validate_bonsai_v2_sidecar(model_path)
    if sidecar_path is None:
        raise ValueError(reason)

    if q35_lang is None or not qwen35_vlm_runtime.apply():
        raise ValueError("mlx-vlm Qwen3.5 runtime unavailable for Ternary Bonsai MTP")

    language_model_cls = getattr(q35_lang, "LanguageModel", None)
    mtp_module_cls = getattr(q35_lang, "MTPModule", None)
    if language_model_cls is None or not callable(mtp_module_cls):
        raise ValueError("mlx-vlm Qwen3.5 runtime unavailable for Ternary Bonsai MTP")

    language_model = getattr(model, "language_model", None)
    if (
        not isinstance(language_model, nn.Module)
        or type(language_model) is not language_model_cls
    ):
        raise ValueError(
            "Ternary Bonsai MTP sidecar requires mlx_vlm Qwen3.5 LanguageModel"
        )

    if not getattr(model, "_omlx_prism_hadamard_installed", False):
        install_prism_hadamard_verify(model)
    mtp_args = copy.copy(language_model.args)
    mtp_args.mtp_num_hidden_layers = 1
    mtp_module = mtp_module_cls(mtp_args)

    loaded_arrays = mx.load(str(sidecar_path))
    if not isinstance(loaded_arrays, dict):
        raise ValueError(
            "Invalid mtp/mtp-v2-head.safetensors header: expected safetensors tensor map"
        )

    for stem, target_path in LINEAR_STEM_MAP.items():
        quant_linear = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
        nn.Module.__init__(quant_linear)
        quant_linear["weight"] = loaded_arrays[f"{stem}.weight"]
        quant_linear["scales"] = loaded_arrays[f"{stem}.scales"]
        quant_linear["biases"] = loaded_arrays[f"{stem}.biases"]
        quant_linear.group_size = BONSAI_V2_QUANT_GROUP_SIZE
        quant_linear.bits = BONSAI_V2_QUANT_BITS
        quant_linear.mode = "affine"
        quant_linear.freeze()
        parent, attr_name = _resolve_parent_module(mtp_module, target_path)
        setattr(parent, attr_name, quant_linear)

    fp16_max = float(mx.finfo(mx.float16).max)
    for stem, target_path in NORM_STEM_MAP.items():
        raw_norm = loaded_arrays[f"{stem}.weight"]
        if raw_norm.dtype != mx.float16:
            if not bool(mx.all(mx.isfinite(raw_norm)).item()) or bool(
                mx.any(mx.abs(raw_norm) > fp16_max).item()
            ):
                raise ValueError(
                    f"Invalid mtp/mtp-v2-head.safetensors values: {stem}.weight exceeds float16 range"
                )
            raw_norm = raw_norm.astype(mx.float16)
        parent, attr_name = _resolve_parent_module(mtp_module, target_path)
        norm_module = getattr(parent, attr_name)
        norm_module.weight = raw_norm

    mx.eval(mtp_module.parameters())
    if is_mtp_active():
        depth = get_mtp_depth()
        depth_fixed = is_mtp_depth_fixed()
    else:
        depth = (
            4
            if getattr(mtp_args, "hidden_size", None) == 5120
            and getattr(mtp_args, "num_hidden_layers", None) == 64
            else 3
        )
        depth_fixed = False
    qwen35_vlm_runtime._configure_vlm_mtp_host(
        language_model,
        mtp_module,
        decode_enabled=True,
        depth=depth,
        depth_fixed=depth_fixed,
    )
    return True
