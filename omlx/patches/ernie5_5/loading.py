# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer, PreTrainedConfig

from ..deepseek_v41.loading import _load_shard
from ..mlx_lm_mtp import get_mtp_depth, is_mtp_active, is_mtp_depth_fixed
from .config import ModelConfig
from .dspark import DSpark
from .model import Model
from .processing import Processor

logger = logging.getLogger(__name__)

_RELEASES = "baidu/ERNIE-5.5-Flash-Preview (FP4) and baidu/ERNIE-5.5-Flash-Preview-FP8"
_DRAFT_PREFIX = "model.mtp."
_OQ_DRAFT_PREFIX = "language_model." + _DRAFT_PREFIX
_BF16_MODULES = ("kv_b_proj",)
_FP8_BLOCK = 128
_MX_GROUP = 32


def repack(weight, scale):
    """Reinterpret a release tensor as MLX ``(weight, scales, ok)`` without rounding.

    FP4 experts (int8 E2M1 pairs, E8M0 scales per 32 values) are already MLX
    mxfp4. FP8 E4M3 tensors with F32 scales per 128x128 block become mxfp8 by
    repeating each block exponent. ``ok`` is false when a block scale is not
    a power of two that E8M0 can hold.
    """
    if weight.dtype == mx.int8:
        return weight.view(mx.uint32), scale, mx.array(True)
    exponent = mx.log2(scale)
    ok = mx.all(
        mx.isfinite(exponent)
        & (exponent == mx.round(exponent))
        & (mx.abs(exponent) <= 127)
    )
    rows, cols = weight.shape
    scales = (exponent + 127).astype(mx.uint8)
    scales = mx.repeat(scales, _FP8_BLOCK, axis=0)[:rows]
    scales = mx.repeat(scales, _FP8_BLOCK // _MX_GROUP, axis=1)
    return weight.view(mx.uint32), scales[:, : cols // _MX_GROUP], ok


def _convert_shard(raw, pending, n_experts, draft=False):
    out, checks = {}, {}
    for key, value in raw.items():
        if (key.startswith(_DRAFT_PREFIX) and not draft) or key.endswith("_scale_inv"):
            continue
        scale = raw.get(key + "_scale_inv")
        if scale is None:
            out[key] = value
            continue
        weight, scales, checks[key] = repack(value, scale)
        parts = key.split(".")
        if parts[3:5] == ["block_sparse_moe", "experts"]:
            experts = pending.setdefault((parts[1], int(parts[2]), parts[6]), {})
            experts[int(parts[5])] = (weight, scales)
        elif any(name in key for name in _BF16_MODULES):
            out[key] = mx.dequantize(
                weight,
                scales,
                group_size=_MX_GROUP,
                bits=8,
                mode="mxfp8",
                dtype=mx.bfloat16,
            )
        else:
            stem = key.removesuffix(".weight")
            out[stem + ".weight"], out[stem + ".scales"] = weight, scales
    passed = mx.stack(list(checks.values())).tolist() if checks else []
    failed = next((key for key, ok in zip(checks, passed) if not ok), None)
    if failed is not None:
        raise ValueError(
            f"FP8 block scales of {failed} are not powers of two; "
            "this checkpoint cannot be repacked to mxfp8 without rounding"
        )
    for kind, layer, projection in [
        k for k, v in pending.items() if len(v) == n_experts
    ]:
        experts = pending.pop((kind, layer, projection))
        stem = f"model.{kind}.{layer}.block_sparse_moe.experts.{projection}"
        out[stem + ".weight"] = mx.stack([experts[e][0] for e in range(n_experts)])
        out[stem + ".scales"] = mx.stack([experts[e][1] for e in range(n_experts)])
    mx.eval(out)
    return out


def _quantization(weights):
    specs = {}
    for key, scales in weights.items():
        if key.endswith(".scales"):
            stem = key.removesuffix(".scales")
            bits = weights[stem + ".weight"].shape[-1] // scales.shape[-1]
            specs[stem] = {"group_size": _MX_GROUP, "bits": bits, "mode": f"mxfp{bits}"}
    return specs


def _load_oq_build(model, path, quantization, draft):
    weights, draft_weights = {}, {}
    for shard in sorted(path.glob("*.safetensors")):
        for key, value in _load_shard(shard).items():
            if not key.startswith(_OQ_DRAFT_PREFIX):
                weights[key] = value
            elif draft:
                draft_weights[key.removeprefix("language_model.")] = value
    draft_weights = model.language_model.sanitize(draft_weights)
    draft_specs = _quantization(draft_weights)
    weights.update({"language_model." + k: v for k, v in draft_weights.items()})
    default = {key: quantization[key] for key in ("group_size", "bits", "mode")}

    def spec(name, _):
        if f"{name}.scales" not in weights:
            return False
        draft_spec = draft_specs.get(name.removeprefix("language_model."))
        return draft_spec or quantization.get(name, default)

    nn.quantize(model, class_predicate=spec)
    model.load_weights(list(weights.items()), strict=True)


def load(path, *, trust_remote_code=False, processor=True):
    """Load the FP4 or FP8 release repacked to mxfp4 and mxfp8, or an oQ build."""
    path = Path(path)
    raw_config = json.loads((path / "config.json").read_text())
    quantization = raw_config.get("quantization_config") or {}
    oq_build = raw_config.get("quantization")
    if oq_build is None and (
        quantization.get("quant_method") != "fp8"
        or quantization.get("weight_block_size") != [_FP8_BLOCK, _FP8_BLOCK]
    ):
        raise ValueError(f"oMLX serves ERNIE 5.5 Flash Preview from {_RELEASES}")
    if processor and not trust_remote_code:
        raise ValueError(
            "ERNIE 5.5 renders chats with tokenizer code shipped in the model "
            "repository. Enable Trust Remote Code for this model to load it."
        )
    config = ModelConfig.from_dict(raw_config)
    model = Model(config)
    draft = is_mtp_active() and bool(config.dspark_target_layer_ids)
    if draft:
        model.language_model.dspark = DSpark(config)
    if oq_build is not None:
        _load_oq_build(model, path, oq_build, draft)
        return _finish(model, path, processor)
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    shards = sorted(
        {
            file
            for key, file in weight_map.items()
            if draft or not key.startswith(_DRAFT_PREFIX)
        }
    )
    pending, weights = {}, {}
    for number, shard in enumerate(shards):
        raw = _load_shard(path / shard)
        weights.update(_convert_shard(raw, pending, config.n_routed_experts, draft))
        if number % 16 == 0:
            logger.info(
                "ERNIE 5.5 loaded %d/%d shards (%.1f GiB active)",
                number + 1,
                len(shards),
                mx.get_active_memory() / 1024**3,
            )
    language_model = model.language_model
    weights = language_model.sanitize(weights)
    specs = _quantization(weights)
    nn.quantize(language_model, class_predicate=lambda name, _: specs.get(name, False))
    language_model.load_weights(list(weights.items()), strict=True)
    return _finish(model, path, processor)


def draft_head_bytes(path):
    config = json.loads((Path(path) / "config.json").read_text())
    if not config.get("dspark_target_layer_ids"):
        return 0
    values = config["vocab_size"] * config["hidden_size"]
    return values // 2 + values // _MX_GROUP


def _finish(model, path, processor):
    language_model = model.language_model
    if "dspark" in language_model:
        language_model.dspark.head = nn.QuantizedLinear.from_linear(
            language_model.lm_head, group_size=_MX_GROUP, bits=4, mode="mxfp4"
        )
    mx.eval(model.parameters())
    model.eval()
    language_model.configure_mtp(True, get_mtp_depth())
    language_model._omlx_mtp_depth_fixed = is_mtp_depth_fixed()
    if not processor:
        return model, None
    # transformers: AutoTokenizer reads config.json and has no ernie5_5_moe config.
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, config=PreTrainedConfig()
    )
    return model, Processor(tokenizer)
