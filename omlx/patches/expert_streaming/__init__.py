# SPDX-License-Identifier: Apache-2.0
"""Expert streaming (SSD) patch for MoE models (glm_moe_dsa, deepseek_v4, ...)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False

# Re-exported from residency.py — the leaf module holds the single copy.
# Keeping a second literal here is what allowed the gates to drift apart
# (engine/batched.py and engine/vlm.py checked this set while the structural
# estimate checked the checkpoint, so a mismatch forced streaming without the
# lazy load and materialized the full MoE banks). Both names stay importable
# from the package root; there is exactly one definition.
from .residency import SUPPORTED_TYPES, normalize_model_type

try:
    # P2 follow-up: public cache-policy API at the package root.
    from .governor import dynamic_residency_enabled
except Exception:  # pragma: no cover - governor has no mlx dependency
    dynamic_residency_enabled = lambda: False  # type: ignore[assignment]

try:
    from .streaming_switch import S3FIFOExpertCache, make_expert_cache
except Exception:  # pragma: no cover - streaming_switch imports mlx
    S3FIFOExpertCache = None  # type: ignore[assignment]
    make_expert_cache = None  # type: ignore[assignment]


def is_supported_model_type(model_type: str | None) -> bool:
    if not model_type:
        return False
    return normalize_model_type(model_type) in SUPPORTED_TYPES


def apply_expert_streaming_patch() -> bool:
    global _APPLIED
    if _APPLIED:
        return False
    _APPLIED = True
    logger.info("Expert streaming patch registered (post-load converter)")
    return True


def _get_budget_bytes(model_settings: Any | None, estimate: Any | None) -> int:
    if model_settings is not None:
        # Preferred name (model_settings.py:222) + legacy cache name.
        # Explicit 0 = page-cache only (no app-level LRU); None falls through
        # to the engine default below.
        for attr in ("expert_streaming_budget_gib", "expert_cache_budget_gib"):
            gib = getattr(model_settings, attr, None)
            if gib is not None:
                try:
                    return max(0, int(float(gib) * 1024**3))
                except (TypeError, ValueError):
                    continue
        # legacy mib
        for attr in ("expert_streaming_budget_mib", "expert_cache_budget_mib"):
            mib = getattr(model_settings, attr, None)
            if mib is not None and int(mib) > 0:
                return int(int(mib) * 1024 * 1024)
    # default: page-cache only. The OS file cache serves expert reuse from
    # clean evictable pages; measured A/B on Qwen3.8-Flash-Next and
    # GLM-5.3-Flash showed it beats a 1-8 GiB app-level LRU in both cold and
    # warm runs while using several GiB less RSS. Pass an explicit budget >0
    # to opt back into the LRU heap.
    return 0


def _prior_usable(cache: Any) -> bool:
    """Cache-prior needs app-level LRU residency as its signal.

    With a page-cache-only budget the resident set is always empty and the
    rerank is pure overhead (autotune b0 trials regressed) — refuse it so
    budget-0 stays on the stock path."""
    try:
        return int(getattr(cache, "capacity", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _io_overrides(model_settings: Any | None) -> dict[str, Any]:
    """Per-model streaming IO overrides with env-fallback semantics.

    Returns a dict whose values are None when the setting is unset (keep the
    env-var / built-in default) or the requested override otherwise.
    """
    raw = {
        "expert_streaming_io_depth": None,
        "expert_streaming_coalesce": None,
        "expert_streaming_readahead": None,
        "expert_streaming_seed": None,
        "expert_streaming_pilot": None,
        "expert_streaming_per_layer_eval": None,
        "expert_streaming_pins": None,
        "expert_streaming_hot_fraction": None,
        "expert_streaming_pin_gib": None,
        "expert_streaming_pin_sync": None,
        "expert_streaming_pin_regime": None,
        "expert_streaming_cache_policy": None,
        "expert_streaming_dynamic": None,
        "expert_streaming_dynamic_max_gib": None,
    }
    if model_settings is None:
        return raw
    for key in raw:
        raw[key] = getattr(model_settings, key, None)
    depth = raw["expert_streaming_io_depth"]
    if depth is not None:
        try:
            depth = int(depth)
        except (TypeError, ValueError):
            depth = None
        else:
            depth = max(1, min(64, depth)) if depth >= 1 else None
        raw["expert_streaming_io_depth"] = depth
    policy = raw["expert_streaming_cache_policy"]
    if policy is not None:
        p = str(policy).strip().lower()
        raw["expert_streaming_cache_policy"] = p if p in ("lru", "s3fifo") else None
    max_gib = raw["expert_streaming_dynamic_max_gib"]
    if max_gib is not None:
        try:
            mg = float(max_gib)
        except (TypeError, ValueError):
            mg = None
        else:
            mg = mg if 0 < mg <= 64 else None
        raw["expert_streaming_dynamic_max_gib"] = mg
    return raw


def _expert_pin_fingerprint(
    model_path: str | Path,
    linears_by_layer: dict[int, list],
    backing: Any,
    cold_root: Any,
    hot_fraction: float | None,
) -> dict:
    """Fase L: profile-identity fields for a loaded model.

    A v2 pin profile applies only when these fields match the model: a
    mismatch logs and ignores the profile (never a silent apply). The
    fingerprint covers the checkpoint (config hash), the source/cold
    packing, the HOBBIT hot fraction and the profile format version.
    """
    import hashlib

    fp: dict = {
        "model": Path(model_path).name,
        "profile_format": 2,  # keep in sync with warmer.PROFILE_VERSION
    }
    cfg = Path(model_path) / "config.json"
    try:
        fp["config_sha"] = hashlib.sha256(cfg.read_bytes()).hexdigest()[:16]
    except Exception:
        fp["config_sha"] = None
    packing = None
    probe = next(
        (
            l
            for ls in linears_by_layer.values()
            for l in ls
            if getattr(l, "stacked_weight_key", None)
        ),
        None,
    )
    if probe is not None:
        src = "oQ4e%d b-gs%d" % (
            int(getattr(probe, "bits", 4)),
            int(getattr(probe, "group_size", 64)),
        )
        fp["source_packing"] = src
        packing = src
        if cold_root is not None and hasattr(backing, "cold_quant_params"):
            try:
                cb, cg = backing.cold_quant_params(probe.stacked_weight_key)
                cold = "cold%d b-gs%d" % (int(cb), int(cg))
                fp["cold_packing"] = cold
                packing = src + "+" + cold
            except Exception:
                fp["cold_packing"] = None
                packing = src + "+cold?"
    else:
        fp["source_packing"] = None
    fp["hot_fraction"] = (
        round(float(hot_fraction), 4) if hot_fraction is not None else None
    )
    fp["packing"] = packing
    return fp


def _wire_streaming_io_overrides(
    layers: Any,
    mtp_stages: Any,
    io_depth: int | None,
    coalesce: bool | None,
) -> int:
    """Attach per-model IO pool / coalesce overrides to streaming linears.

    Returns the number of linears wired (0 when both overrides are unset —
    the module env defaults stay in effect).
    """
    if io_depth is None and coalesce is None:
        return 0
    from .streaming_switch import io_pool_for

    pool = io_pool_for(io_depth) if io_depth is not None else None
    wired = 0
    targets = list(layers or []) + list(mtp_stages or [])
    for lyr in targets:
        moe = getattr(lyr, "mlp", None) or getattr(lyr, "ffn", None)
        if moe is None:
            block = getattr(lyr, "block", None)
            if block is not None:
                moe = getattr(block, "mlp", None) or getattr(block, "ffn", None)
        sm = getattr(moe, "switch_mlp", None) if moe is not None else None
        if sm is None:
            continue
        for proj in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
            lin = getattr(sm, proj, None)
            if lin is not None and hasattr(lin, "_io_pool_override"):
                if pool is not None:
                    lin._io_pool_override = pool  # type: ignore[attr-defined]
                if coalesce is not None:
                    lin._coalesce_override = bool(coalesce)  # type: ignore[attr-defined]
                wired += 1
    return wired


# SwitchGLU bank key prefixes per main layer. GLM/Qwen nest the MoE under
# ``mlp``; DeepSeek V4 nests it under ``ffn`` (and MTP stages under
# ``mtp.<stage>`` — see _mtp_candidate_stacked_keys).
_MAIN_SWITCH_PREFIX_TEMPLATES = (
    "model.layers.{i}.mlp.switch_mlp",
    "model.layers.{i}.ffn.switch_mlp",
    "model.language_model.layers.{i}.mlp.switch_mlp",
    "model.language_model.layers.{i}.ffn.switch_mlp",
    "language_model.model.layers.{i}.mlp.switch_mlp",
    "language_model.model.layers.{i}.ffn.switch_mlp",
    "language_model.layers.{i}.mlp.switch_mlp",
    "language_model.layers.{i}.ffn.switch_mlp",
)


def _candidate_stacked_keys(layer_idx: int, proj: str, suffix: str) -> list[str]:
    return [
        f"{template.format(i=layer_idx)}.{proj}.{suffix}"
        for template in _MAIN_SWITCH_PREFIX_TEMPLATES
    ]


def _mtp_candidate_stacked_keys(stage_idx: int, proj: str, suffix: str, trunk_layers: int | None = None) -> list[str]:
    """Bank key candidates for one DeepSeek V4 MTP/DSpark stage.

    DSpark checkpoints (0731) store ``mtp.<stage>.ffn.switch_mlp.*``; the
    legacy MTPBlock layout nests one level deeper under ``block``.
    With trunk_layers set, GLM-5.3 JANG draft keys are appended: the raw
    export stores the draft as one extra trunk-indexed layer
    (``model.layers.<n>.mlp.switch_mlp.*``) and sanitize remaps it to
    ``language_model.mtp.<stage>.block.mlp.switch_mlp.*``.
    """
    keys = [
        f"mtp.{stage_idx}.ffn.switch_mlp.{proj}.{suffix}",
        f"mtp.{stage_idx}.block.ffn.switch_mlp.{proj}.{suffix}",
    ]
    if trunk_layers is not None:
        base = int(trunk_layers) + int(stage_idx)
        keys.append(f"model.layers.{base}.mlp.switch_mlp.{proj}.{suffix}")
        keys.append(
            f"language_model.mtp.{stage_idx}.block.mlp.switch_mlp.{proj}.{suffix}"
        )
    return keys


def _resolve_stacked_key(
    candidates: list[str],
    proj: str,
    suffix: str,
    backing: Any | None,
    needle: str,
) -> str:
    """Pick the checkpoint key for one stacked bank.

    Prefers exact candidates present in the weight map, then any key
    containing *needle* (layer/scope disambiguation) plus the
    ``switch_mlp.<proj>.<suffix>`` middle. Falls back to the first
    candidate for RAM dicts / missing maps.
    """
    if backing is not None and hasattr(backing, "_weight_map"):
        wm = getattr(backing, "_weight_map", {}) or {}
        for cand in candidates:
            if cand in wm:
                return cand
        mid = f"switch_mlp.{proj}.{suffix}"
        for k in wm:
            if needle in k and mid in k:
                return k
    return candidates[0]


def _source_packing(src: Any, group_size: int, bits: int, mode: str) -> tuple[int, int, str]:
    """Packing for one streaming projection from its source module.

    JANGQ checkpoints mix precisions inside one layer (e.g. a 2-bit gate
    with 3-bit up/down). Each streaming linear keeps its own source
    projection's packing; the layer-level values stay as fallback only.
    P1: a projection missing any packing attr fails loudly (same rule
    as the layer-level detection above) — never inherits silently.
    """
    for _name in ("group_size", "bits", "mode"):
        if getattr(src, _name, None) is None:
            raise ValueError(
                f"Expert streaming: projection {getattr(src, '__class__', type(src)).__name__} "
                f"lacks {(_name)!r} — refusing to inherit packing silently"
            )
    return (
        int(getattr(src, "group_size")),
        int(getattr(src, "bits")),
        str(getattr(src, "mode")),
    )


def _model_config_candidates(model: Any) -> list[Any]:
    """Collect potential config objects for dim resolution (LLM + VLM wrappers)."""
    candidates = []
    for obj in [
        getattr(model, "args", None),
        getattr(getattr(model, "model", None), "args", None),
        getattr(getattr(model, "language_model", None), "args", None),
        getattr(getattr(getattr(model, "language_model", None), "model", None), "args", None),
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "language_model", None), "config", None),
        getattr(getattr(getattr(model, "language_model", None), "model", None), "config", None),
    ]:
        if obj is not None:
            candidates.append(obj)
    return candidates


def _resolve_moe_dims(cfg_candidates: list[Any]) -> tuple[int, int]:
    """Resolve (hidden_size, moe_intermediate_size) from config candidates."""
    hidden: int | None = None
    moe_hidden: int | None = None
    for cand in cfg_candidates:
        try:
            h = getattr(cand, "hidden_size", None)
            if h is None and isinstance(cand, dict):
                h = cand.get("hidden_size")
            if h is not None:
                hidden = int(h)
            m = getattr(cand, "moe_intermediate_size", None)
            if m is None and isinstance(cand, dict):
                m = cand.get("moe_intermediate_size")
            if m is not None:
                moe_hidden = int(m)
            if hidden is not None and moe_hidden is not None:
                break
        except Exception:
            continue
    if hidden is None:
        hidden = 4096
    if moe_hidden is None:
        # qwen4_exp default is 640, glm5_next/deepseek_v4 are 2048; try to
        # infer from first expert — keep original default 1407 for glm_moe_dsa
        moe_hidden = 1407
    # Override for known types when fallback is still generic
    try:
        mt = None
        for c in cfg_candidates:
            mt = getattr(c, "model_type", None) or (c.get("model_type") if isinstance(c, dict) else None)
            if mt:
                break
        mt = str(mt).lower().replace("-", "_") if mt else ""
        if mt in ("qwen4_exp", "qwen4_exp_text") and moe_hidden == 1407:
            moe_hidden = 640
        elif mt in ("glm5_next", "glm5_next_text", "deepseek_v4", "deepseek_v4_mtp") and moe_hidden == 1407:
            moe_hidden = 2048
    except Exception:
        pass
    return hidden, moe_hidden


def _convert_switch_mlp_module(
    moe: Any,
    layer_idx: int,
    *,
    candidates_for: Any,
    needle: str,
    backing: Any,
    backing_kind: str,
    cache: Any,
    estimate: Any,
    hidden: int,
    moe_hidden: int,
    layer: Any | None = None,
    hot_ids: set | None = None,
) -> bool:
    """Replace *moe*.switch_mlp with a StreamingSwitchGLU. Returns True on success.

    ``candidates_for(proj, suffix)`` yields the checkpoint key candidates for
    this module's stacked banks; ``needle`` disambiguates weight-map fallback
    scans (e.g. ``layers.5.`` or ``mtp.2.``). ``hot_ids`` (Fase I6) keeps
    those experts at the SOURCE packing with a dual-tier gather; absent/empty
    keeps the uniform I5 tier (bits overridden to the cold packing).
    """
    import mlx.core as mx

    from .streaming_switch import (
        StreamingQuantizedSwitchLinear,
        StreamingSwitchGLU,
        StreamingSwitchLinear,
    )

    switch_mlp = getattr(moe, "switch_mlp", None)
    if switch_mlp is None:
        return False

    # Determine quantized vs bf16: QuantizedSwitchLinear has 'scales'
    is_quantized = False
    for attr in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
        proj = getattr(switch_mlp, attr, None)
        if proj is not None:
            if hasattr(proj, "scales") or "scales" in getattr(proj, "_data", {}):
                is_quantized = True
                break
            if proj.__class__.__name__ == "QuantizedSwitchLinear":
                is_quantized = True
                break

    n_experts = estimate.experts_per_layer

    fused = hasattr(switch_mlp, "gate_up_proj")
    inv_scatter = getattr(switch_mlp, "inverse_scatter", False)

    # P1: no silent packing defaults. A quantized projection MUST expose
    # its group_size/bits/mode — guessing 64/4/affine for an unknown future
    # quant silently mis-slices every expert bank. Fail loudly instead.
    group_size: int | None = None
    bits: int | None = None
    mode: str | None = None
    if is_quantized:
        for attr in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
            proj = getattr(switch_mlp, attr, None)
            if proj is not None:
                for _name in ("group_size", "bits", "mode"):
                    if getattr(proj, _name, None) is None:
                        raise ValueError(
                            f"Expert streaming: quantized {attr} of layer "
                            f"{layer_idx} lacks {(_name)!r} — refusing to guess "
                            "packing (would mis-slice expert banks)"
                        )
                group_size = int(getattr(proj, "group_size"))
                bits = int(getattr(proj, "bits"))
                mode = str(getattr(proj, "mode"))
                break
        if group_size is None or bits is None or mode is None:
            raise ValueError(
                f"Expert streaming: quantized layer {layer_idx} exposes no "
                "projection to read packing from"
            )

    def _proj_packing(src):
        return _source_packing(src, group_size, bits, mode)

    # Cold precision tier (I5): when the backing serves this layer's banks
    # from expert_cold/, every projection of the layer computes at the
    # tier's packing — override the source bits/group size once, here, so
    # the fused and split branches both build with the tier parameters.
    # HOBBIT split (I6): with a hot set for this layer the linear keeps the
    # SOURCE packing (hot experts) and the cold packing is attached per
    # linear below (dual gather_qmm).
    hobbit_cold_params: tuple[int, int] | None = None
    if hasattr(backing, "cold_quant_params"):
        first_attr = next(
            (
                a
                for a in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
                if getattr(switch_mlp, a, None) is not None
            ),
            None,
        )
        if first_attr is not None:
            probe_key = _resolve_stacked_key(
                candidates_for(first_attr, "weight"),
                first_attr,
                "weight",
                backing,
                needle,
            )
            cold_params = backing.cold_quant_params(probe_key)
            if cold_params is not None:
                if hot_ids:
                    hobbit_cold_params = cold_params
                else:
                    bits, group_size = cold_params

    streaming_glu = StreamingSwitchGLU(
        input_dims=hidden,
        hidden_dims=moe_hidden,
        num_experts=n_experts,
        layer_idx=layer_idx,
        backing=backing,
        cache=cache,
        fused_gate_up=fused,
        inverse_scatter=inv_scatter,
        quantized=is_quantized,
        group_size=group_size,
        bits=bits,
        mode=mode,
        # DeepSeek V4 uses LimitedSwiGLU (swiglu_limit / fp32 on MTP stages);
        # copying it keeps streaming bit-exact with the resident path.
        activation=getattr(switch_mlp, "activation", None),
    )

    # For RAM dict backing, populate dict from resident weights
    if backing_kind == "ram-dict":
        assert isinstance(backing, dict)
        # Map resident stacked banks to per-expert entries
        # need to know stacked keys for file backing naming, but for RAM we key by (layer, proj)
        for proj_name in (["gate_up_proj"] if fused else ["gate_proj", "up_proj", "down_proj"]):
            proj = getattr(switch_mlp, proj_name, None)
            if proj is None:
                continue
            # weight bank
            w = getattr(proj, "weight", None)
            if w is not None:
                mx.eval(w)
                # store stacked for slicing in streaming linear fallback
                backing[(layer_idx, proj_name)] = w  # type: ignore[index]
                # also for quantized scales/biases
                if is_quantized:
                    sc = getattr(proj, "scales", None)
                    if sc is not None:
                        mx.eval(sc)
                        backing[(layer_idx, proj_name, "weight")] = w  # type: ignore[index]
                        backing[(layer_idx, proj_name, "scales")] = sc  # type: ignore[index]
                        b = getattr(proj, "biases", None)
                        if b is not None:
                            mx.eval(b)
                            backing[(layer_idx, proj_name, "biases")] = b  # type: ignore[index]
                        else:
                            # ensure weight/scales keys exist for uniform fallback
                            pass
            # bias
            b = getattr(proj, "bias", None)
            if b is not None:
                mx.eval(b)

    # Now create streaming linears for the projections
    if fused:
        src = switch_mlp.gate_up_proj
        stacked_w_key = _resolve_stacked_key(
            candidates_for("gate_up_proj", "weight"), "gate_up_proj", "weight", backing, needle
        )
        if is_quantized:
            stacked_s_key = _resolve_stacked_key(
                candidates_for("gate_up_proj", "scales"), "gate_up_proj", "scales", backing, needle
            )
            stacked_b_key = _resolve_stacked_key(
                candidates_for("gate_up_proj", "biases"), "gate_up_proj", "biases", backing, needle
            )
            _gu_gs, _gu_bits, _gu_mode = _proj_packing(src)
            proj_stream = StreamingQuantizedSwitchLinear(
                layer_idx=layer_idx,
                proj_name="gate_up_proj",
                stacked_weight_key=stacked_w_key,
                stacked_scales_key=stacked_s_key,
                stacked_biases_key=stacked_b_key,
                num_experts=n_experts,
                input_dims=hidden,
                output_dims=moe_hidden * 2,
                backing=backing,
                cache=cache,
                group_size=_gu_gs,
                bits=_gu_bits,
                mode=_gu_mode,
                has_bias=hasattr(src, "bias"),
            )
            if hasattr(src, "bias"):
                proj_stream.set_bias(src.bias)  # type: ignore[attr-defined]
        else:
            proj_stream = StreamingSwitchLinear(
                layer_idx=layer_idx,
                proj_name="gate_up_proj",
                stacked_key=stacked_w_key,
                num_experts=n_experts,
                input_dims=hidden,
                output_dims=moe_hidden * 2,
                backing=backing,
                cache=cache,
                bias=hasattr(src, "bias"),
            )
            if hasattr(src, "bias"):
                proj_stream.set_bias(src.bias)  # type: ignore[attr-defined]
        streaming_glu.gate_up_proj = proj_stream  # type: ignore[attr-defined]
        # down
        src_down = switch_mlp.down_proj
        stacked_w_key = _resolve_stacked_key(
            candidates_for("down_proj", "weight"), "down_proj", "weight", backing, needle
        )
        if is_quantized:
            stacked_s_key = _resolve_stacked_key(
                candidates_for("down_proj", "scales"), "down_proj", "scales", backing, needle
            )
            stacked_b_key = _resolve_stacked_key(
                candidates_for("down_proj", "biases"), "down_proj", "biases", backing, needle
            )
            _down_gs, _down_bits, _down_mode = _proj_packing(src_down)
            down_stream = StreamingQuantizedSwitchLinear(
                layer_idx=layer_idx,
                proj_name="down_proj",
                stacked_weight_key=stacked_w_key,
                stacked_scales_key=stacked_s_key,
                stacked_biases_key=stacked_b_key,
                num_experts=n_experts,
                input_dims=moe_hidden,
                output_dims=hidden,
                backing=backing,
                cache=cache,
                group_size=_down_gs,
                bits=_down_bits,
                mode=_down_mode,
                has_bias=hasattr(src_down, "bias"),
            )
            if hasattr(src_down, "bias"):
                down_stream.set_bias(src_down.bias)  # type: ignore[attr-defined]
        else:
            down_stream = StreamingSwitchLinear(
                layer_idx=layer_idx,
                proj_name="down_proj",
                stacked_key=stacked_w_key,
                num_experts=n_experts,
                input_dims=moe_hidden,
                output_dims=hidden,
                backing=backing,
                cache=cache,
                bias=hasattr(src_down, "bias"),
            )
            if hasattr(src_down, "bias"):
                down_stream.set_bias(src_down.bias)  # type: ignore[attr-defined]
        streaming_glu.down_proj = down_stream  # type: ignore[attr-defined]
    else:
        for proj_name, out_dim, in_dim in [
            ("gate_proj", moe_hidden, hidden),
            ("up_proj", moe_hidden, hidden),
            ("down_proj", hidden, moe_hidden),
        ]:
            src = getattr(switch_mlp, proj_name, None)
            if src is None:
                continue
            stacked_w_key = _resolve_stacked_key(
                candidates_for(proj_name, "weight"), proj_name, "weight", backing, needle
            )
            if is_quantized:
                stacked_s_key = _resolve_stacked_key(
                    candidates_for(proj_name, "scales"), proj_name, "scales", backing, needle
                )
                stacked_b_key = _resolve_stacked_key(
                    candidates_for(proj_name, "biases"), proj_name, "biases", backing, needle
                )
                _p_gs, _p_bits, _p_mode = _proj_packing(src)
                proj_stream = StreamingQuantizedSwitchLinear(
                    layer_idx=layer_idx,
                    proj_name=proj_name,
                    stacked_weight_key=stacked_w_key,
                    stacked_scales_key=stacked_s_key,
                    stacked_biases_key=stacked_b_key,
                    num_experts=n_experts,
                    input_dims=in_dim,
                    output_dims=out_dim,
                    backing=backing,
                    cache=cache,
                    group_size=_p_gs,
                    bits=_p_bits,
                    mode=_p_mode,
                    has_bias=hasattr(src, "bias"),
                )
                if hasattr(src, "bias"):
                    proj_stream.set_bias(src.bias)  # type: ignore[attr-defined]
            else:
                proj_stream = StreamingSwitchLinear(
                    layer_idx=layer_idx,
                    proj_name=proj_name,
                    stacked_key=stacked_w_key,
                    num_experts=n_experts,
                    input_dims=in_dim,
                    output_dims=out_dim,
                    backing=backing,
                    cache=cache,
                    bias=hasattr(src, "bias"),
                )
                if hasattr(src, "bias"):
                    proj_stream.set_bias(src.bias)  # type: ignore[attr-defined]
            setattr(streaming_glu, proj_name, proj_stream)

    # HOBBIT dual-tier gate (Fase I6): wire the split into every quantized
    # streaming linear of this module — fused AND split projections. With a
    # hot set, the linear keeps the SOURCE packing for hot experts and the
    # cold tier's (hobbit_cold_params) for the rest; the backing already
    # routes the reads (set_hot_experts). Skipping this leaves the linear
    # uniform while the backing splits — mixed packings in one mini-bank.
    if hot_ids and is_quantized and hobbit_cold_params is not None:
        for lin_ in (
            getattr(streaming_glu, a, None)
            for a in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
        ):
            if lin_ is not None and hasattr(lin_, "set_hobbit_split"):
                lin_.set_hobbit_split(hot_ids, hobbit_cold_params[0], hobbit_cold_params[1])

    # Fase K F1: register this layer's quantized streaming linears for the
    # O2 next-layer advisor. P3: MTP/DSpark stages register too — they live
    # in their own layer-id space (len(layers)+stage, no collision with the
    # trunk), so stage s advises s+1 within the draft chain exactly like
    # trunk layers do. Trunk->stage cross-talk stays off (separate spaces,
    # separate routing), which is correct: draft and verify route distinctly.
    if needle.startswith("layers.") or needle.startswith("mtp."):
        # Fase K K1: register on the per-conversion speculation state — the
        # global registry would let one engine's advisor target another
        # engine's linears.
        _spec_state = getattr(cache, "spec_state", None)
        if _spec_state is not None:
            _spec_state.register_linears(
                layer_idx,
                [
                    getattr(streaming_glu, a, None)
                    for a in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
                    if isinstance(getattr(streaming_glu, a, None), StreamingQuantizedSwitchLinear)
                ],
            )

    # P0: per-GLU projection count for cache slot reconciliation. A fused
    # gate_up GLU holds 2 projections (gate_up + down), a split GLU 3 —
    # the global cache was sized for the majority layout, so the convert
    # loop below reconciles any drift (see _reconcile_cache_slots).
    n_proj = len(
        [a for a in ("gate_up_proj", "down_proj") if hasattr(streaming_glu, a)]
        or [a for a in ("gate_proj", "up_proj", "down_proj") if hasattr(streaming_glu, a)]
    )
    streaming_glu.n_proj = n_proj  # type: ignore[attr-defined]

    # Fase J Etapa E: the per-layer load context's projection list (2 fused
    # gate_up+down, 3 split) — consumed by the scheduler's guard accounting
    # (_glu_projection_count).
    streaming_glu.linears = [  # type: ignore[attr-defined]
        getattr(streaming_glu, a, None)
        for a in ("gate_up_proj", "down_proj") if hasattr(streaming_glu, "gate_up_proj")
    ] or [
        getattr(streaming_glu, a, None)
        for a in ("up_proj", "gate_proj", "down_proj")
        if getattr(streaming_glu, a, None) is not None
    ]

    # Replace
    moe.switch_mlp = streaming_glu  # type: ignore[attr-defined]
    # Disable decoder FFN compilation (GLM-5.3 Glm5NextDecoderLayer
    # compiles the FFN when compile_ffn is True): mx.eval(indices) inside
    # the streaming switch is illegal under mx.compile/vmap transforms.
    if layer is not None:
        try:
            layer.compile_ffn = False  # type: ignore[attr-defined]
            layer._ffn_c = None  # type: ignore[attr-defined]
        except Exception:
            pass
        # Evaluate the layer output so the lazy graph does not pin every
        # layer's mini-bank (42 layers x ~13 MB/expert) at once — without
        # this the accumulate graph swaps on GLM-class experts.
        try:
            layer._stream_eval = True  # type: ignore[attr-defined]
        except Exception:
            pass
    return True


def _glu_projection_count(layers: Any) -> int:
    """Projections sharing one per-layer load context on a converted model.

    A converted StreamingSwitchGLU holds linears: 2 when the checkpoint
    fuses gate_up_proj (plus down_proj), 3 when gate/up are split. The
    scheduler's prefill guard charges min(2, projections) banks when a
    per-layer eval boundary is live, so any value >= 2 collapses to the
    same 2; only a 1 (no shared context) changes the charge. Defaults to 3
    (the conservative split case) when no converted GLU is reachable.
    """
    try:
        for layer in layers or ():
            linears = getattr(getattr(layer, "switch_mlp", None), "linears", None)
            # An empty/unsized linears is not a converted GLU — keep
            # looking rather than reporting 0 projections.
            if linears and len(linears) > 0:
                return len(linears)
    except Exception:  # noqa: BLE001
        pass
    return 3


def convert_model_to_streaming(
    model: Any,
    model_path: str | Path,
    model_settings: Any | None = None,
    *,
    budget_bytes: int | None = None,
    use_file_backing: bool = True,
) -> tuple[Any, Any]:
    """Convert MoE layers of *model* to streaming.

    Returns (model, backing_store) where backing_store must be kept alive
    for the model lifetime (holds mmap readers).  When no MoE layers are
    found, returns (model, None) unchanged.
    """
    from .residency import expert_streaming_estimate

    estimate = expert_streaming_estimate(str(model_path))
    if not estimate.supported:
        logger.info("Expert streaming: model %s not supported (%s)", model_path, estimate.reason)
        return model, None

    if budget_bytes is None:
        budget_bytes = _get_budget_bytes(model_settings, estimate)

    logger.info(
        "Expert streaming: converting %s: budget=%.2f GiB (%s), layers=%d, experts/layer=%d, per_expert=%.2f MB, slots/layer=%d",
        Path(model_path).name,
        budget_bytes / 1024**3,
        "page-cache only, no LRU" if budget_bytes <= 0 else "LRU heap",
        estimate.num_moe_layers,
        estimate.experts_per_layer,
        estimate.per_expert_bytes / 1024 / 1024,
        estimate.slots_for_budget(budget_bytes),
    )

    # Import here to avoid circular
    from .streaming_switch import make_expert_cache

    per_expert = estimate.per_expert_bytes or 0
    # One cache slot holds ONE projection's slice (gate/up/down are separate
    # keys), so slot sizing must divide by the projections per expert —
    # otherwise the LRU holds a third of the budget it was promised (F2).
    # Per-GLU detection below (P0): fused gate_up GLUs carry 2 projections
    # (gate_up + down), not 3 — dividing a fused model by 3 over-commits
    # the budget by 1.5x. The global per_slot uses the majority layout;
    # _convert_switch_mlp_module reconciles per-GLU drift after conversion.
    per_slot = max(1, per_expert // 3) if per_expert else 0
    # IO overrides (settings/env resolution) are resolved before the cache:
    # the eviction policy setting and the governor arming below read them,
    # and the later backing-store wiring reuses the same resolved dict.
    io_ov = _io_overrides(model_settings)
    cache = make_expert_cache(
        budget_bytes, per_slot, num_layers=estimate.num_moe_layers,
        policy=io_ov.get("expert_streaming_cache_policy"),
    )

    # Dynamic residency governor: opt-in (env), positive budget only. It
    # revisits capacity at request boundaries from system free memory; a
    # budget-0 run is page-cache-only by operator choice and stays so.
    _governor = None
    # Settings > env: a per-model setting overrides the OMLX_EXPERT_STREAMING_DYNAMIC
    # env (None keeps the env/default).
    _dyn_setting = io_ov.get("expert_streaming_dynamic")
    _dyn_on = _dyn_setting if _dyn_setting is not None else dynamic_residency_enabled()
    if _dyn_on and budget_bytes > 0 and per_slot > 0:
        try:
            from .governor import (
                ExpertResidencyGovernor,
                _max_dynamic_budget_bytes,
            )

            _gov_max_gib = io_ov.get("expert_streaming_dynamic_max_gib")
            _gov_max = (
                int(_gov_max_gib * 1024**3)
                if _gov_max_gib is not None
                else _max_dynamic_budget_bytes()
            )
            _governor = ExpertResidencyGovernor(
                cache,
                per_slot,
                estimate.num_moe_layers,
                max(_gov_max, int(budget_bytes)),
            )
            logger.info(
                "Expert streaming: dynamic residency governor armed (budget %.2f GiB, max %.2f GiB)",
                budget_bytes / 1024**3,
                _governor.max_budget_bytes / 1024**3,
            )
        except Exception:
            logger.debug("governor arming failed", exc_info=True)
            _governor = None

    # Backing store
    backing = None
    backing_kind = "ram"
    # HOBBIT split state (Fase I6): populated only when a complete cold tier
    # exists AND a learned pin profile provides frequencies; otherwise the
    # convert keeps the uniform I5 tier semantics.
    hot_ids_by_layer: dict[int, set] = {}
    if use_file_backing:
        try:
            import os

            from . import shard_bank as _shard_mod
            from .shard_bank import (
                ExpertBackingStore,
                _cold_tier_status_dir,
                cold_tier_status,
            )

            extra_roots = [
                p
                for p in os.environ.get("OMLX_EXPERT_STREAMING_EXTRA_ROOTS", "").split(":")
                if p.strip()
            ]
            # Cold precision tier (I5): expert_streaming_cold_tier ("2"/"3")
            # routes expert reads to <model>/expert_cold/ — a requantized
            # full expert set (tools/requant_cold_tier.py) that cuts the
            # bytes per token pinning decode to the NVMe I/O floor. Partial
            # tiers are rejected: the uniform-packing assumption the linears
            # build on would silently break.
            cold_root = None
            cold_setting = getattr(model_settings, "expert_streaming_cold_tier", None)
            # Gap fix: accept any 2..8-bit label and validate against the
            # tier's own __metadata__ (omlx_cold_bits) — the old ("2","3")
            # tuple rejected tiers the requant tool can already produce.
            # Mismatch disables with a warning, never silently.
            _cold_bits_label = str(cold_setting).strip() if cold_setting else ""
            if cold_setting and _cold_bits_label.isdigit() and 2 <= int(_cold_bits_label) <= 8:
                # Deploy-time override: point the tier at an arbitrary
                # directory (a read-only model volume, a second SSD, a
                # sandboxed checkout) instead of <model>/expert_cold. The
                # runtime only requires the tier SHARDS to be complete
                # (cold_tier_status checks whichever dir is used).
                cold_root = Path(os.environ.get("OMLX_EXPERT_STREAMING_COLD_ROOT", "")) \
                    if os.environ.get("OMLX_EXPERT_STREAMING_COLD_ROOT") else None
                cold_dir = cold_root if cold_root is not None else Path(model_path) / "expert_cold"
                ok, why = _cold_tier_status_dir(cold_dir, Path(model_path))
                if ok:
                    # Validate the requested label against the tier metadata.
                    try:
                        from .shard_bank import _safetensors_header as _cold_hdr

                        _meta_bits: set[str] = set()
                        for _shard in cold_dir.glob("*.safetensors"):
                            try:
                                _meta = (_cold_hdr(_shard).get("__metadata__") or {})
                                if _meta.get("omlx_cold_bits") is not None:
                                    _meta_bits.add(str(_meta["omlx_cold_bits"]))
                            except Exception:
                                continue
                        if _meta_bits and _cold_bits_label not in _meta_bits:
                            ok = False
                            why = (
                                f"tier holds bits {sorted(_meta_bits)}, "
                                f"requested {_cold_bits_label}"
                            )
                    except Exception:
                        pass
                if ok:
                    cold_root = cold_dir
                    logger.info("Expert streaming: cold tier %s-bit active (%s)", _cold_bits_label, why)
                else:
                    # Failed completeness: reset to None so the backing
                    # never engages a partial tier (the env path would leak
                    # a non-None cold_root into ExpertBackingStore below).
                    cold_root = None
                    logger.warning(
                        "Expert streaming: cold tier %s requested but %s — disabled",
                        _cold_bits_label,
                        why,
                    )
            elif cold_setting:
                # Gap fix: an unparsable label used to fall through silently.
                logger.warning(
                    "Expert streaming: cold tier %r not understood "
                    "(want 2..8) — disabled",
                    cold_setting,
                )
            backing = ExpertBackingStore(model_path, extra_roots=extra_roots, cold_root=cold_root)
            # dsv4 spill-stacking: per-expert JANGQ checkpoints serve
            # their stacked banks from spill shards outside the model
            # dir. Absorb the manifest mapping so the stacked keys
            # resolve without header scans.
            try:
                from ..deepseek_v4 import spill as _dsv4_spill

                _spill_dir = _dsv4_spill.spill_is_valid(model_path)
                if _spill_dir is not None:
                    _spill_manifest = _dsv4_spill.read_manifest(_spill_dir) or {}
                    _absorbed = backing.absorb_extra_map(
                        _spill_dir,
                        _dsv4_spill.spill_key_to_file(_spill_manifest),
                    )
                    if _absorbed:
                        logger.info(
                            "Expert streaming: %d spilled banks absorbed from %s",
                            _absorbed,
                            _spill_dir,
                        )
            except Exception:
                logger.debug("Expert streaming: spill absorb skipped", exc_info=True)
            # HOBBIT per-expert hot/cold split (Fase I6): with a cold tier
            # active, the top fraction of experts per layer (by learned
            # pin-profile frequency) keeps the ORIGINAL packing while the
            # rest compute at the tier. No profile = uniform I5 tier.
            if cold_root is not None:
                # Contract (UI/bench): None/unset hot fraction = UNIFORM tier
                # (I5) — the split is opt-in per model, like the tier itself.
                # The env default (OMLX_EXPERT_STREAMING_HOT_FRACTION) stays
                # the bench/developer override and wins only when the
                # setting is unset.
                hf_setting = io_ov.get("expert_streaming_hot_fraction")
                if hf_setting is None:
                    hf_setting = _shard_mod.HOT_FRACTION_ENV or None
                hot_fraction = (
                    None
                    if hf_setting is None
                    else max(0.0, min(1.0, float(hf_setting)))
                )
                hot_keys = (
                    _shard_mod.load_hot_set_from_profile(
                        Path(model_path) / ".omlx" / "expert_pin_profile.json",
                        hot_fraction,
                        num_experts=estimate.experts_per_layer,
                    )
                    if hot_fraction is not None and hot_fraction > 0.0
                    else {}
                )
                if hot_keys:
                    backing.set_hot_experts(hot_keys)
                    hot_ids_by_layer = {
                        int(k.removeprefix("layer_")): v for k, v in hot_keys.items()
                    }
                    logger.info(
                        "Expert streaming: HOBBIT split on %d/%d layers (fraction %.2f)",
                        len(hot_keys),
                        estimate.num_moe_layers,
                        hot_fraction,
                    )
                elif hot_fraction is not None:
                    logger.info(
                        "Expert streaming: no pin profile for HOBBIT split"
                        " — uniform cold tier (I5)"
                    )
                else:
                    logger.debug(
                        "Expert streaming: hot fraction unset — uniform cold tier (I5)"
                    )
            # Guard metadata for the scheduler's prefill chunk sizing: the
            # lazy chunk forward holds every MoE layer's assembled mini-bank
            # until the chunk-end eval, so the peak carries ~one bank per
            # layer simultaneously. Without this term the guard under-predicts
            # and admits chunks whose real peak reaches ~26 GB on qwen4_exp
            # (48 layers x ~215 uniq experts x ~2.5 MB) and squeezes the
            # machine (docs F-series F1).
            # Fase J Etapa E: boundary_active starts False — the per-layer
            # bank charge is the safe default, and it is only relaxed once a
            # per-layer eval boundary has actually been installed on a
            # decoder class (set below, after conversion). projections is
            # the number of projections sharing one per-layer load context
            # (2 fused gate_up+down, or 3 split); the guard charges
            # min(2, projections) banks. bf16/fp16 activation: one
            # materialized layer output per token. 0 when the config hid
            # hidden_size — conservative.
            _hidden_size = int(getattr(estimate, "hidden_size", 0) or 0)
            backing.streaming_guard_info = {
                "num_moe_layers": estimate.num_moe_layers,
                "experts_per_layer": estimate.experts_per_layer,
                "per_expert_bytes": estimate.per_expert_bytes,
                "boundary_active": False,
                "projections": 3,
                "activation_bytes_per_token": 2 * _hidden_size,
            }
            backing_kind = "mmap"
        except Exception as e:
            logger.warning("Expert streaming: file backing failed (%s), falling back to RAM dict", e)
            backing = None

    # RAM dict fallback: copy per-expert slices from resident model into dict
    ram_dict: dict[tuple, Any] | None = None
    if backing is None:
        ram_dict = {}
        backing = ram_dict  # type: ignore[assignment]
        backing_kind = "ram-dict"

    # Fase K K1: one speculation state per conversion. It hangs off the
    # cache (always) and off the backing store (file backing) so close()
    # drains the speculation workers with the readers.
    from .streaming_switch import SpeculationState

    _spec_state = SpeculationState()
    cache.spec_state = _spec_state  # type: ignore[attr-defined]
    if _governor is not None:
        # Same reachability path as the cache: the engine finds the governor
        # through the backing at request boundaries.
        cache.governor = _governor  # type: ignore[attr-defined]
    if not isinstance(backing, dict):
        backing.spec_state = _spec_state  # type: ignore[attr-defined]
        if _governor is not None:
            backing.governor = _governor  # type: ignore[attr-defined]
        # Reload the learned transition table so the k+1 overfetch is
        # warm from token 1 (fingerprintMismatch -> ignored, never silent).
        try:
            load_transition_profile(backing, _spec_state)
        except Exception:
            pass
        # P2: the engine reaches the shared cache through the backing it
        # already holds (for the per-request summary log).
        try:
            backing._streaming_cache = cache  # type: ignore[attr-defined]
        except Exception:
            pass

    converted = 0
    # Walk model.layers — handle LLM (model.model.layers) and VLM wrappers
    # (language_model.model.layers via language_model indirection)
    layers = None
    layers_owner = None
    # candidate attribute paths to try
    candidate_paths = [
        ("model", "layers"),  # mlx_lm LanguageModel.model.layers
        ("layers",),  # VLM Model.layers property (glm5_next)
        ("language_model", "model", "layers"),  # VLM wrapper: Model.language_model.model.layers
        ("language_model", "layers"),  # alternative VLM wrapper
        ("model", "language_model", "model", "layers"),
    ]
    for path in candidate_paths:
        cur = model
        owner = None
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            owner = cur
            cur = getattr(cur, attr)
        if ok and cur is not None:
            # sanity: should be iterable with length ~ num_layers
            try:
                _ = len(cur)  # type: ignore[arg-type]
                layers = cur  # type: ignore[assignment]
                layers_owner = owner
                break
            except Exception:
                continue
    if layers is None:
        logger.warning("Expert streaming: could not find model.layers")
        return model, backing if isinstance(backing, dict) is False else None

    hidden, moe_hidden = _resolve_moe_dims(_model_config_candidates(model))

    # Main decoder layers. GLM/Qwen nest the MoE under ``mlp``; DeepSeek V4
    # nests it under ``ffn`` — prefer whichever holds a switch_mlp.
    for layer_idx, layer in enumerate(layers):
        if layer is None:
            continue
        moe = getattr(layer, "mlp", None)
        if moe is None or getattr(moe, "switch_mlp", None) is None:
            moe = getattr(layer, "ffn", None)
        if moe is None or getattr(moe, "switch_mlp", None) is None:
            continue
        if _convert_switch_mlp_module(
            moe,
            layer_idx,
            candidates_for=lambda proj, suffix, _i=layer_idx: _candidate_stacked_keys(_i, proj, suffix),
            needle=f"layers.{layer_idx}.",
            backing=backing,
            backing_kind=backing_kind,
            cache=cache,
            estimate=estimate,
            hidden=hidden,
            moe_hidden=moe_hidden,
            layer=layer,
            hot_ids=hot_ids_by_layer.get(layer_idx),
        ):
            converted += 1

    # DeepSeek V4 MTP/DSpark stages carry their own SwitchGLU banks
    # (mtp.<stage>.ffn on DSpark checkpoints, mtp.<stage>.block.ffn on the
    # legacy MTPBlock layout). Streaming them keeps the ~3 GB/stage banks
    # out of RAM on low-memory hosts.
    # MTP stages live next to the decoder stack, but not always on the
    # same owner that holds it: glm5_next VLM resolves layers through the
    # root Model.layers property while the draft hangs off
    # language_model.mtp. Walk nearby owners before giving up.
    _mtp_owners: list = []
    for _o in (layers_owner, model):
        if _o is None:
            continue
        for _cand_o in (_o, getattr(_o, "language_model", None), getattr(_o, "model", None)):
            if _cand_o is not None and all(_cand_o is not _seen for _seen in _mtp_owners):
                _mtp_owners.append(_cand_o)
    mtp_stages = None
    for _o in _mtp_owners:
        _cand = getattr(_o, "mtp", None)
        if isinstance(_cand, (list, tuple)) and len(_cand):
            mtp_stages = _cand
            break
    mtp_converted = 0
    if mtp_stages:
        for stage_idx, stage in enumerate(mtp_stages):
            if stage is None:
                continue
            stage_moe = getattr(stage, "ffn", None)
            if stage_moe is None:
                stage_moe = getattr(stage, "mlp", None)
            compile_owner = stage
            if stage_moe is None or getattr(stage_moe, "switch_mlp", None) is None:
                block = getattr(stage, "block", None)
                if block is not None:
                    stage_moe = getattr(block, "ffn", None)
                    if stage_moe is None:
                        stage_moe = getattr(block, "mlp", None)
                    if stage_moe is not None and getattr(stage_moe, "switch_mlp", None) is not None:
                        compile_owner = block
            if stage_moe is None or getattr(stage_moe, "switch_mlp", None) is None:
                continue
            if _convert_switch_mlp_module(
                stage_moe,
                len(layers) + stage_idx,
                candidates_for=lambda proj, suffix, _s=stage_idx, _n=len(layers): _mtp_candidate_stacked_keys(_s, proj, suffix, trunk_layers=_n),
                needle=f"mtp.{stage_idx}.",
                backing=backing,
                backing_kind=backing_kind,
                cache=cache,
                estimate=estimate,
                hidden=hidden,
                moe_hidden=moe_hidden,
                layer=compile_owner,
                hot_ids=None,
            ):
                mtp_converted += 1
                converted += 1
        if mtp_converted:
            logger.info(
                "Expert streaming: converted %d/%d MTP/DSpark stage MoE banks",
                mtp_converted,
                len(mtp_stages),
            )

    # The estimate counts MTP stages as MoE layers; when the runtime MTP is
    # inactive (no model.mtp) fewer layers were converted — rebalance the
    # per-layer LRU split so the converted layers keep a fair share.
    if converted and cache.num_layers != converted and cache.capacity > 0:
        cache.num_layers = converted
        cache._per_layer_cap = max(1, cache.capacity // converted)  # type: ignore[attr-defined]

    # Gap fix: validate the resolved (hidden, moe_hidden) against the
    # converted GLUs' real dims. _resolve_moe_dims falls back to family
    # guesses (1407/640/2048) when the config lacks moe_intermediate_size;
    # a wrong guess silently mis-sizes every streaming linear. The GLUs
    # carry their construction dims — a mismatch fails loudly here.
    if converted:
        try:
            _dim_mismatch = []
            for _layer in layers or ():
                _moe = getattr(_layer, "mlp", None) or getattr(_layer, "ffn", None)
                _sm = getattr(_moe, "switch_mlp", None)
                if _sm is None or not hasattr(_sm, "_input_dims"):
                    continue
                if int(getattr(_sm, "_input_dims", hidden)) != int(hidden) or int(
                    getattr(_sm, "_hidden_dims", moe_hidden)
                ) != int(moe_hidden):
                    _dim_mismatch.append(
                        (
                            getattr(_sm, "layer_idx", "?"),
                            getattr(_sm, "_input_dims", "?"),
                            getattr(_sm, "_hidden_dims", "?"),
                        )
                    )
                    break
            if _dim_mismatch:
                logger.warning(
                    "Expert streaming: resolved dims (hidden=%d, moe=%d) disagree "
                    "with converted GLU %s — check _resolve_moe_dims fallbacks",
                    hidden, moe_hidden, _dim_mismatch[0],
                )
        except Exception:
            pass

    # P0: reconcile cache slots with the converted projection layout. The
    # cache was sized for 3 projections/expert (split); every converted
    # GLU records its real n_proj (2 fused, 3 split). A uniform fused
    # model sized //3 over-commits the budget by 1.5x — resize so
    # capacity * per_slot <= budget. Mixed layouts keep the majority
    # sizing (slots are fungible across projections) and log.
    if converted and per_expert and budget_bytes > 0:
        try:
            _n_projs: list[int] = []
            for _layer in layers or ():
                _moe = getattr(_layer, "mlp", None) or getattr(_layer, "ffn", None)
                _sm = getattr(_moe, "switch_mlp", None)
                _np = getattr(_sm, "n_proj", None)
                if isinstance(_np, int) and _np > 0:
                    _n_projs.append(_np)
            if _n_projs:
                from collections import Counter as _Counter

                _majority = _Counter(_n_projs).most_common(1)[0][0]
                _want_slot = max(1, per_expert // _majority)
                if _want_slot != cache.per_expert_bytes:
                    cache.per_expert_bytes = _want_slot
                    cache.capacity = max(1, budget_bytes // _want_slot)
                    if cache.num_layers > 0:
                        cache._per_layer_cap = max(1, cache.capacity // cache.num_layers)  # type: ignore[attr-defined]
                    cache._global_cap = cache.capacity  # type: ignore[attr-defined]
                    logger.info(
                        "Expert streaming: cache slots reconciled to %d projections "
                        "(per_slot=%d B, capacity=%d)",
                        _majority, _want_slot, cache.capacity,
                    )
                if len(set(_n_projs)) > 1:
                    logger.warning(
                        "Expert streaming: mixed fused/split layouts %s — "
                        "cache sized for majority (%d proj)",
                        sorted(set(_n_projs)), _majority,
                    )
        except Exception:
            logger.debug("Expert streaming: slot reconciliation skipped", exc_info=True)

    if converted:
        import mlx.core as mx

        mx.clear_cache()
        logger.info("Expert streaming: converted %d MoE layers (backing=%s, cache_capacity=%d experts)", converted, backing_kind, cache.capacity)
        # Per-model IO overrides (autotune): pool depth + run coalescing ride
        # the streaming linears; unset values keep the env-var defaults.
        # (io_ov was resolved before the backing block — the HOBBIT split
        # reads hot_fraction from it during backing construction.)
        io_wired = _wire_streaming_io_overrides(
            layers, mtp_stages, io_ov["expert_streaming_io_depth"], io_ov["expert_streaming_coalesce"]
        )
        if io_wired:
            logger.info(
                "Expert streaming: IO overrides wired (io_depth=%s coalesce=%s, %d linears)",
                io_ov["expert_streaming_io_depth"],
                io_ov["expert_streaming_coalesce"],
                io_wired,
            )
        # Opt-in adaptive top-k routing truncation (cumulative mass) and
        # cache-prior rerank. Exact (None/1.0, bonus 0.0) by default — no
        # patch engagement, zero overhead.
        from .adaptive_topk import (
            apply_qwen35_moe_topk_patch,
            cache_prior_from_settings,
            configure_from_settings,
        )

        thr = configure_from_settings(model_settings, model_type=estimate.model_type)
        prior = cache_prior_from_settings(model_settings)
        # Cache-prior needs app-level LRU residency as its signal (see
        # _prior_usable): refuse it at page-cache-only budgets.
        if prior > 0 and not _prior_usable(cache):
            logger.warning(
                "expert_streaming_cache_prior=%.2f ignored: no app-level LRU "
                "(budget 0 = page-cache only, resident set always empty)",
                prior,
            )
            from .adaptive_topk import configure_cache_prior

            prior = configure_cache_prior(0.0)
        if thr is not None or prior > 0:
            # configure() already refused inapplicable types (thr None);
            # the qwen hook engages here while the glm hook is inline in
            # the vendored Glm5NextMoE, so a False return only matters
            # when the type itself has no hook at all.
            from .adaptive_topk import is_topk_applicable

            engaged = apply_qwen35_moe_topk_patch()
            if (
                thr is not None
                and not engaged
                and not is_topk_applicable(estimate.model_type)
            ):
                logger.warning(
                    "Adaptive top-k threshold %.2f set but no truncation hook engaged "
                    "for model type %r (exact routing stays on)",
                    thr,
                    estimate.model_type,
                )
        # Qwen3.5/3.8 prefill eval boundary (G4): the installed qwen decoder
        # ignores _stream_eval; wrap it so long prefill chunks evaluate per
        # layer instead of pinning every layer's mini-bank in the lazy graph
        # and retaining an allocator pool big enough to evict the page cache.
        # Bit-exact; prefill-shaped calls only (decode/MTP verify stay lazy).
        from .qwen35_stream_eval import (
            apply_qwen35_moe_stream_eval,
            configure_from_settings as configure_stream_eval,
        )

        eval_on = configure_stream_eval(io_ov["expert_streaming_per_layer_eval"])
        if apply_qwen35_moe_stream_eval():
            logger.info(
                "Expert streaming: qwen per-layer eval boundary %s",
                "on" if eval_on else "off",
            )
        # Etapa E: tell the scheduler's prefill guard whether a boundary is
        # really live. Only then may it stop charging one mini-bank per MoE
        # layer; the flag stays False when the knob is off or no decoder
        # class was wrapped, so glm5_next (which honors _stream_eval inline
        # and is not wrapped here) keeps the conservative charge.
        from .qwen35_stream_eval import boundary_active as _qse_boundary_active
        from .qwen35_stream_eval import wrapped_class_names as _qse_wrapped

        _guard_info = getattr(backing, "streaming_guard_info", None)
        if isinstance(_guard_info, dict):
            _guard_info["boundary_active"] = bool(
                eval_on and _qse_boundary_active()
            )
            _guard_info["projections"] = _glu_projection_count(layers)
            if _guard_info["boundary_active"]:
                logger.info(
                    "Expert streaming: prefill guard boundary accounting on "
                    "(projections=%d, activation=%d B/token)",
                    _guard_info["projections"],
                    _guard_info.get("activation_bytes_per_token", 0),
                )
            elif _qse_wrapped():
                logger.info(
                    "Expert streaming: boundary installed but eval off/class "
                    "mismatch — per-layer bank charge kept (conservative)"
                )
        # PILOT: async router-lookahead prefetch into the staging buffer.
        # The submit hook lives in the glm5_next vendor loop (it scores
        # the next MoE layer's router against the current layer output) —
        # other families have no hook, so PILOT is a no-op for them until
        # one is added. mmap backing only; off when the RAM dict fallback
        # is in use. Opt in per model (expert_streaming_pilot=True) or
        # globally with OMLX_EXPERT_STREAMING_PILOT=1.
        import os

        pilot_requested = io_ov["expert_streaming_pilot"]
        if pilot_requested is None:
            pilot_requested = os.environ.get("OMLX_EXPERT_STREAMING_PILOT", "0") == "1"
        if pilot_requested and not isinstance(
            backing, dict
        ):
            try:
                from .prefetch import ExpertPrefetcher

                prefetcher = ExpertPrefetcher(cache)
                prefetcher.start()
                for obj_ in (
                    getattr(getattr(model, "language_model", None), "model", None),
                    getattr(model, "language_model", None),
                    model,
                ):
                    match_ = obj_ is not None and getattr(obj_, "layers", None) is layers
                    if match_:
                        obj_._expert_prefetcher = prefetcher  # type: ignore[attr-defined]
                        break
                else:
                    prefetcher.stop()
                    prefetcher = None
                    logger.warning(
                        "Expert streaming: PILOT prefetch attach point not found; disabled"
                    )
                if prefetcher is not None:
                    # P0: owned by the backing so shutdown_expert_streaming
                    # can stop its threads on unload/reload (no leak).
                    try:
                        backing._expert_prefetcher = prefetcher  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    # Wire streaming linears to their prefetcher so the
                    # demand path can drain staged np bundles before a
                    # synchronous backing read. P1: fused gate_up_proj
                    # included — it resolves bundles exactly like split
                    # projections and was silently left without prefetch.
                    wired = 0
                    for lyr_ in layers:
                        mlp_ = getattr(lyr_, "mlp", None)
                        if mlp_ is None:
                            mlp_ = getattr(lyr_, "ffn", None)
                        sm_ = getattr(mlp_, "switch_mlp", None)
                        for proj_ in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
                            lin_ = getattr(sm_, proj_, None)
                            if lin_ is not None and hasattr(lin_, "_load_expert_np"):
                                lin_._prefetcher = prefetcher  # type: ignore[attr-defined]
                                wired += 1
                    logger.info(
                        "Expert streaming: PILOT async prefetch active (%d linears wired)",
                        wired,
                    )
            except Exception as e:
                logger.warning("Expert streaming: PILOT prefetch init failed: %s", e)

        # Opt-in warm-only page-cache prefetch + mlock pins (page-cache
        # complements; replaces the LRU's "keep hot experts in RAM" role).
        # F_RDADVISE readahead (RA) rides the same prediction flow with
        # kernel hints instead of reads and defaults ON, as does the
        # prefill-hotness cache seed (SEED). The per-model readahead/seed
        # settings (autotune) override the env defaults when set.
        from . import warmer as _warmer_mod

        ra_setting = io_ov["expert_streaming_readahead"]
        ra_enabled = (
            _warmer_mod.RA_ENABLED if ra_setting is None else bool(ra_setting)
        )
        seed_setting = io_ov["expert_streaming_seed"]
        seed_enabled = (
            _warmer_mod.SEED_ENABLED if seed_setting is None else bool(seed_setting)
        )
        pins_setting = io_ov["expert_streaming_pins"]
        pins_enabled = (
            _warmer_mod.PIN_ENABLED if pins_setting is None else bool(pins_setting)
        )
        pin_gib = io_ov["expert_streaming_pin_gib"]
        pin_budget_bytes = (
            _warmer_mod.PIN_BUDGET_BYTES
            if pin_gib is None
            else max(0, min(64.0, float(pin_gib))) * 1024**3
        )

        if (
            _warmer_mod.WARM_ENABLED
            or pins_enabled
            or ra_enabled
            or seed_enabled
        ):
            try:
                glus: dict[int, Any] = {}
                for layer_idx_, layer_ in enumerate(layers):
                    moe_ = getattr(layer_, "mlp", None) or getattr(layer_, "ffn", None)
                    sm_ = getattr(moe_, "switch_mlp", None)
                    if sm_ is not None and hasattr(sm_, "down_proj"):
                        glus[layer_idx_] = sm_
                linears_by_layer: dict[int, list] = {
                    i: [
                        getattr(g, p)
                        for p in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
                        if hasattr(g, p)
                    ]
                    for i, g in glus.items()
                }
                if _warmer_mod.WARM_ENABLED:
                    warmer = _warmer_mod.PageCacheWarmer(linears_by_layer)
                elif ra_enabled:
                    warmer = _warmer_mod.PageCacheWarmer(
                        linears_by_layer, advise_only=True
                    )
                else:
                    warmer = None
                pinner = None
                if pins_enabled and backing is not None and not isinstance(backing, dict):
                    # Per-model learned-pin profile so the hot set is wired
                    # from token 1 on the next load (E3). The env path (bench
                    # override) wins when set; otherwise a .omlx sidecar in
                    # the model directory.
                    pin_profile_path = _warmer_mod.PIN_PROFILE_PATH or str(
                        Path(model_path) / ".omlx" / "expert_pin_profile.json"
                    )
                    # Fase M1: effective pin sync/regime — the model setting
                    # wins when set; env constants remain the fallback for
                    # unset models (server compatibility).
                    _pin_regime_eff = io_ov["expert_streaming_pin_regime"]
                    if _pin_regime_eff is None:
                        _pin_regime_eff = _warmer_mod.PIN_REGIME
                    _pin_sync_eff = io_ov["expert_streaming_pin_sync"]
                    if _pin_sync_eff is None:
                        _pin_sync_eff = _warmer_mod.PIN_SYNC_ENABLED
                    # Fase L: the pin profile applies only when the loaded
                    # model's fingerprint matches the one it was learned from.
                    # hot_fraction resolves inside the cold-tier branch above;
                    # resolve it again here so the fingerprint is stable even
                    # when no cold tier is active.
                    _hf_setting = io_ov.get("expert_streaming_hot_fraction")
                    if _hf_setting is None:
                        _hf_setting = _shard_mod.HOT_FRACTION_ENV or None
                    _hf = (
                        max(0.0, min(1.0, float(_hf_setting)))
                        if _hf_setting is not None
                        else None
                    )
                    _pin_fp = _expert_pin_fingerprint(
                        model_path,
                        linears_by_layer,
                        backing,
                        cold_root,
                        _hf,
                    )
                    pinner = _warmer_mod.PinController(
                        linears_by_layer,
                        backing,
                        budget_bytes=int(pin_budget_bytes),
                        observe_calls=_warmer_mod.PIN_OBSERVE_CALLS,
                        per_expert_bytes=estimate.per_expert_bytes,
                        profile_path=pin_profile_path,
                        # I6: expert width — sizes the per-token bincount
                        # payloads from on_layer_plan and validates them.
                        num_experts=estimate.experts_per_layer,
                        model_fingerprint=_pin_fp,
                        packing=_pin_fp.get("packing"),
                        pin_regime=_pin_regime_eff,
                        pin_sync=_pin_sync_eff,
                    )
                    # Save-on-unload hook: engines call save_expert_pin_profile()
                    # in stop() while the backing is still reachable.
                    backing._pin_controller = pinner  # type: ignore[attr-defined]
                recorder = None
                if seed_enabled and backing is not None and not isinstance(backing, dict):
                    recorder = _warmer_mod.PrefillHotnessRecorder(
                        linears_by_layer,
                        backing,
                        cache,
                        per_expert_bytes=estimate.per_expert_bytes,
                    )
                if warmer is not None or pinner is not None or recorder is not None:
                    hook = _warmer_mod.WarmPinHook(warmer, pinner, recorder)
                    for sm_ in glus.values():
                        sm_._warm_pins = hook  # type: ignore[attr-defined]
                    logger.info(
                        "Expert streaming: warm=%s pin=%s readahead=%s seed=%s attached (%d layers)",
                        bool(warmer and not warmer.advise_only),
                        bool(pinner),
                        bool(warmer and warmer.advise_only),
                        bool(recorder),
                        len(glus),
                    )
            except Exception as e:
                logger.warning("Expert streaming: warm/pin init failed: %s", e)
    else:
        logger.info("Expert streaming: no MoE layers converted")

    # ram-dict backing is internal only — never part of the public return
    # (existing contract; file-backed store is the only returned backing).
    return model, backing if not isinstance(backing, dict) else None


def expert_streaming_summary(cache: Any, backing: Any | None = None) -> dict:
    """P1: one-line request/bench summary of streaming health.

    Aggregates the counters the implementation already keeps (LRU hits,
    PILOT staging, stash ring, advisor, ctx fallbacks) into a single
    dict for per-request logs and the admin payload. Never raises;
    missing pieces report as None/0.
    """
    out: dict = {}
    try:
        stats = getattr(cache, "stats", None)
        hits = int(getattr(stats, "hits", 0) or 0)
        misses = int(getattr(stats, "misses", 0) or 0)
        out["lru_hit_rate"] = hits / (hits + misses) if (hits + misses) else 0.0
        out["lru_hits"] = hits
        out["lru_misses"] = misses
        out["lru_evictions"] = int(getattr(stats, "evictions", 0) or 0)
        out["lru_size"] = int(getattr(cache, "size", 0) or 0)
        out["lru_capacity"] = int(getattr(cache, "capacity", 0) or 0)
    except Exception:
        pass
    try:
        pf = getattr(backing, "_expert_prefetcher", None)
        pstats = getattr(pf, "stats", None) if pf is not None else None
        if pstats is not None:
            sub = int(pstats.get("submissions", 0) or 0)
            con = int(pstats.get("staged_consumed", 0) or 0)
            out["prefetch_precision"] = con / sub if sub else 0.0
            out["prefetch_submissions"] = sub
            out["prefetch_consumed"] = con
            out["prefetch_dropped"] = int(pstats.get("staged_dropped", 0) or 0)
    except Exception:
        pass
    try:
        spec = getattr(cache, "spec_state", None)
        sstats = getattr(spec, "stats", None) if spec is not None else None
        if sstats is not None:
            sh = int(sstats.get("stash_hits", 0) or 0)
            sm = int(sstats.get("stash_misses", 0) or 0)
            out["stash_hit_rate"] = sh / (sh + sm) if (sh + sm) else 0.0
            out["stash_hits"] = sh
            out["stash_misses"] = sm
            out["advised"] = int(sstats.get("advised", 0) or 0)
    except Exception:
        pass
    try:
        out["ctx_fallbacks"] = dict(getattr(cache, "ctx_fallback_stats", lambda: {})())
    except Exception:
        out["ctx_fallbacks"] = {}
    try:
        gov = getattr(backing, "governor", None) if backing is not None else None
        if gov is not None:
            out["governor"] = gov.summary()
    except Exception:
        pass
    try:
        from .streaming_switch import _SLOT_BANK_ENV as _slot_on

        out["slotbank_on"] = bool(_slot_on)
        out["cache_policy"] = getattr(cache, "policy", "lru")
    except Exception:
        pass
    try:
        spec = getattr(cache, "spec_state", None)
        if spec is not None:
            out["trans_updates"] = int(getattr(spec, "trans_updates", 0) or 0)
            out["trans_sources"] = len(getattr(spec, "trans", {}) or {})
            sstats = getattr(spec, "stats", None) or {}
            if isinstance(sstats, dict) and sstats.get("trans_overfetch"):
                out["trans_overfetch"] = int(sstats["trans_overfetch"])
    except Exception:
        pass
    return out


def save_transition_profile(backing: Any) -> None:
    """Persist the learned (layer, expert) transition table, if any.

    Writes ``<model>/.omlx/expert_transition.json`` with a config-sha
    fingerprint; a mismatch on load ignores the profile (never a silent
    apply). Best-effort; failures only debug-log.
    """
    try:
        if backing is None or isinstance(backing, dict):
            return
        spec = getattr(backing, "spec_state", None)
        if spec is None or not getattr(spec, "trans_updates", 0):
            return
        import hashlib
        import json

        model_path = Path(getattr(backing, "model_path", "") or "")
        if not model_path.is_dir():
            return
        cfg = model_path / "config.json"
        try:
            sha = hashlib.sha256(cfg.read_bytes()).hexdigest()[:16]
        except Exception:
            sha = None
        payload = spec.to_payload()
        payload["model"] = model_path.name
        payload["config_sha"] = sha
        dest = model_path / ".omlx" / "expert_transition.json"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(payload))
        except Exception:
            logger.debug("expert transition save failed", exc_info=True)
    except Exception:
        pass


def load_transition_profile(backing: Any, spec: Any) -> int:
    """Load a persisted transition table into *spec*; returns sources."""
    try:
        import hashlib
        import json

        if backing is None or isinstance(backing, dict) or spec is None:
            return 0
        model_path = Path(getattr(backing, "model_path", "") or "")
        src = model_path / ".omlx" / "expert_transition.json"
        if not src.is_file():
            return 0
        payload = json.loads(src.read_text())
        cfg = model_path / "config.json"
        try:
            sha = hashlib.sha256(cfg.read_bytes()).hexdigest()[:16]
        except Exception:
            sha = None
        if payload.get("config_sha") != sha or payload.get("model") != model_path.name:
            logger.info("expert transition profile fingerprint mismatch — ignored")
            return 0
        n = spec.load_payload(payload)
        if n:
            logger.info("expert transition profile loaded (%d sources)", n)
        return n
    except Exception:
        return 0


def shutdown_expert_streaming(backing: Any) -> None:
    """Release MoE streaming resources held by *backing* (P0).

    Stops the PILOT prefetcher threads, persists nothing (profiles are
    saved separately), and closes shard fds/mmaps. Idempotent; safe to
    call with None or a RAM-dict backing. Engines call this in stop()
    and before replacing the model on reload so threads/fds never leak
    across model lifetimes.
    """
    if backing is None or isinstance(backing, dict):
        return
    # Persist the learned transition table before threads/fds die.
    try:
        save_transition_profile(backing)
    except Exception:
        pass
    try:
        prefetcher = getattr(backing, "_expert_prefetcher", None)
        if prefetcher is not None:
            try:
                prefetcher.stop()
            except Exception:
                pass
            try:
                backing._expert_prefetcher = None  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass
    try:
        close = getattr(backing, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def save_expert_pin_profile(engine: Any) -> None:
    """Persist the learned pin profile of a streaming engine, if any.

    Called from the engine ``stop()`` paths while the backing store (and the
    PinController attached to it) is still reachable — before teardown drops
    the references. Never raises: a failed save only costs the learned hot
    set, never correctness.
    """
    for holder in (
        engine,
        getattr(engine, "_model", None),
        getattr(engine, "_vlm_model", None),
    ):
        if holder is None:
            continue
        backing = getattr(holder, "_expert_streaming_backing", None)
        pinner = getattr(backing, "_pin_controller", None)
        if pinner is not None:
            try:
                pinner.save_profile()
            except Exception:
                logger.debug(
                    "Expert streaming: pin profile save failed", exc_info=True
                )
            return


__all__ = [
    "apply_expert_streaming_patch",
    "convert_model_to_streaming",
    "save_expert_pin_profile",
    "is_supported_model_type",
    "normalize_model_type",
    "SUPPORTED_TYPES",
]
