# SPDX-License-Identifier: Apache-2.0
"""Model -> streaming conversion pipeline.

Everything that turns a resident MoE checkpoint into streaming layers lives
here: stacked-key resolution, the per-module switch-MLP rewrite, the
``convert_model_to_streaming`` orchestrator, and the transition-profile
persistence for its speculation state. Settings/budget resolution
(``_io_overrides``, ``resolve_budget_bytes`` ...) lives in ``settings.py``
(re-exported through the package ``__init__`` — ``moe_expert_offload`` and
the engines import it from there) and is lazily imported inside the
functions that need it, so this module is import-safe at package-init
time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from ._env import env_bool, env_str
from .model_hooks import (
    DEFAULT_PREFIX_TEMPLATES,
    find_moe_container,
    find_moe_owner,
    find_mtp_stages,
    hooks_for,
)

logger = logging.getLogger(__name__)

# SwitchGLU projection attribute names, canonical order. gate_up_proj is
# the fused gate+up spelling — a module carries it OR gate_proj+up_proj.
_PROJ_ATTRS = ("gate_proj", "up_proj", "down_proj", "gate_up_proj")


def _config_sha(model_path: Any) -> str | None:
    """sha256[:16] of the checkpoint config — the profile fingerprint
    field (None when the file is unreadable)."""
    try:
        return hashlib.sha256(
            (Path(model_path) / "config.json").read_bytes()
        ).hexdigest()[:16]
    except Exception:
        return None


def _expert_pin_fingerprint(
    model_path: str | Path,
    linears_by_layer: dict[int, list],
    backing: Any,
    cold_root: Any,
    hot_fraction: float | None,
) -> dict:
    """Profile-identity fields for a loaded model.

    A v2 pin profile applies only when these fields match the model: a
    mismatch logs and ignores the profile (never a silent apply). The
    fingerprint covers the checkpoint (config hash), the source/cold
    packing, the HOBBIT hot fraction and the profile format version.
    """
    fp: dict = {
        "model": Path(model_path).name,
        "profile_format": 2,  # keep in sync with warmer.PROFILE_VERSION
        "config_sha": _config_sha(model_path),
    }
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
    attr_chain: tuple[str, ...] = ("mlp", "ffn"),
) -> int:
    """Attach per-model IO pool / coalesce overrides to streaming linears.

    Returns the number of linears wired (0 when both overrides are unset —
    the module env defaults stay in effect).
    """
    if io_depth is None and coalesce is None:
        return 0
    from .bank_io import io_pool_for

    pool = io_pool_for(io_depth) if io_depth is not None else None
    wired = 0
    targets = list(layers or []) + list(mtp_stages or [])
    for _i, moe in _iter_moe(targets, attr_chain):
        sm = getattr(moe, "switch_mlp", None)
        if sm is None:
            continue
        for proj in _PROJ_ATTRS:
            lin = getattr(sm, proj, None)
            if lin is not None and hasattr(lin, "_io_pool_override"):
                if pool is not None:
                    lin._io_pool_override = pool  # type: ignore[attr-defined]
                if coalesce is not None:
                    lin._coalesce_override = bool(coalesce)  # type: ignore[attr-defined]
                wired += 1
    return wired


# SwitchGLU bank key prefixes per main layer. The registry owns the list
# (``ModelHooks.prefix_templates``) so a family can reorder or narrow it;
# this alias keeps the shared default importable from the package root.

def _candidate_stacked_keys(
    layer_idx: int,
    proj: str,
    suffix: str,
    templates: tuple[str, ...] | None = None,
) -> list[str]:
    return [
        f"{template.format(i=layer_idx)}.{proj}.{suffix}"
        for template in (templates or DEFAULT_PREFIX_TEMPLATES)
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


def _stacked_key_candidates(backing: Any, mid: str) -> list[str]:
    """All weight-map keys containing *mid* — built once per (backing, mid).

    A 3-component bucket index covers the canonical terminal spelling
    (``switch_mlp.<proj>.<suffix>``); the first miss on a given mid runs
    one full scan and folds exotic spellings (mid as a strict
    substring of a longer component) into the same bucket. Every later
    call for that mid is O(bucket), not O(|weight map|).
    """
    idx = getattr(backing, "_stacked_key_index", None)
    if idx is None:
        idx = {}
        wm0 = getattr(backing, "_weight_map", {}) or {}
        for k in wm0:
            pos = k.find("switch_mlp.")
            if pos < 0:
                continue
            parts = k[pos:].split(".", 3)
            if len(parts) >= 3:
                idx.setdefault(".".join(parts[:3]), []).append(k)
        try:
            backing._stacked_key_index = idx
        except Exception:
            pass  # exotic backing without attribute slots: per-call index
    hit = idx.get(mid)
    if hit is not None:
        return hit
    wm = getattr(backing, "_weight_map", {}) or {}
    out = [k for k in wm if mid in k]
    idx[mid] = out
    return out


def _resolve_stacked_key(
    candidates: list[str],
    proj: str,
    suffix: str,
    backing: Any | None,
    needle: str,
    *,
    required: bool = True,
) -> str:
    """Pick the checkpoint key for one stacked bank.

    Prefers exact candidates present in the weight map, then any key
    containing *needle* (layer/scope disambiguation) plus the
    ``switch_mlp.<proj>.<suffix>`` middle. With a real (non-empty) weight
    map a missing *required* key fails the conversion now — a first-
    candidate fallback could point the streaming linear at a key
    the checkpoint does not have, surfacing only on the first fetch.
    RAM dicts / empty maps keep the first-candidate fallback; optional
    suffixes (``biases``) pass ``required=False`` since mxfp checkpoints
    legitimately omit them.
    """
    wm: dict = {}
    if backing is not None and hasattr(backing, "_weight_map"):
        wm = getattr(backing, "_weight_map", {}) or {}
        for cand in candidates:
            if cand in wm:
                return cand
        mid = f"switch_mlp.{proj}.{suffix}"
        bucket = _stacked_key_candidates(backing, mid)
        for k in bucket:
            if needle in k:
                return k
        # Exotic-spelling fallback: the bucket index covers the canonical
        # ``switch_mlp.<proj>.<suffix>`` terminal; a key where mid is a
        # strict substring of a longer component (e.g. ``.weights``) lands
        # in a different bucket. One full scan per (mid, needle) miss;
        # hits merge into the live bucket so the next lookup stays
        # O(bucket).
        scanned = getattr(backing, "_stacked_key_scanned", None)
        if scanned is None:
            scanned = set()
            try:
                backing._stacked_key_scanned = scanned
            except Exception:
                pass
        tag = (mid, needle)
        if tag not in scanned:
            scanned.add(tag)
            bucket_keys = frozenset(bucket)
            extra = [k for k in wm if needle in k and mid in k and k not in bucket_keys]
            if extra:
                # bucket IS idx[mid] — extending it folds the exotic
                # spelling into the index for repeat lookups.
                bucket.extend(extra)
                return extra[0]
    if wm and required:
        raise ValueError(
            f"Expert streaming: no checkpoint key for {proj}.{suffix} "
            f"matching {needle!r} — tried {candidates[:2]}… "
            f"({len(wm)} keys scanned)"
        )
    return candidates[0]


def _source_packing(src: Any) -> tuple[int, int, str]:
    """Packing for one streaming projection from its source module.

    JANGQ checkpoints mix precisions inside one layer (e.g. a 2-bit gate
    with 3-bit up/down). Each streaming linear keeps its own source
    projection's packing. A projection missing any packing attr fails
    loudly — never inherits silently.
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


# Attribute paths from the model root that may carry a config object with
# hidden_size / moe_intermediate_size (LLM + VLM wrapper spellings).
_CONFIG_ATTR_PATHS = (
    ("args",),
    ("model", "args"),
    ("language_model", "args"),
    ("language_model", "model", "args"),
    ("config",),
    ("config", "text_config"),
    ("language_model", "config"),
    ("language_model", "model", "config"),
)


def _model_config_candidates(model: Any) -> list[Any]:
    """Collect potential config objects for dim resolution (LLM + VLM wrappers)."""
    candidates = []
    for path in _CONFIG_ATTR_PATHS:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            candidates.append(obj)
    return candidates


def _resolve_moe_dims(cfg_candidates: list[Any], estimate: Any = None) -> tuple[int, int]:
    """Resolve (hidden_size, moe_intermediate_size) for the converter.

    Sources, in order: the runtime model config candidates, then the
    checkpoint's own header-derived estimate (``residency`` measures the
    projection out-dims). There is deliberately no per-family defaults
    table — a guessed moe_intermediate mis-slices every expert
    bank silently, so an unresolvable pair fails loudly here instead.
    """
    found: dict[str, int | None] = {
        "hidden_size": None,
        "moe_intermediate_size": None,
    }
    for cand in cfg_candidates:
        try:
            for attr in found:
                v = getattr(cand, attr, None)
                if v is None and isinstance(cand, dict):
                    v = cand.get(attr)
                if v is not None:
                    found[attr] = int(v)
            if all(v is not None for v in found.values()):
                break
        except Exception:
            continue
    for attr in found:
        if found[attr] is None or found[attr] <= 0:
            try:
                v = int(getattr(estimate, attr, 0) or 0)
                found[attr] = v if v > 0 else None
            except (TypeError, ValueError):
                found[attr] = None
    hidden, moe_hidden = found["hidden_size"], found["moe_intermediate_size"]
    if hidden is None or moe_hidden is None:
        raise ValueError(
            "Expert streaming: could not resolve (hidden, moe_intermediate) "
            "from model config or checkpoint headers — refusing to guess "
            "bank geometry"
        )
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
    scans (e.g. ``layers.5.`` or ``mtp.2.``). ``hot_ids`` keeps
    those experts at the SOURCE packing with a dual-tier gather; absent/empty
    keeps the uniform cold tier (bits overridden to the cold packing).
    """
    import mlx.core as mx

    from .streaming_layers import (
        StreamingQuantizedSwitchLinear,
        StreamingSwitchGLU,
        StreamingSwitchLinear,
    )

    switch_mlp = getattr(moe, "switch_mlp", None)
    if switch_mlp is None:
        return False

    # (attr, projection) pairs in canonical order — every "first present
    # projection" lookup below reads projs[0].
    projs = [
        (a, p) for a in _PROJ_ATTRS if (p := getattr(switch_mlp, a, None)) is not None
    ]

    # Determine quantized vs bf16: QuantizedSwitchLinear has 'scales'
    is_quantized = any(
        hasattr(p, "scales")
        or "scales" in getattr(p, "_data", {})
        or p.__class__.__name__ == "QuantizedSwitchLinear"
        for _, p in projs
    )

    n_experts = estimate.experts_per_layer

    fused = hasattr(switch_mlp, "gate_up_proj")
    inv_scatter = getattr(switch_mlp, "inverse_scatter", False)

    # Real dims cross-check: the resolved (hidden, moe_hidden) must match
    # the source projections' out dims — shape[1] of the stacked bank is
    # the true out dim whether the tensor is packed or dense. A wrong
    # guess mis-slices every expert bank, so fail loudly here.
    try:
        _down = getattr(switch_mlp, "down_proj", None)
        _gate = getattr(
            switch_mlp, "gate_up_proj" if fused else "up_proj", None
        ) or getattr(switch_mlp, "gate_proj", None)
        _w_down = getattr(_down, "weight", None)
        _w_gate = getattr(_gate, "weight", None)
        if (
            _w_down is not None
            and _w_gate is not None
            and getattr(_w_down, "ndim", 0) == 3
            and getattr(_w_gate, "ndim", 0) == 3
        ):
            real_hidden = int(_w_down.shape[1])
            real_moe = int(_w_gate.shape[1]) // (2 if fused else 1)
            if real_hidden != int(hidden) or real_moe != int(moe_hidden):
                raise ValueError(
                    f"Expert streaming: resolved dims (hidden={hidden}, "
                    f"moe={moe_hidden}) disagree with layer {layer_idx} "
                    f"bank shapes ({real_hidden}/{real_moe})"
                )
    except ValueError:
        raise
    except Exception:
        pass

    # No silent packing defaults. A quantized projection MUST expose
    # its group_size/bits/mode — guessing 64/4/affine for an unknown future
    # quant silently mis-slices every expert bank. Fail loudly instead.
    group_size: int | None = None
    bits: int | None = None
    mode: str | None = None
    if is_quantized:
        if not projs:
            raise ValueError(
                f"Expert streaming: quantized layer {layer_idx} exposes no "
                "projection to read packing from"
            )
        pack_attr, pack_proj = projs[0]
        for _name in ("group_size", "bits", "mode"):
            if getattr(pack_proj, _name, None) is None:
                raise ValueError(
                    f"Expert streaming: quantized {pack_attr} of layer "
                    f"{layer_idx} lacks {_name!r} — refusing to guess "
                    "packing (would mis-slice expert banks)"
                )
        group_size = int(pack_proj.group_size)
        bits = int(pack_proj.bits)
        mode = str(pack_proj.mode)

    # Cold precision tier: when the backing serves this layer's banks
    # from expert_cold/, every projection of the layer computes at the
    # tier's packing — override the source bits/group size once, here, so
    # the fused and split branches both build with the tier parameters.
    # HOBBIT split: with a hot set for this layer the linear keeps the
    # SOURCE packing (hot experts) and the cold packing is attached per
    # linear below (dual gather_qmm).
    hobbit_cold_params: tuple[int, int] | None = None
    if hasattr(backing, "cold_quant_params") and projs:
        first_attr = projs[0][0]
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
    # Family hook: fused weighted-sum kernel for the sorted-decode path
    # (glm5_next/glm_moe_dsa/deepseek_v32). None keeps the plain
    # scatter-unsort tail — the caller's ndim contract then applies scores.
    try:
        from .model_hooks import resolve_weighted_sum_kernel

        streaming_glu._weighted_sum_kernel = resolve_weighted_sum_kernel(
            getattr(estimate, "model_type", "")
        )
    except Exception:
        pass

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
                    # bf16 streaming linear slices the bank row directly
                    backing[(layer_idx, proj_name)] = w  # type: ignore[index]

    # Now create streaming linears for the projections. Fused and split
    # layouts share one builder: resolve the stacked keys, read the source
    # packing, attach the bias when present.
    def _make_proj(proj_name: str, src: Any, in_dim: int, out_dim: int):
        stacked_w_key = _resolve_stacked_key(
            candidates_for(proj_name, "weight"), proj_name, "weight", backing, needle
        )
        if is_quantized:
            stacked_s_key = _resolve_stacked_key(
                candidates_for(proj_name, "scales"), proj_name, "scales", backing, needle
            )
            stacked_b_key = _resolve_stacked_key(
                candidates_for(proj_name, "biases"), proj_name, "biases", backing, needle, required=False
            )
            _p_gs, _p_bits, _p_mode = _source_packing(src)
            lin = StreamingQuantizedSwitchLinear(
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
        else:
            lin = StreamingSwitchLinear(
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
            lin.set_bias(src.bias)  # type: ignore[attr-defined]
        return lin

    if fused:
        streaming_glu.gate_up_proj = _make_proj(  # type: ignore[attr-defined]
            "gate_up_proj", switch_mlp.gate_up_proj, hidden, moe_hidden * 2
        )
        streaming_glu.down_proj = _make_proj(  # type: ignore[attr-defined]
            "down_proj", switch_mlp.down_proj, moe_hidden, hidden
        )
    else:
        for proj_name, out_dim, in_dim in [
            ("gate_proj", moe_hidden, hidden),
            ("up_proj", moe_hidden, hidden),
            ("down_proj", hidden, moe_hidden),
        ]:
            src = getattr(switch_mlp, proj_name, None)
            if src is None:
                continue
            setattr(streaming_glu, proj_name, _make_proj(proj_name, src, in_dim, out_dim))

    # HOBBIT dual-tier gate: wire the split into every quantized
    # streaming linear of this module — fused AND split projections. With a
    # hot set, the linear keeps the SOURCE packing for hot experts and the
    # cold tier's (hobbit_cold_params) for the rest; the backing already
    # routes the reads (set_hot_experts). Skipping this leaves the linear
    # uniform while the backing splits — mixed packings in one mini-bank.
    if hot_ids and is_quantized and hobbit_cold_params is not None:
        for lin_ in (getattr(streaming_glu, a, None) for a in _PROJ_ATTRS):
            if lin_ is not None and hasattr(lin_, "set_hobbit_split"):
                lin_.set_hobbit_split(hot_ids, hobbit_cold_params[0], hobbit_cold_params[1])

    # Register this layer's quantized streaming linears for the
    # next-layer advisor. MTP/DSpark stages register too — they live
    # in their own layer-id space (len(layers)+stage, no collision with the
    # trunk), so stage s advises s+1 within the draft chain exactly like
    # trunk layers do. Trunk->stage cross-talk stays off (separate spaces,
    # separate routing), which is correct: draft and verify route distinctly.
    if needle.startswith("layers.") or needle.startswith("mtp."):
        # Register on the per-conversion speculation state — a global
        # registry would let one engine's advisor target another
        # engine's linears.
        _spec_state = getattr(cache, "spec_state", None)
        if _spec_state is not None:
            _spec_state.register_linears(
                layer_idx,
                [
                    lin
                    for a in _PROJ_ATTRS
                    if isinstance(
                        (lin := getattr(streaming_glu, a, None)),
                        StreamingQuantizedSwitchLinear,
                    )
                ],
            )

    # Per-GLU projection count + the per-layer load context's projection
    # list (2 fused gate_up+down, 3 split) — consumed by the scheduler's
    # guard accounting (_glu_projection_count) and by cache slot
    # reconciliation. The global cache was sized for the majority layout;
    # the convert loop below reconciles any drift (see _reconcile_cache).
    # Discriminate on gate_up_proj, NOT on list truthiness: down_proj
    # exists in BOTH layouts, so an `or` fallback would short-circuit
    # every split GLU to n_proj=1 (3x budget under-fill).
    _lin_attrs = (
        ("gate_up_proj", "down_proj")
        if hasattr(streaming_glu, "gate_up_proj")
        else ("up_proj", "gate_proj", "down_proj")
    )
    streaming_glu.linears = [  # type: ignore[attr-defined]
        lin for a in _lin_attrs if (lin := getattr(streaming_glu, a, None)) is not None
    ]
    streaming_glu.n_proj = len(streaming_glu.linears) or 1  # type: ignore[attr-defined]

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


def _iter_moe(nodes: Any, attr_chain: tuple[str, ...] = ("mlp", "ffn")):
    """Yield (idx, moe_container) for each node carrying a MoE block.

    The one ``find_moe_container`` walk shared by the conversion scans
    (switch-MLP rewrite, IO overrides, per-instance routing stamp) —
    nodes without a MoE container are skipped here so every caller sees
    only real containers.
    """
    for idx, node in enumerate(nodes or ()):
        moe = find_moe_container(node, attr_chain)
        if moe is not None:
            yield idx, moe


def _switch_glus(layers: Any, attr_chain: tuple[str, ...] = ("mlp", "ffn")):
    """Yield (layer_idx, switch_mlp) for each layer holding a switch_mlp."""
    for layer_idx, moe in _iter_moe(layers, attr_chain):
        sm = getattr(moe, "switch_mlp", None)
        if sm is not None:
            yield layer_idx, sm


def _glu_projection_count(layers: Any, attr_chain: tuple[str, ...] = ("mlp", "ffn")) -> int:
    """Projections sharing one per-layer load context on a converted model.

    A converted StreamingSwitchGLU holds linears: 2 when the checkpoint
    fuses gate_up_proj (plus down_proj), 3 when gate/up are split. The
    scheduler's prefill guard charges min(2, projections) banks when a
    per-layer eval boundary is live, so any value >= 2 collapses to the
    same 2; only a 1 (no shared context) changes the charge. Defaults to 3
    (the conservative split case) when no converted GLU is reachable.
    """
    try:
        for _idx, sm in _switch_glus(layers, attr_chain):
            linears = getattr(sm, "linears", None)
            # An empty/unsized linears is not a converted GLU — keep
            # looking rather than reporting 0 projections.
            if linears and len(linears) > 0:
                return len(linears)
    except Exception:  # noqa: BLE001
        pass
    return 3


def _plan_conversion(
    model_path: str | Path,
    model_settings: Any | None,
    budget_bytes: int | None,
) -> tuple[Any, int, int] | None:
    """Pre-conversion gate: kill switch, estimate, budget, slot sizing.

    Returns (estimate, budget_bytes, per_slot) or None when conversion
    must not run (env opt-out / unsupported checkpoint) — the caller
    returns (model, None) untouched.
    """
    # Canonical kill switch, checked at the converter itself (not just the
    # callers): the engines gate this path on expert_streaming_enabled
    # alone. Defense in depth for every entry point, including direct
    # callers.
    if not env_bool("OMLX_MOE_EXPERT_OFFLOAD", True):
        logger.info(
            "Expert streaming: disabled by OMLX_MOE_EXPERT_OFFLOAD=0 (%s)",
            model_path,
        )
        return None

    from . import resolve_budget_bytes
    from .residency import expert_streaming_estimate

    estimate = expert_streaming_estimate(str(model_path))
    if not estimate.supported:
        logger.info("Expert streaming: model %s not supported (%s)", model_path, estimate.reason)
        return None

    if budget_bytes is None:
        budget_bytes = resolve_budget_bytes(model_settings)

    per_expert = estimate.per_expert_bytes or 0
    # One cache slot holds ONE projection's slice (gate/up/down are separate
    # keys), so slot sizing must divide by the projections per expert —
    # otherwise the LRU holds a third of the budget it was promised.
    # Per-GLU detection below: fused gate_up GLUs carry 2 projections
    # (gate_up + down), not 3 — dividing a fused model by 3 over-commits
    # the budget by 1.5x. The global per_slot uses the majority layout;
    # _convert_switch_mlp_module reconciles per-GLU drift after conversion.
    per_slot = max(1, per_expert // 3) if per_expert else 0

    # Report slots and experts separately: `slots_for_budget` counts EXPERTS
    # (it divides by the whole per_expert_bytes), while the cache counts
    # SLOTS — the units differ by n_proj.
    logger.info(
        "Expert streaming: converting %s: budget=%.2f GiB (%s), layers=%d, experts/layer=%d, "
        "per_expert=%.2f MB, per_slot=%.2f MB, slots/layer=%d, experts resident/layer=%d",
        Path(model_path).name,
        budget_bytes / 1024**3,
        "page-cache only, no LRU" if budget_bytes <= 0 else "LRU heap",
        estimate.num_moe_layers,
        estimate.experts_per_layer,
        estimate.per_expert_bytes / 1024 / 1024,
        per_slot / 1024 / 1024,
        (
            ((budget_bytes // per_slot) // estimate.num_moe_layers)
            if per_slot and estimate.num_moe_layers
            else 0
        ),
        estimate.slots_for_budget(budget_bytes),
    )

    return estimate, budget_bytes, per_slot


def _make_streaming_cache_and_governor(
    model_settings: Any | None,
    budget_bytes: int,
    per_slot: int,
    estimate: Any,
) -> tuple[Any, dict, Any]:
    """Expert cache + optional dynamic governor.

    Returns (cache, io_overrides, governor|None). io_overrides rides
    along because every later phase reads per-model settings from it.
    """
    from . import _dynamic_armed, _io_overrides

    # Import here to avoid circular
    from .expert_cache import make_expert_cache

    # IO overrides (settings/env resolution) are resolved before the cache:
    # the eviction policy setting and the governor arming below read them,
    # and the later backing-store wiring reuses the same resolved dict.
    io_ov = _io_overrides(model_settings)
    cache = make_expert_cache(
        budget_bytes, per_slot, num_layers=estimate.num_moe_layers,
        policy=io_ov.get("expert_streaming_cache_policy"),
    )

    # Dynamic residency governor (AUTO DEFAULT): hunger + pressure driven,
    # positive budget only. A budget-0 run is page-cache-only by operator
    # choice and stays so. Precedence: explicit setting (True forces on
    # over a pinned budget, False opts out) > OMLX_EXPERT_STREAMING_DYNAMIC=1
    # env > auto rule (on when the budget itself is automatic, off when
    # the user pinned an explicit budget).
    _governor = None
    _dyn_setting = io_ov.get("expert_streaming_dynamic")
    _dyn_on = _dynamic_armed(_dyn_setting, model_settings)
    if _dyn_on and budget_bytes > 0 and per_slot > 0:
        from .governor import arm_dynamic

        def _gib(key: str) -> int | None:
            """io_ov GiB knob -> bytes (already range-validated); None
            keeps arm_dynamic's env/default resolution."""
            v = io_ov.get(key)
            return int(v * 1024**3) if v is not None else None

        # initial_total_bytes: the resolved LRU budget is the real
        # starting footprint — the default base_cap*per_slot product
        # would under-report it (capacity floors at 1 while the budget
        # is the byte figure the governor must never shrink-past
        # without operator intent). base_cap only feeds the arm log's
        # "per-layer slots" field here; pass the cache's real per-layer
        # cap so the line stays truthful.
        try:
            _governor = arm_dynamic(
                cache,
                per_slot,
                max(1, int(getattr(cache, "capacity", 0) or 0))
                // max(1, int(getattr(estimate, "num_moe_layers", 0) or 0)),
                floor_default=None,
                max_budget_bytes=_gib("expert_streaming_dynamic_max_gib"),
                min_budget_bytes=_gib("expert_streaming_dynamic_min_gib"),
                stall_target=io_ov.get("expert_streaming_dynamic_stall_target"),
                num_layers=estimate.num_moe_layers,
                initial_total_bytes=int(budget_bytes),
                label="Expert streaming",
            )
        except Exception:
            # arm_dynamic itself soft-fails; this guard covers the
            # argument coercion above (int casts on duck-typed fields).
            logger.debug("governor arming failed", exc_info=True)
            _governor = None
    # Phase-aware prefill budget: explicit pin wins, else the cache
    # derives prefill caps from the decode pair (see _derive_prefill_caps).
    # The pin is applied AFTER the slot reconciliation below — converting
    # GiB to slots with the pre-reconciliation per_slot (per_expert // 3)
    # over-pins fused models (2 projections) by 1.5x.

    return cache, io_ov, _governor


def _resolve_cold_tier_root(model_path, model_settings) -> "Path | None":
    """Cold precision tier resolution → the active tier dir or None.

    expert_streaming_cold_tier ("2".."8") routes expert reads to
    <model>/expert_cold/ — a requantized full expert set that cuts the
    bytes per token pinning decode to the NVMe I/O floor. Partial tiers
    are rejected: the uniform-packing assumption the linears build on
    would silently break.
    """
    cold_root = None
    cold_setting = getattr(model_settings, "expert_streaming_cold_tier", None)
    # Accept any 2..8-bit label and validate against the
    # tier's own __metadata__ (omlx_cold_bits).
    # Mismatch disables with a warning, never silently.
    _cold_bits_label = str(cold_setting).strip() if cold_setting else ""
    if cold_setting and _cold_bits_label.isdigit() and 2 <= int(_cold_bits_label) <= 8:
        from .shard_bank import _cold_tier_status_dir

        # Deploy-time override: point the tier at an arbitrary
        # directory (a read-only model volume, a second SSD, a
        # sandboxed checkout) instead of <model>/expert_cold. The
        # runtime only requires the tier SHARDS to be complete
        # (cold_tier_status checks whichever dir is used).
        _cr_env = env_str("OMLX_EXPERT_STREAMING_COLD_ROOT", None)
        cold_root = Path(_cr_env) if _cr_env else None
        cold_dir = cold_root if cold_root is not None else Path(model_path) / "expert_cold"
        ok, why = _cold_tier_status_dir(cold_dir, Path(model_path))
        if ok:
            # Validate the requested label against the tier metadata.
            # The import sits outside the try so a regression fails
            # loudly instead of silently skipping validation.
            from .residency import _safetensors_header as _cold_hdr

            try:
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
        # An unparsable label must warn rather than fall through silently.
        logger.warning(
            "Expert streaming: cold tier %r not understood "
            "(want 2..8) — disabled",
            cold_setting,
        )
    return cold_root


def _resolve_hot_fraction(io_ov: dict) -> float | None:
    """HOBBIT hot fraction for this model.

    Contract (UI/bench): None/unset = UNIFORM tier — the split is opt-in
    per model, like the tier itself. The env default
    (OMLX_EXPERT_STREAMING_HOT_FRACTION) stays the bench/developer
    override and wins only when the setting is unset.
    """
    from . import shard_bank as _shard_mod

    hf = io_ov.get("expert_streaming_hot_fraction")
    if hf is None:
        hf = _shard_mod.HOT_FRACTION_ENV or None
    return max(0.0, min(1.0, float(hf))) if hf is not None else None


def _apply_hobbit_split(
    backing, model_path, io_ov, estimate, cold_root
) -> tuple[dict[int, set], float | None]:
    """HOBBIT per-expert hot/cold split: with a cold tier
    active, the top fraction of experts per layer (by learned
    pin-profile frequency) keeps the ORIGINAL packing while the
    rest compute at the tier. No profile = uniform cold tier.

    Returns (per-layer hot id sets, resolved hot fraction) — the
    fraction is resolved once here so the pin fingerprint downstream
    records the SAME value the split used (empty dict without a
    split; the fraction still reports what was resolved)."""
    from . import shard_bank as _shard_mod

    hot_ids_by_layer: dict[int, set] = {}
    hot_fraction = _resolve_hot_fraction(io_ov)
    if cold_root is not None:
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
    return hot_ids_by_layer, hot_fraction


def _build_expert_backing(
    model_path: str | Path,
    model_settings: Any | None,
    io_ov: dict,
    estimate: Any,
    use_file_backing: bool,
) -> tuple[Any, str, dict, Any, float | None]:
    """SSD/RAM backing construction: cold tier, spill absorb, expert bank,
    HOBBIT hot/cold split, guard-info stamp.

    Returns (backing, backing_kind, hot_ids_by_layer, cold_root,
    hot_fraction). Raises RuntimeError when file backing was requested
    but cannot be built — never silently retains expert banks in RAM.
    """
    cold_root = None
    # Backing store
    backing = None
    backing_kind = "ram"
    # HOBBIT split state: populated only when a complete cold tier
    # exists AND a learned pin profile provides frequencies; otherwise the
    # convert keeps the uniform cold tier semantics. hot_fraction is the
    # resolved knob value (also recorded in the pin fingerprint).
    hot_ids_by_layer: dict[int, set] = {}
    hot_fraction: float | None = None
    if use_file_backing:
        try:
            from .shard_bank import ExpertBackingStore

            cold_root = _resolve_cold_tier_root(model_path, model_settings)
            backing = ExpertBackingStore(model_path, cold_root=cold_root)
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
            hot_ids_by_layer, hot_fraction = _apply_hobbit_split(
                backing, model_path, io_ov, estimate, cold_root
            )
            # Guard metadata for the scheduler's prefill chunk sizing:
            # the lazy chunk forward holds every MoE layer's assembled
            # mini-bank until the chunk-end eval, so the peak carries
            # ~one bank per layer simultaneously. Without this term the
            # guard under-predicts and admits chunks whose real peak
            # reaches ~26 GB on qwen4_exp (48 layers x ~215 uniq experts
            # x ~2.5 MB). boundary_active starts False — the per-layer
            # bank charge is the safe default, relaxed only once a
            # per-layer eval boundary is actually installed (set after
            # conversion). projections is the number of projections
            # sharing one per-layer load context (2 fused gate_up+down,
            # 3 split); the guard charges min(2, projections) banks.
            # activation_bytes_per_token: one materialized bf16/fp16
            # layer output per token; 0 when the config hid
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
            # Fail clean: a model admitted only because SSD streaming fits
            # must not silently retain every expert bank in RAM (OOM).
            # The engine surfaces this instead of proceeding to materialize.
            raise RuntimeError(
                f"Expert streaming: SSD backing creation failed for {model_path}: {e}"
            ) from e

    # RAM dict path: only for explicit use_file_backing=False (unit tests).
    # Production (file backing) either has an mmap backing by now or raised
    # above — never silently retain all banks in RAM.
    if backing is None:
        if use_file_backing:
            raise RuntimeError(
                f"Expert streaming: SSD backing missing for {model_path} "
                "(refusing RAM fallback that would OOM)"
            )
        backing = {}
        backing_kind = "ram-dict"

    return backing, backing_kind, hot_ids_by_layer, cold_root, hot_fraction


def _find_decoder_layers(model: Any) -> tuple[Any, Any]:
    """Locate the decoder layer list (LLM + VLM wrapper spellings).

    Returns (layers, layers_owner) or (None, None) — the caller aborts
    conversion on None.
    """
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
        # A live backing with zero converted layers makes the engine
        # believe streaming is active (serialization + guard) while
        # materialize_lazy_state then evaluates every expert bank — the
        # silent-OOM path the feature exists to avoid. Return no backing.
        return None, None

    return layers, layers_owner


def _convert_moe_layers(
    layers: Any,
    layers_owner: Any,
    model: Any,
    _hooks: Any,
    _moe_chain: tuple[str, ...],
    _templates: tuple[str, ...] | None,
    backing: Any,
    backing_kind: str,
    cache: Any,
    estimate: Any,
    hidden: int,
    moe_hidden: int,
    hot_ids_by_layer: dict,
) -> tuple[int, Any]:
    """Convert main decoder + MTP/DSpark stage banks.

    Returns (converted, mtp_stages); the converted count already
    includes any MTP-stage conversions.
    """
    # One worklist: main decoder layers first (whichever chain attribute
    # holds a switch_mlp is the MoE container — ``mlp`` for GLM/Qwen,
    # ``ffn`` for DeepSeek V4), then DeepSeek V4 MTP/DSpark stages, which
    # carry their own SwitchGLU banks (mtp.<stage>.ffn on DSpark
    # checkpoints, mtp.<stage>.block.ffn on the legacy MTPBlock layout).
    # Streaming them keeps the ~3 GB/stage banks out of RAM on
    # low-memory hosts. MTP stages live next to the decoder stack, but
    # not always on the same owner that holds it: glm5_next VLM resolves
    # layers through the root Model.layers property while the draft
    # hangs off language_model.mtp. Walk nearby owners before giving up.
    mtp_stages = find_mtp_stages((layers_owner, model), _hooks.mtp_owner_chain)
    n_trunk = len(layers)
    worklist = [
        (
            layer,
            layer_idx,
            (
                lambda proj, suffix, _i=layer_idx: _candidate_stacked_keys(
                    _i, proj, suffix, templates=_templates
                )
            ),
            f"layers.{layer_idx}.",
            hot_ids_by_layer.get(layer_idx),
        )
        for layer_idx, layer in enumerate(layers)
        if layer is not None
    ] + [
        (
            stage,
            n_trunk + stage_idx,
            (
                lambda proj, suffix, _s=stage_idx: _mtp_candidate_stacked_keys(
                    _s, proj, suffix, trunk_layers=n_trunk
                )
            ),
            f"mtp.{stage_idx}.",
            None,
        )
        for stage_idx, stage in enumerate(mtp_stages or ())
        if stage is not None
    ]

    converted = 0
    mtp_converted = 0
    for node, conv_idx, candidates_for, needle, hot_ids in worklist:
        moe = find_moe_container(node, _moe_chain)
        if moe is None:
            continue
        if _convert_switch_mlp_module(
            moe,
            conv_idx,
            candidates_for=candidates_for,
            needle=needle,
            backing=backing,
            backing_kind=backing_kind,
            cache=cache,
            estimate=estimate,
            hidden=hidden,
            moe_hidden=moe_hidden,
            layer=find_moe_owner(node, moe, _moe_chain),
            hot_ids=hot_ids,
        ):
            converted += 1
            if needle.startswith("mtp."):
                mtp_converted += 1
    if mtp_converted:
        logger.info(
            "Expert streaming: converted %d/%d MTP/DSpark stage MoE banks",
            mtp_converted,
            len(mtp_stages),
        )

    return converted, mtp_stages


def _reconcile_cache(
    cache: Any,
    _governor: Any,
    layers: Any,
    _moe_chain: tuple[str, ...],
    per_expert: int,
    budget_bytes: int,
    io_ov: dict,
    converted: int,
) -> None:
    """Post-conversion slot/governor/prefill-pin reconciliation."""
    # The estimate counts MTP stages as MoE layers; when the runtime MTP is
    # inactive (no model.mtp) fewer layers were converted — rebalance the
    # per-layer LRU split so the converted layers keep a fair share.
    if converted and cache.num_layers != converted and cache.capacity > 0:
        cache.num_layers = converted
        cache._per_layer_cap = max(1, cache.capacity // converted)  # type: ignore[attr-defined]

    # Reconcile cache slots with the converted projection layout. The
    # cache was sized for 3 projections/expert (split); every converted
    # GLU records its real n_proj (2 fused, 3 split). A uniform fused
    # model sized //3 over-commits the budget by 1.5x — resize so
    # capacity * per_slot <= budget. Mixed layouts keep the majority
    # sizing (slots are fungible across projections) and log.
    if converted and per_expert and budget_bytes > 0:
        try:
            _n_projs: list[int] = []
            for _i, _sm in _switch_glus(layers, _moe_chain):
                _np = getattr(_sm, "n_proj", None)
                if isinstance(_np, int) and _np > 0:
                    _n_projs.append(_np)
            if _n_projs:
                from collections import Counter as _Counter

                _majority = _Counter(_n_projs).most_common(1)[0][0]
                _want_slot = max(1, per_expert // _majority)
                if _want_slot != cache.per_slot_bytes:
                    cache.per_slot_bytes = _want_slot
                    # resize() retargets capacity + per-layer cap + global cap
                    # atomically (and drains under the cache lock) instead of
                    # three unguarded field writes.
                    _want_cap = max(1, budget_bytes // _want_slot)
                    _want_per_layer = (
                        max(1, _want_cap // cache.num_layers) if cache.num_layers > 0 else None
                    )
                    cache.resize(_want_cap, _want_per_layer)
                    logger.info(
                        "Expert streaming: cache slots reconciled to %d projections "
                        "(per_slot=%d B, capacity=%d)",
                        _majority, _want_slot, cache.capacity,
                    )
                # The governor was armed with the pre-reconciliation
                # per_slot (per_expert // 3). Without this its first grow
                # computes cap = budget // stale_per_slot and pins a fused
                # model ~1.5x over the user budget, past max_budget_bytes.
                if _governor is not None:
                    _governor.reconcile_per_slot(cache.per_slot_bytes)
                if len(set(_n_projs)) > 1:
                    logger.warning(
                        "Expert streaming: mixed fused/split layouts %s — "
                        "cache sized for majority (%d proj)",
                        sorted(set(_n_projs)), _majority,
                    )
        except Exception:
            logger.debug("Expert streaming: slot reconciliation skipped", exc_info=True)

    # Phase-aware prefill budget: explicit pin wins, else the cache
    # derives prefill caps from the decode pair (see _derive_prefill_caps).
    # Runs AFTER the slot reconciliation above so the GiB->slots conversion
    # uses the reconciled per_slot_bytes — converting with the
    # pre-reconciliation value (per_expert // 3) would over-pin fused
    # models (2 projections) by 1.5x.
    if converted:
        try:
            _prefill_gib = io_ov.get("expert_streaming_prefill_budget_gib")
            if _prefill_gib is not None and cache.per_slot_bytes > 0:
                _prefill_slots = int(float(_prefill_gib) * 1024**3) // int(
                    cache.per_slot_bytes
                )
                _pin = getattr(cache, "set_prefill_budget_slots", None)
                if callable(_pin):
                    _pin(max(0, _prefill_slots))
        except Exception:
            logger.debug("prefill budget pin failed", exc_info=True)


def _stamp_instance_routing(
    model_settings, layers, mtp_stages, _moe_chain, _hooks, cache, estimate
) -> None:
    """Opt-in adaptive top-k routing truncation (cumulative mass) and
    cache-prior rerank. Exact (None/1.0, bonus 0.0) by default — no
    patch engagement, zero overhead.

    Per-model routing isolation: resolve WITHOUT touching
    the module globals, then stamp every MoE block. The shared Qwen /
    GLM classes must not observe another model's settings via global.
    """
    from . import _prior_usable
    from .adaptive_topk import (
        resolve_prior_from_settings,
        resolve_threshold_from_settings,
        set_instance_routing,
    )

    thr = resolve_threshold_from_settings(model_settings, model_type=estimate.model_type)
    prior = resolve_prior_from_settings(model_settings)
    # Cache-prior needs app-level LRU residency as its signal (see
    # _prior_usable): refuse it at page-cache-only budgets.
    if prior > 0 and not _prior_usable(cache):
        logger.warning(
            "expert_streaming_cache_prior=%.2f ignored: no app-level LRU "
            "(budget 0 = page-cache only, resident set always empty)",
            prior,
        )
        prior = 0.0
    # Stamp per-instance routing onto every MoE block of THIS model
    # (including exact None/0.0) so later models cannot move it.
    try:
        _targets = list(layers or []) + list(mtp_stages or [])
        for _i, _moe in _iter_moe(_targets, _moe_chain):
            set_instance_routing(_moe, thr, prior)
    except Exception:
        logger.debug("Expert streaming: per-instance routing stamp skipped", exc_info=True)
    if thr is not None or prior > 0:
        # resolve_* already refused inapplicable types (thr None); the
        # family's patch engages here while an inline hook (vendored
        # Glm5NextMoE) has nothing to apply — a missing patch only
        # matters when the family has no truncation hook at all.
        engaged = _hooks.apply_topk() if _hooks.apply_topk else False
        if (
            thr is not None
            and not engaged
            and not _hooks.topk_supported
        ):
            logger.warning(
                "Adaptive top-k threshold %.2f set but no truncation hook engaged "
                "for model type %r (exact routing stays on)",
                thr,
                estimate.model_type,
            )


def _wire_stream_eval_boundary(layers, io_ov, _moe_chain, _hooks, backing) -> None:
    """Qwen3.5/3.8 prefill eval boundary: the installed qwen decoder
    ignores _stream_eval; wrap it so long prefill chunks evaluate per
    layer instead of pinning every layer's mini-bank in the lazy graph
    and retaining an allocator pool big enough to evict the page cache.
    Bit-exact; prefill-shaped calls only (decode/MTP verify stay lazy).

    Then the definitive streaming guard: tell the scheduler's prefill
    guard whether a boundary is really live ON THIS MODEL.
    """
    from .qwen35_stream_eval import (
        apply_qwen35_moe_stream_eval,
        boundary_active_for_layers as _qse_boundary_for_layers,
        configure_from_settings as configure_stream_eval,
        wrapped_class_names as _qse_wrapped,
    )

    eval_on = configure_stream_eval(io_ov["expert_streaming_per_layer_eval"])
    if _hooks.stream_eval_targets and apply_qwen35_moe_stream_eval():
        logger.info(
            "Expert streaming: per-layer eval boundary %s",
            "on" if eval_on else "off",
        )
    # Definitive streaming guard: tell the scheduler's prefill guard
    # whether a boundary is really live ON THIS MODEL. Only then may it
    # stop charging one mini-bank per MoE layer; the flag stays False
    # when the knob is off or none of this model's layers runs a
    # boundary (wrapped qwen decoder class, or an inline-honoring
    # decoder such as Glm5NextDecoderLayer). The verdict is per-model:
    # a process-global "some qwen class is wrapped" must not credit
    # (or discredit) an unrelated family.

    _guard_info = getattr(backing, "streaming_guard_info", None)
    if isinstance(_guard_info, dict):
        _guard_info["boundary_active"] = bool(
            eval_on and _qse_boundary_for_layers(layers)
        )
        _guard_info["projections"] = _glu_projection_count(layers, _moe_chain)
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


def _attach_warm_pin_hooks(
    model_path,
    layers,
    io_ov,
    _moe_chain,
    backing,
    cache,
    estimate,
    cold_root,
    spec_state,
    hot_fraction,
) -> None:
    """mlock pins (PIN) + prefill-hotness cache seed (SEED): both ride
    the per-layer hook the streaming GLU calls. F_RDADVISE readahead (RA)
    has a single predictor now — the spec-state advisor in
    streaming_switch — so the resolved per-model flag is only STAMPED on
    ``spec_state.readahead_enabled`` for it here (the advisor reads
    ``getattr(spec_state, "readahead_enabled", None)``, None = env
    default). The per-model readahead/seed/pin settings (autotune)
    override the env defaults when set; *hot_fraction* is the value the
    HOBBIT split already resolved (single resolution, shared with the
    fingerprint)."""
    from . import warmer as _warmer_mod

    def _env_or(key: str, env_default: bool) -> bool:
        """Per-model bool knob: the env default applies only when the
        setting is unset."""
        setting = io_ov[key]
        return env_default if setting is None else bool(setting)

    ra_enabled = _env_or("expert_streaming_readahead", _warmer_mod.RA_ENABLED)
    if spec_state is not None:
        spec_state.readahead_enabled = ra_enabled
    seed_enabled = _env_or("expert_streaming_seed", _warmer_mod.SEED_ENABLED)
    pins_enabled = _env_or("expert_streaming_pins", _warmer_mod.PIN_ENABLED)
    pin_gib = io_ov["expert_streaming_pin_gib"]
    pin_budget_bytes = (
        _warmer_mod.PIN_BUDGET_BYTES
        if pin_gib is None
        else float(pin_gib) * 1024**3  # io_ov already validated (0, 64]
    )

    if pins_enabled or seed_enabled:
        try:
            glus: dict[int, Any] = {
                i: sm
                for i, sm in _switch_glus(layers, _moe_chain)
                if hasattr(sm, "down_proj")
            }
            linears_by_layer: dict[int, list] = {
                i: [getattr(g, p) for p in _PROJ_ATTRS if hasattr(g, p)]
                for i, g in glus.items()
            }
            pinner = None
            if pins_enabled and backing is not None and not isinstance(backing, dict):
                # Per-model learned-pin profile so the hot set is wired
                # from token 1 on the next load. The env path (bench
                # override) wins when set; otherwise a .omlx sidecar in
                # the model directory.
                pin_profile_path = _warmer_mod.PIN_PROFILE_PATH or str(
                    Path(model_path) / ".omlx" / "expert_pin_profile.json"
                )
                # Effective pin sync/regime: the model setting
                # wins when set; env constants remain the fallback for
                # unset models (server compatibility).
                _pin_regime_eff = io_ov["expert_streaming_pin_regime"]
                if _pin_regime_eff is None:
                    _pin_regime_eff = _warmer_mod.PIN_REGIME
                _pin_sync_eff = io_ov["expert_streaming_pin_sync"]
                if _pin_sync_eff is None:
                    _pin_sync_eff = _warmer_mod.PIN_SYNC_ENABLED
                # The pin profile applies only when the loaded
                # model's fingerprint matches the one it was learned from.
                # hot_fraction was resolved once by the backing build —
                # the fingerprint records the SAME value the split used,
                # even when no cold tier is active.
                _pin_fp = _expert_pin_fingerprint(
                    model_path,
                    linears_by_layer,
                    backing,
                    cold_root,
                    hot_fraction,
                )
                pinner = _warmer_mod.PinController(
                    linears_by_layer,
                    backing,
                    budget_bytes=int(pin_budget_bytes),
                    observe_calls=_warmer_mod.PIN_OBSERVE_CALLS,
                    per_expert_bytes=estimate.per_expert_bytes,
                    profile_path=pin_profile_path,
                    # Expert width — sizes the per-token bincount
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
            if pinner is not None or recorder is not None:
                hook = _warmer_mod.WarmPinHook(pinner, recorder)
                for sm_ in glus.values():
                    sm_._warm_pins = hook  # type: ignore[attr-defined]
                logger.info(
                    "Expert streaming: pin=%s seed=%s attached "
                    "(readahead=%s via spec_state, %d layers)",
                    bool(pinner),
                    bool(recorder),
                    ra_enabled,
                    len(glus),
                )
        except Exception as e:
            logger.warning(
                "Expert streaming: warm/pin init failed: %s", e, exc_info=True
            )


def _clear_cache_synced() -> None:
    """Release the reusable Metal pool, but only after a full sync.

    ``mx.clear_cache`` must never run with command buffers in flight: on
    M4 the driver panics with 'completeMemory() prepare count underflow'
    (#300/#888). ``omlx.utils.metal_sync._sync_and_clear_cache`` drains
    the stream first and takes the buffer-access lock that keeps the
    async store-cache worker from observing a half-reclaimed pool
    (#1106), so it is preferred whenever it is importable.
    """
    # The bare clear is ONLY the import-failure fallback. A failure
    # INSIDE the synced helper (lock, sync, clear) must not fall through
    # to the unsynchronized clear — that reintroduces the very 'clear
    # with command buffers in flight' race the helper exists to prevent.
    try:
        from omlx.utils.metal_sync import _sync_and_clear_cache
    except Exception:
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
        return
    try:
        _sync_and_clear_cache()
    except Exception:
        logger.warning(
            "Expert streaming: synced cache clear failed — skipping",
            exc_info=True,
        )


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
    plan = _plan_conversion(model_path, model_settings, budget_bytes)
    if plan is None:
        return model, None
    estimate, budget_bytes, per_slot = plan

    cache, io_ov, _governor = _make_streaming_cache_and_governor(
        model_settings, budget_bytes, per_slot, estimate
    )
    backing, backing_kind, hot_ids_by_layer, cold_root, hot_fraction = (
        _build_expert_backing(
            model_path, model_settings, io_ov, estimate, use_file_backing
        )
    )

    # Wire one SpeculationState (and the governor) onto cache+backing.
    # One speculation state per conversion: it hangs off the cache
    # (always) and off the backing store (file backing) so close()
    # drains the speculation workers with the readers.
    from .speculation import SpeculationState

    _spec_state = SpeculationState()
    cache.spec_state = _spec_state  # type: ignore[attr-defined]
    if _governor is not None:
        # Same reachability path as the cache: the engine finds the
        # governor through the backing at request boundaries.
        cache.governor = _governor  # type: ignore[attr-defined]
    if not isinstance(backing, dict):
        backing.spec_state = _spec_state  # type: ignore[attr-defined]
        if _governor is not None:
            backing.governor = _governor  # type: ignore[attr-defined]
        # Reload the learned transition table so the k+1 overfetch is
        # warm from token 1 (fingerprint mismatch -> ignored, never
        # silent).
        try:
            load_transition_profile(backing, _spec_state)
        except Exception:
            logger.debug(
                "Expert streaming: transition profile load failed",
                exc_info=True,
            )
        # The engine reaches the shared cache through the backing it
        # already holds (for the per-request summary log).
        try:
            backing._streaming_cache = cache  # type: ignore[attr-defined]
        except Exception:
            pass

    layers, layers_owner = _find_decoder_layers(model)
    if layers is None:
        # A live backing with zero converted layers makes the engine
        # believe streaming is active (serialization + guard) while
        # materialize_lazy_state then evaluates every expert bank — the
        # silent-OOM path the feature exists to avoid. Return no backing.
        return model, None
    hidden, moe_hidden = _resolve_moe_dims(_model_config_candidates(model), estimate)

    # Family structural spelling (attr chain, key prefixes, owner chain)
    # comes from the hook registry — resolved once for the whole walk.
    _hooks = hooks_for(estimate.model_type)
    _moe_chain = _hooks.moe_attr_chain
    _templates = _hooks.prefix_templates

    converted, mtp_stages = _convert_moe_layers(
        layers,
        layers_owner,
        model,
        _hooks,
        _moe_chain,
        _templates,
        backing,
        backing_kind,
        cache,
        estimate,
        hidden,
        moe_hidden,
        hot_ids_by_layer,
    )
    _reconcile_cache(
        cache,
        _governor,
        layers,
        _moe_chain,
        estimate.per_expert_bytes or 0,
        budget_bytes,
        io_ov,
        converted,
    )

    if not converted:
        logger.info("Expert streaming: no MoE layers converted")
        # Never hand the engine a live backing when nothing converted —
        # its presence alone enables request serialization and the prefill
        # guard while every expert bank still materializes.
        return model, None

    # Post-conversion wiring: IO overrides, adaptive top-k, eval
    # boundary, guard info, warm/pin/seed hooks.
    _clear_cache_synced()
    logger.info(
        "Expert streaming: converted %d MoE layers "
        "(backing=%s, cache_capacity=%d experts)",
        converted,
        backing_kind,
        cache.capacity,
    )
    # Per-model IO overrides (autotune): pool depth + run coalescing ride
    # the streaming linears; unset values keep the env-var defaults.
    # (io_ov was resolved before the backing block — the HOBBIT split
    # reads hot_fraction from it during backing construction.)
    io_wired = _wire_streaming_io_overrides(
        layers,
        mtp_stages,
        io_ov["expert_streaming_io_depth"],
        io_ov["expert_streaming_coalesce"],
        attr_chain=_moe_chain,
    )
    if io_wired:
        logger.info(
            "Expert streaming: IO overrides wired "
            "(io_depth=%s coalesce=%s, %d linears)",
            io_ov["expert_streaming_io_depth"],
            io_ov["expert_streaming_coalesce"],
            io_wired,
        )
    _stamp_instance_routing(
        model_settings, layers, mtp_stages, _moe_chain, _hooks, cache, estimate
    )
    _wire_stream_eval_boundary(layers, io_ov, _moe_chain, _hooks, backing)
    _attach_warm_pin_hooks(
        model_path,
        layers,
        io_ov,
        _moe_chain,
        backing,
        cache,
        estimate,
        cold_root,
        _spec_state,
        hot_fraction,
    )

    try:
        # Stamped so ensure_streaming_backing_or_raise can verify real
        # conversion — backing presence alone is not evidence.
        # converted already includes the MTP-stage conversions.
        backing.streaming_converted = converted  # type: ignore[attr-defined]
    except Exception:
        pass

    # ram-dict backing is internal only — never part of the public return
    # (the file-backed store is the only returned backing).
    return model, backing if not isinstance(backing, dict) else None


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

        model_path = Path(getattr(backing, "model_path", "") or "")
        if not model_path.is_dir():
            return
        payload = spec.to_payload()
        payload["model"] = model_path.name
        payload["config_sha"] = _config_sha(model_path)
        dest = model_path / ".omlx" / "expert_transition.json"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # tmp+replace like warmer.save_profile: a crash mid-write must
            # not leave a truncated profile behind (self-heals via the
            # config_sha fingerprint, but atomicity keeps it consistent).
            tmp = dest.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, dest)
        except Exception:
            logger.debug("expert transition save failed", exc_info=True)
    except Exception:
        pass


def load_transition_profile(backing: Any, spec: Any) -> int:
    """Load a persisted transition table into *spec*; returns sources."""
    try:
        if backing is None or isinstance(backing, dict) or spec is None:
            return 0
        model_path = Path(getattr(backing, "model_path", "") or "")
        src = model_path / ".omlx" / "expert_transition.json"
        if not src.is_file():
            return 0
        payload = json.loads(src.read_text())
        sha = _config_sha(model_path)
        if payload.get("config_sha") != sha or payload.get("model") != model_path.name:
            logger.info("expert transition profile fingerprint mismatch — ignored")
            return 0
        n = spec.load_payload(payload)
        if n:
            logger.info("expert transition profile loaded (%d sources)", n)
        return n
    except Exception:
        return 0
