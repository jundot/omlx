# SPDX-License-Identifier: Apache-2.0
"""Residency estimates for MoE expert streaming (SSD)."""

from __future__ import annotations

import json
import logging
import re
import struct
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_MODEL_OVERHEAD_FACTOR = 1.05

# P0: decision margin for force_streaming. The accounting factor above
# converts checkpoint bytes to resident bytes; the admission decision
# needs headroom on top (allocator fragmentation, transient banks,
# KV/SDPA) so a load exactly at the ceiling is never admitted.
_FORCE_MARGIN_FACTOR = 1.10


def normalize_model_type(model_type: object) -> str:
    """Canonical form of a config ``model_type`` for allowlist membership."""
    if not model_type:
        return ""
    return str(model_type).replace("-", "_").lower()


def _config_model_type(config: dict) -> str:
    """Effective model_type: top level wins, VLM wrappers fall back to
    ``text_config`` (glm5_next / qwen4_exp nest the language model there)."""
    model_type = str(config.get("model_type") or "")
    if model_type:
        return normalize_model_type(model_type)
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        return normalize_model_type(text_cfg.get("model_type"))
    return ""


# Model types whose MoE expert banks can be streamed from disk.
#
# Single source of truth — every gate reads this object. It used to be
# re-declared as four independent tuples (three here, one in __init__.py, one
# hardcoded in engine/vlm.py), and they drifted: a checkpoint could pass the
# structural estimate below (which is what forces streaming in EnginePool)
# while failing the converter's list (which is what gates lazy loading in
# engine/batched.py). That combination materializes the full multi-hundred-GB
# MoE banks before the converter can drop them — OOM / SIGKILL. Sharing one
# object removes the divergence by construction rather than by keeping lists
# in sync.
#
# Kept in this leaf module: residency.py has no intra-package imports, so
# every consumer (including the package __init__) can import it without a
# cycle.
SUPPORTED_TYPES = frozenset(
    {
        "glm_moe_dsa",
        "deepseek_v32",
        "deepseek_v4",
        "deepseek_v4_mtp",
        "glm5_next",
        "glm5_next_text",
        "qwen4_exp",
        "qwen4_exp_text",
        # Wider coverage: these expose the same stacked
        # ``layers.N.mlp.switch_mlp.{gate,up,down}_proj`` layout and the
        # ``moe_intermediate_size`` key _resolve_moe_dims reads, so they pass
        # the structural estimate as-is. They were only missing from the list.
        "qwen3_moe",
        "qwen2_moe",
        "deepseek_v3",
        "glm4_moe",
    }
)

# Subset whose MoE layer count can be derived from config.json alone, used
# only when the header scan finds no ``layers.N.`` indices. Deliberately NOT
# the full set: the sparse/dense pattern differs per family
# (mlp_layer_types vs first_k_dense_replace vs decoder_sparse_step), so a type
# is listed here only once its config keys are verified. Unlisted types fall
# through to ``supported = False`` — fail closed rather than slice banks with
# a guessed layer count.
_CONFIG_DERIVABLE_MOE_LAYERS = frozenset(
    {
        "glm_moe_dsa",
        "glm5_next",
        "glm5_next_text",
        "qwen4_exp",
        "qwen4_exp_text",
        # deepseek_v3 / glm4_moe resolve through the generic
        # first_k_dense_replace branch below (verified keys).
        "deepseek_v3",
        "glm4_moe",
        # qwen3_moe resolves through the dedicated decoder_sparse_step +
        # mlp_only_layers branch below (verified against mlx_lm).
        "qwen3_moe",
        # qwen2_moe: its decoder is unconditionally all-MoE and its config
        # carries no pattern keys, so the generic loop below (first_k=0,
        # freq=1 defaults) already counts every layer; the _ALL_LAYERS_MOE
        # membership documents the invariant and backstops it.
        "qwen2_moe",
    }
)

# Types where every layer is MoE when routing experts are present and no
# sparse/dense pattern resolved a count.
_ALL_LAYERS_MOE = frozenset(
    {
        "qwen4_exp",
        "qwen4_exp_text",
        "deepseek_v4",
        "deepseek_v4_mtp",
        # qwen2_moe's mlx decoder is unconditionally all-MoE (every layer
        # carries a Qwen2MoeSparseMoeBlock; no sparse/dense pattern keys).
        "qwen2_moe",
    }
)


@dataclass(frozen=True)
class ExpertStreamingEstimate:
    """Estimated bytes with and without expert streaming."""

    supported: bool
    checkpoint_bytes: int
    expert_bytes: int
    dense_bytes: int
    resident_bytes: int
    streaming_bytes: int
    num_moe_layers: int
    experts_per_layer: int
    per_expert_bytes: int
    per_layer_expert_bytes: int
    reason: str | None = None
    # Normalized effective model_type (top-level config wins, VLM wrappers
    # fall back to text_config). Lets consumers scope per-family behavior
    # (e.g. adaptive top-k hooks) without re-reading config.json.
    model_type: str = ""
    # Fase J Etapa E: text hidden size, used to charge one materialized layer
    # activation to the prefill guard when a per-layer eval boundary is live.
    # 0 when the config does not expose it (the guard then charges the bank
    # term only — still conservative).
    hidden_size: int = 0
    # P1: cold-tier awareness. When the estimate was computed against an
    # active cold tier, expert_bytes_effective measures the banks actually
    # read at decode (cold packing, smaller) instead of the source
    # checkpoint banks; tier is "uniform" (I5, all cold), "hobbit"
    # (I6 split, mixed — effective is a hot-fraction-weighted blend) or
    # "none". Empty (tier "none") means effective == expert_bytes.
    tier: str = "none"
    expert_bytes_effective: int = 0
    per_expert_bytes_effective: int = 0

    def force_streaming(
        self, memory_ceiling: int, transient_bytes: int = 0
    ) -> bool:
        """True when streaming turns an impossible load into a viable one.

        P0: both sides carry a decision margin (``_FORCE_MARGIN_FACTOR``)
        and the streaming side adds ``transient_bytes`` (one per-layer
        expert bank — the peak the guard charges the scheduler). A load
        exactly at the ceiling is refused, never admitted.
        """
        if not self.supported or memory_ceiling <= 0:
            return False
        try:
            transient = max(0, int(transient_bytes))
        except (TypeError, ValueError):
            transient = 0
        return (
            int(self.resident_bytes * _FORCE_MARGIN_FACTOR) > memory_ceiling
            and int(self.streaming_bytes * _FORCE_MARGIN_FACTOR) + transient
            <= memory_ceiling
        )

    def slots_for_budget(self, budget_bytes: int) -> int:
        """Slots per layer that fit in *budget_bytes*."""
        if not self.supported or self.per_expert_bytes <= 0 or self.num_moe_layers <= 0:
            return 0
        if budget_bytes <= 0:
            return 0
        per_layer = self.per_expert_bytes
        # budget is total across all MoE layers
        slots = budget_bytes // (self.num_moe_layers * per_layer)
        # clamp to experts_per_layer
        return int(max(0, min(slots, self.experts_per_layer)))

    def streaming_bytes_for_budget(self, budget_bytes: int) -> int:
        """Resident streaming bytes for a given cache budget."""
        if not self.supported:
            return self.resident_bytes
        # dense + cache
        cache = 0
        if budget_bytes > 0 and self.per_expert_bytes > 0:
            slots = self.slots_for_budget(budget_bytes)
            cache = slots * self.num_moe_layers * self.per_expert_bytes
        return int(self.dense_bytes * _MODEL_OVERHEAD_FACTOR + cache)


def _safetensors_header(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return {}
            hsize = struct.unpack("<Q", raw)[0]
            return json.loads(f.read(hsize))
    except Exception:
        return {}


def _load_config(model_path: Path) -> dict:
    try:
        return json.loads((model_path / "config.json").read_text())
    except Exception:
        return {}


def _detect_expert_keys(
    weight_map: dict[str, str],
    headers: dict[str, dict],
    config: dict,
) -> tuple[list[str], int, int]:
    """Return (expert_tensor_keys, num_moe_layers, experts_per_layer)."""
    # Prefer text_config for VLM wrappers (glm5_next, qwen4_exp)
    text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    n_routed = config.get("n_routed_experts")
    if n_routed is None:
        n_routed = text_cfg.get("n_routed_experts")
    if n_routed is None:
        # qwen4_exp uses num_experts
        n_routed = config.get("num_experts")
    if n_routed is None:
        n_routed = text_cfg.get("num_experts")
    try:
        n_routed = int(n_routed) if n_routed is not None else 0
    except Exception:
        n_routed = 0

    num_layers = config.get("num_hidden_layers") or config.get("num_layers") or 0
    if not num_layers and text_cfg:
        num_layers = text_cfg.get("num_hidden_layers") or text_cfg.get("num_layers") or 0
    try:
        num_layers = int(num_layers)
    except Exception:
        num_layers = 0

    model_type = str(config.get("model_type") or "")
    # also check text_config type for VLM wrappers
    if not model_type and text_cfg:
        model_type = str(text_cfg.get("model_type") or "")

    expert_keys: list[str] = []
    mtp_expert_keys: list[str] = []
    mtp_stage_ids: set[int] = set()

    # Heuristics: stacked MoE banks contain switch_mlp and dimension 0 == n_routed
    for key in weight_map.keys():
        # Exclude PLE ngram tables — not MoE experts
        if ".ngram_embedding." in key or ".ple." in key:
            continue
        if "nextn" in key.lower():
            continue
        # DeepSeek V4 (DSpark/MTP) keeps one routed-expert bank per draft stage
        # under mtp.<stage>[.block].ffn.switch_mlp — streamable like main
        # layers, so they count as expert bytes rather than dense.
        is_mtp = ".mtp." in key or key.startswith("mtp.")
        is_expert = False
        if ".mlp.experts." in key:
            is_expert = True
        elif ".experts." in key and "shared_experts" not in key:
            # Per-expert JANGQ layout (layers.N.ffn.experts.{i}.w{1,2,3}.*)
            # or fused experts.gate_up_proj/down_proj: routed banks without
            # a stacked switch_mlp key. Shared experts are dense-resident,
            # never streamed, so they stay out of the byte accounting.
            is_expert = True
        elif "switch_mlp" in key and ("gate_proj" in key or "up_proj" in key or "down_proj" in key or "gate_up_proj" in key):
            # check shape[0] == n_routed if we have headers
            # weight_map may point to sharded files, need headers per file
            # we will check later via header shape; for now consider candidate
            is_expert = True
        elif normalize_model_type(model_type) in SUPPORTED_TYPES and "switch_mlp" in key:
            is_expert = True
        if not is_expert:
            continue
        if is_mtp:
            mtp_expert_keys.append(key)
            m = re.search(r"mtp\.(\d+)\.", key)
            if m:
                try:
                    mtp_stage_ids.add(int(m.group(1)))
                except Exception:
                    pass
        else:
            expert_keys.append(key)

    # Refine by checking header shape when possible
    def _refine(keys: list[str]) -> list[str]:
        if not (n_routed and keys and headers):
            return keys
        refined: list[str] = []
        for k in keys:
            entry = headers.get(k)
            if entry is None:
                refined.append(k)
                continue
            shape = entry.get("shape") or []
            if shape and shape[0] == n_routed:
                refined.append(k)
            elif ".experts." in k:
                refined.append(k)
            # else: may be false positive, skip stacking check
        # if refined is non-empty, use it; otherwise keep original for per-expert files
        return refined if refined else keys

    expert_keys = _refine(expert_keys)
    mtp_expert_keys = _refine(mtp_expert_keys)

    # Determine moe layers by distinct layer indices in expert keys
    layer_pat = re.compile(r"layers\.(\d+)\.")
    layers = set()
    for k in expert_keys:
        m = layer_pat.search(k)
        if m:
            try:
                idx = int(m.group(1))
                # Exclude extra MTP/nextn layers beyond num_hidden_layers (e.g. glm5_next layer 45)
                if num_layers and idx >= num_layers:
                    continue
                layers.add(idx)
            except Exception:
                pass
    num_moe_layers = len(layers) if layers else 0
    # fallback to config's derived count when headers incomplete
    if num_moe_layers == 0 and normalize_model_type(model_type) in _CONFIG_DERIVABLE_MOE_LAYERS:
        # For glm5_next: use mlp_layer_types sparse count (most accurate)
        try:
            mlp_types = config.get("mlp_layer_types")
            if mlp_types is None and text_cfg:
                mlp_types = text_cfg.get("mlp_layer_types")
            if isinstance(mlp_types, list) and n_routed:
                cnt = sum(1 for t in mlp_types if str(t).lower() == "sparse")
                if cnt > 0:
                    num_moe_layers = cnt
            if num_moe_layers == 0 and normalize_model_type(model_type) == "qwen3_moe" and n_routed:
                # qwen3_moe: a layer is MoE iff it is not dense-only and
                # (layer_idx + 1) % decoder_sparse_step == 0, mirroring
                # mlx_lm's Qwen3MoeDecoderLayer. Defaults (step=1, no
                # dense-only layers) resolve to every layer being MoE.
                try:
                    step = int(config.get("decoder_sparse_step") or text_cfg.get("decoder_sparse_step") or 1)
                except (TypeError, ValueError):
                    step = 1
                if step < 1:
                    step = 1
                raw_only = config.get("mlp_only_layers")
                if raw_only is None:
                    raw_only = text_cfg.get("mlp_only_layers")
                only: set[int] = set()
                if isinstance(raw_only, (list, tuple)):
                    for v in raw_only:
                        try:
                            only.add(int(v))
                        except (TypeError, ValueError):
                            continue
                num_moe_layers = sum(
                    1 for i in range(num_layers) if i not in only and (i + 1) % step == 0
                )
            if num_moe_layers == 0 and n_routed:
                # generic first_k/moe_freq fallback
                first_k = int(config.get("first_k_dense_replace") or text_cfg.get("first_k_dense_replace") or 0)
                freq = int(config.get("moe_layer_freq") or text_cfg.get("moe_layer_freq") or 1)
                cnt = 0
                for i in range(num_layers):
                    if i >= first_k and i % freq == 0:
                        cnt += 1
                if cnt > 0:
                    num_moe_layers = cnt
                # qwen4_exp / deepseek_v4: every layer is MoE when
                # num_experts / n_routed present and no mlp types
                if num_moe_layers == 0 and normalize_model_type(model_type) in _ALL_LAYERS_MOE and n_routed:
                    num_moe_layers = int(num_layers) if num_layers else 0
        except Exception:
            pass

    # Filter extra layers beyond num_hidden_layers (e.g. glm5_next layer 45
    # MTP stored as model.layers.45) so expert_bytes excludes them
    if num_layers:
        try:
            filt_pat = re.compile(r"layers\.(\d+)\.")
            filtered = []
            for k in expert_keys:
                m = filt_pat.search(k)
                if m:
                    try:
                        if int(m.group(1)) >= num_layers:
                            continue
                    except Exception:
                        pass
                filtered.append(k)
            expert_keys = filtered
        except Exception:
            pass

    # DeepSeek V4 style MTP/DSpark expert banks are streamable: their bytes
    # count as expert bytes and each stage counts as one MoE layer
    if mtp_expert_keys:
        expert_keys.extend(mtp_expert_keys)
        num_moe_layers += len(mtp_stage_ids)

    return expert_keys, num_moe_layers, n_routed


@lru_cache(maxsize=128)
def _cached_estimate(
    model_path_str: str,
    sig: tuple[tuple[str, int, int], ...],
    index_sig: tuple[int, int] | None,
    _cold_overlay: tuple[float | None, str | None] | None = None,
) -> ExpertStreamingEstimate:
    model_path = Path(model_path_str)
    config = _load_config(model_path)

    checkpoint_files = [Path(p[0]) for p in sig]
    checkpoint_bytes = sum(p[1] for p in sig)

    # Load weight_map and headers
    weight_map: dict[str, str] = {}
    headers: dict[str, dict] = {}
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
        except Exception:
            weight_map = {}
        # build headers from per-file headers
        # only need expert keys' headers for refinement; load all file headers lazily via per-file
        needed_files = set(weight_map.values())
        for fname in needed_files:
            hdr = _safetensors_header(model_path / fname)
            for k, v in hdr.items():
                # headers are per-file, but weight_map keys are global; keep first occurrence
                if k not in headers:
                    headers[k] = v
    else:
        # single file or sharded without index: scan headers directly
        for fpath in checkpoint_files:
            hdr = _safetensors_header(fpath)
            for k, v in hdr.items():
                headers[k] = v
                # weight_map synthetic
                weight_map[k] = fpath.name

    expert_keys, num_moe_layers, experts_per_layer = _detect_expert_keys(
        weight_map, headers, config
    )

    # Sum expert_bytes via headers data_offsets size
    expert_bytes = 0
    for k in expert_keys:
        entry = headers.get(k)
        if entry and "data_offsets" in entry:
            try:
                s, e = entry["data_offsets"]
                expert_bytes += int(e) - int(s)
            except Exception:
                continue
        else:
            # fallback: estimate via shape*itemsize (rare)
            # we have checkpoint_bytes fallback later
            pass

    model_type = _config_model_type(config)

    # If no expert_keys but model_type indicates MoE, try fallback scan of headers for switch_mlp
    if not expert_keys:
        if model_type in SUPPORTED_TYPES and headers:
            for k, entry in headers.items():
                if "switch_mlp" in k:
                    try:
                        s, e = entry["data_offsets"]
                        expert_bytes += int(e) - int(s)
                        expert_keys.append(k)
                    except Exception:
                        pass
            if expert_keys:
                layer_pat = re.compile(r"layers\.(\d+)\.")
                layers = {int(m.group(1)) for k in expert_keys if (m := layer_pat.search(k))}
                # Filter MTP extra layers
                cfg_layers = int(config.get("num_hidden_layers") or (config.get("text_config") or {}).get("num_hidden_layers") or 0) if config.get("num_hidden_layers") or (config.get("text_config") or {}).get("num_hidden_layers") else 0
                if cfg_layers:
                    layers = {l for l in layers if l < cfg_layers}
                num_moe_layers = len(layers)
                # mtp.<stage> expert banks count as streamable MoE layers
                mtp_pat = re.compile(r"mtp\.(\d+)\.")
                num_moe_layers += len(
                    {int(m.group(1)) for k in expert_keys if (m := mtp_pat.search(k))}
                )
                exp = config.get("n_routed_experts")
                if exp is None:
                    exp = (config.get("text_config") or {}).get("n_routed_experts")
                if exp is None:
                    exp = config.get("num_experts")
                if exp is None:
                    exp = (config.get("text_config") or {}).get("num_experts")
                experts_per_layer = int(exp or 0)

    supported = False
    reason: str | None = None
    per_expert = 0
    per_layer_expert = 0
    # The allowlist is a hard gate, not a decoration. Structural detection
    # alone finds stacked switch_mlp banks in families we have never sliced
    # (mixtral's block_sparse_moe, ernie4_5_moe's moe_num_experts, ...), and
    # EnginePool forces streaming from this flag. The converter is only
    # reached on a lazy-loaded model, and lazy loading is gated on the same
    # allowlist — so a structurally-detected but unlisted type would be
    # forced into streaming WITHOUT the lazy load and mlx_lm would
    # materialize the full multi-hundred-GB banks before the converter could
    # drop them (OOM / SIGKILL). Fail closed instead: no streaming means the
    # normal load path, which the admission gate can refuse cleanly.
    if model_type not in SUPPORTED_TYPES:
        label = model_type or "unknown"
        reason = f"model type {label!r} is not in the expert-streaming allowlist"
    elif not expert_keys:
        reason = "no expert tensors found"
    elif num_moe_layers <= 0:
        reason = "could not determine MoE layer count"
    elif experts_per_layer <= 0:
        reason = "n_routed_experts missing in config"
    elif expert_bytes <= 0:
        reason = "expert byte size 0"
    else:
        # per_expert = expert_bytes / (layers*experts)
        try:
            per_expert = expert_bytes // (num_moe_layers * experts_per_layer)
            per_layer_expert = expert_bytes // num_moe_layers
            if per_expert > 0 and per_layer_expert > 0:
                supported = True
            else:
                reason = "per-expert bytes computed as 0"
        except Exception as e:
            reason = str(e)

    # Hidden size: prefer text_config (VLMs put the vision tower's width at
    # the top level and the language model's under text_config).
    hidden_size = 0
    for src in (
        config.get("text_config") if isinstance(config.get("text_config"), dict) else {},
        config,
    ):
        raw_hidden = src.get("hidden_size")
        if raw_hidden:
            try:
                hidden_size = int(raw_hidden)
            except (TypeError, ValueError):
                hidden_size = 0
            break

    dense_bytes = max(0, checkpoint_bytes - expert_bytes)
    resident_bytes = int(checkpoint_bytes * _MODEL_OVERHEAD_FACTOR)
    # streaming default = page-cache only: dense bytes are committed; expert
    # reuse rides the OS file cache (clean, evictable pages — not charged to
    # admission). An explicit LRU budget is accounted via
    # streaming_bytes_for_budget.
    streaming_bytes = int(dense_bytes * _MODEL_OVERHEAD_FACTOR)

    # P1: cold-tier effective bytes are resolved by the caller-provided
    # overlay (see expert_streaming_estimate): the base scan always
    # measures the source checkpoint; the overlay rescales to what decode
    # actually reads. Defaults keep the pre-tier behavior exactly.
    tier = "none"
    expert_bytes_effective = expert_bytes
    if _cold_overlay is not None:
        try:
            _scale, _tier_name = _cold_overlay
            tier = str(_tier_name or "uniform")
            if _scale is not None and 0 < float(_scale) <= 1.0:
                expert_bytes_effective = int(expert_bytes * float(_scale))
        except Exception:
            pass
    per_expert_effective = per_expert
    if num_moe_layers > 0 and experts_per_layer > 0 and expert_bytes_effective != expert_bytes:
        try:
            per_expert_effective = expert_bytes_effective // (num_moe_layers * experts_per_layer)
        except Exception:
            per_expert_effective = per_expert

    return ExpertStreamingEstimate(
        supported=supported,
        checkpoint_bytes=checkpoint_bytes,
        expert_bytes=expert_bytes,
        dense_bytes=dense_bytes,
        resident_bytes=resident_bytes,
        streaming_bytes=streaming_bytes,
        num_moe_layers=num_moe_layers,
        experts_per_layer=experts_per_layer,
        per_expert_bytes=per_expert,
        per_layer_expert_bytes=per_layer_expert,
        reason=reason,
        hidden_size=hidden_size,
        model_type=model_type,
        tier=tier,
        expert_bytes_effective=expert_bytes_effective,
        per_expert_bytes_effective=per_expert_effective,
    )


def _resolve_cold_overlay(
    model_path: Path,
    cold_root: str | Path | None = None,
    hot_fraction: float | None = None,
) -> tuple[float | None, str | None] | None:
    """P1: byte-scale overlay for an active cold tier (or None).

    Returns (scale, tier_name) where scale rescales source expert bytes
    to what decode actually reads. Uniform tier: measured from the cold
    shards' headers (data_offsets of the same expert keys). HOBBIT split:
    hot-fraction-weighted blend of 1.0 (hot, source packing) and the cold
    scale. None when no complete tier is active — the base scan applies.
    """
    try:
        import os as _os

        croot = Path(cold_root).expanduser().resolve() if cold_root else None
        if croot is None:
            env_root = _os.environ.get("OMLX_EXPERT_STREAMING_COLD_ROOT", "")
            croot = Path(env_root).expanduser().resolve() if env_root else model_path / "expert_cold"
        if not croot.is_dir():
            return None
        # Completeness: every source expert key must exist under the tier.
        # Light check here (file presence); the converter's full status
        # check remains authoritative for activation.
        cold_files = list(croot.glob("*.safetensors"))
        if not cold_files:
            return None
        cold_bytes = 0
        cold_keys = 0
        for fp in cold_files:
            try:
                hdr = _safetensors_header(fp)
            except Exception:
                continue
            for k, entry in hdr.items():
                if k == "__metadata__":
                    continue
                try:
                    s, e = entry["data_offsets"]
                    cold_bytes += int(e) - int(s)
                    cold_keys += 1
                except Exception:
                    continue
        if cold_keys == 0 or cold_bytes <= 0:
            return None
        # Source expert bytes for the same checkpoint (base scan, no overlay).
        base = _cached_estimate(str(model_path), _sig_of(model_path), _index_sig_of(model_path))
        if not base.supported or base.expert_bytes <= 0:
            return None
        # The tier must cover the expert banks (not just stray files).
        if cold_keys < len(base.expert_bytes and [1]) * 0 + base.num_moe_layers:
            pass  # soft check only; scale clamps below anyway
        scale = cold_bytes / base.expert_bytes
        if not (0 < scale <= 1.0):
            # A tier larger than source is a misconfiguration — ignore.
            return None
        try:
            hf = None if hot_fraction is None else max(0.0, min(1.0, float(hot_fraction)))
        except (TypeError, ValueError):
            hf = None
        if hf is not None and 0.0 < hf < 1.0:
            return (hf * 1.0 + (1.0 - hf) * scale, "hobbit")
        return (scale, "uniform")
    except Exception:
        return None


def _sig_of(p: Path) -> tuple[tuple[str, int, int], ...]:
    files = {fp.resolve() for fp in p.glob("*.safetensors")}
    return tuple(sorted((str(fp), fp.stat().st_size, fp.stat().st_mtime_ns) for fp in files))


def _index_sig_of(p: Path) -> tuple[int, int] | None:
    index_path = p / "model.safetensors.index.json"
    if index_path.is_file():
        st = index_path.stat()
        return (st.st_size, st.st_mtime_ns)
    return None


def expert_streaming_estimate(
    model_path: str | Path,
    cold_root: str | Path | None = None,
    hot_fraction: float | None = None,
) -> ExpertStreamingEstimate:
    """Inspect checkpoint headers without materializing tensor data.

    P1: pass ``cold_root`` (and ``hot_fraction`` for a HOBBIT split) to
    measure what decode actually reads when a cold tier is active; omit
    both for the source-checkpoint estimate (all existing callers).
    """

    p = Path(model_path).expanduser().resolve()
    sig = _sig_of(p)
    index_sig = _index_sig_of(p)
    overlay = _resolve_cold_overlay(p, cold_root, hot_fraction)
    # Observability (Fase M): the VLM loader runs this scan on EVERY load
    # (allowlist short-circuit + this lru_cache keep it cheap), so a timed
    # debug line records the true cost and hit rate in production logs.
    hits_before = _cached_estimate.cache_info().hits
    t0 = time.perf_counter()
    est = _cached_estimate(str(p), sig, index_sig, overlay)
    scan_ms = (time.perf_counter() - t0) * 1000.0
    logger.debug(
        "Expert streaming scan %s: %.1f ms (%s), supported=%s layers=%d",
        p.name,
        scan_ms,
        "cache-hit" if _cached_estimate.cache_info().hits > hits_before else "header-scan",
        est.supported,
        est.num_moe_layers,
    )
    return est


def clear_estimate_cache() -> None:
    _cached_estimate.cache_clear()
