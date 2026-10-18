# SPDX-License-Identifier: Apache-2.0
"""Hardware-aware model recommendations for the admin downloader.

Reads the oQ checkpoints published under one Hugging Face account, prices
every build against this Mac's memory guard and Metal limit with the same
rules the engine pool uses at load time, and groups the builds that load
into the sections of the Recommended tab.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from huggingface_hub import HfApi, get_session, hf_hub_url
from huggingface_hub.utils import build_hf_headers, hf_raise_for_status

from .. import settings as _settings
from ..cluster.planner import _config_int, _kv_bytes_per_token_per_layer
from ..process_memory_enforcer import (
    _PREFILL_HEADROOM_SAFETY,
    _wired_limit_suggestion_bytes,
    get_effective_metal_cap_bytes,
    get_iogpu_wired_limit_bytes,
    tier_reserve_bytes,
)
from ..utils.hardware import get_chip_name, get_gpu_core_count, parse_chip_info
from .hf_downloader import (
    _HF_API_TIMEOUT,
    _blob_bytes_for_repos,
    _get_hf_api,
    _list_models_stale_token_fallback,
    _safetensors_parameters,
)

logger = logging.getLogger(__name__)

RECOMMENDATION_AUTHOR = "Jundot"

_BUILD_NAME = re.compile(
    r"^(?P<family>.+)-oQ(?P<level>\d+(?:\.\d+)?)e(?P<fp16>-fp16)?(?P<mtp>-mtp)?$"
)
# Share of a model's quality an oQe level keeps, by level.
_RETENTION = {
    2.0: 0.70,
    2.5: 0.80,
    2.7: 0.84,
    3.0: 0.88,
    3.5: 0.92,
    4.0: 0.96,
    5.0: 0.98,
    6.0: 0.99,
    8.0: 1.00,
}
# Decision models served through /v1/systemone, not chat.
_EXCLUDED_TAGS = frozenset({"systemone"})
_CHAT_PIPELINES = frozenset(
    {None, "", "text-generation", "image-text-to-text", "any-to-any"}
)
_LIST_EXPAND = ["safetensors", "config", "tags", "downloads", "pipeline_tag", "sha"]

_BASE_LEVEL = 4.0
_COMFY_CONTEXT = 32 * 1024
_MIN_CONTEXT = 8 * 1024
# Models within 15% of the group leader's smarts count as similar.
_SIMILAR_SMARTS = math.log2(1.15)
_SECTION_SIZE = 3
_SMARTER_GROUP_SIZE = 2
_FASTER_MIN_RATIO = 1.3
# Builds below this level are never promoted into the best-match row.
_PROMOTE_MIN_LEVEL = 3.0

# Allocator slack and runtime buffers that are not weights or KV.
_RUNTIME_OVERHEAD_BYTES = 512 * 1024**2
# Same margins as the PLE/Engram residency estimates.
_WEIGHT_OVERHEAD = 1.05
_ENGRAM_MMAP_SLACK = 32 * 1024**2
_NGRAM_EMBEDDING_MARKER = ".ngram_embedding."

# Decode speed model: share of memory bandwidth decode reaches, Lightning MTP
# acceptance, and the PLE-on-SSD cost the engine pool warns about (~2.5x).
_BANDWIDTH_EFFICIENCY = 0.75
_MTP_SPEEDUP = 1.5
_SSD_TABLE_SPEED = {"qwen4_exp": 0.4}
_SPEED_LEVELS = ((60.0, 5), (35.0, 4), (18.0, 3), (9.0, 2))

# Unified memory bandwidth in GB/s. Missing variants scale the base chip.
_BANDWIDTH_GBPS = {
    ("M1", ""): 68,
    ("M1", "Pro"): 200,
    ("M1", "Max"): 400,
    ("M1", "Ultra"): 800,
    ("M2", ""): 100,
    ("M2", "Pro"): 200,
    ("M2", "Max"): 400,
    ("M2", "Ultra"): 800,
    ("M3", ""): 100,
    ("M3", "Pro"): 150,
    ("M3", "Max"): 400,
    ("M3", "Ultra"): 819,
    ("M4", ""): 120,
    ("M4", "Pro"): 273,
    ("M4", "Max"): 546,
    ("M5", ""): 153,
}
_VARIANT_BANDWIDTH_SCALE = {"": 1, "Pro": 2, "Max": 4, "Ultra": 8}
# Chips without native bf16 run the -fp16 builds faster.
_FP16_CHIPS = frozenset({"M1", "M2"})

_CATALOG_TTL = 3600.0
_INCOMPLETE_CATALOG_TTL = 60.0
_FAMILY_FETCH_CONCURRENCY = 8
_SHARD_FETCH_CONCURRENCY = 8


@dataclass(frozen=True)
class Build:
    """One quantized checkpoint of a family."""

    repo_id: str
    level: float
    fp16: bool
    mtp: bool
    file_bytes: int
    # N-gram / Engram lookup tables the runtime can leave on SSD.
    table_bytes: int = 0


@dataclass(frozen=True)
class Family:
    """Facts shared by every build of one model."""

    name: str
    model_type: str
    total_params: int
    active_params: int
    kv_bytes_per_token: int
    kv_fixed_bytes: int
    max_context: int
    vision: bool
    audio: bool
    downloads: int
    builds: tuple[Build, ...] = field(default=())


@dataclass(frozen=True)
class MemoryInputs:
    """Machine and memory guard state the budgets are computed from."""

    total_ram: int
    tier: str
    guard_enabled: bool
    custom_ceiling: int
    metal_cap: int


@dataclass(frozen=True)
class Budget:
    """Memory one configuration leaves for weights plus KV cache."""

    tier: str
    metal_cap: int
    ceiling: int
    headroom: float
    usable: int


@dataclass(frozen=True)
class Fit:
    """A build that loads under a budget, with its estimates."""

    family: Family
    build: Build
    resident_bytes: int
    table_on_ssd: bool
    max_context: int
    smarts: float
    tokens_per_second: float

    @property
    def comfortable(self) -> bool:
        return self.max_context >= _COMFY_CONTEXT


# =============================================================================
# Memory budgets
# =============================================================================


def _static_ceiling(memory: MemoryInputs, tier: str) -> int:
    ceiling = max(0, memory.total_ram - tier_reserve_bytes(tier, memory.total_ram))
    if tier == "custom" and memory.custom_ceiling > 0:
        ceiling = min(ceiling, memory.custom_ceiling)
    return ceiling


def budget_for(memory: MemoryInputs, tier: str, metal_cap: int) -> Budget:
    """Stable ceiling (static and Metal parts only) minus prefill headroom."""
    ceiling = _static_ceiling(memory, tier)
    if memory.guard_enabled and metal_cap > 0:
        ceiling = min(ceiling, metal_cap)
    headroom = _PREFILL_HEADROOM_SAFETY.get(tier, _PREFILL_HEADROOM_SAFETY["balanced"])
    usable = max(0, int(ceiling * headroom) - _RUNTIME_OVERHEAD_BYTES)
    return Budget(tier, metal_cap, ceiling, headroom, usable)


def memory_inputs(memory_settings: Any) -> MemoryInputs:
    """Read RAM, the memory guard settings and the effective Metal cap."""
    custom_gb = float(
        getattr(memory_settings, "memory_guard_custom_ceiling_gb", 0) or 0
    )
    return MemoryInputs(
        total_ram=int(_settings.get_system_memory()),
        tier=str(
            getattr(memory_settings, "memory_guard_tier", "balanced") or "balanced"
        ),
        guard_enabled=bool(getattr(memory_settings, "prefill_memory_guard", True)),
        custom_ceiling=int(custom_gb * 1024**3),
        metal_cap=int(get_effective_metal_cap_bytes()),
    )


def memory_scenarios(memory: MemoryInputs) -> dict[str, Budget]:
    """Budgets now, with the aggressive tier, with a raised wired limit, and both."""

    def raised(tier: str) -> int:
        suggestion = _wired_limit_suggestion_bytes(_static_ceiling(memory, tier))
        return max(memory.metal_cap, suggestion)

    return {
        "now": budget_for(memory, memory.tier, memory.metal_cap),
        "memory_guard": budget_for(memory, "aggressive", memory.metal_cap),
        "wired_limit": budget_for(memory, memory.tier, raised(memory.tier)),
        "both": budget_for(memory, "aggressive", raised("aggressive")),
    }


# =============================================================================
# Per-build estimates
# =============================================================================


def _table_goes_to_ssd(resident: int, mmap: int, budget: Budget) -> bool:
    """EnginePool's forced PLE/Engram offload rule for a stable ceiling."""
    ceiling = budget.ceiling
    if ceiling <= 0:
        return False
    line = int(ceiling * budget.headroom)
    footprint = int(resident / _WEIGHT_OVERHEAD)
    return (resident > ceiling and mmap <= ceiling) or footprint > line >= mmap


def dense_equivalent_params(family: Family) -> float:
    """Dense size an MoE behaves like, in billions of parameters."""
    total = family.total_params / 1e9
    active = family.active_params / 1e9
    return total**0.6 * active**0.4


def evaluate(
    family: Family, build: Build, budget: Budget, bandwidth_gbps: float
) -> Fit | None:
    """Price one build against a budget. None when it cannot hold 8K tokens."""
    retention = _RETENTION.get(build.level)
    if retention is None or build.file_bytes <= 0:
        return None
    resident = int(build.file_bytes * _WEIGHT_OVERHEAD)
    table_on_ssd = False
    if build.table_bytes > 0:
        mmap = int((build.file_bytes - build.table_bytes) * _WEIGHT_OVERHEAD)
        if family.model_type == "deepseek_v41":
            mmap += _ENGRAM_MMAP_SLACK
        if _table_goes_to_ssd(resident, mmap, budget):
            resident, table_on_ssd = mmap, True

    free = budget.usable - resident - family.kv_fixed_bytes
    if free <= 0:
        return None
    if family.kv_bytes_per_token > 0:
        context = min(family.max_context, free // family.kv_bytes_per_token)
    else:
        context = family.max_context
    if context < _MIN_CONTEXT:
        return None

    smarts = math.log2(dense_equivalent_params(family) * retention)
    weight_bytes = max(1, build.file_bytes - build.table_bytes)
    bytes_per_token = family.active_params * weight_bytes / max(1, family.total_params)
    tps = bandwidth_gbps * 1e9 * _BANDWIDTH_EFFICIENCY / max(1.0, bytes_per_token)
    if build.mtp:
        tps *= _MTP_SPEEDUP
    if table_on_ssd:
        tps *= _SSD_TABLE_SPEED.get(family.model_type, 1.0)
    return Fit(family, build, resident, table_on_ssd, int(context), smarts, tps)


def smarts_level(smarts: float) -> float:
    """Half-step 0.5..5 dots for the UI."""
    return max(0.5, min(5.0, round((smarts - 1.0) / 1.2 * 2) / 2))


def speed_level(tps: float) -> int:
    return next((level for floor, level in _SPEED_LEVELS if tps >= floor), 1)


# =============================================================================
# Sections
# =============================================================================


def _builds_for_chip(family: Family, prefer_fp16: bool) -> list[Build]:
    """One build per level: Lightning MTP first, fp16 only where it is faster."""
    chosen: dict[float, Build] = {}
    for build in family.builds:
        current = chosen.get(build.level)
        key = (build.fp16 == prefer_fp16, build.mtp)
        if current is None or key > (current.fp16 == prefer_fp16, current.mtp):
            chosen[build.level] = build
    return list(chosen.values())


def _by_smarts(fit: Fit) -> tuple[float, float]:
    return (-fit.smarts, -fit.tokens_per_second)


def _best_per_family(fits: list[Fit]) -> list[Fit]:
    best: dict[str, Fit] = {}
    for fit in fits:
        current = best.get(fit.family.name)
        if current is None or _by_smarts(fit) < _by_smarts(current):
            best[fit.family.name] = fit
    return list(best.values())


def rank(fits: list[Fit]) -> list[Fit]:
    """Smartest first; within 15% of a group leader, most downloaded first."""
    rest = sorted(fits, key=_by_smarts)
    ordered: list[Fit] = []
    while rest:
        leader = rest[0].smarts
        group = [fit for fit in rest if leader - fit.smarts <= _SIMILAR_SMARTS]
        group.sort(key=lambda fit: (-fit.family.downloads, *_by_smarts(fit)))
        ordered.extend(group)
        taken = {id(fit) for fit in group}
        rest = [fit for fit in rest if id(fit) not in taken]
    return ordered


def _evaluate_all(
    families: list[Family], budget: Budget, bandwidth: float, prefer_fp16: bool
) -> list[Fit]:
    fits = []
    for family in families:
        for build in _builds_for_chip(family, prefer_fp16):
            fit = evaluate(family, build, budget, bandwidth)
            if fit is not None:
                fits.append(fit)
    return fits


def build_sections(
    families: list[Family],
    scenarios: dict[str, Budget],
    bandwidth_gbps: float,
    prefer_fp16: bool,
) -> dict[str, list[dict]]:
    """Group the builds that load into the Recommended tab's sections."""
    fits = _evaluate_all(families, scenarios["now"], bandwidth_gbps, prefer_fp16)
    comfortable = [fit for fit in fits if fit.comfortable]

    best = rank(
        _best_per_family([f for f in comfortable if f.build.level == _BASE_LEVEL])
    )[:_SECTION_SIZE]
    if len(best) < _SECTION_SIZE:
        taken = {fit.family.name for fit in best}
        best += rank(
            _best_per_family(
                [
                    f
                    for f in fits
                    if f.build.level == _BASE_LEVEL and f.family.name not in taken
                ]
            )
        )[: _SECTION_SIZE - len(best)]
    if not best:
        best = rank(_best_per_family(fits))[:_SECTION_SIZE]
    if not best:
        return {"best": [], "bigger": [], "precise": [], "faster": [], "unlock": []}

    # A lower-bit build that beats the best match on both axes belongs with it.
    lead = best[0]
    dominant = [
        fit
        for fit in comfortable
        if fit.build.level >= _PROMOTE_MIN_LEVEL
        and fit.smarts > lead.smarts
        and fit.tokens_per_second >= lead.tokens_per_second
    ]
    if dominant:
        best = rank(_best_per_family(dominant + best))[:_SECTION_SIZE]
        lead = best[0]

    best_names = {fit.family.name for fit in best}
    top_smarts = max(fit.smarts for fit in best)
    top_size = max(dense_equivalent_params(fit.family) for fit in best)

    bigger = rank(
        _best_per_family(
            [
                fit
                for fit in comfortable
                if fit.family.name not in best_names
                and fit.build.level < _BASE_LEVEL
                and dense_equivalent_params(fit.family) > top_size
                and fit.smarts > top_smarts
            ]
        )
    )[:_SMARTER_GROUP_SIZE]
    precise = rank(
        _best_per_family(
            [
                fit
                for fit in comfortable
                if fit.family.name in best_names and fit.build.level > _BASE_LEVEL
            ]
        )
    )[:_SMARTER_GROUP_SIZE]
    shown = best_names | {fit.family.name for fit in bigger}
    # Faster means a smaller model, not a squeezed build of a bigger one.
    faster = rank(
        _best_per_family(
            [
                fit
                for fit in fits
                if fit.family.name not in shown
                and fit.build.level >= _BASE_LEVEL
                and fit.smarts < top_smarts
                and fit.tokens_per_second >= lead.tokens_per_second * _FASTER_MIN_RATIO
            ]
        )
    )[:_SECTION_SIZE]

    unlock = _unlock_section(
        families, scenarios, bandwidth_gbps, prefer_fp16, fits, best
    )

    best_by_family = {fit.family.name: fit for fit in best}
    return {
        "best": [_serialize(fit, lead) for fit in best],
        "bigger": [_serialize(fit, lead) for fit in bigger],
        "precise": [
            _serialize(fit, best_by_family[fit.family.name]) for fit in precise
        ],
        "faster": [_serialize(fit, lead) for fit in faster],
        "unlock": unlock,
    }


def _unlock_section(
    families: list[Family],
    scenarios: dict[str, Budget],
    bandwidth: float,
    prefer_fp16: bool,
    fits_now: list[Fit],
    best: list[Fit],
) -> list[dict]:
    """Builds that only load after raising the memory guard and/or wired limit."""
    both = scenarios["both"]
    if both.usable <= scenarios["now"].usable:
        return []
    loaded = {fit.build.repo_id for fit in fits_now}
    best_now: dict[str, float] = {}
    for fit in fits_now:
        best_now[fit.family.name] = max(
            best_now.get(fit.family.name, -math.inf), fit.smarts
        )

    def needs(fit: Fit, check) -> list[str]:
        for key in ("memory_guard", "wired_limit"):
            if check(_reevaluate(fit, scenarios[key], bandwidth)):
                return [key]
        return ["memory_guard", "wired_limit"]

    candidates = [
        fit
        for fit in _evaluate_all(families, both, bandwidth, prefer_fp16)
        if fit.comfortable
        and fit.build.repo_id not in loaded
        and fit.smarts > best_now.get(fit.family.name, -math.inf)
    ]
    rows = [
        {**_serialize(fit, best[0]), "kind": "fits", "needs": needs(fit, _comfortable)}
        for fit in rank(_best_per_family(candidates))[:_SECTION_SIZE]
    ]
    for fit in best:
        if not fit.table_on_ssd:
            continue
        resident = _reevaluate(fit, both, bandwidth)
        # Only offer it where the table on SSD costs speed.
        if (
            resident is None
            or resident.table_on_ssd
            or resident.tokens_per_second <= fit.tokens_per_second
        ):
            continue
        rows.append(
            {
                **_serialize(resident, fit),
                "kind": "table_in_memory",
                "needs": needs(fit, lambda f: f is not None and not f.table_on_ssd),
            }
        )
    return rows


def _comfortable(fit: Fit | None) -> bool:
    return fit is not None and fit.comfortable


def _reevaluate(fit: Fit, budget: Budget, bandwidth: float) -> Fit | None:
    return evaluate(fit.family, fit.build, budget, bandwidth)


def _serialize(fit: Fit, base: Fit) -> dict:
    family, build = fit.family, fit.build
    return {
        "repo_id": build.repo_id,
        "name": family.name,
        "model_type": family.model_type,
        "level": build.level,
        "fp16": build.fp16,
        "mtp": build.mtp,
        "vision": family.vision,
        "audio": family.audio,
        "moe": family.active_params < family.total_params,
        "total_params": family.total_params,
        "active_params": family.active_params,
        "downloads": family.downloads,
        "download_bytes": build.file_bytes,
        "resident_bytes": fit.resident_bytes,
        "table_on_ssd": fit.table_on_ssd,
        "table_bytes": build.table_bytes,
        "max_context": fit.max_context,
        "model_max_context": family.max_context,
        "smarts": round(fit.smarts, 3),
        "smarts_level": smarts_level(fit.smarts),
        "speed_level": speed_level(fit.tokens_per_second),
        "tokens_per_second": round(fit.tokens_per_second, 1),
        "base_name": base.family.name,
        "relative_speed": round(
            fit.tokens_per_second / max(1e-6, base.tokens_per_second), 2
        ),
    }


# =============================================================================
# Device
# =============================================================================


@functools.lru_cache(maxsize=1)
def _gpu_cores() -> int | None:
    # system_profiler takes about a second; the answer never changes.
    return get_gpu_core_count()


def device_info() -> dict:
    chip = get_chip_name()
    generation, variant = parse_chip_info(chip)
    bandwidth = _BANDWIDTH_GBPS.get((generation, variant))
    if bandwidth is None:
        base = _BANDWIDTH_GBPS.get((generation, ""), _BANDWIDTH_GBPS[("M5", "")])
        bandwidth = base * _VARIANT_BANDWIDTH_SCALE.get(variant, 1)
    return {
        "chip": chip,
        "generation": generation,
        "variant": variant,
        "gpu_cores": _gpu_cores(),
        "memory_bytes": int(_settings.get_system_memory()),
        "bandwidth_gbps": bandwidth,
        "prefers_fp16": generation in _FP16_CHIPS,
    }


# =============================================================================
# Catalog
# =============================================================================


def parse_build_name(repo_id: str) -> tuple[str, float, bool, bool] | None:
    """(family, level, fp16, mtp) from `owner/<family>-oQ<level>e[-fp16][-mtp]`."""
    match = _BUILD_NAME.match(repo_id.split("/", 1)[-1])
    if match is None:
        return None
    return (
        match["family"],
        float(match["level"]),
        bool(match["fp16"]),
        bool(match["mtp"]),
    )


def _text_config(config: dict) -> dict:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def active_params(config: dict, total_params: int) -> int:
    """Total minus the routed experts a token does not visit."""
    text = _text_config(config)
    experts = next(
        (
            v
            for k in ("num_experts", "n_routed_experts", "num_local_experts")
            if isinstance(v := text.get(k), int) and v > 0
        ),
        0,
    )
    top_k = next(
        (
            v
            for k in ("num_experts_per_tok", "top_k_experts", "moe_top_k")
            if isinstance(v := text.get(k), int) and v > 0
        ),
        0,
    )
    width = text.get("moe_intermediate_size")
    hidden = text.get("hidden_size")
    layers = text.get("num_hidden_layers")
    if not (experts and top_k and width and hidden and layers) or top_k >= experts:
        return total_params
    dense_layers = int(text.get("first_k_dense_replace") or 0)
    dense_layers += len(text.get("mlp_only_layers") or [])
    moe_layers = max(0, layers - dense_layers)
    routed = moe_layers * experts * 3 * hidden * width
    routed = min(routed, int(total_params * 0.98))
    return max(1, int(total_params - routed * (1 - top_k / experts)))


def kv_cache_cost(config: dict) -> tuple[int, int]:
    """(bytes per token, fixed bytes) of an fp16 KV cache.

    Linear-attention layers keep a small fixed state and are skipped.
    Sliding-window layers cost at most one window. Layers that reuse an
    earlier layer's KV store nothing.
    """
    text = _text_config(config)
    layers = _config_int(config, "num_hidden_layers", 0)
    if layers <= 0:
        return 0, 0
    if int(text.get("kv_lora_rank") or 0) > 0:
        return _kv_bytes_per_token_per_layer(config) * layers, 0

    layer_types = text.get("layer_types")
    if not (isinstance(layer_types, list) and len(layer_types) == layers):
        interval = int(text.get("full_attention_interval") or 0)
        layer_types = [
            (
                "full_attention"
                if not interval or (i + 1) % interval == 0
                else "linear_attention"
            )
            for i in range(layers)
        ]
    shared = int(text.get("num_kv_shared_layers") or 0)
    if 0 < shared < layers:
        layer_types = layer_types[: layers - shared]

    kv_heads = int(
        text.get("num_key_value_heads") or text.get("num_attention_heads") or 0
    )
    head_dim = int(text.get("head_dim") or 0)
    if head_dim <= 0 and kv_heads:
        hidden = int(text.get("hidden_size") or 0)
        heads = int(text.get("num_attention_heads") or 0)
        head_dim = hidden // heads if heads else 0
    if kv_heads <= 0 or head_dim <= 0:
        return 0, 0
    global_heads = int(text.get("num_global_key_value_heads") or kv_heads)
    global_dim = int(text.get("global_head_dim") or head_dim)
    window = int(text.get("sliding_window") or 0)

    per_token = 0
    fixed = 0
    for layer_type in layer_types:
        if layer_type == "linear_attention":
            continue
        if layer_type == "sliding_attention" and window > 0:
            fixed += window * kv_heads * head_dim * 4
        else:
            per_token += global_heads * global_dim * 4
    return per_token, fixed


def _read_json(
    endpoint: str | None, repo_id: str, filename: str, revision: str | None
) -> dict:
    """Fetch a small JSON file into memory; the HF cache would keep a weightless entry."""
    url = hf_hub_url(repo_id, filename, revision=revision, endpoint=endpoint)
    response = get_session().get(
        url,
        headers=build_hf_headers(token=False),
        timeout=_HF_API_TIMEOUT,
        follow_redirects=True,
    )
    hf_raise_for_status(response)
    return response.json()


def _ngram_table_bytes(
    api: HfApi, endpoint: str | None, repo_id: str, revision: str | None
) -> int:
    """Bytes of Qwen4-Exp PLE n-gram tables, from the shards that hold them."""
    index = _read_json(endpoint, repo_id, "model.safetensors.index.json", revision)
    files = {
        filename
        for name, filename in (index.get("weight_map") or {}).items()
        if _NGRAM_EMBEDDING_MARKER in name
    }

    def shard_bytes(filename: str) -> int:
        metadata = api.parse_safetensors_file_metadata(
            repo_id, filename, revision=revision, token=False, timeout=_HF_API_TIMEOUT
        )
        return sum(
            tensor.data_offsets[1] - tensor.data_offsets[0]
            for name, tensor in metadata.tensors.items()
            if _NGRAM_EMBEDDING_MARKER in name
        )

    with ThreadPoolExecutor(max_workers=_SHARD_FETCH_CONCURRENCY) as pool:
        return sum(pool.map(shard_bytes, sorted(files)))


def _model_type(config: dict) -> str:
    return str(config.get("model_type") or "").replace("-", "_").lower()


def _family_facts(
    api: HfApi, endpoint: str | None, repo_id: str, revision: str | None
) -> dict:
    """Config-derived facts for one family, read from its representative build."""
    config = _read_json(endpoint, repo_id, "config.json", revision)
    model_type = _model_type(config)
    facts: dict[str, Any] = {"config": config, "model_type": model_type}
    if model_type == "deepseek_v41":
        report = (config.get("omlx_deepseek_v41") or {}).get(
            "quantization_report"
        ) or {}
        facts["table_bytes"] = int(
            (report.get("engram") or {}).get("tensor_bytes") or 0
        )
        facts["total_params"] = int(
            (report.get("remaining_weights") or {}).get("logical_parameters") or 0
        )
    return facts


# Facts and per-build table bytes keyed by (repo_id, sha); a re-upload changes the sha.
_facts_cache: dict[tuple[str, str | None], dict] = {}
_table_cache: dict[tuple[str, str | None], int] = {}
_catalog: tuple[float, float, list[Family]] | None = None
_catalog_lock = asyncio.Lock()


def _assemble_family(
    name: str,
    rows: list[tuple[Any, tuple]],
    facts: dict,
    sizes: dict,
    tables: dict[str, int],
) -> Family | None:
    """`tables` maps each build with known lookup-table bytes to that size."""
    config = facts["config"]
    representative = rows[0][0]
    if representative.id not in tables:
        return None
    table_bytes = tables[representative.id]
    total = int(facts.get("total_params") or 0)
    if total <= 0:
        params = _safetensors_parameters(getattr(representative, "safetensors", None))
        total = sum(int(v) for v in params.values() if isinstance(v, int))
        rep_bytes = sizes.get(representative.id, 0)
        if table_bytes and rep_bytes > table_bytes:
            # Listing totals include the lookup tables; drop their byte share.
            total = int(total * (1 - table_bytes / rep_bytes))
    if total <= 0:
        return None
    per_token, fixed = kv_cache_cost(config)
    builds = tuple(
        Build(row.id, level, fp16, mtp, int(sizes[row.id]), tables[row.id])
        for row, (_, level, fp16, mtp) in rows
        if sizes.get(row.id, 0) > 0 and row.id in tables
    )
    if not builds:
        return None
    return Family(
        name=name,
        model_type=facts["model_type"],
        total_params=total,
        active_params=active_params(config, total),
        kv_bytes_per_token=per_token,
        kv_fixed_bytes=fixed,
        max_context=_config_int(
            config, "max_position_embeddings", 32768, maximum=100_000_000
        ),
        vision=isinstance(config.get("vision_config"), dict),
        audio=isinstance(config.get("audio_config"), dict),
        downloads=sum(int(getattr(row, "downloads", 0) or 0) for row, _ in rows),
        builds=builds,
    )


async def _fetch_families(author: str) -> tuple[list[Family], bool]:
    """List the author's oQ builds and resolve each family. Returns (families, complete)."""
    api, endpoint = _get_hf_api()
    rows, _ = await asyncio.wait_for(
        asyncio.to_thread(
            _list_models_stale_token_fallback,
            api,
            {"author": author, "limit": 1000, "expand": _LIST_EXPAND},
        ),
        timeout=_HF_API_TIMEOUT,
    )
    groups: dict[str, list[tuple[Any, tuple]]] = {}
    for row in rows:
        parsed = parse_build_name(row.id)
        if parsed is None or parsed[1] not in _RETENTION:
            continue
        if _EXCLUDED_TAGS & set(getattr(row, "tags", None) or ()):
            continue
        if getattr(row, "pipeline_tag", None) not in _CHAT_PIPELINES:
            continue
        groups.setdefault(parsed[0], []).append((row, parsed))

    # Separate slots, so slow table reads do not hold back the family configs.
    config_slots = asyncio.Semaphore(_FAMILY_FETCH_CONCURRENCY)
    table_slots = asyncio.Semaphore(_FAMILY_FETCH_CONCURRENCY)

    async def cached(cache: dict, row: Any, read, slots: asyncio.Semaphore) -> Any:
        key = (row.id, getattr(row, "sha", None))
        if key in cache:
            return cache[key]
        async with slots:
            try:
                value = await asyncio.wait_for(
                    asyncio.to_thread(read, api, endpoint, row.id, key[1]),
                    timeout=_HF_API_TIMEOUT * 3,
                )
            except Exception as e:  # noqa: BLE001 - one build must not sink the list
                logger.warning("Could not read model facts for %s: %s", row.id, e)
                return None
        cache[key] = value
        return value

    async def resolve(rows_: list[tuple[Any, tuple]]) -> tuple[dict, dict] | None:
        # The most downloaded build stands in for the family.
        rows_.sort(key=lambda item: -(getattr(item[0], "downloads", 0) or 0))
        representative = rows_[0][0]
        # The n-gram table bits follow the oQ level, so every Qwen4-Exp build
        # is read, alongside the family config.
        listed = getattr(representative, "config", None) or {}
        table_rows = (
            [row for row, _ in rows_] if _model_type(listed) == "qwen4_exp" else []
        )
        facts, *values = await asyncio.gather(
            cached(_facts_cache, representative, _family_facts, config_slots),
            *(
                cached(_table_cache, row, _ngram_table_bytes, table_slots)
                for row in table_rows
            ),
        )
        if facts is None:
            return None
        table_bytes = int(facts.get("table_bytes") or 0)
        tables = {row.id: table_bytes for row, _ in rows_}
        for row, value in zip(table_rows, values):
            if value is None:
                del tables[row.id]
            else:
                tables[row.id] = value
        return facts, tables

    names = list(groups)
    sizes, resolved = await asyncio.gather(
        _blob_bytes_for_repos(
            api, [row.id for rows_ in groups.values() for row, _ in rows_]
        ),
        asyncio.gather(*(resolve(groups[name]) for name in names)),
    )
    families = []
    complete = True
    for name, result in zip(names, resolved):
        if result is None:
            complete = False
            continue
        facts, tables = result
        family = _assemble_family(name, groups[name], facts, sizes, tables)
        if family is None:
            complete = False
            continue
        if len(family.builds) < len(groups[name]):
            complete = False
        families.append(family)
    return families, complete


async def load_families(
    author: str = RECOMMENDATION_AUTHOR,
) -> tuple[list[Family], float, bool]:
    """Cached families. Returns (families, fetched_at, stale)."""
    global _catalog
    async with _catalog_lock:
        now = time.time()
        if _catalog is not None and now < _catalog[1]:
            return _catalog[2], _catalog[0], False
        try:
            families, complete = await _fetch_families(author)
        except Exception:
            if _catalog is not None:
                logger.warning(
                    "Model catalog refresh failed; serving the last copy", exc_info=True
                )
                return _catalog[2], _catalog[0], True
            raise
        ttl = _CATALOG_TTL if complete else _INCOMPLETE_CATALOG_TTL
        _catalog = (now, now + ttl, families)
        return families, now, False


def _budget_dict(budget: Budget) -> dict:
    return {
        "tier": budget.tier,
        "metal_cap_bytes": budget.metal_cap,
        "ceiling_bytes": budget.ceiling,
        "usable_bytes": budget.usable,
    }


async def recommendations(memory_settings: Any) -> dict:
    """Device, memory budgets and sectioned recommendations for the admin UI."""
    families, fetched_at, stale = await load_families()
    device = await asyncio.to_thread(device_info)
    memory = memory_inputs(memory_settings)
    scenarios = memory_scenarios(memory)
    now = scenarios["now"]
    static = _static_ceiling(memory, memory.tier)
    sections = build_sections(
        families, scenarios, device["bandwidth_gbps"], device["prefers_fp16"]
    )
    metal_binds = memory.guard_enabled and 0 < now.metal_cap < static
    return {
        "device": device,
        "memory": {
            "total_bytes": memory.total_ram,
            "tier": memory.tier,
            "guard_enabled": memory.guard_enabled,
            "reserve_bytes": tier_reserve_bytes(memory.tier, memory.total_ram),
            "aggressive_reserve_bytes": tier_reserve_bytes(
                "aggressive", memory.total_ram
            ),
            "static_ceiling_bytes": static,
            "wired_limit_set": get_iogpu_wired_limit_bytes() > 0,
            "binding": "metal" if metal_binds else "static",
            "suggested_wired_limit_mb": scenarios["both"].metal_cap // 1024**2,
            "scenarios": {key: _budget_dict(b) for key, b in scenarios.items()},
        },
        "sections": sections,
        "author": RECOMMENDATION_AUTHOR,
        "fetched_at": fetched_at,
        "stale": stale,
    }
