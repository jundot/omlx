"""Fase 0 bench: expert streaming TTFT, steady-state tok/s, hit rate, memory, stage profile.

Usage:
    .venv/bin/python bench/bench_expert_streaming.py --model qwen --budget 1.0 --decode 96 --out bench/results/qwen_1g.json
    .venv/bin/python bench/bench_expert_streaming.py --model glm --budget 1.0 --decode 16

Protocol (B6): use --single-request and the same --decode for all A/B arms so
TTFT and tok/s are comparable. Every arm writes tokens + chunk_schedule + metal peaks.

Controls:
    OMLX_EXPERT_STREAMING_PROFILE=1  (integer per-stage profiling per layer)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL_PATHS = {
    "qwen": "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-oQ4e-mtp",
    "qwen-jang": "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-JANG_4S",
    "qwen-jang4m": "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-JANG_4M",
    "glm": "/Volumes/SSD 4TB/AI Models/GLM-5.3-Flash-oQ4e",
    "glm-jang": "/Volumes/SSD 4TB/AI Models/GLM-5.3-Flash-JANG-MTP",
    "dsv4": "/Volumes/SSD 4TB/AI Models/DeepSeek-V4-Flash-0731-oQ4e-mtp",
    "dsv4-jang": "/Volumes/SSD 4TB/AI Models/DeepSeek-V4-Flash-0731-JANG",
}
DEFAULT_ENTRIES = {
    "qwen": "Qwen3.8-Flash-Next-oQ4e-mtp",
    "qwen-jang": "Qwen3.8-Flash-Next-JANG_4S",
    "qwen-jang4m": "Qwen3.8-Flash-Next-JANG_4M",
    "glm": "GLM-5.3-Flash-oQ4e",
    "glm-jang": "GLM-5.3-Flash-JANG-MTP",
    "dsv4": "DeepSeek-V4-Flash-0731-oQ4e-mtp",
    "dsv4-jang": "DeepSeek-V4-Flash-0731-JANG",
}
PROMPTS = {
    "qwen": [{"role": "user", "content": "Hello, how are you?"}],
    "qwen-jang": [{"role": "user", "content": "Hello, how are you?"}],
    "qwen-jang4m": [{"role": "user", "content": "Hello, how are you?"}],
    "glm": [{"role": "user", "content": "Hello, how are you?"}],
    "glm-jang": [{"role": "user", "content": "Hello, how are you?"}],
    "dsv4": [{"role": "user", "content": "Hello, how are you?"}],
    "dsv4-jang": [{"role": "user", "content": "Hello, how are you?"}],
}

_FILLER = (
    "The scientist wrote a detailed report about the river ecosystem, "
    "describing how the water temperature changes with the seasons and "
    "which fish species migrate through the valley each year. "
)


def build_prompt(model_key: str, prompt_len: str) -> list[dict]:
    """Synthetic prompts: short (7 tok), 512, 2k, 8k (approximate word targets)."""
    if prompt_len == "short":
        return list(PROMPTS[model_key])
    words = {"512": 400, "2k": 1600, "8k": 6400}[prompt_len]
    content = (_FILLER * (words // 26 + 1))[: words * 7]
    return [{"role": "user", "content": content}]


class FakeEnforcer:
    memory_guard_tier = "balanced"

    def __init__(self, ceiling_gib=32.0):
        self._ceiling = int(ceiling_gib * 1024**3)

    def get_ceiling_breakdown(self):
        return {"static": self._ceiling, "dynamic": 64 * 1024**3, "metal_cap": 64 * 1024**3}

    def get_final_ceiling(self):
        return self._ceiling

    def get_admission_ceiling(self):
        return self._ceiling

    def get_admission_soft_target(self):
        return int(self._ceiling * 0.875)

    def wake(self, active=False):
        pass

    def _propagate_memory_limit(self):
        pass


def _bench_settings(
    pins: bool,
    pin_gib: float | None,
    pin_regime: str,
    budget: float,
    topk: float | None,
    prior: float | None,
    cold_tier: str | None,
    hot_fraction: float | None,
    mtp: bool,
    mtp_block: int | None,
    ane: bool,
    specprefill_draft: str | None,
    specprefill_keep: float | None,
    mtp_native: bool = False,
):
    """Fase M1: the bench's ModelSettings, wired EXPLICITLY.

    Pins arrive as model settings (pin regime + sync), never as a late
    os.environ mutation after engine load — the PinController is built
    inside get_engine, so env written later cannot be relied on.
    """
    from omlx.model_settings import ModelSettings

    return ModelSettings(
        expert_streaming_enabled=True,
        expert_streaming_budget_gib=budget,
        expert_streaming_topk_threshold=topk,
        expert_streaming_cache_prior=prior,
        expert_streaming_cold_tier=cold_tier,
        # Fase I6 HOBBIT split: top fraction of experts per layer (by
        # learned pin-profile frequency) keeps the ORIGINAL packing while
        # the rest read the cold tier. Requires --cold-tier + a profile.
        expert_streaming_hot_fraction=hot_fraction,
        # --pins (parity with the ppl harness): mlock the observed hot
        # experts and LEARN the pin profile this run persists on unload —
        # the decode-dominant hot set for the prefill x decode overlap study.
        expert_streaming_pins=pins or None,
        expert_streaming_pin_gib=(pin_gib if pin_gib is not None else 0.25)
        if pins
        else None,
        # Fase M1: explicit wiring — the controller receives these BEFORE
        # the first request; no reliance on late env mutation.
        expert_streaming_pin_regime=pin_regime if pins else None,
        expert_streaming_pin_sync=True if pins else None,
        qwen4_ple_ssd_offload=True,
        # The two speculative paths are mutually exclusive by design:
        # qwen4_exp runs native Lightning MTP (mtp_enabled); other VLM
        # types use the external-assistant path (vlm_mtp_enabled).
        vlm_mtp_enabled=mtp and not mtp_native,
        mtp_enabled=mtp and mtp_native,
        vlm_mtp_draft_block_size=mtp_block,
        qwen35_ane_prefill_enabled=ane,
        specprefill_enabled=bool(specprefill_draft),
        specprefill_draft_model=specprefill_draft,
        specprefill_keep_pct=specprefill_keep,
        # The bench prompt is 7440 tokens; the product default threshold
        # (8192) would never trigger. Score any long-prompt run.
        specprefill_threshold=2048,
    )


def _effective_config(
    *,
    git_sha: str | None,
    single_request: bool,
    decode_tokens: int,
    chunk_schedule: dict,
    budget_gib: float,
    cold_tier: str | None,
    hot_fraction: float | None,
    pins: bool,
    pinner: Any,
    model_fingerprint: Any,
    run_qd: int,
    expert_qd: int,
    prefill_qd: int,
    knobs: list[str] | None = None,
) -> dict:
    """Fase M5: the immutable effective-config block of one bench run.

    Every result carries the EFFECTIVE state (module constants read here,
    not the CLI intent), so compare_results.py can refuse A/B comparisons
    across incompatible instrumentation, schedules or cache protocols.
    """
    from omlx.patches.expert_streaming import streaming_switch as _ss
    from omlx.patches.expert_streaming import warmer as _warmer_mod
    from omlx.patches.expert_streaming.memtrace import memtrace as _mt
    from omlx.patches.expert_streaming.shard_bank import _PROFILE_READS as _prof

    return {
        "git_sha": git_sha,
        "model_fingerprint": model_fingerprint,
        "single_request": bool(single_request),
        "decode_tokens": int(decode_tokens),
        "chunk_schedule": dict(chunk_schedule),
        "budget_gib": float(budget_gib),
        "cold_tier": cold_tier,
        "hot_fraction": hot_fraction,
        "ctx_mode_policy": "hybrid" if _ss._CTX_ROLLING_ENV else "union",
        "decode_union_rows": int(_ss._DECODE_UNION_MAX_ROWS),
        "ctx_ahead": int(_ss._CTX_PREFETCH_AHEAD),
        "expert_qd": int(expert_qd),
        "run_qd": int(run_qd),
        "prefill_qd": int(prefill_qd),
        "run_merge_gap": int(_ss._RUN_MERGE_GAP),
        "ra_enabled": bool(_warmer_mod.RA_ENABLED),
        "pins_enabled": bool(pins),
        "pin_sync_effective": bool(
            getattr(pinner, "pin_sync", False) if pinner is not None else False
        ),
        "pin_regime_effective": (
            getattr(pinner, "pin_regime", None) if pinner is not None else None
        ),
        "profile_enabled": bool(_prof),
        "memtrace_enabled": bool(_mt.enabled),
        "read_sampling_mode": "profile" if _prof else "off",
        "cache_cool_protocol": "warm-page-cache",
        "cache_policy": str(getattr(_ss, "_CACHE_POLICY_ENV", "lru")),
        "transition_overfetch": os.environ.get("OMLX_EXPERT_STREAMING_TRANSITION", "1") != "0",
        "experiment_knobs": list(knobs or []),
        "active_engines": int(
            os.environ.get("OMLX_EXPERT_STREAMING_ACTIVE_ENGINES", "1")
        ),
    }


# Fase A1: phase switches shared by BOTH bench paths. The legacy path must
# never close "prefill" before the first engine.chat() — that chat RUNS the
# prefill — and the read_stats + memtrace telemetries must switch in the
# SAME order so the two agree on the boundary. All three helpers are
# unit-tested (Fase A6): the legacy flow is exactly open/switch/close
# around the two chats.
def open_phase(tel, mt6, phase: str, engine_id: str) -> None:
    """Open a phase in read telemetry and memtrace together."""
    if tel is not None and tel.enabled:
        tel.begin_phase(phase, request_id="bench-1", engine_id=engine_id)
    if mt6 is not None:
        try:
            mt6.set_context(phase=phase, request_id="bench-1", engine_id=engine_id)
        except Exception:
            pass


def switch_phase(tel, mt6, phase: str, engine_id: str) -> None:
    """Close the open read-stats scope, open a phase, then move memtrace —
    one boundary, both telemetries (Fase A1)."""
    if tel is not None and tel.enabled:
        tel.end_phase()
        tel.begin_phase(phase, request_id="bench-1", engine_id=engine_id)
    if mt6 is not None:
        try:
            mt6.set_context(phase=phase, request_id="bench-1", engine_id=engine_id)
        except Exception:
            pass


def close_phase(tel) -> None:
    """Close the open read-stats scope (memtrace keeps its teardown ctx)."""
    if tel is not None and tel.enabled:
        tel.end_phase()


# Fase A2: fields that must be present and non-empty in EVERY effective
# config — a result without them is un-comparable by construction. The
# nullable reporter fields (cold_tier, hot_fraction, model_fingerprint,
# pin_*) stay out by design: they are legitimately absent without pins.
_REQUIRED_EFFECTIVE_CONFIG_FIELDS = (
    "git_sha",
    "single_request",
    "decode_tokens",
    "chunk_schedule",
    "budget_gib",
    "ctx_mode_policy",
    "decode_union_rows",
    "ctx_ahead",
    "expert_qd",
    "run_qd",
    "prefill_qd",
    "run_merge_gap",
    "ra_enabled",
    "pins_enabled",
    "profile_enabled",
    "memtrace_enabled",
    "read_sampling_mode",
    "cache_cool_protocol",
    "active_engines",
)


def assert_effective_config_complete(cfg, *, gate: bool) -> None:
    """Fase A2 fail-high: a null or incomplete effective_config must never
    land in a gated artifact. Under --gate-tokens this aborts BEFORE any
    result is written; outside gate mode it warns loudly."""
    missing = [
        f
        for f in _REQUIRED_EFFECTIVE_CONFIG_FIELDS
        if cfg is None or cfg.get(f) in (None, "")
    ]
    if not missing:
        return
    _msg = (
        "effective_config incomplete (missing %s); a silent artifact "
        "would be un-comparable by construction" % ", ".join(missing)
    )
    if gate:
        raise SystemExit("bench aborted: " + _msg)
    print("WARNING: " + _msg)


def find_streaming_cache(vlm_model):
    layers = None
    for path in [
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("layers",),
    ]:
        cur = vlm_model
        ok = True
        for a in path:
            if not hasattr(cur, a):
                ok = False
                break
            cur = getattr(cur, a)
        if ok and cur is not None and len(cur) > 0:
            layers = cur
            break
    if layers is None:
        return None
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        sm = getattr(mlp, "switch_mlp", None) if mlp else None
        if sm is None:
            continue
        cache = getattr(sm, "_cache", None) or getattr(sm, "cache", None)
        if cache is not None:
            return cache
        for attr in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
            proj = getattr(sm, attr, None)
            if proj is not None and hasattr(proj, "cache"):
                return proj.cache
    return None


async def run(
    model_key: str,
    budget: float,
    decode: int,
    mtp: bool,
    out: str | None,
    topk: float | None = None,
    prior: float | None = None,
    cold_tier: str | None = None,
    prompt_len: str = "short",
    hot_fraction: float | None = None,
    pins: bool = False,
    mtp_block: int | None = None,
    ane: bool = False,
    mem_ceiling: float = 28.0,
    specprefill_draft: str | None = None,
    specprefill_keep: float | None = None,
    out_dir: str = "bench/results",
    single_request: bool = False,
    gate_tokens: bool = False,
    pin_gib: float | None = None,
    pin_regime: str = "decode",
    knobs: list[str] | None = None,
    mtp_depth: int | None = None,
    arm_telemetry: bool = False,
):
    from omlx.engine_pool import EnginePool
    from omlx.model_settings import ModelSettings
    from omlx.scheduler import SchedulerConfig
    from omlx.utils.proc_memory import get_phys_footprint
    import mlx.core as mx

    # Storage-roofline derivation support: arm demand-read telemetry in
    # runtime (no env-var restart needed) so read_stats carries decode-phase
    # bytes for both base and MTP arms. The tiny per-call bookkeeping only
    # exists while armed.
    if arm_telemetry:
        from omlx.patches.expert_streaming.shard_bank import arm_read_telemetry

        _prev_arm = arm_read_telemetry(True)
        print("read telemetry armed (runtime)")

    model_path = MODEL_PATHS[model_key]
    entry_name = DEFAULT_ENTRIES[model_key]
    # Native Lightning MTP serves qwen4_exp; every other bench type keeps
    # the external-assistant VLM path. Read from config.json (cheap) so a
    # new model folder picks the right path without bench edits.
    _mtp_native = False
    if mtp:
        try:
            with open(os.path.join(model_path, "config.json")) as f:
                # Native Lightning MTP serves qwen4_exp and glm5_next (the
                # vendored mlx-vlm GLM-5.3 module gained its JANG draft
                # head); every other bench type keeps the external-assistant
                # VLM path. Read from config.json (cheap) so a new model
                # folder picks the right path without bench edits.
                _model_type = json.load(f).get("model_type")
                if _model_type == "glm5_next":
                    # glm5_next may nest it under text_config.
                    with open(os.path.join(model_path, "config.json")) as f2:
                        _cfg = json.load(f2)
                    _mtp_native = (
                        _cfg.get("model_type") == "glm5_next"
                        and int(
                            (_cfg.get("text_config") or {}).get(
                                "num_nextn_predict_layers", 0
                            )
                            or 0
                        )
                        > 0
                    )
                else:
                    _mtp_native = _model_type == "qwen4_exp"
        except Exception:
            _mtp_native = False
    # Fase M5: record the exact code revision of the run.
    _GIT_SHA = None
    try:
        import subprocess

        _GIT_SHA = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        _GIT_SHA = None

    print(f"=== {model_key} budget {budget}G decode {decode} mtp {mtp} block {mtp_block} ane {ane} prompt {prompt_len} ===")
    pool = EnginePool(scheduler_config=SchedulerConfig(hot_cache_max_size=0))
    pool._process_memory_enforcer = FakeEnforcer()
    pool.discover_models("/Volumes/SSD 4TB/AI Models")
    entry = pool.get_entry(entry_name)
    if not entry:
        print("entry not found")
        return
    pool._process_memory_enforcer = None  # keep _propagate no-op path quiet

    settings = _bench_settings(
        pins, pin_gib, pin_regime, budget, topk, prior, cold_tier, hot_fraction,
        mtp, mtp_block, ane, specprefill_draft, specprefill_keep,
        mtp_native=_mtp_native,
    )
    if mtp_depth is not None:
        settings.mtp_num_draft_tokens = int(mtp_depth)
    if mtp and not _mtp_native:
        # Chain-level MTP aggregate for this arm only (roofline F1):
        # reset before the first request so the snapshot carries exactly
        # this arm's verify cycles and accepts.
        try:
            from omlx.patches.mlx_lm_mtp.batch_generator import mtp_stats_reset

            mtp_stats_reset()
        except Exception:
            pass
    runtime = pool._entry_runtime_resident_size(entry, settings)
    print(f"runtime est {runtime / 1024**3:.2f}G")
    # Structural estimate block: what the tuner sized against (layer geometry
    # + resident/streaming GiB), plus the header-scan cost itself in ms.
    _estimate_out = None
    try:
        import time as _time

        from omlx.patches.expert_streaming.residency import (
            expert_streaming_estimate as _estimate,
        )

        _t0 = _time.perf_counter()
        _est = _estimate(model_path)
        _scan_ms = (_time.perf_counter() - _t0) * 1000.0
        _estimate_out = {
            "model_type": _est.model_type,
            "supported": _est.supported,
            "num_moe_layers": _est.num_moe_layers,
            "experts_per_layer": _est.experts_per_layer,
            "per_expert_mb": round(_est.per_expert_bytes / 1024**2, 2),
            "resident_gib": round(_est.resident_bytes / 1024**3, 2),
            "streaming_gib": round(_est.streaming_bytes / 1024**3, 2),
            "scan_ms": round(_scan_ms, 1),
        }
        print(f"estimate {_estimate_out}")
    except Exception as _exc:  # never cost the run its numbers
        print(f"estimate unavailable: {_exc}")

    phys0 = get_phys_footprint() / 1024**3
    t0 = time.perf_counter()
    engine = await pool.get_engine(entry_name, runtime_settings=settings)
    t_load = time.perf_counter() - t0
    phys_loaded = get_phys_footprint() / 1024**3
    print(f"engine loaded {t_load:.1f}s phys {phys_loaded:.2f}G active {mx.get_active_memory() / 1024**3:.2f}G")

    # Honest memory limits: without an enforcer the scheduler's prefill
    # throttle/guard never engage (limits stay 0) and the lazy chunk forward's
    # measured ~17MB/token transient (streaming expert mini-banks) runs
    # unbounded — the Metal buffer pool reached ~30 GiB on 8k prompts and
    # squeezed the machine into swap (F-series F1). Set the same watermarks
    # the server's ProcessMemoryEnforcer would propagate.
    try:
        _eng = pool.get_entry(entry_name).engine
        _sched = getattr(getattr(getattr(_eng, "_engine", None), "engine", None), "scheduler", None)
        if _sched is not None:
            gib = 1024**3
            _sched._memory_hard_limit_bytes = int(mem_ceiling * gib)
            _sched._memory_limit_bytes = int(mem_ceiling * 0.9 * gib)
            _sched._memory_abort_limit_bytes = int(mem_ceiling * 0.95 * gib)
            print(
                f"scheduler memory limits: hard {mem_ceiling:.0f}G "
                f"soft {mem_ceiling * 0.9:.0f}G abort {mem_ceiling * 0.95:.0f}G"
            )
    except Exception as e:
        print(f"scheduler limit setup skipped: {e}")

    # Fase M1: pin sync/regime are wired through ModelSettings BEFORE
    # get_engine (see _bench_settings) — no late os.environ mutation here.

    vlm_model = getattr(engine, "_vlm_model", None)
    cache = find_streaming_cache(vlm_model)
    # Reference chunk schedule for bit-exactness (B4): fixed per prompt_len
    # so that divergence from different step sizes is explicit and comparable.
    _CHUNK_SCHEDULE_REF = {"short": 512, "512": 512, "2k": 1024, "8k": 4096}
    chunk_schedule = {
        "prompt_len": prompt_len,
        "reference_step": _CHUNK_SCHEDULE_REF.get(prompt_len, 512),
        "single_request": single_request,
    }
    results = {
        "model": model_key,
        "budget_gib": budget,
        "topk_threshold": topk,
        "cache_prior": prior,
        "cold_tier": cold_tier,
        "hot_fraction": hot_fraction,
        "mtp": mtp,
        "mtp_block": mtp_block,
        "ane": ane,
        "prompt_len": prompt_len,
        "single_request": single_request,
        "chunk_schedule": chunk_schedule,
        "runtime_est_gib": runtime / 1024**3,
        "estimate": _estimate_out,
        "load_s": t_load,
        "phys_before_gib": round(phys0, 2),
        "phys_after_load_gib": round(phys_loaded, 2),
    }
    if cache is not None:
        results["cache_per_expert_cap"] = getattr(cache, "capacity", None)
        results["cache_per_layer_cap"] = getattr(cache, "_per_layer_cap", None)

    messages = build_prompt(model_key, prompt_len)
    from resource_sampler import ResourceSampler

    sampler = ResourceSampler(
        interval=1.0,
        mlx_callbacks={
            "mlx_active_gib": mx.get_active_memory,
            "mlx_cache_gib": mx.get_cache_memory,
            # Fase J: high-water mark per phase to distinguish prefill transient
            # from decode residency (mlx_peak_gib is process-global, so reset per phase).
            "mlx_peak_gib": mx.get_peak_memory,
        },
    )
    _metal_peak: dict[str, float] = {}
    _reset_peak = getattr(mx, "reset_peak_memory", None)

    def _peak_phase(label: str) -> None:
        try:
            _metal_peak[label] = round(mx.get_peak_memory() / 1024**3, 3)
        except Exception:
            pass
        if _reset_peak is not None:
            try:
                _reset_peak()
            except Exception:
                pass

    if _reset_peak is not None:
        try:
            _reset_peak()
        except Exception:
            pass
    # Fase M3: the streaming backing is resolved ONCE (walked from the
    # engine holders) and feeds the read telemetry + ctx fallback + pin
    # exports below — one source of truth, available before the request.
    _bk = None
    _pinner = None
    for holder in (
        engine,
        getattr(engine, "_model", None),
        getattr(engine, "_vlm_model", None),
    ):
        if holder is None:
            continue
        _cand = getattr(holder, "_expert_streaming_backing", None)
        if _cand is not None:
            _bk = _cand
            _pinner = getattr(_bk, "_pin_controller", None)
            break
    sampler.start()
    sampler.mark("prefill")
    # Fase M3: phase-scope the backing telemetry so read_stats splits
    # prefill vs decode without cross-contamination.
    _tel = getattr(_bk, "read_telemetry", None) if _bk is not None else None
    _pool_before = None
    # Fase A1: one memtrace handle for both paths (inert when unavailable).
    try:
        from omlx.patches.expert_streaming.memtrace import memtrace as _mt6
    except Exception:
        _mt6 = None
    # Fase M4/A4: the POOL snapshot brackets the whole request (prefill +
    # decode). Its delta is owner-filtered to THIS backing, so a second
    # engine's pool traffic can never skew the bench's run_pool block
    # (without a backing or telemetry, owner=None gives the process-wide
    # view by design).
    _owner_tag = id(_bk) if _bk is not None else None
    _owner_filter = _owner_tag if _tel is not None and _tel.enabled else None
    try:
        from omlx.patches.expert_streaming.shard_bank import _run_io_pool as _rip

        _ptel = getattr(_rip(), "telemetry", None)
        if _ptel is not None:
            _pool_before = _ptel.snapshot(owner=_owner_filter)
    except Exception:
        _pool_before = None
    if single_request:
        # Single-request avoids the second full prefill; TTFT is first streamed token.
        t_request = time.perf_counter()
        first_output_at = None
        out2 = None
        # Fase A1: the prefill scope opens immediately before the call that
        # RUNS it (the first stream iteration) — never earlier.
        open_phase(_tel, _mt6, "prefill", entry_name)
        async for output in engine.stream_chat(
            messages, max_tokens=decode, temperature=0.0
        ):
            out2 = output
            if first_output_at is None and (
                output.completion_tokens > 0 or output.new_text or getattr(output, "tokens", None)
            ):
                first_output_at = time.perf_counter()
                _peak_phase("prefill")
                # Fase A1: first streamed token == prefill done; both
                # telemetries switch to decode at the SAME boundary.
                switch_phase(_tel, _mt6, "decode", entry_name)
                sampler.mark("decode")
        if out2 is None:
            raise SystemExit("single-request benchmark produced no output")
        if _tel is not None and _tel.enabled and first_output_at is None:
            switch_phase(_tel, _mt6, "decode", entry_name)
        close_phase(_tel)
        end_request = time.perf_counter()
        if first_output_at is None:
            first_output_at = end_request
            _peak_phase("prefill")
            sampler.mark("decode")
        ttft = first_output_at - t_request
        t_decode = end_request - first_output_at
        n = int(out2.completion_tokens)
        prompt_tokens = getattr(out2, "prompt_tokens", None)
        print(f"TTFT (stream first token) {ttft:.1f}s prompt {prompt_tokens}")
    else:
        # Fase A1: the prefill scope opens right before the FIRST chat —
        # that chat RUNS the prefill. It closes ONLY after the chat
        # returns; the decode chat runs under the decode scope.
        open_phase(_tel, _mt6, "prefill", entry_name)
        t1 = time.perf_counter()
        out1 = await engine.chat(messages, max_tokens=1, temperature=0.0)
        ttft = time.perf_counter() - t1
        _peak_phase("prefill")
        sampler.mark("decode")
        print(f"TTFT (1 tok) {ttft:.1f}s prompt {out1.prompt_tokens}")
        switch_phase(_tel, _mt6, "decode", entry_name)
        t2 = time.perf_counter()
        out2 = await engine.chat(messages, max_tokens=decode, temperature=0.0)
        t_decode = time.perf_counter() - t2
        n = int(out2.completion_tokens)
        close_phase(_tel)
    try:
        _mt6.set_context(phase="teardown", request_id="bench-1", engine_id=entry_name)
    except Exception:
        pass
    _pool_after = None
    if _pool_before is not None:
        try:
            from omlx.patches.expert_streaming.shard_bank import _run_io_pool as _rip

            _ptel2 = getattr(_rip(), "telemetry", None)
            if _ptel2 is not None:
                # Fase A4: attribute ONLY this bench's backing to run_pool;
                # foreign engines' traffic is excluded by the owner filter.
                _pool_after = _ptel2.delta(_pool_before, owner=_owner_filter)
        except Exception:
            _pool_after = None
    if n <= 0:
        raise SystemExit("benchmark produced zero completion tokens")
    tokps = n / max(t_decode, 1e-9)
    _peak_phase("decode")
    sampler.mark("teardown")
    sampler.stop()
    print(f"decode {n} tok in {t_decode:.1f}s -> {tokps:.3f} tok/s")
    res_summary = sampler.summary()
    print(f"resources {res_summary['phases']}")
    import json as _json

    # Side-effect artifacts land in out_dir so concurrent/sequential trials
    # (autotune) never overwrite each other's raw sampler series.
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    _json.dump(
        sampler.samples(),
        open(out_dir_p / f"{model_key}_{budget}g_samples.json", "w"),
    )
    # Generated output for bit-exactness comparison across runs. The VLM path
    # forwards RequestOutput.output_token_ids when available. Prefer token IDs
    # for the gate; keep textual fallback. Fail-high when neither exists.
    _text = getattr(out2, "text", None)
    _tokens = getattr(out2, "tokens", None)
    if _tokens is None:
        _tokens = getattr(out2, "token_ids", None)
    if isinstance(_tokens, list) and _tokens:
        _bit_exact = _tokens
        _bit_exact_kind = "tokens"
    elif isinstance(_text, str) and _text:
        _bit_exact = _text
        _bit_exact_kind = "text"
    else:
        raise SystemExit(
            f"bit-exactness gate FAILED: out2 has neither tokens nor text "
            f"(tokens={type(_tokens).__name__}, text={type(_text).__name__}); "
            "cannot compare runs. Aborting."
        )
    # Fase K K8: arms that REQUIRE the token-ID gate must fail high when
    # the engine produced no token list — a text-only gate cannot prove
    # identical token IDs, so it must never silently pass.
    if gate_tokens and _bit_exact_kind != "tokens":
        raise SystemExit(
            f"token-ID gate FAILED: bit_exact_kind={_bit_exact_kind} "
            f"(tokens={type(_tokens).__name__}); run with the engine fix that "
            "populates output_token_ids. Aborting."
        )
    _json.dump(
        {
            "bit_exact_kind": _bit_exact_kind,
            "text": _text if isinstance(_text, str) else None,
            "completion_tokens": n,
            "tokens": _tokens if isinstance(_tokens, list) else None,
        },
        open(out_dir_p / f"{model_key}_{budget}g_output.json", "w"),
    )

    stats = None
    profile = None
    pf_stats = None
    # Non-streaming (resident-expert) models skip the telemetry block below;
    # pre-initialize every var it assigns so results.update stays bound.
    advise_stats = None
    _read_stats_out = None
    _pool_after = None
    _memtrace_summary = None
    _ctx_fb = None
    _pin_out = None
    if cache is not None:
        stats = {
            "hits": cache.stats.hits,
            "misses": cache.stats.misses,
            "evictions": cache.stats.evictions,
            "hit_rate": cache.stats.hit_rate(),
            "size": cache.size,
            "capacity": cache.capacity,
            # FU2: policy + transition-table state for A/B arms.
            "policy": getattr(cache, "policy", "lru"),
            "trans_updates": int(getattr(getattr(cache, "spec_state", None), "trans_updates", 0) or 0),
            "trans_sources": len(getattr(getattr(cache, "spec_state", None), "trans", {}) or {}),
            "trans_overfetch": int((getattr(getattr(cache, "spec_state", None), "stats", {}) or {}).get("trans_overfetch", 0)),
        }
        print(f"cache {stats}")
        if cache.profile.enabled:
            profile = cache.profile.report()
            print(f"profile totals {profile['totals']}")
        # PILOT prefetcher stats (attached on language_model.model or wrapper)
        # The MTP accept counters live on the VLMModelAdapter
        # (engine._adapter / engine.model); native Lightning MTP does not
        # go through them, leaving None there.
        mtp_adapter_stats = None
        _holders = (
            getattr(vlm_model, "language_model", None),
            getattr(getattr(vlm_model, "language_model", None), "model", None),
            getattr(engine, "_adapter", None),
            getattr(engine, "model", None),
            vlm_model,
        )
        for holder in _holders:
            if mtp_adapter_stats is None:
                cand2 = getattr(holder, "mtp_stats", None)
                if isinstance(cand2, dict) and cand2.get("cycles", 0) > 0:
                    mtp_adapter_stats = dict(cand2)
            if mtp_adapter_stats is not None:
                break
        if mtp_adapter_stats is not None:
            print(f"mtp accept {mtp_adapter_stats}")
        # Chain-level aggregate (authoritative): the batch_generator logs
        # per-request cycles/accepts; summed here they cover FULL-accept
        # cycles the adapter clamp hook misses. Native Lightning MTP never
        # reaches this chain, leaving both None there.
        chain_mtp_stats = None
        if mtp:
            try:
                from omlx.patches.mlx_lm_mtp.batch_generator import mtp_stats_snapshot

                chain_mtp_stats = mtp_stats_snapshot()
                if chain_mtp_stats.get("cycles", 0) > 0:
                    print(f"mtp chain {chain_mtp_stats}")
            except Exception:
                chain_mtp_stats = None
        # Fase K F3: export the O2 F_RDADVISE speculation counters so
        # the readahead coverage is measurable (advised experts).
        # K1: the counters live on the per-conversion SpeculationState.
        try:
            _cache_spec = getattr(cache, "spec_state", None)
            advise_stats = dict(_cache_spec.stats) if _cache_spec is not None else None
            print(f"advise {advise_stats}")
        except Exception:
            advise_stats = None
        # Fase 2: demand-read telemetry (armed only by PROFILE=1).
        try:
            from omlx.patches.expert_streaming.shard_bank import read_stats as _rs

            _read_stats_out = _rs(_bk)
        except Exception:
            _read_stats_out = None
        # Fase L1: per-frame ctx observability — memtrace aggregates
        # (ctx_mode/positions/bank/inflight/prefetch per ctx.ensure event)
        # and the fallback-to-legacy counter by reason.
        try:
            from omlx.patches.expert_streaming.memtrace import memtrace as _mt

            _memtrace_summary = _mt.summary() if _mt.enabled else None
        except Exception:
            _memtrace_summary = None
        try:
            _ctx_fb = cache.ctx_fallback_stats()
        except Exception:
            _ctx_fb = None

        # Fase L: pin accounting (only when --pins armed a PinController).
        _pin_out = {
            "requested": pins,
            "pin_budget_gib": round((pin_gib if pin_gib is not None else 0.25), 3)
            if pins
            else 0.0,
            "pinned_bytes": 0,
            "pinned_experts": 0,
            "pinned_pages_estimate": 0,
            "profile_regime": pin_regime if pins else None,
            "pin_sync_requested": pins,
            "pin_sync_effective": False,
            "pin_regime_requested": pin_regime if pins else None,
            "pin_regime_effective": None,
            "pin_profile_loaded_at_engine_load": False,
            "pin_applied_before_first_request": False,
            "profile_fingerprint_match": None,
            "pin_load_time_ms": 0.0,
        }
        if _pinner is not None:
            _pin_out.update(
                {
                    "pin_budget_gib": round(_pinner.budget_bytes / 1024**3, 3),
                    "pinned_bytes": getattr(_bk, "pinned_bytes", 0),
                    "pinned_experts": getattr(_bk, "pinned_count", 0),
                    "pinned_pages_estimate": _pinner.pinned_pages_estimate,
                    "profile_regime": _pinner.profile_regime,
                    "pin_sync_effective": getattr(_pinner, "pin_sync", False),
                    "pin_regime_effective": _pinner.pin_regime,
                    "pin_profile_loaded_at_engine_load": getattr(
                        _pinner, "pins_applied_at_load", False
                    ),
                    "pin_applied_before_first_request": (
                        getattr(_pinner, "pins_applied_at_load", False)
                        and bool(getattr(_pinner, "pin_sync", False))
                    ),
                    "profile_fingerprint_match": _pinner.fingerprint_match,
                    "pin_load_time_ms": round(_pinner.pin_load_time_ms, 1),
                }
            )

    # Fase M5: the effective-config block — everything a fair comparison
    # must hold constant, read from the EFFECTIVE state.
    try:
        from omlx.patches.expert_streaming import streaming_switch as _ss_cfg
        from omlx.patches.expert_streaming import warmer as _warmer_cfg

        _expert_qd = getattr(_ss_cfg._EXPERT_IO_POOL, "_max_workers", None)
        _effective_config_out = _effective_config(
            git_sha=_GIT_SHA,
            single_request=single_request,
            decode_tokens=decode,
            chunk_schedule=chunk_schedule,
            budget_gib=budget,
            cold_tier=cold_tier,
            hot_fraction=hot_fraction,
            pins=pins,
            pinner=_pinner,
            model_fingerprint=(
                getattr(_pinner, "model_fingerprint", None)
                if _pinner is not None
                else None
            ),
            run_qd=0,
            expert_qd=_expert_qd or 0,
            prefill_qd=_ss_cfg._PREFILL_QD_ENV,
            knobs=knobs,
        )
        from omlx.patches.expert_streaming.shard_bank import _RUN_IO_QD as _rqd_cfg

        _effective_config_out["run_qd"] = int(_rqd_cfg)
    except Exception:
        _effective_config_out = None
    # Fase A2 fail-high: under --gate-tokens a null/incomplete block
    # ABORTS here, before any artifact is written; otherwise it warns.
    assert_effective_config_complete(_effective_config_out, gate=gate_tokens)
    phys_end = get_phys_footprint() / 1024**3
    try:
        from omlx.utils.proc_memory import get_lifetime_max_phys_footprint

        phys_lifetime_max = round(
            get_lifetime_max_phys_footprint() / 1024**3, 2
        )
    except Exception:
        phys_lifetime_max = None
    results.update(
        {
            "ttft_s": round(ttft, 2),
            "decode_tokens": n,
            "decode_s": round(t_decode, 2),
            "tok_s": round(tokps, 4),
            "phys_after_decode_gib": round(phys_end, 2),
            "phys_lifetime_max_gib": phys_lifetime_max,
            "metal_peak_prefill_gib": _metal_peak.get("prefill"),
            "metal_peak_decode_gib": _metal_peak.get("decode"),
            "active_after_decode_gib": round(mx.get_active_memory() / 1024**3, 2),
            "cache_stats": stats,
            "profile": profile,
            "prefetcher": pf_stats,
            "advise_stats": advise_stats,
            "read_stats": _read_stats_out,
            "mtp_accept_stats": chain_mtp_stats or mtp_adapter_stats,
            "run_pool": _pool_after,
            "memtrace_summary": _memtrace_summary,
            "ctx_fallback_to_legacy": _ctx_fb,
            "pin": _pin_out,
            "effective_config": _effective_config_out,
            # Fase A5: left null by the bench; the PROFILE=0 vs PROFILE=1
            # gate pair (bench/overhead_probe.py) fills it with the
            # per-call instrumentation cost when the machine frees up.
            "instrumentation_overhead": None,
            "resources": res_summary,
            "tokens": _tokens if isinstance(_tokens, list) else None,
            "bit_exact_kind": _bit_exact_kind,
        }
    )

    # Persist the learned pin profile when pins are active (the server does
    # this in stop(); the harness tears down via release/unload, so save
    # explicitly — parity with the ppl harness, which needs the frequencies
    # for the next HOBBIT-split load).
    if pins:
        from omlx.patches.expert_streaming import save_expert_pin_profile

        try:
            save_expert_pin_profile(engine)
        except Exception as exc:  # never cost the run its numbers
            print(f"pin profile save failed: {exc}")

    await pool.release_engine(entry_name)
    await pool._unload_engine(entry_name)

    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {out}")
    print("=== DONE ===")


def main():
    # INFO logs (streaming conversion, pool releases) are bench evidence —
    # without a handler Python drops them below WARNING.
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_PATHS))
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--decode", type=int, default=96)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--topk", type=float, default=None, help="adaptive top-k mass threshold (default exact)")
    ap.add_argument("--cache-prior", type=float, default=None, help="cache-prior logit bonus for resident experts (default exact)")
    ap.add_argument("--cold-tier", default=None, metavar="BITS",
                    help="route expert reads to the <model>/expert_cold/ 3/2-bit tier (I5)")
    ap.add_argument("--hot-fraction", type=float, default=None, metavar="FRAC",
                    help="HOBBIT split fraction (I6): with --cold-tier and a learned pin "
                         "profile, this fraction of each layer's most-used experts keeps the "
                         "original packing; the rest read the cold tier")
    ap.add_argument("--pins", action="store_true",
                    help="mlock-pin observed hot experts (default 0.25 GiB) and persist the "
                         "learned pin profile on unload (parity with the ppl harness)")
    ap.add_argument("--prompt-len", choices=["short", "512", "2k", "8k"], default="short")
    ap.add_argument("--mtp-block", type=int, default=None, help="vlm_mtp_draft_block_size (MTP tokens per round)")
    ap.add_argument("--ane", action="store_true", help="enable qwen35 ANE prefill")
    ap.add_argument("--specprefill", default=None, metavar="PATH",
                    help="draft model path for SpecPrefill (scores the prompt and prefills only the important tokens)")
    ap.add_argument("--specprefill-keep", type=float, default=None, metavar="PCT",
                    help="keep rate for SpecPrefill (default 0.2)")
    ap.add_argument("--pin-gib", type=float, default=None, metavar="GIB",
                    help="pin budget for --pins arms (default 0.25) — L2 matrix: 0.25/0.5/1.25")
    ap.add_argument("--knob", action="append", default=None, metavar="KNOB",
                    help="declare an experiment knob (e.g. pins_enabled) that A/B"
                         "comparison may differ on (Fase M5)")
    ap.add_argument("--pin-regime", choices=["decode", "prefill"], default="decode",
                    help="regime whose learned profile drives the pin selection (arm E: prefill)")
    ap.add_argument("--mtp-depth", type=int, default=None, metavar="N",
                    help="max native-MTP draft depth (mtp_num_draft_tokens); "
                         "default leaves the model default (glm5_next: 3)")
    ap.add_argument("--min-free-gb", type=float, default=22.0, metavar="GB",
                    help="abort when available memory is below this (memory-starved runs fragment prefill "
                         "into many chunks, re-stream experts, and thrash the page cache)")
    ap.add_argument("--mem-ceiling-gib", type=float, default=28.0, metavar="GIB",
                    help="scheduler memory ceiling propagated as throttle/guard watermarks (the server "
                         "gets this from the ProcessMemoryEnforcer; the bench has no enforcer)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gate-tokens", action="store_true",
                    help="require non-empty token-ID lists for the bit-exactness gate; fail high on empty")
    ap.add_argument("--out-dir", default="bench/results", metavar="DIR",
                    help="directory for the _samples/_output side-effect files (default bench/results)")
    ap.add_argument(
        "--single-request",
        action="store_true",
        help="measure TTFT and decode from one streaming request (avoids a second prefill; B6)",
    )
    ap.add_argument(
        "--arm-read-telemetry",
        action="store_true",
        help="arm demand-read telemetry in runtime (storage-roofline "
             "derivation: decode-phase byte ratio + MTP accept stats)",
    )
    ap.add_argument(
        "--cache-policy", choices=["lru", "s3fifo"], default="lru",
        help="FU2: LRU eviction policy for the app-level cache "
             "(page-cache-only budgets ignore it). A/B vs lru.",
    )
    ap.add_argument(
        "--no-transition", action="store_true",
        help="FU1: disable the transition-table k+1 overfetch in the RA "
             "advisor (A/B arm).",
    )
    args = ap.parse_args()
    # FU1/FU2/FU3: env must be set before any omlx import (singletons are
    # read at import time). All omlx imports in this file are lazy, so
    # main-time mutation is in time.
    os.environ["OMLX_EXPERT_STREAMING_CACHE"] = args.cache_policy
    if args.no_transition:
        os.environ["OMLX_EXPERT_STREAMING_TRANSITION"] = "0"
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / 1024**3
        if free_gb < args.min_free_gb:
            raise SystemExit(
                f"bench aborted: only {free_gb:.1f} GB available (need {args.min_free_gb:.0f}+). "
                "Memory-starved runs fragment prefill into many chunks and re-stream experts — "
                "close apps or lower --min-free-gb to override."
            )
        print(f"memory preflight ok: {free_gb:.1f} GB available", flush=True)
    except ImportError:
        pass
    asyncio.run(
        run(
            args.model,
            args.budget,
            args.decode,
            args.mtp,
            args.out,
            args.topk,
            args.cache_prior,
            args.cold_tier,
            prompt_len=args.prompt_len,
            hot_fraction=args.hot_fraction,
            pins=args.pins,
            mtp_block=args.mtp_block,
            ane=args.ane,
            specprefill_draft=args.specprefill,
            specprefill_keep=args.specprefill_keep,
            mem_ceiling=args.mem_ceiling_gib,
            out_dir=args.out_dir,
            single_request=args.single_request,
            gate_tokens=args.gate_tokens,
            mtp_depth=args.mtp_depth,
            pin_gib=args.pin_gib,
            pin_regime=args.pin_regime,
            knobs=args.knob,
            arm_telemetry=args.arm_read_telemetry,
        )
    )


if __name__ == "__main__":
    main()