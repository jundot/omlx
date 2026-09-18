"""Fase 0 bench: expert streaming TTFT, steady-state tok/s, hit rate, memory, stage profile.

Usage:
    .venv/bin/python bench/bench_expert_streaming.py --model qwen --budget 1.0 --decode 96 --out bench/results/qwen_1g.json
    .venv/bin/python bench/bench_expert_streaming.py --model glm --budget 1.0 --decode 16

Protocol (B6): use --single-request and the same --decode for all A/B arms so
TTFT and tok/s are comparable. Every arm writes tokens + chunk_schedule + metal peaks.

"""

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))


class _Heartbeat:
    """Periodic flushed progress line so long phases never look frozen.

    A daemon thread prints one ``[hb ...]`` line every ``interval`` seconds:
    phase + elapsed, tokens emitted so far and demand hit/miss counters.
    Everything is
    duck-typed and exception-swallowed — a heartbeat must never break a run.
    """

    def __init__(self, interval_s: float):
        self.interval = float(interval_s)
        self.state = {"phase": "load", "t0": time.perf_counter(), "tok": 0}
        self._stop = threading.Event()
        self._cache = None
        self._thread = None

    def attach(self, cache=None) -> None:
        self._cache = cache

    def mark(self, phase: str) -> None:
        self.state["phase"] = phase
        self.state["t0"] = time.perf_counter()
        self.state["tok"] = 0

    def tick_tok(self, n: int) -> None:
        self.state["tok"] = int(n)

    def start(self) -> None:
        if self.interval <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="bench-heartbeat", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                el = time.perf_counter() - self.state["t0"]
                parts = [f"[hb {self.state['phase']} +{el:.0f}s"]
                if self.state.get("tok"):
                    parts.append(f"tok={self.state['tok']}")
                st = (
                    getattr(self._cache, "stats", None)
                    if self._cache is not None
                    else None
                )
                if st is not None:
                    h = (getattr(st, "decode_hits", 0) or 0) + (
                        getattr(st, "prefill_hits", 0) or 0
                    )
                    m = (getattr(st, "decode_misses", 0) or 0) + (
                        getattr(st, "prefill_misses", 0) or 0
                    )
                    parts.append(f"demand h={h} m={m}")
                print(" ".join(parts) + "]", flush=True)
            except Exception:
                pass

MODEL_PATHS = {
    "qwen": "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-oQ4e-mtp",
    "qwen-jang": "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-JANG_4S",
    "qwen-jang4m": "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-JANG_4M",
    "glm": "/Volumes/SSD 4TB/AI Models/GLM-5.3-Flash-oQ4e",
    "glm-jang": "/Volumes/SSD 4TB/AI Models/GLM-5.3-Flash-JANG-MTP",
    "dsv4": "/Volumes/SSD 4TB/AI Models/DeepSeek-V4-Flash-0731-oQ4e-mtp",
    "dsv4-jang": "/Volumes/SSD 4TB/AI Models/DeepSeek-V4-Flash-0731-JANG",
    "v41": "/Volumes/SSD 4TB/AI Models/Jundot/DeepSeek-V4.1-Flash-oQ4e-mtp",
}
DEFAULT_ENTRIES = {
    "qwen": "Qwen3.8-Flash-Next-oQ4e-mtp",
    "qwen-jang": "Qwen3.8-Flash-Next-JANG_4S",
    "qwen-jang4m": "Qwen3.8-Flash-Next-JANG_4M",
    "glm": "GLM-5.3-Flash-oQ4e",
    "glm-jang": "GLM-5.3-Flash-JANG-MTP",
    "dsv4": "DeepSeek-V4-Flash-0731-oQ4e-mtp",
    "dsv4-jang": "DeepSeek-V4-Flash-0731-JANG",
    "v41": "DeepSeek-V4.1-Flash-oQ4e-mtp",
}
PROMPTS = {
    "qwen": [{"role": "user", "content": "Hello, how are you?"}],
    "qwen-jang": [{"role": "user", "content": "Hello, how are you?"}],
    "qwen-jang4m": [{"role": "user", "content": "Hello, how are you?"}],
    "glm": [{"role": "user", "content": "Hello, how are you?"}],
    "glm-jang": [{"role": "user", "content": "Hello, how are you?"}],
    "dsv4": [{"role": "user", "content": "Hello, how are you?"}],
    "dsv4-jang": [{"role": "user", "content": "Hello, how are you?"}],
    "v41": [{"role": "user", "content": "Hello, how are you?"}],
}

_FILLER = (
    "The scientist wrote a detailed report about the river ecosystem, "
    "describing how the water temperature changes with the seasons and "
    "which fish species migrate through the valley each year. "
)


def build_prompt(model_key: str, prompt_len: str) -> list[dict]:
    """Synthetic prompts: short (7 tok), 512, 2k, 8k (approximate word targets)."""
    if prompt_len == "short":
        if model_key not in PROMPTS:
            # Direct-path models share the generic short prompt.
            return [{"role": "user", "content": "Hello, how are you?"}]
        return list(PROMPTS[model_key])
    words = {"512": 400, "2k": 1600, "8k": 6400}[prompt_len]
    content = (_FILLER * (words // 26 + 1))[: words * 7]
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Corpus continuation prompts
#
# The shipped prompt is "Hello, how are you?" with a chat template. The model
# answers and emits EOS after ~43 tokens, which is what made the hotness-decay
# A/B structurally unmeasurable: a 1024-token sweep needs ~22x more tokens than
# the model will ever produce. The admin bench suite (omlx/admin/benchmark.py)
# sidesteps this by feeding raw token IDs from a corpus to stream_generate --
# no chat template, so the model continues the text instead of terminating.
# Same idea here, so we keep our instrumentation and get their prompt.
# ---------------------------------------------------------------------------

_CORPORA_DIR = Path(__file__).resolve().parent.parent / "omlx" / "admin" / "bench_corpora"

# filename, chars_per_token, content start marker (None = whole file).
# Mirrors BENCHMARK_CONTEXT_PROFILES in omlx/admin/benchmark.py.
_CORPUS_SPECS: dict[str, tuple[str, float, str | None]] = {
    "novel_en": ("novel_en.txt", 4.0, "Call me Ishmael."),
    "novel_ko": ("novel_ko.txt", 1.35, None),
    "novel_ja": ("novel_ja.txt", 1.6, None),
    "code_python": ("code_python.txt", 4.0, None),
    "code_mixed": ("code_mixed.txt", 3.5, None),
}


def build_corpus_prompt_ids(
    tokenizer, corpus: str, target_tokens: int, *, offset: int = 0
) -> list[int]:
    """Exactly ``target_tokens`` raw token IDs of a corpus excerpt.

    Deterministic: the same (corpus, target_tokens, offset) always yields the
    same IDs, so every A/B arm sees an identical prompt and the run measures
    the port rather than prompt difficulty.
    """
    filename, chars_per_token, start_marker = _CORPUS_SPECS[corpus]
    text = (_CORPORA_DIR / filename).read_text(encoding="utf-8")
    if start_marker:
        idx = text.find(start_marker)
        if idx < 0:
            raise SystemExit(f"corpus {filename} lost its start marker {start_marker!r}")
        text = text[idx:]
    if not text:
        raise SystemExit(f"corpus {filename} is empty")
    target_chars = max(int(target_tokens * chars_per_token), 1)
    start = int(offset) % max(1, len(text))
    for _ in range(8):
        body = (text + text)[start : start + target_chars]
        ids = [int(t) for t in tokenizer.encode(body)]
        if len(ids) >= target_tokens:
            return ids[:target_tokens]
        if not ids:
            raise SystemExit(f"corpus {filename} tokenized to 0 tokens")
        # Scale by the ratio this tokenizer actually produced.
        target_chars = max(
            target_chars + 1,
            (target_chars * target_tokens + len(ids) - 1) // len(ids) + 1,
        )
    raise SystemExit(f"could not build a {target_tokens}-token prompt from {filename}")


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


def _parse_budget(raw: str | float | None) -> float | None:
    """Parse --budget: GiB float, or None for the auto stack."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in ("auto", "none", ""):
        return None
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise SystemExit(f"bad --budget {raw!r}: GiB number or 'auto'")


def _bench_settings(
    pins: bool,
    pin_gib: float | None,
    pin_regime: str,
    budget: float | None,
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
    moe_offload_frac: float | None = None,
    dflash: bool = False,
    dflash_draft: str | None = None,
    engram_ssd: bool = False,
    model_type: str | None = None,
):
    """Fase M1: the bench's ModelSettings, wired EXPLICITLY.

    Pins arrive as model settings (pin regime + sync), never as a late
    os.environ mutation after engine load — the PinController is built
    inside get_engine, so env written later cannot be relied on.
    """
    from omlx.model_settings import ModelSettings

    return ModelSettings(
        # Validation scope for model-dependent rules (e.g. the DSpark
        # offload exception); auto-managed, never a user toggle.
        model_type=model_type,
        # --moe-offload FRAC arm: upstream official MoE offload OWNS the
        # experts, ours stands down so exactly one system is measured.
        expert_streaming_enabled=(moe_offload_frac is None),
        moe_expert_offload_enabled=(moe_offload_frac is not None),
        moe_expert_offload_resident_fraction=(
            moe_offload_frac if moe_offload_frac is not None else 0.25
        ),
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
        dflash_enabled=dflash,
        dflash_draft_model=dflash_draft or None,
        deepseek_v41_engram_ssd_offload=engram_ssd,
    )


def _effective_config(
    *,
    git_sha: str | None,
    single_request: bool,
    decode_tokens: int,
    chunk_schedule: dict,
    budget_gib: float | None,
    budget_mode: str = "pinned",
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

    return {
        "git_sha": git_sha,
        "model_fingerprint": model_fingerprint,
        "single_request": bool(single_request),
        "decode_tokens": int(decode_tokens),
        "chunk_schedule": dict(chunk_schedule),
        "budget_gib": float(budget_gib) if budget_gib is not None else -1.0,
        "budget_mode": str(budget_mode),
        "cold_tier": cold_tier,
        "hot_fraction": hot_fraction,
        # OMLX_EXPERT_STREAMING_CTX_ROLLING is gone (one-path cleanup):
        # mode is picked per call — union for decode-shaped calls while
        # DECODE_UNION_ROWS > 0, rolling everywhere when it is 0.
        "ctx_mode_policy": (
            "hybrid" if _ss._DECODE_UNION_MAX_ROWS > 0 else "rolling"
        ),
        "decode_union_rows": int(_ss._DECODE_UNION_MAX_ROWS),
        "ctx_ahead": int(_ss._CTX_PREFETCH_AHEAD),
        "expert_qd": int(expert_qd),
        "run_qd": int(run_qd),
        "prefill_qd": int(prefill_qd),
        # Dedicated prefetch queue removed 2026-09-09 (audit P2-13, no win in
        # three campaigns). Prefetch shares the demand pool; nothing to
        # record here.
        "ra_enabled": bool(_ss._RA_ENV),
        "pins_enabled": bool(pins),
        "pin_sync_effective": bool(
            getattr(pinner, "pin_sync", False) if pinner is not None else False
        ),
        "pin_regime_effective": (
            getattr(pinner, "pin_regime", None) if pinner is not None else None
        ),
        "cache_cool_protocol": "warm-page-cache",
        "cache_policy": str(getattr(_ss, "_CACHE_POLICY_ENV", "lru")),
        "transition_overfetch": os.environ.get("OMLX_EXPERT_STREAMING_TRANSITION", "1") != "0",
        "experiment_knobs": list(knobs or []),
    }


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
    # Removed (audit 2026-09-09, P2-13): "run_merge_gap" — the knob is gone,
    # so new results no longer carry it. Archived results that still do are
    # unaffected: this is a presence check, not an equality check.
    "ra_enabled",
    "pins_enabled",
    "cache_cool_protocol",
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
        if mlp is None:
            # deepseek_v4 nests the MoE under layer.ffn, not layer.mlp.
            mlp = getattr(layer, "ffn", None)
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


def _iter_switch_mlps(vlm_model):
    """Yield every MoE `switch_mlp` in the model, whatever the wrapper path."""
    for path in (
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("layers",),
    ):
        cur = vlm_model
        for a in path:
            if not hasattr(cur, a):
                cur = None
                break
            cur = getattr(cur, a)
        if cur:
            for layer in cur:
                mlp = getattr(layer, "mlp", None)
                sm = getattr(mlp, "switch_mlp", None) if mlp else None
                if sm is not None:
                    yield sm
            return


def collect_seed_vs_resident(vlm_model, cache):
    """Did the seeder's chosen hot set actually become the resident set?

    bench/bench_seed_static.py replays the captured JANG_4M trace and says a
    *static* seed of the top-k by prefill frequency should hit 0.1935 at
    4 GiB, but e2e measures 0.0618 -- which is exactly the slot coverage
    95/(512*3), i.e. what a uniform random 6.2% sample would give. Those two
    facts together say the resident set is uncorrelated with the routing, but
    they cannot say *which* step lost the signal. This splits them:

      resident == seed  -> the ranking itself is wrong (bad frequency oracle)
      resident != seed  -> the ranking was fine and never became resident
                           (key mismatch, dropped warm-pool jobs, or the
                           per-layer cap truncating the hottest first)
    """
    seed: dict[int, set[int]] = {}
    for sm in _iter_switch_mlps(vlm_model):
        hook = getattr(sm, "_warm_pins", None)
        rec = getattr(hook, "recorder", None)
        hot = getattr(rec, "last_hot", None) if rec is not None else None
        if not hot:
            continue
        for lay, eids in hot.items():
            seed.setdefault(int(lay), set()).update(int(e) for e in eids)
        break  # one recorder is shared across layers

    resident: dict[int, set[int]] = {}
    store = getattr(cache, "_store", None)
    if store:
        for key in store:
            try:
                lay, eid = int(key[0]), int(key[1])
            except Exception:
                continue
            resident.setdefault(lay, set()).add(eid)

    per_layer = []
    for lay in sorted(set(seed) | set(resident)):
        s = seed.get(lay, set())
        r = resident.get(lay, set())
        per_layer.append(
            {
                "layer": lay,
                "seed_n": len(s),
                "resident_n": len(r),
                "intersection": len(s & r),
                "seed_only": len(s - r),
                "resident_only": len(r - s),
            }
        )
    tot_seed = sum(p["seed_n"] for p in per_layer)
    tot_res = sum(p["resident_n"] for p in per_layer)
    tot_int = sum(p["intersection"] for p in per_layer)
    summary = {
        "layers": len(per_layer),
        "seed_total": tot_seed,
        "resident_total": tot_res,
        "intersection": tot_int,
        "seed_kept_frac": round(tot_int / tot_seed, 4) if tot_seed else None,
        "resident_from_seed_frac": round(tot_int / tot_res, 4) if tot_res else None,
    }
    return {
        "summary": summary,
        "per_layer": per_layer,
        # Actual IDs, so the ranking can be compared against a routing trace
        # replayed offline (bench/bench_seed_static.py --trace ...).
        "seed_experts": {str(lay): sorted(s) for lay, s in seed.items()},
        "resident_experts": {str(lay): sorted(r) for lay, r in resident.items()},
    }


async def run(
    model_key: str,
    budget: float | None,
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
    corpus: str | None = None,
    corpus_tokens: int = 1024,
    requests: int = 1,
    prompt_text: str | None = None,
    moe_offload: float | None = None,
    dflash: bool = False,
    dflash_draft: str | None = None,
    v41_offload: float | None = None,
    engram_ssd: bool = False,
    heartbeat_s: float = 5.0,
):
    from omlx.engine_pool import EnginePool
    from omlx.scheduler import SchedulerConfig
    from omlx.utils.proc_memory import get_phys_footprint
    import mlx.core as mx

    if "/" in model_key:
        # Direct model dir (must be pool-discovered under its basename).
        model_path = os.path.expanduser(model_key)
        entry_name = os.path.basename(model_path.rstrip("/"))
    else:
        model_path = MODEL_PATHS[model_key]
        entry_name = DEFAULT_ENTRIES[model_key]
    # V4.1 keeps its own adapter mechanics: --v41-offload drives the SAME
    # legacy keys through the V4.1 engine branch (moe_expert_offload_enabled
    # + resident fraction + engram SSD). The fraction is the INITIAL
    # per-layer budget; the model-agnostic dynamic governor then manages
    # it (same precedence as the unified backend). A direct path to a
    # deepseek_v41 checkpoint gets the same treatment as the v41 alias —
    # sniff the arch cheaply from config.
    _is_v41_arch = model_key == "v41"
    if not _is_v41_arch:
        try:
            with open(os.path.join(model_path, "config.json")) as _cf:
                _is_v41_arch = (
                    json.load(_cf).get("model_type") == "deepseek_v41"
                )
        except Exception:
            pass
    if _is_v41_arch and budget is None:
        print("note: v41 fraction is the initial dynamic budget; "
              "governor manages it from pressure + hunger", flush=True)
    # v41 offload + native MTP (DSpark) is a supported arm: verify blocks
    # run under frozen slot recency (moe_offload.verify_scope).
    _eff_offload = (
        moe_offload
        if moe_offload is not None
        else (v41_offload if _is_v41_arch else None)
    )
    if dflash and not dflash_draft:
        raise ValueError("--dflash needs --dflash-draft")
    if dflash and _eff_offload is not None:
        raise ValueError("--dflash cannot combine with expert offload")
    _offload_backend = None
    if _eff_offload is not None:
        # Compat arm (unified backend): print the upstream system's own
        # verdict for this checkpoint plus which backend serves it —
        # expert_streaming on owned types, the legacy adapter elsewhere.
        try:
            from omlx.patches.moe_offload_compat import (
                moe_offload_compatibility,
            )

            _sup, _why = moe_offload_compatibility(model_path)
            print(
                f"upstream moe-offload verdict: supported={_sup} reason={_why}",
                flush=True,
            )
        except Exception as _ve:
            print(f"upstream moe-offload verdict: probe failed: {_ve}", flush=True)
        try:
            from omlx.patches.moe_expert_offload import _streaming_owns_model

            _offload_backend = (
                "v41-native-adapter"
                if _is_v41_arch
                else (
                    "unified-expert-streaming"
                    if _streaming_owns_model(model_path)
                    else "legacy-adapter"
                )
            )
            print(f"moe-offload backend: {_offload_backend}", flush=True)
        except Exception as _be:
            print(f"moe-offload backend: probe failed: {_be}", flush=True)
    # Native Lightning MTP serves qwen4_exp; every other bench type keeps
    # the external-assistant VLM path. Read from config.json (cheap) so a
    # new model folder picks the right path without bench edits.
    _mtp_native = False
    _bench_model_type = None
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            # Native Lightning MTP serves qwen4_exp and glm5_next (the
            # vendored mlx-vlm GLM-5.3 module gained its JANG draft
            # head); every other bench type keeps the external-assistant
            # VLM path. Read from config.json (cheap) so a new model
            # folder picks the right path without bench edits.
            _cfg = json.load(f)
            _model_type = _cfg.get("model_type")
            _nested_type = (_cfg.get("text_config") or {}).get(
                "model_type"
            )
            # ModelSettings.model_type is the validation scope for
            # model-dependent rules (streaming x MTP exclusivity included);
            # mirror residency._config_model_type: top level, text_config
            # fallback. Without it the mtp+streaming arm is rejected.
            _bench_model_type = _model_type or _nested_type
            if mtp:
                if _nested_type == "deepseek_v41" or _model_type == "deepseek_v41":
                    # DSpark lives in-checkpoint: the v41+MTP arm needs the
                    # native path (vlm_mtp + offload is rightly rejected).
                    _mtp_native = True
                elif _model_type == "glm5_next":
                    # glm5_next may nest it under text_config.
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

    _budget_mode = "auto" if budget is None else "pinned"
    try:
        from omlx.patches.expert_streaming import _auto_budget_bytes as _auto_b
        _budget_init_gib = budget if budget is not None else _auto_b() / 1024**3
    except Exception:
        _budget_init_gib = budget if budget is not None else 1.0
    _budget_tag = "auto" if budget is None else f"{budget}g"
    print(f"=== {model_key} budget {_budget_mode}({_budget_init_gib:.2f}G init) decode {decode} mtp {mtp} block {mtp_block} ane {ane} prompt {prompt_len} ===")
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
        moe_offload_frac=_eff_offload,
        dflash=dflash,
        dflash_draft=dflash_draft,
        engram_ssd=engram_ssd,
        model_type=_bench_model_type
        or ("deepseek_v41" if _is_v41_arch else None),
    )
    if mtp_depth is not None:
        settings.mtp_num_draft_tokens = int(mtp_depth)
    if mtp:
        # Chain-level MTP aggregate for this arm only (roofline F1):
        # reset before the first request so the snapshot carries exactly
        # this arm's verify cycles and accepts. Native Lightning MTP also
        # drives the GenerationBatch chain, so reset it too.
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
    # Progress heartbeat: engine load (model conversion + streaming attach)
    # is the longest silent stretch of the run — start before it.
    hb = _Heartbeat(heartbeat_s)
    hb.mark("load")
    hb.start()
    t0 = time.perf_counter()
    engine = await pool.get_engine(entry_name, runtime_settings=settings)
    t_load = time.perf_counter() - t0
    hb.mark("postload")
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
    hb.attach(cache=cache)
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
        "budget_gib": _budget_init_gib,
        "budget_mode": _budget_mode,
        "moe_offload_frac": moe_offload,
        "offload_backend": _offload_backend,
        "v41_offload_frac": v41_offload,
        "engram_ssd": engram_ssd,
        "dflash": dflash,
        "dflash_draft": dflash_draft,
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

    if prompt_text:
        messages = [{"role": "user", "content": prompt_text}]
    else:
        messages = build_prompt(model_key, prompt_len)
    raw_prompt_ids = None
    if corpus is not None:
        _tok = getattr(engine, "tokenizer", None)
        if _tok is None:
            raise SystemExit("--corpus needs engine.tokenizer, engine has none")
        raw_prompt_ids = build_corpus_prompt_ids(_tok, corpus, int(corpus_tokens))
        print(
            f"corpus prompt: {corpus} -> {len(raw_prompt_ids)} raw token IDs "
            f"(stream_generate, no chat template)",
            flush=True,
        )
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
    # The streaming backing is resolved ONCE (walked from the engine
    # holders) and feeds the governor/ctx-fallback/pin exports below —
    # one source of truth, available before the request.
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
    hb.attach(cache=cache)
    sampler.start()
    sampler.mark("prefill")
    if single_request:
        # Single-request avoids the second full prefill; TTFT is first streamed
        # token.
        #
        # --requests N>1 serves N sequential requests on the SAME engine. That
        # is the point: the cache lives for the process, not the request, so
        # a one-request run throws away exactly the state cross-request reuse
        # exercises. N requests accumulate it.

        def _stream_once(max_tokens: int):
            if raw_prompt_ids is not None:
                return engine.stream_generate(
                    prompt=list(raw_prompt_ids),
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
            return engine.stream_chat(
                messages, max_tokens=max_tokens, temperature=0.0
            )

        n_req = max(1, int(requests))
        ttft_first = None
        ttfts: list[float] = []
        toks_per_req: list[int] = []
        n_total = 0
        t_decode_total = 0.0
        out2 = None
        for _req_idx in range(n_req):
            t_request = time.perf_counter()
            first_output_at = None
            hb.mark("prefill")
            async for output in _stream_once(decode):
                out2 = output
                hb.tick_tok(getattr(output, "completion_tokens", 0) or 0)
                if first_output_at is None and (
                    output.completion_tokens > 0 or output.new_text or getattr(output, "tokens", None)
                ):
                    first_output_at = time.perf_counter()
                    if _req_idx == 0:
                        # Peak-memory high-water marks are process-global and
                        # reset on read; only the first request's prefill peak
                        # is the cold one worth recording.
                        _peak_phase("prefill")
                        sampler.mark("decode")
                    hb.mark("decode")
            if out2 is None:
                raise SystemExit("single-request benchmark produced no output")
            if first_output_at is None:
                hb.mark("decode")
            end_request = time.perf_counter()
            if first_output_at is None:
                first_output_at = end_request
                if _req_idx == 0:
                    _peak_phase("prefill")
                    sampler.mark("decode")
            _ttft = first_output_at - t_request
            _n_req_tok = int(out2.completion_tokens)
            ttfts.append(_ttft)
            toks_per_req.append(_n_req_tok)
            if ttft_first is None:
                ttft_first = _ttft
            t_decode_total += end_request - first_output_at
            n_total += _n_req_tok
            if n_req > 1:
                print(
                    f"  req {_req_idx}: {_n_req_tok} tok, TTFT {_ttft:.1f}s",
                    flush=True,
                )
        ttft = ttft_first if ttft_first is not None else 0.0
        t_decode = t_decode_total
        n = n_total
        prompt_tokens = getattr(out2, "prompt_tokens", None)
        if raw_prompt_ids is not None and prompt_tokens is None:
            prompt_tokens = len(raw_prompt_ids)
        print(
            f"TTFT (first request) {ttft:.1f}s prompt {prompt_tokens} "
            f"| {n_req} requests, {n_total} tok total, "
            f"tok/req {toks_per_req}"
        )
    else:
        hb.mark("prefill")
        t1 = time.perf_counter()
        out1 = await engine.chat(messages, max_tokens=1, temperature=0.0)
        ttft = time.perf_counter() - t1
        _peak_phase("prefill")
        sampler.mark("decode")
        print(f"TTFT (1 tok) {ttft:.1f}s prompt {out1.prompt_tokens}")
        hb.mark("decode")
        t2 = time.perf_counter()
        out2 = await engine.chat(messages, max_tokens=decode, temperature=0.0)
        t_decode = time.perf_counter() - t2
        n = int(out2.completion_tokens)
    if n <= 0:
        raise SystemExit("benchmark produced zero completion tokens")
    tokps = n / max(t_decode, 1e-9)
    hb.stop()
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
        open(out_dir_p / f"{model_key}_{_budget_tag}_samples.json", "w"),
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
        open(out_dir_p / f"{model_key}_{_budget_tag}_output.json", "w"),
    )

    stats = None
    pf_stats = None
    # Dynamic governor summary (auto default): actions taken at request
    # boundaries, hunger windows, per-layer targeting. Best-effort.
    _governor_out = None
    try:
        _gov = getattr(cache, "governor", None) if cache is not None else None
        _gov_src = cache
        if _gov is None and _bk is not None:
            # V4.1 native adapter: same model-agnostic governor, per-layer
            # capacity units (see streaming_backing module docstring).
            _gov = getattr(_bk, "governor", None)
            _gov_src = _bk
        if _gov is not None and hasattr(_gov, "summary"):
            _governor_out = _gov.summary()
            _governor_out["capacity_slots"] = getattr(_gov_src, "capacity", None)
            try:
                _governor_out["layer_overrides"] = getattr(
                    _gov_src, "layer_cap_overrides", lambda: {}
                )()
            except Exception:
                pass
    except Exception:
        _governor_out = None
    # Non-streaming (resident-expert) models skip the stats block below;
    # pre-initialize every var it assigns so results.update stays bound.
    advise_stats = None
    _ctx_fb = None
    _pin_out = None
    mtp_adapter_stats = None
    chain_mtp_stats = None
    if cache is not None:
        _cst = getattr(cache, "stats", None)
        if _cst is None:
            # Foreign cache (e.g. upstream moe_expert_offload.ExpertCache
            # exposes bare hits/misses counters, no stats object).
            _h = getattr(cache, "hits", 0) or 0
            _m = getattr(cache, "misses", 0) or 0
            _n = _h + _m
            _cst = {
                "hits": _h,
                "misses": _m,
                "evictions": 0,
                "hit_rate": (_h / _n) if _n else 0.0,
                "puts": _m,
                "retain_evicted": 0,
            }
            _cst_get = lambda k, d=0: _cst.get(k, d)
            _cst_rate = lambda: _cst["hit_rate"]
        else:
            _cst_get = lambda k, d=0: getattr(_cst, k, d)
            # CacheStats exposes raw counter fields only — derive the
            # rates here instead of calling methods that don't exist.
            _cst_rate = lambda: _cst_get("hits") / max(
                1, _cst_get("hits") + _cst_get("misses")
            )

        def _rate(n, *d):
            return _cst_get(n) / max(1, sum(_cst_get(k) for k in d))

        def _cst_call(k):
            # Bound methods (ours: decode_hit_rate()) arrive uncalled
            # through _cst_get; foreign-dict values pass through untouched.
            v = _cst_get(k)
            return v() if callable(v) else v
        stats = {
            "hits": _cst_get("hits"),
            "misses": _cst_get("misses"),
            "evictions": _cst_get("evictions"),
            "hit_rate": _cst_rate(),
            "size": int(getattr(cache, "size", 0) or 0),
            "capacity": int(getattr(cache, "capacity", 0) or 0),
            # Fase M4 residency diagnosis: puts vs misses, seeder damage, and
            # the per-layer occupancy histogram (a frozen cache shows every
            # layer pinned at the seed count with 0 evictions).
            "puts": _cst_get("puts"),
            "retain_evicted": _cst_get("retain_evicted"),
            # Demand-scoped: hit_rate above counts every get (prefill + the
            # rolling double split), which diluted it 2.3x below the decode
            # truth and misled the 0.637-vs-0.062 comparison. Compare offline
            # predictions against decode_hit_rate, not hit_rate.
            "decode_hits": _cst_get("decode_hits"),
            "decode_misses": _cst_get("decode_misses"),
            "decode_hit_rate": _rate("decode_hits", "decode_hits", "decode_misses"),
            "prefill_hits": _cst_get("prefill_hits"),
            "prefill_misses": _cst_get("prefill_misses"),
            # Per-LAYER stall (mihailescu2m/llama.cpp): stall is per layer,
            # not per miss, so decode_stall_rate -- not decode_hit_rate -- is
            # what tracks wall time. Read it together with
            # decode_misses_per_stalled_layer: >>1 means misses cluster inside
            # layers (widen prefetch, it is cheap); ~1 means they are spread
            # one per layer (only residency fixes it).
            "decode_layers": _cst_get("decode_layers"),
            "decode_layers_missed": _cst_get("decode_layers_missed"),
            "decode_stall_rate": _rate("decode_layers_missed", "decode_layers"),
            "decode_misses_per_stalled_layer": _rate(
                "decode_misses", "decode_layers_missed"
            ),
            "prefill_layers": _cst_get("prefill_layers"),
            "prefill_layers_missed": _cst_get("prefill_layers_missed"),
            "prefill_stall_rate": _rate(
                "prefill_layers_missed", "prefill_layers"
            ),
            "admission_drops": int(getattr(cache, "admission_drops", 0) or 0),
            # P2 detached-admission worker counters.
            "admit_mode": os.environ.get("OMLX_EXPERT_STREAMING_ADMIT", "detached"),
            "admit_submitted": int(
                getattr(getattr(cache, "admission", None), "submitted", 0) or 0
            ),
            "admit_admitted": int(
                getattr(getattr(cache, "admission", None), "admitted", 0) or 0
            ),
            "admit_filtered": int(
                getattr(getattr(cache, "admission", None), "filtered", 0) or 0
            ),
            "admit_queue_drops": int(
                getattr(getattr(cache, "admission", None), "queue_drops", 0) or 0
            ),
            "per_layer_cap": int(getattr(cache, "_per_layer_cap", 0) or 0),
            "per_layer_counts": {
                "min": min(getattr(cache, "_layer_counts", {}).values() or [0]),
                "max": max(getattr(cache, "_layer_counts", {}).values() or [0]),
                "distinct": sorted(
                    set(getattr(cache, "_layer_counts", {}).values())
                )[:8],
            },
            # FU2: policy + transition-table state for A/B arms.
            "policy": getattr(cache, "policy", "lru"),
            "trans_updates": int(getattr(getattr(cache, "spec_state", None), "trans_updates", 0) or 0),
            "trans_sources": len(getattr(getattr(cache, "spec_state", None), "trans", {}) or {}),
            "trans_overfetch": int((getattr(getattr(cache, "spec_state", None), "stats", {}) or {}).get("trans_overfetch", 0)),
            # Union-latch waits (all-or-nothing serialization) and
            # per-projection completion spread.
            "union_ensures": _cst_get("union_ensures"),
            "union_wait_us": _cst_get("union_wait_us"),
            "union_spread_us": _cst_get("union_spread_us"),
        }
        print(f"cache {stats}")
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
            if advise_stats is not None and _cache_spec is not None:
                try:
                    advise_stats["trans_precision"] = round(
                        _cache_spec.trans_precision(), 4
                    )
                except Exception:
                    pass
            print(f"advise {advise_stats}")
        except Exception:
            advise_stats = None
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

        _expert_qd = getattr(_ss_cfg._EXPERT_IO_POOL, "_max_workers", None)
        _effective_config_out = _effective_config(
            git_sha=_GIT_SHA,
            single_request=single_request,
            decode_tokens=decode,
            chunk_schedule=chunk_schedule,
            budget_gib=_budget_init_gib,
            budget_mode=_budget_mode,
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
            # The dedicated prefill pool is gone (one-path cleanup):
            # prefill shares the demand pool. 0 = "no separate queue",
            # the same value the removed knob defaulted to.
            prefill_qd=int(getattr(_ss_cfg, "_PREFILL_QD_ENV", 0)),
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
            # V4.1 native adapter residency (per-layer slots): hits/misses,
            # governor capacity, and P8 staged-prefetch counters.
            "v41_backing": (
                _bk.summary()
                if _bk is not None and hasattr(_bk, "summary")
                else None
            ),
            "prefetcher": pf_stats,
            "advise_stats": advise_stats,
            "mtp_accept_stats": chain_mtp_stats or mtp_adapter_stats,
            "ctx_fallback_to_legacy": _ctx_fb,
            "pin": _pin_out,
            "governor": _governor_out,
            "effective_config": _effective_config_out,
            "resources": res_summary,
            "tokens": _tokens if isinstance(_tokens, list) else None,
            "bit_exact_kind": _bit_exact_kind,
        }
    )
    # Opt-in diagnosis (OMLX_BENCH_SEED_DUMP=1). Writes seed_vs_resident.json
    # next to the other artifacts: did the seeder's top-k actually become the
    # resident set? See collect_seed_vs_resident for how to read it.
    if os.environ.get("OMLX_BENCH_SEED_DUMP", "") == "1":
        try:
            _svr = collect_seed_vs_resident(vlm_model, cache)
            with open(out_dir_p / "seed_vs_resident.json", "w") as _fh:
                _json.dump(_svr, _fh, indent=1)
            print(f"seed_vs_resident {_svr['summary']}")
        except Exception as e:  # never fail a run over a diagnostic
            print(f"seed_vs_resident FAILED: {e!r}")

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
    ap.add_argument("--model", required=True,
                    help=f"one of {sorted(MODEL_PATHS)} or a direct model dir")
    ap.add_argument("--budget", default="1.0",
                    help="LRU budget GiB, or 'auto' for the RAM-scaled dynamic stack (default 1.0 pinned)")
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
    ap.add_argument("--prompt", default=None, metavar="TEXT",
                    help="raw user prompt text (overrides --prompt-len)")
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
        "--heartbeat",
        type=float,
        default=5.0,
        metavar="SEC",
        help="progress heartbeat interval in seconds (0 disables; default 5). "
             "Prints phase/elapsed/tokens/demand counters so long runs never "
             "look frozen.",
    )
    ap.add_argument(
        "--corpus", choices=sorted(_CORPUS_SPECS), default=None,
        help="feed a raw corpus excerpt (token IDs, no chat template) instead "
             "of the shipped 'Hello, how are you?' prompt. A continuation does "
             "not self-terminate, so long decodes are actually reachable.",
    )
    ap.add_argument(
        "--corpus-tokens", type=int, default=1024, metavar="N",
        help="prompt length in tokens when --corpus is set (default 1024)",
    )
    ap.add_argument(
        "--requests", type=int, default=1, metavar="N",
        help="serve N sequential requests on one engine. Process-lifetime "
             "state (hotness counters, LRU evictions) accumulates across "
             "them, which a single request cannot exercise.",
    )
    ap.add_argument(
        "--cache-policy", choices=["lru", "s3fifo"],
        default="lru",
        help="FU2: eviction policy for the app-level cache "
             "(page-cache-only budgets ignore it). A/B vs lru.",
    )
    ap.add_argument(
        "--no-transition", action="store_true",
        help="FU1: disable the transition-table k+1 overfetch in the RA "
             "advisor (A/B arm).",
    )
    ap.add_argument(
        "--moe-offload", type=float, default=None, metavar="FRAC",
        help="upstream official MoE-expert-offload arm: resident fraction "
             "(e.g. 0.25). Disables OUR streaming so exactly one offload "
             "system owns the experts.",
    )
    ap.add_argument(
        "--dflash", action="store_true",
        help="enable DFlash speculative decoding (needs --dflash-draft).",
    )
    ap.add_argument(
        "--dflash-draft", default=None, metavar="PATH",
        help="draft checkpoint for --dflash.",
    )
    ap.add_argument(
        "--v41-offload", type=float, default=None, metavar="FRAC",
        help="DeepSeek-V4.1 native adapter arm: resident expert fraction "
             "(e.g. 0.125). Only meaningful with --model v41; the fraction "
             "is the initial dynamic budget.",
    )
    ap.add_argument(
        "--engram-ssd", action="store_true",
        help="DeepSeek-V4.1 Engram SSD offload (its ~74G tables stay on "
             "SSD instead of RAM).",
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
            _parse_budget(args.budget),
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
            corpus=args.corpus,
            corpus_tokens=args.corpus_tokens,
            requests=args.requests,
            prompt_text=args.prompt,
            moe_offload=args.moe_offload,
            dflash=args.dflash,
            dflash_draft=args.dflash_draft,
            v41_offload=args.v41_offload,
            engram_ssd=args.engram_ssd,
            heartbeat_s=args.heartbeat,
        )
    )


if __name__ == "__main__":
    main()