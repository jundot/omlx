"""Fase H autotuner: per-machine parameter search for expert-streaming models.

Runs coordinate-descent trials of the streaming bench (bench_expert_streaming.py)
under a hard memory-safety rail (watchdog kills the trial on swap growth or an
available-memory floor — the run must never push the box into swap), scores
results against a balanced objective (TTFT 50% + decode tok/s 50%), validates
the winner at the long context, and — with --apply — persists the winning
knobs into the model's per-model settings (the same profile the app edits).

Safety model (the hard rules):
  1. Ceiling per trial = min(enforcer static/metal cap, available − reserve):
     sized to the machine's *available* memory, never its capacity (F1 lesson).
  2. Watchdog: swap growth > max or available < floor (2 consecutive samples)
     → SIGKILL the trial, record a safe-failure, raise the reserve.
  3. Drain: the next trial only starts once available recovers to
     baseline − drain_slack; 2 consecutive watchdog kills abort the session.
  4. Nothing here runs on its own — a human launches the tuner.

Usage:
    .venv/bin/python bench/autotune_expert_streaming.py --model qwen
    .venv/bin/python bench/autotune_expert_streaming.py --model qwen --apply
    .venv/bin/python bench/autotune_expert_streaming.py --model qwen --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_expert_streaming import DEFAULT_ENTRIES, MODEL_PATHS  # noqa: E402

GIB = 1024**3
logger = logging.getLogger("autotune")


# ---------------------------------------------------------------------------
# Knobs (one trial configuration)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Knobs:
    budget_gib: float = 0.0
    io_depth: int = 16
    coalesce: bool = True
    readahead: bool = True
    seed: bool = True
    topk: float | None = None
    prior: float = 0.0
    cold_tier: str | None = None
    hot_fraction: float | None = None

    def label(self) -> str:
        parts = [
            f"b{self.budget_gib:g}",
            f"qd{self.io_depth}",
            f"c{int(self.coalesce)}",
            f"ra{int(self.readahead)}",
            f"s{int(self.seed)}",
        ]
        if self.topk is not None:
            parts.append(f"tk{self.topk:g}")
        if self.prior > 0:
            parts.append(f"cp{self.prior:g}")
        if self.cold_tier is not None:
            parts.append(f"ct{self.cold_tier}")
        if self.hot_fraction is not None:
            parts.append(f"hf{self.hot_fraction:g}")
        return "_".join(parts)

    def env(self) -> dict[str, str]:
        """Env vars for the bench subprocess (bench delivers IO knobs via env)."""
        return {
            "OMLX_EXPERT_STREAMING_QD": str(self.io_depth),
            "OMLX_EXPERT_STREAMING_COALESCE": "1" if self.coalesce else "0",
            "OMLX_EXPERT_STREAMING_RA": "1" if self.readahead else "0",
            "OMLX_EXPERT_STREAMING_SEED": "1" if self.seed else "0",
            # Hermeticity: pin the fidelity fallback per trial so an ambient
            # shell export cannot leak into base/calibration trials (audit).
            "OMLX_EXPERT_STREAMING_CACHE_PRIOR": str(float(self.prior)),
        }

    def profile_kwargs(self) -> dict:
        """Per-model settings kwargs (--apply). topk None → exact routing."""
        return {
            "expert_streaming_budget_gib": float(self.budget_gib),
            "expert_streaming_io_depth": int(self.io_depth),
            "expert_streaming_coalesce": bool(self.coalesce),
            "expert_streaming_readahead": bool(self.readahead),
            "expert_streaming_seed": bool(self.seed),
            "expert_streaming_topk_threshold": self.topk,
            "expert_streaming_cache_prior": (float(self.prior) if self.prior > 0 else None),
            "expert_streaming_cold_tier": self.cold_tier,
            "expert_streaming_hot_fraction": self.hot_fraction,
        }


# Coordinate-descent sweep order: biggest measured lever first.
KNOB_SWEEP_ORDER = ("budget_gib", "io_depth", "coalesce", "readahead", "seed", "topk", "prior", "cold_tier")

KNOB_ATTRS = {
    "budget_gib": "budget_gib",
    "io_depth": "io_depth",
    "coalesce": "coalesce",
    "readahead": "readahead",
    "seed": "seed",
    "topk": "topk",
    "prior": "prior",
    "cold_tier": "cold_tier",
}


# ---------------------------------------------------------------------------
# Scoring (balanced objective) and the memory watchdog (pure, unit-tested)
# ---------------------------------------------------------------------------


def balanced_score(ttft_s: float, tok_s: float, ref_ttft_s: float, ref_toks: float) -> float:
    """Speedup vs the reference trial, TTFT 50% + decode throughput 50%.

    Reference (calibration) scores 0.0; >0 is better than default.
    """
    if ttft_s <= 0 or tok_s <= 0 or ref_ttft_s <= 0 or ref_toks <= 0:
        return float("-inf")
    return 0.5 * (ref_ttft_s / ttft_s - 1.0) + 0.5 * (tok_s / ref_toks - 1.0)


def trial_score(result: "TrialResult", ref: "TrialResult") -> float:
    """Balanced score minus a soft penalty for any swap growth observed."""
    if not result.ok:
        return float("-inf")
    score = balanced_score(
        result.ttft_s or 0.0, result.tok_s or 0.0, ref.ttft_s or 0.0, ref.tok_s or 0.0
    )
    return score - 0.25 * max(0.0, result.swap_growth_gib)


def budget_knee_gib(budget_scores: list[tuple[float, float]]) -> float | None:
    """Smallest budget reaching >=95% of the best score (the LRU knee).

    Input is (budget_gib, score) pairs for completed same-context trials.
    Empty/all-failed input returns None (no knee data). Best-at-zero
    returns 0.0: the bench measured the LRU as adding nothing. The runtime
    caps ``expert_streaming_budget_auto`` at min(8 GiB, knee).
    """
    if not budget_scores:
        return None
    best = max(s for _, s in budget_scores)
    if best == float("-inf"):
        return None
    target = 0.95 * best if best > 0 else best
    cands = sorted(b for b, s in budget_scores if s >= target)
    return float(cands[0]) if cands else None


def write_budget_knee(model_path: Path, model_key: str, knee_gib: float) -> Path:
    """Persist the knee next to the checkpoint for budget_auto to read."""
    dest = Path(model_path) / ".omlx" / "expert_budget_knee.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "version": 1,
                "model": model_key,
                "knee_gib": round(float(knee_gib), 3),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "note": (
                    "smallest budget reaching ~95% of the best autotune "
                    "score; caps expert_streaming_budget_auto"
                ),
            },
            indent=2,
        )
    )
    return dest


@dataclass(frozen=True)
class WatchdogPolicy:
    floor_available_gib: float = 5.0
    max_swap_growth_gib: float = 2.0
    floor_consecutive: int = 2  # available-floor needs N consecutive hits
    poll_s: float = 2.0


def watchdog_eval(
    available_gib: float,
    swap_growth_gib: float,
    consecutive_floor_hits: int,
    policy: WatchdogPolicy,
) -> tuple[str | None, int]:
    """Return (kill_reason, new_consecutive_floor_hits).

    Swap growth is unambiguous → immediate kill. The available floor can dip
    transiently while the OS reclaims, so it requires `floor_consecutive`
    consecutive samples.
    """
    if swap_growth_gib > policy.max_swap_growth_gib:
        return "swap-growth", 0
    if available_gib < policy.floor_available_gib:
        hits = consecutive_floor_hits + 1
        if hits >= policy.floor_consecutive:
            return "available-floor", hits
        return None, hits
    return None, 0


# ---------------------------------------------------------------------------
# Ceiling math and trial planning (pure)
# ---------------------------------------------------------------------------


def compute_ceiling_gib(
    static_ceiling_gib: float,
    metal_cap_gib: float,
    available_gib: float,
    reserve_gib: float,
    floor_gib: float = 6.0,
) -> float:
    """Per-trial scheduler ceiling: min(cap, available − reserve).

    The ceiling is sized to the machine's *available* memory (the F1 lesson:
    a capacity-sized ceiling let a 2k prefill transient hit 38 GiB wired into
    swap on a 22 GB-available box).
    """
    cap = static_ceiling_gib
    if metal_cap_gib > 0:
        cap = min(cap, metal_cap_gib)
    raw = min(cap, available_gib - reserve_gib)
    return round(max(raw, floor_gib), 1)


def prune_depth_candidates(rand_gbps: float | None, seq_gbps: float | None, candidates: list[int]) -> list[int]:
    """Skip the QD sweep when the disk is already near-saturated at QD1."""
    if rand_gbps is None or seq_gbps is None or seq_gbps <= 0:
        return candidates
    if rand_gbps >= 0.75 * seq_gbps:
        return [candidates[len(candidates) // 2]] if candidates else candidates
    return candidates


def screen_candidates(
    base: Knobs,
    *,
    budgets: list[float],
    depths: list[int],
    sweep_topk: bool,
    sweep_prior: bool = False,
    priors: list[float] | None = None,
    cold_tier_available: bool = False,
    hot_fractions: list[float] | None = None,
    sweep_cold_tier: bool = False,
    loaded_est_gib: float | None = None,
    available_gib: float | None = None,
    reserve_gib: float = 10.0,
) -> list[tuple[str, Knobs]]:
    """One-factor-at-a-time trial list.

    Every candidate is carried on `base` (not on an intermediate winner) so
    each trial scores directly against the same calibration reference — the
    head-to-head phase then re-verifies the combination before it ships.
    """
    trials: list[tuple[str, Knobs]] = []
    for knob in KNOB_SWEEP_ORDER:
        if knob == "topk" and not sweep_topk:
            continue
        if knob == "prior" and not sweep_prior:
            continue
        if knob == "budget_gib":
            cands: list = list(budgets)
            # A positive budget lives in RSS: only sweep values that fit
            # alongside the loaded runtime and the machine reserve.
            if loaded_est_gib is not None and available_gib is not None:
                room = available_gib - reserve_gib - loaded_est_gib - 2.0
                cands = [b for b in budgets if b <= max(0.0, room)] or [0.0]
        elif knob == "io_depth":
            cands = list(depths)
        elif knob == "topk":
            cands = [None, 0.85]
        elif knob == "prior":
            cands = list(priors) if priors else [0.0, 1.0, 2.0]
            # The reranker is refused at budget 0 (no LRU to rank with), so
            # a prior arm on a zero-budget base is a no-op re-measurement of
            # the base. Carry prior arms on a positive budget that fits
            # alongside the loaded runtime, mirroring how the budget knob
            # itself picks candidates; with no room at all, skip the arms.
            if getattr(base, "budget_gib", 0.0) <= 0.0:
                room = None
                if loaded_est_gib is not None and available_gib is not None:
                    room = available_gib - reserve_gib - loaded_est_gib - 2.0
                if room is None:
                    carry_budget = 1.0
                else:
                    carry_budget = 1.0 if room >= 1.0 else room
                if carry_budget <= 0.0:
                    cands = []
                else:
                    base = replace(base, budget_gib=carry_budget)
        elif knob == "cold_tier":
            # Only sweep when the model has a materialized expert_cold/ dir
            # (the tier must exist before the runtime can route to it).
            # Opt-in quality lever: requantizing to a cold tier is
            # near-lossless, NOT bit-exact. Never sweep it automatically —
            # only when --sweep-cold-tier is passed AND the tier is on disk
            # (project policy: defaults stay bit-exact).
            if not (sweep_cold_tier and cold_tier_available):
                continue
            hf = replace(base, cold_tier="3", hot_fraction=(hot_fractions[0] if hot_fractions else None))
            trials.append(("cold_tier", hf))
            for frac in hot_fractions[1:] if hot_fractions else []:
                trials.append(("cold_tier", replace(base, cold_tier="3", hot_fraction=frac)))
            continue
        else:
            cands = [True, False]
        for value in cands:
            if getattr(base, KNOB_ATTRS[knob]) == value:
                continue
            trials.append((knob, replace(base, **{KNOB_ATTRS[knob]: value})))
    return trials


def select_best(
    base: Knobs,
    candidates: list[tuple[str, Knobs]],
    evaluate,
    base_score: float = 0.0,
) -> tuple[Knobs, float]:
    """Return the best-scoring config (base included) from the candidate list."""
    best = base
    best_score = base_score
    for _knob, cand in candidates:
        score = evaluate(cand)
        if score > best_score:
            best, best_score = cand, score
    return best, best_score


# ---------------------------------------------------------------------------
# Trial execution (subprocess + watchdog)
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    cfg: Knobs
    context: str
    decode: int
    ceiling_gib: float
    status: str  # ok | failed | killed | skipped
    reason: str | None = None
    ttft_s: float | None = None
    tok_s: float | None = None
    swap_growth_gib: float = 0.0
    phys_after_load_gib: float | None = None
    result_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def row(self) -> dict:
        return {
            "config": self.cfg.label(),
            "context": self.context,
            "decode": self.decode,
            "ceiling_gib": self.ceiling_gib,
            "status": self.status,
            "reason": self.reason,
            "ttft_s": self.ttft_s,
            "tok_s": self.tok_s,
            "swap_growth_gib": round(self.swap_growth_gib, 2),
        }


def bench_command(
    cfg: Knobs,
    *,
    python: str,
    bench_path: Path,
    model_key: str,
    context: str,
    decode: int,
    ceiling_gib: float,
    min_free_gb: float,
    out_dir: Path,
) -> list[str]:
    cmd = [
        python,
        str(bench_path),
        "--model",
        model_key,
        "--budget",
        str(cfg.budget_gib),
        "--decode",
        str(decode),
        "--prompt-len",
        context,
        "--mem-ceiling-gib",
        str(ceiling_gib),
        "--min-free-gb",
        str(min_free_gb),
        "--out",
        str(out_dir / "result.json"),
        "--out-dir",
        str(out_dir),
    ]
    if cfg.topk is not None:
        cmd += ["--topk", str(cfg.topk)]
    if cfg.prior > 0:
        cmd += ["--cache-prior", str(cfg.prior)]
    if cfg.cold_tier is not None:
        cmd += ["--cold-tier", str(cfg.cold_tier)]
        if cfg.hot_fraction is not None:
            cmd += ["--hot-fraction", str(cfg.hot_fraction)]
    return cmd


def run_trial(
    cfg: Knobs,
    *,
    model_key: str,
    context: str,
    decode: int,
    ceiling_gib: float,
    min_free_gb: float,
    out_dir: Path,
    repo_root: Path,
    policy: WatchdogPolicy,
    min_available_gib: float,
    loaded_est_gib: float | None = None,
    reserve_gib: float = 10.0,
) -> TrialResult:
    """Run one bench subprocess under the watchdog rail. Blocking."""
    import psutil

    out_dir.mkdir(parents=True, exist_ok=True)
    vm = psutil.virtual_memory()
    available = vm.available / GIB

    # Trial preflight: skip (don't fail) when the machine can't take this
    # trial right now — either below the bench's own floor or unable to hold
    # the loaded runtime plus the reserve.
    if available < min_available_gib:
        return TrialResult(cfg, context, decode, ceiling_gib, "skipped",
                           reason=f"available {available:.1f}G < floor {min_available_gib:.0f}G")
    if loaded_est_gib is not None and available < loaded_est_gib + reserve_gib + 2.0:
        return TrialResult(cfg, context, decode, ceiling_gib, "skipped",
                           reason=f"available {available:.1f}G < runtime {loaded_est_gib:.1f}G + reserve {reserve_gib:.0f}G + margin")

    baseline_swap = psutil.swap_memory().used / GIB
    log_path = out_dir / "bench.log"
    cmd = bench_command(
        cfg,
        python=sys.executable,
        bench_path=repo_root / "bench" / "bench_expert_streaming.py",
        model_key=model_key,
        context=context,
        decode=decode,
        ceiling_gib=ceiling_gib,
        min_free_gb=min_free_gb,
        out_dir=out_dir,
    )
    env = {**os.environ, **cfg.env()}
    logger.info("trial %s: ceiling %.1fG ctx %s decode %d", cfg.label(), ceiling_gib, context, decode)
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

        stop = threading.Event()
        kill = {"reason": None}
        peak_swap_growth = 0.0
        floor_hits = 0

        def _watch() -> None:
            nonlocal peak_swap_growth, floor_hits
            while not stop.wait(policy.poll_s):
                if proc.poll() is not None:
                    return
                try:
                    vm_now = psutil.virtual_memory()
                    sw_now = psutil.swap_memory()
                except Exception:
                    continue
                growth = sw_now.used / GIB - baseline_swap
                peak_swap_growth = max(peak_swap_growth, growth)
                reason, floor_hits = watchdog_eval(
                    vm_now.available / GIB, growth, floor_hits, policy
                )
                if reason is not None:
                    kill["reason"] = reason
                    logger.warning(
                        "WATCHDOG killing trial %s: %s (available %.1fG, swap growth %.2fG)",
                        cfg.label(), reason, vm_now.available / GIB, growth,
                    )
                    proc.kill()
                    return

        watcher = threading.Thread(target=_watch, daemon=True, name="autotune-watchdog")
        watcher.start()
        try:
            proc.wait()
        finally:
            stop.set()
            watcher.join(timeout=5)

    status, reason = "ok", None
    if kill["reason"] is not None:
        status, reason = "killed", kill["reason"]
        _print_log_tail(log_path)
    elif proc.returncode != 0:
        status = "failed"
        reason = f"bench exit {proc.returncode} (tail: {log_path})"
        _print_log_tail(log_path)

    result = TrialResult(
        cfg=cfg,
        context=context,
        decode=decode,
        ceiling_gib=ceiling_gib,
        status=status,
        reason=reason,
        swap_growth_gib=peak_swap_growth,
        result_path=str(out_dir / "result.json") if (out_dir / "result.json").exists() else None,
    )
    if (out_dir / "result.json").exists():
        try:
            data = json.loads((out_dir / "result.json").read_text())
            result.ttft_s = data.get("ttft_s")
            result.tok_s = data.get("tok_s")
            result.phys_after_load_gib = data.get("phys_after_load_gib")
            if status == "ok" and (result.ttft_s is None or result.tok_s is None):
                result.status, result.reason = "failed", "result.json missing ttft/tok"
        except (OSError, json.JSONDecodeError) as exc:
            if status == "ok":
                result.status, result.reason = "failed", f"result.json unreadable: {exc}"
    return result


def _print_log_tail(log_path: Path, lines: int = 25) -> None:
    try:
        tail = log_path.read_text(errors="replace").splitlines()[-lines:]
        for line in tail:
            logger.warning("bench| %s", line)
    except OSError:
        pass


def wait_for_drain(target_available_gib: float, timeout_s: float = 180.0, poll_s: float = 2.0) -> bool:
    """Wait until available memory recovers to the target (trial teardown)."""
    import psutil

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if psutil.virtual_memory().available / GIB >= target_available_gib:
                return True
        except Exception:
            return True  # telemetry broken → don't deadlock the session
        time.sleep(poll_s)
    return False


# ---------------------------------------------------------------------------
# Machine probe (no model load)
# ---------------------------------------------------------------------------


@dataclass
class MachineProfile:
    total_gib: float
    available_gib: float
    swap_used_gib: float
    static_ceiling_gib: float
    metal_cap_gib: float
    tier: str
    checkpoint_gib: float
    seq_gbps: float | None = None
    rand_gbps: float | None = None

    def dict(self) -> dict:
        d = asdict(self)
        return d


def probe_machine(model_path: Path, tier: str = "balanced") -> MachineProfile:
    """RAM/swap snapshot + enforcer ceiling components + checkpoint size."""
    import psutil

    from omlx.process_memory_enforcer import (
        ProcessMemoryEnforcer,
        get_effective_metal_cap_bytes,
    )

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    # Constructor is pure (no monitor thread, no wired-limit side effect —
    # start() is what launches the poll loop).
    enforcer = ProcessMemoryEnforcer(None, memory_guard_tier=tier)
    breakdown = enforcer._get_ceiling_breakdown()
    checkpoint = sum(p.stat().st_size for p in model_path.glob("*.safetensors"))
    return MachineProfile(
        total_gib=vm.total / GIB,
        available_gib=vm.available / GIB,
        swap_used_gib=sw.used / GIB,
        static_ceiling_gib=breakdown["static"] / GIB,
        metal_cap_gib=breakdown["metal_cap"] / GIB,
        tier=tier,
        checkpoint_gib=checkpoint / GIB,
    )


def probe_ssd(
    model_path: Path,
    seq_bytes: int = 2 * GIB,
    rand_reads: int = 48,
    rand_size: int = 16 * 1024 * 1024,
) -> tuple[float, float]:
    """(sequential GB/s, random-expert-size GB/s) on the model's largest shard.

    The random probe mimics expert demand reads (single ~12.75 MiB preads,
    issued serially = QD1) and feeds the QD-sweep pruning decision.
    """
    shards = sorted(model_path.glob("*.safetensors"), key=lambda p: p.stat().st_size, reverse=True)
    if not shards:
        return (0.0, 0.0)
    shard = shards[0]
    size = shard.stat().st_size
    import os
    import random as _random

    fd = os.open(shard, os.O_RDONLY)
    try:
        # Sequential: streaming prefill reads big contiguous runs.
        t0 = time.perf_counter()
        transferred = 0
        offset = 0
        chunk = 8 * 1024 * 1024
        while transferred < seq_bytes and offset < size:
            n = os.pread(fd, chunk, offset)
            if not n:
                break
            transferred += len(n)
            offset += len(n)
        seq_gbps = transferred / (time.perf_counter() - t0) / GIB

        # Random expert-granularity (serial issue ≈ QD1).
        rng = _random.Random(0)
        t0 = time.perf_counter()
        rand_bytes = 0
        for _ in range(rand_reads):
            off = rng.randrange(0, max(1, size - rand_size)) // rand_size * rand_size
            n = os.pread(fd, rand_size, off)
            rand_bytes += len(n)
        rand_gbps = rand_bytes / (time.perf_counter() - t0) / GIB
    finally:
        os.close(fd)
    return (round(seq_gbps, 2), round(rand_gbps, 2))


# ---------------------------------------------------------------------------
# Recommendation + --apply (per-model profile via ModelSettingsManager)
# ---------------------------------------------------------------------------


def build_recommendation(
    *,
    model_key: str,
    model_id: str,
    machine: MachineProfile,
    winner: Knobs,
    winner_score: float,
    trials: list[TrialResult],
    screen_context: str,
    validate_context: str,
    applied: bool,
    notes: list[str] | None = None,
    budget_knee: float | None = None,
) -> dict:
    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "balanced",
        "model": model_key,
        "model_id": model_id,
        "machine": machine.dict(),
        "screen_context": screen_context,
        "validate_context": validate_context,
        "budget_knee_gib": budget_knee,
        "winner": winner.profile_kwargs() | {"label": winner.label()}, 
        "winner_score": round(winner_score, 4) if winner_score == winner_score else None,
        "trials": [t.row() for t in trials],
        "applied_to_profile": applied,
        "notes": notes or [],
    }


def apply_to_profile(model_id: str, knobs: Knobs) -> Path:
    """Persist the winning knobs as the model's per-model configuration.

    Uses the same store the app edits (~/.omlx/model_settings.json). Run this
    with the server stopped (or reload the model afterwards): a running
    server keeps its own in-memory manager and will overwrite on its next
    settings save.
    """
    from omlx.model_settings import ModelSettingsManager
    from omlx.settings import resolve_default_base_path

    mgr = ModelSettingsManager(resolve_default_base_path())
    current = mgr.get_settings(model_id)
    for key, value in knobs.profile_kwargs().items():
        setattr(current, key, value)
    mgr.set_settings(model_id, current)
    logger.info("applied %s to %s in %s", knobs.label(), model_id, resolve_default_base_path())
    return resolve_default_base_path() / "model_settings.json"


# ---------------------------------------------------------------------------
# Session orchestration
# ---------------------------------------------------------------------------


def run_session(opts: argparse.Namespace) -> int:
    import psutil

    model_path = Path(MODEL_PATHS[opts.model])
    model_id = DEFAULT_ENTRIES[opts.model]
    # Cold-tier autotuning is gated on a materialized tier: only sweep
    # expert_streaming_cold_tier when this checkpoint actually has
    # <model>/expert_cold/ on disk (generated by tools/requant_cold_tier.py).
    cold_tier_available = (model_path / "expert_cold").is_dir()
    hot_fractions = [float(x) for x in str(opts.hot_fractions).split(",") if x.strip()] if getattr(opts, "hot_fractions", None) else []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(opts.out_root) / f"{opts.model}_{stamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent
    policy = WatchdogPolicy(
        floor_available_gib=opts.watchdog_floor_gib,
        max_swap_growth_gib=opts.watchdog_swap_gib,
    )
    reserve = opts.reserve_gib

    # ---- Phase 0: machine probe -------------------------------------------
    print("=== Fase H autotune: machine probe ===", flush=True)
    machine = probe_machine(model_path, tier=opts.tier)
    seq_gbps, rand_gbps = probe_ssd(model_path)
    machine.seq_gbps = seq_gbps
    machine.rand_gbps = rand_gbps
    print(
        f"ram {machine.total_gib:.0f}G available {machine.available_gib:.1f}G "
        f"swap {machine.swap_used_gib:.1f}G | static ceiling {machine.static_ceiling_gib:.1f}G "
        f"metal cap {machine.metal_cap_gib:.1f}G (tier {machine.tier})",
        flush=True,
    )
    print(
        f"checkpoint {machine.checkpoint_gib:.1f}G | ssd seq {seq_gbps:.2f} GB/s "
        f"random(QD1) {rand_gbps:.2f} GB/s",
        flush=True,
    )
    if machine.available_gib < opts.min_free_gb and not opts.dry_run:
        print(
            f"ABORT: only {machine.available_gib:.1f}G available (need {opts.min_free_gb:.0f}G). "
            "The autotuner refuses to start on a loaded machine.",
            flush=True,
        )
        return 2
    if machine.available_gib < opts.min_free_gb:
        print(
            f"WARNING: only {machine.available_gib:.1f}G available — real runs would refuse to start.",
            flush=True,
        )

    depths = prune_depth_candidates(rand_gbps, seq_gbps, opts.qd)
    if depths != opts.qd:
        print(f"QD sweep pruned by SSD probe (near-saturated at QD1): {depths}", flush=True)
    budgets = list(opts.budgets)
    base = Knobs()

    baseline_available = machine.available_gib
    trials: list[TrialResult] = []
    consecutive_kills = 0
    loaded_est: float | None = None
    trial_no = 0

    def _do_trial(cfg: Knobs, context: str, decode: int) -> TrialResult:
        nonlocal trial_no, reserve, consecutive_kills
        trial_no += 1
        vm_now = psutil.virtual_memory()
        avail_now = vm_now.available / GIB
        ceiling = compute_ceiling_gib(
            machine.static_ceiling_gib, machine.metal_cap_gib, avail_now, reserve
        )
        result = run_trial(
            cfg,
            model_key=opts.model,
            context=context,
            decode=decode,
            ceiling_gib=ceiling,
            min_free_gb=opts.min_free_gb,
            out_dir=session_dir / f"trial_{trial_no:02d}_{context}_{cfg.label()}",
            repo_root=repo_root,
            policy=policy,
            min_available_gib=opts.min_free_gb,
            loaded_est_gib=loaded_est,
            reserve_gib=reserve,
        )
        trials.append(result)
        row = result.row()
        print(
            f"  [{trial_no:02d}] {cfg.label()} {context} → {result.status}"
            + (f" ttft {result.ttft_s:.1f}s {result.tok_s:.3f} tok/s" if result.ok else f" ({result.reason})"),
            flush=True,
        )
        (session_dir / f"trial_{trial_no:02d}_row.json").write_text(json.dumps(row, indent=2))
        if result.status == "killed":
            consecutive_kills += 1
            reserve = min(reserve + 2.0, baseline_available / 2)  # back off
            if consecutive_kills >= 2:
                print(
                    "ABORT: two watchdog kills in a row — the machine is too loaded "
                    "for tuning right now. Close apps or rerun later.",
                    flush=True,
                )
                raise SystemExit(2)
        elif result.ok:
            consecutive_kills = 0
        # Drain: wait for teardown memory to come back before the next trial.
        target = max(opts.min_free_gb, baseline_available - opts.drain_slack_gib)
        if not wait_for_drain(target):
            print(
                f"ABORT: machine did not recover to {target:.1f}G available after the trial — "
                "something else grabbed the memory (or teardown leaked). Session stopped.",
                flush=True,
            )
            raise SystemExit(3)
        return result

    if opts.dry_run:
        cands = screen_candidates(
            base,
            budgets=budgets,
            depths=depths,
            sweep_topk=opts.sweep_topk,
            sweep_prior=opts.sweep_prior,
            priors=opts.priors,
            cold_tier_available=cold_tier_available,
        sweep_cold_tier=opts.sweep_cold_tier,
            hot_fractions=hot_fractions,
            loaded_est_gib=None,
            available_gib=baseline_available,
            reserve_gib=reserve,
        )
        print(f"[dry-run] would run 1 calibration + {len(cands)} screening trials (+head-to-head/validation)")
        for knob, cfg in cands:
            print(f"[dry-run]   {knob}: {cfg.label()}")
        rec = build_recommendation(
            model_key=opts.model, model_id=model_id, machine=machine,
            winner=base, winner_score=0.0, trials=[], screen_context=opts.screen_context,
            validate_context=opts.validate_context, applied=False,
            notes=["dry-run — no trials executed"],
        )
        (session_dir / "recommendation.json").write_text(json.dumps(rec, indent=2))
        print(f"[dry-run] wrote {session_dir / 'recommendation.json'}")
        return 0

    # ---- Phase 1: calibration (discarded; warms page cache, measures load) -
    print("=== Phase 1: calibration trial (default config, discarded) ===", flush=True)
    calib = _do_trial(base, opts.screen_context, opts.screen_decode)
    if not calib.ok:
        print(f"ABORT: calibration trial did not complete ({calib.reason}).", flush=True)
        return 2
    loaded_est = calib.phys_after_load_gib or machine.checkpoint_gib * 0.2

    def evaluate_factory(context: str, decode: int):
        def _evaluate(cfg: Knobs) -> float:
            r = _do_trial(cfg, context, decode)
            if not r.ok:
                return float("-inf")
            return trial_score(r, calib)

        return _evaluate

    # ---- Phase 2: one-factor-at-a-time screening ---------------------------
    print("=== Phase 2: screening (one factor at a time) ===", flush=True)
    cands = screen_candidates(
        base,
        budgets=budgets,
        depths=depths,
        sweep_topk=opts.sweep_topk,
        sweep_prior=opts.sweep_prior,
        priors=opts.priors,
        cold_tier_available=cold_tier_available,
        hot_fractions=hot_fractions,
        loaded_est_gib=loaded_est,
        available_gib=baseline_available,
        reserve_gib=reserve,
    )
    best_cfg, _best_score = select_best(
        base, cands, evaluate_factory(opts.screen_context, opts.screen_decode), base_score=0.0
    )

    # ---- Phase 3: head-to-head + long-context validation --------------------
    notes: list[str] = []
    final_cfg = best_cfg
    if best_cfg != base and not opts.skip_validation:
        print("=== Phase 3: head-to-head (best vs default) ===", flush=True)
        rerun_best = _do_trial(best_cfg, opts.screen_context, opts.screen_decode)
        rerun_base = _do_trial(base, opts.screen_context, opts.screen_decode)
        if rerun_best.ok and rerun_base.ok and trial_score(rerun_base, calib) > trial_score(rerun_best, calib):
            final_cfg = base
            notes.append("head-to-head favored the default config; screening win did not reproduce")
        best_cfg = final_cfg

        print(f"=== Phase 4: validation at {opts.validate_context} (winner vs default) ===", flush=True)
        val_winner = _do_trial(best_cfg, opts.validate_context, opts.validate_decode)
        val_base = _do_trial(base, opts.validate_context, opts.validate_decode)
        if val_winner.ok and val_base.ok:
            w = balanced_score(val_winner.ttft_s, val_winner.tok_s, val_base.ttft_s, val_base.tok_s)
            if w < 0:
                notes.append(
                    f"winner regressed at {opts.validate_context} (score {w:+.3f} vs default); "
                    "recommending the default config for the profile"
                )
                best_cfg = base
            else:
                notes.append(f"winner confirmed at {opts.validate_context} (score {w:+.3f} vs default)")
        else:
            notes.append(
                "long-context validation incomplete "
                f"(winner {val_winner.status}, default {val_base.status}); profile carries the screening winner"
            )

    print(f"=== Winner: {best_cfg.label()} ===", flush=True)
    # LRU knee from same-context screening scores: caps budget_auto so the
    # cache stops growing where the bench stopped paying off.
    knee_gib = budget_knee_gib(
        [
            (t.cfg.budget_gib, trial_score(t, calib))
            for t in trials
            if t.ok and t.context == opts.screen_context
        ]
    )
    print(f"Budget knee: {knee_gib if knee_gib is not None else 'n/a'} GiB", flush=True)
    applied = False
    if opts.apply:
        try:
            path = apply_to_profile(model_id, best_cfg)
            applied = True
            print(f"Applied to per-model profile: {path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: --apply failed: {exc}", flush=True)
        if knee_gib is not None:
            try:
                knee_path = write_budget_knee(model_path, opts.model, knee_gib)
                print(f"Wrote budget knee: {knee_path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: knee write failed: {exc}", flush=True)

    rec = build_recommendation(
        model_key=opts.model,
        model_id=model_id,
        machine=machine,
        winner=best_cfg,
        winner_score=trial_score(
            next((t for t in trials if t.cfg == best_cfg and t.ok), calib), calib
        ),
        trials=trials,
        screen_context=opts.screen_context,
        validate_context=opts.validate_context,
        applied=applied,
        notes=notes,
        budget_knee=knee_gib,
    )
    (session_dir / "recommendation.json").write_text(json.dumps(rec, indent=2))
    (session_dir / "trials.json").write_text(json.dumps([t.row() for t in trials], indent=2))
    print(f"Session artifacts: {session_dir}", flush=True)
    print("Run again with --apply to persist the winner, or inspect recommendation.json first.", flush=True)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=list(MODEL_PATHS))
    ap.add_argument("--screen-context", choices=["short", "512", "2k"], default="2k")
    ap.add_argument("--validate-context", choices=["512", "2k", "8k"], default="8k")
    ap.add_argument("--screen-decode", type=int, default=32)
    ap.add_argument("--validate-decode", type=int, default=96)
    ap.add_argument("--budgets", default="0,2,4", help="comma-separated budget_gib candidates")
    ap.add_argument("--qd", type=int, nargs="+", default=[8, 16, 32], help="IO depth candidates")
    ap.add_argument("--sweep-topk", action="store_true",
                    help="also sweep expert_streaming_topk_threshold (trades output fidelity)")
    ap.add_argument("--sweep-cold-tier", action="store_true",
                    help="sweep expert_streaming_cold_tier (requires <model>/expert_cold/; gated automatically)")
    ap.add_argument("--hot-fractions", default="0.25,0.5",
                    help="comma-separated hot_fraction candidates for the cold-tier arm")
    ap.add_argument("--sweep-prior", action="store_true",
                    help="also sweep expert_streaming_cache_prior (trades output fidelity)")
    ap.add_argument("--priors", default="0.0,1.0,2.0", help="comma-separated cache_prior candidates (calibration: 2.0 short+2k winner, 4.0 degenerates)")
    ap.add_argument("--reserve-gib", type=float, default=10.0,
                    help="memory kept away from the bench ceiling: your apps + KV + headroom")
    ap.add_argument("--min-free-gb", type=float, default=22.0)
    ap.add_argument("--watchdog-swap-gib", type=float, default=2.0,
                    help="kill the trial when swap grows beyond this (hard rail)")
    ap.add_argument("--watchdog-floor-gib", type=float, default=5.0,
                    help="kill the trial when available stays below this")
    ap.add_argument("--drain-slack-gib", type=float, default=2.0)
    ap.add_argument("--tier", choices=["safe", "balanced", "aggressive", "custom"], default="balanced")
    ap.add_argument("--out-root", default="bench/results/autotune")
    ap.add_argument("--apply", action="store_true",
                    help="persist the winner into the model's per-model settings (~/.omlx)")
    ap.add_argument("--skip-validation", action="store_true", help="screening only (no 8k validation)")
    ap.add_argument("--dry-run", action="store_true", help="probe the machine, print the plan, run nothing")
    opts = ap.parse_args()
    opts.budgets = [float(b) for b in opts.budgets.split(",")]
    opts.priors = [float(p) for p in opts.priors.split(",")]
    return run_session(opts)


if __name__ == "__main__":
    raise SystemExit(main())
