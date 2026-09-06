"""Fase M5: compare two bench result JSONs and refuse invalid A/B.

A comparison is only fair when the EFFECTIVE configuration matches on
every critical field (instrumentation, token gate, chunk schedule, cache
protocol, knobs). The comparator reads each result's "effective_config"
block, allows the experiment knobs each side declared, and reports the
performance delta only when the comparison is fair.

Usage:
    python bench/compare_results.py arm_A.json arm_B.json
    python bench/compare_results.py --allow pins_enabled a.json b.json
"""

import argparse
import json
import sys
from pathlib import Path

# Fields that must match exactly between two comparable arms. The metrics
# (tok_s, ttft_s, metal peaks, read_stats) are the OUTPUT, never input.
CRITICAL_FIELDS = (
    "git_sha",
    "model_fingerprint",
    "single_request",
    "decode_tokens",
    "chunk_schedule",
    "budget_gib",
    "cold_tier",
    "hot_fraction",
    "ctx_mode_policy",
    "decode_union_rows",
    "ctx_ahead",
    "expert_qd",
    "run_qd",
    "prefill_qd",
    "run_merge_gap",
    "ra_enabled",
    "pins_enabled",
    "pin_sync_effective",
    "pin_regime_effective",
    "profile_enabled",
    "memtrace_enabled",
    "read_sampling_mode",
    "cache_cool_protocol",
    # Fase A4: an A/B where one side ran with a second engine in-process
    # and the other did not compares different pool worlds — refused.
    "active_engines",
)

METRIC_FIELDS = ("ttft_s", "tok_s", "metal_peak_prefill_gib", "metal_peak_decode_gib")


def load_result(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def declared_knobs(data: dict) -> set[str]:
    cfg = data.get("effective_config") or {}
    return set(cfg.get("experiment_knobs") or [])


def collect_mismatches(a: dict, b: dict, allow: set[str] | None = None) -> list[str]:
    """Critical-field differences between two results. Every mismatch is
    one human-readable string; empty means the comparison is fair."""
    allow = allow or set()
    issues: list[str] = []
    ca = a.get("effective_config")
    cb = b.get("effective_config")
    if not ca or not cb:
        issues.append(

            "missing effective_config block: %s"

            % ("a" if not ca else "b")

        )
        return issues
    for field in CRITICAL_FIELDS:
        if field in allow or field in declared_knobs(a) and field in declared_knobs(b):
            continue
        va, vb = ca.get(field), cb.get(field)
        if va != vb:
            issues.append(f"{field}: {va!r} vs {vb!r}")
    # Token-gate kind must match when both results carry it.
    ka, kb = a.get("bit_exact_kind"), b.get("bit_exact_kind")
    if ka is not None and kb is not None and ka != kb:
        issues.append(f"bit_exact_kind: {ka!r} vs {kb!r}")
    # Fase A2 fail-high: a text-only "gate" proves nothing about token
    # identity, and an EMPTY token list is a broken gate by construction —
    # neither may enter a comparison, even when both sides share the flaw.
    for side, d in (("a", a), ("b", b)):
        toks = d.get("tokens")
        if isinstance(toks, list) and len(toks) == 0:
            issues.append(f"{side}: tokens list is empty — token gate cannot pass")
        if d.get("bit_exact_kind") == "text":
            issues.append(
                f"{side}: bit_exact_kind == 'text' cannot prove token identity"
            )
    # Fase A3: the stage-bucket VOCABULARY must match. read_stats with
    # different stage keys (e.g. the pre-A3 metric names) cannot be
    # compared stage-wise, so the comparison is refused.
    def _stage_keys(d: dict) -> set:
        rs = d.get("read_stats") or {}
        acc = rs.get("lifetime") or {}
        st = acc.get("stages_us") or {}
        return set(st)

    _ka3, _kb3 = _stage_keys(a), _stage_keys(b)
    if _ka3 and _kb3 and _ka3 != _kb3:
        issues.append(
            "read_stats stage keys: %s vs %s" % (sorted(_ka3), sorted(_kb3))
        )
    return issues


def comparison_summary(a: dict, b: dict) -> dict:
    out = {}
    for f in METRIC_FIELDS:
        va, vb = a.get(f), b.get(f)
        if va is None or vb is None:
            continue
        out[f] = {
            "a": round(float(va), 4),
            "b": round(float(vb), 4),
            "delta_pct": round((float(vb) - float(va)) / float(va) * 100.0, 2),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", help="first bench result JSON")
    ap.add_argument("b", help="second bench result JSON")
    ap.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="FIELD",
        help="additionally allow this effective-config field to differ",
    )
    args = ap.parse_args()
    a = load_result(args.a)
    b = load_result(args.b)
    shared = declared_knobs(a) & declared_knobs(b)
    issues = collect_mismatches(a, b, set(args.allow) | shared)
    if issues:
        print("INCOMPARABLE (%d mismatch(es)):" % len(issues))
        for issue in issues:
            print("  -", issue)
        print("declared knobs:", sorted(shared))
        return 2
    print("FAIR COMPARISON")
    for field, v in comparison_summary(a, b).items():
        print(

            "  %-24s a=%-10s b=%-10s delta=%+.2f%%"
            % (field, v["a"], v["b"], v["delta_pct"]),

        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
