"""Storage roofline: measure the SSD ceiling for MoE expert streaming.

Measures uncached sequential + random bandwidth on the volume holding the
checkpoint, derives stored expert bytes per decode step from the checkpoint
headers, and prints the predicted tok/s ceiling plus the structural MTP
verdict (tok/cycle vs verify byte multiplier).

Usage:
    .venv/bin/python bench/bench_storage_roofline.py --model qwen-jang4m
    .venv/bin/python bench/bench_storage_roofline.py --model qwen-jang --tok-per-cycle 1.79 --verify-mult 2.3 --measured-base 3.55
    .venv/bin/python bench/bench_storage_roofline.py --volume-only --dir "/Volumes/SSD 4TB/AI Models"

Protocol: run with inference IDLE (no loaded model, no downloads) — any
concurrent IO on the same volume steals bandwidth and the ceiling reads low.
The report lands in bench/results/storage_roofline/ (gitignored, like benches).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_expert_streaming import MODEL_PATHS  # noqa: E402

from omlx.utils.storage_roofline import (  # noqa: E402
    build_report,
    measure_storage,
    moe_step_profile,
    predict_roofline,
    save_report,
    volume_info_for,
)


def _slug(model_key: str | None, directory: Path) -> str:
    base = model_key or directory.name.replace(" ", "_")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{base}_{stamp}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=list(MODEL_PATHS),
                    help="model key (same table as bench_expert_streaming)")
    ap.add_argument("--model-dir", default=None,
                    help="explicit checkpoint dir (overrides --model)")
    ap.add_argument("--volume-only", action="store_true",
                    help="measure storage only, no model profile/prediction")
    ap.add_argument("--dir", default=None,
                    help="directory on the target volume (volume-only mode)")
    ap.add_argument("--file-gb", type=float, default=2.0)
    ap.add_argument("--read-mb", type=int, default=2)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tok-per-cycle", type=float, default=1.0,
                    help="measured MTP tok/cycle (1 + accept for depth-1)")
    ap.add_argument("--verify-mult", type=float, default=2.3,
                    help="verify/base byte ratio (Gap-2 measured 2.3)")
    ap.add_argument("--measured-base", type=float, default=None,
                    help="measured base tok/s to calibrate efficiency")
    ap.add_argument("--out", default=None, help="explicit report path")
    args = ap.parse_args()

    if args.volume_only:
        target = Path(args.dir or ".").expanduser()
        model_dir = None
        model_key = None
    else:
        model_dir = Path(args.model_dir or (MODEL_PATHS[args.model] if args.model else ""))
        if not args.model and not args.model_dir:
            ap.error("need --model, --model-dir, or --volume-only")
        if not model_dir.is_dir():
            ap.error(f"model dir not found: {model_dir}")
        target = model_dir
        model_key = args.model or model_dir.name.replace(" ", "_")

    print(f"volume : {target}", flush=True)
    vol = volume_info_for(target)
    print(f"media  : {vol.media_name or '?'} ({vol.protocol or '?'}, "
          f"{'SSD' if vol.solid_state else 'HDD?'}, {vol.location or '?'})", flush=True)
    print(f"free   : {vol.free_bytes / 1024**3:.0f} GiB", flush=True)
    if vol.free_bytes < int(args.file_gb * 1024**3) + 1024**3:
        ap.error("not enough free space for the scratch file + 1 GiB headroom")

    def progress(phase: str, done: int, total: int) -> None:
        # Throttled single-line progress; benches run for minutes otherwise.
        if phase == "done":
            return
        if total and (done == total or done // (64 * 1024 * 1024) != getattr(progress, "_last", -1)):
            progress._last = done // (64 * 1024 * 1024)  # type: ignore[attr-defined]
            print(f"  {phase}: {done / 1024**2:.0f}/{total / 1024**2:.0f} MiB", flush=True)

    meas = measure_storage(
        target, file_gb=args.file_gb, read_mb=args.read_mb,
        samples=args.samples, seed=args.seed, progress=progress,
    )
    print(f"seq    : {meas.seq_read_Bps / 1024**3:.2f} GiB/s (spill/load predictor)", flush=True)
    print(f"rand{meas.read_mb}MB : {meas.rand_read_Bps / 1024**2:.0f} MB/s, "
          f"{meas.rand_iops:.0f} IOPS, "
          f"p50/p99 {meas.rand_lat_ms_p50:.2f}/{meas.rand_lat_ms_p99:.2f} ms "
          f"(decode predictor, method={meas.method}, cache_clean={meas.cache_clean})",
          flush=True)
    print(f"write  : {meas.write_Bps / 1024**3:.2f} GiB/s", flush=True)
    for w in meas.warnings:
        print(f"warn   : {w}", flush=True)

    profile = None
    prediction = None
    if not args.volume_only and model_dir is not None:
        profile = moe_step_profile(model_dir)
        if not profile.supported:
            print(f"profile: unsupported ({profile.reason})", flush=True)
        else:
            print(f"profile: {profile.model_type} {profile.num_moe_layers} MoE layers x "
                  f"top{profile.top_k}/{profile.routed_total_per_layer} = "
                  f"{profile.bytes_per_step / 1024**2:.1f} MiB/step "
                  f"(+{profile.shared_bytes_per_layer / 1024**2:.1f} MiB resident shared/layer)",
                  flush=True)
            prediction = predict_roofline(
                profile, meas, tok_per_cycle=args.tok_per_cycle,
                verify_byte_mult=args.verify_mult,
            )
            print(f"ceiling: base {prediction.ceiling_base_tok_s:.2f} tok/s", flush=True)
            print(f"verdict: {prediction.explanation}", flush=True)

    report = build_report(vol, meas, profile, prediction,
                          measured_base_tok_s=args.measured_base)
    if args.measured_base and "calibration" in report:
        cal = report["calibration"]
        eff = cal["efficiency"] * 100
        if eff > 100:
            note = "(above the cold ceiling: temporal locality + prefetch dividend)"
        elif eff >= 70:
            note = "(near the cold ceiling: decode is SSD-bound)"
        else:
            note = "(well under ceiling: bottleneck is elsewhere — CPU/Metal/scheduler)"
        print(f"calib  : measured {cal['measured_base_tok_s']:.2f} tok/s = "
              f"{eff:.0f}% of cold ceiling {note}", flush=True)
    out = Path(args.out) if args.out else None
    if out is None:
        saved = save_report(report, _slug(model_key, target))
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        saved = out
    print(f"report : {saved}", flush=True)


if __name__ == "__main__":
    main()
