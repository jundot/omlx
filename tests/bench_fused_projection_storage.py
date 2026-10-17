# SPDX-License-Identifier: Apache-2.0
"""Measure retained fused projection storage and warm projection timings."""

import argparse
import ast
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.speculative.ops import linear as ops

from omlx.patches import qwen35_gdn_prework as prework


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:omlx/patches/qwen35_gdn_prework.py"],
        text=True,
    )
    node = next(
        item
        for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_one_row_fused_projections"
    )
    namespace = dict(vars(prework))
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "<baseline>", "exec"),
        namespace,
    )
    before_local = namespace["_one_row_fused_projections"]
    prework.apply_qwen35_gdn_prework_patch()
    before_upstream = getattr(
        ops._decode_quantized_linears_fused,
        "__wrapped__",
        ops._decode_quantized_linears_fused,
    )
    rows = []
    for path in ("omlx", "mlx-vlm"):
        for cycle in range(3):
            for arm in ("main", "shared", "shared", "main"):
                mx.random.seed(82)
                linears = [
                    nn.QuantizedLinear(5120, count, bias=False, group_size=64, bits=4)
                    for count in (10240, 6144, 48, 48)
                ]
                for linear in linears:
                    linear.set_dtype(mx.bfloat16)
                linear = None
                mx.eval([item.parameters() for item in linears])
                x = mx.random.normal((1, 1, 5120)).astype(mx.bfloat16)
                mx.eval(x)
                mx.synchronize()
                gc.collect()
                mx.clear_cache()
                initial = mx.get_active_memory()
                mx.reset_peak_memory()
                if path == "omlx":
                    selected = (
                        before_local
                        if arm == "main"
                        else prework._one_row_fused_projections
                    )

                    def call(selected=selected, linears=linears, x=x):
                        w, s, b, split, group_size, bits, mode = selected(linears)
                        return tuple(
                            mx.split(
                                mx.quantized_matmul(
                                    x,
                                    w,
                                    scales=s,
                                    biases=b,
                                    transpose=True,
                                    group_size=group_size,
                                    bits=bits,
                                    mode=mode,
                                ),
                                split,
                                axis=-1,
                            )
                        )

                else:
                    selected = (
                        before_upstream
                        if arm == "main"
                        else ops._decode_quantized_linears_fused
                    )

                    def call(selected=selected, linears=linears, x=x):
                        return selected(linears, x)

                for _ in range(3):
                    mx.eval(call())
                first_peak = mx.get_peak_memory()
                mx.synchronize()
                gc.collect()
                mx.clear_cache()
                retained = mx.get_active_memory()
                cache = linears[0]._fused_decode_linears
                mx.reset_peak_memory()
                timings = []
                for _ in range(20):
                    start = time.perf_counter()
                    mx.eval(call())
                    timings.append((time.perf_counter() - start) * 1000)
                reused = cache is linears[0]._fused_decode_linears
                rows.append(
                    {
                        "path": path,
                        "cycle": cycle,
                        "arm": arm,
                        "initial_bytes": initial,
                        "retained_bytes": retained,
                        "cold_fusion_peak_bytes": first_peak,
                        "warm_peak_bytes": mx.get_peak_memory(),
                        "warm_median_ms": statistics.median(timings),
                        "cache_reused": reused,
                    }
                )
                assert reused
                del linears, x, cache, call, selected
                mx.synchronize()
                gc.collect()
                mx.clear_cache()
    print(
        json.dumps(
            {
                "base": args.baseline,
                "conditions": "M1 Max validation: 27B GDN geometry; original4bit/group64/BF16 scales; three ABBA blocks/path;3discarded warmups;20evaluated warm calls/sample; memory includes only this process; no engine-throughput claim",
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
