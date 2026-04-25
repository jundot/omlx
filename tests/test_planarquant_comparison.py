# SPDX-License-Identifier: Apache-2.0
"""PlanarQuant synthetic performance/accuracy comparison smoke tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_planarquant.py"
_SPEC = importlib.util.spec_from_file_location("compare_planarquant", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
run_comparisons = _MODULE.run_comparisons


def test_synthetic_comparison_smoke():
    results = run_comparisons(lengths=[32], batch=1, heads=2, dim=128, iters=2)

    assert {r["mode"] for r in results} == {"k_v", "k_only"}
    for row in results:
        assert row["cosine"] > 0.95
        assert row["packed_compression"] > 1.0
        assert row["fp16_decode_ms"] > 0.0
        assert row["pq_decode_warm_ms"] > 0.0
