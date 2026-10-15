# SPDX-License-Identifier: Apache-2.0
"""Finished drafter contexts should retain only their sliding window."""

import json
import subprocess
import sys
import textwrap

import mlx.core as mx
import pytest


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal allocator required")
@pytest.mark.parametrize("case", ["history", "batch"])
def test_finished_context_releases_discarded_history(case):
    # Isolate allocator accounting from other tests and retained model fixtures.
    code = textwrap.dedent("""
        import gc
        import json
        import sys
        from types import SimpleNamespace
        import mlx.core as mx
        from omlx.speculative.dflash.drafter import DFlashDrafter, _RowContext

        slots, hidden = 2047, 5120
        model = SimpleNamespace(
            config=SimpleNamespace(target_layer_ids=[0], sliding_window=slots + 1),
            layers=[SimpleNamespace(self_attn=SimpleNamespace(causal=False))],
        )
        drafter = DFlashDrafter(model, block_size=8, source_path="synthetic")
        mx.synchronize()
        gc.collect()
        mx.clear_cache()
        baseline = mx.get_active_memory()
        case = sys.argv[1]
        if case == "history":
            inputs = [
                mx.full((1, slots, hidden), 1, dtype=mx.bfloat16),
                mx.full((1, slots - 1, hidden), 2, dtype=mx.bfloat16),
            ]
            mx.eval(inputs)
            end = 2 * slots - 1
            row = _RowContext(history=inputs.copy(), history_len=end, history_end=end)
        else:
            inputs = [mx.full((8, slots, hidden), 3, dtype=mx.bfloat16)]
            mx.eval(inputs)
            end = slots
            row = _RowContext(end=end, fed=end)
            drafter._remember(row, inputs[0][3:4])
        drafter._rows[0] = row
        drafter._request_uids["finished"] = 0
        drafter._uid_requests[0] = "finished"
        tokens = list(range(end))
        drafter.release_request("finished", tokens)
        drafter.release([0])
        del inputs, row
        kept = drafter._kept["finished"].context
        # Wait for submitted copies, without realizing an unused snapshot.
        mx.synchronize()
        gc.collect()
        mx.clear_cache()
        retained = mx.get_active_memory() - baseline
        correct = (
            kept.shape == (1, slots, hidden)
            and (
                (mx.all(kept[:, :1] == 1).item() and mx.all(kept[:, 1:] == 2).item())
                if case == "history" else mx.all(kept == 3).item()
            )
        )
        restored = drafter.prepare_request("next", tokens, end)
        print(json.dumps({
            "retained": retained, "logical": kept.nbytes,
            "correct": correct, "restored": restored,
            "start": drafter._request_restored["next"][1],
        }))
        """)
    result = subprocess.run(
        [sys.executable, "-c", code, case],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    assert measured["correct"]
    assert measured["restored"] == 2047
    assert measured["start"] == (2046 if case == "history" else 0)
    # Permit allocator alignment, not discarded history or other batch rows.
    assert measured["retained"] <= measured["logical"] + 128 * 1024, measured
