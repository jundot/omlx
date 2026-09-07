# SPDX-License-Identifier: Apache-2.0
"""Phase 0.3 repro: does a mid-generation rank failure crash or wedge the
lockstep collective, and does the runtime marker/heartbeat notice?

Not a production code path. An offline instrumentation worker driven by
``run_local_generation_wedge_smoke`` (collective.py) to produce ground
truth for the design doc's §A1 "wedge vs. crash" question ahead of the
2.1 fix.

Every step is a single one-hot "abort vote" ``all_sum`` every rank always
joins (mirrors §A2/1.1's "always cast a vote, even on failure" pattern),
followed by a "work" collective only when nobody voted to abort. Two
failure modes on the designated fatal rank, at a designated step:

``caught`` -- the step raises inside a try/except; the handler still
casts this rank's abort vote before returning, so every rank sees the
vote on the very next collective and stops together. Expect: a clean,
symmetric, coordinated stop.

``killed`` -- the generation thread is abandoned mid-step: it returns
without joining even the abort-vote collective. The rank's *process*
stays alive afterward (it does not exit -- see the module docstring in
collective.py for why) and its heartbeat thread, independent of the
now-dead generation thread, keeps refreshing the runtime marker file.
Expect: the peer blocks forever in that same collective call, with
nothing watching it -- and the marker keeps looking healthy throughout.
"""

from __future__ import annotations

import json
import os
import threading
import time


def _fatal_rank() -> int:
    return int(os.environ.get("OMLX_WEDGE_FATAL_RANK", "1"))


def _fatal_step() -> int:
    return int(os.environ.get("OMLX_WEDGE_FATAL_STEP", "2"))


def _steps() -> int:
    return int(os.environ.get("OMLX_WEDGE_STEPS", "5"))


def _mode() -> str:
    return os.environ.get("OMLX_WEDGE_MODE", "caught")


def main() -> int:
    from .inference_worker import RuntimeMarker

    import mlx.core as mx

    group = mx.distributed.init(backend="ring", strict=True)
    rank = group.rank()
    size = group.size()

    marker = RuntimeMarker(
        state_dir=os.environ["OMLX_WEDGE_STATE_DIR"],
        deployment_id=os.environ.get("OMLX_WEDGE_DEPLOYMENT_ID", "wedge-repro"),
        rank=rank,
        world_size=size,
        model="wedge-repro",
        backend="ring",
        plan_hash="0" * 8,
    )
    marker.update("ready")
    # Independent of the generation loop below by design -- this is the
    # exact mechanism §A1 flags: the heartbeat cannot tell "generation is
    # progressing" from "generation silently died", because it never asks.
    marker.start_heartbeat(interval=0.15)

    mode = _mode()
    fatal_rank = _fatal_rank()
    fatal_step = _fatal_step()
    steps = _steps()
    is_fatal_rank = rank == fatal_rank
    outcome: dict = {"type": "wedge_result", "rank": rank, "mode": mode}
    abandoned = False

    def run_steps() -> None:
        nonlocal abandoned
        for step in range(steps):
            if is_fatal_rank and step == fatal_step and mode == "killed":
                marker.update("generating", step=step, wedged=True)
                outcome["abandoned_at_step"] = step
                abandoned = True
                return
            abort_votes = [0] * size
            if is_fatal_rank and step == fatal_step and mode == "caught":
                abort_votes[rank] = 1
            summed = mx.distributed.all_sum(mx.array(abort_votes))
            mx.eval(summed)
            if any(int(v) for v in summed.tolist()):
                marker.update("failed" if is_fatal_rank else "aborted", step=step)
                outcome["aborted_at_step"] = step
                return
            mx.eval(mx.distributed.all_sum(mx.array(step + 1)))
            marker.update("generating", step=step)
        outcome["completed"] = True

    thread = threading.Thread(target=run_steps, name="wedge-generation", daemon=True)
    thread.start()
    thread.join()

    if abandoned:
        # A killed generation thread doesn't take the process down with it.
        # Exiting here would tear down the TCP ring connection and let the
        # peer fail fast on a socket error instead of truly wedging -- the
        # exact "crash vs. wedge" distinction §A1/C1 need settled. Stay up
        # and keep heartbeating, same as a real stuck-but-alive rank would.
        print(json.dumps(outcome, sort_keys=True), flush=True)
        while True:
            time.sleep(3600)

    marker.stop_heartbeat()
    print(json.dumps(outcome, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
