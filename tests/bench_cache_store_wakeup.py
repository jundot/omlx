# SPDX-License-Identifier: Apache-2.0
"""CPU admission-stage benchmark; run with --baseline <main-revision>."""

import argparse
import ast
import asyncio
import concurrent.futures
import json
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_cache_store_wakeup as fixture

import omlx.engine_core as core
import omlx.scheduler as scheduler

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", required=True)
args = parser.parse_args()


def baseline_method(module, cls, name):
    source = subprocess.check_output(
        ["git", "show", f'{args.baseline}:{module.__name__.replace(".", "/")}.py'],
        text=True,
    )
    tree = ast.parse(source)
    node = next(
        method
        for definition in tree.body
        if isinstance(definition, ast.ClassDef) and definition.name == cls.__name__
        for method in definition.body
        if getattr(method, "name", None) == name
    )
    namespace = dict(vars(module))
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "<baseline>", "exec"),
        namespace,
    )
    return namespace[name]


old_loop = baseline_method(core, core.EngineCore, "_engine_loop")
old_submit = baseline_method(scheduler, scheduler.Scheduler, "_submit_abort_store")
changed_submit = scheduler.Scheduler._submit_abort_store


async def trial(arm, delay):
    engine = fixture.make_engine()
    engine.config.step_interval = 0.05
    engine.scheduler._submit_abort_store = types.MethodType(
        old_submit if arm == "main" else changed_submit, engine.scheduler
    )
    if arm == "main":
        engine._engine_loop = types.MethodType(old_loop, engine)
    future = concurrent.futures.Future()
    fixture.submit_store(engine.scheduler, future)
    admitted = asyncio.Event()
    completed_at = []
    admitted_at = []

    def step():
        engine.scheduler._drain_pending_async_removes()
        if engine.scheduler._store_cache_gate.has_capacity:
            admitted_at.append(time.perf_counter())
            engine._loop.call_soon_threadsafe(admitted.set)
        return fixture.SchedulerOutput(has_work=False)

    engine.scheduler.step = step
    with patch("omlx.scheduler._safe_sync_stream"):
        task = asyncio.create_task(engine._engine_loop())
        try:
            await asyncio.wait_for(engine._wake_event.wait_entered.wait(), timeout=1)
            await asyncio.sleep(delay)

            def complete():
                completed_at.append(time.perf_counter())
                future.set_result(None)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
                worker.submit(complete).result(timeout=1)
            await asyncio.wait_for(admitted.wait(), timeout=1)
            return {
                "arm": arm,
                "offset_ms": delay * 1000,
                "completion_to_admission_ms": (admitted_at[0] - completed_at[0]) * 1000,
            }
        finally:
            engine._running = False
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            engine._mlx_executor.shutdown(wait=True)


async def main():
    for arm in ("main", "changed", "changed", "main"):
        await trial(arm, 0.02)
    rows = []
    for delay in (0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.045):
        for arm in ("main", "changed", "changed", "main"):
            rows.append(await trial(arm, delay))
    result = {
        "base": args.baseline,
        "conditions": "4 discarded warmups; 28 ABBA trials; 50ms default poll; production EngineCore loop and Scheduler abort-store enqueue/drain; synthetic Future; mocked MLX sync; CPU stage only",
        "rows": rows,
        "summary": {
            arm: statistics.median(
                row["completion_to_admission_ms"] for row in rows if row["arm"] == arm
            )
            for arm in ("main", "changed")
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
