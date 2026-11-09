# SPDX-License-Identifier: Apache-2.0
"""Tests for ``_prepare_prompt_off_loop`` in ``omlx/server.py``.

Prompt preparation (chat template + tokenize) is pure CPU work. Running it on
the event loop stalls every other stream and ``/health``, so both batched
engines hand it to a worker thread.
"""

import asyncio
import threading
import time

import pytest

from omlx.engine.batched import BatchedEngine
from omlx.server import _prepare_prompt_off_loop


def _slow_prepare(engine, sink):
    sink["ident"] = threading.get_ident()
    time.sleep(0.2)
    return "prepared"


@pytest.mark.asyncio
async def test_text_engine_prepare_runs_off_loop():
    """The text engine prepares on a worker thread and keeps the loop alive."""
    engine = BatchedEngine(model_name="test-model")
    sink = {}
    gaps = []

    async def heartbeat():
        last = time.perf_counter()
        for _ in range(10):
            await asyncio.sleep(0.02)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    task = asyncio.create_task(heartbeat())
    result = await _prepare_prompt_off_loop(engine, _slow_prepare, sink)
    await task

    assert result == "prepared"
    assert sink["ident"] != threading.get_ident()
    assert max(gaps) < 0.1, gaps


@pytest.mark.asyncio
async def test_subclass_engine_stays_on_loop():
    """Only the exact batched engine types are off-loaded.

    ``DistributedBatchedEngine`` subclasses ``BatchedEngine`` but does not run
    ``prepare_chat_prompt``, so it must keep the synchronous path.
    """

    class _SubclassEngine(BatchedEngine):
        pass

    engine = _SubclassEngine(model_name="test-model")
    sink = {}

    result = await _prepare_prompt_off_loop(engine, _slow_prepare, sink)

    assert result == "prepared"
    assert sink["ident"] == threading.get_ident()
