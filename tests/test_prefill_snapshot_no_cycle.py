"""Prefill boundary snapshots must not strand cache arrays in a cycle.

``_extract_prefill_snapshot_states`` evaluates the boundary leaves before
handing them off, and it used to gather them with a *recursive nested*
function. A recursive closure reaches itself through its own cell, so the
closure — and every container it captured — is reachable only through a
reference cycle and survives until the generational collector happens to run.
The captured leaf list names every array in the boundary state, and
``mx.array`` is a tiny object on the Python heap while backing GBs of Metal
memory, so nothing about the Python heap tells the collector to run.

Caches that grow in place (``KVCache`` reuses its preallocated buffer) hide
this: the stranded references alias the live chain and cost no extra bytes.
Caches that reallocate on growth do not — with TurboQuant KV every turn
stranded a full extra chain, measured at 0.74 GiB per turn on a 32k
Qwen3.8-27B conversation (usage 20.4 -> 24.5 GiB over six turns, flat once
collected).
"""

import gc
from types import SimpleNamespace

import mlx.core as mx

from omlx.scheduler import Scheduler


class _Cache:
    """Minimal sliceable cache, like mlx_lm's KVCache."""

    def __init__(self, seq_len: int = 4):
        self.keys = mx.zeros((1, 1, seq_len, 2))
        self.values = mx.zeros((1, 1, seq_len, 2))
        self.offset = seq_len

    @property
    def state(self):
        return self.keys, self.values

    @property
    def meta_state(self):
        return ()


def _stub():
    stub = SimpleNamespace(
        _stream=mx.default_stream(mx.default_device()),
        _PREFILL_SNAPSHOT_MARKER=Scheduler._PREFILL_SNAPSHOT_MARKER,
        model_name="",
    )
    stub._extract_cache_states = lambda caches: Scheduler._extract_cache_states(
        stub, caches
    )
    stub._extract_snapshot_cache_states = (
        lambda caches: Scheduler._extract_snapshot_cache_states(stub, caches)
    )
    return stub


def _closures_capturing_arrays() -> list[str]:
    """Qualnames of cyclic-garbage closures that captured cache arrays."""
    names = []
    for obj in gc.garbage:
        closure = getattr(obj, "__closure__", None) if callable(obj) else None
        if not closure:
            continue
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:  # cell still empty
                continue
            if isinstance(value, (list, tuple)) and any(
                isinstance(item, mx.array) for item in value
            ):
                names.append(getattr(obj, "__qualname__", repr(obj)))
    return names


def test_prefill_snapshot_extraction_strands_no_cache_arrays():
    stub = _stub()

    gc.collect()
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        gc.collect()
        del gc.garbage[:]

        result = Scheduler._extract_prefill_snapshot_states(stub, [_Cache()])
        assert result is not None, "extraction returned nothing"
        del result

        gc.collect()
        stranded = _closures_capturing_arrays()
    finally:
        gc.set_debug(0)
        del gc.garbage[:]
        gc.collect()

    assert not stranded, (
        "boundary-snapshot extraction left cache arrays reachable only through "
        f"a reference cycle, captured by: {sorted(set(stranded))}"
    )


def test_prefill_snapshot_extraction_still_evaluates_leaves():
    """The walk that replaced the recursion must still reach every leaf."""
    stub = _stub()
    result = Scheduler._extract_prefill_snapshot_states(stub, [_Cache(seq_len=3)])

    assert result is not None
    marker, extracted = result
    assert marker == Scheduler._PREFILL_SNAPSHOT_MARKER
    assert len(extracted) == 1
    keys, values = extracted[0]["state"]
    # mx.eval() on the leaves means reading them needs no further evaluation.
    assert keys.shape == (1, 1, 3, 2)
    assert values.shape == (1, 1, 3, 2)
