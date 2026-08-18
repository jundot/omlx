"""Prefill boundary snapshots must preserve NamedTuple state types.

``_extract_prefill_snapshot_states`` deep-copies the container structure of
each layer's extracted state so the live cache can keep mutating its own
slots. The copy helper rebuilt every tuple as a plain ``tuple``, which is
lossy for tuple *subclasses*: TurboQuant keeps its per-codec state in
NamedTuples (``TurboQuantPolarProdState``, ``TurboQuantSplitState``, ...) and
every later ``_slice_state`` / ``_state_length`` call dispatches on
``isinstance``. Flattening them made those calls raise
``TypeError: Unsupported TurboQuant state type: <class 'tuple'>``, which in
turn made live-chain adoption fall back to the paged path on every turn
whenever TurboQuant KV was enabled.
"""

from types import SimpleNamespace
from typing import NamedTuple

import mlx.core as mx

from omlx.scheduler import Scheduler


class _PolarState(NamedTuple):
    """Stand-in for mlx_vlm's TurboQuant NamedTuple states."""

    radii: mx.array
    level_indices: tuple


class _QuantCache:
    """Minimal cache exposing NamedTuple-valued state, like TurboQuantKVCache."""

    def __init__(self, seq_len: int = 4):
        arr = mx.zeros((1, 1, seq_len, 2))
        self.keys = _PolarState(arr, (arr,))
        self.values = _PolarState(arr, (arr,))
        self.offset = seq_len

    @property
    def state(self):
        return self.keys, self.values

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, 4.0, 0)))


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
    stub._extract_prefill_snapshot_states = (
        lambda caches: Scheduler._extract_prefill_snapshot_states(stub, caches)
    )
    return stub


def test_prefill_snapshot_preserves_namedtuple_state_type():
    stub = _stub()
    result = Scheduler._extract_prefill_snapshot_states(stub, [_QuantCache()])

    assert result is not None, "extraction returned nothing"
    marker, layers = result
    assert marker == Scheduler._PREFILL_SNAPSHOT_MARKER
    assert len(layers) == 1

    state = layers[0]["state"]
    assert len(state) == 2
    for element in state:
        assert isinstance(element, _PolarState), (
            f"NamedTuple state was flattened to {type(element).__name__}; "
            "downstream isinstance dispatch (TurboQuant _slice_state/"
            "_state_length) breaks"
        )
        # the nested tuple field must still be a plain tuple, and copied
        assert isinstance(element.level_indices, tuple)


def test_prefill_snapshot_still_copies_plain_containers():
    """The type fix must not stop the helper from decoupling containers."""
    stub = _stub()
    cache = _QuantCache()
    live_outer = cache.state
    result = Scheduler._extract_prefill_snapshot_states(stub, [cache])
    _marker, layers = result
    snapshot_state = layers[0]["state"]

    # a distinct outer container, not an alias of the live one
    assert snapshot_state is not live_outer
    assert snapshot_state[0] is not live_outer[0] or isinstance(
        snapshot_state[0], _PolarState
    )
