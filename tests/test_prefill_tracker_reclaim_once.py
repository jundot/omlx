# SPDX-License-Identifier: Apache-2.0
"""The reclaim charge is priced ONCE, not accumulated without bound.

When a prefill chunk ends up holding LESS memory than before, the tracker
records the difference: MLX's buffer pool may need to allocate that space
again on the next chunk, so the guard prices that risk up front.

The field's own comment says it is priced **once** ("prices it once until a
positive measurement confirms reallocation"). The code summed instead. Each
release stacked on the previous one, and since the sum only clears when a
chunk grows, a short conversation that releases memory several times left the
charge armed for the next conversation.

Measured on GLM-5.3-Flash (M1 Ultra, 128GB): with the same 2048-token chunk
and the same context length, the predicted charge ran from 1.02GB (clean) to
23.99GB (poisoned). The computed term is worth 1.02GB; the other 22.97GB were
stacked releases. With the machine's ceiling at 124GB and the model holding
107GB, that refuses every long prompt.
"""

from omlx.prefill_transient_tracker import PrefillTransientTracker

GB = 1024**3
MB = 1024**2


def _tracker():
    return PrefillTransientTracker(model_id="test")


def test_a_single_release_is_priced_at_its_own_value():
    t = _tracker()
    t.record_reclaim(512 * MB)
    assert t.recent_reclaim_bytes == 512 * MB


def test_consecutive_releases_do_not_accumulate():
    """Twenty 512MB releases are not 10GB to reallocate at once."""
    t = _tracker()
    for _ in range(20):
        t.record_reclaim(512 * MB)
    assert t.recent_reclaim_bytes == 512 * MB


def test_the_largest_release_is_the_one_that_counts():
    t = _tracker()
    t.record_reclaim(200 * MB)
    t.record_reclaim(1500 * MB)
    t.record_reclaim(300 * MB)
    assert t.recent_reclaim_bytes == 1500 * MB


def test_a_growing_chunk_clears_the_charge():
    """Regression: the behaviour that already existed still holds."""
    t = _tracker()
    t.record_reclaim(800 * MB)
    t.clear_reclaim()
    assert t.recent_reclaim_bytes == 0


def test_a_positive_chunk_clears_the_charge_through_update():
    t = _tracker()
    t.record_reclaim(800 * MB)
    t.update(2048, 512 * MB)
    assert t.recent_reclaim_bytes == 0


def test_a_zero_or_negative_release_does_not_count():
    t = _tracker()
    t.record_reclaim(0)
    t.record_reclaim(-100 * MB)
    assert t.recent_reclaim_bytes == 0


def test_the_measured_incident():
    """The short conversation releases repeatedly; the long prompt came next.

    Before: 22.97GB of charge. After: the largest single event, 1.52GB —
    which is what the server log showed being reclaimed at once.
    """
    t = _tracker()
    for released in (526 * MB, 1520 * MB, 900 * MB, 480 * MB, 1200 * MB):
        t.record_reclaim(released)
    assert t.recent_reclaim_bytes == 1520 * MB
    assert t.recent_reclaim_bytes < 2 * GB
