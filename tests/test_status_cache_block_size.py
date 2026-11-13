"""cache_block_size in /v1/models/status entries.

The effective paged-cache block size is a load-time value (the scheduler
adjusts the configured size per model cache type), so status reports the live
value for loaded entries and None otherwise. Clients use it to pad stable
prefixes to block multiples (see #3348).
"""

from types import SimpleNamespace

from omlx.engine_pool import EnginePool


def _entry(engine):
    return SimpleNamespace(engine=engine)


def _core(block_size):
    return SimpleNamespace(
        scheduler=SimpleNamespace(
            config=SimpleNamespace(paged_cache_block_size=block_size)
        )
    )


def test_loaded_entry_reports_scheduler_block_size():
    # BatchedEngine / VLM engine shape: engine._engine (AsyncEngineCore)
    # wraps .engine (EngineCore), which owns the scheduler.
    engine = SimpleNamespace(_engine=SimpleNamespace(engine=_core(2048)))
    assert EnginePool._entry_cache_block_size(_entry(engine)) == 2048


def test_engine_exposing_scheduler_directly():
    assert EnginePool._entry_cache_block_size(_entry(_core(4096))) == 4096


def test_unloaded_entry_reports_none():
    assert EnginePool._entry_cache_block_size(_entry(None)) is None


def test_engine_without_scheduler_reports_none():
    engine = SimpleNamespace(_engine=SimpleNamespace(engine=SimpleNamespace()))
    assert EnginePool._entry_cache_block_size(_entry(engine)) is None
