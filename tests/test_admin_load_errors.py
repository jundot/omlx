# SPDX-License-Identifier: Apache-2.0
"""Status codes returned by POST /api/models/{model_id}/load.

Loading a model used to report every failure as 500, discarding the
actionable ``EnginePoolError`` subclasses that say whether the model cannot
fit, is busy, or disappeared.
"""

import asyncio

import pytest
from fastapi import HTTPException

import omlx.admin.routes as admin_routes
import omlx.server  # noqa: F401 — ensure server module is imported first
from omlx.exceptions import (
    EnginePoolError,
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
    ModelUnavailableError,
)


class _Pool:
    def __init__(self, error):
        self._error = error

    def get_entry(self, model_id):
        entry = type("E", (), {})()
        entry.engine = None
        entry.is_loading = False
        return entry

    async def get_engine(self, model_id):
        raise self._error


def _load(pool):
    originals = admin_routes._get_engine_pool
    admin_routes._get_engine_pool = lambda: pool
    try:
        return asyncio.run(admin_routes.load_model("m", is_admin=True))
    finally:
        admin_routes._get_engine_pool = originals


@pytest.mark.parametrize(
    "error, expected",
    [
        (ModelNotFoundError("m", ["a"]), 404),
        # "Cannot fit" is 507 on both load and inference (server.py maps
        # ModelTooLargeError the same way).
        (ModelTooLargeError("m", 100 << 30, 50 << 30, binding="final"), 507),
        (InsufficientMemoryError(100 << 30, 50 << 30, "need 100, have 50"), 507),
        (ModelUnavailableError("m"), 409),
        (ModelLoadingError("m"), 409),
        (ModelBusyError("m", "unload"), 409),
        # Anything else from the pool is still a server-side fault.
        (EnginePoolError("boom"), 500),
        (RuntimeError("VLM load failed"), 500),
    ],
)
def test_load_error_maps_to_status(error, expected):
    """Each EnginePoolError keeps its own status instead of a blanket 500."""
    with pytest.raises(HTTPException) as excinfo:
        _load(_Pool(error))
    assert excinfo.value.status_code == expected
    # The message carries the actionable detail; it must survive the mapping.
    assert str(excinfo.value.detail) == str(error)
