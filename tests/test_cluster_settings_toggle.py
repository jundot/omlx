# SPDX-License-Identifier: Apache-2.0
"""Persistence contract for the distributed-inference exposure toggle."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from omlx.admin import routes


def test_distributed_opt_in_is_saved_for_the_next_restart():
    settings = MagicMock()
    settings.server = SimpleNamespace(distributed_inference_enabled=False)
    settings.validate.return_value = []
    settings.save.return_value = None
    original = routes._get_global_settings
    routes._get_global_settings = lambda: settings
    try:
        result = asyncio.run(
            routes.update_global_settings(
                request=routes.GlobalSettingsRequest(
                    distributed_inference_enabled=True
                ),
                is_admin=True,
            )
        )
    finally:
        routes._get_global_settings = original

    assert result["success"] is True
    assert "distributed_inference_enabled" not in result["runtime_applied"]
    assert settings.server.distributed_inference_enabled is True
    settings.save.assert_called_once()
