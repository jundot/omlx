# SPDX-License-Identifier: Apache-2.0
"""Tests for oQ admin routes: option forwarding."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import omlx.admin.routes as admin_routes
import omlx.server  # noqa: F401 — ensure server module is imported first


def _recording_manager():
    """A stand-in manager that captures the keyword arguments it receives."""
    calls = []

    async def _start(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {"output_name": kwargs["oq_level"]})

    return SimpleNamespace(start_quantization=_start), calls


class TestOqStartRoute:
    def test_forwards_preserve_ngram_table(self):
        manager, calls = _recording_manager()
        request = admin_routes.OQStartRequest(
            model_path="/models/Qwen3.8-Flash-Next",
            oq_level=4,
            preserve_ngram_table=True,
        )

        with patch.object(admin_routes, "_oq_manager", manager), patch.object(
            admin_routes, "_paroquant_compat_for_model", lambda _info: (False, "")
        ):
            result = asyncio.run(
                admin_routes.start_oq_quantization(request, is_admin=True)
            )

        assert result["success"] is True
        assert calls[0]["preserve_ngram_table"] is True

    def test_defaults_to_quantizing_the_ngram_table(self):
        request = admin_routes.OQStartRequest(model_path="/models/x", oq_level=4)

        assert request.preserve_ngram_table is False


class TestOqEstimateRoute:
    def test_forwards_preserve_ngram_table_to_the_estimator(self):
        seen = []

        def _fake_estimate(*args, **kwargs):
            seen.append(args)
            return {"effective_bpw": 4.7}

        with patch("omlx.oq.estimate_bpw_and_size", _fake_estimate):
            result = asyncio.run(
                admin_routes.estimate_oq(
                    model_path="/models/Qwen3.8-Flash-Next",
                    oq_level=4,
                    preserve_ngram_table=True,
                    is_admin=True,
                )
            )

        assert result == {"effective_bpw": 4.7}
        assert seen == [("/models/Qwen3.8-Flash-Next", 4, 64, False, True)]

    def test_estimate_defaults_to_packed_table(self):
        seen = []

        def _fake_estimate(*args, **kwargs):
            seen.append(args)
            return {"effective_bpw": 4.6}

        with patch("omlx.oq.estimate_bpw_and_size", _fake_estimate):
            asyncio.run(
                admin_routes.estimate_oq(
                    model_path="/models/Qwen3.8-Flash-Next",
                    oq_level=4,
                    is_admin=True,
                )
            )

        assert seen == [("/models/Qwen3.8-Flash-Next", 4, 64, False, False)]
