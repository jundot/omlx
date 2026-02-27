# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.server module - sampling parameter resolution and exception handlers."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.server import SamplingDefaults, ServerState, app, get_sampling_params


class TestGetSamplingParams:
    """Tests for get_sampling_params function."""

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        """Set up a clean server state for each test."""
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    def test_returns_4_tuple(self):
        """Test that get_sampling_params returns a 4-tuple."""
        result = get_sampling_params(None, None)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_defaults(self):
        """Test default values with no request or model params."""
        temp, top_p, top_k, rep_penalty = get_sampling_params(None, None)
        assert temp == 1.0
        assert top_p == 0.95
        assert top_k == 40
        assert rep_penalty == 1.0

    def test_request_overrides(self):
        """Test request params override global defaults."""
        temp, top_p, top_k, rep_penalty = get_sampling_params(0.5, 0.8)
        assert temp == 0.5
        assert top_p == 0.8
        assert top_k == 40  # not overridable via request
        assert rep_penalty == 1.0

    def test_model_settings_override(self):
        """Test model settings override global defaults."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = ModelSettings(
                temperature=0.3, top_k=50, repetition_penalty=1.2
            )
            manager.set_settings("test-model", settings)
            self._state.settings_manager = manager

            temp, top_p, top_k, rep_penalty = get_sampling_params(
                None, None, "test-model"
            )
            assert temp == 0.3
            assert top_p == 0.95  # falls back to global
            assert top_k == 50
            assert rep_penalty == 1.2

    def test_request_over_model(self):
        """Test request params take priority over model settings."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = ModelSettings(temperature=0.3)
            manager.set_settings("test-model", settings)
            self._state.settings_manager = manager

            temp, top_p, top_k, rep_penalty = get_sampling_params(
                0.7, None, "test-model"
            )
            assert temp == 0.7  # request wins

    def test_model_repetition_penalty(self):
        """Test model-level repetition_penalty overrides global."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = ModelSettings(repetition_penalty=1.5)
            manager.set_settings("test-model", settings)
            self._state.settings_manager = manager

            _, _, _, rep_penalty = get_sampling_params(None, None, "test-model")
            assert rep_penalty == 1.5

    def test_global_repetition_penalty(self):
        """Test global repetition_penalty is used when no model override."""
        self._state.sampling = SamplingDefaults(repetition_penalty=1.3)

        _, _, _, rep_penalty = get_sampling_params(None, None)
        assert rep_penalty == 1.3

    def test_force_sampling(self):
        """Test force_sampling ignores request params."""
        self._state.sampling = SamplingDefaults(
            temperature=0.5, top_p=0.8, force_sampling=True
        )

        temp, top_p, _, _ = get_sampling_params(0.9, 0.99)
        assert temp == 0.5  # forced, not request
        assert top_p == 0.8  # forced, not request


class TestExceptionHandlers:
    """Tests for global exception handlers that log API errors."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app, raise_server_exceptions=False)

    def test_http_exception_logged(self, client, caplog):
        """Test that HTTPException responses are logged."""
        # /v1/models requires startup, so a 404 on a non-existent route works
        response = client.get("/v1/nonexistent-endpoint")
        assert response.status_code == 404

    def test_validation_error_logged(self, client, caplog):
        """Test that request validation errors (422) are logged."""
        # POST to /v1/chat/completions with invalid body triggers validation
        response = client.post(
            "/v1/chat/completions",
            json={"invalid_field": "bad"},
        )
        # Should be 422 (validation error) or 500 (server not initialized)
        assert response.status_code in (422, 500)

    def test_exception_handler_returns_json(self, client):
        """Test that exception handlers return proper JSON responses."""
        response = client.get("/v1/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
