# SPDX-License-Identifier: Apache-2.0
"""Regression tests for classifying a dead MTLCompilerService.

The compiler daemon is a system-level service outside oMLX's control. Once
its connection is gone the process cannot recover it (#1335), but the server
can still tell the caller what happened instead of returning an opaque 500.
These tests run without a GPU: they only exercise classification and error
mapping through the production code paths.
"""

import pytest

from omlx.exceptions import (
    MetalCompilerUnavailableError,
    RequestAbortedError,
    is_metal_compiler_unavailable,
)
from omlx.request import RequestOutput
from omlx.engine_core import (
    _raise_request_output_error,
    is_metal_compiler_unavailable_text,
)

COMPILER_ERROR = (
    "[metal::Device] Unable to load function "
    "sdpa_vector_2pass_1_bfloat16_t_128_128\n"
    "Unable to reach MTLCompilerService. The process is unavailable because "
    "the compiler is no longer active. Latest invalidation reason: "
    "Connection init failed at lookup with error 32 - Broken pipe"
)


class TestPatternMatching:
    def test_matches_real_mlx_error(self):
        assert is_metal_compiler_unavailable(RuntimeError(COMPILER_ERROR)) is True

    def test_matches_wrapped_in_cause_chain(self):
        inner = RuntimeError(COMPILER_ERROR)
        outer = RuntimeError("Prefill failed")
        outer.__cause__ = inner
        assert is_metal_compiler_unavailable(outer) is True

    def test_no_false_positive_on_other_kernel_errors(self):
        # A missing/ordinary kernel library is a different failure and must
        # not be mapped to "compiler service unavailable".
        for msg in (
            "Unable to load kernel some_kernel",
            "kIOGPUCommandBufferCallbackErrorOutOfMemory",
            "Broken pipe on socket",
            "",
        ):
            assert is_metal_compiler_unavailable(RuntimeError(msg)) is False

    def test_none_and_cyclic_chain(self):
        assert is_metal_compiler_unavailable(None) is False
        loop = RuntimeError("loop")
        loop.__cause__ = loop
        assert is_metal_compiler_unavailable(loop) is False

    def test_text_helper(self):
        assert is_metal_compiler_unavailable_text(COMPILER_ERROR) is True
        assert is_metal_compiler_unavailable_text(None) is False
        assert is_metal_compiler_unavailable_text("ordinary error") is False


class TestRequestOutputClassification:
    def _output(self, error, error_code=None):
        return RequestOutput(
            request_id="req-1335",
            error=error,
            error_code=error_code,
            finished=True,
        )

    def test_compiler_loss_raises_typed_error(self):
        with pytest.raises(MetalCompilerUnavailableError) as excinfo:
            _raise_request_output_error(self._output(COMPILER_ERROR))
        assert excinfo.value.request_id == "req-1335"
        assert "MTLCompilerService" in str(excinfo.value)

    def test_other_errors_still_raise_runtime_error(self):
        with pytest.raises(RuntimeError) as excinfo:
            _raise_request_output_error(self._output("some other prefill failure"))
        assert not isinstance(excinfo.value, MetalCompilerUnavailableError)

    def test_structured_error_codes_take_precedence(self):
        # An admin abort keeps its own 409 mapping even if the text happens
        # to mention the compiler; classification must not override it.
        with pytest.raises(RequestAbortedError):
            _raise_request_output_error(
                self._output(COMPILER_ERROR, error_code="model_unloading")
            )


class TestServerMapping:
    """The 503 handler and the SSE in-band payload must stay classifiable."""

    def _server(self):
        from omlx import server

        return server

    def test_503_handler_body_and_headers(self):
        import asyncio
        import json
        from unittest.mock import MagicMock

        server = self._server()
        request = MagicMock()
        request.method = "POST"
        request.url.path = "/v1/chat/completions"
        original = server._is_api_route
        server._is_api_route = lambda r: True
        try:
            response = asyncio.run(
                server.metal_compiler_unavailable_handler(
                    request,
                    MetalCompilerUnavailableError(COMPILER_ERROR, request_id="req-1"),
                )
            )
        finally:
            server._is_api_route = original

        assert response.status_code == 503
        assert response.headers["retry-after"] == "10"
        body = json.loads(response.body)
        assert body["error"]["code"] == "metal_compiler_unavailable"
        assert body["error"]["omlx_code"] == "metal_compiler_unavailable"
        assert "restart the oMLX server" in body["error"]["message"]

    def test_streaming_payload_keeps_the_code(self):
        from omlx import server

        payload = server._streaming_error_payload(
            MetalCompilerUnavailableError(COMPILER_ERROR, request_id="req-1"),
            "chat streaming",
        )
        assert payload["error"]["code"] == "metal_compiler_unavailable"
        assert payload["error"]["omlx_code"] == "metal_compiler_unavailable"
        # An unrelated error must still fall back to a plain server_error.
        plain = server._streaming_error_payload(
            RuntimeError("some other failure"), "chat streaming"
        )
        assert "code" not in plain["error"]
