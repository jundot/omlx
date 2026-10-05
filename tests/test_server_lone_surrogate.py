# SPDX-License-Identifier: Apache-2.0
"""Unpaired UTF-16 surrogates in request text answer 400, not 500 (#4247)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.api.anthropic_models import MessagesRequest
from omlx.api.openai_models import ChatCompletionRequest, CompletionRequest
from omlx.exceptions import InvalidRequestError

LONE = '"hi \\ud83d"'
PAIRED = '"hi \\ud83d\\ude00"'


def _build_test_app():
    """Register the real 400 handler and the real check on a fresh app."""
    import omlx.server as srv

    app = FastAPI()
    app.add_exception_handler(InvalidRequestError, srv.invalid_request_error_handler)

    @app.post("/v1/chat/completions")
    def chat(request: ChatCompletionRequest):
        srv._reject_lone_surrogates(request)
        return {"ok": True}

    @app.post("/v1/completions")
    def completion(request: CompletionRequest):
        srv._reject_lone_surrogates(request)
        return {"ok": True}

    @app.post("/v1/messages")
    def messages(request: MessagesRequest):
        srv._reject_lone_surrogates(request)
        return {"ok": True}

    return app


def _post(client, path, text):
    if path == "/v1/completions":
        body = '{"model": "m", "prompt": %s}' % text
    elif path == "/v1/messages":
        body = '{"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": %s}]}' % text
    else:
        body = '{"model": "m", "messages": [{"role": "user", "content": %s}]}' % text
    return client.post(path, content=body, headers={"content-type": "application/json"})


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions", "/v1/messages"])
class TestLoneSurrogate:
    def test_lone_surrogate_is_400(self, path):
        with TestClient(_build_test_app()) as client:
            resp = _post(client, path, LONE)
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "surrogate" in err["message"]
        assert err["param"] in ("messages[0].content", "prompt")

    def test_paired_surrogate_passes(self, path):
        with TestClient(_build_test_app()) as client:
            resp = _post(client, path, PAIRED)
        assert resp.status_code == 200
