# SPDX-License-Identifier: Apache-2.0
"""Remote prefill settings come from OMLX_REMOTE_PREFILL_*, and vLLM is asked in its own API."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from omlx.remote_prefill.client import request_prefill
from omlx.remote_prefill.receiver import HandoffError, HandoffTimeoutError
from omlx.remote_prefill.settings import RemotePrefillSettings

_ENV = {
    "OMLX_REMOTE_PREFILL_URL": "http://prefill.invalid:8000/",
    "OMLX_REMOTE_PREFILL_MODEL": "org/model",
    "OMLX_REMOTE_PREFILL_FOR": "local-model",
    "OMLX_REMOTE_PREFILL_LINKS": "sparka, sparkb",
}


def test_settings_read_the_server_its_model_and_links_in_rank_order():
    settings = RemotePrefillSettings.from_env(
        {
            **_ENV,
            "OMLX_REMOTE_PREFILL_MIN_TOKENS": "8192",
            "OMLX_REMOTE_PREFILL_CHECKSUM": "off",
        }
    )
    assert settings.url == "http://prefill.invalid:8000"
    assert settings.links == ("sparka", "sparkb")
    assert settings.min_tokens == 8192 and settings.checksum is False
    assert settings.timeout_s == 120


def test_no_url_means_no_remote_prefill():
    assert RemotePrefillSettings.from_env({}) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"OMLX_REMOTE_PREFILL_URL": "prefill.invalid:8000"},
        {"OMLX_REMOTE_PREFILL_LINKS": ""},
        {"OMLX_REMOTE_PREFILL_LINKS": "bad link"},
        {"OMLX_REMOTE_PREFILL_FOR": ""},
        {"OMLX_REMOTE_PREFILL_MIN_TOKENS": "0"},
    ],
)
def test_incomplete_settings_are_refused(changes):
    with pytest.raises(ValueError):
        RemotePrefillSettings.from_env({**_ENV, **changes})


class _Response:
    def __init__(self):
        self.read_called = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        self.read_called = True
        return b"{}"


def _settings(**changes):
    return RemotePrefillSettings.from_env({**_ENV, **changes})


def test_vllm_is_asked_for_one_token_with_the_handoff_in_kv_transfer_params():
    seen = {}

    def opener(request, timeout):
        seen["url"], seen["timeout"] = request.full_url, timeout
        seen["headers"] = dict(request.header_items())
        seen["body"] = json.loads(request.data)
        return _Response()

    request_prefill(
        _settings(OMLX_REMOTE_PREFILL_API_KEY="secret-token"),
        [5, 6, 7],
        "ab" * 16,
        4,
        opener=opener,
    )
    assert seen["url"] == "http://prefill.invalid:8000/v1/completions"
    assert seen["body"] == {
        "model": "org/model",
        "prompt": [5, 6, 7],
        "max_tokens": 1,
        "temperature": 0.0,
        "kv_transfer_params": {"mcdma_handoff": {"id": "ab" * 16, "export_from": 4}},
    }
    assert seen["headers"]["Authorization"] == "Bearer secret-token"


def test_a_vllm_error_becomes_a_handoff_error_with_its_answer():
    def opener(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 400, "Bad Request", {}, io.BytesIO(b"prompt too long")
        )

    with pytest.raises(HandoffError, match="vLLM answered 400: prompt too long"):
        request_prefill(_settings(), [1], "cd" * 16, 0, opener=opener)


def test_an_unreachable_vllm_becomes_a_handoff_error():
    def opener(request, timeout):
        raise urllib.error.URLError("connection refused")

    with pytest.raises(HandoffError, match="did not answer"):
        request_prefill(_settings(), [1], "cd" * 16, 0, opener=opener)


def test_a_vllm_that_stays_silent_becomes_a_timeout():
    def opener(request, timeout):
        raise TimeoutError("timed out")

    with pytest.raises(HandoffTimeoutError, match="did not answer within 120 s"):
        request_prefill(_settings(), [1], "cd" * 16, 0, opener=opener)
