# SPDX-License-Identifier: Apache-2.0
"""`/v1/messages` accepts the same SpecPrefill overrides as `/v1/chat/completions`.

Forwarding to the engine is tested in
tests/integration/test_server_endpoints.py::TestAnthropicMessagesEndpoint.
"""

import pytest
from pydantic import ValidationError

from omlx.api.anthropic_models import MessagesRequest
from omlx.api.openai_models import ChatCompletionRequest

SPECPREFILL_FIELDS = ("specprefill", "specprefill_keep_pct", "specprefill_threshold")


def _messages(**extra):
    payload = {
        "model": "m",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(extra)
    return MessagesRequest(**payload)


def _chat(**extra):
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    payload.update(extra)
    return ChatCompletionRequest(**payload)


@pytest.mark.parametrize("field", SPECPREFILL_FIELDS)
def test_both_endpoints_expose_the_field(field):
    assert field in MessagesRequest.model_fields
    assert field in ChatCompletionRequest.model_fields


@pytest.mark.parametrize("field", SPECPREFILL_FIELDS)
def test_omitted_is_none_on_both(field):
    """None is what lets the handler leave the model/server setting alone."""
    assert getattr(_messages(), field) is None
    assert getattr(_chat(), field) is None


@pytest.mark.parametrize("value", [True, False])
def test_explicit_enable_and_disable_round_trip(value):
    assert _messages(specprefill=value).specprefill is value
    assert _chat(specprefill=value).specprefill is value


def test_tuning_fields_round_trip():
    req = _messages(specprefill_keep_pct=0.25, specprefill_threshold=4096)
    assert req.specprefill_keep_pct == 0.25
    assert req.specprefill_threshold == 4096


def test_rejects_wrong_types():
    with pytest.raises(ValidationError):
        _messages(specprefill="maybe")
    with pytest.raises(ValidationError):
        _messages(specprefill_threshold="lots")
