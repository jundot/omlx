# SPDX-License-Identifier: Apache-2.0
"""`/v1/messages` accepts the same SpecPrefill overrides as `/v1/chat/completions`.

The two endpoints previously disagreed: a client could tune SpecPrefill per
request on the OpenAI endpoint but not on the Anthropic one, where the field
was silently dropped because `MessagesRequest` neither declared it nor allowed
extras. These tests pin the parity and the shared semantics — omitted leaves
the model/server setting in charge, true forces on, false forces off.
"""

import ast
import inspect

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


def _forwarding_sources(func, tree=None):
    """Statements in `func` that assign chat_kwargs["specprefill*"].

    Accepts a pre-parsed tree so a caller that also needs parent links can work
    against the same node objects; parsing twice yields two disjoint trees.
    """
    if tree is None:
        tree = ast.parse(inspect.getsource(func))
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Subscript)
                    and getattr(target.value, "id", None) == "chat_kwargs"
                    and isinstance(getattr(target, "slice", None), ast.Constant)
                    and str(target.slice.value).startswith("specprefill")):
                out.setdefault(target.slice.value, []).append(node)
    return out


def test_anthropic_handler_forwards_every_field():
    import omlx.server as srv

    forwarded = _forwarding_sources(srv.create_anthropic_message)
    for field in SPECPREFILL_FIELDS:
        assert field in forwarded, f"{field} is not forwarded by /v1/messages"


def test_anthropic_forwarding_matches_the_openai_endpoint():
    """Parity, not a second policy: neither endpoint may set a transport default.

    Every assignment must be reached only when the client supplied something,
    or via the model-settings fallback — never unconditionally.
    """
    import omlx.server as srv

    for func in (srv.create_anthropic_message, srv.create_chat_completion):
        tree = ast.parse(inspect.getsource(func))
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for field, assigns in _forwarding_sources(func, tree).items():
            for assign in assigns:
                guarded = False
                cur = parents.get(assign)
                while cur is not None:
                    if isinstance(cur, ast.If):
                        guarded = True
                        break
                    cur = parents.get(cur)
                assert guarded, (
                    f"{func.__name__} sets chat_kwargs[{field!r}] unconditionally; "
                    "that would be a transport default, not parity"
                )
