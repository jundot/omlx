# SPDX-License-Identifier: Apache-2.0

import pytest

from omlx.reasoning_effort import (
    apply_chat_template_with_reasoning_effort_fallback,
)

MESSAGES = [{"role": "user", "content": "Hello"}]


class EnumTemplate:
    def __init__(self, accepted, default):
        self.accepted = set(accepted)
        self.default = default
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        value = kwargs.get("reasoning_effort", self.default)
        self.calls.append(value)
        if value not in self.accepted:
            raise ValueError(f"unsupported effort: {value}")
        return f"effort={value}"


class InklingTemplate:
    EFFORTS = {
        "none": 0.0,
        "minimal": 0.1,
        "low": 0.2,
        "medium": 0.7,
        "high": 0.9,
        "max": 0.99,
    }

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        value = kwargs.get("reasoning_effort", 0.9)
        self.calls.append(value)
        if isinstance(value, str):
            if value not in self.EFFORTS:
                raise ValueError(f"unsupported effort: {value}")
            value = self.EFFORTS[value]
        value = float(value)
        if value < 0.0 or value > 0.99:
            raise ValueError(f"effort out of range: {value}")
        return f"effort={value}"


class HarmonyTemplate:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        value = kwargs.get("reasoning_effort", "medium")
        self.calls.append(value)
        return f"Reasoning: {value}"


def render(target, value, *, is_harmony=False):
    return apply_chat_template_with_reasoning_effort_fallback(
        target,
        MESSAGES,
        {"tokenize": False, "reasoning_effort": value},
        is_harmony=is_harmony,
    )


def test_native_value_renders_once():
    target = EnumTemplate({"low", "medium", "xhigh"}, "xhigh")

    assert render(target, "medium") == "effort=medium"
    assert target.calls == ["medium"]


def test_template_that_ignores_reasoning_effort_renders_once():
    class IgnoringTemplate:
        def __init__(self):
            self.calls = 0

        def apply_chat_template(self, messages, **kwargs):
            self.calls += 1
            return "unchanged"

    target = IgnoringTemplate()

    assert render(target, "maximum") == "unchanged"
    assert target.calls == 1


@pytest.mark.parametrize(
    "value,expected,calls",
    [
        ("high", "xhigh", ["high", "xhigh"]),
        ("maximum", "xhigh", ["maximum", "max", "xhigh"]),
        (" MINIMAL ", "low", ["minimal", "low"]),
    ],
)
def test_qwen_aliases_and_native_default(value, expected, calls):
    target = EnumTemplate({"low", "medium", "xhigh"}, "xhigh")

    assert render(target, value) == f"effort={expected}"
    assert target.calls == calls


def test_unknown_value_removes_the_key_instead_of_passing_none():
    target = EnumTemplate({"low", "medium", "xhigh"}, "xhigh")

    assert render(target, "bogus") == "effort=xhigh"
    assert target.calls == ["bogus", "xhigh"]


def test_original_error_is_preserved_when_default_also_fails():
    class AlwaysFail:
        def apply_chat_template(self, messages, **kwargs):
            value = kwargs.get("reasoning_effort", "missing")
            raise RuntimeError(f"failed:{value}")

    with pytest.raises(RuntimeError, match="failed:bogus"):
        render(AlwaysFail(), "bogus")


@pytest.mark.parametrize(
    "value,expected,calls",
    [
        (0.5, 0.5, [0.5]),
        ("0.5", 0.5, ["0.5", 0.5]),
        ("xhigh", 0.99, ["xhigh", "max"]),
        ("maximum", 0.99, ["maximum", "max"]),
        (1.0, 0.9, [1.0, 0.9]),
    ],
)
def test_inkling_numbers_aliases_and_default(value, expected, calls):
    target = InklingTemplate()

    assert render(target, value) == f"effort={expected}"
    assert target.calls == calls


@pytest.mark.parametrize(
    "value,expected",
    [
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "high"),
        ("maximum", "high"),
        ("off", "low"),
    ],
)
def test_harmony_normalizes_to_protocol_levels_in_one_render(value, expected):
    target = HarmonyTemplate()

    assert render(target, value, is_harmony=True) == f"Reasoning: {expected}"
    assert target.calls == [expected]


@pytest.mark.parametrize("value", ["bogus", 0.9])
def test_harmony_unknown_values_use_native_default(value):
    target = HarmonyTemplate()

    assert render(target, value, is_harmony=True) == "Reasoning: medium"
    assert target.calls == ["medium"]


def test_input_kwargs_are_not_mutated():
    target = EnumTemplate({"low", "medium", "xhigh"}, "xhigh")
    kwargs = {"tokenize": False, "reasoning_effort": "high"}

    apply_chat_template_with_reasoning_effort_fallback(target, MESSAGES, kwargs)

    assert kwargs == {"tokenize": False, "reasoning_effort": "high"}
