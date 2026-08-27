# SPDX-License-Identifier: Apache-2.0
"""Pure-policy contracts for prompt/decode continuous-batch budgets."""

import pytest

from omlx.continuous_batching import _positive_env_int, continuous_batch_budget


def _budget(rows, **kwargs):
    return continuous_batch_budget(
        configured_prefill_tokens=kwargs.pop("configured", 2048),
        active_decode_rows=rows,
        decode_row_budget=kwargs.pop("decode_budget", 32),
        max_prompt_tokens=kwargs.pop("max_prompt", 8192),
        mixed_prefill_quantum=kwargs.pop("quantum", 512),
        **kwargs,
    )


def test_idle_budget_preserves_configured_prefill_width_and_rows():
    budget = _budget(0, configured=4096)

    assert not budget.mixed
    assert budget.prefill_quantum == 4096
    assert budget.prompt_token_budget == 0
    assert budget.prompt_row_budget == 0


@pytest.mark.parametrize(
    ("rows", "expected"),
    ((1, 512), (2, 256), (4, 128)),
)
def test_b1_b2_b4_have_independent_prompt_and_decode_budgets(rows, expected):
    budget = _budget(rows, decode_budget=8)

    assert budget.prefill_quantum == expected
    assert budget.prompt_token_budget == expected
    assert budget.prompt_row_budget == 1
    assert budget.decode_row_budget == 8


def test_quantum_never_exceeds_configured_or_prompt_token_cap():
    assert _budget(1, configured=256).prefill_quantum == 256
    assert _budget(1, max_prompt=192).prefill_quantum == 192


def test_pressure_quantum_stays_on_kernel_grid():
    for rows in range(1, 17):
        assert _budget(rows).prefill_quantum % 64 == 0


@pytest.mark.parametrize("raw", ("", "not-an-int", "0", "-64"))
def test_malformed_or_nonpositive_env_quantum_falls_back(monkeypatch, raw):
    monkeypatch.setenv("OMLX_TEST_BATCH_QUANTUM", raw)
    assert _positive_env_int("OMLX_TEST_BATCH_QUANTUM", 512) == 512


def test_positive_env_quantum_is_accepted(monkeypatch):
    monkeypatch.setenv("OMLX_TEST_BATCH_QUANTUM", "640")
    assert _positive_env_int("OMLX_TEST_BATCH_QUANTUM", 512) == 640


@pytest.mark.parametrize(
    ("rows", "expected"),
    ((1, 1024), (2, 1024), (3, 512), (4, 512)),
)
def test_ds4_two_decode_rows_share_one_pressure_tier(rows, expected):
    budget = _budget(
        rows,
        quantum=1024,
        decode_rows_per_pressure_tier=2,
    )
    assert budget.active_decode_rows == rows
    assert budget.prefill_quantum == expected
