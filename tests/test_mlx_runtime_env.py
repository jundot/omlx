# SPDX-License-Identifier: Apache-2.0
"""Tests for early MLX Metal runtime policy."""

from omlx._mlx_runtime import configure_mlx_runtime_environment


def test_high_memory_policy_enables_validated_defaults():
    environ = {}

    applied = configure_mlx_runtime_environment(
        environ,
        memory_bytes=128 * 1024**3,
    )

    assert applied == {
        "MLX_METAL_FAST_SYNCH": "1",
        "MLX_MAX_MB_PER_BUFFER": "512",
        "MLX_MAX_OPS_PER_BUFFER": "100",
    }
    assert environ == applied


def test_low_memory_policy_only_enables_fast_synchronization():
    environ = {}

    applied = configure_mlx_runtime_environment(
        environ,
        memory_bytes=32 * 1024**3,
    )

    assert applied == {"MLX_METAL_FAST_SYNCH": "1"}


def test_operator_values_override_every_default():
    environ = {
        "MLX_METAL_FAST_SYNCH": "0",
        "MLX_MAX_MB_PER_BUFFER": "320",
        "MLX_MAX_OPS_PER_BUFFER": "50",
    }

    applied = configure_mlx_runtime_environment(
        environ,
        memory_bytes=128 * 1024**3,
    )

    assert applied == environ


def test_runtime_policy_can_be_disabled():
    environ = {"OMLX_MLX_RUNTIME_TUNING": "off"}

    assert configure_mlx_runtime_environment(
        environ,
        memory_bytes=128 * 1024**3,
    ) == {}
    assert environ == {"OMLX_MLX_RUNTIME_TUNING": "off"}
