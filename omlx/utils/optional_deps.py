# SPDX-License-Identifier: Apache-2.0
"""Availability checks for lazily imported image dependencies."""

from importlib.util import find_spec

MLX_VLM_MISSING_MESSAGE = (
    "mlx-vlm image support is unavailable. "
    "Reinstall oMLX to restore its required mlx-vlm dependency."
)


def require_mlx_vlm_available() -> None:
    """Fail before loading models or processing image inputs."""
    if find_spec("mlx_vlm") is None:
        raise ImportError(MLX_VLM_MISSING_MESSAGE)
