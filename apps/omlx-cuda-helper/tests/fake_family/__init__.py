# SPDX-License-Identifier: Apache-2.0
"""A family with numpy arithmetic for layers, to test stages without a GPU."""

FIRST_STAGE_MIN = 2


def plan(model_dir):
    """(bytes outside the layers, bytes per layer) of the fake checkpoint."""
    return 1 << 20, [1 << 20] * 8
