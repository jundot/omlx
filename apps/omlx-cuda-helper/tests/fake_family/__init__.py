# SPDX-License-Identifier: Apache-2.0
"""A family with numpy arithmetic for layers, to test stages without a GPU."""

FIRST_STAGE_MIN = 2
PACKS_WEIGHTS = True


def plan(model_dir, packed_weights=False):
    """(bytes outside the layers, bytes per layer) of the fake checkpoint."""
    return 1 << 20, [(1 << 19) if packed_weights else (1 << 20)] * 8
