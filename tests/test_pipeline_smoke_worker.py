# SPDX-License-Identifier: Apache-2.0

import pytest

from omlx.cluster.pipeline_smoke_worker import _smoke_assignments


def test_smoke_assignments_cover_layers_in_reverse_pipeline_order():
    assignments = _smoke_assignments(4)

    assert [(item.rank, item.start_layer, item.end_layer) for item in assignments] == [
        (0, 6, 8),
        (1, 4, 6),
        (2, 2, 4),
        (3, 0, 2),
    ]
    assert all(item.layer_count == 2 for item in assignments)


@pytest.mark.parametrize("world_size", [0, 1, 17])
def test_smoke_assignments_reject_unsafe_world_sizes(world_size):
    with pytest.raises(ValueError, match="between 2 and 16"):
        _smoke_assignments(world_size)
