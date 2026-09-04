# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 must be splittable across Macs — and unchanged on one.

Spreading the shard files over two nodes does nothing by itself: the vendored
forward was ``for layer in self.layers``, with no notion of another machine.
These pin the three things mlx-lm requires, and that adding them did not alter
single-node serving, which runs the same class.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

# The vendored MiniMax implementation needs mlx-vlm; a runner without it
# should skip these, not error collecting them.
pytest.importorskip("mlx_vlm")
from mlx.utils import tree_flatten

from omlx.patches.minimax_m3_mlx_lm import apply_minimax_m3_mlx_lm_patch

CONFIG = {
    "model_type": "minimax_m3_vl",
    "text_config": {
        "num_hidden_layers": 4, "hidden_size": 64, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 32,
        "shared_intermediate_size": 32, "num_local_experts": 2,
        "num_experts_per_tok": 1, "n_shared_experts": 1, "vocab_size": 128,
        "rms_norm_eps": 1e-6, "rope_theta": 10000, "max_position_embeddings": 512,
    },
}

# Eight layers, and the layer mix a real checkpoint has: MiniMax's default
# sparse frequency is [0]*3 + [1]*(n-3), so 0-2 are dense-attention dense-MLP
# and 3-7 are sparse-index MoE. Split 2 ways, rank 1 holds 0-3 — its *last*
# layer is a sparse one, which is what made the send-dependency bug fire on
# every rank but rank 0.
MIXED_CONFIG = {
    "model_type": "minimax_m3_vl",
    "text_config": {**CONFIG["text_config"], "num_hidden_layers": 8},
}

# The same eight layers made uniform — every layer MoE and sparse-index — so
# "half the layers" is genuinely "half the bytes" and a ratio means something.
# Real MiniMax-M3 is 57 of 60 layers MoE, so this is the honest shape; the
# tiny default config is lopsided only because two experts are smaller than
# one dense MLP.
UNIFORM_CONFIG = {
    "model_type": "minimax_m3_vl",
    "text_config": {
        **CONFIG["text_config"],
        "num_hidden_layers": 8,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "mlp_layer_types": ["sparse"] * 8,
        "layer_types": ["minimax_m3_sparse"] * 8,
    },
}


class _Group:
    def __init__(self, rank: int, size: int) -> None:
        self._rank, self._size = rank, size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


@pytest.fixture(autouse=True)
def _no_stage_pin_leaks_between_tests():
    """A pin is a process global; one test's must never reach the next."""

    from omlx.patches.minimax_m3_mlx_lm.pipeline_patch import clear_assigned_stage

    clear_assigned_stage()
    yield
    clear_assigned_stage()


def _model(config: dict = CONFIG):
    from mlx_lm.utils import _get_classes

    apply_minimax_m3_mlx_lm_patch()
    model_cls, args_cls = _get_classes(config)
    return model_cls(args_cls.from_dict(config))


def _rank_of(config: dict, rank: int, world_size: int):
    """Build the model and take a stage exactly the way the loader does.

    ``mlx_lm.utils.sharded_load`` constructs the model and then calls
    ``model.model.pipeline(pipeline_group)`` — the top-level wrapper is never
    told anything. That asymmetry is the whole of finding 1: a wrapper that
    cached its own reference to the layer list kept the model the inner tree
    just dropped.
    """

    model = _model(config)
    model.model.pipeline(_Group(rank, world_size))
    return model


def _unique_bytes(params) -> int:
    """Bytes of distinct arrays, deduped by identity.

    Summing ``tree_flatten`` naively double-counts a model reachable by two
    paths, which is how a rank holding 1.00x the model looked like 1.07x
    instead of like a bug.
    """

    seen: set[int] = set()
    total = 0
    for _, array in tree_flatten(params):
        if not isinstance(array, mx.array) or id(array) in seen:
            continue
        seen.add(id(array))
        total += array.nbytes
    return total


def _stub_collectives(monkeypatch) -> None:
    """Let a single process run a rank's forward without a peer.

    Only the transport is faked; every line of the forward under test runs.
    """

    monkeypatch.setattr(mx.distributed, "send", lambda x, dst, **k: x)
    monkeypatch.setattr(mx.distributed, "recv_like", lambda x, src, **k: x)
    monkeypatch.setattr(mx.distributed, "all_gather", lambda x, **k: x)


def test_mlx_lm_now_considers_the_model_pipelinable():
    """The exact gate: hasattr(model, "model") and hasattr(model.model, "pipeline").

    Failing it produced "The model does not support pipelining but a
    pipeline_group was provided" after 61.7 GiB had already been staged.
    """

    model = _model()
    assert hasattr(model, "model")
    assert hasattr(model.model, "pipeline")


def test_each_rank_keeps_only_its_own_layers():
    """Blanked layers are never built, so a rank needs only its own shards."""

    rank0, rank1 = _model(), _model()
    rank0.model.pipeline(_Group(0, 2))
    rank1.model.pipeline(_Group(1, 2))

    # PipelineMixin numbers in reverse: rank 0 holds the *last* layers.
    assert rank0.model.start_idx == 2 and rank0.model.num_layers == 2
    assert rank1.model.start_idx == 0 and rank1.model.num_layers == 2
    assert sum(1 for layer in rank0.model.layers if layer is None) == 2
    assert len(rank1.model.layers) == 2


def test_the_two_stages_cover_every_layer_exactly_once():
    ranks = []
    for rank in range(2):
        model = _model()
        model.model.pipeline(_Group(rank, 2))
        start = model.model.start_idx
        ranks.append(set(range(start, start + model.model.num_layers)))
    assert ranks[0] | ranks[1] == set(range(4))
    assert not (ranks[0] & ranks[1]), "no layer may be computed twice"


def test_an_even_split_across_four_ranks_covers_every_layer():
    covered = set()
    for rank in range(4):
        model = _model()
        model.model.pipeline(_Group(rank, 4))
        start = model.model.start_idx
        covered |= set(range(start, start + model.model.num_layers))
    assert covered == set(range(4))


def test_a_split_that_does_not_divide_evenly_is_an_upstream_limitation():
    """Documents a constraint in mlx-lm's PipelineMixin, not in this patch.

    ``start_idx`` is computed from the *adjusted* per-rank count, so when
    ``layers % ranks != 0`` the first rank's range runs past the end: 4 layers
    across 3 ranks gives rank 0 ``start_idx=4``. The planner should keep stage
    counts to a divisor of the layer count until that is fixed upstream.
    """

    model = _model()
    model.model.pipeline(_Group(0, 3))
    assert model.model.start_idx >= len(model.model.layers), (
        "if this now passes, mlx-lm fixed uneven splits and the planner may "
        "stop restricting stage counts to divisors of the layer count"
    )


# --- Single-node serving must be untouched ---------------------------------


def test_one_node_runs_every_layer_and_produces_logits():
    """The same class serves single-node; pipelining must not disturb it."""

    model = _model()
    assert model.model.pipeline_size == 1
    assert model.model.start_idx == 0

    logits = model(mx.array([[1, 2, 3]]))
    mx.eval(logits)
    assert logits.shape[:2] == (1, 3)
    assert bool(mx.all(mx.isfinite(logits))), "un-pipelined forward must be sane"


def test_no_collective_is_attempted_on_one_node(monkeypatch):
    """A single node must never call send/recv — there is no peer to answer."""

    called = []
    monkeypatch.setattr(
        mx.distributed, "send",
        lambda *a, **k: called.append("send"),
    )
    monkeypatch.setattr(
        mx.distributed, "recv_like",
        lambda *a, **k: called.append("recv"),
    )
    model = _model()
    mx.eval(model(mx.array([[1, 2, 3]])))
    assert not called, f"single node attempted {called}"


def test_the_patch_is_idempotent():
    from omlx.patches.minimax_m3_mlx_lm.pipeline_patch import (
        apply_minimax_m3_pipeline_patch,
    )

    assert apply_minimax_m3_pipeline_patch()
    assert apply_minimax_m3_pipeline_patch()


# --- The plan must survive contact with the loader -------------------------


def test_the_planners_uneven_split_is_honoured_not_recomputed():
    """The failure that OOMed a MacBook.

    The plan gave rank 0 layers 46-60 (14 layers, 56 GiB). PipelineMixin
    recomputed an even 30/30 and the rank loaded 112 GiB against a 109 GiB
    limit. Pinning the range is what makes a Workstation reserve mean anything.
    """

    from omlx.patches.minimax_m3_mlx_lm.pipeline_patch import (
        clear_assigned_stage,
        set_assigned_stage,
    )

    model = _model()  # 4 layers
    set_assigned_stage(3, 4)
    try:
        model.model.pipeline(_Group(0, 2))
    finally:
        clear_assigned_stage()

    assert model.model.start_idx == 3
    assert model.model.num_layers == 1, "must hold the assigned range, not half"
    assert sum(1 for layer in model.model.layers if layer is None) == 3


def test_without_an_assignment_the_even_split_is_unchanged():
    """Outside oMLX's launcher nothing is pinned; behaviour must not change."""

    from omlx.patches.minimax_m3_mlx_lm.pipeline_patch import clear_assigned_stage

    clear_assigned_stage()
    model = _model()
    model.model.pipeline(_Group(0, 2))
    assert model.model.start_idx == 2 and model.model.num_layers == 2


def test_the_effective_stage_reports_what_will_load_not_what_was_planned():
    """What the memory guard must consult."""

    from omlx.patches.minimax_m3_mlx_lm.pipeline_patch import (
        clear_assigned_stage,
        effective_stage,
        set_assigned_stage,
    )

    clear_assigned_stage()
    # 60 layers, rank 0 of 2: PipelineMixin takes the last 30, not the last 14.
    assert effective_stage(60, 0, 2) == (30, 60)

    set_assigned_stage(46, 60)
    try:
        assert effective_stage(60, 0, 2) == (46, 60)
    finally:
        clear_assigned_stage()
