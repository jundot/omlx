"""The measured prefill working set of the stage holding the SSD lookup table.

Live measurement (2026-11-12, 128 GiB + 32 GiB Qwen4-Exp split): the stage
holding the PLE n-gram layer grew +7.25 GiB above its loaded footprint at a
1024-token chunk, +13.25 GiB at 2048, and completed a 26k-token prefill at
4096 inside a 16.25 GiB envelope. The chunk sets the size; the context
saturates above ~13k tokens. These tests pin the reservation to those numbers
and to the two live outcomes: the 0-4 split plans, the 0-8 split refuses.

Fake-driven: no Mac, no checkpoint, no endpoint.
"""

from __future__ import annotations

import inspect

import pytest

from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PlanningError,
    _partition_layers,
    plan_unequal_pipeline,
)

GiB = 1024**3

# The measured envelope: (chunk tokens, working set bytes).
MEASURED = ((1024, int(3.848 * GiB)), (2048, int(5.400 * GiB)), (4096, int(8.958 * GiB)))


def _layout(table_layers: tuple[int, ...] = (1,), **overrides) -> ModelLayout:
    """A Qwen4-Exp-shaped layout: 48 layers, the PLE table on layer 1.

    Held bytes reproduce the live split: layers 0-4 hold 8.55 GiB with the
    38.4 GiB table on SSD, layers 4-48 hold 80.2 GiB.
    """

    layer_bytes = [int(1.8 * GiB)] * 48
    layer_bytes[1] = int(38.4 * GiB) + int(1.8 * GiB)
    ssd = [0] * 48
    ssd[1] = int(38.4 * GiB)
    defaults = dict(
        source="/fake/qwen4-exp",
        fixed_weight_bytes=int(0.1 * GiB),
        layer_weight_bytes=tuple(layer_bytes),
        tensor_count=1000,
        supports_pipeline=True,
        kv_bytes_per_token_per_layer=512,
        prefill_transient_bytes_per_context_token=1400,
        tensor_parallel_ssd_layer_bytes=tuple(ssd),
        table_layer_ids=table_layers,
    )
    defaults.update(overrides)
    return ModelLayout(**defaults)


def _nodes(small_usable: float = 26.0, large_usable: float = 111.0) -> list[NodeBudget]:
    return [
        NodeBudget(
            node_id="large",
            rank=0,
            capacity_bytes=int(large_usable * GiB),
        ),
        NodeBudget(
            node_id="small",
            rank=1,
            capacity_bytes=int(small_usable * GiB),
        ),
    ]


class TestMeasuredEnvelope:
    def test_the_measured_points_are_the_reservation(self):
        model = _layout()
        for chunk, expected in MEASURED:
            assert model.table_prefill_transient_bytes(chunk) == expected

    def test_between_points_interpolates_linearly(self):
        model = _layout()
        value = model.table_prefill_transient_bytes(1536)
        assert MEASURED[0][1] < value < MEASURED[1][1]
        assert value == MEASURED[0][1] + (MEASURED[1][1] - MEASURED[0][1]) // 2

    def test_below_the_smallest_measured_chunk_scales_down(self):
        model = _layout()
        assert model.table_prefill_transient_bytes(512) == MEASURED[0][1] // 2

    def test_above_the_largest_measured_chunk_keeps_the_envelope(self):
        # Larger chunks are unmeasured; the reservation keeps the largest
        # measured value and the rank guard stays the request-time authority.
        model = _layout()
        assert model.table_prefill_transient_bytes(8192) == MEASURED[2][1]

    def test_no_table_layer_means_no_reservation(self):
        assert _layout(table_layers=()).table_prefill_transient_bytes(4096) == 0

    def test_non_positive_chunk_means_no_reservation(self):
        assert _layout().table_prefill_transient_bytes(0) == 0


class TestLayoutPricing:
    def test_ple_layer_ids_are_one_based_in_the_config(self):
        from omlx.cluster.planner import _qwen4_ple_layer_ids

        config = {"text_config": {"ple_layer_ids": [2]}}
        assert _qwen4_ple_layer_ids(config) == (1,)
        assert _qwen4_ple_layer_ids({"text_config": {"ple_layer_ids": [1, 3]}}) == (0, 2)

    def test_malformed_declarations_mean_no_table(self):
        from omlx.cluster.planner import _qwen4_ple_layer_ids

        assert _qwen4_ple_layer_ids({}) == ()
        assert _qwen4_ple_layer_ids({"text_config": {"ple_layer_ids": []}}) == ()
        assert _qwen4_ple_layer_ids({"text_config": {"ple_layer_ids": [0]}}) == ()
        assert _qwen4_ple_layer_ids({"text_config": {"ple_layer_ids": ["2"]}}) == ()
        assert _qwen4_ple_layer_ids({"text_config": {"ple_layer_ids": [True]}}) == ()

    def test_layout_without_a_table_keeps_its_hash_payload(self):
        payload = _layout(table_layers=()).to_dict()
        assert "table_layer_ids" not in payload
        assert ModelLayout.from_dict(payload).table_layer_ids == ()

    def test_table_layer_ids_round_trip(self):
        model = _layout(table_layers=(0, 1))
        restored = ModelLayout.from_dict(model.to_dict())
        assert restored.table_layer_ids == (0, 1)

    def test_table_layer_ids_must_name_existing_layers(self):
        with pytest.raises(ValueError):
            _layout(table_layers=(48,))
        with pytest.raises(ValueError):
            _layout(table_layers=(2, 1))


class TestTheLiveOutcomes:
    """The reservation must reproduce both measured splits."""

    def test_the_completed_0_4_split_plans_at_throughput(self):
        plan = plan_unequal_pipeline(
            _layout(),
            _nodes(),
            workload_profile="throughput",
            context_tokens=32768,
            engram_ssd_offload=True,
        )
        small = next(a for a in plan.assignments if a.node_id == "small")
        # The live run completed layers 0-4 at a 4096 chunk; the planner must
        # not hand the small Mac the 0-7/0-8 stage that ran out of memory.
        assert small.end_layer <= 5
        assert small.prefill_transient_bytes == MEASURED[2][1]
        assert small.table_prefill_transient_bytes == MEASURED[2][1]

    def test_the_pre_fix_oom_split_now_plans_under_the_valve(self):
        # Before the PLE prefill pool valve this 0-8 cut OOMed in a live run
        # and the old, ceiling-limited envelope refused it at plan time. The
        # valve removed the cause and the envelope was re-measured below the
        # ceiling (1024 3.85 / 2048 5.40 / 4096 8.96 GiB), so the cut fits. A split that still does not
        # fit weights + KV + the envelope refuses normally — that rule is
        # unchanged (test_a_split_that_does_not_fit_still_refuses below).
        model = _layout(pipeline_stage_starts=(8,))
        plan = plan_unequal_pipeline(
            model,
            _nodes(),
            workload_profile="throughput",
            context_tokens=32768,
            engram_ssd_offload=True,
        )
        small = next(a for a in plan.assignments if a.node_id == "small")
        assert small.table_prefill_transient_bytes == MEASURED[2][1]

    def test_a_split_that_does_not_fit_still_refuses(self):
        # The same geometry with a stage budget the envelope cannot fit: the
        # refusal law survives the re-measurement.
        model = _layout(pipeline_stage_starts=(8,))
        with pytest.raises(PlanningError):
            plan_unequal_pipeline(
                model,
                _nodes(small_usable=14.0, large_usable=200.0),
                workload_profile="throughput",
                context_tokens=32768,
                engram_ssd_offload=True,
            )

    def test_only_the_table_stage_pays_the_envelope(self):
        plan = plan_unequal_pipeline(
            _layout(),
            _nodes(),
            workload_profile="throughput",
            context_tokens=32768,
            engram_ssd_offload=True,
        )
        large = next(a for a in plan.assignments if a.node_id == "large")
        assert large.table_prefill_transient_bytes == 0
        assert large.prefill_transient_bytes < MEASURED[0][1]

    def test_a_resident_table_is_not_charged_the_streaming_envelope(self):
        # The envelope was measured with the table on SSD; a resident table
        # is not charged it (the generic price is withdrawn, so the plan
        # reserves nothing for it — its working set is unmeasured).
        plan = plan_unequal_pipeline(
            _layout(),
            _nodes(small_usable=60.0, large_usable=200.0),
            workload_profile="throughput",
            context_tokens=32768,
            engram_ssd_offload=False,
        )
        small = next(a for a in plan.assignments if a.node_id == "small")
        assert small.table_prefill_transient_bytes == 0

    def test_the_deprecated_generic_price_never_overrides_the_envelope(self):
        # The generic per-token price is withdrawn (#5033 B1/B2). A persisted
        # layout that still carries it must not change the table stage's
        # reservation: the measured envelope is the only plan charge.
        model = _layout(prefill_transient_bytes_per_context_token=100_000)
        plan = plan_unequal_pipeline(
            model,
            _nodes(small_usable=200.0, large_usable=400.0),
            workload_profile="throughput",
            context_tokens=1_048_576,
            engram_ssd_offload=True,
        )
        small = next(a for a in plan.assignments if a.node_id == "small")
        assert small.prefill_transient_bytes == MEASURED[2][1]
        assert small.table_prefill_transient_bytes == MEASURED[2][1]

    def test_partition_charges_the_table_stage_only(self):
        model = _layout()
        ranges = _partition_layers(
            model.tensor_parallel_layer_bytes,
            tuple(sorted(_nodes(), key=lambda n: n.rank, reverse=True)),
            fixed_weight_bytes=model.fixed_weight_bytes,
            node_overhead_bytes=1 * GiB,
            table_layer_ids=model.table_layer_ids,
            table_overhead_bytes=MEASURED[2][1],
        )
        # Reverse rank order: ranges[0] is rank 1 (the small Mac, early
        # layers), ranges[1] is rank 0. The small node holds layer 1; the
        # large node must not lose the envelope from its budget.
        small_range = ranges[0]
        assert small_range[0] <= 1 < small_range[1]
        large = _nodes()[0]
        large_held = sum(model.tensor_parallel_layer_bytes[ranges[1][0]:ranges[1][1]])
        assert (
            model.fixed_weight_bytes + large_held + 1 * GiB <= large.usable_bytes
        )


# The table envelope and routed-expert offload are two separate planner
# features: the offload kwargs arrive with the expert-offload PR. When this
# branch does not carry them yet, the interaction is skipped rather than
# silently untested — it runs as soon as both are in the same tree.
_NEEDS_OFFLOAD = (
    "moe_offload_layer_bytes" in inspect.signature(plan_unequal_pipeline).parameters
)


@pytest.mark.skipif(
    not _NEEDS_OFFLOAD,
    reason="needs the expert-offload planner arguments (the MoE offload PR)",
)
class TestExpertOffloadVersusTheTableLayer:
    """PLE-SSD + expert offload: the streamed share must fit the resident
    layer bytes. Found live on a two-Mac pipeline cell, where a
    stale expert-offload setting met the PLE table layer and the dataclass'
    bare ValueError surfaced as a 500 at the route."""

    @staticmethod
    def _small_layout(table_bytes: float = 18.0, layer_bytes: float = 2.0) -> ModelLayout:
        layers = [int(layer_bytes * GiB)] * 4
        layers[1] = int(table_bytes * GiB) + int(layer_bytes * GiB)
        ssd = [0] * 4
        ssd[1] = int(table_bytes * GiB)
        return ModelLayout(
            source="/fake/small-qwen4-exp",
            fixed_weight_bytes=int(0.1 * GiB),
            layer_weight_bytes=tuple(layers),
            tensor_count=100,
            supports_pipeline=True,
            kv_bytes_per_token_per_layer=128,
            prefill_transient_bytes_per_context_token=64,
            tensor_parallel_ssd_layer_bytes=tuple(ssd),
            table_layer_ids=(1,),
        )

    @staticmethod
    def _small_nodes() -> list[NodeBudget]:
        return [
            NodeBudget(node_id="large", rank=0, capacity_bytes=int(100 * GiB)),
            NodeBudget(node_id="small", rank=1, capacity_bytes=int(30 * GiB)),
        ]

    def test_offload_that_fits_the_table_layer_plans(self):
        model = self._small_layout()
        offload = [int(0.25 * GiB)] * 4  # small share fits every stage
        plan = plan_unequal_pipeline(
            model,
            self._small_nodes(),
            workload_profile="throughput",
            context_tokens=4096,
            engram_ssd_offload=True,
            moe_offload_layer_bytes=tuple(offload),
            moe_expert_offload_resident_fraction=0.5,
        )
        assert plan.assignments
        for assignment in plan.assignments:
            assert (
                assignment.moe_offload_weight_bytes
                <= assignment.layer_weight_bytes - assignment.ssd_weight_bytes
            )
