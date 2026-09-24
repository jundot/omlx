"""Pure memory admission checks for batched dense text prefill."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from omlx.prefill.geometry import PrefillExpertGeometry
from omlx.prefill.memory import (
    PrefillMemoryContext,
    PrefillTransition,
    estimate_batched_prefill_memory,
    estimate_prefill_transition_memory,
)
from omlx.prefill.models import model_geometry_from_args


@pytest.fixture
def model_args():
    return SimpleNamespace(
        model_type="llama",
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_size=512,
        intermediate_size=1536,
        vocab_size=4096,
    )


@pytest.fixture
def geometry(model_args):
    return model_geometry_from_args(model_args, dtype_size=2)


def estimate(geometry, **overrides):
    arguments = {
        "batch_size": 2,
        "query_tokens": 128,
        "max_prompt_tokens": 2048,
    }
    arguments.update(overrides)
    return estimate_batched_prefill_memory(geometry, **arguments)


def test_context_prices_incremental_allocations_against_one_usage_snapshot(geometry):
    context = PrefillMemoryContext(geometry, 1_000_000, 2_000_000, 10_000, 1, 1024)
    cost = context.estimate(2, 128, 2048)
    expected = estimate(
        geometry,
        current_cache_bytes=10_000,
        decode_batch_size=1,
        decode_max_tokens=1024,
    )
    assert cost == expected
    assert context.headroom_bytes == 1_000_000
    assert cost.peak_bytes(context.current_usage_bytes) == (
        context.current_usage_bytes + expected.additional_peak_bytes
    )


def test_context_does_not_invent_headroom_or_a_missing_safety_ceiling(geometry):
    assert PrefillMemoryContext(geometry, 200, 100).headroom_bytes == 0
    assert PrefillMemoryContext(geometry, 200, 0).estimate(2, 1, 100) is None
    assert PrefillMemoryContext(None, 200, 1000).estimate(2, 1, 100) is None


def transition_cost(geometry, **overrides):
    arguments = {
        "transition": PrefillTransition.HANDOFF,
        "physical_rows": 4,
        "owned_rows": 4,
        "cache_tokens": 128,
        "current_cache_bytes": 4 * 256 * 2048,
        "decode_batch_size": 2,
        "decode_max_tokens": 1024,
    }
    arguments.update(overrides)
    return estimate_prefill_transition_memory(geometry, **arguments)


def test_handoff_does_not_reserve_committed_decode_rows_twice(geometry):
    before = transition_cost(geometry)
    after = transition_cost(geometry, owned_rows=2, decode_batch_size=4)
    assert after.cache_transition_bytes == (
        before.cache_transition_bytes - 2 * 256 * 2048
    )
    assert after.additional_peak_bytes < before.additional_peak_bytes
    assert before.new_kv_bytes == after.new_kv_bytes == 0


@pytest.mark.parametrize(
    "transition", [PrefillTransition.COMPACT, PrefillTransition.DEMOTE]
)
def test_non_decode_transitions_reserve_only_surviving_cache_copies(
    geometry, transition
):
    cost = transition_cost(geometry, transition=transition, owned_rows=2)
    assert cost.cache_transition_bytes == 2 * 256 * 2048
    assert cost.new_kv_bytes == 0
    assert cost == transition_cost(
        geometry,
        transition=transition,
        owned_rows=2,
        decode_batch_size=32,
        decode_max_tokens=16384,
    )


def test_transition_context_prices_live_cache_not_future_prompt_growth(geometry):
    context = PrefillMemoryContext(geometry, 1_000_000, 100_000_000, 32768, 2, 1024)
    cost = context.estimate_transition(
        PrefillTransition.COMPACT, physical_rows=4, owned_rows=2, cache_tokens=16
    )
    assert cost.cache_transition_bytes == 16384
    assert cost.new_kv_bytes == 0
    assert cost.peak_bytes(context.current_usage_bytes) == (
        context.current_usage_bytes + cost.additional_peak_bytes
    )
    assert (
        replace(context, limit_bytes=0).estimate_transition(
            PrefillTransition.COMPACT, physical_rows=4, owned_rows=2, cache_tokens=16
        )
        is None
    )


def test_handoff_keeps_padding_and_retained_capacity_allowances(geometry):
    kv_bytes_per_token = (
        2
        * geometry.num_layers
        * geometry.num_kv_heads
        * geometry.head_dim
        * geometry.cache_dtype_size
    )
    retained_tokens = 16384
    row_bytes = retained_tokens * kv_bytes_per_token
    cost = transition_cost(
        geometry, current_cache_bytes=4 * row_bytes, owned_rows=2, decode_batch_size=32
    )
    assert cost.cache_transition_bytes >= 2 * row_bytes + 2 * 34 * row_bytes


@pytest.mark.parametrize(
    "overrides",
    [
        {"transition": "handoff"},
        {"physical_rows": 0},
        {"physical_rows": True},
        {"owned_rows": 0},
        {"owned_rows": 5},
        {"owned_rows": 1.5},
        {"cache_tokens": -1},
        {"cache_tokens": False},
        {"current_cache_bytes": -1},
        {"current_cache_bytes": 0},
        {"decode_batch_size": 0},
        {"decode_max_tokens": 0},
    ],
)
def test_invalid_transition_observations_fail_closed(geometry, overrides):
    assert transition_cost(geometry, **overrides) is None


def test_transition_requires_verified_geometry():
    assert transition_cost(None) is None


def test_model_args_support_attributes_and_mappings(model_args):
    assert model_geometry_from_args(model_args) == model_geometry_from_args(
        vars(model_args)
    )
    geometry = model_geometry_from_args(model_args)
    assert geometry.head_dim == 64
    assert geometry.compute_dtype_size == 4
    assert geometry.cache_dtype_size == 4


def test_llama_defaults_missing_kv_heads_to_attention_heads(model_args):
    model_args.num_key_value_heads = None
    geometry = model_geometry_from_args(model_args)
    assert geometry.num_kv_heads == geometry.num_attention_heads


def test_qwen3_explicit_head_dimension_can_differ_from_hidden_width(model_args):
    model_args.model_type = "qwen3"
    model_args.head_dim = 128
    geometry = model_geometry_from_args(model_args)
    assert geometry.head_dim == 128
    assert geometry.hidden_size == 512


def test_qwen2_does_not_price_an_ignored_head_dimension(model_args):
    model_args.model_type = "qwen2"
    model_args.head_dim = 32

    assert model_geometry_from_args(model_args) is None


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen2_moe", "unknown", None])
def test_unknown_architectures_are_not_priced_as_dense(model_args, model_type):
    model_args.model_type = model_type
    assert model_geometry_from_args(model_args) is None


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("num_hidden_layers", 0),
        ("num_attention_heads", False),
        ("num_key_value_heads", 3),
        ("num_key_value_heads", -1),
        ("hidden_size", 513),
        ("intermediate_size", None),
        ("vocab_size", "4096"),
        ("head_dim", 0),
        ("head_dim", 64.0),
    ],
)
def test_invalid_geometry_disables_batching(model_args, field_name, invalid_value):
    setattr(model_args, field_name, invalid_value)
    assert model_geometry_from_args(model_args) is None


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
def test_qwen_missing_required_dimensions_is_unsupported(model_args, model_type):
    model_args.model_type = model_type
    model_args.num_key_value_heads = None
    assert model_geometry_from_args(model_args) is None


def test_qwen3_does_not_infer_an_unspecified_head_dimension(model_args):
    model_args.model_type = "qwen3"
    assert model_geometry_from_args(model_args) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"dtype_size": 1},
        {"dtype_size": True},
        {"cache_dtype_size": 0.5},
        {"cache_step": 0},
    ],
)
def test_unknown_dtype_or_allocator_geometry_disables_batching(model_args, overrides):
    assert model_geometry_from_args(model_args, **overrides) is None


def test_batch_width_prices_every_row(geometry):
    single = estimate(geometry, batch_size=1)
    batch = estimate(geometry, batch_size=4)
    assert batch.new_kv_bytes == 4 * single.new_kv_bytes
    assert batch.temporary_workspace_bytes == 4 * single.temporary_workspace_bytes
    assert batch.cache_transition_bytes == 4 * single.cache_transition_bytes


def test_full_prompt_growth_is_reserved_before_the_first_chunk(geometry):
    first_chunk_only = estimate(geometry, max_prompt_tokens=128)
    full_prompt = estimate(geometry, max_prompt_tokens=8192)
    assert full_prompt.new_kv_bytes > first_chunk_only.new_kv_bytes
    assert (
        full_prompt.temporary_workspace_bytes
        > first_chunk_only.temporary_workspace_bytes
    )
    assert full_prompt.cache_transition_bytes > first_chunk_only.cache_transition_bytes


def test_current_footprint_and_live_cache_are_counted_once(geometry):
    cold_cost = estimate(geometry)
    current_cache = cold_cost.new_kv_bytes // 2
    active_cost = estimate(geometry, current_cache_bytes=current_cache)
    model_and_other_allocations = 10 * 1024**3
    assert active_cost.new_kv_bytes == cold_cost.new_kv_bytes - current_cache
    assert active_cost.peak_bytes(model_and_other_allocations + current_cache) == (
        cold_cost.peak_bytes(model_and_other_allocations)
    )


def test_cache_capacity_is_not_assumed_to_align_to_allocator_step(geometry):
    cost = estimate(
        geometry,
        batch_size=2,
        query_tokens=300,
        max_prompt_tokens=500,
    )
    kv_bytes_per_token = (
        2
        * geometry.num_layers
        * geometry.num_kv_heads
        * geometry.head_dim
        * geometry.cache_dtype_size
    )
    capacity_after_chunks_of_200_then_300 = 200 + 512
    actual_cache_bytes = 2 * capacity_after_chunks_of_200_then_300 * kv_bytes_per_token
    assert cost.new_kv_bytes >= actual_cache_bytes


def test_retained_allocator_capacity_is_not_subtracted_from_other_work(geometry):
    cold_cost = estimate(geometry)
    retained_cache = cold_cost.new_kv_bytes * 2
    active_cost = estimate(geometry, current_cache_bytes=retained_cache)
    assert active_cost.new_kv_bytes == 0
    assert active_cost.temporary_workspace_bytes == cold_cost.temporary_workspace_bytes
    assert active_cost.cache_transition_bytes >= retained_cache


def test_retained_group_capacity_is_included_when_padding_a_wide_decode_join(geometry):
    batch_size = 2
    decode_batch_size = 32
    retained_tokens = 16_384
    kv_bytes_per_token = (
        2
        * geometry.num_layers
        * geometry.num_kv_heads
        * geometry.head_dim
        * geometry.cache_dtype_size
    )
    retained_cache = batch_size * retained_tokens * kv_bytes_per_token

    cost = estimate(
        geometry,
        batch_size=batch_size,
        max_prompt_tokens=128,
        current_cache_bytes=retained_cache,
        decode_batch_size=decode_batch_size,
        decode_max_tokens=128,
    )

    joined_cache = (
        (batch_size + decode_batch_size) * retained_tokens * kv_bytes_per_token
    )
    assert cost.cache_transition_bytes >= retained_cache + 2 * joined_cache


def test_cache_transition_reserves_extracted_rows_and_decode_join(geometry):
    cost = estimate(geometry)
    assert cost.cache_transition_bytes >= 2 * cost.new_kv_bytes


def test_long_incoming_prompt_reserves_padding_for_existing_decode_rows(geometry):
    no_decode = estimate(geometry)
    short_decode = estimate(geometry, decode_batch_size=3, decode_max_tokens=128)
    long_decode = estimate(geometry, decode_batch_size=3, decode_max_tokens=2048)
    assert short_decode.cache_transition_bytes == long_decode.cache_transition_bytes
    assert short_decode.cache_transition_bytes > no_decode.cache_transition_bytes
    assert short_decode.new_kv_bytes == no_decode.new_kv_bytes


def test_existing_long_decode_context_expands_new_rows_at_handoff(geometry):
    short_decode = estimate(geometry, decode_batch_size=3, decode_max_tokens=128)
    long_decode = estimate(geometry, decode_batch_size=3, decode_max_tokens=8192)
    assert long_decode.cache_transition_bytes > short_decode.cache_transition_bytes


@pytest.mark.parametrize(
    ("decode_batch_size", "decode_tokens", "prompt_tokens"),
    [(64, 128, 8192), (2, 8192, 128)],
    ids=["pad-existing-wide-decode", "pad-incoming-short-prompts"],
)
def test_padded_merge_inputs_coexist_with_concatenated_output(
    geometry, decode_batch_size, decode_tokens, prompt_tokens
):
    shallow_geometry = replace(
        geometry,
        num_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        hidden_size=128,
        intermediate_size=256,
    )
    batch_size = 2
    kv_bytes_per_token = 2 * 64 * 2
    extracted_group_bytes = batch_size * prompt_tokens * kv_bytes_per_token
    joined_tokens = max(decode_tokens, prompt_tokens)
    padded_rows = decode_batch_size if decode_tokens < prompt_tokens else batch_size
    padded_input_bytes = padded_rows * joined_tokens * kv_bytes_per_token
    concatenated_bytes = (
        (batch_size + decode_batch_size) * joined_tokens * kv_bytes_per_token
    )
    cost = estimate(
        shallow_geometry,
        batch_size=batch_size,
        query_tokens=1,
        max_prompt_tokens=prompt_tokens,
        current_cache_bytes=extracted_group_bytes,
        decode_batch_size=decode_batch_size,
        decode_max_tokens=decode_tokens,
    )

    assert padded_input_bytes > cost.temporary_workspace_bytes
    assert cost.cache_transition_bytes >= (
        extracted_group_bytes + padded_input_bytes + concatenated_bytes
    )


def test_unfused_attention_and_dense_mlp_are_both_reserved(geometry):
    cost = estimate(geometry)
    score_matrix = 2 * geometry.num_attention_heads * 128 * 2048 * 4
    gated_mlp = 2 * 128 * 3 * geometry.intermediate_size * 4
    assert cost.temporary_workspace_bytes >= 2 * score_matrix + gated_mlp


def test_query_chunk_reduction_does_not_release_future_kv_reservation(geometry):
    small_chunk = estimate(geometry, query_tokens=64)
    large_chunk = estimate(geometry, query_tokens=256)
    assert small_chunk.new_kv_bytes == large_chunk.new_kv_bytes
    assert small_chunk.cache_transition_bytes == large_chunk.cache_transition_bytes
    assert small_chunk.temporary_workspace_bytes < large_chunk.temporary_workspace_bytes


def test_kv_storage_width_is_distinct_from_compute_workspace(model_args):
    wide_cache = model_geometry_from_args(model_args, dtype_size=4)
    narrow_cache = model_geometry_from_args(
        model_args, dtype_size=4, cache_dtype_size=2
    )
    wide_cost = estimate(wide_cache)
    narrow_cost = estimate(narrow_cache)
    assert narrow_cost.new_kv_bytes * 2 == wide_cost.new_kv_bytes
    assert narrow_cost.temporary_workspace_bytes == wide_cost.temporary_workspace_bytes


def test_logits_are_reserved_when_executor_evaluates_the_output(geometry):
    cache_only = estimate(geometry)
    evaluated_logits = estimate(geometry, evaluate_logits=True)
    assert evaluated_logits.temporary_workspace_bytes >= (
        cache_only.temporary_workspace_bytes + 2 * 128 * geometry.vocab_size * 4
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"query_tokens": -1},
        {"query_tokens": 2049},
        {"max_prompt_tokens": None},
        {"current_cache_bytes": -1},
        {"current_cache_bytes": 1.5},
        {"decode_batch_size": 1},
        {"decode_max_tokens": 1},
        {"decode_batch_size": False},
        {"evaluate_logits": 1},
    ],
)
def test_invalid_execution_geometry_is_unsupported(geometry, overrides):
    assert estimate(geometry, **overrides) is None


def test_missing_or_invalid_geometry_never_returns_a_zero_cost(geometry):
    assert estimate(None) is None
    assert estimate(replace(geometry, num_kv_heads=0)) is None


def test_invalid_physical_occupancy_is_rejected(geometry):
    with pytest.raises(ValueError, match="current_usage_bytes"):
        estimate(geometry).peak_bytes(-1)


@pytest.fixture
def expert_geometry(model_args):
    return model_geometry_from_args(
        {
            **vars(model_args),
            "model_type": "hy_v3",
            "head_dim": 128,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "num_shared_experts": 1,
            "expert_hidden_dim": 768,
            "first_k_dense_replace": 1,
        }
    )


def test_expert_workspace_includes_routes_selected_experts_and_shared_mlp(
    expert_geometry,
):
    assert expert_geometry.head_dim == 128
    assert expert_geometry.experts == PrefillExpertGeometry(128, 8, 768, 768)
    cost = estimate(expert_geometry)
    dense = estimate(replace(expert_geometry, experts=None))
    assert cost.new_kv_bytes == dense.new_kv_bytes
    assert cost.cache_transition_bytes == dense.cache_transition_bytes
    assert (
        cost.temporary_workspace_bytes
        > dense.temporary_workspace_bytes + 2 * 128 * 8 * 768 * 4
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_experts", 0),
        ("experts_per_token", 129),
        ("intermediate_size", -1),
        ("shared_intermediate_size", None),
    ],
)
def test_unknown_expert_geometry_fails_closed(expert_geometry, field, value):
    invalid = replace(
        expert_geometry, experts=replace(expert_geometry.experts, **{field: value})
    )
    assert estimate(invalid) is None
