"""Cold text batching accepts only proven model and cache combinations."""

from types import SimpleNamespace

import mlx.nn as nn
import pytest
from mlx_lm.models import llama, qwen2, qwen3
from mlx_lm.models.cache import KVCache, RotatingKVCache

from omlx.prefill.policy import prefill_eligibility


@pytest.fixture(params=[llama, qwen2, qwen3], ids=["llama", "qwen2", "qwen3"])
def dense_model(request):
    module = request.param
    args = {
        "model_type": module.__name__.rsplit(".", 1)[-1],
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": 64,
        "max_position_embeddings": 128,
        "rope_theta": 10_000,
        "tie_word_embeddings": True,
    }
    if module is qwen3:
        args["head_dim"] = 8
    return module.Model(module.ModelArgs(**args))


@pytest.fixture
def cold_request():
    return SimpleNamespace(
        remaining_tokens=[1, 2, 3],
        cached_tokens=0,
        rope_deltas=0.0,
    )


def _fresh_cache():
    return [KVCache(), KVCache()]


def test_known_dense_models_with_fresh_cache_are_eligible(dense_model, cold_request):
    result = prefill_eligibility(dense_model, cold_request, _fresh_cache())

    assert result.eligible
    assert result.reason == "supported"


def test_architecture_name_alone_does_not_prove_support(cold_request):
    result = prefill_eligibility(
        SimpleNamespace(model_type="llama"), cold_request, _fresh_cache()
    )

    assert not result.eligible
    assert result.reason == "unsupported_model"


def test_custom_subclasses_do_not_inherit_batching_support(cold_request):
    class CustomModel(llama.Model):
        pass

    model = CustomModel(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=2,
            intermediate_size=32,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
        )
    )

    assert not prefill_eligibility(model, cold_request, _fresh_cache()).eligible


@pytest.mark.parametrize(
    ("attribute", "value", "reason"),
    [
        ("cached_tokens", 1, "prefix_cache"),
        ("remaining_tokens", [], "no_prefill_tokens"),
        ("remaining_tokens", [1], "no_prefill_tokens"),
        ("remaining_tokens", None, "no_prefill_tokens"),
        ("vlm_inputs_embeds", object(), "multimodal_request"),
        ("vlm_extra_kwargs", {"position_ids": object()}, "multimodal_request"),
        ("images", [object()], "multimodal_request"),
        ("videos", [object()], "multimodal_request"),
        ("specprefill_indices", object(), "speculative_prefill"),
        ("_specprefill_enabled", True, "speculative_prefill"),
        ("rope_deltas", 1.0, "unsupported_position_state"),
    ],
)
def test_request_features_select_singleton_fallback(
    dense_model, cold_request, attribute, value, reason
):
    setattr(cold_request, attribute, value)

    result = prefill_eligibility(dense_model, cold_request, _fresh_cache())

    assert not result.eligible
    assert result.reason == reason


def test_one_prefill_token_is_valid_after_reserving_kickoff(dense_model, cold_request):
    cold_request.remaining_tokens = [7, 8]

    assert prefill_eligibility(dense_model, cold_request, _fresh_cache()).eligible


def test_unprepared_tokens_do_not_confuse_empty_prefix_hit(dense_model, cold_request):
    cold_request.remaining_tokens = []
    cold_request.prompt_token_ids = [1, 2, 3]

    assert not prefill_eligibility(dense_model, cold_request, _fresh_cache()).eligible


@pytest.mark.parametrize(
    "setting", ["turboquant_enabled", "speculative_enabled", "distributed_enabled"]
)
def test_scheduler_execution_modes_veto_cold_cache(dense_model, cold_request, setting):
    assert not prefill_eligibility(
        dense_model, cold_request, _fresh_cache(), **{setting: True}
    ).eligible


@pytest.mark.parametrize(
    "attribute",
    [
        "_uses_mrope",
        "_omlx_mtp_decode_enabled",
        "_omlx_ane_mlp_prefill_count",
        "_omlx_ane_gdn_prefill_count",
        "_omlx_ane_down_prefill_count",
        "_omlx_ane_dual_prefill_count",
    ],
)
@pytest.mark.parametrize("target", ["model", "backbone"])
def test_model_execution_flags_veto_batching(
    dense_model, cold_request, attribute, target
):
    model_part = dense_model if target == "model" else dense_model.model
    setattr(model_part, attribute, True)

    assert not prefill_eligibility(dense_model, cold_request, _fresh_cache()).eligible


@pytest.mark.parametrize("target", ["model", "backbone"])
def test_custom_prefill_entrypoint_selects_existing_path(
    dense_model, cold_request, target
):
    model_part = dense_model if target == "model" else dense_model.model
    model_part._omlx_prefill = lambda *args, **kwargs: None

    result = prefill_eligibility(dense_model, cold_request, _fresh_cache())

    assert not result.eligible
    assert result.reason == "custom_execution"


def test_sliding_attention_is_not_inferred_from_plain_caches(dense_model, cold_request):
    dense_model.args.layer_types = ["full_attention", "sliding_attention"]

    result = prefill_eligibility(dense_model, cold_request, _fresh_cache())

    assert not result.eligible
    assert result.reason == "unsupported_attention"


@pytest.mark.parametrize(
    "caches",
    [None, [], [KVCache()], [KVCache(), RotatingKVCache(max_size=8)]],
)
def test_missing_partial_and_special_cache_layouts_are_rejected(
    dense_model, cold_request, caches
):
    result = prefill_eligibility(dense_model, cold_request, caches)

    assert not result.eligible
    assert result.reason == "unsupported_cache"


def test_cache_subclass_with_merge_is_not_implicitly_supported(
    dense_model, cold_request
):
    class CustomCache(KVCache):
        pass

    result = prefill_eligibility(dense_model, cold_request, [KVCache(), CustomCache()])

    assert not result.eligible
    assert result.reason == "unsupported_cache"


@pytest.mark.parametrize("populated_field", ["offset", "keys", "values"])
def test_cache_state_must_be_fresh_even_if_request_metadata_says_cold(
    dense_model, cold_request, populated_field
):
    caches = _fresh_cache()
    setattr(caches[0], populated_field, 1)

    result = prefill_eligibility(dense_model, cold_request, caches)

    assert not result.eligible
    assert result.reason == "prefix_cache"


@pytest.mark.parametrize("world_size", [1, 2])
def test_tensor_sharding_does_not_inherit_outer_model_support(
    dense_model, cold_request, world_size
):
    model_type = type(dense_model)
    dense_model.shard(SimpleNamespace(size=lambda: world_size, rank=lambda: 0))

    result = prefill_eligibility(dense_model, cold_request, _fresh_cache())

    assert type(dense_model) is model_type
    assert not result.eligible
    assert result.reason == "unsupported_geometry"


@pytest.mark.parametrize("target", ["model", "backbone"])
def test_pipeline_assignment_does_not_inherit_local_support(
    dense_model, cold_request, target
):
    model_part = dense_model if target == "model" else dense_model.model
    model_part.pipeline_size = 2
    model_part.pipeline_rank = 0

    result = prefill_eligibility(dense_model, cold_request, _fresh_cache())

    assert not result.eligible
    assert result.reason == "distributed_execution"


def test_unsupported_projection_cannot_hide_inside_supported_model(
    dense_model, cold_request
):
    class CustomLinear(nn.Linear):
        pass

    dense_model.layers[0].mlp.gate_proj = CustomLinear(32, 64)

    result = prefill_eligibility(dense_model, cold_request, _fresh_cache())

    assert not result.eligible
    assert result.reason == "unsupported_geometry"


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("num_hidden_layers", 1),
        ("head_dim", 4),
        ("intermediate_size", 32),
        ("vocab_size", 32),
    ],
)
def test_live_model_dimensions_must_match_memory_geometry(
    dense_model, cold_request, attribute, value
):
    setattr(dense_model.args, attribute, value)

    result = prefill_eligibility(dense_model, cold_request, _fresh_cache())

    assert not result.eligible
    assert result.reason == "unsupported_geometry"


def test_custom_rope_cannot_hide_inside_supported_model(dense_model, cold_request):
    dense_model.layers[0].self_attn.rope = nn.Identity()

    result = prefill_eligibility(dense_model, cold_request, _fresh_cache())

    assert not result.eligible
    assert result.reason == "unsupported_position_state"


@pytest.mark.parametrize("type_key", ["type", "rope_type"])
def test_mrope_arguments_do_not_inherit_plain_rope_support(
    dense_model, cold_request, type_key
):
    dense_model.args.rope_scaling = {type_key: "mrope"}

    result = prefill_eligibility(dense_model, cold_request, _fresh_cache())

    assert not result.eligible
    assert result.reason == "unsupported_position_state"


def test_affine_weight_quantization_preserves_local_support(dense_model, cold_request):
    nn.quantize(dense_model, group_size=32, bits=4)

    assert prefill_eligibility(dense_model, cold_request, _fresh_cache()).eligible


def test_unknown_quantization_mode_falls_back(dense_model, cold_request):
    nn.quantize(dense_model, group_size=32, bits=4)
    dense_model.layers[0].mlp.gate_proj.mode = "unverified"

    result = prefill_eligibility(dense_model, cold_request, _fresh_cache())

    assert not result.eligible
    assert result.reason == "unsupported_geometry"
