import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.dspark.compat import (
    detect_format,
    model_fingerprint,
    probe_drafter,
    validate_pair,
)
from omlx.dspark.provider import NativeDSparkProvider, SpeculativeDraftProvider
from omlx.dspark.smoke import verify_cross_target_smoke
from omlx.model_discovery import is_helper_model_config
from omlx.model_settings import ModelSettings
from omlx.output_collector import RequestOutputCollector
from omlx.request import RequestOutput
from omlx.scheduler import Scheduler


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _target(tmp_path):
    root = tmp_path / "target"
    root.mkdir()
    _write_json(
        root / "config.json",
        {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "vocab_size": 100,
                "hidden_size": 64,
                "num_hidden_layers": 8,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
            },
        },
    )
    return root


def _drafter(tmp_path, target, *, vocab=100, hidden=64, taps=None):
    root = tmp_path / f"draft-{vocab}-{hidden}"
    root.mkdir()
    _write_json(
        root / "config.json",
        {
            "architectures": ["Qwen3DSparkModel"],
            "vocab_size": vocab,
            "hidden_size": hidden,
            "target_layer_ids": taps or [1, 4, 7],
            "block_size": 4,
        },
    )
    (root / "model.safetensors").write_bytes(b"test inventory")
    _write_json(
        root / "dspark_manifest.json",
        {
            "target_fingerprint": model_fingerprint(target),
            "target_num_hidden_layers": 8,
            "owns_embedding": True,
            "owns_output_head": True,
            "quantization": {"status": "ready", "bits": 2, "group_size": 64},
        },
    )
    return root


def test_dspark_is_mutually_exclusive_with_other_speculators():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ModelSettings(dspark_enabled=True, dflash_enabled=True)


def test_dspark_reuses_native_cache_features():
    settings = ModelSettings(
        dspark_enabled=True,
        specprefill_enabled=True,
        turboquant_kv_enabled=True,
        turboquant_kv_bits=3,
    )
    assert settings.dspark_enabled
    assert settings.specprefill_enabled
    assert settings.turboquant_kv_enabled


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"architectures": ["Qwen3DSparkModel"]}, "deepspec"),
        ({"architectures": ["DFlashDraftModel"], "markov_rank": 256}, "speculators"),
        ({"architectures": ["DFlashDraftModel"]}, "higgs_sidecar"),
    ],
)
def test_format_detection(config, expected):
    assert detect_format(config) == expected


@pytest.mark.parametrize(
    "architecture",
    ["Qwen3DSparkModel", "DeepSpecModel", "SpeculatorsDraftModel"],
)
def test_dspark_architectures_are_hidden_helpers(architecture):
    assert is_helper_model_config({"architectures": [architecture]})


def test_exact_pair_accepts_matching_managed_checkpoint(tmp_path):
    target = _target(tmp_path)
    result = validate_pair(target, _drafter(tmp_path, target), pairing_mode="exact")
    assert result.compatible
    assert result.blocked_reasons == ()
    assert result.capabilities["continuous_batching"] is True
    assert result.capabilities["continuous_scheduling"] is True
    assert result.capabilities["batched_target_verify"] is True
    assert result.capabilities["turboquant"] is True
    assert result.capabilities["specprefill"] is True
    assert result.capabilities["specprefill_mode"] == "native_target_only_cap_zero"


def test_higgs_sidecar_is_a_loadable_shared_head_format(tmp_path):
    target = _target(tmp_path)
    root = tmp_path / "higgs"
    root.mkdir()
    _write_json(
        root / "config.json",
        {
            "architectures": ["DFlashDraftModel"],
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "intermediate_size": 128,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 1024,
            "num_target_layers": 8,
            "target_layer_ids": [1, 4, 7],
            "block_size": 4,
        },
    )
    (root / "model.safetensors").write_bytes(b"inventory")
    _write_json(
        root / "dspark_manifest.json",
        {
            "target_fingerprint": model_fingerprint(target),
            "target_num_hidden_layers": 8,
            "owns_embedding": False,
            "owns_output_head": False,
            "quantization": {"status": "ready", "bits": 2, "group_size": 64},
        },
    )
    result = validate_pair(target, root, pairing_mode="exact")
    assert result.compatible
    assert result.capabilities["shares_target_embedding"] is True


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"vocab": 101}, "vocab_size mismatch"),
        ({"hidden": 32}, "hidden_size mismatch"),
        ({"taps": [9]}, "target_layer_ids out of range"),
    ],
)
def test_structural_mismatch_is_blocked(tmp_path, changes, message):
    target = _target(tmp_path)
    result = validate_pair(
        target,
        _drafter(tmp_path, target, **changes),
        pairing_mode="verified_cross_target",
    )
    assert not result.compatible
    assert any(message in reason for reason in result.blocked_reasons)


def test_cross_target_requires_explicit_mode(tmp_path):
    target = _target(tmp_path)
    drafter = _drafter(tmp_path, target)
    manifest = drafter / "dspark_manifest.json"
    data = json.loads(manifest.read_text())
    data["target_fingerprint"] = hashlib.sha256(b"other target").hexdigest()
    _write_json(manifest, data)
    exact = validate_pair(target, drafter, pairing_mode="exact")
    cross = validate_pair(target, drafter, pairing_mode="verified_cross_target")
    assert not exact.compatible
    assert cross.compatible
    assert cross.warnings


def test_speculators_nested_geometry_and_taps(tmp_path):
    root = tmp_path / "speculators"
    root.mkdir()
    _write_json(
        root / "config.json",
        {
            "architectures": ["DSparkDraftModel"],
            "speculators_config": {"algorithm": "dspark"},
            "aux_hidden_state_layer_ids": [9, 19, 29],
            "draft_vocab_size": 32000,
            "markov_rank": 256,
            "block_size": 8,
            "transformer_layer_config": {
                "model_type": "qwen3",
                "vocab_size": 248320,
                "hidden_size": 2048,
                "num_hidden_layers": 3,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
            },
        },
    )
    (root / "model.safetensors").write_bytes(b"inventory")
    _write_json(
        root / "dspark_manifest.json",
        {
            "target_num_hidden_layers": 40,
            "owns_embedding": True,
            "owns_output_head": True,
        },
    )
    probe = probe_drafter(root)
    assert probe.format == "speculators"
    assert probe.vocab_size == 248320
    assert probe.hidden_size == 2048
    assert probe.target_layers == 40
    assert probe.target_layer_ids == (9, 19, 29)
    assert probe.owns_embedding and probe.owns_output_head


def test_cross_target_smoke_delegates_to_native_provider():
    class Tokenizer:
        @staticmethod
        def encode(_text):
            return [1, 2]

    class Provider:
        @staticmethod
        def greedy_smoke(prompt_ids, max_tokens):
            assert prompt_ids == [1, 2]
            assert max_tokens == 4
            raise ValueError("diverged at position 1")

    with pytest.raises(ValueError, match="diverged at position 1"):
        verify_cross_target_smoke(Provider(), Tokenizer())


def test_logprobs_are_merged_by_native_output_collector():
    collector = RequestOutputCollector()
    first = RequestOutput(
        request_id="r",
        new_token_ids=[1],
        logprobs=[{"token_id": 1, "logprob": -0.1, "top": []}],
    )
    second = RequestOutput(
        request_id="r",
        new_token_ids=[2],
        output_token_ids=[1, 2],
        logprobs=[{"token_id": 2, "logprob": -0.2, "top": []}],
    )
    merged = collector._merge_outputs(first, second)
    assert [entry["token_id"] for entry in merged.logprobs] == [1, 2]


def test_scheduler_steps_all_native_dspark_requests_without_private_queue():
    scheduler = object.__new__(Scheduler)
    states = {
        -1: SimpleNamespace(finished=False),
        -2: SimpleNamespace(finished=False),
    }

    class Provider:
        @staticmethod
        def step_batch(items):
            for uid, state in items:
                state.finished = uid == -1
            return [uid for uid, _ in items]

    scheduler._dspark_provider = Provider()
    scheduler._dspark_active = states
    assert scheduler._step_dspark() == [-1, -2]
    assert list(scheduler._dspark_active) == [-2]


def test_native_provider_batches_equal_width_target_verify(monkeypatch):
    provider = object.__new__(NativeDSparkProvider)
    provider._totals = {}
    provider.tap = [1]

    states = [
        SimpleNamespace(
            finished=False,
            ready=[],
            emitted=1,
            cap=2,
            auto_cap=False,
            native_speculation=True,
            pending=10 + index,
            cache=[f"cache-{index}"],
            rng_state=list(mx.random.state),
        )
        for index in range(2)
    ]
    calls = []

    class Target:
        @staticmethod
        def verify_batch(ids, caches, tap):
            calls.append((ids.tolist(), caches, tap))
            return (
                mx.zeros((2, 3, 8)),
                mx.zeros((2, 3, 4)),
                [["advanced-0"], ["advanced-1"]],
                [None, None],
            )

    provider.target = Target()
    monkeypatch.setattr(provider, "_propose_only", lambda state: ([1, 2], None))
    monkeypatch.setattr(provider, "snapshot", lambda state: {"history_len": 1})
    monkeypatch.setattr(
        provider,
        "_finish_verified",
        lambda state, *args, **kwargs: [(state.pending, None)],
    )
    monkeypatch.setattr(provider, "_emit_pairs", lambda uid, state, pairs: [uid])

    assert provider.step_batch([(-1, states[0]), (-2, states[1])]) == [-1, -2]
    assert len(calls) == 1
    assert calls[0][0] == [[10, 1, 2], [11, 1, 2]]
    assert [state.cache for state in states] == [["advanced-0"], ["advanced-1"]]
    assert provider._totals["batched_verify_rounds"] == 1
    assert provider._totals["batched_verify_rows"] == 2


def test_native_provider_protocol_has_full_lifecycle_contract():
    names = {
        "probe",
        "load",
        "close",
        "create_request_state",
        "step_batch",
        "prefill_context",
        "propose",
        "snapshot",
        "rollback",
        "commit",
        "update_after_verify",
        "memory_usage",
        "stats",
    }
    assert names <= set(SpeculativeDraftProvider.__dict__)


def test_runtime_sources_do_not_import_external_server():
    root = Path(__file__).parents[1] / "omlx"
    source = "\n".join(path.read_text() for path in root.rglob("*.py"))
    assert "mlx_dspark.server" not in source
