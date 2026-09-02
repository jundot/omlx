# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.planner import PipelineAssignment
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.exceptions import ModelUnavailableError


def _deployment(model_path: str) -> ClusterDeployment:
    return ClusterDeployment(
        deployment_id="pool-test",
        model=model_path,
        backend="ring",
        hosts=(
            ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("peer", "peer.local", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment("local", 0, 3, 8, 80, 10, 8, 128),
            PipelineAssignment("peer", 1, 0, 3, 40, 10, 8, 64),
        ),
        plan_hash="f" * 64,
    )


def _entry(model_path: str) -> EngineEntry:
    return EngineEntry(
        model_id="nemotron",
        model_path=model_path,
        model_type="llm",
        engine_type="batched",
        estimated_size=300,
    )


def _vision_entry(model_path) -> EngineEntry:
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "vision_n_layers": 32,
                "vision_dim": 1024,
                "vision_n_heads": 16,
                "vision_inter_dim": 2816,
                "vision_patch_size": 14,
                "vision_rope_theta": 10000.0,
                "vision_downsample_ratio": 3,
                "vision_max_n_token": 384,
                "vision_min_pixels": 147456,
                "vision_max_wh_ratio": 8,
            }
        )
    )
    entry = _entry(str(model_path))
    entry.model_id = "deepseek-v4-vision"
    entry.model_type = "vlm"
    entry.engine_type = "vlm"
    return entry


def test_engine_pool_admits_only_rank_zero_resident_weight(tmp_path):
    model_path = str(tmp_path / "nemotron")
    deployment = _deployment(model_path)
    pool = EnginePool()
    pool._cluster_registry = SimpleNamespace(
        get_for_model=lambda model: deployment if model == model_path else None
    )
    entry = _entry(model_path)

    assert pool._entry_resident_size(entry) == 90
    assert entry.estimated_size == 300


def test_loaded_engine_retains_resident_accounting_after_deactivation(tmp_path):
    model_path = str(tmp_path / "nemotron")
    deployment = _deployment(model_path)
    pool = EnginePool()
    pool._cluster_registry = SimpleNamespace(get_for_model=lambda model: None)
    entry = _entry(model_path)
    entry.engine = MagicMock(deployment=deployment)

    assert pool._entry_resident_size(entry) == 90


def test_activation_does_not_relabel_an_already_loaded_local_engine(tmp_path):
    model_path = str(tmp_path / "nemotron")
    deployment = _deployment(model_path)
    pool = EnginePool()
    pool._cluster_registry = SimpleNamespace(
        get_for_model=lambda model: deployment if model == model_path else None
    )
    entry = _entry(model_path)
    entry.engine = object()

    assert pool._distributed_deployment_for_entry(entry) is None
    assert pool._entry_resident_size(entry) == 300


def test_pool_status_reports_full_and_local_cluster_sizes(tmp_path):
    model_path = str(tmp_path / "nemotron")
    deployment = _deployment(model_path)
    pool = EnginePool()
    pool._cluster_registry = SimpleNamespace(
        get_for_model=lambda model: deployment if model == model_path else None
    )
    pool._entries["nemotron"] = _entry(model_path)

    model = pool.get_status()["models"][0]

    assert model["estimated_size"] == 300
    assert model["resident_estimated_size"] == 90
    assert model["distributed"] is True


def test_cluster_model_path_resolves_to_public_model_id(tmp_path):
    model_path = tmp_path / "nemotron"
    model_path.mkdir()
    pool = EnginePool()
    pool._entries["friendly-name"] = _entry(str(model_path))

    assert pool.resolve_cluster_model_id(str(model_path)) == "friendly-name"


def test_cluster_model_path_collapses_equivalent_public_aliases(tmp_path):
    model_path = tmp_path / "snapshot"
    model_path.mkdir()
    pool = EnginePool()
    hashed = _entry(str(model_path))
    repo = _entry(str(model_path))
    repo.source_type = "huggingface"
    repo.source_repo_id = "owner/model"
    pool._entries["87e768fb"] = hashed
    pool._entries["owner--model"] = repo

    assert pool.resolve_cluster_model_id(str(model_path)) == "owner--model"


def test_cluster_model_path_rejects_incompatible_public_aliases(tmp_path):
    model_path = tmp_path / "snapshot"
    model_path.mkdir()
    pool = EnginePool()
    text = _entry(str(model_path))
    vision = _entry(str(model_path))
    vision.model_type = "vlm"
    vision.engine_type = "vlm"
    pool._entries["text"] = text
    pool._entries["vision"] = vision

    with pytest.raises(ValueError, match="incompatible public model IDs"):
        pool.resolve_cluster_model_id(str(model_path))


def test_active_cluster_deployment_id_resolves_to_public_model_id(tmp_path):
    model_path = tmp_path / "nemotron"
    model_path.mkdir()
    deployment = _deployment(str(model_path))
    pool = EnginePool()
    pool._entries["friendly-name"] = _entry(str(model_path))
    pool._cluster_registry = SimpleNamespace(
        get=lambda deployment_id: (
            deployment if deployment_id == deployment.deployment_id else None
        )
    )

    assert (
        pool.resolve_model_id(deployment.deployment_id, settings_manager=None)
        == "friendly-name"
    )


def test_stale_cluster_deployment_id_preserves_normal_not_found_behavior(tmp_path):
    deployment = _deployment(str(tmp_path / "missing"))
    pool = EnginePool()
    pool._cluster_registry = SimpleNamespace(
        get=lambda deployment_id: (
            deployment if deployment_id == deployment.deployment_id else None
        )
    )

    assert (
        pool.resolve_model_id(deployment.deployment_id, settings_manager=None)
        == deployment.deployment_id
    )


def test_cluster_model_path_rejects_non_text_model(tmp_path):
    model_path = tmp_path / "vision"
    model_path.mkdir()
    pool = EnginePool()
    entry = _entry(str(model_path))
    entry.model_type = "vlm"
    entry.engine_type = "vlm"
    pool._entries["vision"] = entry

    with pytest.raises(ValueError, match="text LLM models only"):
        pool.resolve_cluster_model_id(str(model_path))


def test_cluster_model_path_accepts_gated_deepseek_v4_vision(tmp_path):
    model_path = tmp_path / "deepseek-v4-vision"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "vision_n_layers": 32,
                "vision_dim": 1024,
                "vision_n_heads": 16,
                "vision_inter_dim": 2816,
                "vision_patch_size": 14,
                "vision_rope_theta": 10000.0,
                "vision_downsample_ratio": 3,
                "vision_max_n_token": 384,
                "vision_min_pixels": 147456,
                "vision_max_wh_ratio": 8,
            }
        )
    )
    pool = EnginePool()
    entry = _entry(str(model_path))
    entry.model_type = "vlm"
    entry.engine_type = "vlm"
    pool._entries["deepseek-v4-vision"] = entry

    assert (
        pool.resolve_cluster_model_id(str(model_path)) == "deepseek-v4-vision"
    )


@pytest.mark.parametrize("force_lm", [False, True])
async def test_gated_vision_deployment_uses_distributed_engine(
    tmp_path,
    monkeypatch,
    force_lm,
):
    entry = _vision_entry(tmp_path / "deepseek-v4-vision")
    deployment = _deployment(entry.model_path)
    pool = EnginePool()
    pool._entries[entry.model_id] = entry
    pool._cluster_registry = SimpleNamespace(
        get_for_model=lambda model: deployment if model == entry.model_path else None
    )
    engine = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    distributed = MagicMock(return_value=engine)
    local_vlm = MagicMock(side_effect=AssertionError("local VLM load attempted"))
    local_llm = MagicMock(side_effect=AssertionError("local LLM load attempted"))
    monkeypatch.setattr(
        "omlx.engine.distributed.DistributedBatchedEngine", distributed
    )
    monkeypatch.setattr("omlx.engine_pool.VLMBatchedEngine", local_vlm)
    monkeypatch.setattr("omlx.engine_pool.BatchedEngine", local_llm)

    await pool._load_engine(entry.model_id, force_lm=force_lm)

    distributed.assert_called_once()
    engine.start.assert_awaited_once()
    assert entry.engine is engine
    local_vlm.assert_not_called()
    local_llm.assert_not_called()


async def test_failed_gated_vision_deployment_never_falls_back_locally(
    tmp_path,
    monkeypatch,
):
    entry = _vision_entry(tmp_path / "deepseek-v4-vision")
    deployment = _deployment(entry.model_path)
    pool = EnginePool()
    pool._entries[entry.model_id] = entry
    pool._cluster_registry = SimpleNamespace(
        get_for_model=lambda model: deployment if model == entry.model_path else None
    )
    engine = SimpleNamespace(
        start=AsyncMock(side_effect=RuntimeError("rank start failed")),
        stop=AsyncMock(),
    )
    local_vlm = MagicMock(side_effect=AssertionError("local VLM load attempted"))
    local_llm = MagicMock(side_effect=AssertionError("local LLM load attempted"))
    monkeypatch.setattr(
        "omlx.engine.distributed.DistributedBatchedEngine",
        MagicMock(return_value=engine),
    )
    monkeypatch.setattr("omlx.engine_pool.VLMBatchedEngine", local_vlm)
    monkeypatch.setattr("omlx.engine_pool.BatchedEngine", local_llm)
    monkeypatch.setattr(pool, "_schedule_failed_load_reclaim", MagicMock())

    with pytest.raises(ModelUnavailableError, match="rank start failed"):
        await pool._load_engine(entry.model_id)

    assert entry.engine is None
    local_vlm.assert_not_called()
    local_llm.assert_not_called()


def test_remote_only_cluster_model_gets_a_batched_pool_entry(tmp_path):
    model_path = tmp_path / "minimax"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"model_type":"minimax_m3","max_position_embeddings":262144}'
    )
    pool = EnginePool()

    model_id, created = pool.register_cluster_model(
        str(model_path),
        estimated_size=236 * 1024**3,
    )

    entry = pool.get_entry(model_id)
    assert created is True
    assert model_id == "minimax"
    assert entry is not None
    assert entry.engine_type == "batched"
    assert entry.model_type == "llm"
    assert entry.source_type == "cluster"
    assert entry.model_context_length == 262144
    assert pool.resolve_cluster_model_id(str(model_path)) == model_id


def test_cluster_only_pool_entry_is_removed_after_registry_deactivation(tmp_path):
    model_path = tmp_path / "minimax"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"minimax_m3"}')
    pool = EnginePool()
    pool._cluster_registry = SimpleNamespace(get_for_model=lambda _model: None)
    model_id, _ = pool.register_cluster_model(
        str(model_path),
        estimated_size=236 * 1024**3,
    )

    assert pool.unregister_cluster_model(model_id) is True
    assert pool.get_entry(model_id) is None


async def test_distributed_unload_uses_process_teardown_as_memory_barrier(
    tmp_path,
    monkeypatch,
):
    model_path = str(tmp_path / "nemotron")
    deployment = _deployment(model_path)
    pool = EnginePool()
    entry = _entry(model_path)
    stop = AsyncMock()
    entry.engine = SimpleNamespace(deployment=deployment, stop=stop)
    pool._entries["nemotron"] = entry
    pool._current_model_memory = 90
    monkeypatch.setattr(
        "omlx.engine_pool.mx.get_active_memory",
        MagicMock(side_effect=AssertionError("main MLX gauge is unrelated")),
    )

    await pool._unload_engine("nemotron")

    stop.assert_awaited_once()
    assert entry.engine is None
    assert pool.current_model_memory == 0


async def test_failed_distributed_teardown_keeps_supervisor_reachable(tmp_path):
    model_path = str(tmp_path / "nemotron")
    deployment = _deployment(model_path)
    pool = EnginePool()
    entry = _entry(model_path)
    stop = AsyncMock(side_effect=RuntimeError("rank did not exit"))
    engine = SimpleNamespace(deployment=deployment, stop=stop)
    entry.engine = engine
    pool._entries["nemotron"] = entry
    pool._current_model_memory = 90

    try:
        await pool._unload_engine("nemotron")
    except RuntimeError as exc:
        assert "rank did not exit" in str(exc)
    else:
        raise AssertionError("distributed teardown failure was swallowed")

    assert entry.engine is engine
    assert pool.current_model_memory == 90
