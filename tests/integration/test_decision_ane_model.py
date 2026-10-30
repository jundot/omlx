# SPDX-License-Identifier: Apache-2.0
"""Optional real-checkpoint test of ANE through the actual HTTP handler."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from omlx.custom_kernels.quant_matmul import ane
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.scheduler import SchedulerConfig
from omlx.settings import GlobalSettings
from scripts.benchmark_decision_ane import check_answers, request


@pytest.mark.slow
def test_real_clef_endpoint_dispatches_ane_and_releases_it(tmp_path, monkeypatch):
    source = os.environ.get("OMLX_TEST_CLEF_MODEL")
    if not source:
        pytest.skip("Set OMLX_TEST_CLEF_MODEL to a local quantized Clef checkpoint")
    if not ane.ane_available():
        pytest.skip("Build native kernels on an ANE-capable Mac first")
    model = Path(source).expanduser().resolve()
    assert (model / "joint_head.safetensors").is_file()
    from omlx import server
    from omlx.cluster import worker_shim

    monkeypatch.setattr(worker_shim, "ensure_cluster_python_shim", lambda: None)
    settings = GlobalSettings(base_path=tmp_path)
    settings.model.model_dirs = [str(model.parent)]
    settings.server.server_aliases = ["127.0.0.1"]
    settings.memory.memory_guard_tier = "balanced"
    manager = ModelSettingsManager(tmp_path)
    manager.set_settings(
        model.name,
        ModelSettings(
            ane_prefill_enabled=True,
            ane_prefill_sequence_length=1024,
            ane_prefill_fraction=0.25,
            ane_prefill_max_layers=32,
            ane_prefill_dual_ane=False,
            ane_prefill_gdn=False,
        ),
    )
    key = "local-ane-integration-test"
    server.init_server(
        model_dirs=[str(model.parent)],
        scheduler_config=SchedulerConfig(prefill_step_size=2048),
        api_key=key,
        global_settings=settings,
    )
    headers = {"Authorization": f"Bearer {key}"}
    try:
        with TestClient(server.app) as client:
            # Load and compile with a short request; exclude warmup dispatches.
            response = client.post(
                "/v1/systemone",
                json={"model": model.name, **request(0)},
                headers=headers,
            )
            assert response.status_code == 200
            check_answers(response.json())
            assert ane.ane_profile_set_enabled(True)
            ane.ane_profile_reset()
            response = client.post(
                "/v1/systemone",
                json={"model": model.name, **request(160)},
                headers=headers,
            )
            assert response.status_code == 200
            check_answers(response.json())
            assert response.json()["usage"]["output_tokens"] == 0
            assert ane.ane_profile_snapshot()["mlp"]["operations"] > 0
            status = client.get("/api/status", headers=headers)
            assert status.status_code == 200
            assert status.json()["ane_prefill"]["configured_models"] == 1
            engine = server._server_state.engine_pool._entries[model.name].engine
            assert engine.ane_status()["driver_weight_bytes"] > 0
        assert engine.backbone_model is None
    finally:
        ane.ane_profile_set_enabled(False)
