# SPDX-License-Identifier: Apache-2.0

from omlx.cluster import model_inventory
from omlx.cluster.model_inventory import (
    engine_pool_model_inventory,
    merge_model_inventories,
    remote_model_inventory,
)
from omlx.cluster.planner import ModelLayout


def _model(
    *,
    size: int,
    path: str,
    model_id: str = "MiniMax-M3-4bit",
    model_type: str = "vlm",
):
    return {
        "id": model_id,
        "display_name": f"mlx-community/{model_id}",
        "model_path": path,
        "model_type": model_type,
        "config_model_type": "minimax_m3_vl",
        "estimated_size": size,
        "model_context_length": 1_048_576,
        "source_repo_id": f"mlx-community/{model_id}",
    }


def test_a_shared_model_is_listed_once_with_every_location():
    local = _model(size=62, path="/Users/omlx/.omlx/models/MiniMax-M3-4bit")
    studio = _model(size=236, path="/Users/omlx/.omlx/models/MiniMax-M3-4bit")

    [merged] = merge_model_inventories(
        [
            ("MacBook Pro", "127.0.0.1", [local]),
            ("Mac Studio", "studio", [studio]),
        ]
    )

    assert merged["location_count"] == 2
    assert [item["node_id"] for item in merged["locations"]] == [
        "MacBook Pro",
        "Mac Studio",
    ]
    assert merged["model_source"] == "studio"
    assert merged["source_node_id"] == "Mac Studio"
    assert merged["estimated_size"] == 236


def test_an_equal_complete_local_copy_is_preferred_over_ssh():
    model = _model(size=236, path="/models/m")

    [merged] = merge_model_inventories(
        [
            ("Studio", "studio", [model]),
            ("MacBook", "127.0.0.1", [model]),
        ]
    )

    assert merged["model_source"] == "127.0.0.1"


def test_remote_inventory_runs_the_peers_own_discovery(monkeypatch):
    captured = {}

    def fake_run(host, snippet, argument, **kwargs):
        captured.update(host=host, snippet=snippet, argument=argument, kwargs=kwargs)
        return [
            {
                "model_id": "m",
                "model_path": "/models/m",
                "model_type": "llm",
                "engine_type": "batched",
                "estimated_size": 10,
                "config_model_type": "llama",
            },
            {
                "model_id": "embed",
                "model_path": "/models/embed",
                "model_type": "embedding",
                "engine_type": "embedding",
                "estimated_size": 2,
            },
        ]

    monkeypatch.setattr(model_inventory, "run_remote_python", fake_run)

    models = remote_model_inventory("studio")

    assert [item["id"] for item in models] == ["m"]
    assert captured["host"] == "studio"
    assert "discover_models_from_dirs" in captured["snippet"]
    assert "GlobalSettings.load" in captured["snippet"]


def test_local_pool_inventory_keeps_vlms_for_cluster_compatibility():
    class Pool:
        def get_status(self):
            return {
                "models": [
                    {
                        "id": "m3",
                        "model_path": "/models/m3",
                        "model_type": "vlm",
                        "config_model_type": "minimax_m3_vl",
                        "estimated_size": 236,
                    },
                    {
                        "id": "embed",
                        "model_path": "/models/embed",
                        "model_type": "embedding",
                        "estimated_size": 2,
                    },
                ]
            }

    models = engine_pool_model_inventory(Pool())

    assert [item["id"] for item in models] == ["m3"]


def _client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from omlx.cluster.routes import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_cluster_inventory_endpoint_unions_local_and_peer_models(monkeypatch):
    from omlx.cluster import routes

    class Pool:
        def get_status(self):
            return {"models": [_model(size=62, path="/models/m3")]}

    monkeypatch.setattr(routes, "_get_engine_pool", lambda: Pool())
    monkeypatch.setattr(
        routes,
        "remote_model_inventory",
        lambda host, *, python_executable: [
            _model(size=236, path="/models/m3")
        ],
    )

    response = _client().post(
        "/admin/api/cluster/models",
        json={
            "hosts": [
                {"node_id": "MacBook", "ssh": "127.0.0.1"},
                {
                    "node_id": "Mac Studio",
                    "ssh": "studio",
                    "python_executable": "/opt/omlx/bin/python",
                },
            ]
        },
    )

    assert response.status_code == 200
    [model] = response.json()["models"]
    assert model["model_source"] == "studio"
    assert model["python_executable"] == "/opt/omlx/bin/python"
    assert model["location_count"] == 2


def test_catalogue_measures_a_peer_owned_model_on_the_peer(monkeypatch):
    from omlx.cluster import routes

    asked = {}

    def fake_layout(host, path, *, python_executable):
        asked.update(host=host, path=path, python=python_executable)
        return ModelLayout(
            source=path,
            fixed_weight_bytes=0,
            layer_weight_bytes=(1024,) * 8,
            supports_pipeline=True,
            kv_bytes_per_token_per_layer=128,
        )

    monkeypatch.setattr(routes, "remote_model_layout", fake_layout)
    response = _client().post(
        "/admin/api/cluster/catalogue",
        json={
            "nodes": [
                {
                    "node_id": "MacBook",
                    "capacity_bytes": 1 << 30,
                    "reserve_bytes": 1 << 20,
                },
                {
                    "node_id": "Studio",
                    "capacity_bytes": 1 << 30,
                    "reserve_bytes": 1 << 20,
                },
            ],
            "models": [
                {
                    "id": "m3",
                    "model_path": "/models/m3",
                    "model_source": "studio",
                    "model_source_python": "/opt/omlx/bin/python",
                    "source_node_id": "Studio",
                    "model_context_length": 262144,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert asked == {
        "host": "studio",
        "path": "/models/m3",
        "python": "/opt/omlx/bin/python",
    }
    assert response.json()["models"][0]["model_source"] == "studio"
    assert response.json()["models"][0]["fits"] is True


def test_plan_carries_the_selected_model_holder_to_remote_measurement(monkeypatch):
    from omlx.cluster import routes

    asked = {}

    def fake_layout(host, path, *, python_executable):
        asked.update(host=host, path=path, python=python_executable)
        return ModelLayout(
            source=path,
            fixed_weight_bytes=0,
            layer_weight_bytes=(1024,) * 8,
            supports_pipeline=True,
        )

    monkeypatch.setattr(routes, "remote_model_layout", fake_layout)
    response = _client().post(
        "/admin/api/cluster/plan",
        json={
            "model_path": "/models/m3",
            "model_source": "studio",
            "model_source_python": "/opt/omlx/bin/python",
            "nodes": [
                {"node_id": "MacBook", "capacity_bytes": 1 << 30},
                {"node_id": "Studio", "capacity_bytes": 1 << 30},
            ],
        },
    )

    assert response.status_code == 200
    assert asked == {
        "host": "studio",
        "path": "/models/m3",
        "python": "/opt/omlx/bin/python",
    }
