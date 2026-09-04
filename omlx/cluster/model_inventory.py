# SPDX-License-Identifier: Apache-2.0
"""Discover and merge models held by every Mac in a cluster.

The ordinary oMLX model list is intentionally server-local.  A cluster picker
has a different job: it must show the union of every selected Mac and remember
which Mac owns the complete copy that planning and staging should read.

Inventories are read over the already-enrolled SSH connection by running the
same discovery code on the peer.  No model weights are loaded and no remote
admin credential is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .staging import DEFAULT_REMOTE_PYTHON, run_remote_python

_REMOTE_MODEL_INVENTORY_SNIPPET = (
    "import json;"
    "from dataclasses import asdict;"
    "from omlx.model_discovery import discover_models_from_dirs;"
    "from omlx.settings import GlobalSettings;"
    "s=GlobalSettings.load();"
    "m=discover_models_from_dirs(s.get_effective_model_dirs());"
    "print(json.dumps([asdict(v) for v in m.values()]))"
)

_CLUSTER_MODEL_TYPES = {"llm", "vlm"}


def remote_model_inventory(
    ssh_target: str,
    *,
    python_executable: str = DEFAULT_REMOTE_PYTHON,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """Return the peer's discoverable LLM/VLM models without loading them."""

    payload = run_remote_python(
        ssh_target,
        _REMOTE_MODEL_INVENTORY_SNIPPET,
        "inventory",
        description="read the model inventory",
        python_executable=python_executable,
        timeout=timeout,
    )
    if not isinstance(payload, list):
        raise RuntimeError(f"invalid model inventory from {ssh_target}")

    models: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model_id")
        model_path = item.get("model_path")
        model_type = item.get("model_type")
        estimated_size = item.get("estimated_size")
        if (
            not isinstance(model_id, str)
            or not model_id
            or not isinstance(model_path, str)
            or not model_path
            or model_type not in _CLUSTER_MODEL_TYPES
            or not isinstance(estimated_size, int)
            or isinstance(estimated_size, bool)
            or estimated_size < 0
        ):
            continue
        models.append(
            {
                "id": model_id,
                "display_name": item.get("source_repo_id") or model_id,
                "model_path": model_path,
                "model_type": model_type,
                "config_model_type": str(item.get("config_model_type") or ""),
                "estimated_size": estimated_size,
                "model_context_length": item.get("model_context_length"),
                "source_type": str(item.get("source_type") or "local"),
                "source_repo_id": item.get("source_repo_id"),
                "is_helper": bool(item.get("is_helper", False)),
                "loaded": False,
                "is_loading": False,
                "is_default": False,
                "is_favorite": False,
            }
        )
    return models


def engine_pool_model_inventory(pool: Any) -> list[dict[str, Any]]:
    """Normalize the coordinator's live engine-pool status for cluster use."""

    status = pool.get_status()
    raw_models = status.get("models", []) if isinstance(status, dict) else []
    models: list[dict[str, Any]] = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        model_path = item.get("model_path")
        model_type = item.get("model_type", "llm")
        if not isinstance(model_path, str) or not model_path:
            continue
        if model_type not in _CLUSTER_MODEL_TYPES:
            continue
        estimated_size = item.get("estimated_size", 0)
        if (
            not isinstance(estimated_size, int)
            or isinstance(estimated_size, bool)
            or estimated_size < 0
        ):
            estimated_size = 0
        model_id = str(item.get("id") or model_path.rstrip("/").split("/")[-1])
        models.append(
            {
                "id": model_id,
                "display_name": item.get("source_repo_id") or model_id,
                "model_path": model_path,
                "model_type": model_type,
                "config_model_type": str(item.get("config_model_type") or ""),
                "estimated_size": estimated_size,
                "model_context_length": item.get("model_context_length"),
                "source_type": str(item.get("source_type") or "local"),
                "source_repo_id": item.get("source_repo_id"),
                "is_helper": bool(item.get("is_helper", False)),
                "loaded": bool(item.get("loaded", False)),
                "is_loading": bool(item.get("is_loading", False)),
                "is_default": bool(item.get("is_default", False)),
                "is_favorite": bool(item.get("is_favorite", False)),
            }
        )
    return models


def _model_identity(model: dict[str, Any]) -> str:
    repo = model.get("source_repo_id")
    if isinstance(repo, str) and repo:
        return f"repo:{repo.lower()}"
    return (
        f"id:{str(model.get('id') or '').lower()}:"
        f"{str(model.get('config_model_type') or '').lower()}"
    )


def merge_model_inventories(
    inventories: Sequence[tuple[str, str, Sequence[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Deduplicate shared models and choose the most complete source copy.

    A previous pipeline run can leave only one stage on the coordinator.  That
    directory still looks like the same model, but its measured size is much
    smaller than the complete Studio copy.  Choosing the largest location is a
    deterministic, useful proxy for completeness; the planner independently
    verifies the declared layer count before trusting it.
    """

    grouped: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    order: list[str] = []
    for node_id, ssh_target, models in inventories:
        for raw in models:
            model = dict(raw)
            identity = _model_identity(model)
            if identity not in grouped:
                grouped[identity] = []
                order.append(identity)
            grouped[identity].append((node_id, ssh_target, model))

    merged: list[dict[str, Any]] = []
    for identity in order:
        locations = grouped[identity]
        source_node, source_ssh, source_model = max(
            locations,
            key=lambda item: (
                int(item[2].get("estimated_size") or 0),
                item[1] in {"127.0.0.1", "localhost", "::1"},
            ),
        )
        result = dict(source_model)
        result.update(
            {
                "model_key": identity,
                "model_source": source_ssh,
                "source_node_id": source_node,
                "location_count": len(locations),
                "locations": [
                    {
                        "node_id": node_id,
                        "ssh": ssh_target,
                        "model_path": model.get("model_path"),
                        "estimated_size": int(model.get("estimated_size") or 0),
                        "python_executable": model.get("python_executable"),
                    }
                    for node_id, ssh_target, model in locations
                ],
                "loaded": any(bool(model.get("loaded")) for _, _, model in locations),
                "is_loading": any(
                    bool(model.get("is_loading")) for _, _, model in locations
                ),
                "is_default": any(
                    bool(model.get("is_default")) for _, _, model in locations
                ),
                "is_favorite": any(
                    bool(model.get("is_favorite")) for _, _, model in locations
                ),
            }
        )
        merged.append(result)
    return merged


__all__ = [
    "engine_pool_model_inventory",
    "merge_model_inventories",
    "remote_model_inventory",
]
