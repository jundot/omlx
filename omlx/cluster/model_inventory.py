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
    "import json\n"
    "from dataclasses import asdict\n"
    "from omlx.model_discovery import discover_models_from_dirs\n"
    "from omlx.settings import GlobalSettings\n"
    # A stock-oMLX peer predates this fork's MTP helper. Losing that one fact
    # must not lose the peer's whole inventory, so the import is guarded and
    # the fallback answers fail-closed (False) per model.
    "try:\n"
    "    from omlx.cluster.model_inventory import mtp_compatible_for_model\n"
    "except ImportError:\n"
    "    def mtp_compatible_for_model(model_path):\n"
    "        return False\n"
    "s=GlobalSettings.load()\n"
    "m=discover_models_from_dirs(s.get_effective_model_dirs())\n"
    "print(json.dumps([{**d, 'mtp_compatible': mtp_compatible_for_model(d.get('model_path', ''))} for d in (asdict(v) for v in m.values())]))"
)

_CLUSTER_MODEL_TYPES = {"llm", "vlm"}


def mtp_compatible_for_model(model_path: str) -> bool:
    """Whether this checkpoint can take oMLX's Lightning MTP draft head.

    The admin model list already computes this from the config and the weight
    index. The cluster inventory needs the same fact so the assistant's
    ``assistantMtpFor`` does not read a missing key and plan every MTP-capable
    model with ``mtp: false``. Importing the admin helper lazily keeps the
    cluster module free of an import cycle; any read failure is fail-closed.
    """

    path = str(model_path or "").strip()
    if not path:
        return False
    try:
        from ..admin.routes import _mtp_compat_for_model

        compatible, _reason = _mtp_compat_for_model({"model_path": path})
        return bool(compatible)
    except Exception:
        return False


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
                # Carried so the assistant can decide MTP from the same model
                # list it already loads. Missing/failed reads are false, never
                # an invented true.
                "mtp_compatible": bool(item.get("mtp_compatible", False)),
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
        # GGUF models run on an external engine, never across the cluster.
        if model_type not in _CLUSTER_MODEL_TYPES or item.get("model_format") == "gguf":
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
                "mtp_compatible": (
                    bool(item.get("mtp_compatible"))
                    or mtp_compatible_for_model(model_path)
                ),
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
                "mtp_compatible": any(
                    bool(model.get("mtp_compatible"))
                    for _, _, model in locations
                ),
            }
        )
        merged.append(result)
    return merged


__all__ = [
    "engine_pool_model_inventory",
    "merge_model_inventories",
    "mtp_compatible_for_model",
    "remote_model_inventory",
]


_REMOTE_MODEL_LAYOUT_SNIPPET = (
    "import json,sys;"
    "from omlx.cluster.planner import inspect_safetensors_layout;"
    "print(json.dumps(inspect_safetensors_layout(sys.argv[1]).to_dict()))"
)


def remote_model_layout(
    ssh_target: str,
    model_path: str,
    *,
    python_executable: str = DEFAULT_REMOTE_PYTHON,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """The planner layout of a checkpoint read *on the Mac that holds it*.

    A model that lives only on a peer cannot be read locally, and planning from
    a guess would split a model that does not exist. The peer answers with the
    same ``inspect_safetensors_layout`` this node would run, so the measurement
    records the shape of the real checkpoint and the plan, apply and start can
    all use it (``MeasurementRecord.layout``).
    """

    payload = run_remote_python(
        ssh_target,
        _REMOTE_MODEL_LAYOUT_SNIPPET,
        model_path,
        description="read the model layout",
        python_executable=python_executable,
        timeout=timeout,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid model layout from {ssh_target}")
    return payload
