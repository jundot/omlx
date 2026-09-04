# SPDX-License-Identifier: Apache-2.0
"""Atomic persistence for user-approved distributed model deployments."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

from .deployment import ClusterDeployment


def _model_key(model: str) -> str:
    path = Path(model).expanduser()
    if path.exists():
        return str(path.resolve())
    return model.strip()


class ClusterRegistry:
    """Thread-safe deployment registry with no credentials or private keys."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)
        self.path = self.base_path / "cluster" / "deployments.json"
        self._lock = threading.RLock()
        self._deployments: dict[str, ClusterDeployment] = {}
        self.load_error: str | None = None
        try:
            self._load()
        except ValueError as exc:
            # A corrupt optional cluster file must never prevent the local
            # oMLX server and GUI from starting. Fail closed: activate none.
            self._deployments = {}
            self.load_error = str(exc)

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._deployments = {}
                return
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"could not read cluster deployment registry: {exc}"
                ) from exc
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise ValueError("unsupported cluster deployment registry schema")
            raw_deployments = payload.get("deployments")
            if not isinstance(raw_deployments, list):
                raise ValueError("cluster deployment registry is malformed")
            deployments = [
                ClusterDeployment.from_dict(item) for item in raw_deployments
            ]
            self._deployments = {
                _model_key(deployment.model): deployment
                for deployment in deployments
            }
            if len(self._deployments) != len(deployments):
                raise ValueError("cluster deployment registry has duplicate models")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "deployments": [
                deployment.to_dict()
                for _, deployment in sorted(self._deployments.items())
            ],
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".deployments.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)

    def list(self) -> tuple[ClusterDeployment, ...]:
        with self._lock:
            return tuple(
                deployment
                for _, deployment in sorted(self._deployments.items())
            )

    def get_for_model(self, model: str) -> ClusterDeployment | None:
        with self._lock:
            return self._deployments.get(_model_key(model))

    def get(self, deployment_id: str) -> ClusterDeployment | None:
        with self._lock:
            return next(
                (
                    deployment
                    for deployment in self._deployments.values()
                    if deployment.deployment_id == deployment_id
                ),
                None,
            )

    def upsert(self, deployment: ClusterDeployment) -> None:
        with self._lock:
            duplicate = self.get(deployment.deployment_id)
            if duplicate is not None and _model_key(duplicate.model) != _model_key(
                deployment.model
            ):
                raise ValueError(
                    f"deployment ID {deployment.deployment_id!r} is already in use"
                )
            previous = dict(self._deployments)
            self._deployments[_model_key(deployment.model)] = deployment
            try:
                self._save()
            except Exception:
                self._deployments = previous
                raise
            self.load_error = None

    def remove(self, deployment_id: str) -> bool:
        with self._lock:
            key = next(
                (
                    key
                    for key, deployment in self._deployments.items()
                    if deployment.deployment_id == deployment_id
                ),
                None,
            )
            if key is None:
                return False
            previous = dict(self._deployments)
            del self._deployments[key]
            try:
                self._save()
            except Exception:
                self._deployments = previous
                raise
            return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "deployments": [deployment.to_dict() for deployment in self.list()],
            "load_error": self.load_error,
        }


_registry_lock = threading.Lock()
_configured_registry: ClusterRegistry | None = None


def configure_cluster_registry(base_path: Path) -> ClusterRegistry:
    global _configured_registry
    with _registry_lock:
        _configured_registry = ClusterRegistry(base_path)
        return _configured_registry


def get_cluster_registry() -> ClusterRegistry:
    with _registry_lock:
        if _configured_registry is None:
            raise RuntimeError("cluster registry is not configured")
        return _configured_registry
