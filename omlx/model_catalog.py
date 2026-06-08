# SPDX-License-Identifier: Apache-2.0
"""Persistent model inventory metadata for oMLX."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

CATALOG_VERSION = 1
ModelSource = Literal["hf", "modelscope", "local", "unknown"]


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _infer_download_provenance(path: str) -> dict[str, str]:
    """Infer remote provenance from downloader-created local_dir layouts."""
    if not path:
        return {}

    model_path = Path(path)
    if (
        (model_path / ".cache" / "huggingface").exists()
        and model_path.parent.name
        and model_path.parent.name != "models"
    ):
        repo_id = f"{model_path.parent.name}/{model_path.name}"
        return {
            "source": "hf",
            "provider": "huggingface",
            "repo_id": repo_id,
        }

    return {}


@dataclass
class ModelCatalogEntry:
    """Durable metadata for one discovered local model."""

    model_id: str
    path: str
    source: ModelSource = "unknown"
    provider: str = ""
    repo_id: str = ""
    downloaded_at: str = ""
    local_revision: str = ""
    remote_revision: str = ""
    remote_updated_at: str = ""
    last_checked_at: str = ""
    update_status: str = "not_checked"
    removed: bool = False
    last_perf_result_id: str = ""
    best_perf_summary: dict[str, Any] = field(default_factory=dict)
    last_accuracy_result_id: str = ""
    best_accuracy_summary: dict[str, Any] = field(default_factory=dict)
    accuracy_summaries_by_benchmark: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelCatalogEntry":
        valid = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("removed", None)
        return data


class ModelCatalog:
    """Thread-safe JSON-backed model catalog."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.catalog_file = self.base_path / "model_catalog.json"
        self._lock = threading.Lock()
        self._entries: dict[str, ModelCatalogEntry] = {}
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.catalog_file.exists():
            self._entries = {}
            return
        try:
            with open(self.catalog_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("models", {})
            self._entries = {
                model_id: ModelCatalogEntry.from_dict(entry)
                for model_id, entry in entries.items()
                if isinstance(entry, dict)
            }
        except Exception as e:
            logger.error(f"Failed to load model catalog {self.catalog_file}: {e}")
            self._entries = {}

    def _save_locked(self) -> None:
        data = {
            "version": CATALOG_VERSION,
            "models": {
                model_id: asdict(entry)
                for model_id, entry in sorted(self._entries.items())
            },
        }
        tmp = self.catalog_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp.replace(self.catalog_file)

    def get(self, model_id: str) -> ModelCatalogEntry | None:
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                return None
            return ModelCatalogEntry.from_dict(asdict(entry))

    def get_public(self, model_id: str) -> dict[str, Any]:
        entry = self.get(model_id)
        return entry.to_public_dict() if entry else {}

    def record_download(
        self,
        *,
        model_id: str,
        path: Path,
        source: ModelSource,
        repo_id: str,
        local_revision: str = "",
        remote_revision: str = "",
        remote_updated_at: str = "",
    ) -> None:
        with self._lock:
            existing = self._entries.get(model_id)
            downloaded_at = existing.downloaded_at if existing else ""
            entry = ModelCatalogEntry(
                model_id=model_id,
                path=str(path),
                source=source,
                provider="huggingface" if source == "hf" else source,
                repo_id=repo_id,
                downloaded_at=downloaded_at or utc_now(),
                local_revision=local_revision or remote_revision,
                remote_revision=remote_revision,
                remote_updated_at=remote_updated_at,
                last_checked_at=(
                    utc_now() if remote_revision or remote_updated_at else ""
                ),
                update_status=(
                    "current" if remote_revision or remote_updated_at else "not_checked"
                ),
                removed=False,
                last_perf_result_id=existing.last_perf_result_id if existing else "",
                best_perf_summary=existing.best_perf_summary if existing else {},
                last_accuracy_result_id=(
                    existing.last_accuracy_result_id if existing else ""
                ),
                best_accuracy_summary=(
                    existing.best_accuracy_summary if existing else {}
                ),
                accuracy_summaries_by_benchmark=(
                    existing.accuracy_summaries_by_benchmark if existing else {}
                ),
            )
            self._entries[model_id] = entry
            self._save_locked()

    def reconcile(self, discovered: list[dict[str, Any]]) -> None:
        """Ensure every discovered model has a catalog entry."""
        seen = {m.get("id") for m in discovered if m.get("id")}
        with self._lock:
            for model in discovered:
                model_id = model.get("id")
                path = model.get("model_path") or ""
                if not model_id:
                    continue
                entry = self._entries.get(model_id)
                provenance = _infer_download_provenance(path)
                if entry is None:
                    self._entries[model_id] = ModelCatalogEntry(
                        model_id=model_id,
                        path=path,
                        source=provenance.get(
                            "source",
                            "local" if path else "unknown",
                        ),
                        provider=provenance.get(
                            "provider",
                            "local" if path else "",
                        ),
                        repo_id=provenance.get("repo_id", ""),
                    )
                else:
                    entry.path = path or entry.path
                    if (
                        provenance
                        and entry.source in ("local", "unknown")
                        and not entry.repo_id
                    ):
                        entry.source = provenance["source"]
                        entry.provider = provenance["provider"]
                        entry.repo_id = provenance["repo_id"]
                    entry.removed = False
            for model_id, entry in self._entries.items():
                if model_id not in seen:
                    entry.removed = True
            self._save_locked()

    def update_remote_state(
        self,
        model_id: str,
        *,
        remote_revision: str = "",
        remote_updated_at: str = "",
        error: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                return {}
            entry.last_checked_at = utc_now()
            if error:
                entry.update_status = "check_failed"
            else:
                old_remote_updated_at = entry.remote_updated_at
                entry.remote_revision = remote_revision or entry.remote_revision
                entry.remote_updated_at = remote_updated_at or entry.remote_updated_at
                if remote_revision and entry.local_revision:
                    entry.update_status = (
                        "current"
                        if remote_revision == entry.local_revision
                        else "update_available"
                    )
                elif remote_updated_at and old_remote_updated_at:
                    entry.update_status = (
                        "current"
                        if remote_updated_at == old_remote_updated_at
                        else "update_available"
                    )
                else:
                    entry.update_status = "unknown"
            self._save_locked()
            return entry.to_public_dict()

    def update_perf_summary(
        self, model_id: str, result_id: str, summary: dict[str, Any]
    ) -> None:
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                entry = ModelCatalogEntry(model_id=model_id, path="", source="unknown")
                self._entries[model_id] = entry
            entry.last_perf_result_id = result_id
            entry.best_perf_summary = summary
            self._save_locked()

    def replace_accuracy_summaries(
        self, summaries_by_model: dict[str, dict[str, Any]]
    ) -> None:
        """Replace catalog accuracy summaries with a recomputed snapshot."""
        with self._lock:
            for entry in self._entries.values():
                entry.last_accuracy_result_id = ""
                entry.best_accuracy_summary = {}
                entry.accuracy_summaries_by_benchmark = {}

            for model_id, summary in summaries_by_model.items():
                entry = self._entries.get(model_id)
                if entry is None:
                    entry = ModelCatalogEntry(
                        model_id=model_id, path="", source="unknown"
                    )
                    self._entries[model_id] = entry
                entry.last_accuracy_result_id = summary.get(
                    "last_accuracy_result_id", ""
                )
                entry.best_accuracy_summary = dict(
                    summary.get("best_accuracy_summary") or {}
                )
                entry.accuracy_summaries_by_benchmark = dict(
                    summary.get("accuracy_summaries_by_benchmark") or {}
                )

            self._save_locked()
