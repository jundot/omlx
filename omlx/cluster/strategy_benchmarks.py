# SPDX-License-Identifier: Apache-2.0
"""Durable end-to-end measurements for automatic parallelism selection."""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CONTEXT_BUCKETS = (1024, 8192, 32768, 262144)
_MAX_RECORDS = 256


def _model_key(model: str) -> str:
    path = Path(model).expanduser()
    return str(path.resolve()) if path.exists() else model.strip()


def context_bucket(tokens: int) -> int:
    value = max(1, int(tokens))
    return next((limit for limit in _CONTEXT_BUCKETS if value <= limit), _CONTEXT_BUCKETS[-1])


@dataclass(frozen=True)
class StrategyBenchmark:
    model: str
    node_ids: tuple[str, ...]
    backend: str
    tensor_parallel_size: int
    context_tokens: int
    prompt_tokens_per_second: float
    decode_tokens_per_second: float
    time_to_first_token_seconds: float
    measured_at: str
    samples: int = 1
    source: str = "end_to_end_request"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("benchmark model is required")
        if len(self.node_ids) < 2 or any(not item for item in self.node_ids):
            raise ValueError("benchmark requires at least two node IDs")
        if self.backend not in {"ring", "jaccl", "jaccl-ring"}:
            raise ValueError("benchmark backend is invalid")
        if not 1 <= self.tensor_parallel_size <= len(self.node_ids):
            raise ValueError("benchmark tensor parallel size is invalid")
        if self.context_tokens < 1:
            raise ValueError("benchmark context must be positive")
        for label, value in (
            ("prompt throughput", self.prompt_tokens_per_second),
            ("decode throughput", self.decode_tokens_per_second),
            ("time to first token", self.time_to_first_token_seconds),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"benchmark {label} must be finite and positive")
        if not 1 <= self.samples <= 10_000:
            raise ValueError("benchmark samples are invalid")
        if self.source != "end_to_end_request":
            raise ValueError("benchmark source is unsupported")
        datetime.fromisoformat(self.measured_at.replace("Z", "+00:00"))

    @property
    def strategy(self) -> str:
        return "tensor" if self.tensor_parallel_size > 1 else "pipeline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": _model_key(self.model),
            "node_ids": list(self.node_ids),
            "backend": self.backend,
            "tensor_parallel_size": self.tensor_parallel_size,
            "strategy": self.strategy,
            "context_tokens": self.context_tokens,
            "context_bucket": context_bucket(self.context_tokens),
            "prompt_tokens_per_second": self.prompt_tokens_per_second,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "time_to_first_token_seconds": self.time_to_first_token_seconds,
            "measured_at": self.measured_at,
            "samples": self.samples,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StrategyBenchmark:
        if not isinstance(payload, dict):
            raise ValueError("strategy benchmark must be an object")
        try:
            return cls(
                model=str(payload["model"]),
                node_ids=tuple(str(item) for item in payload["node_ids"]),
                backend=str(payload["backend"]),
                tensor_parallel_size=int(payload["tensor_parallel_size"]),
                context_tokens=int(payload["context_tokens"]),
                prompt_tokens_per_second=float(
                    payload["prompt_tokens_per_second"]
                ),
                decode_tokens_per_second=float(payload["decode_tokens_per_second"]),
                time_to_first_token_seconds=float(
                    payload["time_to_first_token_seconds"]
                ),
                measured_at=str(payload["measured_at"]),
                samples=int(payload.get("samples", 1)),
                source=str(payload.get("source", "end_to_end_request")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and not isinstance(exc, KeyError):
                raise
            raise ValueError("strategy benchmark is missing required fields") from exc


class StrategyBenchmarkStore:
    def __init__(self, base_path: Path) -> None:
        self.path = Path(base_path) / "cluster" / "strategy-benchmarks.json"
        self._lock = threading.RLock()
        self._records: list[StrategyBenchmark] = []
        self.load_error: str | None = None
        try:
            self._load()
        except ValueError as exc:
            self.load_error = str(exc)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read strategy benchmarks: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("unsupported strategy benchmark schema")
        raw = payload.get("benchmarks")
        if not isinstance(raw, list):
            raise ValueError("strategy benchmark store is malformed")
        self._records = [StrategyBenchmark.from_dict(item) for item in raw][
            -_MAX_RECORDS:
        ]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".strategy-benchmarks.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "schema_version": 1,
                        "benchmarks": [item.to_dict() for item in self._records],
                    },
                    stream,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)

    def record(self, benchmark: StrategyBenchmark) -> None:
        with self._lock:
            self._records.append(benchmark)
            self._records = self._records[-_MAX_RECORDS:]
            self._save()
            self.load_error = None

    def measurements(
        self,
        *,
        model: str,
        node_ids: tuple[str, ...],
        backend: str,
        target_context_tokens: int,
    ) -> dict[int, StrategyBenchmark]:
        """Return robust medians for each TP degree at the nearest context bucket."""

        key = _model_key(model)
        target = context_bucket(target_context_tokens)
        # Request completion records from a worker thread while Automatic may
        # read from another request thread. Snapshot under the same lock used
        # by record()/to_dict() so a truncating append cannot change the sample
        # set halfway through the median calculation.
        with self._lock:
            records = tuple(self._records)
        matching = [
            item
            for item in records
            if _model_key(item.model) == key
            # Rank order is part of the measured strategy. It is especially
            # important for pipeline parallelism, where rank zero owns the
            # final stage and reversing two unequal Macs changes the result.
            and item.node_ids == node_ids
            and item.backend == backend
            and context_bucket(item.context_tokens) == target
        ]
        result: dict[int, StrategyBenchmark] = {}
        for tp_size in sorted({item.tensor_parallel_size for item in matching}):
            samples = [
                item for item in matching if item.tensor_parallel_size == tp_size
            ][-7:]
            if not samples:
                continue
            result[tp_size] = StrategyBenchmark(
                model=key,
                node_ids=node_ids,
                backend=backend,
                tensor_parallel_size=tp_size,
                context_tokens=target,
                prompt_tokens_per_second=statistics.median(
                    item.prompt_tokens_per_second for item in samples
                ),
                decode_tokens_per_second=statistics.median(
                    item.decode_tokens_per_second for item in samples
                ),
                time_to_first_token_seconds=statistics.median(
                    item.time_to_first_token_seconds for item in samples
                ),
                measured_at=max(item.measured_at for item in samples),
                samples=len(samples),
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "benchmarks": [item.to_dict() for item in self._records],
                "load_error": self.load_error,
            }


def new_benchmark(
    *,
    model: str,
    node_ids: tuple[str, ...],
    backend: str,
    tensor_parallel_size: int,
    context_tokens: int,
    prompt_tokens_per_second: float,
    decode_tokens_per_second: float,
    time_to_first_token_seconds: float,
) -> StrategyBenchmark:
    return StrategyBenchmark(
        model=model,
        node_ids=node_ids,
        backend=backend,
        tensor_parallel_size=tensor_parallel_size,
        context_tokens=context_tokens,
        prompt_tokens_per_second=prompt_tokens_per_second,
        decode_tokens_per_second=decode_tokens_per_second,
        time_to_first_token_seconds=time_to_first_token_seconds,
        measured_at=datetime.now(UTC).isoformat(),
    )


_configured_store: StrategyBenchmarkStore | None = None
_configured_lock = threading.Lock()


def configure_strategy_benchmark_store(base_path: Path) -> StrategyBenchmarkStore:
    global _configured_store
    with _configured_lock:
        _configured_store = StrategyBenchmarkStore(base_path)
        return _configured_store


def get_strategy_benchmark_store() -> StrategyBenchmarkStore:
    with _configured_lock:
        if _configured_store is None:
            raise RuntimeError("strategy benchmark store is not configured")
        return _configured_store


__all__ = [
    "StrategyBenchmark",
    "StrategyBenchmarkStore",
    "configure_strategy_benchmark_store",
    "context_bucket",
    "get_strategy_benchmark_store",
    "new_benchmark",
]
