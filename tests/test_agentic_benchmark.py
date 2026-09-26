# SPDX-License-Identifier: Apache-2.0
"""Tests for the Harbor-driven agentic benchmarks (no Docker or network).

A fake ``harbor`` script stands in for the real CLI: it honours
``--job-name``/``-o``/``-i`` and writes Harbor-shaped trial ``result.json``
files, so the runner's progress, parsing, and cancellation paths run for real.
"""

import asyncio
import os
import sys
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import omlx.admin.accuracy_benchmark as accuracy_benchmark
import omlx.eval.agentic as agentic
from omlx.admin.accuracy_benchmark import (
    AccuracyBenchmarkRequest,
    _resolve_agent_endpoint,
    create_run,
    run_accuracy_benchmark,
)
from omlx.admin.external_api import ExternalEndpointConfig
from omlx.eval.agentic import AgentEndpoint, HarborBenchmark
from omlx.eval.base import BenchmarkResult, QuestionResult

FAKE_HARBOR = textwrap.dedent(
    """
    import json, os, sys, time
    from datetime import datetime, timedelta, timezone

    args = sys.argv[1:]
    job = args[args.index("--job-name") + 1]
    out = args[args.index("-o") + 1]
    tasks = [args[i + 1] for i, a in enumerate(args) if a == "-i"]
    mode = os.environ["FAKE_HARBOR_MODE"]
    if pidfile := os.environ.get("FAKE_HARBOR_PIDFILE"):
        open(pidfile, "w").write(str(os.getpid()))
    if mode == "fail":
        print("boom: registry unreachable")
        sys.exit(2)
    if mode == "sleep":
        time.sleep(60)
        sys.exit(0)
    outcomes = [
        {"verifier_result": {"rewards": {"reward": 1}}},
        {"verifier_result": {"rewards": {"reward": 0}}},
        {"verifier_result": None, "exception_info": {
            "exception_type": "AgentTimeoutError",
            "exception_message": "agent exceeded 900s",
        }},
    ]
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # The last requested task is deliberately never run.
    for i, task in enumerate(tasks[:-1]):
        trial = os.path.join(out, job, task.replace("/", "__") + "__abc")
        os.makedirs(trial, exist_ok=True)
        data = {
            "task_name": task,
            "started_at": start.isoformat(),
            "finished_at": (start + timedelta(seconds=10 * (i + 1))).isoformat(),
            "agent_result": {"n_input_tokens": 100 * (i + 1), "n_output_tokens": 7},
            **outcomes[i],
        }
        json.dump(data, open(os.path.join(trial, "result.json"), "w"))
        time.sleep(0.2)
    """
)

ITEMS = [
    {"id": "org/alpha", "category": "Software"},
    {"id": "org/beta", "category": "Software"},
    {"id": "org/gamma", "category": "Science"},
    {"id": "org/delta", "category": "Science"},
]


class _Suite(HarborBenchmark):
    name = "fake_suite"
    dataset = "org/fake@1"


@pytest.fixture
def suite(tmp_path, monkeypatch):
    script = tmp_path / "fake_harbor.py"
    script.write_text(FAKE_HARBOR)
    monkeypatch.setattr(
        agentic, "resolve_harbor_command", lambda: [sys.executable, str(script)]
    )
    monkeypatch.setattr(agentic, "check_docker", AsyncMock())
    monkeypatch.setattr(agentic, "check_endpoint", AsyncMock())
    bench = _Suite()
    bench.dataset_total = 100
    bench.configure_agent_run(
        endpoint=AgentEndpoint("http://host.docker.internal:8000/v1", "k", "m"),
        jobs_dir=tmp_path / "jobs",
        job_name="fake-1",
        agent_timeout_multiplier=1.0,
    )
    return bench


@pytest.mark.asyncio
async def test_run_maps_harbor_trials_to_results(suite, monkeypatch):
    monkeypatch.setenv("FAKE_HARBOR_MODE", "ok")
    progress: list[tuple[int, int]] = []

    async def on_progress(current, total):
        progress.append((current, total))

    result = await suite.run(None, ITEMS, on_progress, batch_size=2)

    assert [q.predicted for q in result.question_results] == [
        "pass", "fail", "error: AgentTimeoutError", "not_run",
    ]
    assert result.correct_count == 1
    assert result.accuracy == pytest.approx(0.25)
    assert result.category_scores == {"Science": 0.0, "Software": 0.5}
    alpha, _, gamma, delta = result.question_results
    assert alpha.time_seconds == pytest.approx(10.0)
    assert alpha.prompt_tokens == 100 and alpha.completion_tokens == 7
    assert (suite.job_dir / "org__alpha__abc").samefile(alpha.artifact_path)
    assert gamma.error_message == "agent exceeded 900s"
    assert delta.artifact_path is None
    assert progress[-1] == (3, 4)
    assert [c for c, _ in progress] == sorted(c for c, _ in progress)


@pytest.mark.asyncio
async def test_host_python_env_does_not_leak_into_harbor(suite, monkeypatch, tmp_path):
    # The macOS app runs the server with PYTHONHOME/PYTHONPATH pointing at
    # its bundled CPython; Harbor's own interpreter must not inherit them.
    monkeypatch.setenv("FAKE_HARBOR_MODE", "ok")
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "bundled-cpython-3.11"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "bundled-resources"))

    result = await suite.run(None, ITEMS)

    assert result.question_results[0].predicted == "pass"


@pytest.mark.asyncio
async def test_harbor_failure_without_trials_raises(suite, monkeypatch):
    monkeypatch.setenv("FAKE_HARBOR_MODE", "fail")
    with pytest.raises(RuntimeError, match="Harbor exited with code 2.*boom"):
        await suite.run(None, ITEMS)


@pytest.mark.asyncio
async def test_cancel_terminates_harbor_process_group(suite, monkeypatch, tmp_path):
    pidfile = tmp_path / "harbor.pid"
    monkeypatch.setenv("FAKE_HARBOR_MODE", "sleep")
    monkeypatch.setenv("FAKE_HARBOR_PIDFILE", str(pidfile))

    task = asyncio.create_task(suite.run(None, ITEMS))
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text():
            break
        await asyncio.sleep(0.05)
    pid = int(pidfile.read_text())

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)


def test_agent_timeout_multiplier_is_validated():
    with pytest.raises(ValueError, match="agent_timeout_multiplier"):
        AccuracyBenchmarkRequest(
            model_id="m",
            benchmarks={"swebench_verified": 10},
            agent_timeout_multiplier=3,
        )


def test_external_loopback_endpoint_is_rewritten_for_containers():
    req = AccuracyBenchmarkRequest(
        model_id="remote",
        benchmarks={"terminalbench_4": 5},
        external=ExternalEndpointConfig(
            base_url="http://localhost:1234/v1", model="remote-model"
        ),
    )
    endpoint = _resolve_agent_endpoint(req)
    assert endpoint.base_url == "http://host.docker.internal:1234/v1"
    assert endpoint.model == "remote-model"
    assert endpoint.api_key == "omlx"


@pytest.mark.asyncio
async def test_agentic_suite_is_never_uploaded(tmp_path):
    class Stub(HarborBenchmark):
        name = "terminalbench_4"

        async def load_dataset(self, sample_size=0):
            self.dataset_total = 66
            return [{"id": "t"}]

        async def run(self, engine, items, on_progress=None, **kwargs):
            q = QuestionResult("t", True, "pass", "pass", 1.0, artifact_path="/x")
            return BenchmarkResult("terminalbench_4", 1.0, 1, 1, 1.0, [q])

    req = AccuracyBenchmarkRequest(model_id="m", benchmarks={"terminalbench_4": 1})
    run = create_run(req)
    pool = MagicMock()
    pool.get_loaded_model_ids = MagicMock(return_value=[])
    pool.get_engine = AsyncMock(return_value=MagicMock())
    pool._unload_engine = AsyncMock()
    pool._settings_manager = None
    upload = AsyncMock()
    endpoint = AgentEndpoint("http://host.docker.internal:8000/v1", "k", "m")

    with (
        patch.dict("omlx.eval.BENCHMARKS", {"terminalbench_4": Stub}, clear=True),
        patch.object(accuracy_benchmark, "build_upload_context", return_value={}),
        patch.object(accuracy_benchmark, "upload_intelligence_result", upload),
        patch.object(accuracy_benchmark, "_resolve_agent_endpoint", return_value=endpoint),
        patch.object(accuracy_benchmark, "_agent_jobs_dir", return_value=tmp_path),
    ):
        await run_accuracy_benchmark(run, pool)

    assert run.status == "completed"
    upload.assert_not_awaited()
    result = next(e["data"] for e in run.events if e["type"] == "result")
    assert result["sampling_profile"] == "server_defaults"
    assert result["agent"] == "pi"
    assert result["question_results"][0]["artifact_path"] == "/x"
    assert "upload" not in result
