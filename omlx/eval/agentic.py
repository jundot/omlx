# SPDX-License-Identifier: Apache-2.0
"""Agentic benchmarks (Terminal-Bench 4.0, SWE-bench Verified) via Harbor.

Harbor (https://harborframework.com) owns task containers, agent install,
verification and scoring; it runs its built-in ``pi`` coding agent against
oMLX's OpenAI-compatible endpoint. Harbor needs Python >=3.12 and a heavy
dependency set, so it runs as an isolated ``uvx`` tool, never imported.

Unlike the single-turn evaluators this is not a ``BaseBenchmark``: it
duck-types the ``load_dataset`` / ``run`` / ``dataset_total`` surface that
``omlx.admin.accuracy_benchmark`` drives.
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .base import BenchmarkResult, QuestionResult, compute_category_scores
from .datasets import deterministic_sample, load_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

HARBOR_VERSION = "0.23.0"
HARBOR_AGENT = "pi"
HARBOR_MODEL_API = "openai-completions"
TERMINALBENCH_4_DATASET = "terminal-bench/terminal-bench@4"
SWEBENCH_VERIFIED_DATASET = "swebench-verified@1.0"
DOCKER_HOST_ALIAS = "host.docker.internal"
PREFLIGHT_IMAGE = "curlimages/curl:8.10.1"

_POLL_SECONDS = 2.0
_SIGINT_GRACE_SECONDS = 60.0
_LOG_TAIL_CHARS = 2000


@dataclass(frozen=True)
class AgentEndpoint:
    """OpenAI-compatible endpoint as seen from inside a task container."""

    base_url: str  # includes /v1
    api_key: str
    model: str


# uv's default install dirs. GUI-launched servers (the macOS app) inherit a
# minimal PATH that adds Homebrew but not these.
_USER_TOOL_DIRS = ("~/.local/bin", "~/.cargo/bin")


def _which(name: str) -> Optional[str]:
    extra = os.pathsep.join(os.path.expanduser(d) for d in _USER_TOOL_DIRS)
    path = os.environ.get("PATH", "")
    return shutil.which(name, path=f"{path}{os.pathsep}{extra}" if path else extra)


# The macOS app points PYTHONHOME/PYTHONPATH at its bundled CPython 3.11 for
# the server; inherited by Harbor's own (uv-managed 3.13) interpreter they
# break its startup ("Failed to import encodings module").
_HOST_PYTHON_ENV = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONEXECUTABLE",
    "PYTHONSTARTUP",
    "VIRTUAL_ENV",
    "__PYVENV_LAUNCHER__",
)


def harbor_env(**overrides: str) -> dict[str, str]:
    """Environment for the Harbor CLI, free of this interpreter's settings."""
    env = {k: v for k, v in os.environ.items() if k not in _HOST_PYTHON_ENV}
    env.update(overrides)
    return env


def resolve_harbor_command() -> list[str]:
    """Command prefix that invokes the pinned Harbor CLI."""
    if uvx := _which("uvx"):
        return [
            uvx, "--python", "3.13",
            "--from", f"harbor=={HARBOR_VERSION}", "harbor",
        ]
    if harbor := _which("harbor"):
        return [harbor]
    raise RuntimeError(
        "Harbor is required for agentic benchmarks. Install uv "
        "(https://docs.astral.sh/uv/) or run: "
        f"uv tool install harbor=={HARBOR_VERSION}"
    )


async def _run_capture(
    argv: list[str], timeout: float, env: Optional[dict[str, str]] = None
) -> tuple[int, str]:
    """Run a short command; returns (exit code, stdout). -1 on timeout/missing."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
    except FileNotFoundError:
        return -1, ""
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, ""
    return proc.returncode or 0, stdout.decode(errors="replace").strip()


async def check_docker() -> None:
    code, _ = await _run_capture(["docker", "info"], timeout=20)
    if code != 0:
        raise RuntimeError(
            "Docker is not running. Start Docker Desktop, OrbStack, Colima, "
            "or a Podman machine and retry."
        )


async def check_endpoint(endpoint: AgentEndpoint) -> None:
    """Prove a container can reach the model endpoint with the agent's key."""
    script = (
        'curl -sS -o /dev/null -w "%{http_code}" --max-time 30 '
        '-H "Authorization: Bearer $OMLX_BENCH_KEY" "$0/models"'
    )
    _, code = await _run_capture(
        [
            "docker", "run", "--rm",
            "--add-host", f"{DOCKER_HOST_ALIAS}:host-gateway",
            "-e", "OMLX_BENCH_KEY",
            PREFLIGHT_IMAGE, "sh", "-c", script, endpoint.base_url,
        ],
        timeout=180,
        env={**os.environ, "OMLX_BENCH_KEY": endpoint.api_key},
    )
    if code != "200":
        raise RuntimeError(
            f"Docker containers cannot reach {endpoint.base_url} "
            f"(HTTP {code or 'no response'}). Set Settings → Server host to "
            "0.0.0.0 or configure an API key, then retry."
        )


def _parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _load_trials(job_root: Path) -> dict[str, tuple[Path, dict]]:
    """Finished trial results keyed by task name (latest retry wins)."""
    trials: dict[str, tuple[Path, dict]] = {}
    for path in job_root.glob("*/result.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not data.get("finished_at"):
            continue
        name = data.get("task_name")
        if not isinstance(name, str):
            continue
        prev = trials.get(name)
        if prev is None or str(data["finished_at"]) > str(prev[1]["finished_at"]):
            trials[name] = (path.parent, data)
    return trials


def _question_result(item: dict, trial: Optional[tuple[Path, dict]]) -> QuestionResult:
    task_id = item["id"]
    category = item.get("category")
    if trial is None:
        return QuestionResult(
            question_id=task_id, correct=False, expected="pass",
            predicted="not_run", time_seconds=0.0, category=category,
        )
    trial_dir, data = trial
    rewards = (data.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    if reward is None and len(rewards) == 1:
        reward = next(iter(rewards.values()))
    correct = reward is not None and float(reward) >= 1.0
    exc = data.get("exception_info") or {}
    if correct:
        predicted = "pass"
    elif reward is not None:
        predicted = "fail"
    elif exc:
        predicted = f"error: {exc.get('exception_type', 'Exception')}"
    else:
        predicted = "fail"
    started = _parse_time(data.get("started_at"))
    finished = _parse_time(data.get("finished_at"))
    elapsed = (finished - started).total_seconds() if started and finished else 0.0
    agent = data.get("agent_result") or {}
    return QuestionResult(
        question_id=task_id,
        correct=correct,
        expected="pass",
        predicted=predicted,
        time_seconds=elapsed,
        category=category,
        prompt_tokens=agent.get("n_input_tokens") or 0,
        completion_tokens=agent.get("n_output_tokens") or 0,
        error_message=str(exc.get("exception_message") or "")[:_LOG_TAIL_CHARS],
        artifact_path=str(trial_dir),
    )


class HarborBenchmark:
    """Runs one Harbor dataset with the built-in pi agent."""

    name: str = ""
    dataset: str = ""
    task_list_file: str = ""
    dataset_total: Optional[int] = None
    # Container platform override; None keeps the engine's native platform.
    platform: Optional[str] = None

    def __init__(self) -> None:
        self.endpoint: Optional[AgentEndpoint] = None
        self.jobs_dir: Optional[Path] = None
        self.job_name: str = ""
        self.agent_timeout_multiplier: float = 1.0

    def configure_agent_run(
        self,
        *,
        endpoint: AgentEndpoint,
        jobs_dir: Path,
        job_name: str,
        agent_timeout_multiplier: float,
    ) -> None:
        self.endpoint = endpoint
        self.jobs_dir = jobs_dir
        self.job_name = job_name
        self.agent_timeout_multiplier = agent_timeout_multiplier

    @property
    def job_dir(self) -> Path:
        assert self.jobs_dir is not None
        return self.jobs_dir / self.job_name

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        items = load_jsonl(DATA_DIR / self.task_list_file)
        self.dataset_total = len(items)
        if sample_size == 0:
            return items
        return deterministic_sample(items, sample_size)

    def _command(self, items: list[dict], batch_size: int, compose: Path) -> list[str]:
        assert self.endpoint is not None and self.jobs_dir is not None
        argv = resolve_harbor_command() + [
            "run",
            "-d", self.dataset,
            "-a", HARBOR_AGENT,
            "-m", f"openai/{self.endpoint.model}",
            "--ak", f"model_api={HARBOR_MODEL_API}",
            "-e", "docker",
            "-n", str(batch_size),
            "--agent-timeout-multiplier", str(self.agent_timeout_multiplier),
            "--job-name", self.job_name,
            "-o", str(self.jobs_dir),
            "--extra-docker-compose", str(compose),
            "--allow-agent-host", DOCKER_HOST_ALIAS,
            "-y",
        ]
        if self.dataset_total is None or len(items) < self.dataset_total:
            for item in items:
                argv += ["-i", item["id"]]
        return argv

    async def run(
        self,
        engine: Any,
        items: list[dict],
        on_progress: Optional[Callable[[int, int], Any]] = None,
        batch_size: int = 1,
        sampling_kwargs: Optional[dict] = None,
        enable_thinking: bool = False,
    ) -> BenchmarkResult:
        """Run the sampled tasks through Harbor.

        ``engine``/``sampling_kwargs``/``enable_thinking`` are unused: pi
        reaches the model over HTTP and gets the server's model defaults.
        """
        if self.endpoint is None or self.jobs_dir is None:
            raise RuntimeError("configure_agent_run() must be called before run()")
        start_time = time.time()
        await check_docker()
        await check_endpoint(self.endpoint)

        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        compose = self.jobs_dir / f"{self.job_name}.compose.json"
        main: dict[str, Any] = {"extra_hosts": [f"{DOCKER_HOST_ALIAS}:host-gateway"]}
        if self.platform:
            main["platform"] = self.platform
        compose.write_text(json.dumps({"services": {"main": main}}))
        log_path = self.jobs_dir / f"{self.job_name}.log"
        env = harbor_env(
            OPENAI_API_KEY=self.endpoint.api_key,
            OPENAI_BASE_URL=self.endpoint.base_url,
            PYTHONUNBUFFERED="1",
        )
        argv = self._command(items, batch_size, compose)
        logger.info(f"{self.name}: launching Harbor job {self.job_name}")

        with open(log_path, "wb") as log:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=log, stderr=log, env=env, start_new_session=True,
            )
            try:
                reported = -1
                while True:
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(proc.wait()), _POLL_SECONDS
                        )
                        exited = True
                    except asyncio.TimeoutError:
                        exited = False
                    done = len(_load_trials(self.job_dir))
                    if done != reported and on_progress:
                        reported = done
                        await on_progress(min(done, len(items)), len(items))
                    if exited:
                        break
            except asyncio.CancelledError:
                await self._terminate(proc)
                raise

        trials = _load_trials(self.job_dir)
        if not trials and proc.returncode:
            tail = log_path.read_text(errors="replace")[-_LOG_TAIL_CHARS:]
            raise RuntimeError(f"Harbor exited with code {proc.returncode}: {tail}")

        results = [_question_result(item, trials.get(item["id"])) for item in items]
        correct = sum(1 for r in results if r.correct)
        return BenchmarkResult(
            benchmark_name=self.name,
            accuracy=correct / len(results) if results else 0.0,
            total_questions=len(results),
            correct_count=correct,
            time_seconds=time.time() - start_time,
            question_results=results,
            category_scores=compute_category_scores(results),
            thinking_used=False,
        )

    async def _terminate(self, proc: asyncio.subprocess.Process) -> None:
        """SIGINT Harbor's process group so it tears down containers."""
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGINT)
                await asyncio.wait_for(proc.wait(), _SIGINT_GRACE_SECONDS)
            except ProcessLookupError:
                pass
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
        await self._remove_job_containers()

    async def _remove_job_containers(self) -> None:
        """Remove containers Harbor left behind for this job's trials.

        Harbor names each compose project ``<trial dir, lowercased>__...``
        (e.g. ``shadow-relay__ongxeyv__verifier__trial``); only those match.
        """
        prefixes = tuple(
            f"{p.name.lower()}__" for p in self.job_dir.glob("*") if p.is_dir()
        )
        if not prefixes:
            return
        code, out = await _run_capture(
            [
                "docker", "ps", "-a", "--format",
                '{{.ID}} {{.Label "com.docker.compose.project"}}',
            ],
            timeout=30,
        )
        if code != 0:
            return
        ids = [
            cid
            for cid, _, project in (line.partition(" ") for line in out.splitlines())
            if project.startswith(prefixes)
        ]
        if ids:
            logger.info(f"{self.name}: removing {len(ids)} leftover container(s)")
            await _run_capture(["docker", "rm", "-f", *ids], timeout=120)


class TerminalBench4Benchmark(HarborBenchmark):
    name = "terminalbench_4"
    dataset = TERMINALBENCH_4_DATASET
    task_list_file = "terminalbench_4_tasks.jsonl"


class SWEBenchVerifiedBenchmark(HarborBenchmark):
    name = "swebench_verified"
    dataset = SWEBENCH_VERIFIED_DATASET
    task_list_file = "swebench_verified_tasks.jsonl"
    # SWE-bench images (swebench/sweb.eval.x86_64.*) are amd64-only; arm64
    # engines fail with "no match for platform" unless told to emulate.
    platform = "linux/amd64"
