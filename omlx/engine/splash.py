# SPDX-License-Identifier: Apache-2.0
"""Splash engine: serve a model through a Splash server process.

Splash (https://github.com/incoai/splash) runs Qwen3.8-27B and
Qwen3.6-35B-A3B on its own Metal engine with DFlash2 speculative decoding.
It is a separate program, so this engine does not load weights in-process:
``start()`` launches ``splash serve`` for the model on a private loopback port
and requests are forwarded to its OpenAI-compatible API. Reasoning comes back
inside ``<think>`` tags for oMLX's reasoning parser, and tool calls come back
structured.

Two builds are supported side by side: the official ``splash`` (M3 or later)
and the community ``splash-m1`` port for M1/M2 GPUs; see ``KNOWN_BUILDS``.

Splash resolves models by Hugging Face repository ID from the Hub cache, so a
model qualifies when its snapshot is already cached there (no surprise
downloads) and it is either an MLX affine 4-bit/group-64 checkpoint of a
supported family or a prebuilt Splash package (``manifest.json`` with a
``splash-packed`` format).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..exceptions import InvalidRequestError
from ..model_discovery import is_splash_package
from ..utils.proc_memory import (
    get_lifetime_max_phys_footprint,
    get_phys_footprint,
    list_process_group_pids,
)
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)

# Environment variable naming the splash executable, for installs that are
# not on PATH (a source checkout, for example).
SPLASH_PATH_ENV = "OMLX_SPLASH_PATH"
# Homebrew prefixes: the menu bar app starts with a minimal PATH.
_BREW_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")

STARTUP_TIMEOUT_SECONDS = 1800  # the first start prepares the model
STARTUP_POLL_SECONDS = 2
STOP_TIMEOUT_SECONDS = 60
WATCHDOG_POLL_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 3600
STATUS_POLL_SECONDS = 5
LOG_TAIL_LINES = 5

# Splash accepts top_k from 1 to 32 (server/protocol.py MAX_TOP_K) and
# contexts up to 256K tokens (``--max-context``).
SPLASH_MAX_TOP_K = 32
SPLASH_MAX_CONTEXT = 262144
NEUTRAL_REPETITION_PENALTY = 1.0

# oMLX reasoning effort -> Splash's levels (none, low, medium, xhigh).
_SPLASH_EFFORT = {
    "off": "none",
    "none": "none",
    "minimal": "low",
    "low": "low",
    "moderate": "medium",
    "medium": "medium",
    "high": "xhigh",
    "xhigh": "xhigh",
    "max": "xhigh",
    "maximum": "xhigh",
    "ultra": "xhigh",
}

# The text_config fields that identify each architecture Splash serves, as
# Splash's install/families.py states them.
SPLASH_FAMILIES: dict[str, dict[str, Any]] = {
    "Qwen3.8-27B": {
        "model_type": "qwen3_5_text",
        "max_position_embeddings": 262144,
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "vocab_size": 248320,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
    },
    "Qwen3.6-35B-A3B": {
        "model_type": "qwen3_5_moe_text",
        "max_position_embeddings": 262144,
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "vocab_size": 248320,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "num_experts_per_tok": 8,
    },
}


# -- Splash builds -------------------------------------------------------------

# The launchers oMLX looks for, in its default preference order. The M1 build
# is a community port of Splash to Apple7/8 GPUs (M1/M2), which the official
# build does not support; it installs side by side as ``splash-m1``.
KNOWN_BUILDS = (
    ("splash", "Splash"),
    ("splash-m1", "Splash M1"),
)
CUSTOM_BUILD = "custom"
CUSTOM_LABEL = f"Custom ({SPLASH_PATH_ENV})"
SPLASH_INSTALL_HINT = (
    "Splash is not installed: brew install incoai/tap/splash (M3 or later), "
    "or Splash M1 for M1/M2 Macs (https://github.com/paperniuk/splash/releases)"
)
HELP_TIMEOUT_SECONDS = 30
# Extra place install-m1.sh puts splash-m1 when Homebrew is absent.
_USER_BIN_DIR = Path.home() / ".local" / "bin"


@dataclass(frozen=True)
class SplashBuild:
    """One installed Splash launcher."""

    id: str
    label: str
    path: Path
    serve_options: frozenset[str]

    @property
    def loads_mlx_checkpoints(self) -> bool:
        # Splash 1.1 added loading upstream MLX checkpoints together with
        # --language-only; 1.0 serves prebuilt Splash packages only.
        return "--language-only" in self.serve_options


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _find_launcher(name: str) -> Path | None:
    found = shutil.which(name)
    if found:
        return Path(found)
    for directory in (*_BREW_BIN_DIRS, str(_USER_BIN_DIR)):
        candidate = Path(directory) / name
        if _executable(candidate):
            return candidate
    return None


@functools.lru_cache(maxsize=16)
def _serve_options_cached(path: str, _mtime_ns: int) -> frozenset[str]:
    try:
        result = subprocess.run(
            [path, "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=HELP_TIMEOUT_SECONDS,
            env=_splash_env(None),
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Splash: %s serve --help failed: %s", path, e)
        return frozenset()
    return frozenset(re.findall(r"--[a-z][a-z-]*", result.stdout))


def _serve_options(path: Path) -> frozenset[str]:
    """The options ``<launcher> serve`` accepts, read from its help once."""
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return frozenset()
    return _serve_options_cached(str(path), mtime)


def find_splash_builds() -> list[SplashBuild]:
    """Installed Splash builds: ``$OMLX_SPLASH_PATH``, then the known ones."""
    builds = []
    configured = os.environ.get(SPLASH_PATH_ENV)
    if configured and _executable(Path(configured).expanduser()):
        path = Path(configured).expanduser()
        builds.append(
            SplashBuild(CUSTOM_BUILD, CUSTOM_LABEL, path, _serve_options(path))
        )
    for build_id, label in KNOWN_BUILDS:
        path = _find_launcher(build_id)
        if path is not None:
            builds.append(SplashBuild(build_id, label, path, _serve_options(path)))
    return builds


@functools.lru_cache(maxsize=1)
def _is_m1_or_m2() -> bool:
    """Whether this Mac has an Apple7/8 GPU, which only the M1 build runs on."""
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return re.search(r"\bM[12]\b", brand) is not None


def select_splash_build(
    builds: list[SplashBuild], requested: str | None = None
) -> SplashBuild | None:
    """The build to run: the requested one, or the best installed one.

    Automatic selection takes ``$OMLX_SPLASH_PATH`` first, then the M1 build
    on M1/M2 Macs and the official build elsewhere.
    """
    by_id = {build.id: build for build in builds}
    if requested:
        return by_id.get(requested)
    order = [CUSTOM_BUILD, "splash", "splash-m1"]
    if _is_m1_or_m2():
        order = [CUSTOM_BUILD, "splash-m1", "splash"]
    return next((by_id[i] for i in order if i in by_id), None)


# -- model compatibility -------------------------------------------------------


def splash_family(config: dict) -> str | None:
    """The Splash family whose architecture ``config`` states, or None."""
    text = config.get("text_config") if isinstance(config, dict) else None
    if not isinstance(text, dict):
        return None
    for name, signature in SPLASH_FAMILIES.items():
        if all(text.get(key) == value for key, value in signature.items()):
            return name
    return None


def _is_affine_4bit_group64(config: dict) -> bool:
    quant = config.get("quantization")
    return (
        isinstance(quant, dict)
        and quant.get("mode", "affine") == "affine"
        and quant.get("bits") == 4
        and quant.get("group_size") == 64
    )


def splash_repo_id(model_path: str | Path, source_repo_id: str | None) -> str:
    """The repository ID Splash resolves the model by.

    Hub cache entries carry it; a model folder laid out as
    ``<owner>/<repo>`` (how oMLX and LM Studio download) names it.
    """
    if source_repo_id:
        return source_repo_id
    path = Path(model_path)
    return f"{path.parent.name}/{path.name}"


def _hf_hub_cache() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def is_in_hf_cache(repo_id: str) -> bool:
    """Whether the Hub cache holds a snapshot of ``repo_id``."""
    folder = _hf_hub_cache() / ("models--" + repo_id.replace("/", "--"))
    snapshots = folder / "snapshots"
    try:
        return any(child.is_dir() for child in snapshots.iterdir())
    except OSError:
        return False


def resolve_splash_build(
    model_path: str | Path,
    source_repo_id: str | None = None,
    build_id: str | None = None,
) -> tuple[SplashBuild | None, str]:
    """The Splash build that can serve ``model_path``.

    Returns:
        (build, reason). ``build`` is None and ``reason`` says why when no
        installed build can serve the model.
    """
    path = Path(model_path)
    package = is_splash_package(path)
    if not package:
        try:
            config = json.loads((path / "config.json").read_text())
        except (OSError, ValueError) as e:
            return None, f"failed to read config.json: {e}"
        if not isinstance(config, dict) or splash_family(config) is None:
            supported = ", ".join(SPLASH_FAMILIES)
            return None, f"Splash serves {supported} only"
        if not _is_affine_4bit_group64(config):
            return None, "Splash needs an MLX affine 4-bit/group-64 checkpoint"
    repo_id = splash_repo_id(path, source_repo_id)
    if not is_in_hf_cache(repo_id):
        return None, (
            f"{repo_id} is not in the Hugging Face cache; Splash loads models "
            "from there"
        )
    builds = find_splash_builds()
    build = select_splash_build(builds, build_id)
    if build is None:
        if build_id and builds:
            return None, f"Splash build '{build_id}' is not installed"
        return None, SPLASH_INSTALL_HINT
    if not package and not build.loads_mlx_checkpoints:
        return None, (
            f"{build.label} serves Splash packages only; MLX checkpoints need "
            "Splash 1.1 or later"
        )
    return build, ""


def is_splash_compatible(
    model_path: str | Path,
    source_repo_id: str | None = None,
    build_id: str | None = None,
) -> tuple[bool, str]:
    """Decide whether an installed Splash build can serve ``model_path``.

    Returns:
        (is_compatible, reason). ``reason`` is empty when compatible.
    """
    build, reason = resolve_splash_build(model_path, source_repo_id, build_id)
    return build is not None, reason


# -- request translation -------------------------------------------------------


def splash_effort(chat_template_kwargs: dict | None) -> str | None:
    kwargs = chat_template_kwargs or {}
    if kwargs.get("enable_thinking") is False:
        return "none"
    effort = kwargs.get("reasoning_effort")
    if effort is None:
        return None
    return _SPLASH_EFFORT.get(str(effort).lower())


def splash_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-format messages for Splash.

    oMLX hands engines template-format tool calls, whose arguments may be a
    dict; the OpenAI API Splash implements wants a JSON string.
    """
    converted = []
    for message in messages:
        calls = message.get("tool_calls")
        if not calls:
            converted.append(message)
            continue
        fixed = []
        for call in calls:
            function = dict(call.get("function") or {})
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                function["arguments"] = json.dumps(arguments or {})
            fixed.append({**call, "type": "function", "function": function})
        converted.append({**message, "tool_calls": fixed})
    return converted


@dataclass
class _LiveRequest:
    """One in-flight Splash request, as the admin dashboard shows it.

    Times are time.monotonic(); prefill figures come from Splash's
    prompt_progress chunks.
    """

    started_at: float
    max_tokens: int | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    processed_tokens: int = 0
    prefill_ms: float = 0.0
    first_token_at: float | None = None
    last_token_at: float | None = None
    generated_tokens: int = 0

    def add_progress(self, progress: dict[str, Any]) -> None:
        self.prompt_tokens = int(progress.get("total", self.prompt_tokens))
        self.cached_tokens = int(progress.get("cache", self.cached_tokens))
        self.processed_tokens = int(progress.get("processed", self.processed_tokens))
        self.prefill_ms = float(progress.get("time_ms", self.prefill_ms))

    def add_tokens(self, count: int, now: float) -> None:
        self.first_token_at = self.first_token_at or now
        self.last_token_at = now
        self.generated_tokens += count

    def prefill_row(self, request_id: str, now: float) -> dict[str, Any]:
        # Tokens reused from Splash's prefix cache cost no prefill time.
        computed = self.processed_tokens - self.cached_tokens
        speed = computed / (self.prefill_ms / 1000) if self.prefill_ms > 0 else 0.0
        remaining = self.prompt_tokens - self.processed_tokens
        return {
            "request_id": request_id,
            "processed": self.processed_tokens,
            "total": self.prompt_tokens,
            "speed": speed,
            "eta": remaining / speed if speed > 0 else None,
            "elapsed": now - self.started_at,
            "detail": None,
        }

    def generating_row(self, request_id: str, now: float) -> dict[str, Any]:
        elapsed = max(0.0, now - self.first_token_at)
        return {
            "request_id": request_id,
            "elapsed_seconds": elapsed,
            "generated_tokens": self.generated_tokens,
            "tokens_per_second": self.generated_tokens / elapsed if elapsed else 0.0,
            "last_activity_age_seconds": max(0.0, now - self.last_token_at),
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
        }


class _StreamState:
    """Accumulates one Splash response stream into oMLX outputs."""

    def __init__(self) -> None:
        self.text = ""
        self.in_think = False
        self.calls: dict[int, dict[str, str]] = {}
        self.usage: dict[str, Any] = {}
        self.finish_reason = "stop"
        self.first_token_at: float | None = None

    def add_choice(self, choice: dict[str, Any]) -> tuple[str, bool]:
        """Apply one streamed choice; returns (new text, produced output)."""
        self.finish_reason = choice.get("finish_reason") or self.finish_reason
        delta = choice.get("delta") or {}
        piece = ""
        if delta.get("reasoning_content"):
            piece += ("" if self.in_think else "<think>") + delta["reasoning_content"]
            self.in_think = True
        if delta.get("content"):
            piece += ("</think>" if self.in_think else "") + delta["content"]
            self.in_think = False
        for call in delta.get("tool_calls") or []:
            slot = self.calls.setdefault(
                call.get("index", 0),
                {"id": call.get("id") or "", "name": "", "arguments": ""},
            )
            function = call.get("function") or {}
            slot["name"] += function.get("name") or ""
            slot["arguments"] += function.get("arguments") or ""
        self.text += piece
        return piece, bool(piece or delta.get("tool_calls"))

    def close_think(self) -> str:
        closing = "</think>" if self.in_think else ""
        self.in_think = False
        self.text += closing
        return closing

    def tool_calls(self) -> list[dict[str, str]] | None:
        # oMLX's parser format: flat {"id", "name", "arguments"} dicts.
        calls = [
            {
                "id": slot["id"],
                "name": slot["name"],
                "arguments": slot["arguments"] or "{}",
            }
            for _, slot in sorted(self.calls.items())
        ]
        return calls or None


# -- process management ----------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _signal_group(group: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError):  # the group already exited
        os.killpg(group, sig)


def _splash_env(api_key: str | None, tmpdir: Path | None = None) -> dict[str, str]:
    """The environment for ``splash serve``.

    oMLX's bundled interpreter sets PYTHONHOME/PYTHONPATH, which would point
    Splash's own Python at the wrong standard library. The API key travels
    in the environment so it never shows up in the process list. Setting
    ``tmpdir`` directs Splash's disk cache to that directory via ``TMPDIR``.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONHOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE")
    }
    path = env.get("PATH", "/usr/bin:/bin").split(":")
    env["PATH"] = ":".join([d for d in _BREW_BIN_DIRS if d not in path] + path)
    if api_key is not None:
        env["SPLASH_API_KEY"] = api_key
    if tmpdir is not None:
        env["TMPDIR"] = str(tmpdir)
    return env


def _watchdog_command(command: list[str]) -> list[str]:
    """Wrap ``command`` in a shell that leads its process group.

    The shell stops the whole group when oMLX (its parent) or Splash exits,
    so a crashed or force-killed oMLX never leaves Splash holding memory.
    """
    script = (
        'parent=$1; shift; "$@" & child=$!; '
        'while kill -0 "$parent" 2>/dev/null && kill -0 "$child" 2>/dev/null; '
        f"do sleep {WATCHDOG_POLL_SECONDS}; done; "
        'if kill -0 "$child" 2>/dev/null; then trap "" TERM; kill -TERM 0; fi; '
        'wait "$child"'
    )
    return ["/bin/sh", "-c", script, "splash-watchdog", str(os.getpid()), *command]


def _log_dir() -> Path:
    try:
        from ..settings import get_settings

        settings = get_settings()
        return settings.logging.get_log_dir(settings.base_path)
    except Exception:
        return Path.home() / ".omlx" / "logs"


def _failure_summary(path: Path, offset: int) -> str:
    """Why a Splash start failed, from what it logged since ``offset``.

    Splash prefixes its own failures with ``error:``; the launcher's setup
    output (dependency installs, downloads) around them is noise here.
    """
    try:
        with open(path, "rb") as log:
            log.seek(offset)
            lines = log.read().decode(errors="replace").splitlines()
    except OSError:
        return ""
    lines = [line.strip() for line in lines if line.strip()]
    errors = [line for line in lines if line.lower().startswith("error")]
    return "\n".join((errors or lines)[-LOG_TAIL_LINES:])


# -- engine -------------------------------------------------------------------------


class SplashEngine(BaseEngine):
    """Serves one model through a ``splash serve`` child process."""

    # Splash budgets its own Metal memory (--max-memory); there is no oMLX
    # scheduler here for the process memory enforcer to update.
    _prefill_memory_guard_managed_externally = True
    # Splash validates and enforces response_format in its own engine.
    supports_native_response_format = True

    def __init__(
        self,
        model_name: str,
        build: SplashBuild,
        source_repo_id: str | None = None,
        max_context: int | None = None,
        cache_disk_bytes: int = 0,
        cache_disk_dir: Path | None = None,
    ) -> None:
        self._path = Path(model_name)
        self._build = build
        self._repo_id = splash_repo_id(self._path, source_repo_id)
        self._max_context = max_context
        # Builds before 1.1 have no disk tier.
        self._cache_disk_bytes = (
            cache_disk_bytes if "--max-cache-disk" in build.serve_options else 0
        )
        self._cache_disk_dir = cache_disk_dir
        self._process: subprocess.Popen | None = None
        self._client: httpx.AsyncClient | None = None
        self._base_url = ""
        self._api_key = secrets.token_urlsafe(32)
        self._log_path = (
            _log_dir() / "splash" / (self._repo_id.replace("/", "--") + ".log")
        )
        self._log_offset = 0
        self._active = 0
        self._requests = 0
        self._live: dict[str, _LiveRequest] = {}
        self._weights_size: int | None = None
        self._warned: set[str] = set()
        self._tokenizer: Any = None
        self._model_type: str | None = None
        self._status: dict[str, Any] | None = None
        self._status_task: asyncio.Task | None = None
        self._status_refresh: asyncio.Task | None = None

    # -- identity ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._repo_id

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model_type(self) -> str | None:
        return self._model_type

    def has_active_requests(self) -> bool:
        return self._active > 0

    # -- lifecycle -----------------------------------------------------------------

    def _tokenizer_dir(self) -> Path:
        # Splash packages keep the tokenizer in tokenizer/; MLX checkpoints
        # keep it beside the weights.
        nested = self._path / "tokenizer"
        return nested if (nested / "tokenizer_config.json").is_file() else self._path

    def _load_tokenizer(self) -> None:
        from transformers import AutoTokenizer

        directory = self._tokenizer_dir()
        self._tokenizer = AutoTokenizer.from_pretrained(str(directory))
        try:
            config = json.loads((directory / "config.json").read_text())
            self._model_type = config.get("model_type")
        except (OSError, ValueError):
            self._model_type = None

    def _serve_command(self, port: int) -> list[str]:
        """``splash serve`` with the options this build accepts."""
        options = self._build.serve_options
        command = [
            str(self._build.path),
            "serve",
            "--model",
            self._repo_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        if "--no-webui" in options:
            command.append("--no-webui")
        if "--language-only" in options and not is_splash_package(self._path):
            # Requests arrive as text: skip loading the vision tower. Splash
            # takes this option for upstream checkpoints only.
            command.append("--language-only")
        if self._max_context and "--max-context" in options:
            limit = min(int(self._max_context), SPLASH_MAX_CONTEXT)
            command += ["--max-context", str(limit)]
        if self._cache_disk_bytes:
            command += ["--max-cache-disk", str(self._cache_disk_bytes)]
        return command

    async def start(self) -> None:
        await asyncio.to_thread(self._load_tokenizer)
        port = _free_port()
        self._base_url = f"http://127.0.0.1:{port}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        command = self._serve_command(port)
        tmpdir = self._cache_disk_dir if self._cache_disk_bytes else None
        if tmpdir is not None:
            tmpdir.mkdir(parents=True, exist_ok=True)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Splash: starting %s (log: %s)", " ".join(command), self._log_path)
        with open(self._log_path, "ab") as log:
            self._log_offset = log.tell()
            self._process = subprocess.Popen(
                _watchdog_command(command),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=_splash_env(self._api_key, tmpdir=tmpdir),
                start_new_session=True,
            )
        try:
            await self._wait_until_ready()
            self._status_task = asyncio.create_task(self._poll_status())
        except BaseException:
            await self.stop()
            raise

    async def _refresh_status(self) -> None:
        """Keep the last /status snapshot for the synchronous dashboard getters."""
        if self._client is None:
            return
        try:
            response = await self._client.get("/status", timeout=2)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                self._status = data
        except Exception:
            logger.debug("Splash: failed to fetch /status", exc_info=True)

    async def _poll_status(self) -> None:
        while True:
            await self._refresh_status()
            await asyncio.sleep(STATUS_POLL_SECONDS)

    async def _serves_model(self) -> bool:
        try:
            response = await self._client.get("/v1/models", timeout=2)
            models = response.json()
        except (httpx.HTTPError, ValueError):
            return False
        return any(m.get("id") == self._repo_id for m in models.get("data", []))

    async def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"Splash exited with code {self._process.returncode} "
                    f"while loading {self._repo_id}: "
                    f"{_failure_summary(self._log_path, self._log_offset)} "
                    f"(log: {self._log_path})"
                )
            if await self._serves_model():
                logger.info("Splash: serving %s at %s", self._repo_id, self._base_url)
                return
            await asyncio.sleep(STARTUP_POLL_SECONDS)
        raise RuntimeError(
            f"Splash did not serve {self._repo_id} within "
            f"{STARTUP_TIMEOUT_SECONDS}s; see {self._log_path}"
        )

    async def stop(self) -> None:
        if self._status_task is not None:
            self._status_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._status_task
            self._status_task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        group = process.pid  # the watchdog leads Splash's process group
        _signal_group(group, signal.SIGTERM)
        try:
            await asyncio.to_thread(process.wait, STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning("Splash: server ignored SIGTERM; killing it")
            _signal_group(group, signal.SIGKILL)
            await asyncio.to_thread(process.wait)

    # -- requests ------------------------------------------------------------------

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning("Splash: %s", message)

    def _request_body(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        presence_penalty: float,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Map oMLX sampling options onto the subset Splash accepts.

        Splash rejects min_p, presence/frequency penalties and repetition
        penalty, and caps top_k at 32: those are dropped or clamped with a
        one-time warning rather than failing the request.
        """
        body: dict[str, Any] = {
            "model": self._repo_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_k and top_k > 0:
            body["top_k"] = min(int(top_k), SPLASH_MAX_TOP_K)
            if top_k > SPLASH_MAX_TOP_K:
                self._warn_once("top_k", f"top_k {top_k} clamped to {SPLASH_MAX_TOP_K}")
        ignored = {
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": kwargs.get("frequency_penalty"),
            "thinking_budget": kwargs.get("thinking_budget"),
        }
        for name, value in ignored.items():
            if value:
                self._warn_once(name, f"{name}={value} is unsupported and ignored")
        if repetition_penalty not in (None, NEUTRAL_REPETITION_PENALTY):
            self._warn_once(
                "repetition_penalty", "repetition_penalty is unsupported and ignored"
            )
        if kwargs.get("stop"):
            body["stop"] = kwargs["stop"]
        if kwargs.get("seed") is not None:
            body["seed"] = kwargs["seed"]
        if kwargs.get("response_format"):
            body["response_format"] = kwargs["response_format"]
        return body

    async def _stream(self, body: dict[str, Any]) -> AsyncIterator[GenerationOutput]:
        """POST a streaming chat request to Splash and yield oMLX outputs.

        Timestamps use time.perf_counter(), the clock oMLX measures time to
        first token and generation speed with.
        """
        if self._client is None:
            raise RuntimeError("Splash engine is not started")
        body = {
            **body,
            "stream": True,
            "return_progress": True,
            "stream_options": {"include_usage": True},
        }
        state = _StreamState()
        request_id = uuid.uuid4().hex
        live = _LiveRequest(time.monotonic(), body.get("max_tokens"))
        self._live[request_id] = live
        self._active += 1
        self._requests += 1
        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=body
            ) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode(errors="replace")
                    if 400 <= response.status_code < 500:
                        raise InvalidRequestError(f"Splash: {detail[:500]}")
                    raise RuntimeError(
                        f"Splash returned HTTP {response.status_code}: {detail[:500]}"
                    )
                async for chunk in self._chunks(response):
                    state.usage = chunk.get("usage") or state.usage
                    if isinstance(chunk.get("prompt_progress"), dict):
                        live.add_progress(chunk["prompt_progress"])
                    for choice in chunk.get("choices") or []:
                        piece, produced = state.add_choice(choice)
                        if not produced:
                            continue
                        live.add_tokens(self._count_tokens(piece), time.monotonic())
                        now = time.perf_counter()
                        state.first_token_at = state.first_token_at or now
                        yield GenerationOutput(
                            text=state.text,
                            new_text=piece,
                            finished=False,
                            first_token_at=state.first_token_at,
                            generated_at=now,
                        )
            yield self._final_output(state)
        finally:
            self._live.pop(request_id, None)
            self._active -= 1
            # Pick up the request's cache counters without waiting for the
            # next poll; one refresh in flight is enough.
            if self._client is not None and (
                self._status_refresh is None or self._status_refresh.done()
            ):
                self._status_refresh = asyncio.create_task(self._refresh_status())

    @staticmethod
    async def _chunks(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError(f"Splash error: {chunk['error']}")
            yield chunk

    @staticmethod
    def _final_output(state: _StreamState) -> GenerationOutput:
        closing = state.close_think()
        details = state.usage.get("prompt_tokens_details") or {}
        return GenerationOutput(
            text=state.text,
            new_text=closing,
            finished=True,
            finish_reason=state.finish_reason,
            prompt_tokens=state.usage.get("prompt_tokens", 0),
            completion_tokens=state.usage.get("completion_tokens", 0),
            cached_tokens=details.get("cached_tokens", 0),
            tool_calls=state.tool_calls(),
            first_token_at=state.first_token_at,
            generated_until=time.perf_counter(),
        )

    @staticmethod
    async def _collect(outputs: AsyncIterator[GenerationOutput]) -> GenerationOutput:
        final = None
        async for output in outputs:
            final = output
        final.new_text = final.text
        return final

    # -- chat ----------------------------------------------------------------------

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        body = self._request_body(
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
            kwargs,
        )
        body["messages"] = splash_messages(messages)
        if tools:
            body["tools"] = tools
        effort = splash_effort(kwargs.get("chat_template_kwargs"))
        if effort:
            body["reasoning_effort"] = effort
        async for output in self._stream(body):
            yield output

    async def chat(self, messages: list[dict[str, Any]], **kwargs) -> GenerationOutput:
        return await self._collect(self.stream_chat(messages, **kwargs))

    def count_chat_tokens(
        self, messages, tools=None, chat_template_kwargs=None, is_partial=None
    ) -> int:
        template = dict(chat_template_kwargs or {})
        attempts = (
            template,
            {k: v for k, v in template.items() if k != "reasoning_effort"},
        )
        for attempt in attempts:
            try:
                ids = self._tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **attempt,
                )
                # Newer transformers return a BatchEncoding, not a list of ids.
                return len(ids["input_ids"] if hasattr(ids, "keys") else ids)
            except Exception:  # an effort value this template rejects
                logger.debug("Splash: chat template token count failed", exc_info=True)
        return sum(
            len(self._tokenizer.encode(str(m.get("content", "")))) for m in messages
        )

    # -- plain completions -----------------------------------------------------------

    async def stream_generate(
        self, prompt: str | list[int], **kwargs
    ) -> AsyncIterator[GenerationOutput]:
        # Splash's API serves chat, responses and messages; it has no plain
        # text completion endpoint.
        raise InvalidRequestError(
            "Splash models serve chat requests only; use /v1/chat/completions, "
            "/v1/responses or /v1/messages"
        )
        yield  # pragma: no cover - makes this an async generator

    async def generate(self, prompt: str | list[int], **kwargs) -> GenerationOutput:
        return await self._collect(self.stream_generate(prompt, **kwargs))

    # -- stats -----------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_type": "splash",
            "splash_build": self._build.label,
            "model_name": self._repo_id,
            "server": self._base_url,
            "running": self._process is not None and self._process.poll() is None,
            "requests": self._requests,
            "active_requests": self._active,
            "cache_disk_bytes": self._cache_disk_bytes,
            "process_memory_bytes": self.process_memory_bytes(),
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        return None

    def get_runtime_cache_stats(self) -> dict[str, Any] | None:
        """Splash's cache counters in the shape the cache dashboard reads.

        Mapped from the last /status snapshot: prefix hits and cold misses,
        KV pages holding cached prefixes, and the disk tier. oMLX's hot-cache
        columns stay 0; that budget is oMLX's own.
        """
        if self._status is None:
            return None
        kv, state, disk, cache = (
            self._status.get(key) or {} for key in ("kv", "state", "disk", "cache")
        )
        block_size = int(kv.get("block_tokens") or self._status.get("block_tokens", 0))
        hits = int(cache.get("hits", 0))
        misses = int(cache.get("cold_misses", 0))
        evictions = int(state.get("evictions", 0))
        restores = int(disk.get("kv_restores", 0))
        demotions = int(disk.get("kv_demotions", 0))
        used = int(disk.get("used_bytes", 0))
        return {
            "block_size": block_size,
            "indexed_blocks": int(kv.get("pages_cache", 0)),
            "ssd_cache": {
                "num_files": 1 if used else 0,
                "total_size_bytes": used,
                "max_size_bytes": int(disk.get("capacity_bytes", 0)),
                "hits": hits,
                "misses": misses,
                "evictions": evictions,
                "loads": restores,
                "saves": demotions,
            },
            "prefix_cache": {"block_size": block_size},
            "cache_rates": {
                "cumulative": {
                    "prefix_hits": hits,
                    "prefix_misses": misses,
                    "evictions": evictions,
                    "ssd_disk_loads": restores,
                    "ssd_saves": demotions,
                }
            },
            "splash_cache": {
                "reused_tokens": int(cache.get("reused_tokens", 0)),
                "kv_hit_tokens": int(cache.get("kv_hit_tokens", 0)),
                "kv_disk_hit_tokens": int(cache.get("kv_disk_hit_tokens", 0)),
                "state_entries": int(state.get("entries", 0)),
                "state_bytes": int(state.get("bytes", 0)),
            },
        }

    # -- benchmark and memory --------------------------------------------------------

    def benchmark_endpoint(self) -> dict[str, Any]:
        """Splash's own API, for the admin benchmark's chat-endpoint path.

        Thinking is off so the benchmark measures plain prefill and decode.
        """
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Splash engine is not started")
        return {
            "base_url": f"{self._base_url}/v1",
            "api_key": self._api_key,
            "model": self._repo_id,
            "extra_body": {"reasoning_effort": "none"},
        }

    def _group_pids(self) -> list[int]:
        if self._process is None or self._process.poll() is not None:
            return []
        return list_process_group_pids(self._process.pid)

    def _weights_bytes(self) -> int:
        if self._weights_size is None:
            files = (f for f in self._path.rglob("*") if f.is_file())
            self._weights_size = sum(f.stat().st_size for f in files)
        return self._weights_size

    def _reported_memory(self, key: str) -> int:
        # Splash's own Metal accounting, which includes the mapped weights.
        memory = (self._status or {}).get("memory_actual") or {}
        return int(memory.get(key, 0))

    def process_memory_bytes(self) -> int:
        """Memory the running Splash holds.

        Splash 1.1 maps its weights from disk, and the kernel does not charge
        those pages to the process, so its footprint alone undercounts. The
        Metal memory Splash reports covers them; before the first status
        snapshot, the weights' size on disk stands in.
        """
        pids = self._group_pids()
        if not pids:
            return 0
        footprint = sum(get_phys_footprint(pid) for pid in pids)
        floor = self._reported_memory("current_bytes") or self._weights_bytes()
        return max(footprint, floor)

    def peak_memory_bytes(self) -> int:
        """The highest memory Splash has held so far, measured the same way."""
        pids = self._group_pids()
        if not pids:
            return 0
        peak = max(get_lifetime_max_phys_footprint(pid) for pid in pids)
        floor = self._reported_memory("peak_bytes") or self._weights_bytes()
        return max(peak, floor)

    def _count_tokens(self, text: str) -> int:
        # Splash streams several tokens per chunk (speculative decoding), so
        # count the text rather than the chunks.
        if not text or self._tokenizer is None:
            return 1
        return max(1, len(self._tokenizer.encode(text, add_special_tokens=False)))

    def get_live_requests(self) -> dict[str, list[dict[str, Any]]]:
        """Rows for the admin dashboard's active-models card.

        The same prefilling/generating rows scheduler-backed engines produce,
        built from Splash's prompt_progress chunks and the streamed tokens.
        """
        now = time.monotonic()
        live = list(self._live.items())
        return {
            "prefilling": [
                req.prefill_row(rid, now)
                for rid, req in live
                if req.first_token_at is None
            ],
            "generating": [
                req.generating_row(rid, now)
                for rid, req in live
                if req.first_token_at is not None
            ],
        }
