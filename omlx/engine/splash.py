# SPDX-License-Identifier: Apache-2.0
"""Splash engine: serve a model through a Splash server process.

Splash (https://github.com/incoai/splash) runs Qwen3.8-27B and
Qwen3.6-35B-A3B on its own Metal engine with DFlash2 speculative decoding.
It is a separate program, so this engine does not load weights in-process.
``start()`` stages the oMLX model folder as an entry of a private Hugging Face
cache, launches ``splash serve --offline`` for it on a loopback port and
forwards requests to its OpenAI-compatible API. Reasoning comes back inside
``<think>`` tags for oMLX's reasoning parser, and tool calls come back
structured.

Two builds are supported: the official ``splash`` (M3 or later) and the
community ``splash-m1`` port for M1/M2 GPUs.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
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
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer

from ..exceptions import InvalidRequestError
from ..utils.hardware import get_chip_name, parse_chip_info
from ..utils.proc_memory import get_phys_footprint, list_process_group_pids
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)

# External engines a model can be served by, as model settings name them.
EXTERNAL_ENGINES = {"splash": "Splash"}

# Homebrew prefixes: the menu bar app starts with a minimal PATH.
_BREW_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")
# Where install-m1.sh puts splash-m1 when Homebrew is absent.
_USER_BIN_DIR = Path.home() / ".local" / "bin"

STARTUP_TIMEOUT_SECONDS = 600  # the first start hashes the model files once
STARTUP_POLL_SECONDS = 2
STOP_TIMEOUT_SECONDS = 60
STOP_POLL_SECONDS = 0.2
WATCHDOG_POLL_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 3600
STATUS_POLL_SECONDS = 5
VERSION_TIMEOUT_SECONDS = 30
# A launcher that failed to report its version is asked again after this.
VERSION_RETRY_SECONDS = 60
LOG_TAIL_LINES = 5

# 1.3.0 is the first release that loads MLX checkpoints from a local
# snapshot offline with a local DFlash2 draft; 1.3.1 added mixed-bit ones.
MIN_SPLASH_VERSION = (1, 3, 0)
MIXED_QUANTIZATION_VERSION = (1, 3, 1)
# Splash reads a per-request thinking_token_budget from 1.3.2 on.
THINKING_BUDGET_VERSION = (1, 3, 2)
SPLASH_MAX_CONTEXT = 262144
SPLASH_INSTALL_HINT = (
    "Splash is not installed: brew install incoai/tap/splash (M3 or later), "
    "or splash-m1 for M1/M2 Macs (https://github.com/paperniuk/splash/releases)"
)

# Splash's reasoning levels, and the oMLX spellings that map onto them.
_SPLASH_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_EFFORT_ALIASES = {"off": "none", "moderate": "medium", "maximum": "max"}

# The text_config fields that identify each architecture Splash serves.
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
_REQUIRED_MODEL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")

# Splash quantizes the BF16 DFlash2 draft to 4 bits when it loads it.
_DRAFT_LOAD_RATIO = 1 / 3
# Scratch Splash plans beside the weights: prefill, decode and lane states.
_RUNTIME_RESERVE_BYTES = 3 * 1024**3
# Content parts Splash takes beside text: images (it also reads PDFs, which
# oMLX's request models do not carry).
_SPLASH_MEDIA_TYPES = frozenset({"image_url", "input_image", "image"})
_UNSUPPORTED_MEDIA_TYPES = frozenset({"video_url", "input_audio", "audio_url"})


# -- Splash binary -------------------------------------------------------------


@dataclass(frozen=True)
class SplashBinary:
    """An installed Splash launcher that oMLX can run."""

    path: Path
    version: str
    auto_detected: bool

    @property
    def version_tuple(self) -> tuple[int, int, int]:
        return _version_tuple(self.version) or (0, 0, 0)

    @property
    def label(self) -> str:
        if self.path.name == "splash-m1" or "-m1" in self.version:
            return "Splash M1"
        return "Splash"

    @property
    def loads_mixed_quantization(self) -> bool:
        return self.version_tuple >= MIXED_QUANTIZATION_VERSION

    @property
    def accepts_thinking_budget(self) -> bool:
        return self.version_tuple >= THINKING_BUDGET_VERSION


def _version_tuple(version: str) -> tuple[int, int, int] | None:
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


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


# (path, mtime) -> (version or None, time.monotonic() of the check)
_versions: dict[tuple[str, int], tuple[str | None, float]] = {}


def _read_version(path: str) -> str | None:
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=VERSION_TIMEOUT_SECONDS,
            env=_splash_env(None),
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Splash: %s --version failed: %s", path, e)
        return None
    match = re.search(r"\d+\.\d+\.\d+\S*", result.stdout + result.stderr)
    return match.group(0) if match else None


def _splash_version(path: Path) -> str | None:
    """The version ``<launcher> --version`` reports, read once per file.

    A failed read is retried after VERSION_RETRY_SECONDS: a first run may
    still be setting the launcher up.
    """
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return None
    cached = _versions.get(key)
    now = time.monotonic()
    if cached is not None and (
        cached[0] is not None or now - cached[1] < VERSION_RETRY_SECONDS
    ):
        return cached[0]
    version = _read_version(str(path))
    _versions[key] = (version, now)
    return version


def inspect_splash_binary(
    path: Path, *, auto_detected: bool
) -> tuple[SplashBinary | None, str]:
    """Check that ``path`` is a Splash launcher oMLX supports.

    Returns:
        (binary, reason). ``binary`` is None and ``reason`` says why when
        ``path`` cannot be used.
    """
    if not _executable(path):
        return None, f"{path} is not an executable file"
    version = _splash_version(path)
    if version is None or _version_tuple(version) is None:
        return None, f"{path} did not report a Splash version"
    if _version_tuple(version) < MIN_SPLASH_VERSION:
        minimum = ".".join(str(part) for part in MIN_SPLASH_VERSION)
        return None, f"Splash {version} is too old; oMLX needs {minimum} or later"
    return SplashBinary(path, version, auto_detected), ""


@functools.lru_cache(maxsize=1)
def _is_m1_or_m2() -> bool:
    """Whether this Mac has an M1/M2 GPU, which only splash-m1 supports."""
    chip, _ = parse_chip_info(get_chip_name())
    return chip in ("M1", "M2")


def find_splash_binary(configured: str | None) -> tuple[SplashBinary | None, str]:
    """The Splash launcher to run: ``configured``, or the one found installed.

    Automatic detection prefers splash-m1 on M1/M2 Macs and the official
    build elsewhere, falling back to whichever is installed.
    """
    if configured:
        return inspect_splash_binary(Path(configured).expanduser(), auto_detected=False)
    names = ("splash-m1", "splash") if _is_m1_or_m2() else ("splash", "splash-m1")
    reason = SPLASH_INSTALL_HINT
    for name in names:
        path = _find_launcher(name)
        if path is None:
            continue
        binary, reason = inspect_splash_binary(path, auto_detected=True)
        if binary is not None:
            return binary, ""
    return None, reason


# -- model compatibility -------------------------------------------------------


@functools.lru_cache(maxsize=256)
def _read_json_cached(path: str, _mtime_ns: int) -> dict | None:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_config(model_path: Path) -> dict | None:
    path = model_path / "config.json"
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    return _read_json_cached(str(path), mtime)


def splash_family(config: dict) -> str | None:
    """The Splash family whose architecture ``config`` states, or None."""
    text = config.get("text_config")
    if not isinstance(text, dict):
        return None
    for name, signature in SPLASH_FAMILIES.items():
        if all(text.get(key) == value for key, value in signature.items()):
            return name
    return None


def splash_has_vision(config: dict) -> bool:
    """Whether the checkpoint carries a vision tower Splash can load."""
    return isinstance(config.get("vision_config"), dict)


def _is_affine_4bit_group64(quant: dict) -> bool:
    layers = [quant, *(value for value in quant.values() if isinstance(value, dict))]
    return all(
        layer.get("mode", "affine") == "affine"
        and layer.get("bits") == 4
        and layer.get("group_size") == 64
        for layer in layers
    )


def _quantization_problem(config: dict, binary: SplashBinary) -> str:
    quant = config.get("quantization")
    if not isinstance(quant, dict):
        return "Splash needs a quantized MLX checkpoint"
    # Splash validates the bit widths and modules it accepts itself.
    if binary.loads_mixed_quantization or _is_affine_4bit_group64(quant):
        return ""
    minimum = ".".join(str(part) for part in MIXED_QUANTIZATION_VERSION)
    return (
        f"{binary.label} {binary.version} serves affine 4-bit/group-64 "
        f"checkpoints only; Splash {minimum} also serves mixed-bit ones"
    )


def _is_splash_draft(draft: dict, text: dict) -> bool:
    architectures = draft.get("architectures") or []
    return (
        "DFlash2DraftModel" in architectures
        and draft.get("hidden_size") == text.get("hidden_size")
        and draft.get("vocab_size") == text.get("vocab_size")
        and draft.get("num_target_layers") == text.get("num_hidden_layers")
    )


def find_splash_draft(text_config: dict, candidates: Iterable[str]) -> Path | None:
    """The first DFlash2 draft among ``candidates`` that fits ``text_config``."""
    for candidate in candidates:
        path = Path(candidate)
        draft = _read_config(path)
        if draft is not None and _is_splash_draft(draft, text_config):
            return path
    return None


@dataclass(frozen=True)
class SplashTarget:
    """What a model needs to run on Splash: a launcher and a draft."""

    binary: SplashBinary
    family: str
    draft_path: Path


def resolve_splash_target(
    model_path: str | Path,
    configured_binary: str | None,
    draft_candidates: Iterable[str],
) -> tuple[SplashTarget | None, str]:
    """Check whether Splash can serve ``model_path``.

    ``draft_candidates`` are the paths of the speculative-decoding helper
    models oMLX discovered, the configured DFlash draft first.

    Returns:
        (target, reason). ``target`` is None and ``reason`` says why when
        Splash cannot serve the model.
    """
    path = Path(model_path)
    binary, reason = find_splash_binary(configured_binary)
    if binary is None:
        return None, reason
    config = _read_config(path)
    if config is None:
        return None, "failed to read config.json"
    family = splash_family(config)
    if family is None:
        return None, f"Splash serves {' and '.join(SPLASH_FAMILIES)} only"
    problem = _quantization_problem(config, binary)
    if problem:
        return None, problem
    missing = [name for name in _REQUIRED_MODEL_FILES if not (path / name).is_file()]
    if missing:
        return None, f"Splash needs {', '.join(missing)} in the model folder"
    draft = find_splash_draft(config["text_config"], draft_candidates)
    if draft is None:
        return None, (
            f"Splash needs the {family} DFlash2 draft in a model directory "
            f"(z-lab/{family}-DFlash2 or incoai/{family}-DFlash2)"
        )
    return SplashTarget(binary, family, draft), ""


# -- memory ---------------------------------------------------------------------


def _safetensors_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.glob("*.safetensors"))


def _kv_bytes_per_token(text: dict, kv_format: str) -> int:
    layer_types = text.get("layer_types") or []
    layers = sum(1 for kind in layer_types if kind == "full_attention")
    if not layers:
        layers = text["num_hidden_layers"] // text.get("full_attention_interval", 1)
    values = layers * 2 * text["num_key_value_heads"] * text["head_dim"]
    # int8 pages carry one 1-byte scale per 64 values.
    return values * 2 if kv_format == "bf16" else values + values // 64


def splash_memory_estimate(
    model_path: str | Path,
    draft_path: Path,
    max_context: int | None,
    kv_format: str,
) -> int:
    """Memory for Splash to hold ``max_context`` tokens of context.

    The weights, the draft as Splash loads it, scratch, and KV pages for
    ``max_context`` tokens. The weights include the vision tower Splash loads
    for image input. Splash's context limit is its whole KV pool, and
    concurrent requests that outgrow the pool wait for pages, so one window
    is enough. Splash spends a larger budget on its prefix cache.
    """
    path = Path(model_path)
    text = (_read_config(path) or {}).get("text_config") or {}
    context = min(max_context or SPLASH_MAX_CONTEXT, SPLASH_MAX_CONTEXT)
    kv = _kv_bytes_per_token(text, kv_format) * context
    draft = int(_safetensors_bytes(draft_path) * _DRAFT_LOAD_RATIO)
    return _safetensors_bytes(path) + draft + _RUNTIME_RESERVE_BYTES + kv


# -- model staging -------------------------------------------------------------


def splash_repo_id(model_id: str) -> str:
    """The Hub-style ID oMLX stages ``model_id`` under for Splash.

    Splash takes ``owner/repo`` IDs only. The ``omlx`` owner keeps staged
    models apart from the IDs a standalone Splash installs from the Hub, and
    the hash keeps model IDs that read alike (``a/b``, ``a-b``) apart.
    """
    slug = re.sub(r"[^A-Za-z0-9._]+", "-", model_id)
    slug = re.sub(r"\.{2,}", ".", slug).strip("-.")[:87].strip("-.")
    digest = hashlib.sha1(model_id.encode()).hexdigest()[:8]
    return f"omlx/{slug or 'model'}-{digest}"


def _model_files(model_path: Path) -> list[Path]:
    files = []
    for path in model_path.rglob("*"):
        relative = path.relative_to(model_path)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def _snapshot_revision(model_path: Path, files: list[Path]) -> str:
    """A 40-hex revision that changes whenever the model files change."""
    digest = hashlib.sha1(str(model_path.resolve()).encode())
    for path in files:
        stat = path.stat()
        relative = path.relative_to(model_path)
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def stage_model(model_path: Path, repo_id: str, hub_dir: Path) -> Path:
    """Present ``model_path`` to Splash as a cached snapshot of ``repo_id``.

    Builds ``<hub_dir>/models--<owner>--<repo>/snapshots/<rev>`` from
    symlinks to the model files and points ``refs/main`` at it, which is
    what an offline Splash resolves. A link that resolves into a Hub cache
    ``blobs/`` file spares Splash hashing that file.

    Returns:
        The snapshot directory.
    """
    files = _model_files(model_path)
    revision = _snapshot_revision(model_path, files)
    repo_dir = hub_dir / ("models--" + repo_id.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    snapshot = snapshots / revision
    if not snapshot.is_dir():
        partial = snapshots / f".{revision}.partial"
        shutil.rmtree(partial, ignore_errors=True)
        for path in files:
            link = partial / path.relative_to(model_path)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(path.resolve())
        partial.rename(snapshot)
    for stale in snapshots.iterdir():
        if stale.name != revision:
            shutil.rmtree(stale, ignore_errors=True)
    refs = repo_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision)
    return snapshot


# -- request translation -------------------------------------------------------


def splash_effort(chat_template_kwargs: dict | None) -> str | None:
    kwargs = chat_template_kwargs or {}
    if kwargs.get("enable_thinking") is False:
        return "none"
    effort = kwargs.get("reasoning_effort")
    if effort is None:
        return None
    effort = str(effort).lower()
    effort = _EFFORT_ALIASES.get(effort, effort)
    return effort if effort in _SPLASH_EFFORTS else None


def splash_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-format messages for Splash.

    oMLX hands engines template-format tool calls, whose arguments may be a
    dict; the OpenAI API Splash implements wants a JSON string. The Responses
    API path keeps its ``input_image`` parts, which Splash's chat API takes
    as ``image_url`` parts.
    """
    converted = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            message = {**message, "content": [_chat_part(p) for p in content]}
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


def _chat_part(part: Any) -> Any:
    """A Responses ``input_image`` part as a chat ``image_url`` part."""
    if not isinstance(part, dict) or part.get("type") != "input_image":
        return part
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    image_url = {"url": url}
    if part.get("detail"):
        image_url["detail"] = part["detail"]
    return {"type": "image_url", "image_url": image_url}


def _check_media(messages: list[dict[str, Any]]) -> None:
    """Refuse content parts Splash cannot read before they reach it."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in _UNSUPPORTED_MEDIA_TYPES:
                raise InvalidRequestError(
                    "Splash models take text and image input only, "
                    f"not {part['type']!r} content"
                )


def _text_of(content: Any) -> str:
    """The text of message content, without media payloads."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    texts = []
    for part in content:
        if not isinstance(part, dict):
            texts.append(str(part))
        elif part.get("type") not in _SPLASH_MEDIA_TYPES | _UNSUPPORTED_MEDIA_TYPES:
            texts.append(str(part.get("text") or ""))
    return "\n".join(texts)


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


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except (ProcessLookupError, PermissionError):
        # macOS reports EPERM for a group that only holds zombies.
        return False
    return True


async def _wait_group_exit(group: int, timeout: float) -> bool:
    """Whether every process in ``group`` exited within ``timeout``."""
    deadline = time.monotonic() + timeout
    while _group_alive(group):
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(STOP_POLL_SECONDS)
    return True


def _splash_env(api_key: str | None, hub_dir: Path | None = None) -> dict[str, str]:
    """The environment for a Splash launcher.

    oMLX's bundled interpreter sets PYTHONHOME/PYTHONPATH, which would point
    Splash's own Python at the wrong standard library. The API key travels
    in the environment so it never shows up in the process list. With
    ``hub_dir``, Splash resolves models from that private cache offline.
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
    if hub_dir is not None:
        env["HF_HUB_CACHE"] = str(hub_dir)
        env["HF_HUB_OFFLINE"] = "1"
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


def _failure_summary(path: Path, offset: int) -> str:
    """Why a Splash start failed, from what it logged since ``offset``.

    Splash prefixes its own failures with ``error:``; the launcher's setup
    output around them is noise here.
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


@dataclass(frozen=True)
class SplashOptions:
    """The ``splash serve`` options oMLX sets for one model."""

    max_context: int | None = None
    max_memory_bytes: int = 0
    kv_format: str = "int8"
    ane_prefill: bool = True
    cache_disk_bytes: int = 0
    cache_dir: Path | None = None


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
        model_id: str,
        target: SplashTarget,
        hub_dir: Path,
        log_dir: Path,
        options: SplashOptions,
    ) -> None:
        self._path = Path(model_name)
        self._target = target
        self._repo_id = splash_repo_id(model_id)
        self._hub_dir = hub_dir
        self._options = options
        self._process: subprocess.Popen | None = None
        self._client: httpx.AsyncClient | None = None
        self._base_url = ""
        self._api_key = secrets.token_urlsafe(32)
        self._log_path = log_dir / (self._repo_id.replace("/", "--") + ".log")
        self._log_offset = 0
        self._active = 0
        self._requests = 0
        self._live: dict[str, _LiveRequest] = {}
        self._weights_size: int | None = None
        self._tokenizer: Any = None
        self._model_type: str | None = None
        self._status: dict[str, Any] | None = None
        self._status_task: asyncio.Task | None = None
        self._status_refresh: asyncio.Task | None = None
        self._vision = splash_has_vision(_read_config(self._path) or {})

    # -- identity ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return str(self._path)

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model_type(self) -> str | None:
        return self._model_type

    @property
    def accepts_image_input(self) -> bool:
        """Image content parts reach Splash for a checkpoint with a vision tower."""
        return self._vision

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def max_context_tokens(self) -> int | None:
        """The context window Splash planned within its memory budget."""
        value = (self._status or {}).get("maximum_context_tokens")
        return value if isinstance(value, int) and value > 0 else None

    def has_active_requests(self) -> bool:
        return self._active > 0

    @property
    def runtime_failed_reason(self) -> str | None:
        """Why Splash stopped serving, when its process exited on its own.

        The pool unloads an engine that reports this and reloads it on the
        next request.
        """
        process = self._process
        if process is None or process.poll() is None:
            return None
        return (
            f"Splash exited with code {process.returncode}: "
            f"{_failure_summary(self._log_path, self._log_offset)} "
            f"(log: {self._log_path})"
        )

    @contextlib.contextmanager
    def outside_requests(self, count: int = 1):
        """Count requests sent to Splash's API directly as active ones.

        The admin benchmark talks to Splash without stream_chat(); this keeps
        the model from looking idle to eviction meanwhile.
        """
        self._active += count
        try:
            yield
        finally:
            self._active -= count

    # -- lifecycle -----------------------------------------------------------------

    def _load_tokenizer(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._path))
        config = _read_config(self._path) or {}
        self._model_type = config.get("model_type")

    def _serve_command(self, port: int) -> list[str]:
        options = self._options
        command = [
            str(self._target.binary.path),
            "serve",
            "--model",
            self._repo_id,
            "--offline",
            "--draft-model",
            str(self._target.draft_path.resolve()),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-webui",
            # oMLX unloads idle models itself; Splash must not free weights
            # that oMLX still counts as loaded.
            "--idle-release",
            "off",
            "--kv-format",
            options.kv_format,
        ]
        if not self._vision:
            # Text-only checkpoint: skip Splash's vision preparation.
            command.append("--language-only")
        # Without a limit Splash fits its context to the memory budget; a
        # limit the budget cannot hold makes it refuse to start.
        if options.max_context:
            command += [
                "--max-context",
                str(min(options.max_context, SPLASH_MAX_CONTEXT)),
            ]
        if options.max_memory_bytes > 0:
            command += ["--max-memory", str(options.max_memory_bytes)]
        if not options.ane_prefill:
            command.append("--disable-ane")
        if options.cache_disk_bytes > 0 and options.cache_dir is not None:
            command += [
                "--max-cache-disk",
                str(options.cache_disk_bytes),
                "--persistent-cache",
                "--cache-dir",
                str(options.cache_dir),
            ]
        return command

    async def start(self) -> None:
        # Request paths call start() on loaded engines to load lazy parts;
        # a second call must not launch a second Splash. The pool reloads an
        # engine whose Splash died.
        if self._process is not None:
            if self._process.poll() is None:
                return
            raise RuntimeError(self.runtime_failed_reason)
        await asyncio.to_thread(self._load_tokenizer)
        await asyncio.to_thread(stage_model, self._path, self._repo_id, self._hub_dir)
        port = _free_port()
        self._base_url = f"http://127.0.0.1:{port}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        command = self._serve_command(port)
        if self._options.cache_dir is not None and self._options.cache_disk_bytes:
            self._options.cache_dir.mkdir(parents=True, exist_ok=True)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Splash: starting %s (log: %s)", " ".join(command), self._log_path)
        with open(self._log_path, "ab") as log:
            self._log_offset = log.tell()
            self._process = subprocess.Popen(
                _watchdog_command(command),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=_splash_env(self._api_key, self._hub_dir),
                start_new_session=True,
            )
        try:
            await self._wait_until_ready()
            await self._refresh_status()
            self._status_task = asyncio.create_task(self._poll_status())
        except BaseException:
            await self.stop()
            raise

    async def _refresh_status(self) -> None:
        """Keep the last /status snapshot for the synchronous getters."""
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
            await asyncio.sleep(STATUS_POLL_SECONDS)
            await self._refresh_status()

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
                    f"while loading {self._path.name}: "
                    f"{_failure_summary(self._log_path, self._log_offset)} "
                    f"(log: {self._log_path})"
                )
            if await self._serves_model():
                logger.info("Splash: serving %s at %s", self._repo_id, self._base_url)
                return
            await asyncio.sleep(STARTUP_POLL_SECONDS)
        raise RuntimeError(
            f"Splash did not serve {self._path.name} within "
            f"{STARTUP_TIMEOUT_SECONDS}s; see {self._log_path}"
        )

    async def stop(self) -> None:
        for task in (self._status_task, self._status_refresh):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._status_task = self._status_refresh = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        process, self._process = self._process, None
        if process is None:
            return
        # The watchdog leads Splash's process group and exits on the signal
        # at once, so wait for the whole group: unloading is the memory
        # barrier the pool relies on.
        group = process.pid
        _signal_group(group, signal.SIGTERM)
        if not await _wait_group_exit(group, STOP_TIMEOUT_SECONDS):
            logger.warning("Splash: server ignored SIGTERM; killing it")
            _signal_group(group, signal.SIGKILL)
            await _wait_group_exit(group, STOP_TIMEOUT_SECONDS)
        await asyncio.to_thread(process.wait)

    # -- requests ------------------------------------------------------------------

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
        body: dict[str, Any] = {
            "model": self._repo_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
        }
        for key in ("frequency_penalty", "stop", "seed", "response_format"):
            if kwargs.get(key) is not None:
                body[key] = kwargs[key]
        # oMLX's resolved budget (request over model settings); the in-process
        # engines enforce it in the scheduler, Splash enforces it itself.
        budget = kwargs.get("thinking_budget")
        if budget is not None and self._target.binary.accepts_thinking_budget:
            body["thinking_token_budget"] = budget
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
        _check_media(messages)
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
        # Count the text only: an image's base64 payload is not prompt text.
        return sum(
            len(self._tokenizer.encode(_text_of(m.get("content")))) for m in messages
        )

    # -- plain completions -----------------------------------------------------------

    async def stream_generate(
        self, prompt: str | list[int], **kwargs
    ) -> AsyncIterator[GenerationOutput]:
        # oMLX forwards chat requests only; there is no plain text path.
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
            "splash_build": self._target.binary.label,
            "splash_version": self._target.binary.version,
            "model_name": self.model_name,
            "server": self._base_url,
            "log_path": str(self._log_path),
            "running": self._process is not None and self._process.poll() is None,
            "requests": self._requests,
            "active_requests": self._active,
            "memory_budget_bytes": self.reserved_bytes(),
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
        block_size = int(kv.get("block_tokens") or 0)
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

    def reserved_bytes(self) -> int:
        """The Metal budget Splash planned at start, which it may grow into."""
        budget = ((self._status or {}).get("memory_plan") or {}).get("budget") or {}
        return int(budget.get("hard_budget_bytes") or 0)

    def _group_pids(self) -> list[int]:
        if self._process is None or self._process.poll() is not None:
            return []
        return list_process_group_pids(self._process.pid)

    def _weights_bytes(self) -> int:
        if self._weights_size is None:
            self._weights_size = sum(f.stat().st_size for f in _model_files(self._path))
        return self._weights_size

    def _reported_memory(self, key: str) -> int:
        # Splash's own Metal accounting, which includes the weight images.
        memory = (self._status or {}).get("memory_actual") or {}
        return int(memory.get(key, 0))

    def process_memory_bytes(self) -> int:
        """Memory the running Splash holds.

        Splash keeps its weights in Metal buffers that the process footprint
        may not charge in full. The Metal memory Splash reports covers them;
        before the first status snapshot, the weights' size on disk stands in.
        """
        pids = self._group_pids()
        if not pids:
            return 0
        footprint = sum(get_phys_footprint(pid) for pid in pids)
        floor = self._reported_memory("current_bytes") or self._weights_bytes()
        return max(footprint, floor)

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
