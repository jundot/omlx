# SPDX-License-Identifier: Apache-2.0
"""Sushi engine: serve a model through a Sushi server process.

Sushi (https://github.com/beamivalice/sushi) is a Zig/Metal engine, a fork of
mlx-serve, that serves a few selected models on Apple Silicon with custom
"sushi" quants: packs whose routed experts are EXL3 trellis tensors, which
mlx-lm cannot load. Like Splash it is a separate program, so this engine does
not load weights in-process. ``start()`` launches ``sushi serve`` for the
model folder on a loopback port and forwards chat requests to its
OpenAI-compatible API. Reasoning streams as ``reasoning_content`` and tool
calls stream structured, the same shapes Splash produces, so the request and
stream handling is shared with :class:`SplashEngine`.

Sushi plans its own memory, KV cache, MTP speculation and prefix cache (RAM
and SSD tiers); oMLX manages its lifecycle, routing and memory accounting.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..exceptions import InvalidRequestError
from . import splash as _splash
from .base import GenerationOutput
from .splash import (
    REQUEST_TIMEOUT_SECONDS,
    SplashEngine,
    _failure_summary,
    _free_port,
    _read_config,
    _text_of,
    _watchdog_command,
)

logger = logging.getLogger(__name__)

# Large packs (60+ GB) take minutes to map on first load.
SUSHI_STARTUP_TIMEOUT_SECONDS = 1200
VERSION_TIMEOUT_SECONDS = 30
SUSHI_API_KEY_ENV = "OMLX_SUSHI_API_KEY"

SUSHI_INSTALL_HINT = (
    "Sushi is not installed: download sushi-bin-macos-arm64.tar.gz from "
    "https://github.com/beamivalice/sushi/releases and set its path in "
    "oMLX settings (integrations.sushi_path), or put `sushi` on your PATH"
)

# Architectures Sushi serves (its README / docs/*-arch.md), by config model_type.
SUSHI_MODEL_TYPES = frozenset({"qwen4_exp", "glm5_next", "mimo_v2"})

# Where oMLX installs Sushi itself, and the release asset it fetches.
SUSHI_INSTALL_DIR = Path.home() / ".omlx" / "bin" / "sushi-macos-arm64"
SUSHI_RELEASE_URL = (
    "https://github.com/beamivalice/sushi/releases/latest/download/"
    "sushi-bin-macos-arm64.tar.gz"
)
# Where `hf download` puts Sushi packs in Sushi's own instructions.
SUSHI_MODELS_DIR = Path.home() / ".sushi" / "models"

# Common places a release tarball gets unpacked.
_CANDIDATE_DIRS = (
    str(SUSHI_INSTALL_DIR),
    "/opt/homebrew/bin",
    "/usr/local/bin",
    str(Path.home() / ".local" / "bin"),
    str(Path.home() / "sushi-macos-arm64"),
    str(Path.home() / ".sushi" / "bin"),
    str(Path.home() / "Downloads" / "sushi-macos-arm64"),
)

# Sushi's effort vocabulary, weakest first. Each architecture accepts a
# subset (advertised per /v1/models row) and refuses every other word.
_EFFORT_ORDER = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
_EFFORT_ALIASES = {"none": "off", "moderate": "medium", "maximum": "max"}

# Sushi's KV cache with MTP and --mtp-head-kv-quant at 8 bits, per token of
# context, for Qwen3.8-Flash-Next (README "Memory"). Other arches fall back to
# this figure too; Sushi's own admission check is the real gate.
_KV_BYTES_PER_TOKEN_KV8 = 16_640
_RUNTIME_RESERVE_BYTES = 3 * 1024**3


# -- binary ----------------------------------------------------------------------


@dataclass(frozen=True)
class SushiBinary:
    path: Path
    version: str
    auto_detected: bool

    @property
    def label(self) -> str:
        return "Sushi"

    @property
    def accepts_thinking_budget(self) -> bool:
        # Sushi takes the budget as `reasoning_budget_tokens`, set by
        # SushiEngine._request_body, not as Splash's thinking_token_budget.
        return False


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


@functools.lru_cache(maxsize=16)
def _read_version(path: str, _mtime_ns: int) -> str | None:
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=VERSION_TIMEOUT_SECONDS,
            env=sushi_env(None),
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Sushi: %s --version failed: %s", path, e)
        return None
    match = re.search(r"\d+\.\d+\.\d+\S*", result.stdout + result.stderr)
    return match.group(0) if match else None


def inspect_sushi_binary(
    path: Path, *, auto_detected: bool
) -> tuple[SushiBinary | None, str]:
    if not _executable(path):
        return None, f"{path} is not an executable file"
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None, f"{path} is not readable"
    version = _read_version(str(path), mtime)
    if version is None:
        return None, f"{path} did not report a Sushi version"
    return SushiBinary(path, version, auto_detected), ""


def find_sushi_binary(configured: str | None) -> tuple[SushiBinary | None, str]:
    """The Sushi binary to run: ``configured``, or one found installed."""
    if configured:
        return inspect_sushi_binary(Path(configured).expanduser(), auto_detected=False)
    candidates = []
    found = shutil.which("sushi")
    if found:
        candidates.append(Path(found))
    candidates += [Path(d) / "sushi" for d in _CANDIDATE_DIRS]
    for path in candidates:
        if _executable(path):
            binary, reason = inspect_sushi_binary(path, auto_detected=True)
            if binary is not None:
                return binary, ""
    return None, SUSHI_INSTALL_HINT


# -- model compatibility ---------------------------------------------------------


def is_sushi_pack(config: dict) -> bool:
    """Whether a config describes a Sushi pack (EXL3 routed experts)."""
    quant = config.get("expert_quant")
    if isinstance(quant, dict) and str(quant.get("format", "")).lower() == "exl3":
        return True
    text = config.get("text_config") or {}
    quant = text.get("expert_quant")
    return isinstance(quant, dict) and str(quant.get("format", "")).lower() == "exl3"


def sushi_model_type(config: dict) -> str | None:
    model_type = str(config.get("model_type") or "")
    return model_type if model_type in SUSHI_MODEL_TYPES else None


@dataclass(frozen=True)
class SushiTarget:
    # None: not installed yet; the engine installs it on start.
    binary: SushiBinary | None
    model_type: str
    exl3: bool


def resolve_sushi_target(
    model_path: str | Path,
    configured_binary: str | None,
    auto_install: bool = False,
) -> tuple[SushiTarget | None, str]:
    """Check whether Sushi can serve ``model_path``; else say why not."""
    path = Path(model_path)
    config = _read_config(path)
    if config is None:
        return None, "failed to read config.json"
    model_type = sushi_model_type(config)
    if model_type is None:
        return None, (
            "Sushi serves "
            + ", ".join(sorted(SUSHI_MODEL_TYPES))
            + f" models only (this is {config.get('model_type')!r})"
        )
    binary, reason = find_sushi_binary(configured_binary)
    if binary is None and not (auto_install and not configured_binary):
        return None, reason
    return SushiTarget(binary, model_type, is_sushi_pack(config)), ""


# -- install ---------------------------------------------------------------------


def install_sushi(dest: Path = SUSHI_INSTALL_DIR, url: str = SUSHI_RELEASE_URL) -> Path:
    """Download the latest Sushi release into ``dest`` and return the binary.

    The tarball unpacks to ``sushi-macos-arm64/``; it is extracted next to
    ``dest`` through a temporary folder and swapped in, so a failed or
    interrupted download never leaves a half-written install behind.
    """
    import tarfile
    import tempfile

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Sushi: downloading %s", url)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".sushi-dl-") as tmp:
        archive = Path(tmp) / "sushi.tar.gz"
        with httpx.stream(
            "GET", url, follow_redirects=True, timeout=httpx.Timeout(30, read=600)
        ) as response:
            response.raise_for_status()
            with open(archive, "wb") as out:
                for chunk in response.iter_bytes():
                    out.write(chunk)
        unpack = Path(tmp) / "unpack"
        with tarfile.open(archive) as tar:
            tar.extractall(unpack, filter="data")
        found = next(
            (p for p in unpack.rglob("sushi") if p.is_file()),
            None,
        )
        if found is None:
            raise RuntimeError(f"the Sushi release at {url} has no `sushi` binary")
        found.chmod(found.stat().st_mode | 0o111)
        root = found.parent
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(root), str(dest))
    binary = dest / "sushi"
    logger.info("Sushi: installed %s", binary)
    return binary


# -- wired memory ----------------------------------------------------------------


def physical_memory_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def wired_limit_target_mb(needed_bytes: int, ram_bytes: int) -> int:
    """The ``iogpu.wired_limit_mb`` to set so ``needed_bytes`` fits.

    Sushi's own guidance leaves about 8 GiB of RAM to macOS (58000 MB on a
    64 GB Mac, 88000 on 96 GB, 120000 on 128 GB). Never go above that, even
    if the model wants more: Sushi's admission check then refuses cleanly
    instead of the OS swapping wired pages.
    """
    ceiling_mb = (ram_bytes // 1024**2) - 8 * 1024
    ceiling_mb -= ceiling_mb % 1000
    needed_mb = -(-needed_bytes // 1024**2)
    if needed_mb > ceiling_mb:
        logger.warning(
            "Sushi: model wants %d MB of GPU memory, more than the %d MB "
            "macOS can safely wire on this Mac",
            needed_mb,
            ceiling_mb,
        )
    return max(0, ceiling_mb)


def raise_wired_limit(limit_mb: int) -> None:
    """Set ``iogpu.wired_limit_mb`` through macOS's administrator dialog.

    Runs one fixed command; macOS asks for the password in its own window.
    The setting lasts until the next reboot.
    """
    command = f"/usr/sbin/sysctl iogpu.wired_limit_mb={int(limit_mb)}"
    script = (
        f'do shell script "{command}" with administrator privileges '
        'with prompt "oMLX needs to raise the GPU memory limit to run '
        'this Sushi model."'
    )
    result = subprocess.run(
        ["/usr/bin/osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        if "-128" in detail:
            raise RuntimeError(
                "GPU memory limit not raised (the password dialog was "
                f"cancelled). Sushi needs: sudo sysctl iogpu.wired_limit_mb={limit_mb}"
            )
        raise RuntimeError(f"could not raise the GPU memory limit: {detail}")


# -- memory ----------------------------------------------------------------------


def _weights_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.glob("*.safetensors"))


def sushi_memory_estimate(
    model_path: str | Path,
    ctx_size: int | None,
    kv_quant: str,
    prefix_cache_mem_bytes: int = 0,
) -> int:
    """Memory Sushi will wire for the weights and one ``ctx_size`` window."""
    path = Path(model_path)
    if not ctx_size:
        text = (_read_config(path) or {}).get("text_config") or {}
        ctx_size = int(text.get("max_position_embeddings") or 262_144)
    per_token = _KV_BYTES_PER_TOKEN_KV8
    if kv_quant == "4":
        per_token //= 2
    elif kv_quant in ("16", "off"):
        per_token *= 2
    return (
        _weights_bytes(path)
        + per_token * int(ctx_size)
        + prefix_cache_mem_bytes
        + _RUNTIME_RESERVE_BYTES
    )


# -- options ---------------------------------------------------------------------


@dataclass
class SushiOptions:
    """The ``sushi serve`` options oMLX sets for one model.

    Defaults match the recommended Sushi-4bpw launch on a 96 GB+ Mac.
    """

    ctx_size: int | None = 500_000
    kv_quant: str = "8"
    mtp: bool = True
    mtp_head_kv_quant: bool = True
    prefill_chunk: int | None = 2048
    max_tokens: int | None = 64_000
    temperature: float | None = 1.0
    prefix_cache_disk: str | None = "20GB"
    prefix_cache_entries: int | None = 1
    prefix_cache_mem: str | None = None
    extra_args: list[str] = field(default_factory=list)
    # Download Sushi on first use when no binary is found.
    auto_install: bool = True
    # Raise iogpu.wired_limit_mb (admin dialog) when the model needs it.
    auto_wired_limit: bool = True


def parse_size_bytes(value: str | None) -> int:
    """'20GB' / '1GB' / '512MB' as bytes (binary units, as Sushi reads them)."""
    if not value:
        return 0
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)i?B?\s*", str(value), re.I)
    if not match:
        return 0
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    return int(float(match.group(1)) * scale[match.group(2).upper()])


def sushi_env(api_key: str | None) -> dict[str, str]:
    """oMLX's bundled interpreter's Python variables do not belong to Sushi."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONHOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE")
    }
    if api_key is not None:
        env[SUSHI_API_KEY_ENV] = api_key
    return env


def pick_effort(requested: str | None, accepted: list[str]) -> str | None:
    """Map an oMLX effort word onto one this Sushi model accepts.

    Sushi refuses a word outside the model's list instead of rounding it, so
    round here: the requested word if accepted, else the nearest stronger
    thinking word, else the nearest weaker one. ``off`` maps only to ``off``.
    """
    if requested is None:
        return None
    word = _EFFORT_ALIASES.get(str(requested).lower(), str(requested).lower())
    if not accepted:
        return word
    if word in accepted:
        return word
    if word == "off":
        return None
    if word == "on":
        thinking = [w for w in accepted if w != "off"]
        return thinking[0] if thinking else None
    if word not in _EFFORT_ORDER:
        return None
    rank = _EFFORT_ORDER.index(word)
    ranked = [w for w in accepted if w in _EFFORT_ORDER and w != "off"]
    stronger = [w for w in ranked if _EFFORT_ORDER.index(w) >= rank]
    if stronger:
        return min(stronger, key=_EFFORT_ORDER.index)
    if ranked:
        return max(ranked, key=_EFFORT_ORDER.index)
    return "on" if "on" in accepted else None


# -- engine ----------------------------------------------------------------------


class SushiEngine(SplashEngine):
    """Serves one model through a ``sushi serve`` child process.

    Subclasses SplashEngine for its request streaming, process-group
    management and stats plumbing, so every pool path that treats Splash as
    an out-of-process engine treats Sushi the same way.
    """

    external_engine_id = "sushi"
    supports_native_response_format = True

    def __init__(  # noqa: D107 - SplashEngine's __init__ is not reused
        self,
        model_name: str,
        model_id: str,
        target: SushiTarget,
        log_dir: Path,
        options: SushiOptions,
        memory_budget_bytes: int = 0,
    ) -> None:
        self._path = Path(model_name)
        self._model_id = model_id
        self._target = target
        self._options = options
        self._memory_budget = memory_budget_bytes
        self._process: subprocess.Popen | None = None
        self._client: httpx.AsyncClient | None = None
        self._base_url = ""
        self._api_key = os.urandom(24).hex()
        slug = re.sub(r"[^A-Za-z0-9._-]+", "--", model_id)
        self._log_path = log_dir / f"{slug}.log"
        self._log_offset = 0
        self._active = 0
        self._requests = 0
        self._live: dict = {}
        self._weights_size: int | None = None
        self._tokenizer: Any = None
        self._model_type: str | None = None
        self._status: dict[str, Any] | None = None
        self._status_task: asyncio.Task | None = None
        self._status_refresh: asyncio.Task | None = None
        self._served_id: str = str(self._path)
        self._efforts: list[str] = []
        self._context_length: int | None = None
        config = _read_config(self._path) or {}
        self._vision = bool(config.get("vision_config")) and not config.get(
            "language_model_only", False
        )
        # Splash-only attributes some inherited helpers read.
        self._repo_id = self._served_id

    # -- identity ------------------------------------------------------------------

    @property
    def max_context_tokens(self) -> int | None:
        if self._context_length:
            return self._context_length
        return self._options.ctx_size

    @property
    def runtime_failed_reason(self) -> str | None:
        process = self._process
        if process is None or process.poll() is None:
            return None
        return (
            f"Sushi exited with code {process.returncode}: "
            f"{_failure_summary(self._log_path, self._log_offset)} "
            f"(log: {self._log_path})"
        )

    # -- lifecycle -----------------------------------------------------------------

    def _load_tokenizer(self) -> None:
        config = _read_config(self._path) or {}
        self._model_type = config.get("model_type")
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self._path))
        except Exception:  # noqa: BLE001 - token counts fall back to estimates
            logger.warning(
                "Sushi: no Hugging Face tokenizer for %s; token counts are "
                "estimated",
                self._path,
                exc_info=True,
            )
            self._tokenizer = None

    def _ensure_binary(self) -> None:
        if self._target.binary is not None:
            return
        if not self._options.auto_install:
            raise RuntimeError(SUSHI_INSTALL_HINT)
        binary, _ = find_sushi_binary(None)
        if binary is None:
            path = install_sushi()
            binary, reason = inspect_sushi_binary(path, auto_detected=True)
            if binary is None:
                raise RuntimeError(f"Sushi was downloaded but does not run: {reason}")
        self._target = SushiTarget(binary, self._target.model_type, self._target.exl3)

    def _ensure_wired_limit(self) -> None:
        """Make room in macOS's GPU wired limit for this model."""
        if not self._options.auto_wired_limit or self._memory_budget <= 0:
            return
        from ..process_memory_enforcer import (
            get_effective_metal_cap_bytes,
            get_iogpu_wired_limit_bytes,
        )

        cap = get_iogpu_wired_limit_bytes() or get_effective_metal_cap_bytes()
        if cap >= self._memory_budget:
            return
        target = wired_limit_target_mb(self._memory_budget, physical_memory_bytes())
        if target * 1024**2 <= cap:
            return  # cannot do better than what is set; let Sushi decide
        logger.info(
            "Sushi: %s needs ~%.1f GiB of GPU memory, limit is %.1f GiB; "
            "raising iogpu.wired_limit_mb to %d",
            self._path.name,
            self._memory_budget / 1024**3,
            cap / 1024**3,
            target,
        )
        raise_wired_limit(target)

    def _serve_command(self, port: int) -> list[str]:
        o = self._options
        command = [
            str(self._target.binary.path),
            "serve",
            "--model",
            str(self._path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-key-env",
            SUSHI_API_KEY_ENV,
            # Exit with oMLX even if the watchdog shell is killed first.
            "--parent-pid",
            str(os.getpid()),
            # oMLX owns loading and eviction; one model per Sushi process.
            "--max-resident-models",
            "1",
            "--no-update-check",
            "--log-file",
            "off",
            "--kv-quant",
            o.kv_quant,
        ]
        command.append("--mtp" if o.mtp else "--no-mtp")
        if o.mtp and o.mtp_head_kv_quant:
            command.append("--mtp-head-kv-quant")
        if o.ctx_size:
            command += ["--ctx-size", str(o.ctx_size)]
        if o.prefill_chunk:
            command += ["--prefill-chunk", str(o.prefill_chunk)]
        if o.max_tokens:
            command += ["--max-tokens", str(o.max_tokens)]
        if o.temperature is not None:
            command += ["--temp", f"{o.temperature:g}"]
        if o.prefix_cache_disk:
            command += ["--prefix-cache-disk", o.prefix_cache_disk]
        if o.prefix_cache_entries:
            command += ["--prefix-cache-entries", str(o.prefix_cache_entries)]
        if o.prefix_cache_mem:
            command += ["--prefix-cache-mem", o.prefix_cache_mem]
        command += list(o.extra_args)
        return command

    async def start(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                return
            raise RuntimeError(self.runtime_failed_reason)
        await asyncio.to_thread(self._load_tokenizer)
        await asyncio.to_thread(self._ensure_binary)
        await asyncio.to_thread(self._ensure_wired_limit)
        port = _free_port()
        self._base_url = f"http://127.0.0.1:{port}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        command = self._serve_command(port)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Sushi: starting %s (log: %s)", shlex.join(command), self._log_path)
        with open(self._log_path, "ab") as log:
            self._log_offset = log.tell()
            self._process = subprocess.Popen(
                _watchdog_command(command),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=sushi_env(self._api_key),
                start_new_session=True,
            )
        try:
            await self._wait_until_ready()
            await self._refresh_status()
            self._status_task = asyncio.create_task(self._poll_status())
        except BaseException:
            await self.stop()
            raise

    async def _loaded_row(self) -> dict | None:
        try:
            response = await self._client.get("/v1/models", timeout=5)
            rows = response.json().get("data", [])
        except (httpx.HTTPError, ValueError, AttributeError):
            return None
        for row in rows:
            if isinstance(row, dict) and row.get("loaded"):
                return row
        return None

    async def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + SUSHI_STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"Sushi exited with code {self._process.returncode} "
                    f"while loading {self._path.name}: "
                    f"{_failure_summary(self._log_path, self._log_offset)} "
                    f"(log: {self._log_path})"
                )
            row = await self._loaded_row()
            if row is not None:
                self._served_id = str(row.get("id") or self._served_id)
                self._repo_id = self._served_id
                efforts = row.get("reasoning_efforts")
                if isinstance(efforts, list):
                    self._efforts = [str(e) for e in efforts]
                ctx = row.get("max_model_len") or row.get("context_length")
                if isinstance(ctx, int) and ctx > 0:
                    self._context_length = ctx
                logger.info(
                    "Sushi: serving %s as %r at %s",
                    self._path.name,
                    self._served_id,
                    self._base_url,
                )
                return
            await asyncio.sleep(_splash.STARTUP_POLL_SECONDS)
        raise RuntimeError(
            f"Sushi did not load {self._path.name} within "
            f"{SUSHI_STARTUP_TIMEOUT_SECONDS}s; see {self._log_path}"
        )

    async def _refresh_status(self) -> None:
        """Keep the last /props snapshot for the synchronous getters."""
        if self._client is None:
            return
        try:
            response = await self._client.get("/props", timeout=2)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                self._status = data
        except Exception:  # noqa: BLE001
            logger.debug("Sushi: failed to fetch /props", exc_info=True)

    # -- requests ------------------------------------------------------------------

    def _request_body(self, *args) -> dict[str, Any]:
        body = super()._request_body(*args)
        body["model"] = self._served_id
        # oMLX's resolved thinking budget (request over model settings), in
        # the field Sushi reads; it outranks the effort word's own cap.
        budget = args[-1].get("thinking_budget") if args else None
        if budget is not None:
            body["reasoning_budget_tokens"] = int(budget)
        return body

    async def _stream(self, body: dict[str, Any], request_id: str | None = None):
        # Sushi reports progress through its final `timings`, not
        # prompt_progress chunks; the extra field is ignored either way.
        try:
            async for output in super()._stream(body, request_id):
                yield output
        except InvalidRequestError as e:
            raise InvalidRequestError(str(e).replace("Splash", "Sushi", 1)) from e

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
    ):
        from .splash import splash_messages

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
            if kwargs.get("tool_choice") is not None:
                body["tool_choice"] = kwargs["tool_choice"]
        template = dict(kwargs.get("chat_template_kwargs") or {})
        if template.get("enable_thinking") is False:
            requested = "off"
        else:
            requested = template.pop("reasoning_effort", None)
            if requested is None and template.get("enable_thinking") is True:
                requested = "on"
        effort = pick_effort(requested, self._efforts)
        if effort:
            body["reasoning_effort"] = effort
        # Pass the rest through (e.g. preserve_thinking); Sushi ignores
        # fields it does not read.
        template.pop("enable_thinking", None)
        if template:
            body["chat_template_kwargs"] = template
        async for output in self._stream(body, kwargs.get("_request_id")):
            yield output

    async def chat(self, messages: list[dict[str, Any]], **kwargs) -> GenerationOutput:
        return await self._collect(self.stream_chat(messages, **kwargs))

    def count_chat_tokens(
        self, messages, tools=None, chat_template_kwargs=None, is_partial=None
    ) -> int:
        if self._tokenizer is None:
            # ~4 characters per token for English text and code.
            return sum(len(_text_of(m.get("content"))) for m in messages) // 4 + 1
        return super().count_chat_tokens(
            messages, tools, chat_template_kwargs, is_partial
        )

    def _count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            return max(1, len(text) // 4)
        return super()._count_tokens(text)

    async def stream_generate(self, prompt, **kwargs):
        raise InvalidRequestError(
            "Sushi models serve chat requests only; use /v1/chat/completions, "
            "/v1/responses or /v1/messages"
        )
        yield  # pragma: no cover

    # -- stats -----------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_type": "sushi",
            "sushi_version": self._target.binary.version if self._target.binary else None,
            "sushi_path": str(self._target.binary.path) if self._target.binary else None,
            "model_name": self.model_name,
            "served_model_id": self._served_id,
            "server": self._base_url,
            "log_path": str(self._log_path),
            "running": self._process is not None and self._process.poll() is None,
            "requests": self._requests,
            "active_requests": self._active,
            "reasoning_efforts": list(self._efforts),
            "memory_budget_bytes": self.reserved_bytes(),
            "process_memory_bytes": self.process_memory_bytes(),
        }

    def get_runtime_cache_stats(self) -> dict[str, Any] | None:
        """Sushi's prefix-cache counters from /props, in the dashboard shape."""
        if self._status is None:
            return None
        cache = (self._status.get("settings") or {}).get("prefix_cache") or {}
        used = int(cache.get("disk_used_bytes") or 0)
        return {
            "block_size": 0,
            "indexed_blocks": int(cache.get("disk_entries") or 0),
            "ssd_cache": {
                "num_files": int(cache.get("disk_entries") or 0),
                "total_size_bytes": used,
                "max_size_bytes": int(cache.get("disk_bytes") or 0),
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "loads": 0,
                "saves": 0,
            },
            "prefix_cache": {"block_size": 0},
            "cache_rates": {"cumulative": {}},
            "sushi_cache": {
                "ram_enabled": bool(cache.get("ram_enabled")),
                "mem_bytes": int(cache.get("mem_bytes") or 0),
                "disk_bytes": int(cache.get("disk_bytes") or 0),
                "disk_used_bytes": used,
                "disk_entries": int(cache.get("disk_entries") or 0),
            },
        }

    def benchmark_endpoint(self) -> dict[str, Any]:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Sushi engine is not started")
        extra: dict[str, Any] = {}
        if "off" in self._efforts:
            extra["reasoning_effort"] = "off"
        return {
            "base_url": f"{self._base_url}/v1",
            "api_key": self._api_key,
            "model": self._served_id,
            "extra_body": extra,
        }

    def reserved_bytes(self) -> int:
        """The budget oMLX admitted Sushi with (weights + KV for ctx_size)."""
        return int(self._memory_budget)

    def _reported_memory(self, key: str) -> int:
        # /props memory.active_bytes is Sushi's live Metal usage.
        memory = (self._status or {}).get("memory") or {}
        if key == "current_bytes":
            return int(memory.get("active_bytes") or 0)
        return int(memory.get(key) or 0)

    def _weights_bytes(self) -> int:
        if self._weights_size is None:
            self._weights_size = _weights_bytes(self._path)
        return self._weights_size
