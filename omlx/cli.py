#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI for oMLX.

Commands:
    omlx serve --model-dir /path/to/models    Start multi-model server

Usage:
    # Multi-model serving
    omlx serve --model-dir /path/to/models

    # With pinned models
    omlx serve --model-dir /path/to/models --pin llama-3b,qwen-7b
"""

import argparse
import faulthandler
import math
import sys

from ._version import __version__


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return parsed


def _has_cli_overrides(args) -> bool:
    """Check if CLI args contain non-default values that should be saved.

    All argparse defaults are None, so `is not None` means the user
    explicitly passed the flag on the command line.
    """
    if hasattr(args, "model_dir") and args.model_dir is not None:
        return True
    if hasattr(args, "port") and args.port is not None:
        return True
    if hasattr(args, "host") and args.host is not None:
        return True
    if hasattr(args, "log_level") and args.log_level is not None:
        return True
    if hasattr(args, "embedding_batch_size") and args.embedding_batch_size is not None:
        return True
    if hasattr(args, "memory_guard") and args.memory_guard is not None:
        return True
    if hasattr(args, "memory_guard_gb") and args.memory_guard_gb is not None:
        return True
    if hasattr(args, "mcp_config") and args.mcp_config is not None:
        return True
    if hasattr(args, "hf_endpoint") and args.hf_endpoint is not None:
        return True
    if hasattr(args, "hf_cache_enabled") and args.hf_cache_enabled is not None:
        return True
    if hasattr(args, "ms_endpoint") and args.ms_endpoint is not None:
        return True
    if hasattr(args, "http_proxy") and args.http_proxy is not None:
        return True
    if hasattr(args, "https_proxy") and args.https_proxy is not None:
        return True
    if hasattr(args, "no_proxy") and args.no_proxy is not None:
        return True
    if hasattr(args, "ca_bundle") and args.ca_bundle is not None:
        return True
    return False


def serve_command(args):
    """Start the OpenAI-compatible multi-model server."""
    import logging
    import os
    import uvicorn

    from ._version import __version__
    from .settings import init_settings, get_settings
    from .logging_config import configure_file_logging, AdminStatsAccessFilter

    try:
        from ._build_info import build_number
    except ImportError:
        build_number = None

    # Print version banner
    print(f"\033[33moMLX - LLM inference, optimized for your Mac\033[0m")
    print(f"\033[33m├─ https://github.com/jundot/omlx\033[0m")
    if build_number:
        print(f"\033[33m├─ Version: {__version__}\033[0m")
        print(f"\033[33m└─ Build: {build_number}\033[0m")
    else:
        print(f"\033[33m└─ Version: {__version__}\033[0m")
    print()

    # Initialize global settings first (to get log_level from file if not specified)
    settings = init_settings(base_path=args.base_path, cli_args=args)

    # Register TRACE level (5) — includes full message content
    TRACE = 5
    logging.addLevelName(TRACE, "TRACE")

    # Configure logging (use settings value which has proper priority)
    level_name = settings.server.log_level.upper()
    log_level = (
        TRACE if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Set omlx loggers
    for name in [
        "omlx",
        "omlx.scheduler",
        "omlx.paged_ssd_cache",
        "omlx.memory_monitor",
        "omlx.paged_cache",
        "omlx.prefix_cache",
        "omlx.engine_pool",
        "omlx.model_discovery",
    ]:
        logging.getLogger(name).setLevel(log_level)

    # Suppress repetitive admin stats access logs
    logging.getLogger("uvicorn.access").addFilter(AdminStatsAccessFilter())

    # Suppress noisy third-party loggers unless trace level
    if log_level > TRACE:
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)

    # Ensure required directories exist
    settings.ensure_directories()

    # Apply HuggingFace endpoint if configured
    if settings.huggingface.endpoint:
        os.environ["HF_ENDPOINT"] = settings.huggingface.endpoint

    # Apply ModelScope endpoint if configured
    if settings.modelscope.endpoint:
        os.environ["MODELSCOPE_DOMAIN"] = settings.modelscope.endpoint

    # Apply proxy/TLS settings if configured
    if settings.network.http_proxy:
        os.environ["HTTP_PROXY"] = settings.network.http_proxy
        os.environ["http_proxy"] = settings.network.http_proxy
    if settings.network.https_proxy:
        os.environ["HTTPS_PROXY"] = settings.network.https_proxy
        os.environ["https_proxy"] = settings.network.https_proxy
    if settings.network.no_proxy:
        os.environ["NO_PROXY"] = settings.network.no_proxy
        os.environ["no_proxy"] = settings.network.no_proxy
    if settings.network.ca_bundle:
        os.environ["REQUESTS_CA_BUNDLE"] = settings.network.ca_bundle
        os.environ["SSL_CERT_FILE"] = settings.network.ca_bundle

    # Validate before persisting CLI overrides, so invalid flags never poison
    # settings.json.
    errors = settings.validate()
    if errors:
        for error in errors:
            print(f"Configuration error: {error}")
        sys.exit(1)

    # Save CLI args to settings.json if non-default values provided
    if _has_cli_overrides(args):
        try:
            settings.save()
            print("Saved CLI arguments to settings.json")
        except Exception as e:
            print(f"Warning: Failed to save settings: {e}")

    # Configure file logging (writes to {base_path}/logs/server.log)
    log_dir = settings.logging.get_log_dir(settings.base_path)
    configure_file_logging(
        log_dir=log_dir,
        level=settings.server.log_level,
        include_request_id=True,
        retention_days=settings.logging.retention_days,
    )
    print(f"Log directory: {log_dir}")

    # Enable native crash diagnostics (SIGABRT, SIGSEGV, SIGFPE, SIGBUS).
    # On Metal/MLX crashes (#511, #520), this dumps all Python thread
    # tracebacks to the server log before the process terminates.
    crash_log_path = log_dir / "crash.log"
    _crash_file = open(crash_log_path, "a")
    faulthandler.enable(file=_crash_file, all_threads=True)

    # Bind the socket before importing/initializing the server. Uvicorn's
    # normal startup runs ASGI lifespan before binding host/port, which means
    # pinned models can be preloaded before a port conflict is detected.
    print(f"Binding server at http://{settings.server.host}:{settings.server.port}")
    # uvicorn does not support "trace" — map to "debug" for its internal logging
    uvicorn_level = (
        "debug" if settings.server.log_level == "trace" else settings.server.log_level
    )
    # Only show access logs at trace level
    show_access_log = settings.server.log_level == "trace"
    uvicorn_config = uvicorn.Config(
        "omlx.server:app",
        host=settings.server.host,
        port=settings.server.port,
        log_level=uvicorn_level,
        access_log=show_access_log,
    )
    serve_socket = uvicorn_config.bind_socket()

    try:
        # Import server and config after the port is known to be available.
        from .server import init_server
        from .config import parse_size

        model_dirs = settings.get_effective_model_dirs()
        print(f"Base path: {settings.base_path}")
        print(f"Model directories: {', '.join(str(d) for d in model_dirs)}")
        print(f"Memory guard tier: {settings.memory.memory_guard_tier}")

        # Store MCP config path for FastAPI startup
        # Priority: CLI arg > settings.json
        mcp_config = args.mcp_config or settings.mcp.config_path
        if mcp_config:
            print(f"MCP config: {mcp_config}")
            os.environ["OMLX_MCP_CONFIG"] = mcp_config

        # Determine paged SSD cache directory
        # Priority: --no-cache > CLI arg > settings file
        if args.no_cache:
            paged_ssd_cache_dir = None
        elif args.paged_ssd_cache_dir:
            # CLI argument takes precedence
            paged_ssd_cache_dir = args.paged_ssd_cache_dir
        elif settings.cache.enabled:
            # Use settings file value (resolved path or default)
            paged_ssd_cache_dir = str(
                settings.cache.get_ssd_cache_dir(settings.base_path)
            )
        else:
            # Cache explicitly disabled in settings
            paged_ssd_cache_dir = None

        # Build scheduler config for BatchedEngine
        scheduler_config = settings.to_scheduler_config()
        # Set paged SSD cache options
        scheduler_config.paged_ssd_cache_dir = paged_ssd_cache_dir
        # Determine cache max size: CLI arg > settings (with auto resolution)
        if paged_ssd_cache_dir:
            if args.paged_ssd_cache_max_size:
                # CLI argument specified explicitly
                cache_max_size_bytes = parse_size(args.paged_ssd_cache_max_size)
            else:
                # Use settings value (handles "auto" -> 10% of SSD capacity)
                cache_max_size_bytes = settings.cache.get_ssd_cache_max_size_bytes(
                    settings.base_path
                )
            scheduler_config.paged_ssd_cache_max_size = cache_max_size_bytes
        else:
            scheduler_config.paged_ssd_cache_max_size = 0
            cache_max_size_bytes = 0

        # Hot cache: CLI arg > settings
        if paged_ssd_cache_dir:
            if args.hot_cache_max_size:
                hot_cache_max_bytes = parse_size(args.hot_cache_max_size)
            else:
                hot_cache_max_bytes = settings.cache.get_hot_cache_max_size_bytes()
            scheduler_config.hot_cache_max_size = hot_cache_max_bytes
        else:
            scheduler_config.hot_cache_max_size = 0

        if args.no_cache:
            print(
                "Mode: Multi-model serving (no oMLX cache, mlx-lm BatchGenerator only)"
            )
        elif paged_ssd_cache_dir:
            print("Mode: Multi-model serving (continuous batching + paged SSD cache)")
            # Format cache size for display
            cache_max_size_display = f"{cache_max_size_bytes / (1024**3):.1f}GB"
            print(
                f"paged SSD cache: {paged_ssd_cache_dir} (max: {cache_max_size_display})"
            )
            if scheduler_config.hot_cache_max_size > 0:
                hot_display = f"{scheduler_config.hot_cache_max_size / (1024**3):.1f}GB"
                print(f"Hot cache: {hot_display} (in-memory)")
        else:
            print("Mode: Multi-model serving (continuous batching, no cache)")

        # Set MLX buffer cache limit high to prevent the allocator from
        # immediately releasing Metal buffers when the cache is full.
        # Without this, allocator::free() can call buf->release() while the
        # GPU is still using the buffer, causing kernel panics on M4.
        # With a large cache limit, freed buffers always stay in the pool
        # and are only released via mx.clear_cache() (which we protect
        # with mx.synchronize()). See issue #300.
        import mlx.core as mx

        total_mem = mx.device_info().get("memory_size", 0)
        if total_mem > 0:
            mx.set_cache_limit(total_mem)

        # Initialize server
        # Note: pinned_models and default_model are managed via admin page (model_settings.json)
        # Sampling parameters (max_tokens, temperature, etc.) are per-model settings
        init_server(
            model_dirs=[str(d) for d in model_dirs],
            scheduler_config=scheduler_config,
            api_key=settings.auth.api_key,
            global_settings=settings,
        )

        print(
            f"Starting server at http://{settings.server.host}:{settings.server.port}"
        )
        try:
            uvicorn.Server(uvicorn_config).run(sockets=[serve_socket])
        except KeyboardInterrupt:
            pass
    finally:
        # Uvicorn closes sockets during normal shutdown; this covers failures
        # after bind succeeds but before the server takes ownership.
        serve_socket.close()


def launch_command(args, extra_args: list[str] | None = None):
    """Launch an external tool integrated with oMLX.

    extra_args are unknown CLI tokens forwarded to the underlying tool binary
    (e.g. ``-r`` / ``--resume <id>`` for Claude Code).
    """
    import requests

    from .integrations import IntegrationContext, get_integration, list_integrations
    from .settings import GlobalSettings

    def _optional_str(value) -> str | None:
        return value if isinstance(value, str) and value else None

    tool_name = args.tool

    if tool_name == "list":
        print("Available integrations:")
        for integ in list_integrations():
            installed = "installed" if integ.is_installed() else "not installed"
            print(f"  {integ.name:12s} {integ.display_name} ({installed})")
        return

    integration = get_integration(tool_name)
    if integration is None:
        print(f"Unknown integration: {tool_name}")
        print("Available: " + ", ".join(i.name for i in list_integrations()))
        sys.exit(1)

    # Resolve host/port: CLI args > env vars > settings.json > defaults
    settings = GlobalSettings.load()
    host = args.host or settings.server.host
    port = args.port or settings.server.port

    # 0.0.0.0 is a valid bind address but not a valid connect address.
    # Fall back to localhost so launch can reach the server regardless
    # of which interface it was bound to.
    connect_host = host if host and host != "0.0.0.0" else "127.0.0.1"

    # Check if oMLX server is running
    base_url = f"http://{connect_host}:{port}"
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        resp.raise_for_status()
    except Exception:
        print(f"oMLX server is not running at {base_url}")
        print("Start the server first: omlx start")
        sys.exit(1)

    # Get API key: CLI args > settings.json > empty
    api_key = getattr(args, "api_key", None) or settings.auth.api_key or ""

    claude_settings = getattr(settings, "claude_code", None)
    cli_opus_model = _optional_str(getattr(args, "opus_model", None))
    cli_sonnet_model = _optional_str(getattr(args, "sonnet_model", None))
    cli_haiku_model = _optional_str(getattr(args, "haiku_model", None))
    settings_opus_model = _optional_str(getattr(claude_settings, "opus_model", None))
    settings_sonnet_model = _optional_str(
        getattr(claude_settings, "sonnet_model", None)
    )
    settings_haiku_model = _optional_str(getattr(claude_settings, "haiku_model", None))
    opus_model = cli_opus_model or settings_opus_model
    sonnet_model = cli_sonnet_model or settings_sonnet_model
    haiku_model = cli_haiku_model or settings_haiku_model

    # Build headers for authenticated requests
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Pre-fetch model status (context_window, max_tokens, model_type per model)
    models_status_map: dict[str, dict] = {}
    try:
        resp = requests.get(f"{base_url}/v1/models/status", headers=headers, timeout=5)
        if resp.ok:
            for m in resp.json().get("models", []):
                if m_id := m.get("id"):
                    models_status_map[m_id] = m
                if model_alias := m.get("model_alias"):
                    models_status_map[model_alias] = m
    except Exception:
        pass

    # Determine model. Explicit CLI tier flags bypass the picker; otherwise always
    # prompt interactively so the user's selection is honoured.
    model = args.model
    if not model and (cli_opus_model or cli_sonnet_model or cli_haiku_model):
        model = cli_sonnet_model or cli_opus_model or cli_haiku_model or ""
    elif not model:
        # Fetch available models from server
        try:
            resp = requests.get(f"{base_url}/v1/models", headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = [
                m["id"]
                for m in data.get("data", [])
                if m.get("model_type") in ("llm", "vlm", None)
            ]
        except Exception:
            models = []

        if not models:
            print("No models available. Load a model first.")
            sys.exit(1)

        if len(models) == 1:
            model = models[0]
            print(f"Using model: {model}")
        else:
            models_info_list = [
                {"id": m_id, **models_status_map.get(m_id, {})} for m_id in models
            ]
            model = integration.select_model(models_info_list, integration.display_name)

    # Check if tool is installed
    if not integration.is_installed():
        print(f"{integration.display_name} is not installed.")
        print(f"Install: {integration.install_hint}")
        sys.exit(1)

    # If the model was chosen interactively (no --model and no explicit tier flags),
    # use the picked model for all tiers instead of letting settings-based tier
    # models override the user's selection.
    if args.model is None and not (cli_opus_model or cli_sonnet_model or cli_haiku_model):
        opus_model = None
        sonnet_model = None
        haiku_model = None

    # Resolve model limits from pre-fetched status
    model_info = models_status_map.get(model, {})
    ctx = IntegrationContext(
        host=connect_host,
        port=port,
        api_key=api_key,
        model=model,
        opus_model=opus_model if tool_name == "claude" else None,
        sonnet_model=sonnet_model if tool_name == "claude" else None,
        haiku_model=haiku_model if tool_name == "claude" else None,
        context_window=model_info.get("max_context_window"),
        max_tokens=model_info.get("max_tokens"),
        model_type=model_info.get("model_type"),
        reasoning=model_info.get("enable_thinking"),
        tools_profile=getattr(args, "tools_profile", "coding"),
        extra_args=tuple(extra_args or ()),
    )

    # Launch
    print(f"Launching {integration.display_name} with model {model}...")
    integration.launch(ctx)


def _app_control_socket_path():
    from pathlib import Path

    return Path.home() / "Library" / "Application Support" / "oMLX" / "control.sock"


def _app_bundle_path():
    from pathlib import Path

    from .utils.install import get_app_bundle_cli_path

    cli_path = get_app_bundle_cli_path()
    try:
        return cli_path.parents[2]
    except IndexError:
        return Path("/Applications/oMLX.app")


def _open_macos_app() -> None:
    import subprocess

    app_path = _app_bundle_path()
    subprocess.run(
        ["/usr/bin/open", "-gj", str(app_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _send_app_control(command: str, timeout: float = 2.0) -> dict:
    import json
    import socket

    sock_path = _app_control_socket_path()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(sock_path))
        sock.sendall(json.dumps({"command": command}).encode("utf-8") + b"\n")
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    raw = b"".join(chunks).split(b"\n", 1)[0]
    return json.loads(raw.decode("utf-8"))


def _send_app_control_with_launch(command: str, timeout: float) -> dict:
    import time

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    _open_macos_app()
    while time.monotonic() < deadline:
        try:
            return _send_app_control(command)
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"Could not reach oMLX.app control socket: {last_error}")


def _wait_app_control_state(states: set[str], timeout: float) -> dict:
    import time

    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = _send_app_control("status")
        if last.get("state") in states:
            return last
        time.sleep(0.5)
    return last


def _run_brew_services(command: str) -> int:
    import shutil
    import subprocess

    brew = shutil.which("brew")
    if not brew:
        print("Homebrew is not available on PATH.")
        return 1
    result = subprocess.run([brew, "services", command, "omlx"])
    return result.returncode


def lifecycle_command(args) -> int:
    """Run background lifecycle commands for the current installation."""
    from .utils.install import is_app_bundle, is_homebrew

    command = args.command
    timeout = getattr(args, "timeout", 60.0)
    no_wait = getattr(args, "no_wait", False)

    if is_app_bundle():
        try:
            if command == "stop":
                try:
                    response = _send_app_control(command)
                except OSError:
                    print("oMLX stopped")
                    return 0
            else:
                response = _send_app_control_with_launch(command, timeout=timeout)
            if not response.get("ok"):
                print(response.get("message") or f"oMLX {command} failed")
                return 1

            if command in {"start", "restart"} and not no_wait:
                response = _wait_app_control_state({"running", "unresponsive"}, timeout)
                if response.get("state") not in {"running", "unresponsive"}:
                    print(
                        f"oMLX server is {response.get('state', 'unknown')} "
                        f"after {int(timeout)}s."
                    )
                    return 1

            if command == "stop":
                print("oMLX stopped")
            elif command == "start":
                print(
                    f"oMLX server {response.get('state')} on port {response.get('port')}"
                )
            elif command == "restart":
                print(f"oMLX server restarted on port {response.get('port')}")
            return 0
        except Exception as exc:
            print(f"Failed to control oMLX.app: {exc}")
            return 1

    if is_homebrew():
        mapping = {"start": "start", "stop": "stop", "restart": "restart"}
        return _run_brew_services(mapping[command])

    if command == "start":
        print("Background start is available for the macOS app and Homebrew installs.")
        print("For this install, run foreground server mode with: omlx serve")
    else:
        print("Background stop/restart requires the macOS app or Homebrew service.")
    return 1


def diagnose_menubar() -> int:
    """Diagnose why the oMLX menubar icon might be missing.

    Reports macOS version, app install path, running menubar process, and the
    most recent visibility warning from the log. Prints manual recovery steps
    since Tahoe's ControlCenter doesn't expose a public API to re-enable a
    hidden status item.
    """
    import platform
    import subprocess
    from pathlib import Path

    print("oMLX menubar diagnostics")
    print("=" * 40)

    mac_ver = platform.mac_ver()[0] or "unknown"
    print(f"macOS:          {mac_ver}")
    print(f"Bundle ID:      app.omlx")

    app_path = Path("/Applications/oMLX.app")
    print(f"App installed:  {'yes' if app_path.exists() else 'NO (install DMG first)'}")

    try:
        res = subprocess.run(
            ["pgrep", "-af", "oMLX"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        running = bool(res.stdout.strip())
        print(f"Menubar app:    {'running' if running else 'NOT running'}")
        if running:
            first_line = res.stdout.strip().splitlines()[0]
            pid = first_line.split()[0] if first_line else "?"
            print(f"PID:            {pid}")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Menubar app:    check failed ({e})")

    # The Swift app writes `server.log` (stdout/stderr of the Python child).
    # No separate menubar.log — visibility-probe lines are logged into the
    # same file via OSLog.
    log_dir = Path.home() / "Library" / "Application Support" / "oMLX" / "logs"
    log_candidates = [log_dir / "server.log"]
    print(f"Log dir:        {log_dir}")

    hits: list[tuple[str, str]] = []
    for path in log_candidates:
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 131072))
                tail = f.read().decode("utf-8", errors="replace")
        except OSError as e:
            print(f"Could not read {path.name}: {e}")
            continue
        for ln in tail.splitlines():
            if (
                "menubar visibility probe" in ln
                or "NSStatusItem" in ln
                or "ControlCenter" in ln
                or "Menu Bar" in ln
            ):
                hits.append((path.name, ln))

    if hits:
        print("\nRecent visibility log entries (last 10):")
        for src, ln in hits[-10:]:
            print(f"  [{src}] {ln}")
    else:
        print("\nNo visibility log entries found (app may not have probed yet).")

    print()
    print("If the icon is missing on macOS Tahoe (26.x):")
    print("  1. Open System Settings > Menu Bar")
    print(
        "     open 'x-apple.systempreferences:com.apple.ControlCenter-Settings.extension?MenuBar'"
    )
    print("  2. Find 'oMLX' and set it to 'Show in Menu Bar'")
    print("  3. If oMLX isn't in the list, quit the app and relaunch oMLX.app")
    print()
    print("Note: Apple's sandbox policy prevents third-party apps from")
    print("programmatically re-enabling their own menubar visibility on Tahoe.")
    return 0


def diagnose_command(args) -> int:
    """Dispatch 'omlx diagnose <target>' to the appropriate subcommand."""
    target = getattr(args, "target", None)
    if target == "menubar":
        return diagnose_menubar()
    print(f"Unknown diagnose target: {target}")
    print("Available: menubar")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="omlx: Production-ready LLM server for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  omlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
  omlx launch codex --model qwen3.5
        """,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print the oMLX version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    for name, help_text in (
        ("start", "Start oMLX as a managed background server"),
        ("stop", "Stop the managed background oMLX server"),
        ("restart", "Restart the managed background oMLX server"),
    ):
        lifecycle_parser = subparsers.add_parser(
            name,
            help=help_text,
            description=help_text,
        )
        lifecycle_parser.add_argument(
            "--timeout",
            type=float,
            default=60.0,
            help="Seconds to wait for the macOS app/server to reach the requested state",
        )
        if name in {"start", "restart"}:
            lifecycle_parser.add_argument(
                "--no-wait",
                action="store_true",
                help="Return after sending the request without waiting for server health",
            )

    # Serve command (multi-model)
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start multi-model OpenAI-compatible server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Start a multi-model inference server with LRU-based memory management.

Models are discovered from subdirectories of --model-dir. Each subdirectory
should contain a valid model with config.json and *.safetensors files.

Example directory structure:
  /path/to/models/
  ├── llama-3b/           → model_id: "llama-3b"
  │   ├── config.json
  │   └── model.safetensors
  ├── qwen-7b/            → model_id: "qwen-7b"
  └── mistral-7b/         → model_id: "mistral-7b"
""",
    )

    # Required arguments
    serve_parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing model subdirectories (default: ~/.omlx/models)",
    )
    # Server options
    serve_parser.add_argument(
        "--host", type=str, default=None, help="Host to bind (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=None, help="Port to bind (default: 8000)"
    )
    serve_parser.add_argument(
        "--log-level",
        type=str,
        choices=["trace", "debug", "info", "warning", "error"],
        default=None,
        help="Log level (default: info). trace includes full message content",
    )
    serve_parser.add_argument(
        "--sse-keepalive-mode",
        type=str,
        choices=["chunk", "comment", "off"],
        default=None,
        help="SSE keepalive emission mode (default: chunk). 'chunk' emits "
        "protocol-aware no-op events compatible with strict clients like "
        "OpenClaw / WorkBuddy; 'comment' emits the legacy ': keep-alive' SSE "
        "comment; 'off' disables keepalive entirely",
    )

    # Scheduler options (for BatchedEngine)
    serve_parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=None,
        help="Max requests processed simultaneously. Higher values increase throughput but use more memory. (default: 8)",
    )
    serve_parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Max embedding inputs processed in one forward pass. Higher values increase throughput but use more memory. (default: 32)",
    )

    # Memory guard options
    serve_parser.add_argument(
        "--memory-guard",
        type=str,
        choices=["safe", "balanced", "aggressive"],
        default=None,
        help="Memory guard tier. safe reserves more system memory; aggressive allows more oMLX memory use. (default: balanced)",
    )
    serve_parser.add_argument(
        "--memory-guard-gb",
        type=_positive_float,
        default=None,
        help="Custom memory guard ceiling in GB. Sets memory guard tier to custom.",
    )

    # paged SSD cache options
    serve_parser.add_argument(
        "--paged-ssd-cache-dir",
        type=str,
        default=None,
        help="Directory for paged SSD cache storage (enables oMLX prefix cache)",
    )
    serve_parser.add_argument(
        "--paged-ssd-cache-max-size",
        type=str,
        default=None,
        help="Maximum paged SSD cache size (e.g., '100GB', '50GB'). Default: 100GB",
    )
    serve_parser.add_argument(
        "--hot-cache-max-size",
        type=str,
        default=None,
        help="Maximum in-memory hot cache size (e.g., '8GB', '4GB'). Default: 0 (disabled)",
    )
    serve_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable oMLX paged SSD cache. mlx-lm BatchGenerator still manages KV states internally.",
    )
    serve_parser.add_argument(
        "--initial-cache-blocks",
        type=int,
        default=None,
        help="Number of cache blocks to pre-allocate at startup (default: 256). "
        "Higher values reduce dynamic allocation overhead for large contexts.",
    )

    # MCP options
    serve_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML) for tool integration",
    )

    # HuggingFace options
    serve_parser.add_argument(
        "--hf-endpoint",
        type=str,
        default=None,
        help="Custom HuggingFace Hub endpoint URL (e.g., https://hf-mirror.com)",
    )
    serve_parser.add_argument(
        "--hf-cache",
        dest="hf_cache_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Discover models from the standard HuggingFace Hub local cache (default: enabled)",
    )

    # ModelScope options
    serve_parser.add_argument(
        "--ms-endpoint",
        type=str,
        default=None,
        help="Custom ModelScope Hub endpoint URL",
    )

    # Network options
    serve_parser.add_argument(
        "--http-proxy",
        type=str,
        default=None,
        help="HTTP proxy URL (e.g., http://proxy.company.com:8080)",
    )
    serve_parser.add_argument(
        "--https-proxy",
        type=str,
        default=None,
        help="HTTPS proxy URL (e.g., http://proxy.company.com:8080)",
    )
    serve_parser.add_argument(
        "--no-proxy",
        type=str,
        default=None,
        help="Comma-separated hosts/IPs to bypass proxy (e.g., localhost,127.0.0.1)",
    )
    serve_parser.add_argument(
        "--ca-bundle",
        type=str,
        default=None,
        help="Path to CA bundle PEM file for TLS interception environments",
    )

    # Base path and auth
    serve_parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Base directory for oMLX data (default: ~/.omlx)",
    )
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (optional)",
    )

    # Launch command
    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch an external tool with oMLX integration",
        description="Configure and launch external coding tools (Claude Code, Copilot, Codex, OpenCode, OpenClaw, Hermes Agent, Pi) "
        "to use the running oMLX server.",
    )
    launch_parser.add_argument(
        "tool",
        type=str,
        help="Tool to launch: claude, copilot, codex, opencode, openclaw, hermes, pi, or 'list' to show available",
    )
    launch_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (interactive selection if not specified)",
    )
    launch_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="oMLX server host (default: from settings or 127.0.0.1)",
    )
    launch_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="oMLX server port (default: from settings or 8000)",
    )
    launch_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for oMLX server authentication",
    )
    launch_parser.add_argument(
        "--tools-profile",
        type=str,
        default="coding",
        choices=["minimal", "coding", "messaging", "full"],
        help="OpenClaw tools profile (default: coding)",
    )
    launch_parser.add_argument(
        "--opus",
        dest="opus_model",
        type=str,
        default=None,
        help="Claude Code Opus tier model (Claude integration only)",
    )
    launch_parser.add_argument(
        "--sonnet",
        dest="sonnet_model",
        type=str,
        default=None,
        help="Claude Code Sonnet tier model (Claude integration only)",
    )
    launch_parser.add_argument(
        "--haiku",
        dest="haiku_model",
        type=str,
        default=None,
        help="Claude Code Haiku tier model (Claude integration only)",
    )

    # Diagnose command
    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Diagnose installation or runtime issues",
        description="Run diagnostic checks and print recovery steps.",
    )
    diagnose_parser.add_argument(
        "target",
        type=str,
        choices=["menubar"],
        help="What to diagnose. 'menubar' checks Tahoe ControlCenter visibility.",
    )

    # Use parse_known_args so `omlx launch <tool> -- ...` can forward unknown
    # tokens (e.g. `-r`, `--resume <id>`) to the underlying tool binary.
    # Non-launch commands keep the previous strictness by rejecting unknowns.
    args, extra_args = parser.parse_known_args()

    if args.command == "launch":
        launch_command(args, extra_args=extra_args)
    else:
        if extra_args:
            parser.error(f"unrecognized arguments: {' '.join(extra_args)}")
        if args.command == "serve":
            serve_command(args)
        elif args.command in {"start", "stop", "restart"}:
            sys.exit(lifecycle_command(args))
        elif args.command == "diagnose":
            sys.exit(diagnose_command(args))
        else:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
