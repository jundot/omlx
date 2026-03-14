#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI for oMLX.

Commands:
    omlx serve --model-dir /path/to/models    Start multi-model server

Usage:
    # Multi-model serving
    omlx serve --model-dir /path/to/models --max-model-memory 32GB

    # With pinned models
    omlx serve --model-dir /path/to/models --max-model-memory 48GB --pin llama-3b,qwen-7b
"""

import argparse
import sys


def _has_cli_overrides(args) -> bool:
    """Check if CLI args contain non-default values that should be saved."""
    # model_dir: default=None (don't save None)
    if hasattr(args, "model_dir") and args.model_dir is not None:
        return True
    # port: default=8000
    if hasattr(args, "port") and args.port != 8000:
        return True
    # max_model_memory: default="auto"
    if hasattr(args, "max_model_memory") and args.max_model_memory != "auto":
        return True
    # max_process_memory: default=None (don't save None)
    if hasattr(args, "max_process_memory") and args.max_process_memory is not None:
        return True
    # host: default="127.0.0.1"
    if hasattr(args, "host") and args.host != "127.0.0.1":
        return True
    # log_level: default="info"
    if hasattr(args, "log_level") and args.log_level != "info":
        return True
    return False


def serve_command(args):
    """Start the OpenAI-compatible multi-model server."""
    import logging
    import os
    import uvicorn

    from ._version import __version__
    from .settings import init_settings, get_settings
    from .logging_config import configure_file_logging

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
    log_level = TRACE if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Set omlx loggers
    for name in ["omlx", "omlx.scheduler", "omlx.paged_ssd_cache",
                 "omlx.memory_monitor", "omlx.paged_cache", "omlx.prefix_cache",
                 "omlx.engine_pool", "omlx.model_discovery"]:
        logging.getLogger(name).setLevel(log_level)

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

    # Validate settings
    errors = settings.validate()
    if errors:
        for error in errors:
            print(f"Configuration error: {error}")
        sys.exit(1)

    # Import server and config
    from .server import app, init_server
    from .config import parse_size

    model_dirs = settings.model.get_model_dirs(settings.base_path)
    print(f"Base path: {settings.base_path}")
    print(f"Model directories: {', '.join(str(d) for d in model_dirs)}")
    print(f"Max model memory: {settings.model.max_model_memory}")
    print(f"Max process memory: {settings.memory.max_process_memory}")

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
        paged_ssd_cache_dir = str(settings.cache.get_ssd_cache_dir(settings.base_path))
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
            cache_max_size_bytes = settings.cache.get_ssd_cache_max_size_bytes(settings.base_path)
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
        print("Mode: Multi-model serving (no oMLX cache, mlx-lm BatchGenerator only)")
    elif paged_ssd_cache_dir:
        print("Mode: Multi-model serving (continuous batching + paged SSD cache)")
        # Format cache size for display
        cache_max_size_display = f"{cache_max_size_bytes / (1024**3):.1f}GB"
        print(f"paged SSD cache: {paged_ssd_cache_dir} (max: {cache_max_size_display})")
        if scheduler_config.hot_cache_max_size > 0:
            hot_display = f"{scheduler_config.hot_cache_max_size / (1024**3):.1f}GB"
            print(f"Hot cache: {hot_display} (in-memory)")
    else:
        print("Mode: Multi-model serving (continuous batching, no cache)")

    # Initialize server
    # Note: pinned_models and default_model are managed via admin page (model_settings.json)
    # Sampling parameters (max_tokens, temperature, etc.) are per-model settings
    init_server(
        model_dirs=[str(d) for d in model_dirs],
        max_model_memory=settings.model.get_max_model_memory_bytes(),
        scheduler_config=scheduler_config,
        api_key=settings.auth.api_key,
        global_settings=settings,
    )

    # Start server
    print(f"Starting server at http://{settings.server.host}:{settings.server.port}")
    # uvicorn does not support "trace" — map to "debug" for its internal logging
    uvicorn_level = "debug" if settings.server.log_level == "trace" else settings.server.log_level
    # Only show access logs at trace level
    show_access_log = settings.server.log_level == "trace"
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level=uvicorn_level,
        access_log=show_access_log,
    )



def launch_command(args):
    """Launch an external tool integrated with oMLX."""
    import requests

    from .integrations import get_integration, list_integrations
    from .settings import GlobalSettings

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

    # Check if oMLX server is running
    base_url = f"http://{host}:{port}"
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        resp.raise_for_status()
    except Exception:
        print(f"oMLX server is not running at {base_url}")
        print("Start the server first: omlx serve")
        sys.exit(1)

    # Get API key from CLI args
    api_key = getattr(args, "api_key", None) or ""

    # Build headers for authenticated requests
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Determine model
    model = args.model
    if not model:
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
            print("Available models:")
            for i, m in enumerate(models, 1):
                print(f"  {i}. {m}")
            while True:
                try:
                    choice = input("Select model number: ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(models):
                        model = models[idx]
                        break
                    print(f"Please enter 1-{len(models)}")
                except (ValueError, EOFError):
                    print(f"Please enter 1-{len(models)}")

    # Check if tool is installed
    if not integration.is_installed():
        print(f"{integration.display_name} is not installed.")
        print(f"Install: {integration.install_hint}")
        sys.exit(1)

    # Launch
    print(f"Launching {integration.display_name} with model {model}...")
    tools_profile = getattr(args, "tools_profile", "coding")
    integration.launch(
        port=port, api_key=api_key, model=model, host=host, tools_profile=tools_profile
    )


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
    subparsers = parser.add_subparsers(dest="command", help="Commands")

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
    serve_parser.add_argument(
        "--max-model-memory",
        type=str,
        default=None,
        help="Maximum memory for loaded models (e.g., 32GB, 'disabled'). Default: 80%% of system memory.",
    )
    serve_parser.add_argument(
        "--max-process-memory",
        type=str,
        default=None,
        help=(
            "Max total process memory as percentage of system RAM (10-99%%), "
            "'auto' (RAM - 8GB), or 'disabled'. Default: auto."
        ),
    )

    # Server options
    serve_parser.add_argument("--host", type=str, default=None, help="Host to bind (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=None, help="Port to bind (default: 8000)")
    serve_parser.add_argument(
        "--log-level",
        type=str,
        choices=["trace", "debug", "info", "warning", "error"],
        default=None,
        help="Log level (default: info). trace includes full message content",
    )

    # Scheduler options (for BatchedEngine)
    serve_parser.add_argument(
        "--max-num-seqs", type=int, default=None, help="Max concurrent sequences (default: 256)"
    )
    serve_parser.add_argument(
        "--completion-batch-size",
        type=int,
        default=None,
        help="Max sequences for mlx-lm BatchGenerator completion phase (token generation). (default: 32)",
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

    # ModelScope options
    serve_parser.add_argument(
        "--ms-endpoint",
        type=str,
        default=None,
        help="Custom ModelScope Hub endpoint URL",
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
        description="Configure and launch external coding tools (Codex, OpenCode, OpenClaw) "
        "to use the running oMLX server.",
    )
    launch_parser.add_argument(
        "tool",
        type=str,
        help="Tool to launch: codex, opencode, openclaw, or 'list' to show available",
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

    args = parser.parse_args()

    if args.command == "serve":
        serve_command(args)
    elif args.command == "launch":
        launch_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
