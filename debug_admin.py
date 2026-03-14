#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Debug admin panel launcher for oMLX.

A standalone admin panel server for UI development and debugging.
Does not require MLX or any model dependencies.

Usage:
    python debug_admin.py
    python debug_admin.py --port 8080
    python debug_admin.py --no-auth  # Skip authentication
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Mock MLX ecosystem BEFORE any imports
# =============================================================================

# Create a comprehensive mock system
class MockModule:
    """A mock module that handles any attribute access gracefully."""
    
    def __init__(self, name):
        self.__name__ = name
        self.__file__ = None
        self.__path__ = []
        self.__package__ = name
    
    def __getattr__(self, name):
        # Return another mock module for any attribute
        return MockModule(f"{self.__name__}.{name}")
    
    def __call__(self, *args, **kwargs):
        # Allow being called as a function
        return None
    
    def __iter__(self):
        return iter([])


def install_mock_module(name: str):
    """Install a mock module and return it."""
    mod = MockModule(name)
    sys.modules[name] = mod
    return mod


# Install all MLX-related mocks
mlx_mock = install_mock_module("mlx")
install_mock_module("mlx.core")
install_mock_module("mlx.utils")
install_mock_module("mlx.metal")
install_mock_module("mlx.nn")
install_mock_module("mlx.optimizers")

install_mock_module("mlx_lm")
install_mock_module("mlx_lm.utils")
install_mock_module("mlx_lm.generate")
install_mock_module("mlx_lm.sample_utils")
install_mock_module("mlx_lm.tokenizer")
install_mock_module("mlx_lm.tokenizer_utils")
install_mock_module("mlx_lm.models")
install_mock_module("mlx_lm.cache")
install_mock_module("mlx_lm.tuner")
install_mock_module("mlx_lm.lora")

install_mock_module("mlx_vlm")
install_mock_module("mlx_vlm.utils")
install_mock_module("mlx_vlm.generate")
install_mock_module("mlx_vlm.models")

install_mock_module("mlx_embeddings")
install_mock_module("mlx_embeddings.utils")

# Mock numpy with basic functionality
numpy_mock = install_mock_module("numpy")

# NOTE: Do NOT mock huggingface_hub, requests, modelscope - we want real download functionality

# Create a mock omlx package to prevent importing the real __init__.py
# which imports scheduler (which requires mlx)
omlx_mock = install_mock_module("omlx")
install_mock_module("omlx.config")
install_mock_module("omlx._version")
# Don't mock omlx.admin - we'll import it properly below
install_mock_module("omlx.scheduler")
install_mock_module("omlx.engine")
install_mock_module("omlx.engine_core")
install_mock_module("omlx.engine_pool")
install_mock_module("omlx.cache")
install_mock_module("omlx.cache.paged_cache")
install_mock_module("omlx.cache.prefix_cache")
install_mock_module("omlx.cache.paged_ssd_cache")
install_mock_module("omlx.cache.boundary_snapshot_store")
install_mock_module("omlx.memory_monitor")
install_mock_module("omlx.request")
install_mock_module("omlx.exceptions")
install_mock_module("omlx.models")
install_mock_module("omlx.models.base_model")
install_mock_module("omlx.models.llm")
install_mock_module("omlx.models.vlm")
install_mock_module("omlx.models.embedding")
install_mock_module("omlx.models.reranker")
install_mock_module("omlx.api")
install_mock_module("omlx.api.openai_models")
install_mock_module("omlx.api.anthropic_models")
install_mock_module("omlx.api.anthropic_utils")
install_mock_module("omlx.mcp")
install_mock_module("omlx.mcp.manager")
install_mock_module("omlx.mcp.client")
install_mock_module("omlx.mcp.executor")
install_mock_module("omlx.mcp.tools")
install_mock_module("omlx.output_collector")
install_mock_module("omlx.process_memory_enforcer")

# Create mock server_metrics with get_server_metrics function
server_metrics_mock = install_mock_module("omlx.server_metrics")


class MockServerMetrics:
    """Mock server metrics."""

    def get_snapshot(self, model_id=None, scope="session"):
        return {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_prompt_tokens": 0,
            "avg_tokens_per_second": 0,
            "avg_time_to_first_token": 0,
            "cache_hit_rate": 0,
        }

    def clear_metrics(self):
        pass

    def clear_alltime_metrics(self):
        pass


_mock_server_metrics = MockServerMetrics()
server_metrics_mock.get_server_metrics = lambda: _mock_server_metrics
server_metrics_mock.ServerMetrics = MockServerMetrics

install_mock_module("omlx.optimizations")
install_mock_module("omlx.model_discovery")
install_mock_module("omlx.model_registry")
install_mock_module("omlx.integrations")
install_mock_module("omlx.integrations.codex")
install_mock_module("omlx.integrations.opencode")
install_mock_module("omlx.integrations.openclaw")
install_mock_module("omlx.adapter")
install_mock_module("omlx.adapter.harmony")

# Now we can safely import omlx modules
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Import admin modules directly (avoiding omlx.__init__ which imports scheduler)
# =============================================================================

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import directly from file paths to avoid omlx.__init__
import importlib.util


def import_module_from_file(name: str, file_path: Path):
    """Import a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# First, import settings (it has no MLX dependencies)
settings_module = import_module_from_file("omlx.settings", project_root / "omlx" / "settings.py")
SubKeyEntry = settings_module.SubKeyEntry

# Create a proper omlx.admin package for relative imports to work
admin_package = type(sys)("omlx.admin")
admin_package.__file__ = str(project_root / "omlx" / "admin" / "__init__.py")
admin_package.__path__ = [str(project_root / "omlx" / "admin")]
admin_package.__package__ = "omlx.admin"
sys.modules["omlx.admin"] = admin_package

# Import admin submodules in order (dependencies first)
# Import hf_downloader (used by ms_downloader)
hf_downloader_module = import_module_from_file("omlx.admin.hf_downloader", project_root / "omlx" / "admin" / "hf_downloader.py")

# Import ms_downloader
ms_downloader_module = import_module_from_file("omlx.admin.ms_downloader", project_root / "omlx" / "admin" / "ms_downloader.py")

# Import auth module
auth_module = import_module_from_file("omlx.admin.auth", project_root / "omlx" / "admin" / "auth.py")
create_session_token = auth_module.create_session_token
require_admin = auth_module.require_admin
verify_api_key = auth_module.verify_api_key
_RedirectToLogin = auth_module._RedirectToLogin

# Import routes module (depends on above modules)
routes_module = import_module_from_file("omlx.admin.routes", project_root / "omlx" / "admin" / "routes.py")
admin_router = routes_module.router
set_admin_getters = routes_module.set_admin_getters
set_hf_downloader = routes_module.set_hf_downloader
set_ms_downloader = routes_module.set_ms_downloader

# =============================================================================
# Mock Data Classes
# =============================================================================


class MockModelSettings:
    """Mock per-model settings."""

    def __init__(self, model_id: str):
        self.model_alias = f"alias_{model_id}"
        self.model_type_override = None
        self.max_context_window = 8192
        self.max_tokens = 4096
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 40
        self.repetition_penalty = 1.0
        self.min_p = 0.0
        self.presence_penalty = 0.0
        self.force_sampling = False
        self.max_tool_result_tokens = 4096
        self.chat_template_kwargs = {}
        self.forced_ct_kwargs = []
        self.ttl_seconds = 3600
        self.is_pinned = False
        self.is_default = False
        self.display_name = model_id.replace("-", " ").title()
        self.description = f"Mock model for debugging: {model_id}"


class MockEnginePool:
    """Mock engine pool with fake model data."""

    def __init__(self):
        self.models = [
            {
                "id": "llama-3.2-3b-instruct",
                "loaded": True,
                "is_loading": False,
                "estimated_size": 6_500_000_000,
                "pinned": True,
                "engine_type": "batched",
                "model_type": "llm",
                "config_model_type": "llama",
                "last_access": "2025-03-14T10:30:00Z",
            },
            {
                "id": "qwen2.5-7b-instruct",
                "loaded": True,
                "is_loading": False,
                "estimated_size": 14_200_000_000,
                "pinned": False,
                "engine_type": "batched",
                "model_type": "llm",
                "config_model_type": "qwen2",
                "last_access": "2025-03-14T09:15:00Z",
            },
            {
                "id": "mistral-nemo-12b-instruct",
                "loaded": False,
                "is_loading": False,
                "estimated_size": 24_000_000_000,
                "pinned": False,
                "engine_type": "batched",
                "model_type": "llm",
                "config_model_type": "mistral",
                "last_access": "2025-03-13T16:45:00Z",
            },
            {
                "id": "gemma-3-4b-it",
                "loaded": False,
                "is_loading": True,
                "estimated_size": 8_100_000_000,
                "pinned": False,
                "engine_type": "batched",
                "model_type": "llm",
                "config_model_type": "gemma3",
                "last_access": None,
            },
            {
                "id": "qwen2.5-vl-7b-instruct",
                "loaded": True,
                "is_loading": False,
                "estimated_size": 15_000_000_000,
                "pinned": False,
                "engine_type": "vlm",
                "model_type": "vlm",
                "config_model_type": "qwen2_vl",
                "last_access": "2025-03-14T08:00:00Z",
            },
            {
                "id": "bge-m3",
                "loaded": True,
                "is_loading": False,
                "estimated_size": 2_200_000_000,
                "pinned": False,
                "engine_type": "embedding",
                "model_type": "embedding",
                "config_model_type": "xlm-roberta",
                "last_access": "2025-03-14T11:00:00Z",
            },
        ]
        self._model_settings = {m["id"]: MockModelSettings(m["id"]) for m in self.models}

    def get_status(self) -> Dict[str, Any]:
        return {"models": self.models}

    def get_model(self, model_id: str) -> Optional[Dict]:
        for m in self.models:
            if m["id"] == model_id:
                return m
        return None


class MockSettingsManager:
    """Mock settings manager."""

    def __init__(self, engine_pool: MockEnginePool):
        self._engine_pool = engine_pool

    def get_all_settings(self) -> Dict[str, MockModelSettings]:
        return self._engine_pool._model_settings


class MockServerState:
    """Mock server state."""

    def __init__(self, engine_pool: MockEnginePool):
        self._engine_pool = engine_pool
        self.default_model = "llama-3.2-3b-instruct"


class MockGlobalSettings:
    """Mock global settings."""

    def __init__(self):
        self.base_path = Path.home() / ".omlx"
        self.server = type("ServerSettings", (), {"host": "0.0.0.0", "port": 8000, "log_level": "INFO"})()
        self.model = type(
            "ModelSettings",
            (),
            {
                "get_model_dirs": lambda self, base_path=None: [Path("/Users/mock/.omlx/models")],
                "get_model_dir": lambda self, base_path=None: Path("/Users/mock/.omlx/models"),
                "max_model_memory": "16GB",
                "model_fallback": True,
            },
        )()
        self.memory = type("MemorySettings", (), {"max_process_memory": "32GB"})()
        self.scheduler = type(
            "SchedulerSettings",
            (),
            {"max_num_seqs": 64, "completion_batch_size": 32},
        )()
        self.cache = type(
            "CacheSettings",
            (),
            {
                "enabled": True,
                "ssd_cache_dir": "/Users/mock/.omlx/cache",
                "get_ssd_cache_max_size_bytes": lambda self, base_path=None: 10 * 1024**3,
                "hot_cache_max_size": "2GB",
                "initial_cache_blocks": 256,
            },
        )()
        self.mcp = type("MCPSettings", (), {"config_path": ""})()
        self.huggingface = type("HFSettings", (), {"endpoint": ""})()
        self.modelscope = type("MSSettings", (), {"endpoint": ""})()
        self.sampling = type(
            "SamplingSettings",
            (),
            {
                "max_context_window": 8192,
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.0,
            },
        )()
        self.auth = type(
            "AuthSettings",
            (),
            {
                "api_key": "sk-debug-admin-key-12345",
                "skip_api_key_verification": False,
                "sub_keys": [],
            },
        )()
        self.claude_code = type(
            "ClaudeCodeSettings",
            (),
            {
                "context_scaling_enabled": False,
                "target_context_size": 128000,
                "mode": "auto",
                "opus_model": "claude-3-opus",
                "sonnet_model": "claude-3-sonnet",
                "haiku_model": "claude-3-haiku",
            },
        )()
        self.integrations = type(
            "IntegrationsSettings",
            (),
            {
                "codex_model": "",
                "opencode_model": "",
                "openclaw_model": "",
                "openclaw_tools_profile": "default",
            },
        )()
        self.ui = type("UISettings", (), {"language": "en"})()

    def save(self):
        logger.info("Mock: Settings saved")


# =============================================================================
# FastAPI Application
# =============================================================================

# Create mock instances for MLX-dependent components only
mock_engine_pool = MockEnginePool()
mock_settings_manager = MockSettingsManager(mock_engine_pool)
mock_server_state = MockServerState(mock_engine_pool)
mock_global_settings = MockGlobalSettings()

# Use REAL downloaders (imported from the actual modules)
HFDownloader = hf_downloader_module.HFDownloader
MSDownloader = ms_downloader_module.MSDownloader

# Create real downloader instances with a temp model directory
import tempfile
temp_model_dir = tempfile.mkdtemp(prefix="omlx_debug_")
real_hf_downloader = HFDownloader(model_dir=temp_model_dir)
real_ms_downloader = MSDownloader(model_dir=temp_model_dir)

# Create FastAPI app
app = FastAPI(title="oMLX Admin Debug Server")

# Setup paths
admin_dir = project_root / "omlx" / "admin"
templates = Jinja2Templates(directory=admin_dir / "templates")


# =============================================================================
# Helper Functions
# =============================================================================


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    elif size_bytes < 1024**4:
        return f"{size_bytes / 1024**3:.2f} GB"
    else:
        return f"{size_bytes / 1024**4:.2f} TB"


# =============================================================================
# Admin Router Setup
# =============================================================================

# Set up the mock getters
set_admin_getters(
    lambda: mock_server_state,
    lambda: mock_engine_pool,
    lambda: mock_settings_manager,
    lambda: mock_global_settings,
)
set_hf_downloader(real_hf_downloader)
set_ms_downloader(real_ms_downloader)

# Include the admin router
app.include_router(admin_router)


@app.exception_handler(_RedirectToLogin)
async def redirect_to_login_handler(request, exc):
    """Redirect unauthenticated browser requests to the admin login page."""
    return RedirectResponse(url="/admin", status_code=302)


# =============================================================================
# Additional Debug Routes
# =============================================================================


@app.get("/debug/status")
async def debug_status():
    """Debug endpoint to check mock data status."""
    return {
        "status": "ok",
        "models_count": len(mock_engine_pool.models),
        "loaded_models": sum(1 for m in mock_engine_pool.models if m["loaded"]),
        "hf_downloads": len(real_hf_downloader.get_tasks()),
        "ms_downloads": len(real_ms_downloader.get_tasks()),
    }


@app.get("/debug/models")
async def debug_models():
    """Debug endpoint to list all mock models."""
    return {"models": mock_engine_pool.models}


# =============================================================================
# Static Files Mount
# =============================================================================

static_dir = admin_dir / "static"
if static_dir.exists():
    app.mount("/admin/static", StaticFiles(directory=static_dir), name="admin-static")


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Debug admin panel for oMLX")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind (default: 8001)",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable authentication (auto-login)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_auth:
        # Patch the require_admin dependency to always pass
        from omlx.admin import auth

        async def mock_require_admin():
            return True

        auth.require_admin = mock_require_admin
        logger.warning("Authentication disabled - use --no-auth for development only!")

    logger.info(f"Starting debug admin server on http://{args.host}:{args.port}")
    logger.info(f"Admin panel: http://{args.host}:{args.port}/admin")
    logger.info(f"Debug API: http://{args.host}:{args.port}/debug/status")
    logger.info("")
    logger.info("Default API key for login: sk-debug-admin-key-12345")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
