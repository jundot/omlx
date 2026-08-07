"""Codex (OpenAI Codex CLI) integration."""

from __future__ import annotations

from pathlib import Path

from omlx.codex_interceptor.manager import (
    CodexInterceptorConfig,
    CodexInterceptorManager,
)
from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_command_prefix

CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"


def interceptor_config(
    ctx: IntegrationContext, *, launch_app: bool
) -> CodexInterceptorConfig:
    """Translate an integration context without mutating Codex settings."""
    if not ctx.model:
        raise ValueError("select a local model before launching Codex")
    return CodexInterceptorConfig(
        model=ctx.model,
        upstream_url=ctx.openai_base_url.rstrip("/") + "/responses",
        api_key=ctx.auth_token,
        auth_header=bool(ctx.auth_token),
        project=Path.cwd(),
        local_label=f"Local · oMLX · {ctx.model.rsplit('/', 1)[-1]}",
        launch_app=launch_app,
    )


class CodexIntegration(Integration):
    """Codex CLI integration using a transparent, process-scoped proxy."""

    CONFIG_PATH = CODEX_CONFIG_PATH

    def __init__(self):
        super().__init__(
            name="codex",
            display_name="Codex",
            type="environment",
            install_check="codex",
            install_hint="npm install -g @openai/codex",
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        return (
            f"{get_cli_command_prefix()} "
            f"launch codex --model {ctx.model or 'select-a-model'}"
        )

    def configure(self, ctx: IntegrationContext) -> None:
        # Intentionally a no-op: changing ~/.codex/config.toml disables or
        # alters native Codex features. Routing is scoped to the child process.
        return None

    def launch(self, ctx: IntegrationContext) -> None:
        manager = CodexInterceptorManager()
        raise SystemExit(
            manager.run_cli(interceptor_config(ctx, launch_app=False), ctx.extra_args)
        )
