"""Swival integration."""

from __future__ import annotations

import os

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_command_prefix


class SwivalIntegration(Integration):
    """Swival integration that launches the swival tool with model configuration."""

    def __init__(self):
        super().__init__(
            name="swival",
            display_name="Swival",
            type="env_var",
            install_check="swival",
            install_hint="Install swival by following instructions at https://github.com/your-repo/swival",
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        # Only flags `omlx launch` itself accepts belong here; the provider /
        # base-url / context wiring is applied to the real swival binary in
        # launch() below.
        return (
            f"{get_cli_command_prefix()} "
            f"launch swival --model {ctx.model or 'select-a-model'}"
        )

    def launch(self, ctx: IntegrationContext) -> None:
        # Set up environment for swival
        env = self._scrubbed_env()
        
        # Build the command arguments
        args = ["swival"]
        args.extend(["--provider", "generic"])
        args.extend(["--base-url", ctx.base_url])
        
        # Add API key if provided
        if ctx.api_key:
            args.extend(["--api-key", ctx.api_key])
        
        # Add model name (this is the required injection)
        if ctx.model:
            args.extend(["--model", ctx.model])
        
        # Add max context tokens
        if ctx.context_window:
            args.extend(["--max-context-tokens", str(ctx.context_window)])
            
        # Add any extra arguments
        args.extend(ctx.extra_args)
        
        os.execvpe("swival", args, env)