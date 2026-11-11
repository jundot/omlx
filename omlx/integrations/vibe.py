# SPDX-License-Identifier: Apache-2.0
"""Mistral Vibe integration using process-scoped configuration."""

from __future__ import annotations

import hashlib
import json
import os
import shlex

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_command_prefix


class VibeIntegration(Integration):
    """Launch Vibe against oMLX without rewriting the user's config.toml."""

    def __init__(self):
        super().__init__(
            name="vibe",
            display_name="Mistral Vibe",
            type="env_var",
            install_check="vibe",
            install_hint="uv tool install mistral-vibe",
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        model = shlex.quote(ctx.model or "select-a-model")
        return f"{get_cli_command_prefix()} launch vibe --model {model}"

    def launch(self, ctx: IntegrationContext) -> None:
        env = self._scrubbed_env()
        # Saved Vibe sessions refer to this alias. Keep it stable for the same
        # endpoint/model while avoiding collisions with user-defined entries.
        identity = json.dumps([ctx.openai_base_url, ctx.model]).encode()
        alias = f"omlx-{hashlib.sha256(identity).hexdigest()[:16]}"
        env["OMLX_API_KEY"] = ctx.auth_token
        # Vibe's environment layer merges these named entries with the existing
        # user/project TOML. The key itself never goes into config or arguments.
        env["VIBE_PROVIDERS"] = json.dumps(
            [
                {
                    "name": alias,
                    "api_base": ctx.openai_base_url,
                    "api_key_env_var": "OMLX_API_KEY",
                    "api_style": "openai",
                    "backend": "generic",
                }
            ]
        )
        model = {
            "name": ctx.model,
            "provider": alias,
            "alias": alias,
            "display_name": ctx.model,
            "input_price": 0.0,
            "output_price": 0.0,
            "supports_images": ctx.supports_images,
            "thinking": "off",
        }
        if ctx.context_window:
            model["max_context_length"] = ctx.context_window
        env["VIBE_MODELS"] = json.dumps([model])
        env["VIBE_ACTIVE_MODEL"] = alias
        os.execvpe("vibe", ["vibe", *ctx.extra_args], env)
