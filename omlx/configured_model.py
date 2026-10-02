# SPDX-License-Identifier: Apache-2.0
"""Resolve a model's effective settings across the layers that define them.

Per-model settings win over the model's own defaults, which win over the
server's sampling defaults. Code that resolves a setting across these layers
should go through ConfiguredModel rather than repeat the lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from .engine_pool import EngineEntry
from .model_settings import ModelSettings, merge_chat_template_kwargs

if TYPE_CHECKING:
    # Circular at runtime: server.py imports this module.
    from .server import SamplingDefaults

T = TypeVar("T")


def first_present(*args: T | None) -> T | None:
    """The first non-None value passed, or None if all are None."""
    for x in args:
        if x is not None:
            return x
    return None


def _empty_engine_entry() -> EngineEntry:
    """An entry with no discovered defaults, for models the pool does not know."""
    return EngineEntry(
        model_id="",
        model_path="",
        model_type="llm",
        engine_type="batched",
        estimated_size=0,
    )


@dataclass(frozen=True)
class ConfiguredModel:
    """A model's settings, its discovered defaults, and the server's sampling
    defaults, with accessors that resolve each effective value across them.

    Build with :func:`new_configured_model`; every layer must be present.
    """

    settings: ModelSettings
    model_entry: EngineEntry
    sampling: SamplingDefaults
    # Accessors take the first non-None value: settings, then model_entry,
    # then sampling. A layer falls through only on a field whose default
    # is None, so give new fields a None default, not a real value.

    def _settings_template_kwargs(self) -> dict[str, Any]:
        """Chat-template kwargs the settings layer sends when a request overrides nothing."""
        # Same helper as the request path, so what we report cannot drift
        # from what gets rendered.
        return merge_chat_template_kwargs(
            self.settings,
            None,
            preserve_thinking_default=self.model_entry.preserve_thinking_default,
        )

    @property
    def enable_thinking(self) -> bool | None:
        """Whether the model will think: the per-model toggle, else the
        ``chat_template_kwargs`` value, else ``True`` if a thinking budget is
        active, else the template's default. ``None`` if the model has no
        thinking switch."""
        return first_present(
            self._settings_template_kwargs().get("enable_thinking"),
            self.model_entry.thinking_default,
        )

    @property
    def preserve_thinking(self) -> bool | None:
        """Whether earlier ``<think>`` blocks are kept when rendering history:
        the per-model toggle, else the ``chat_template_kwargs`` value, else
        ``True`` when the template supports it and thinking is on. ``None``
        if the template has no such flag."""
        return self._settings_template_kwargs().get("preserve_thinking")

    @property
    def max_context_window(self) -> int | None:
        """Context limit in tokens: the per-model override, else the model's
        native length capped by ``sampling.max_context_window_policy``, else
        the global default. ``None`` only if no layer sets one."""
        if self.settings.max_context_window is not None:
            return self.settings.max_context_window
        # Only the discovered length is policy-capped: the override and the
        # global default are explicit operator choices. Policy <= 0 is unset.
        native = self.model_entry.model_context_length
        policy = self.sampling.max_context_window_policy
        if native is not None and policy is not None and policy > 0:
            native = min(native, policy)
        return first_present(native, self.sampling.max_context_window)

    @property
    def max_tokens(self) -> int | None:
        """Output token limit: the per-model setting, else the global default."""
        return first_present(
            self.settings.max_tokens,
            self.sampling.max_tokens,
        )


def new_configured_model(
    settings: ModelSettings | None = None,
    model_entry: EngineEntry | None = None,
    sampling: SamplingDefaults | None = None,
) -> ConfiguredModel:
    """Build a ConfiguredModel, filling absent layers with empty defaults."""
    # Circular at runtime: server.py imports this module.
    from .server import SamplingDefaults

    return ConfiguredModel(
        settings=ModelSettings() if settings is None else settings,
        model_entry=_empty_engine_entry() if model_entry is None else model_entry,
        sampling=SamplingDefaults() if sampling is None else sampling,
    )
