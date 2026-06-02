"""Centralized resolution for model configurations.

In general, model settings override model defaults, model defaults override
sampling defaults, and sampling defaults override global defaults. Settings
not related to sampling are expected to be absent from sampling defaults, and
not all of these settings objects use the same name for the same concept.

This object is intended to be incrementally extended and incrementally adopted:
when updating code that performs this kind of resolution, migrate it to use
this object if practical.

In addition to resolving a setting from multiple layers, this allows passing
a model and its configuration around as a single object, which can clean up
function interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, TypeVar

from .engine_pool import EngineEntry
from .model_settings import ModelSettings

if TYPE_CHECKING:
    # SamplingDefaults lives in server.py, which imports this module.
    from .server import SamplingDefaults

T = TypeVar('T')

def first_present(*args: Optional[T]) -> Optional[T]:
    """The first non-None value passed, or None if all are None."""
    for x in args:
        if x is not None:
            return x
    return None


_EMPTY_ENGINE_ENTRY = EngineEntry(
    model_id="", model_path="", model_type="llm",
    engine_type="batched", estimated_size=0,
)


@dataclass(frozen=True)
class ConfiguredModel:
    """A model, its configuration, and its sources for fallback settings.

    Every layer is always present.  Layers with no real data use default-
    constructed instances (e.g. ``ModelSettings()`` with every field
    ``None``, an ``EngineEntry`` with empty strings and zeroes, etc.).
    This keeps property accessors simple — no ``None`` guards needed.
    However, multilayer resolution will not continue past a field that
    has a default value; currently this does not affect any lookups
    but an alternative approach may be required in the future.
    """

    # Priority is generally settings > model_entry > sampling.

    settings: ModelSettings
    model_entry: EngineEntry
    sampling: SamplingDefaults

    @property
    def enable_thinking(self) -> Optional[bool]:
        """Effective thinking state as the engine will serve it.

        ``settings.enable_thinking`` (the explicit toggle from Model settings)
        takes precedence; otherwise the model's chat-template default
        (``model_entry.thinking_default``). ``None`` means the model exposes no
        thinking toggle.
        """
        return first_present(
            self.settings.enable_thinking,
            self.model_entry.thinking_default
        )

    @property
    def preserve_thinking(self) -> Optional[bool]:
        """Effective ``preserve_thinking`` state (keep <think> blocks in
        historical turns), resolved the same way as :attr:`enable_thinking`.
        """
        return first_present(
            self.settings.preserve_thinking,
            self.model_entry.preserve_thinking_default
        )

    def thinking_template_overrides(self) -> Dict[str, Any]:
        """Chat-template kwargs to merge from the explicit per-model toggles.

        Uses only ``self.settings``, allowing the model's chat template to use
        its own defaults directly.
        """
        overrides: Dict[str, Any] = {}
        if self.settings.enable_thinking is not None:
            overrides["enable_thinking"] = self.settings.enable_thinking
        if self.settings.preserve_thinking is not None:
            overrides["preserve_thinking"] = self.settings.preserve_thinking
        return overrides

    @property
    def max_context_window(self) -> int | None:
        """Effective max context window limit."""
        return first_present(
            self.settings.max_context_window,
            self.model_entry.model_context_length,
            self.sampling.max_context_window
        )

    @property
    def max_tokens(self) -> int | None:
        """Effective max output tokens."""
        return first_present(
            self.settings.max_tokens,
            self.sampling.max_tokens
        )


def new_configured_model(
    settings: ModelSettings | None = None,
    model_entry: EngineEntry | None = None,
    sampling: SamplingDefaults | None = None,
) -> ConfiguredModel:
    """Build a ConfiguredModel, filling absent layers with defaults."""
    # Lazy import: SamplingDefaults lives in server.py, which imports us.
    from .server import SamplingDefaults as _SD
    return ConfiguredModel(
        settings=settings or ModelSettings(),
        model_entry=model_entry or _EMPTY_ENGINE_ENTRY,
        sampling=sampling or _SD(),
    )
