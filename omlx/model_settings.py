"""Per-model settings management for oMLX.

This module provides dataclasses and a manager for storing and retrieving
per-model configuration settings, including sampling parameters, pinned/default
flags, and metadata.
"""

import json
import logging
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Current settings file format version
SETTINGS_VERSION = 1


@dataclass
class ModelSettings:
    """Per-model configuration settings.

    Attributes:
        max_context_window: Maximum prompt token count before rejection (None = use global default).
        max_tokens: Maximum number of tokens to generate (None = use global default).
        temperature: Sampling temperature (None = use global default).
        top_p: Nucleus sampling probability (None = use global default).
        top_k: Top-k sampling parameter (None = use global default).
        repetition_penalty: Repetition penalty (None = use default 1.0, i.e. disabled).
        force_sampling: Force sampling even with temperature=0.
        is_pinned: Keep model loaded in memory.
        is_default: Use this model when no model is specified.
        display_name: Human-readable name for UI display.
        description: Optional description of the model.
    """

    # Sampling parameters (None means use global default)
    max_context_window: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    force_sampling: bool = False
    max_tool_result_tokens: Optional[int] = None

    # Model management flags
    is_pinned: bool = False
    is_default: bool = False  # Only one model can be default

    # Metadata
    display_name: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values.

        Returns:
            Dictionary representation with None values filtered out.
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ModelSettings":
        """Create ModelSettings from a dictionary.

        Args:
            data: Dictionary containing settings values.

        Returns:
            New ModelSettings instance with values from dict.
        """
        # Get valid field names
        valid_fields = {f.name for f in fields(cls)}

        # Filter to only valid keys
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)


class ModelSettingsManager:
    """Manager for per-model settings with file persistence.

    Handles loading, saving, and accessing model settings from a JSON file.
    Thread-safe for concurrent access.

    Attributes:
        base_path: Base directory for settings storage.
        settings_file: Path to the settings JSON file.
    """

    def __init__(self, base_path: Path):
        """Initialize the settings manager.

        Args:
            base_path: Base directory for settings storage.
        """
        self.base_path = Path(base_path)
        self.settings_file = self.base_path / "model_settings.json"
        self._lock = threading.Lock()
        self._settings: Dict[str, ModelSettings] = {}

        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Load existing settings
        self._load()

    def _load(self) -> None:
        """Load settings from the JSON file.

        If the file doesn't exist or is invalid, starts with empty settings.
        """
        if not self.settings_file.exists():
            logger.debug(f"Settings file not found: {self.settings_file}")
            self._settings = {}
            return

        try:
            with open(self.settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check version
            version = data.get("version", 1)
            if version != SETTINGS_VERSION:
                logger.warning(
                    f"Settings file version {version} differs from current {SETTINGS_VERSION}"
                )

            # Load model settings
            models_data = data.get("models", {})
            self._settings = {}

            for model_id, model_data in models_data.items():
                try:
                    self._settings[model_id] = ModelSettings.from_dict(model_data)
                except Exception as e:
                    logger.warning(
                        f"Failed to load settings for model '{model_id}': {e}"
                    )

            logger.info(f"Loaded settings for {len(self._settings)} models")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in settings file: {e}")
            self._settings = {}
        except Exception as e:
            logger.error(f"Failed to load settings file: {e}")
            self._settings = {}

    def _save(self) -> None:
        """Save settings to the JSON file.

        Must be called while holding the lock.
        """
        data = {
            "version": SETTINGS_VERSION,
            "models": {
                model_id: settings.to_dict()
                for model_id, settings in self._settings.items()
            }
        }

        try:
            # Write to temp file first, then rename for atomicity
            temp_file = self.settings_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            temp_file.replace(self.settings_file)
            logger.debug(f"Saved settings for {len(self._settings)} models")

        except Exception as e:
            logger.error(f"Failed to save settings file: {e}")
            raise

    def get_settings(self, model_id: str) -> ModelSettings:
        """Get settings for a specific model.

        Args:
            model_id: The model identifier.

        Returns:
            ModelSettings for the model, or default settings if not found.
        """
        with self._lock:
            if model_id in self._settings:
                # Return a copy to prevent external modification
                settings = self._settings[model_id]
                return ModelSettings.from_dict(settings.to_dict())

            logger.log(5, "No settings found for model '%s', using defaults", model_id)
            return ModelSettings()

    def set_settings(self, model_id: str, settings: ModelSettings) -> None:
        """Set settings for a specific model.

        If the new settings have is_default=True, clears is_default from all
        other models to maintain the exclusive default constraint.

        Args:
            model_id: The model identifier.
            settings: The settings to apply.
        """
        with self._lock:
            # Handle exclusive default constraint
            if settings.is_default:
                for mid, s in self._settings.items():
                    if mid != model_id and s.is_default:
                        s.is_default = False
                        logger.info(
                            f"Cleared is_default from model '{mid}' "
                            f"(new default: '{model_id}')"
                        )

            # Store a copy of the settings
            self._settings[model_id] = ModelSettings.from_dict(settings.to_dict())
            logger.info(f"Updated settings for model '{model_id}'")

            self._save()

    def get_default_model_id(self) -> Optional[str]:
        """Get the ID of the default model.

        Returns:
            The model ID marked as default, or None if no default is set.
        """
        with self._lock:
            for model_id, settings in self._settings.items():
                if settings.is_default:
                    return model_id
            return None

    def get_pinned_model_ids(self) -> list[str]:
        """Get list of all pinned model IDs.

        Returns:
            List of model IDs that are marked as pinned.
        """
        with self._lock:
            return [
                model_id
                for model_id, settings in self._settings.items()
                if settings.is_pinned
            ]

    def get_all_settings(self) -> Dict[str, ModelSettings]:
        """Get a copy of all model settings.

        Returns:
            Dictionary mapping model IDs to their settings (deep copy).
        """
        with self._lock:
            return {
                model_id: ModelSettings.from_dict(settings.to_dict())
                for model_id, settings in self._settings.items()
            }
