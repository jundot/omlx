"""Per-model settings management for oMLX.

This module provides dataclasses and a manager for storing and retrieving
per-model configuration settings, including sampling parameters, pinned/default
flags, and metadata.
"""

import copy
import json
import logging
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

from .model_profiles import (
    filter_profile_fields,
    filter_universal_fields,
    validate_profile_name,
    utcnow,
)

logger = logging.getLogger(__name__)

# Current settings file format version
SETTINGS_VERSION = 1
PROFILES_VERSION = 1
TEMPLATES_VERSION = 1


@dataclass
class ModelSettings:
    """Per-model configuration settings.

    Attributes:
        max_context_window: Maximum prompt token count before rejection (None = use global default).
        max_tokens: Maximum number of tokens to generate (None = use global default).
        temperature: Sampling temperature (None = use global default).
        top_p: Nucleus sampling probability (None = use global default).
        top_k: Top-k sampling parameter (None = use global default).
        min_p: Minimum probability threshold (None = use global default).
        repetition_penalty: Repetition penalty (None = use default 1.0, i.e. disabled).
        presence_penalty: Presence penalty (None = use global default).
        force_sampling: Force sampling even with temperature=0.
        max_tool_result_tokens: Maximum tokens in tool result (None = use global default).
        chat_template_kwargs: Extra chat template keyword arguments.
        forced_ct_kwargs: Keys in chat_template_kwargs that cannot be overridden.
        ttl_seconds: Auto-unload after idle seconds (None = no TTL).
        model_type_override: "llm", "vlm", "embedding", "reranker", or None (auto-detect).
        model_alias: API-visible alternative to the directory name.
        index_cache_freq: IndexCache: every Nth layer keeps indexer (DSA models only).
        enable_thinking: Explicit toggle for thinking/reasoning mode (None = auto).
        thinking_budget_enabled: Whether a thinking token budget is active.
        thinking_budget_tokens: Max tokens for thinking/reasoning.
        reasoning_parser: xgrammar builtin name: "qwen", "harmony", "llama", etc.
        turboquant_kv_enabled: Enable TurboQuant KV cache compression.
        turboquant_kv_bits: TurboQuant bit depth (2/2.5/3/3.5/4/6/8).
        turboquant_skip_last: Skip last KVCache layer to prevent corruption.
        specprefill_enabled: Enable SpecPrefill (experimental sparse prefill for MoE).
        specprefill_draft_model: Path to draft model for SpecPrefill.
        specprefill_keep_pct: Keep rate for SpecPrefill (0.1–0.5).
        specprefill_threshold: Min tokens to trigger SpecPrefill.
        dflash_enabled: Enable DFlash speculative decoding.
        dflash_draft_model: Path/repo for DFlash draft checkpoint.
        dflash_draft_quant_bits: Draft model quantization bits.
        dflash_max_ctx: Token threshold to fall back to BatchedEngine (None = unlimited).
        dflash_in_memory_cache: Enable DFlash L1 (RAM) prefix cache.
        dflash_in_memory_cache_max_bytes: L1 cache byte budget.
        dflash_ssd_cache: Enable DFlash L2 (SSD) prefix cache spill (uses omlx SSD cache dir).
        mtp_enabled: Enable native multi-token prediction (mlx-lm PR 990 / PR 15 monkey-patch).
            When True, the BatchGenerator uses an MTP draft+verify path for single-request
            decoding. Compatible model_types: qwen3_5*, qwen3_6*, deepseek_v4*. Mutually
            exclusive with dflash_enabled and turboquant_kv_enabled. Concurrent requests on
            the same model fall back to standard continuous batching automatically.
        is_pinned: Keep model loaded in memory.
        is_default: Use this model when no model is specified.
        display_name: Human-readable name for UI display.
        description: Optional description of the model.
        active_profile_name: Name of the currently-applied profile (None = no profile).
    """

    # Sampling parameters (None means use global default)
    max_context_window: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    force_sampling: bool = False
    max_tool_result_tokens: Optional[int] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    forced_ct_kwargs: Optional[list[str]] = None  # Keys that cannot be overridden by API requests
    ttl_seconds: Optional[int] = None  # Auto-unload after idle seconds (None = no TTL)
    model_type_override: Optional[str] = None  # "llm", "vlm", "embedding", "reranker", or None (auto-detect)
    model_alias: Optional[str] = None  # API-visible name (alternative to directory name)
    index_cache_freq: Optional[int] = None  # IndexCache: every Nth layer keeps indexer (DSA models only)
    enable_thinking: Optional[bool] = None  # Explicit toggle for thinking/reasoning mode (None = auto)
    preserve_thinking: Optional[bool] = None  # Keep <think> blocks in historical turns (None = auto, True when template supports it)
    thinking_budget_enabled: bool = False
    thinking_budget_tokens: Optional[int] = None
    reasoning_parser: Optional[str] = None  # xgrammar builtin name: "qwen", "harmony", "llama", etc.

    # TurboQuant KV cache (mlx-vlm backend)
    turboquant_kv_enabled: bool = False
    turboquant_kv_bits: float = 4  # 2, 2.5, 3, 3.5, 4, 6, 8
    turboquant_skip_last: bool = True  # Skip last KVCache layer (prevents corruption on sensitive models)

    # SpecPrefill (experimental: attention-based sparse prefill for MoE models)
    specprefill_enabled: bool = False
    specprefill_draft_model: Optional[str] = None  # Path to draft model (must share tokenizer)
    specprefill_keep_pct: Optional[float] = None  # Keep rate (0.1-0.5, default 0.2)
    specprefill_threshold: Optional[int] = None  # Min tokens to trigger (default 8192)

    # DFlash (block diffusion speculative decoding)
    dflash_enabled: bool = False
    dflash_draft_model: Optional[str] = None  # Path/repo for DFlash draft checkpoint
    dflash_draft_quant_bits: Optional[int] = None  # Draft model quantization (None=bf16, 4)
    dflash_max_ctx: Optional[int] = None  # None = unlimited; trigger BatchedEngine fallback when prompt_len >= this
    # DFlash prefix cache (private to dflash; separate from omlx tiered cache because
    # snapshots include draft model GDN state and target hidden chunks omlx never tracks)
    dflash_in_memory_cache: bool = True
    dflash_in_memory_cache_max_bytes: int = 8 * 1024 * 1024 * 1024  # 8 GiB (balanced profile default)
    dflash_ssd_cache: bool = False  # Requires in-memory cache and an omlx paged SSD cache dir

    # Native MTP (mlx-lm PR 990 / PR 15 monkey-patch). When enabled, BatchGenerator
    # uses MTP draft+verify path for single-request decoding. Compatible model_types:
    # qwen3_5*, qwen3_6*, deepseek_v4*. Mutually exclusive with dflash and turboquant.
    mtp_enabled: bool = False

    # Model management flags
    is_pinned: bool = False
    is_default: bool = False  # Only one model can be default

    # Security: opt-in per model. When True, mlx-lm/mlx-vlm/mlx-embeddings/reranker
    # loaders are allowed to execute custom Python from the model repository
    # (modeling_*.py, tokenization_*.py). Off by default — see issue #926.
    trust_remote_code: bool = False

    # Metadata
    display_name: Optional[str] = None
    description: Optional[str] = None
    active_profile_name: Optional[str] = None  # Name of the currently-applied profile

    def __post_init__(self) -> None:
        # Native MTP is mutually exclusive with DFlash (also speculative) and
        # TurboQuant KV (patches the same attention path). Reject combos at
        # construction time so the conflict surfaces in the admin UI / API
        # rather than at model load.
        if self.mtp_enabled and self.dflash_enabled:
            raise ValueError(
                "mtp_enabled and dflash_enabled cannot both be True; choose one "
                "speculative-decoding path per model"
            )
        if self.mtp_enabled and self.turboquant_kv_enabled:
            raise ValueError(
                "mtp_enabled and turboquant_kv_enabled cannot both be True; "
                "TurboQuant patches the attention path that MTP relies on"
            )

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
        self.profiles_file = self.base_path / "model_profiles.json"
        self.templates_file = self.base_path / "global_templates.json"
        self._lock = threading.Lock()
        self._settings: Dict[str, ModelSettings] = {}
        self._profiles: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._templates: Dict[str, Dict[str, Any]] = {}

        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Load existing settings
        self._load()
        self._load_profiles()
        self._load_templates()

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

    # ==================== Profiles ====================

    def _load_profiles(self) -> None:
        if not self.profiles_file.exists():
            self._profiles = {}
            return
        try:
            with open(self.profiles_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            version = data.get("version", 1)
            if version != PROFILES_VERSION:
                logger.warning(
                    f"Profiles file version {version} differs from current {PROFILES_VERSION}"
                )
            self._profiles = data.get("profiles", {}) or {}
            # Migration: strip ttl_seconds from existing profile settings
            for model_id, profiles in self._profiles.items():
                for name, profile in profiles.items():
                    settings = profile.get("settings")
                    if settings and "ttl_seconds" in settings:
                        del settings["ttl_seconds"]
        except Exception as e:
            logger.error(f"Failed to load profiles file: {e}")
            self._profiles = {}

    def _save_profiles(self) -> None:
        """Write profiles to disk atomically (temp file + rename)."""
        data = {"version": PROFILES_VERSION, "profiles": self._profiles}
        temp_file = self.profiles_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            temp_file.replace(self.profiles_file)
        except Exception as e:
            logger.error(f"Failed to save profiles file: {e}")
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise

    def list_profiles(self, model_id: str) -> list[dict]:
        """Return all profiles for ``model_id`` as serializable dicts."""
        with self._lock:
            per_model = self._profiles.get(model_id, {})
            return [dict(p) for p in per_model.values()]

    def get_profile(self, model_id: str, name: str) -> Optional[dict]:
        with self._lock:
            return dict(self._profiles.get(model_id, {}).get(name, {})) or None

    def save_profile(
        self,
        model_id: str,
        name: str,
        display_name: str,
        description: Optional[str],
        settings: Dict[str, Any],
        source_template: Optional[str] = None,
    ) -> dict:
        """Create a new profile. Raises if name is invalid or already exists."""
        validate_profile_name(name)
        filtered = filter_profile_fields(settings or {})
        with self._lock:
            per_model = self._profiles.setdefault(model_id, {})
            if name in per_model:
                raise ValueError(f"Profile '{name}' already exists for model '{model_id}'")
            now = utcnow().isoformat()
            per_model[name] = {
                "name": name,
                "display_name": display_name or name,
                "description": description,
                "created_at": now,
                "updated_at": now,
                "settings": filtered,
                "source_template": source_template,
            }
            self._save_profiles()
            return dict(per_model[name])

    def update_profile(
        self,
        model_id: str,
        name: str,
        *,
        new_name: Optional[str] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        source_template: Optional[str] = None,
    ) -> Optional[dict]:
        """Update a profile's metadata/settings. Returns updated dict or None if not found."""
        with self._lock:
            per_model = self._profiles.get(model_id, {})
            if name not in per_model:
                return None
            profile = dict(per_model[name])
            target_name = name
            rename_mode = False
            if new_name is not None and new_name != name:
                validate_profile_name(new_name)
                if new_name in per_model:
                    raise ValueError(
                        f"Profile '{new_name}' already exists for model '{model_id}'"
                    )
                target_name = new_name
                profile["name"] = new_name
                rename_mode = True
            if display_name is not None:
                profile["display_name"] = display_name
            if description is not None:
                profile["description"] = description
            if settings is not None:
                profile["settings"] = filter_profile_fields(settings)
            if source_template is not None:
                profile["source_template"] = source_template or None
            profile["updated_at"] = utcnow().isoformat()

            # Snapshot for rollback on write failure
            profiles_snapshot = copy.deepcopy(self._profiles)
            settings_snapshot = copy.deepcopy(self._settings)

            # Also update ModelSettings.active_profile_name if renamed and it was active
            old_active = None
            if rename_mode:
                old_active = self._settings.get(model_id)
                if old_active is not None and old_active.active_profile_name == name:
                    old_active.active_profile_name = target_name
                del per_model[name]

            per_model[target_name] = profile

            # Write profiles first; if this throws, rollback everything
            try:
                self._save_profiles()
                if rename_mode and old_active is not None:
                    self._save()
            except Exception:
                self._profiles = profiles_snapshot
                self._settings = settings_snapshot
                raise

            return dict(profile)

    def delete_profile(self, model_id: str, name: str) -> bool:
        with self._lock:
            per_model = self._profiles.get(model_id, {})
            if name not in per_model:
                return False

            profiles_snapshot = copy.deepcopy(self._profiles)
            settings_snapshot = copy.deepcopy(self._settings)

            del per_model[name]
            if not per_model and model_id in self._profiles:
                del self._profiles[model_id]
            # Clear active_profile_name if it referenced this profile
            old_active = self._settings.get(model_id)
            if old_active is not None and old_active.active_profile_name == name:
                old_active.active_profile_name = None

            try:
                self._save_profiles()
                if old_active is not None and old_active.active_profile_name is None:
                    self._save()
            except Exception:
                self._profiles = profiles_snapshot
                self._settings = settings_snapshot
                raise
            return True

    def apply_profile(self, model_id: str, name: str) -> Optional[ModelSettings]:
        """Merge profile settings into the model's live settings and persist."""
        with self._lock:
            per_model = self._profiles.get(model_id, {})
            if name not in per_model:
                return None
            profile_settings = per_model[name].get("settings", {}) or {}

            settings_snapshot = copy.deepcopy(self._settings)

            current = self._settings.get(model_id)
            if current is None:
                current = ModelSettings()
            merged = current.to_dict()
            for k, v in profile_settings.items():
                merged[k] = v
            merged["active_profile_name"] = name
            new_settings = ModelSettings.from_dict(merged)
            self._settings[model_id] = new_settings
            try:
                self._save()
            except Exception:
                self._settings = settings_snapshot
                raise
            return ModelSettings.from_dict(new_settings.to_dict())

    # ==================== Templates ====================

    def _load_templates(self) -> None:
        if not self.templates_file.exists():
            self._templates = {}
            return
        try:
            with open(self.templates_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            version = data.get("version", 1)
            if version != TEMPLATES_VERSION:
                logger.warning(
                    f"Templates file version {version} differs from current {TEMPLATES_VERSION}"
                )
            self._templates = data.get("templates", {}) or {}
            # Migration: strip ttl_seconds from existing template settings
            for name, template in self._templates.items():
                settings = template.get("settings")
                if settings and "ttl_seconds" in settings:
                    del settings["ttl_seconds"]
        except Exception as e:
            logger.error(f"Failed to load templates file: {e}")
            self._templates = {}

    def _save_templates(self) -> None:
        """Must be called while holding the lock."""
        data = {"version": TEMPLATES_VERSION, "templates": self._templates}
        try:
            temp_file = self.templates_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            temp_file.replace(self.templates_file)
        except Exception as e:
            logger.error(f"Failed to save templates file: {e}")
            raise

    def list_templates(self) -> list[dict]:
        with self._lock:
            return [dict(t) for t in self._templates.values()]

    def get_template(self, name: str) -> Optional[dict]:
        with self._lock:
            return dict(self._templates.get(name, {})) or None

    def save_template(
        self,
        name: str,
        display_name: str,
        description: Optional[str],
        settings: Dict[str, Any],
    ) -> dict:
        validate_profile_name(name)
        filtered = filter_universal_fields(settings or {})
        with self._lock:
            if name in self._templates:
                raise ValueError(f"Template '{name}' already exists")
            now = utcnow().isoformat()
            self._templates[name] = {
                "name": name,
                "display_name": display_name or name,
                "description": description,
                "created_at": now,
                "updated_at": now,
                "settings": filtered,
            }
            self._save_templates()
            return dict(self._templates[name])

    def upsert_template(
        self,
        name: str,
        display_name: str,
        description: Optional[str],
        settings: Dict[str, Any],
    ) -> dict:
        """Create or replace a template with the given settings."""
        validate_profile_name(name)
        filtered = filter_universal_fields(settings or {})
        with self._lock:
            now = utcnow().isoformat()
            existing = self._templates.get(name)
            created_at = existing["created_at"] if existing else now
            self._templates[name] = {
                "name": name,
                "display_name": display_name or name,
                "description": description,
                "created_at": created_at,
                "updated_at": now,
                "settings": filtered,
            }
            self._save_templates()
            return dict(self._templates[name])

    def update_template(
        self,
        name: str,
        *,
        new_name: Optional[str] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[dict]:
        with self._lock:
            if name not in self._templates:
                return None
            template = dict(self._templates[name])
            target = name
            if new_name is not None and new_name != name:
                validate_profile_name(new_name)
                if new_name in self._templates:
                    raise ValueError(f"Template '{new_name}' already exists")
                target = new_name
                template["name"] = new_name
            if display_name is not None:
                template["display_name"] = display_name
            if description is not None:
                template["description"] = description
            if settings is not None:
                template["settings"] = filter_universal_fields(settings)
            template["updated_at"] = utcnow().isoformat()
            if target != name:
                del self._templates[name]
            self._templates[target] = template
            self._save_templates()
            return dict(template)

    def delete_template(self, name: str) -> bool:
        with self._lock:
            if name not in self._templates:
                return False
            del self._templates[name]
            self._save_templates()
            return True
