# SPDX-License-Identifier: Apache-2.0
"""Settings recipes: portable model-settings snapshots.

A recipe is a one-line ASCII string a user copies from an omlx.ai benchmark
detail page (or any other oMLX install) and pastes into Model Settings. The
same codec shape is used by the omlx.ai worker (``src/lib/recipe.ts``): a
version prefix, then base64url without padding of a zlib deflate stream of
the settings JSON object. The pure helpers here are shared by the reset,
recipe and optimal-settings admin endpoints in ``routes.py``.
"""

import base64
import json
import os
import re
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Any
from urllib.parse import urlencode

from ..model_profiles import PROFILE_FIELDS_SET, filter_profile_fields
from ..model_settings import ModelSettings

RECIPE_PREFIX = "omlx-recipe:"
RECIPE_VERSION = 1

_ENCODED_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_ENCODED_CHARS = 8192
# Decompression stops here so a crafted recipe cannot inflate into a large
# buffer before the JSON size check.
_MAX_DECODED_BYTES = 16384
_MAX_KEYS = 120
_MAX_STRING_VALUE = 500

OMLX_AI_PERFORMANCE_URL = "https://omlx.ai/benchmarks/performance"

# Profile fields a snapshot never imports: a grammar constrains output
# content for the local use case, not performance, so the local value stays.
SNAPSHOT_EXCLUDED_FIELDS = frozenset({"guided_grammar", "guided_grammar_enabled"})


class RecipeError(ValueError):
    """The recipe text is malformed or violates the size limits."""


def encode_recipe(settings: dict) -> str:
    """Encode a settings dict. Mirrors the worker encoder byte for byte."""
    payload = {
        key: value
        for key, value in sorted(settings.items())
        if key != "benchmark_context" and value is not None
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    packed = zlib.compress(raw, 9)
    encoded = base64.urlsafe_b64encode(packed).rstrip(b"=").decode("ascii")
    return f"{RECIPE_PREFIX}{RECIPE_VERSION}:{encoded}"


def decode_recipe(text: str) -> dict:
    """Decode recipe text into a settings dict, or raise ``RecipeError``.

    The result is the raw payload; callers filter it with
    ``filter_profile_fields`` before use.
    """
    text = (text or "").strip()
    if not text.startswith(RECIPE_PREFIX):
        raise RecipeError("Not a recipe: expected text starting with 'omlx-recipe:'")
    rest = text[len(RECIPE_PREFIX) :]
    version, sep, encoded = rest.partition(":")
    if not sep or not version.isdigit():
        raise RecipeError("Malformed recipe header")
    if int(version) != RECIPE_VERSION:
        raise RecipeError(f"Unsupported recipe version {version}")
    if (
        not encoded
        or len(encoded) > _MAX_ENCODED_CHARS
        or not _ENCODED_RE.match(encoded)
    ):
        raise RecipeError("Recipe body is not valid base64url text or is too long")
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        packed = base64.urlsafe_b64decode(padded)
    except (ValueError, TypeError) as error:
        raise RecipeError("Recipe body is not valid base64url text") from error
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(packed, _MAX_DECODED_BYTES)
    except zlib.error as error:
        raise RecipeError("Recipe body is not a valid compressed stream") from error
    if inflater.unconsumed_tail or not inflater.eof:
        raise RecipeError("Recipe payload is too large")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise RecipeError("Recipe payload is not valid JSON") from error
    if not isinstance(payload, dict):
        raise RecipeError("Recipe payload must be a JSON object")
    if len(payload) > _MAX_KEYS:
        raise RecipeError(f"Recipe payload has more than {_MAX_KEYS} keys")
    for key, value in payload.items():
        if not isinstance(key, str) or len(key) > 60:
            raise RecipeError("Recipe payload has an invalid key")
        if isinstance(value, str) and len(value) > _MAX_STRING_VALUE:
            raise RecipeError(f"Recipe value for '{key}' is too long")
        if not isinstance(value, (bool, int, float, str, list, dict)):
            raise RecipeError(f"Recipe value for '{key}' has an unsupported type")
    return payload


@dataclass(frozen=True)
class FeatureGroup:
    """Settings keys that stand or fall together when applying a snapshot."""

    name: str
    enabled_key: str
    prefixes: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()

    def owns(self, key: str) -> bool:
        return key in self.keys or key.startswith(self.prefixes)


# Speculative paths come first so a conflicting MoE offload or ANE request is
# the one dropped in lenient mode, matching validate_moe_expert_offload's
# "disable speculative decoding first" framing in reverse: the recipe author
# chose the speculative path deliberately.
FEATURE_GROUPS: tuple[FeatureGroup, ...] = (
    FeatureGroup("dflash", "dflash_enabled", prefixes=("dflash_",)),
    FeatureGroup("specprefill", "specprefill_enabled", prefixes=("specprefill_",)),
    FeatureGroup("vlm_mtp", "vlm_mtp_enabled", prefixes=("vlm_mtp_",)),
    FeatureGroup("mtp", "mtp_enabled", keys=("mtp_enabled", "mtp_num_draft_tokens")),
    FeatureGroup("turboquant", "turboquant_kv_enabled", prefixes=("turboquant_",)),
    FeatureGroup(
        "ane_prefill", "qwen35_ane_prefill_enabled", prefixes=("qwen35_ane_prefill_",)
    ),
    FeatureGroup("oq_a8", "qwen35_oq_a8_enabled", prefixes=("qwen35_oq_a8_",)),
    FeatureGroup(
        "moe_expert_offload",
        "moe_expert_offload_enabled",
        prefixes=("moe_expert_offload_",),
    ),
    FeatureGroup("index_cache", "index_cache_freq", keys=("index_cache_freq",)),
)


def group_enabled(snapshot: dict, group: FeatureGroup) -> bool:
    return bool(snapshot.get(group.enabled_key))


def drop_group(snapshot: dict, group: FeatureGroup) -> dict:
    """Return a copy without the group's keys; they fall back to defaults."""
    return {key: value for key, value in snapshot.items() if not group.owns(key)}


def _leaf(value: str) -> str:
    return os.path.basename(value.rstrip("/")) or value


def resolve_draft_reference(
    value: str, entries: Iterable[tuple[str, str | None]]
) -> tuple[str | None, str | None]:
    """Map a draft-model reference to an installed ``(model_id, model_path)``.

    Benchmark uploads reduce draft paths to a basename, and a recipe may come
    from another machine, so an exact id/path match is tried first and a
    basename match second. Ties resolve to the lowest model id.
    """
    if not isinstance(value, str) or not value.strip():
        return None, None
    value = value.strip()
    entries = sorted(entries, key=lambda item: item[0])
    for model_id, model_path in entries:
        if value == model_id or (model_path and value == model_path):
            return model_id, model_path
    wanted = _leaf(value)
    for model_id, model_path in entries:
        if _leaf(model_id) == wanted or (model_path and _leaf(model_path) == wanted):
            return model_id, model_path
    return None, None


# Dataclass defaults for every ModelSettings field. Sending these instead of
# null matters: the settings PUT handler rejects null for the ANE numeric
# controls and for qwen35_oq_a8_min_tokens.
DEFAULTS: dict[str, Any] = {
    f.name: getattr(ModelSettings(), f.name) for f in fields(ModelSettings)
}
ALL_FIELDS: frozenset[str] = frozenset(DEFAULTS)


def recipe_scope(snapshot: dict, uploaded_fields: Iterable[str]) -> set[str]:
    """Fields a recipe replaces: every shareable profile field plus its own keys.

    Profile fields outside the upload allowlist (chat template kwargs) keep
    their current value unless the recipe names them.
    """
    scope = (PROFILE_FIELDS_SET & set(uploaded_fields)) | set(snapshot)
    return scope - SNAPSHOT_EXCLUDED_FIELDS


def build_candidate(current: dict, snapshot: dict, scope: set[str]) -> dict:
    """Current settings with every scoped field replaced by the snapshot."""
    candidate = {key: value for key, value in current.items() if key not in scope}
    candidate.update(snapshot)
    return candidate


def settings_diff(candidate: dict, current: dict, scope: set[str]) -> dict:
    """Scoped fields whose effective value changes between current and candidate."""
    diff: dict[str, Any] = {}
    for key in scope:
        if key not in DEFAULTS:
            continue
        target = candidate.get(key, DEFAULTS[key])
        if target != current.get(key, DEFAULTS[key]):
            diff[key] = target
    return diff


def clean_snapshot(snapshot: dict) -> dict:
    """Profile-eligible keys with real values, minus labels and excluded fields."""
    return filter_profile_fields(
        {
            key: value
            for key, value in snapshot.items()
            if key != "benchmark_context" and key not in SNAPSHOT_EXCLUDED_FIELDS
        }
    )


def search_url(
    chip: str, variant: str, model_name: str, context: int, memory_gb: int | None = None
) -> str:
    """Leaderboard link pre-filtered to the device and model that had no match."""
    query: dict[str, Any] = {
        "device": json.dumps([chip, variant or ""]),
        "model_exact": model_name,
        "context": context,
    }
    if memory_gb:
        query["memory_max"] = memory_gb
    return f"{OMLX_AI_PERFORMANCE_URL}?{urlencode(query)}"
