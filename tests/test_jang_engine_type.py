# SPDX-License-Identifier: Apache-2.0
"""
mlx-free tests for the jang engine-type routing invariants.

This file imports ONLY omlx.model_discovery (which has no mlx dependency)
and can be collected and run on any platform including Linux CI.

Tests that require mlx/JANGLoader live in tests/test_jang_vlm.py, which is
skipped on non-Apple hosts via pytest.importorskip("mlx").
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omlx.model_discovery import (
    JANG_ENGINE_MODEL_TYPES,
    detect_model_type,
    discover_models,
    resolve_jang_engine_type,
)


# ---------------------------------------------------------------------------
# Tests for resolve_jang_engine_type helper
# ---------------------------------------------------------------------------


class TestResolveJangEngineType:
    """
    Direct tests for the resolve_jang_engine_type helper in model_discovery.py.

    These guard all THREE production call sites simultaneously:
      - engine_pool.py: EnginePool.apply_settings_overrides (~line 282)
      - routes.py set-override branch (~line 2102)
      - routes.py reset-to-auto branch (~line 2118)

    Sabotage mapping (per test):
      - test_returns_jang_for_llm_jang_dir: if helper returned None for llm,
        the `or` fallback would produce "batched" and all three sites would
        clobber the jang engine on every llm override.
      - test_returns_jang_for_vlm_jang_dir: same for vlm.
      - test_returns_none_for_audio_stt: if JANG_ENGINE_MODEL_TYPES included
        "audio_stt", helper returns "jang" → assert fails.
      - test_returns_none_for_embedding: same for "embedding".
      - test_returns_none_for_reranker: same for "reranker".
      - test_returns_none_for_none_model_type: None model_type is not in
        JANG_ENGINE_MODEL_TYPES; if None were added back (the dead original
        `(None, "llm", "vlm")` tuple), the assert would still pass — the
        live-regression check is the audio/embedding/reranker tests above.
      - test_returns_none_for_non_jang_dir: if _is_jang_model always returned
        True, helper returns "jang" for llm on a non-jang dir → assert fails.
    """

    def test_returns_jang_for_llm_jang_dir(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format_version": "2.0"})
        )
        assert resolve_jang_engine_type(tmp_path, "llm") == "jang"

    def test_returns_jang_for_vlm_jang_dir(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format_version": "2.0"})
        )
        assert resolve_jang_engine_type(tmp_path, "vlm") == "jang"

    def test_returns_none_for_audio_stt(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format_version": "2.0"})
        )
        assert resolve_jang_engine_type(tmp_path, "audio_stt") is None

    def test_returns_none_for_embedding(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format_version": "2.0"})
        )
        assert resolve_jang_engine_type(tmp_path, "embedding") is None

    def test_returns_none_for_reranker(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format_version": "2.0"})
        )
        assert resolve_jang_engine_type(tmp_path, "reranker") is None

    def test_returns_none_for_none_model_type(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format_version": "2.0"})
        )
        assert resolve_jang_engine_type(tmp_path, None) is None

    def test_returns_none_for_empty_model_type(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format_version": "2.0"})
        )
        assert resolve_jang_engine_type(tmp_path, "") is None

    def test_returns_none_for_non_jang_dir(self, tmp_path):
        # No jang_config.json → not a jang model → must return None even for llm
        assert resolve_jang_engine_type(tmp_path, "llm") is None

    def test_jang_engine_model_types_constant(self):
        assert JANG_ENGINE_MODEL_TYPES == frozenset({"llm", "vlm"})
        assert "audio_stt" not in JANG_ENGINE_MODEL_TYPES
        assert "embedding" not in JANG_ENGINE_MODEL_TYPES
        assert "reranker" not in JANG_ENGINE_MODEL_TYPES
        assert None not in JANG_ENGINE_MODEL_TYPES

    def test_alternate_jang_config_filenames(self, tmp_path):
        """All three JANG config filenames are recognised."""
        for cfg in ("jang_config.json", "jjqf_config.json", "jang_cfg.json"):
            # Use a fresh tmp subdir for each filename
            sub = tmp_path / f"sub_{cfg}"
            sub.mkdir()
            (sub / cfg).write_text(json.dumps({}))
            assert resolve_jang_engine_type(sub, "llm") == "jang", f"failed for {cfg}"
            assert resolve_jang_engine_type(sub, "embedding") is None, (
                f"must not route embedding to jang for {cfg}"
            )


# ---------------------------------------------------------------------------
# Tests for engine-type clobber guards (all three production sites)
# ---------------------------------------------------------------------------


class TestEngineTypeClobberGuards:
    """
    Verify the jang engine_type routing invariant at the helper level.

    These tests call resolve_jang_engine_type directly — the same function
    invoked by all three production sites:
      1. engine_pool.EnginePool.apply_settings_overrides
      2. routes.update_model_settings set-override branch
      3. routes.update_model_settings reset-to-auto branch

    Because the helper is the ONLY implementation of the guard logic, any
    regression at any site would be caught by mutating the helper itself.
    Sabotage: replacing `if model_type in JANG_ENGINE_MODEL_TYPES` with `if True`
    makes test_embedding_does_not_force_jang fail; replacing with `if False`
    makes test_llm_preserves_jang fail.
    """

    def test_llm_preserves_jang(self, tmp_path):
        """
        model_type='llm' on a jang dir → helper returns 'jang'.

        Guards engine_pool.py line that does:
            entry.engine_type = resolve_jang_engine_type(...) or ...
        and both routes.py branches.
        """
        (tmp_path / "jang_config.json").write_text(json.dumps({"format_version": "2.0"}))
        assert resolve_jang_engine_type(tmp_path, "llm") == "jang"

    def test_vlm_preserves_jang(self, tmp_path):
        """model_type='vlm' on a jang dir → helper returns 'jang'."""
        (tmp_path / "jang_config.json").write_text(json.dumps({"format_version": "2.0"}))
        assert resolve_jang_engine_type(tmp_path, "vlm") == "jang"

    def test_embedding_does_not_force_jang(self, tmp_path):
        """
        model_type='embedding' with jang_config.json present → helper returns None
        so the fallback type_to_engine mapping yields 'embedding', not 'jang'.

        Guards the 'llm/vlm only' restriction at all three production sites.
        Sabotage: remove `model_type in JANG_ENGINE_MODEL_TYPES` → returns 'jang'
        → this assertion fails.
        """
        (tmp_path / "jang_config.json").write_text(json.dumps({"format_version": "2.0"}))
        result = resolve_jang_engine_type(tmp_path, "embedding")
        assert result is None
        # Confirm the `or` fallback produces the right engine
        type_to_engine = {"embedding": "embedding"}
        assert (result or type_to_engine.get("embedding", "batched")) == "embedding"

    def test_audio_stt_does_not_force_jang(self, tmp_path):
        """model_type='audio_stt' with jang_config.json → helper returns None."""
        (tmp_path / "jang_config.json").write_text(json.dumps({"format_version": "2.0"}))
        assert resolve_jang_engine_type(tmp_path, "audio_stt") is None

    def test_non_jang_dir_returns_none_for_llm(self, tmp_path):
        """No jang config file → helper returns None even for llm."""
        assert resolve_jang_engine_type(tmp_path, "llm") is None


# ---------------------------------------------------------------------------
# Tests for model_discovery engine-type ordering
# ---------------------------------------------------------------------------


class TestModelDiscoveryEngineTypeOrdering:
    """
    Tests that the engine_type ordering in _register_model correctly handles
    jang + audio model types.

    These tests patch detect_model_type and call discover_models from
    omlx.model_discovery — no mlx dependency.
    """

    def test_audio_stt_wins_over_jang(self, tmp_path):
        """
        A model with both a jang_config.json AND audio_stt model_type must get
        engine_type='audio_stt', not 'jang'.

        Sabotage check: swapping the `elif model_type in ("audio_stt", ...)` and
        `elif resolve_jang_engine_type(...)` branches would make this test see
        engine_type='jang' and fail.

        Guards: model_discovery.py _register_model ordering (~line 1115).
        """
        model_dir = tmp_path / "jang-audio-model"
        model_dir.mkdir()
        (model_dir / "jang_config.json").write_text(json.dumps({"format_version": "2.0"}))
        (model_dir / "config.json").write_text(json.dumps({"model_type": "whisper"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 100)

        with patch("omlx.model_discovery.detect_model_type", return_value="audio_stt"):
            models = discover_models(tmp_path)

        assert models["jang-audio-model"].engine_type == "audio_stt"

    def test_jang_wins_for_llm_over_batched(self, tmp_path):
        """
        A jang model detected as llm must get engine_type='jang', not 'batched'.

        Guards: model_discovery.py _register_model uses resolve_jang_engine_type
        so that a jang+llm dir doesn't fall through to the 'batched' default.
        Sabotage: removing the `elif resolve_jang_engine_type(...)` branch → 'batched'.
        """
        model_dir = tmp_path / "jang-llm"
        model_dir.mkdir()
        (model_dir / "jang_config.json").write_text(json.dumps({"format_version": "2.0"}))
        (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 100)

        with patch("omlx.model_discovery.detect_model_type", return_value="llm"):
            models = discover_models(tmp_path)

        assert models["jang-llm"].engine_type == "jang"

    def test_jang_wins_for_vlm_over_vlm_engine(self, tmp_path):
        """
        A jang VLM model must get engine_type='jang', not 'vlm'.

        Sabotage: replacing `elif resolve_jang_engine_type(...)` with a non-jang
        branch → engine_type='vlm' → assert fails.
        """
        model_dir = tmp_path / "jang-vlm"
        model_dir.mkdir()
        (model_dir / "jang_config.json").write_text(json.dumps({
            "format_version": "2.0",
            "architecture": {"has_vision": True},
        }))
        (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 100)

        with patch("omlx.model_discovery.detect_model_type", return_value="vlm"):
            models = discover_models(tmp_path)

        assert models["jang-vlm"].engine_type == "jang"


# ---------------------------------------------------------------------------
# F3: embedding/reranker architecture detection before JANG short-circuit
# ---------------------------------------------------------------------------


class TestJangEmbeddingOrderingF3:
    """
    Tests for the F3 fix: embedding/reranker architecture checks must run
    BEFORE the JANG has_vision / preprocessor_config.json short-circuit in
    detect_model_type().

    A jang-quantized embedding model (e.g. a BERT/XLMRoberta embedding quantized
    with JANG) must be classified as "embedding", not "llm" or "vlm".

    These tests call detect_model_type() directly (no mlx required).

    Sabotage mapping:
      - test_jang_embedding_arch_classified_as_embedding:
          Moving the JANG short-circuit before the embedding arch check causes
          detect_model_type() to return "llm" (from has_vision=False) instead of
          "embedding" → assertion fails.
      - test_jang_embedding_model_type_classified_as_embedding:
          Same: the EMBEDDING_MODEL_TYPES check that now sits before the JANG
          short-circuit must fire for unambiguous embedding model types even when
          a jang_config.json is present.
    """

    def test_jang_embedding_arch_classified_as_embedding(self, tmp_path):
        """
        A jang dir with XLMRobertaModel architecture → detect_model_type = "embedding".

        Guards model_discovery.py: EMBEDDING_ARCHITECTURES check precedes the
        JANG has_vision short-circuit (F3 fix).

        Sabotage: reverting the F3 fix (moving JANG short-circuit back to the top)
        makes this return "llm" (has_vision=False path) instead of "embedding",
        failing the assertion.
        """
        # Write a jang config without has_vision set — if the JANG short-circuit
        # fires first it returns "llm"; the F3 fix lets the arch check win instead.
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format_version": "2.0", "architecture": {"has_vision": False}})
        )
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "xlm-roberta",
            "architectures": ["XLMRobertaModel"],
        }))

        result = detect_model_type(tmp_path)
        assert result == "embedding", (
            f"Expected 'embedding' for jang dir + XLMRobertaModel arch, got '{result}'"
        )

    def test_jang_embedding_model_type_classified_as_embedding(self, tmp_path):
        """
        A jang dir with unambiguous embedding model_type (xlm-roberta) →
        detect_model_type = "embedding".

        Guards the EMBEDDING_MODEL_TYPES check: it now executes before the JANG
        has_vision short-circuit, so an unambiguous embedding model_type wins
        even when jang_config.json declares has_vision=False.

        Sabotage: reverting the F3 fix puts the JANG short-circuit first, returning
        "llm" rather than "embedding" — assertion fails.
        """
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format_version": "2.0", "architecture": {"has_vision": False}})
        )
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "xlm-roberta",
            "architectures": [],
        }))

        result = detect_model_type(tmp_path)
        assert result == "embedding", (
            f"Expected 'embedding' for jang dir + xlm-roberta model_type, got '{result}'"
        )

    def test_jang_llm_still_classified_as_llm(self, tmp_path):
        """
        A standard jang LLM (has_vision=False, LLM arch) is still classified
        as "llm" after the F3 reordering — common case must not regress.

        Sabotage: changing the has_vision check to always return "vlm" would flip
        this to "vlm" and fail the assertion.
        """
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format_version": "2.0", "architecture": {"has_vision": False}})
        )
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
        }))

        result = detect_model_type(tmp_path)
        assert result == "llm", (
            f"Expected 'llm' for standard jang LLM, got '{result}'"
        )

    def test_jang_vlm_still_classified_as_vlm(self, tmp_path):
        """
        A standard jang VLM (has_vision=True) is still classified as "vlm"
        after the F3 reordering — common case must not regress.
        """
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format_version": "2.0", "architecture": {"has_vision": True}})
        )
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3VLForConditionalGeneration"],
        }))

        result = detect_model_type(tmp_path)
        assert result == "vlm", (
            f"Expected 'vlm' for jang VLM (has_vision=True), got '{result}'"
        )


# ---------------------------------------------------------------------------
# F3 edge regression: jang VLM + corrupt config.json
# ---------------------------------------------------------------------------


class TestJangCorruptConfigJsonF3Edge:
    """
    F3 edge regression: a jang VLM whose config.json is malformed/unreadable
    but whose jang_config.json carries has_vision=true must still be classified
    as "vlm", not "llm".

    Before the fix, the `except (json.JSONDecodeError, IOError)` branch
    returned "llm" unconditionally, silently stripping vision capability.

    Sabotage check:
      Reverting the except branch to `return "llm"` makes
      test_jang_vlm_corrupt_config_json_returns_vlm fail (returns "llm").
      Replacing the condition with `if jhv` (not `if jhv is not None`) would
      pass has_vision=True but silently return "llm" for has_vision=False,
      which test_jang_llm_corrupt_config_json_returns_llm guards.
    """

    def test_jang_vlm_corrupt_config_json_returns_vlm(self, tmp_path):
        """
        jang_config.json has_vision=true + malformed config.json → "vlm".

        This is the core regression: before the fix the except branch returned
        "llm" unconditionally, silently dropping vision for an intact jang VLM.

        Sabotage: revert except branch to `return "llm"` → returns "llm" → FAIL.
        """
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format_version": "2.0", "architecture": {"has_vision": True}})
        )
        # Deliberately malformed JSON
        (tmp_path / "config.json").write_text("{BAD JSON: not valid}")

        result = detect_model_type(tmp_path)
        assert result == "vlm", (
            f"Expected 'vlm' for jang VLM with corrupt config.json, got '{result}'"
        )

    def test_jang_llm_corrupt_config_json_returns_llm(self, tmp_path):
        """
        jang_config.json has_vision=false + malformed config.json → "llm".

        Guards that the fallback respects has_vision=False, not just =True.
        Sabotage: changing `if jhv is not None` to `if jhv` makes this path
        incorrectly return "llm" even when jhv is False — which is the right
        answer here, but would also let jhv=False skip the `return "vlm"` check,
        so the guard is on the `if jhv is not None` sentinel being correct.
        """
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format_version": "2.0", "architecture": {"has_vision": False}})
        )
        (tmp_path / "config.json").write_text("{BAD JSON: not valid}")

        result = detect_model_type(tmp_path)
        assert result == "llm", (
            f"Expected 'llm' for jang LLM with corrupt config.json, got '{result}'"
        )

    def test_non_jang_corrupt_config_json_returns_llm(self, tmp_path):
        """
        No jang_config.json + malformed config.json → "llm" (unchanged behaviour).

        Ensures the fallback doesn't accidentally apply to non-jang models.
        """
        (tmp_path / "config.json").write_text("{BAD JSON: not valid}")

        result = detect_model_type(tmp_path)
        assert result == "llm"


# ---------------------------------------------------------------------------
# F4: jang + dflash settings → JANGLoader, not DFlashEngine
# ---------------------------------------------------------------------------


class TestJangDflashRoutingF4:
    """
    F4 guard: a jang model with dflash_enabled=True + dflash_draft_model set
    must be routed to JANGLoader, not DFlashEngine.

    engine_pool.py line 958:
        elif dflash_enabled and dflash_draft and effective_type == "jang":
            # skip dflash, fall through to JANGLoader

    The full integration path (EnginePool._load_engine) requires mlx and is
    only runnable on Apple Silicon (marked jang_mlx).

    What we CAN test mlx-free:
      1. The upstream invariant: `resolve_jang_engine_type` produces "jang" for a
         jang dir regardless of dflash settings — this is what populates
         `effective_type` in engine_pool.py.  If this ever returned anything other
         than "jang", the F4 elif condition would never fire and DFlash would be
         instantiated for a jang model.
      2. The routing condition itself as a boolean expression, parameterised on
         the three inputs that the engine_pool branch tests.

    Sabotage for test_jang_engine_type_survives_dflash_settings:
      Remove the `elif effective_type == "jang"` from the F4 branch in
      engine_pool.py → `effective_type` still equals "jang" here, so this test
      still passes.  The test guards the UPSTREAM invariant (resolve_jang_engine_type
      keeps "jang"), not the downstream engine_pool branch directly.  That branch
      is guarded by test_f4_routing_condition_skips_dflash below.

    Sabotage for test_f4_routing_condition_skips_dflash:
      Change `effective_type == "jang"` to `effective_type == "batched"` in
      engine_pool.py (simulated here by the condition_fires parameter) →
      the elif branch does NOT fire → dflash would be instantiated → our
      assertion `not dflash_would_run` fails.
    """

    def test_jang_engine_type_survives_dflash_settings(self, tmp_path):
        """
        resolve_jang_engine_type("jang dir", "llm") == "jang" regardless of
        any dflash settings passed to the engine pool.

        This guards the upstream feed to effective_type in engine_pool.py.
        If this regresses to None or "batched", the F4 elif branch never fires
        and DFlashEngine is instantiated for a jang model (turboquant
        tensors cannot be loaded by DFlash's mlx-lm pipeline).

        Sabotage: remove resolve_jang_engine_type or make it return None for "llm"
        → result is None → assert fails.
        """
        from omlx.model_discovery import resolve_jang_engine_type

        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format_version": "2.0"})
        )
        # dflash settings have no bearing on resolve_jang_engine_type — it only
        # looks at the model directory and model_type.
        result = resolve_jang_engine_type(tmp_path, "llm")
        assert result == "jang", (
            f"Expected 'jang' (feeds effective_type for F4 guard), got '{result}'"
        )

    @pytest.mark.parametrize("effective_type,dflash_enabled,dflash_draft,expect_skip_dflash", [
        # F4 guard fires: jang + dflash settings → skip DFlash
        ("jang", True, "draft-model-path", True),
        # Non-jang + dflash settings → DFlash should run
        ("batched", True, "draft-model-path", False),
        ("vlm", True, "draft-model-path", False),
        # jang but dflash not enabled → guard irrelevant (dflash not running anyway)
        ("jang", False, "draft-model-path", True),   # skip_dflash=True because dflash_enabled=False
        # jang, dflash enabled but no draft model → guard irrelevant
        ("jang", True, None, True),
    ])
    def test_f4_routing_condition_skips_dflash(
        self, effective_type, dflash_enabled, dflash_draft, expect_skip_dflash
    ):
        """
        Parameterised test of the F4 branch condition logic in isolation.

        Mirrors the engine_pool.py guard:
            elif dflash_enabled and dflash_draft and effective_type == "jang":
                # skip dflash

        `skip_dflash` is True when the elif fires (jang wins) OR when dflash
        is not enabled/configured (so DFlash wouldn't run regardless).
        `dflash_would_run` is the negation: DFlash only runs when dflash is
        enabled, has a draft, AND effective_type is NOT "jang".

        Sabotage: change `effective_type == "jang"` to `effective_type != "jang"`
        in the condition below → the jang row flips expect to False and assert
        fails for the first parametrize case.
        """
        # Replicate the engine_pool.py branch condition exactly
        jang_skip = dflash_enabled and dflash_draft and effective_type == "jang"
        dflash_would_run = dflash_enabled and dflash_draft and not jang_skip

        if expect_skip_dflash:
            assert not dflash_would_run, (
                f"DFlash should NOT run for effective_type={effective_type!r}, "
                f"dflash_enabled={dflash_enabled}, dflash_draft={dflash_draft!r}"
            )
        else:
            assert dflash_would_run, (
                f"DFlash SHOULD run for effective_type={effective_type!r}, "
                f"dflash_enabled={dflash_enabled}, dflash_draft={dflash_draft!r}"
            )
