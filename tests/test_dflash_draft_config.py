# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.dflash_draft_config (issue #2317)."""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("dflash_mlx", reason="dflash-mlx not installed")

# All fields DFlashDraftModelArgs requires with no default, minus
# rope_theta/block_size (added per-layout below). Values mirror the real
# z-lab/Qwen3.6-35B-A3B-DFlash config excerpt.
_BASE_FIELDS = {
    "model_type": "qwen3",
    "hidden_size": 2048,
    "num_hidden_layers": 6,
    "intermediate_size": 6144,
    "num_attention_heads": 32,
    "rms_norm_eps": 1e-06,
    "vocab_size": 248320,
    "num_key_value_heads": 8,
    "max_position_embeddings": 262144,
    "head_dim": 128,
    "tie_word_embeddings": False,
    "num_target_layers": 40,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "layer_types": (
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ),
    "sliding_window": 4096,
    "use_sliding_window": True,
}


def _new_layout_config() -> dict:
    """Real z-lab/Qwen3.6-35B-A3B-DFlash config (current, transformers 5.x)."""
    return {
        **_BASE_FIELDS,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "dflash_config": {
            "block_size": 16,
            "mask_token_id": 248077,
            "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37],
        },
    }


def _old_layout_config() -> dict:
    """Same repo's April revision config (working, root-level keys)."""
    return {
        **_BASE_FIELDS,
        "rope_theta": 10000000,
        "block_size": 16,
        "rope_scaling": {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 64.0,
            "original_max_position_embeddings": 4096,
            "rope_type": "yarn",
            "type": "yarn",
        },
        "dflash_config": {
            "mask_token_id": 248070,
            "target_layer_ids": [1, 10, 19, 28, 37],
        },
    }


class TestNormalizeDflashDraftConfig:
    """Unit tests for the pure hoisting function."""

    def test_new_layout_hoists_rope_theta_and_block_size(self):
        from omlx.patches.dflash_draft_config import normalize_dflash_draft_config

        result = normalize_dflash_draft_config(_new_layout_config())

        assert result["rope_theta"] == 10000000
        assert result["block_size"] == 16
        # rope_type is "default" — no rope_scaling should be synthesized.
        assert "rope_scaling" not in result

    def test_old_layout_passes_through_unchanged(self):
        from omlx.patches.dflash_draft_config import normalize_dflash_draft_config

        config = _old_layout_config()
        result = normalize_dflash_draft_config(config)

        assert result == config

    def test_root_value_wins_on_conflict(self):
        from omlx.patches.dflash_draft_config import normalize_dflash_draft_config

        config = {
            "rope_theta": 999,
            "block_size": 5,
            "rope_parameters": {"rope_theta": 111, "rope_type": "default"},
            "dflash_config": {"block_size": 99},
        }
        result = normalize_dflash_draft_config(config)

        assert result["rope_theta"] == 999
        assert result["block_size"] == 5

    def test_yarn_rope_parameters_hoist_scaling_and_theta(self):
        from omlx.patches.dflash_draft_config import normalize_dflash_draft_config

        config = {
            "rope_parameters": {
                "rope_theta": 5000000,
                "rope_type": "yarn",
                "factor": 64.0,
            },
        }
        result = normalize_dflash_draft_config(config)

        assert result["rope_scaling"] == {"rope_type": "yarn", "factor": 64.0}
        assert result["rope_theta"] == 5000000

    def test_type_key_precedence_mirrors_initialize_rope(self):
        # mlx_lm's initialize_rope reads "type" first, then "rope_type" —
        # the hoist decision must use the same precedence or it can disagree
        # with the consumer when the two keys contradict.
        from omlx.patches.dflash_draft_config import normalize_dflash_draft_config

        type_wins_yarn = normalize_dflash_draft_config(
            {
                "rope_parameters": {
                    "rope_theta": 1,
                    "type": "yarn",
                    "rope_type": "default",
                    "factor": 2.0,
                },
            }
        )
        assert type_wins_yarn["rope_scaling"]["type"] == "yarn"

        type_wins_default = normalize_dflash_draft_config(
            {
                "rope_parameters": {
                    "rope_theta": 1,
                    "type": "default",
                    "rope_type": "yarn",
                },
            }
        )
        assert "rope_scaling" not in type_wins_default

    def test_explicit_null_root_keys_count_as_missing(self):
        # transformers 4.x-derived configs commonly serialize
        # "rope_scaling": null — an explicit null must not block the hoist.
        from omlx.patches.dflash_draft_config import normalize_dflash_draft_config

        config = {
            "rope_theta": None,
            "rope_scaling": None,
            "block_size": None,
            "rope_parameters": {
                "rope_theta": 5000000,
                "rope_type": "yarn",
                "factor": 64.0,
            },
            "dflash_config": {"block_size": 16},
        }
        result = normalize_dflash_draft_config(config)

        assert result["rope_theta"] == 5000000
        assert result["block_size"] == 16
        assert result["rope_scaling"] == {"rope_type": "yarn", "factor": 64.0}

    def test_null_nested_values_are_not_hoisted(self):
        from omlx.patches.dflash_draft_config import normalize_dflash_draft_config

        config = {
            "rope_parameters": {"rope_theta": None},
            "dflash_config": {"block_size": None},
        }
        result = normalize_dflash_draft_config(config)

        assert result.get("rope_theta") is None
        assert result.get("block_size") is None

    def test_non_dict_nested_values_are_ignored(self):
        from omlx.patches.dflash_draft_config import normalize_dflash_draft_config

        config = {"rope_parameters": "bogus", "dflash_config": 7}
        result = normalize_dflash_draft_config(config)

        assert result == config

    def test_input_not_mutated(self):
        from omlx.patches.dflash_draft_config import normalize_dflash_draft_config

        config = _new_layout_config()
        snapshot = copy.deepcopy(config)

        normalize_dflash_draft_config(config)

        assert config == snapshot


class TestInstallDflashDraftConfigNormalizer:
    """Integration tests against the real dflash-mlx DFlashDraftModelArgs.

    The install is process-global (it monkey-patches the class-level
    ``from_dict`` classmethod) and is never unwound — unlike
    ``dflash_lifecycle``'s restore-on-stop idiom, there is nothing to
    restore: normalization is fill-only per key, so leaving it installed
    for the rest of the test process is output-identical for old-layout
    configs (root keys set, no non-"default" ``rope_parameters``).
    """

    def test_new_layout_constructs_with_hoisted_values(self):
        from dflash_mlx.model import DFlashDraftModelArgs

        from omlx.patches.dflash_draft_config import (
            install_dflash_draft_config_normalizer,
        )

        install_dflash_draft_config_normalizer()
        args = DFlashDraftModelArgs.from_dict(_new_layout_config())

        assert args.rope_theta == 10000000
        assert args.block_size == 16
        assert args.rope_scaling is None
        assert args.dflash_config["mask_token_id"] == 248077
        assert args.dflash_config["target_layer_ids"] == [
            1, 6, 11, 16, 22, 27, 32, 37,
        ]

    def test_old_layout_still_works(self):
        from dflash_mlx.model import DFlashDraftModelArgs

        from omlx.patches.dflash_draft_config import (
            install_dflash_draft_config_normalizer,
        )

        install_dflash_draft_config_normalizer()
        args = DFlashDraftModelArgs.from_dict(_old_layout_config())

        assert args.rope_theta == 10000000
        assert args.block_size == 16
        assert args.rope_scaling == {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 64.0,
            "original_max_position_embeddings": 4096,
            "rope_type": "yarn",
            "type": "yarn",
        }

    def test_missing_rope_theta_and_block_size_still_raises(self):
        from dflash_mlx.model import DFlashDraftModelArgs

        from omlx.patches.dflash_draft_config import (
            install_dflash_draft_config_normalizer,
        )

        install_dflash_draft_config_normalizer()
        # Neither root nor nested layout supplies rope_theta/block_size —
        # the normalizer must not paper over a genuinely incomplete config.
        broken = dict(_BASE_FIELDS)

        with pytest.raises(TypeError):
            DFlashDraftModelArgs.from_dict(broken)

    def test_install_is_idempotent(self):
        from dflash_mlx.model import DFlashDraftModelArgs

        from omlx.patches.dflash_draft_config import (
            install_dflash_draft_config_normalizer,
        )

        install_dflash_draft_config_normalizer()
        wrapped_once = DFlashDraftModelArgs.from_dict.__func__

        install_dflash_draft_config_normalizer()
        wrapped_twice = DFlashDraftModelArgs.from_dict.__func__

        # Second install must not re-wrap the already-wrapped classmethod.
        assert wrapped_once is wrapped_twice
        assert DFlashDraftModelArgs._omlx_draft_config_normalizer_installed is True

        args = DFlashDraftModelArgs.from_dict(_new_layout_config())
        assert args.rope_theta == 10000000
        assert args.block_size == 16
