# SPDX-License-Identifier: Apache-2.0
"""Tests for chat-template patches declaring their own reasoning-effort vocabulary."""

import json
from pathlib import Path

from omlx.chat_template_patches import reasoning_effort_vocabulary_for
from omlx.model_discovery import detect_reasoning_effort
from omlx.patches.deepseek_v4.chat_template_v4 import (
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORT_PROMPTS,
)


def _model_dir(tmp_path: Path, model_type: str, template: str | None = None) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
    if template is not None:
        (model_dir / "chat_template.jinja").write_text(template)
    return model_dir


def test_unpatched_model_type_is_not_claimed():
    assert reasoning_effort_vocabulary_for("qwen3_5") is None
    assert reasoning_effort_vocabulary_for("") is None


def test_deepseek_v4_patch_declares_its_vocabulary():
    assert reasoning_effort_vocabulary_for("deepseek_v4") == (
        list(REASONING_EFFORT_PROMPTS),
        DEFAULT_REASONING_EFFORT,
    )


def test_patch_matches_model_type_by_prefix():
    """Variants such as the MTP checkpoint share the patched template."""
    assert reasoning_effort_vocabulary_for("deepseek_v4_mtp") is not None


def test_detector_asks_the_patch_before_reading_disk(tmp_path):
    """DeepSeek V4 ships no template on disk; the vocabulary comes from oMLX."""
    model_dir = _model_dir(tmp_path, "deepseek_v4")
    assert detect_reasoning_effort(model_dir) == (
        list(REASONING_EFFORT_PROMPTS),
        DEFAULT_REASONING_EFFORT,
    )


def test_patch_wins_over_an_unrelated_template_on_disk(tmp_path):
    """The served template is the patch's, so the checkpoint's own is irrelevant."""
    template = "{%- if reasoning_effort not in ('a', 'b') %}{%- endif %}"
    model_dir = _model_dir(tmp_path, "deepseek_v4", template)
    options, _ = detect_reasoning_effort(model_dir)
    assert options == list(REASONING_EFFORT_PROMPTS)


def test_caller_supplied_model_type_skips_the_config_read(tmp_path):
    model_dir = tmp_path / "no-config"
    model_dir.mkdir()
    options, default = detect_reasoning_effort(model_dir, "deepseek_v4")
    assert options == list(REASONING_EFFORT_PROMPTS)
    assert default == DEFAULT_REASONING_EFFORT


def test_ordinary_model_still_reads_its_template(tmp_path):
    template = (
        "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}"
        "{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}{%- endif %}"
    )
    model_dir = _model_dir(tmp_path, "qwen3_5", template)
    assert detect_reasoning_effort(model_dir) == (["xhigh", "medium", "low"], "xhigh")
