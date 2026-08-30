# SPDX-License-Identifier: Apache-2.0
"""Tests for the DFlash draft-model path resolution (issue #3019).

A bare local model name (``dflash_draft_model="Qwen3.8-27B-DFlash2"``, the
form the admin UI accepts) must resolve against the discovered pool before
reaching dflash-mlx, which otherwise treats it as a Hub repo id and calls
``snapshot_download()`` -- silently falling back to the plain engine on
networks where huggingface.co is unreachable.
"""

from __future__ import annotations

from omlx.engine_pool import EngineEntry, _resolve_local_draft_model_path


def _make_entry(model_id: str, model_path: str) -> EngineEntry:
    return EngineEntry(
        model_id=model_id,
        model_path=model_path,
        model_type="llm",
        engine_type="batched",
        estimated_size=0,
    )


class TestResolveLocalDraftModelPath:
    def test_bare_name_hits_pool_entry(self):
        """A bare local name matching a discovered model resolves to its path."""
        pool = {
            "Qwen3.8-27B-DFlash2": _make_entry(
                "Qwen3.8-27B-DFlash2", "/models/Qwen3.8-27B-DFlash2"
            )
        }
        assert (
            _resolve_local_draft_model_path(pool, "Qwen3.8-27B-DFlash2")
            == "/models/Qwen3.8-27B-DFlash2"
        )

    def test_unmatched_repo_id_passes_through(self):
        """HF repo ids are a legitimate input and must stay untouched."""
        pool = {
            "Qwen3.8-27B-DFlash2": _make_entry(
                "Qwen3.8-27B-DFlash2", "/models/Qwen3.8-27B-DFlash2"
            )
        }
        assert (
            _resolve_local_draft_model_path(pool, "jundot/Qwen3.8-27B-DFlash2")
            == "jundot/Qwen3.8-27B-DFlash2"
        )

    def test_unmatched_bare_name_passes_through(self):
        """No entry -> original value, so dflash-mlx keeps its own fallback."""
        assert _resolve_local_draft_model_path({}, "Qwen3.8-27B-DFlash2") == (
            "Qwen3.8-27B-DFlash2"
        )

    def test_absolute_path_passes_through(self):
        assert _resolve_local_draft_model_path(
            {}, "/models/Qwen3.8-27B-DFlash2"
        ) == "/models/Qwen3.8-27B-DFlash2"

    def test_empty_and_none_return_unchanged(self):
        pool = {"any": _make_entry("any", "/models/any")}
        assert _resolve_local_draft_model_path(pool, None) is None
        assert _resolve_local_draft_model_path(pool, "") == ""