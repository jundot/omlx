# SPDX-License-Identifier: Apache-2.0
"""EnginePool resolution tests for canonical DS4 model identities."""

import json
from unittest.mock import MagicMock

from omlx.engine_pool import EnginePool
from omlx.model_settings import ModelSettings


def _pool_with_gguf(tmp_path, filename: str = "Foo.gguf") -> EnginePool:
    (tmp_path / filename).write_bytes(b"0" * 1000)
    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    return pool


def test_ds4_semantic_suffixes_do_not_resolve(tmp_path):
    """Semantic suffixes are not alternative identities for a DS4 entry."""
    pool = _pool_with_gguf(tmp_path)

    for suffix in ("-" + "chat", "-" + "reasoner", "-think-" + "max"):
        requested = f"foo{suffix}"
        assert pool.resolve_model_id(requested, None) == requested


def test_ds4_provider_namespace_does_not_resolve(tmp_path):
    """Provider prefixes are not alternative identities for a DS4 entry."""
    pool = _pool_with_gguf(tmp_path)

    assert pool.resolve_model_id("provider/foo", None) == "provider/foo"


def test_user_model_alias_does_not_gain_semantic_variants(tmp_path):
    """A configured alias resolves exactly, without generated variants."""
    pool = _pool_with_gguf(tmp_path)
    settings_manager = MagicMock()
    settings_manager.get_exposed_profile_source_model_id.return_value = None
    settings_manager.get_all_settings.return_value = {
        "foo": ModelSettings(model_alias="gpt-4o"),
    }

    assert pool.resolve_model_id("gpt-4o", settings_manager) == "foo"
    generated = "gpt-4o-" + "reasoner"
    assert pool.resolve_model_id(generated, settings_manager) == generated


def test_model_alias_ignores_undiscovered_settings_entry(tmp_path):
    """Stale per-model settings do not resolve aliases to missing models."""
    pool = _pool_with_gguf(tmp_path)
    settings_manager = MagicMock()
    settings_manager.get_exposed_profile_source_model_id.return_value = None
    settings_manager.get_all_settings.return_value = {
        "missing": ModelSettings(model_alias="gpt-4o"),
    }

    assert pool.resolve_model_id("gpt-4o", settings_manager) == "gpt-4o"


def test_ds4_gguf_file_path_is_not_dropped_as_missing(tmp_path):
    """DS4 GGUF entries are files, not directories with config.json."""
    pool = _pool_with_gguf(tmp_path)
    entry = pool.get_entry("foo")

    assert entry is not None
    pool._raise_if_model_path_missing_locked("foo", entry)
    assert pool.get_entry("foo") is entry


def test_ds4_source_filename_and_path_resolve_to_normalized_id(tmp_path):
    """Original GGUF filenames and paths still identify the physical entry."""
    filename = "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-v2.gguf"
    gguf = tmp_path / filename
    pool = _pool_with_gguf(tmp_path, filename)
    model_id = "deepseek-v4-flash-iq2xxs-w2q2k-aprojq8-sexpq8-outq8-v2"

    assert pool.resolve_model_id(filename, None) == model_id
    assert pool.resolve_model_id(filename.removesuffix(".gguf"), None) == model_id
    assert pool.resolve_model_id(str(gguf), None) == model_id


def test_ds4_source_filename_disambiguates_mlx_collision(tmp_path):
    """A .gguf source name can select DS4 when the stem collides with MLX."""
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    mlx_model = tmp_path / "foo"
    mlx_model.mkdir()
    (mlx_model / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (mlx_model / "model.safetensors").write_bytes(b"0" * 1000)
    pool = EnginePool()
    pool.discover_models(str(tmp_path))

    assert pool.resolve_model_id("Foo", None) == "foo"
    assert pool.resolve_model_id("Foo.gguf", None) == "foo:ds4"


def test_ds4_full_path_matches_exact_path_before_basename(tmp_path):
    """Full GGUF paths do not resolve to another entry with the same basename."""
    first = tmp_path / "a" / "model.gguf"
    second = tmp_path / "b" / "model.gguf"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"0" * 1000)
    second.write_bytes(b"1" * 1000)
    pool = EnginePool()
    pool.discover_models(str(tmp_path))

    assert pool.resolve_model_id(str(first), None) == "a"
    assert pool.resolve_model_id(str(second), None) == "b"
