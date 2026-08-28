# SPDX-License-Identifier: Apache-2.0
"""Tests for safetensors shard identity across symlinked checkpoint layouts."""

from __future__ import annotations

import os

import pytest

from omlx.utils.safetensors_paths import model_shard_matcher


@pytest.fixture
def snapshot(tmp_path):
    """A model directory whose shards are symlinks into a blob store."""
    store = tmp_path / "blobs"
    model_dir = tmp_path / "snapshots" / "rev"
    store.mkdir()
    model_dir.mkdir(parents=True)
    for name in (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ):
        (store / f"blob-{name}").write_bytes(b"")
        os.symlink(store / f"blob-{name}", model_dir / name)
    return model_dir


def test_matches_plain_shards(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard = model_dir / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"")

    matches = model_shard_matcher(model_dir)
    assert matches(shard)
    assert matches(str(shard))
    assert not matches(model_dir / "config.json")
    assert not matches(tmp_path / "other.safetensors")


def test_matches_symlinked_shards(snapshot):
    store = snapshot.parent.parent / "blobs"
    matches = model_shard_matcher(snapshot)

    for shard in sorted(snapshot.glob("*.safetensors")):
        assert matches(shard), shard
    # The blob store holds the same bytes but is not the model directory.
    assert not matches(store / "blob-model-00001-of-00002.safetensors")


def test_matches_shards_through_a_symlinked_model_dir(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    shard = real / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"")
    link = tmp_path / "link"
    os.symlink(real, link)

    via_link = model_shard_matcher(link)
    assert via_link(link / shard.name)
    assert via_link(shard)
    assert model_shard_matcher(real)(link / shard.name)


def test_matches_relative_shard_names(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard = model_dir / "model.safetensors"
    shard.write_bytes(b"")

    monkeypatch.chdir(model_dir)
    assert model_shard_matcher(model_dir)("model.safetensors")


@pytest.mark.parametrize(
    "filename",
    [
        None,
        1234,
        "",
        ".",
        b"/tmp/model/x.safetensors",
        "/tmp/model/nested/model.safetensors",
        "/tmp/model/other.JSON",
    ],
)
def test_non_matching_input_never_matches_and_never_raises(tmp_path, filename):
    """Callers hand ``safe_open`` filenames oMLX does not control."""
    assert not model_shard_matcher(tmp_path / "model")(filename)
