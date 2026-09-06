"""JANGQ PLE layout support in the qwen4_exp compat loader.

JANGQ checkpoints nest the ngram table one level shallower than the
oQ layout (ple.ngram_embedding, no ple_embedding level, no model.
prefix) and retain the historical spellings as phantom index keys
with no tensor behind them. These tests use a synthetic safetensors
directory: no model download, no GPU, no SSD array needed.
""",

import json

import numpy as np
import pytest

safetensors_numpy = pytest.importorskip("safetensors.numpy")

from omlx.patches.mlx_vlm_qwen4_exp_compat import (
    apply_mlx_vlm_qwen4_exp_compat_patch,
)

apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    DiskBackedShardedEmbedding,
    _SafeTensorMMap,
)

_DOUBLED = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
_RUNTIME_DOUBLED = (
    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
)
_JANG = "language_model.layers.1.ple.ngram_embedding"


def _triple(base):
    return [f"{base}.weight", f"{base}.scales", f"{base}.biases"]


def _write_fake_ckpt(tmp_path, real_bases, phantom_bases):
    """Write one tiny safetensors file with real triples; the index also
    lists phantom triples that have no tensor behind them."""
    tensors = {}
    for shard in (0, 1):
        for base in real_bases:
            stem = f"{base}.shards.{shard}"
            tensors[f"{stem}.weight"] = np.zeros((4, 4), dtype=np.uint32)
            tensors[f"{stem}.scales"] = np.ones((4, 1), dtype=np.float16)
            tensors[f"{stem}.biases"] = np.zeros((4, 1), dtype=np.float16)
    fname = "model-00001-of-00001.safetensors"
    safetensors_numpy.save_file(tensors, str(tmp_path / fname))
    weight_map = {}
    for shard in (0, 1):
        for base in real_bases + phantom_bases:
            for key in _triple(f"{base}.shards.{shard}"):
                weight_map[key] = fname
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map})
    )
    return tmp_path


def test_disk_backed_ple_selects_header_backed_jang_base(tmp_path):
    """Phantom doubled keys must not shadow the real JANG triple."""
    root = _write_fake_ckpt(tmp_path, [_JANG], [_DOUBLED, _RUNTIME_DOUBLED])
    emb = DiskBackedShardedEmbedding(root, _DOUBLED, 8, 32, 2)
    try:
        for shard in (0, 1):
            weight_key, scales_key, biases_key, bits, group = emb._shard_specs[
                shard
            ]
            assert weight_key == f"{_JANG}.shards.{shard}.weight"
            assert "ple_embedding" not in weight_key
            assert scales_key.endswith(".scales") and biases_key.endswith(
                ".biases"
            )
            assert (bits, group) == (4, 32)
    finally:
        emb.close()


def test_disk_backed_ple_keeps_historical_order(tmp_path):
    """When the doubled triple is real, it still wins over a phantom
    shallow spelling: existing checkpoints resolve exactly as before."""
    root = _write_fake_ckpt(tmp_path, [_DOUBLED], [_JANG])
    emb = DiskBackedShardedEmbedding(root, _DOUBLED, 8, 32, 2)
    try:
        weight_key = emb._shard_specs[0][0]
        assert weight_key == f"{_DOUBLED}.shards.0.weight"
    finally:
        emb.close()


def test_disk_backed_ple_reports_absent_shard(tmp_path):
    """No spelling backed anywhere still raises instead of picking a
    phantom."""
    root = _write_fake_ckpt(tmp_path, [], [_DOUBLED, _RUNTIME_DOUBLED, _JANG])
    with pytest.raises(KeyError):
        DiskBackedShardedEmbedding(root, _DOUBLED, 8, 32, 2)


def test_virtual_ple_candidates_include_jang_spelling():
    from omlx.patches.mlx_vlm_qwen4_exp_compat.virtual_ple import (
        _candidate_keys,
    )

    cfg = {
        "model_type": "qwen4_exp",
        "text_config": {"ple_layer_ids": [2], "split_ngram_parts": 2},
    }
    keys = set(_candidate_keys(cfg))
    assert f"{_JANG}.shards.0.weight" in keys
    assert f"{_DOUBLED}.shards.0.weight" in keys
    assert f"{_RUNTIME_DOUBLED}.shard_1.weight" in keys


def test_storage_regex_matches_both_layouts():
    from mlx_vlm.models.qwen4_exp.qwen4_exp import _NGRAM_STORAGE_RE

    assert _NGRAM_STORAGE_RE.search(f"{_JANG}.shards.0.weight")
    assert _NGRAM_STORAGE_RE.search(f"{_JANG}.shards.12.biases")
    assert _NGRAM_STORAGE_RE.search(f"{_RUNTIME_DOUBLED}.shards.3.scales")
    assert _NGRAM_STORAGE_RE.search(f"{_DOUBLED}.shard_0.weight")
    assert not _NGRAM_STORAGE_RE.search(
        "language_model.layers.1.mlp.gate_proj.weight"
    )


def test_checkpoint_key_normalization():
    from mlx_vlm.models.qwen4_exp.qwen4_exp import _normalize_checkpoint_key

    assert (
        _normalize_checkpoint_key("visual.blocks.0.attn.qkv.weight")
        == "vision_tower.blocks.0.attn.qkv.weight"
    )
    assert (
        _normalize_checkpoint_key("language_model.layers.0.mlp.gate_proj.weight")
        == "language_model.model.layers.0.mlp.gate_proj.weight"
    )
    assert (
        _normalize_checkpoint_key("language_model.embed_tokens.weight")
        == "language_model.model.embed_tokens.weight"
    )
    assert (
        _normalize_checkpoint_key(f"{_JANG}.shards.0.weight")
        == f"{_RUNTIME_DOUBLED}.shards.0.weight"
    )
    # Historical spellings pass through unchanged.
    assert (
        _normalize_checkpoint_key(f"{_RUNTIME_DOUBLED}.shards.0.weight")
        == f"{_RUNTIME_DOUBLED}.shards.0.weight"
    )
    assert (
        _normalize_checkpoint_key("language_model.mtp.layers.0.foo.weight")
        == "language_model.mtp.layers.0.foo.weight"
    )
    assert _normalize_checkpoint_key("lm_head.weight") == "lm_head.weight"
    assert (
        _normalize_checkpoint_key("language_model.layers.1.ple.conv1d_weight")
        == "language_model.model.layers.1.ple.conv1d.weight"
    )
    assert (
        _normalize_checkpoint_key(
            "language_model.layers.1.ple.ngram_heads_vocab_sizes"
        )
        == "language_model.model.layers.1.ple.ple_embedding."
        "ngram_heads_vocab_sizes"
    )


def test_oq_ngram_predicate_matches_jang():
    from omlx.oq import _is_qwen4_exp_ngram_embedding_tensor

    cfg = {"model_type": "qwen4_exp", "text_config": {}}
    assert _is_qwen4_exp_ngram_embedding_tensor(f"{_JANG}.shards.0", cfg)
    assert not _is_qwen4_exp_ngram_embedding_tensor(
        "language_model.layers.1.mlp.gate_proj", cfg
    )
