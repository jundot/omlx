# SPDX-License-Identifier: Apache-2.0
"""Tests for the Bonsai DSpark drafter support.

Covers:
  - BonsaiDSparkConfig: round-trip, from_json, properties
  - GGUF parser: header parsing, Q4_1 dequant, BF16 reader, key remapping
  - BonsaiDSparkDrafter: construction, bind_target_embedding, log-SNR embedding
  - BonsaiTarget: construction, is_vlm flag
  - generate loop: prefill, greedy round-trip with mock target+drafter
  - model_settings: dspark_enabled fields, conflict with dflash/mtp
  - engine_pool wiring: DSParkEngine selected when dspark_enabled+draft set
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import mlx.core as mx

from omlx.custom_kernels.bonsai.dspark.config import BonsaiDSparkConfig
from omlx.custom_kernels.bonsai.dspark.convert import (
    _dequant_q4_0,
    _dequant_q4_1,
    _parse_gguf_header,
    _read_bf16,
    _remap_key,
)
from omlx.custom_kernels.bonsai.dspark.drafter import (
    BonsaiDSparkDrafter,
    _log_snr_sinusoidal,
)
from omlx.custom_kernels.bonsai.dspark.target import BonsaiTarget


# ---------------------------------------------------------------------------
# BonsaiDSparkConfig
# ---------------------------------------------------------------------------


class TestBonsaiDSparkConfig:
    def test_defaults(self):
        cfg = BonsaiDSparkConfig()
        assert cfg.family == "bonsai"
        assert cfg.hidden_size == 5120
        assert cfg.block_size == 4
        assert cfg.markov_rank == 256
        assert cfg.log_snr_dim == 128
        assert cfg.log_snr_inference == 10.0

    def test_properties(self):
        cfg2 = BonsaiDSparkConfig(hidden_size=5120, num_attention_heads=20, head_dim=256)
        assert cfg2.scaling == pytest.approx(1.0 / 256.0 ** 0.5)
        assert cfg2.n_kv_heads == cfg2.num_key_value_heads
        assert cfg2.attention_k_eq_v is False

    def test_round_trip_json(self, tmp_path):
        cfg = BonsaiDSparkConfig(
            hidden_size=5120,
            vocab_size=248320,
            num_hidden_layers=6,
            target_layer_ids=[1, 16, 31, 46, 61],
            block_size=4,
        )
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg.to_dict()))
        cfg2 = BonsaiDSparkConfig.from_json(p)
        assert cfg2.hidden_size == 5120
        assert cfg2.vocab_size == 248320
        assert cfg2.target_layer_ids == [1, 16, 31, 46, 61]
        assert cfg2.block_size == 4
        assert cfg2.family == "bonsai"

    def test_from_json_wrong_family_raises(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text(json.dumps({"family": "qwen3", "hidden_size": 1024, "vocab_size": 100,
                                  "num_hidden_layers": 2, "num_attention_heads": 4,
                                  "block_size": 4, "target_layer_ids": [1]}))
        with pytest.raises(ValueError, match="family='bonsai'"):
            BonsaiDSparkConfig.from_json(p)


# ---------------------------------------------------------------------------
# GGUF convert helpers
# ---------------------------------------------------------------------------


class TestKeyRemapping:
    def test_token_embd_skipped(self):
        assert _remap_key("token_embd.weight") is None

    def test_block_attention(self):
        assert _remap_key("blk.0.attn_q.weight") == "layers.0.self_attn.q_proj.weight"
        assert _remap_key("blk.3.attn_k.weight") == "layers.3.self_attn.k_proj.weight"
        assert _remap_key("blk.5.attn_output.weight") == "layers.5.self_attn.o_proj.weight"
        assert _remap_key("blk.2.ffn_gate.weight") == "layers.2.mlp.gate_proj.weight"
        assert _remap_key("blk.1.ffn_down.weight") == "layers.1.mlp.down_proj.weight"
        assert _remap_key("blk.0.attn_norm.weight") == "layers.0.input_layernorm.weight"
        assert _remap_key("blk.0.ffn_norm.weight") == "layers.0.post_attention_layernorm.weight"

    def test_dspark_tensors(self):
        assert _remap_key("dspark.fc.weight") == "fc.weight"
        assert _remap_key("dspark.hidden_norm.weight") == "hidden_norm.weight"
        assert _remap_key("dspark.log_snr_fc1.weight") == "log_snr_fc1.weight"
        assert _remap_key("dspark.log_snr_fc1.bias") == "log_snr_fc1.bias"
        assert _remap_key("dspark.log_snr_fc2.weight") == "log_snr_fc2.weight"
        assert _remap_key("dspark.log_snr_fc2.bias") == "log_snr_fc2.bias"
        assert _remap_key("dspark.markov_head_a.weight") == "markov_head.markov_w1.weight"
        assert _remap_key("dspark.markov_head_b.weight") == "markov_head.markov_w2.weight"
        assert _remap_key("dspark.confidence_head.weight") == "confidence_head.proj.weight"
        assert _remap_key("dspark.confidence_head.bias") == "confidence_head.proj.bias"

    def test_output_tensors(self):
        assert _remap_key("output.weight") == "lm_head.weight"
        assert _remap_key("output_norm.weight") == "norm.weight"


class TestQ4_1Dequant:
    def _make_q4_1_block(self, delta: float, d_min: float, nibbles: list[int]) -> bytes:
        """Build one 20-byte Q4_1 block."""
        assert len(nibbles) == 32
        # Pack 32 nibbles as 16 bytes (lo nibble first)
        packed = bytearray(16)
        for i in range(16):
            packed[i] = (nibbles[2 * i] & 0x0F) | ((nibbles[2 * i + 1] & 0x0F) << 4)
        delta_bytes = np.array([delta], dtype=np.float16).tobytes()
        dmin_bytes = np.array([d_min], dtype=np.float16).tobytes()
        return bytes(packed) + delta_bytes + dmin_bytes

    def test_single_block_zero(self):
        block = self._make_q4_1_block(1.0, 0.0, [0] * 32)
        result = _dequant_q4_1(block, 32)
        assert result.shape == (32,)
        np.testing.assert_allclose(result, 0.0, atol=1e-3)

    def test_single_block_values(self):
        delta, d_min = 2.0, -1.0
        nibbles = list(range(16)) + list(range(16))  # 0..15 twice
        block = self._make_q4_1_block(delta, d_min, nibbles)
        result = _dequant_q4_1(block, 32)
        expected = np.array(nibbles, dtype=np.float32) * delta + d_min
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)

    def test_two_blocks(self):
        b1 = self._make_q4_1_block(1.0, 0.0, [3] * 32)
        b2 = self._make_q4_1_block(2.0, -5.0, [7] * 32)
        result = _dequant_q4_1(b1 + b2, 64)
        assert result.shape == (64,)
        np.testing.assert_allclose(result[:32], 3.0, rtol=1e-2)
        np.testing.assert_allclose(result[32:], 7.0 * 2.0 + (-5.0), rtol=1e-2)


class TestBF16Reader:
    def test_reads_known_value(self):
        # BF16 representation of 1.0: sign=0, exp=127 (0x7F), mantissa=0 → 0x3F80
        val = np.array([0x3F80], dtype=np.uint16).tobytes()
        result = _read_bf16(val, 1)
        np.testing.assert_allclose(result, [1.0], atol=1e-5)

    def test_reads_negative(self):
        # BF16 representation of -2.0: 0xC000
        val = np.array([0xC000], dtype=np.uint16).tobytes()
        result = _read_bf16(val, 1)
        np.testing.assert_allclose(result, [-2.0], atol=1e-5)

    def test_multiple_values(self):
        vals = np.array([0.5, 1.5, -3.0], dtype=np.float32)
        # Pack as BF16 (take upper 2 bytes of float32)
        bf16_vals = (vals.view(np.uint32) >> 16).astype(np.uint16)
        raw = bf16_vals.tobytes()
        result = _read_bf16(raw, 3)
        np.testing.assert_allclose(result, vals, rtol=1e-2)


# ---------------------------------------------------------------------------
# Minimal GGUF header building for parse tests
# ---------------------------------------------------------------------------

def _build_minimal_gguf(kv: dict, tensor_infos: list[dict]) -> bytes:
    """Build a minimal valid GGUF binary for testing _parse_gguf_header."""
    def _pack_str(s: str) -> bytes:
        enc = s.encode()
        return struct.pack("<Q", len(enc)) + enc

    def _pack_kv(key: str, vtype: int, value) -> bytes:
        out = _pack_str(key) + struct.pack("<I", vtype)
        if vtype == 4:    # uint32
            out += struct.pack("<I", value)
        elif vtype == 8:  # string
            out += _pack_str(value)
        elif vtype == 10: # uint64
            out += struct.pack("<Q", value)
        return out

    def _pack_tensor(t: dict) -> bytes:
        out = _pack_str(t["name"])
        dims = t["dims"]
        out += struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", t["dtype"])
        out += struct.pack("<Q", t["offset"])
        return out

    # Build header
    kv_bytes = b""
    for k, (vtype, v) in kv.items():
        kv_bytes += _pack_kv(k, vtype, v)

    tensor_bytes = b""
    for t in tensor_infos:
        tensor_bytes += _pack_tensor(t)

    n_tensors = len(tensor_infos)
    n_kv = len(kv)
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", n_tensors) + struct.pack("<Q", n_kv)
    return header + kv_bytes + tensor_bytes


class TestGGUFHeaderParser:
    def test_basic_parse(self):
        data = _build_minimal_gguf(
            kv={"dspark.block_size": (4, 4), "general.name": (8, "test")},
            tensor_infos=[
                {"name": "output_norm.weight", "dims": [16], "dtype": 30, "offset": 0},
            ],
        )
        # Pad to 32-byte alignment + add dummy data
        padding = (32 - (len(data) % 32)) % 32
        data += b"\x00" * (padding + 64)

        kv, tensors, data_offset = _parse_gguf_header(data)
        assert kv["dspark.block_size"] == 4
        assert kv["general.name"] == "test"
        assert len(tensors) == 1
        assert tensors[0]["name"] == "output_norm.weight"
        assert tensors[0]["dtype"] == 30
        assert tensors[0]["shape"] == [16]

    def test_bad_magic_raises(self):
        with pytest.raises(ValueError, match="Not a GGUF"):
            _parse_gguf_header(b"BLAH\x03\x00\x00\x00" + b"\x00" * 20)

    def test_unsupported_version_raises(self):
        data = b"GGUF" + struct.pack("<I", 99) + b"\x00" * 16
        with pytest.raises(ValueError, match="version 99 not supported"):
            _parse_gguf_header(data)


# ---------------------------------------------------------------------------
# BonsaiDSparkDrafter
# ---------------------------------------------------------------------------


class TestBonsaiDSparkDrafter:
    @pytest.fixture
    def small_config(self):
        return BonsaiDSparkConfig(
            hidden_size=64,
            vocab_size=256,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            block_size=4,
            mask_token_id=1,
            target_layer_ids=[0, 1],
            markov_rank=8,
            enable_confidence_head=True,
        )

    def test_construction(self, small_config):
        drafter = BonsaiDSparkDrafter(small_config)
        assert drafter.block_size == 4
        assert drafter.mask_token_id == 1
        assert drafter.markov_head is not None
        assert drafter.confidence_head is not None

    def test_make_ctx_cache(self, small_config):
        drafter = BonsaiDSparkDrafter(small_config)
        caches = drafter.make_ctx_cache()
        assert len(caches) == 2
        assert caches[0].k is None

    def test_bind_target_embedding_mlxlm(self, small_config):
        drafter = BonsaiDSparkDrafter(small_config)
        import mlx.nn as nn
        fake_embed = nn.Embedding(256, 64)
        fake_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=fake_embed))
        drafter.bind_target_embedding(fake_model)
        assert drafter.embed_tokens is fake_embed

    def test_bind_target_embedding_vlm(self, small_config):
        drafter = BonsaiDSparkDrafter(small_config)
        import mlx.nn as nn
        fake_embed = nn.Embedding(256, 64)
        fake_model = SimpleNamespace(
            language_model=SimpleNamespace(
                model=SimpleNamespace(embed_tokens=fake_embed)
            )
        )
        drafter.bind_target_embedding(fake_model)
        assert drafter.embed_tokens is fake_embed

    def test_bind_target_embedding_unknown_raises(self, small_config):
        drafter = BonsaiDSparkDrafter(small_config)
        with pytest.raises(ValueError, match="Cannot find embed_tokens"):
            drafter.bind_target_embedding(SimpleNamespace(x=1))


class TestLogSnrEmbedding:
    def test_shape(self):
        emb = _log_snr_sinusoidal(10.0, 128)
        assert emb.shape == (1, 1, 128)

    def test_different_values_differ(self):
        e1 = _log_snr_sinusoidal(0.0, 128)
        e2 = _log_snr_sinusoidal(10.0, 128)
        diff = float(mx.abs(e1 - e2).max())
        assert diff > 0.01

    def test_same_value_same(self):
        e1 = _log_snr_sinusoidal(5.0, 128)
        e2 = _log_snr_sinusoidal(5.0, 128)
        diff = float(mx.abs(e1 - e2).max())
        assert diff == 0.0

    def test_dtype_float32(self):
        emb = _log_snr_sinusoidal(1.0, 64)
        assert emb.dtype == mx.float32


# ---------------------------------------------------------------------------
# BonsaiTarget
# ---------------------------------------------------------------------------


class TestBonsaiTarget:
    def _make_fake_vlm(self):
        """Build a fake VLM model with language_model."""
        import mlx.nn as nn

        class _FakeLM:
            def make_cache(self):
                return [None] * 64

            def __call__(self, inputs, cache=None, **kwargs):
                capture = kwargs.get("capture_layer_ids")
                B, L = inputs.shape
                H = 32
                V = 128
                logits = mx.zeros((B, L, V))
                if capture is not None:
                    hidden_states = [mx.zeros((B, L, H)) for _ in capture]
                    return SimpleNamespace(logits=logits, hidden_states=hidden_states)
                return SimpleNamespace(logits=logits, hidden_states=None)

        model = SimpleNamespace(language_model=_FakeLM())
        tokenizer = SimpleNamespace()
        return model, tokenizer

    def test_construction_requires_language_model(self):
        with pytest.raises(ValueError, match="language_model"):
            BonsaiTarget(SimpleNamespace(x=1), None)

    def test_is_vlm(self):
        model, tok = self._make_fake_vlm()
        target = BonsaiTarget(model, tok)
        assert target.is_vlm is True

    def test_make_cache(self):
        model, tok = self._make_fake_vlm()
        target = BonsaiTarget(model, tok)
        cache = target.make_cache()
        assert isinstance(cache, list)

    def test_plain_forward(self):
        model, tok = self._make_fake_vlm()
        target = BonsaiTarget(model, tok)
        cache = target.make_cache()
        ids = mx.array([[1, 2, 3]])
        logits = target.plain(ids, cache)
        assert logits.shape == (1, 3, 128)

    def test_run_with_tap(self):
        model, tok = self._make_fake_vlm()
        target = BonsaiTarget(model, tok)
        cache = target.make_cache()
        ids = mx.array([[1, 2]])
        logits, fused = target.run(ids, cache, tap=[0, 1, 2, 3, 4])
        assert logits.shape == (1, 2, 128)
        # fused = concatenation of 5 tapped hidden states [1, 2, 32] → [1, 2, 160]
        assert fused.shape == (1, 2, 5 * 32)

    def test_verify_tap_noop(self):
        model, tok = self._make_fake_vlm()
        target = BonsaiTarget(model, tok)
        target.verify_tap()   # should not raise


# ---------------------------------------------------------------------------
# Generate loop (with mocked target + drafter)
# ---------------------------------------------------------------------------


class TestSpeculativeGenerate:
    """Smoke-test the generate loop with minimal fake implementations."""

    def _make_target_and_drafter(self, vocab=32, hidden=16, n_tap=2):
        """Build tiny fake target + drafter for a round-trip test."""
        import mlx.nn as nn

        class _FakeLM:
            def make_cache(self):
                return [None] * 4

            def __call__(self, inputs, cache=None, **kwargs):
                capture = kwargs.get("capture_layer_ids")
                B, L = inputs.shape
                # Always predict token 5 (argmax)
                # Build logits by embedding a constant index
                import mlx.nn as _nn
                logits = _nn.Embedding(vocab, vocab)(mx.zeros((B * L,), dtype=mx.int32))
                logits = mx.zeros((B, L, vocab))
                # Use scatter via eye row: pick row 5 of identity matrix → shape [vocab]
                eye_row = mx.equal(mx.arange(vocab), mx.array(5)).astype(mx.float32) * 10.0
                logits = mx.broadcast_to(eye_row.reshape(1, 1, vocab), (B, L, vocab))
                if capture is not None:
                    hs = [mx.zeros((B, L, hidden)) for _ in capture]
                    return SimpleNamespace(logits=logits, hidden_states=hs)
                return SimpleNamespace(logits=logits)

        model = SimpleNamespace(
            language_model=_FakeLM(),
        )
        tokenizer = SimpleNamespace(
            eos_token_id=3,
            decode=lambda ids: " ".join(str(i) for i in ids if i != 3),
        )
        target = BonsaiTarget(model, tokenizer)

        cfg = BonsaiDSparkConfig(
            hidden_size=hidden,
            vocab_size=vocab,
            num_hidden_layers=1,
            intermediate_size=hidden * 2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=hidden // 2,
            block_size=3,
            mask_token_id=2,
            target_layer_ids=list(range(n_tap)),
            markov_rank=4,
            enable_confidence_head=False,
            log_snr_dim=8,
        )
        drafter = BonsaiDSparkDrafter(cfg)
        drafter.embed_tokens = nn.Embedding(vocab, hidden)
        return target, drafter, tokenizer

    def test_generate_produces_tokens(self):
        from omlx.custom_kernels.bonsai.dspark.generate import speculative_generate

        target, drafter, tokenizer = self._make_target_and_drafter()

        result = speculative_generate(
            target,
            tokenizer,
            drafter,
            prompt_ids=[1, 2],
            max_new_tokens=5,
            temperature=0.0,
        )
        assert result.num_tokens > 0
        assert len(result.token_ids) > 0
        assert result.target_forwards >= 1

    def test_generate_respects_eos(self):
        from omlx.custom_kernels.bonsai.dspark.generate import speculative_generate

        target, drafter, tokenizer = self._make_target_and_drafter()
        # Override so model always predicts eos_token_id=3
        import mlx.nn as nn

        class _EosLM:
            def make_cache(self):
                return [None] * 4

            def __call__(self, inputs, cache=None, **kwargs):
                capture = kwargs.get("capture_layer_ids")
                B, L = inputs.shape
                eye_row = mx.equal(mx.arange(32), mx.array(3)).astype(mx.float32) * 10.0
                logits = mx.broadcast_to(eye_row.reshape(1, 1, 32), (B, L, 32))
                if capture is not None:
                    hs = [mx.zeros((B, L, 16)) for _ in capture]
                    return SimpleNamespace(logits=logits, hidden_states=hs)
                return SimpleNamespace(logits=logits)

        target._lm = _EosLM()
        result = speculative_generate(
            target, tokenizer, drafter,
            prompt_ids=[1, 2],
            max_new_tokens=20,
            temperature=0.0,
        )
        # Should stop quickly once eos is committed
        assert result.finish_reason == "stop"
        assert result.num_tokens <= 5  # at most a few rounds before eos


# ---------------------------------------------------------------------------
# ModelSettings: dspark fields and conflict
# ---------------------------------------------------------------------------


class TestModelSettingsDSpark:
    def test_dspark_fields_default(self):
        from omlx.model_settings import ModelSettings
        ms = ModelSettings()
        assert ms.dspark_enabled is False
        assert ms.dspark_draft_model is None
        assert ms.dspark_max_draft_tokens == 2
        assert ms.dspark_log_snr is None

    def test_dspark_enabled_with_dflash_raises(self):
        from omlx.model_settings import ModelSettings
        with pytest.raises(ValueError, match="mutually exclusive"):
            ModelSettings(dspark_enabled=True, dflash_enabled=True,
                          dspark_draft_model="/tmp/x", dflash_draft_model="/tmp/y")

    def test_dspark_enabled_with_mtp_raises(self):
        from omlx.model_settings import ModelSettings
        with pytest.raises(ValueError, match="mutually exclusive"):
            ModelSettings(dspark_enabled=True, mtp_enabled=True,
                          dspark_draft_model="/tmp/x")

    def test_dspark_alone_is_fine(self):
        from omlx.model_settings import ModelSettings
        ms = ModelSettings(dspark_enabled=True, dspark_draft_model="/tmp/drafter")
        assert ms.dspark_enabled is True
        assert ms.dspark_draft_model == "/tmp/drafter"
