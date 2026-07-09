# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.model_loading.maybe_load_custom_quantization."""

import sys
import types
from unittest.mock import MagicMock

import pytest

from omlx.utils import model_loading
from omlx.utils.model_loading import (
    maybe_apply_pre_load_patches,
    maybe_load_custom_quantization,
)


def _write_config(tmp_path, body: str) -> str:
    (tmp_path / "config.json").write_text(body)
    return str(tmp_path)


def _write_mtp_index(tmp_path, has_mtp: bool) -> None:
    """Drop a stub ``model.safetensors.index.json`` next to config.json so
    ``_checkpoint_has_mtp_weights`` resolves deterministically in tests."""
    keys = {"language_model.model.embed_tokens.weight": "model.safetensors"}
    if has_mtp:
        keys["language_model.mtp.fc.weight"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata": {}, "weight_map": ' + str(keys).replace("'", '"') + "}"
    )


class TestNoDispatch:
    """Cases where the dispatcher should return None and let the caller
    fall back to the standard mlx-lm/mlx-vlm load path."""

    def test_missing_config_returns_none(self, tmp_path):
        # tmp_path has no config.json
        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) is None

    def test_malformed_config_returns_none(self, tmp_path):
        path = _write_config(tmp_path, "{not valid json")
        assert maybe_load_custom_quantization(path, is_vlm=False) is None

    def test_no_quantization_config_returns_none(self, tmp_path):
        path = _write_config(tmp_path, '{"model_type": "llama"}')
        assert maybe_load_custom_quantization(path, is_vlm=False) is None

    def test_empty_quant_method_returns_none(self, tmp_path):
        path = _write_config(tmp_path, '{"quantization_config": {}}')
        assert maybe_load_custom_quantization(path, is_vlm=False) is None

    def test_unknown_quant_method_returns_none(self, tmp_path):
        # Methods we don't dispatch on (mlx-lm may or may not handle them
        # natively; either way the dispatcher stays out of the way).
        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "awq"}}',
        )
        assert maybe_load_custom_quantization(path, is_vlm=False) is None


def _install_paroquant_stub(monkeypatch, load_impl):
    """Register a minimal paroquant.inference.backends.mlx.load stub."""
    names = [
        "paroquant",
        "paroquant.inference",
        "paroquant.inference.backends",
        "paroquant.inference.backends.mlx",
    ]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    load_mod = types.ModuleType("paroquant.inference.backends.mlx.load")
    load_mod.load = load_impl
    monkeypatch.setitem(sys.modules, "paroquant.inference.backends.mlx.load", load_mod)


class TestParoquantDispatch:
    """Cases where quant_method == 'paroquant'."""

    def test_paroquant_missing_raises_install_hint(self, tmp_path, monkeypatch):
        # Force the import to fail: shadow the package with a sentinel that
        # blocks submodule resolution. setitem(..., None) makes `import X`
        # raise ImportError in the standard machinery.
        for name in [
            "paroquant",
            "paroquant.inference",
            "paroquant.inference.backends",
            "paroquant.inference.backends.mlx",
            "paroquant.inference.backends.mlx.load",
        ]:
            monkeypatch.setitem(sys.modules, name, None)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        with pytest.raises(ImportError, match="paroquant"):
            maybe_load_custom_quantization(path, is_vlm=False)

    def test_paroquant_text_load_returns_tuple(self, tmp_path, monkeypatch):
        def fake_load(model_path, force_text):
            assert force_text is True
            return "MODEL", "PROC", False

        _install_paroquant_stub(monkeypatch, fake_load)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        result = maybe_load_custom_quantization(path, is_vlm=False)
        assert result == ("MODEL", "PROC")

    def test_paroquant_vlm_load_returns_tuple(self, tmp_path, monkeypatch):
        def fake_load(model_path, force_text):
            assert force_text is False
            return "VLM_MODEL", "VLM_PROC", True

        _install_paroquant_stub(monkeypatch, fake_load)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        result = maybe_load_custom_quantization(path, is_vlm=True)
        assert result == ("VLM_MODEL", "VLM_PROC")

    def test_paroquant_text_only_for_vlm_load_raises(self, tmp_path, monkeypatch):
        # is_vlm=True but the loader returned (..., loaded_is_vlm=False).
        def fake_load(model_path, force_text):
            return "MODEL", "PROC", False

        _install_paroquant_stub(monkeypatch, fake_load)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        with pytest.raises(ValueError, match="text-only"):
            maybe_load_custom_quantization(path, is_vlm=True)

    def test_quant_method_case_insensitive(self, tmp_path, monkeypatch):
        # The dispatcher lowercases quant_method, so mixed-case configs
        # (e.g. produced by other tooling) still hit the paroquant path.
        captured = {}

        def fake_load(model_path, force_text):
            captured["called"] = True
            return "M", "P", False

        _install_paroquant_stub(monkeypatch, fake_load)
        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "ParoQuant"}}',
        )
        assert maybe_load_custom_quantization(path, is_vlm=False) == ("M", "P")
        assert captured["called"] is True


class TestLlama4PreLoadDispatch:
    @pytest.mark.parametrize(
        "body",
        [
            '{"model_type": "llama4", "text_config": {}}',
            '{"model_type": "mllama", "text_config": {"model_type": "llama4"}}',
        ],
    )
    def test_llama4_attention_patch_applies(self, tmp_path, monkeypatch, body):
        monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
        monkeypatch.setitem(
            sys.modules,
            "omlx.patches.mlx_lm_mtp",
            MagicMock(set_mtp_active=MagicMock()),
        )
        apply_mock = MagicMock(return_value=True)
        monkeypatch.setitem(
            sys.modules,
            "omlx.patches.llama4_attention",
            MagicMock(apply_llama4_attention_patch=apply_mock),
        )

        path = _write_config(tmp_path, body)
        maybe_apply_pre_load_patches(path)

        apply_mock.assert_called_once_with()


class TestLoadTextModel:
    def test_forwards_trust_remote_code_when_mlx_lm_supports_it(
        self, tmp_path, monkeypatch
    ):
        path = _write_config(tmp_path, '{"model_type": "llama"}')
        maybe_apply = MagicMock()
        monkeypatch.setattr(model_loading, "maybe_apply_pre_load_patches", maybe_apply)
        # Pin the capability flag so the test is deterministic regardless of the
        # installed mlx-lm version (lm_load_compat reads this global at call time).
        monkeypatch.setattr(model_loading, "_LM_LOAD_ACCEPTS_TRC", True)

        load_mock = MagicMock(return_value=("MODEL", "TOKENIZER"))
        monkeypatch.setitem(sys.modules, "mlx_lm", MagicMock(load=load_mock))

        settings = types.SimpleNamespace(trust_remote_code=True)
        result = model_loading.load_text_model(
            path,
            tokenizer_config={"trust_remote_code": True},
            model_settings=settings,
        )

        assert result == ("MODEL", "TOKENIZER")
        maybe_apply.assert_called_once_with(path, model_settings=settings)
        load_mock.assert_called_once_with(
            path,
            tokenizer_config={"trust_remote_code": True},
            trust_remote_code=True,
        )

    def test_omits_trust_remote_code_when_mlx_lm_lacks_it(self, tmp_path, monkeypatch):
        # Some mlx-lm releases dropped ``trust_remote_code`` from ``load``.
        # lm_load_compat must omit the kwarg there rather than raise TypeError.
        path = _write_config(tmp_path, '{"model_type": "llama"}')
        monkeypatch.setattr(
            model_loading, "maybe_apply_pre_load_patches", MagicMock()
        )
        monkeypatch.setattr(model_loading, "_LM_LOAD_ACCEPTS_TRC", False)

        load_mock = MagicMock(return_value=("MODEL", "TOKENIZER"))
        monkeypatch.setitem(sys.modules, "mlx_lm", MagicMock(load=load_mock))

        settings = types.SimpleNamespace(trust_remote_code=True)
        result = model_loading.load_text_model(
            path,
            tokenizer_config={"trust_remote_code": True},
            model_settings=settings,
        )

        assert result == ("MODEL", "TOKENIZER")
        load_mock.assert_called_once_with(
            path,
            tokenizer_config={"trust_remote_code": True},
        )


class TestVlmMtpPreLoadDispatch:
    """maybe_apply_pre_load_patches must wire the mlx-vlm MTP sanitize
    patch alongside the runtime patch for MTP-capable VLM checkpoints.

    The dense Qwen3.5/3.6 VLM runtime patch does not touch Model.sanitize;
    it relies on apply_mlx_vlm_mtp_patch having installed the mtp.*
    preservation first. If only the runtime patch runs, stock mlx-vlm
    sanitize strips every mtp.* key and the MTP head loads at random
    init (PR #1320).

    MoE VLMs without declared MTP heads still need the sanitize replacement
    so pre-converted switch_mlp weights load (issue #1261); runtime patch
    must not run on that path."""

    def _stub_patches(self, monkeypatch):
        """Replace the patch modules with mocks that record call order.

        Returns the recorded-order list plus the sanitize/runtime/attach
        mocks."""
        calls: list[str] = []
        sanitize_mock = MagicMock(side_effect=lambda: calls.append("sanitize") or True)
        runtime_mock = MagicMock(side_effect=lambda: calls.append("runtime") or True)
        attach_mock = MagicMock(
            side_effect=lambda enabled: calls.append(f"attach={enabled}")
        )
        # Side-step the real mlx-lm load_config monkey-patch.
        monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
        monkeypatch.setitem(
            sys.modules,
            "omlx.patches.mlx_lm_mtp",
            MagicMock(
                set_mtp_active=MagicMock(),
                apply_mlx_lm_mtp_patch=MagicMock(return_value=True),
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "omlx.patches.mlx_vlm_mtp",
            MagicMock(
                apply_mlx_vlm_mtp_patch=sanitize_mock,
                apply_mlx_vlm_mtp_runtime_patch=runtime_mock,
                set_mtp_attach_enabled=attach_mock,
            ),
        )
        return calls, sanitize_mock, runtime_mock, attach_mock

    def test_sanitize_patch_runs_before_runtime_for_vlm_mtp(
        self, tmp_path, monkeypatch
    ):
        calls, sanitize_mock, runtime_mock, attach_mock = self._stub_patches(
            monkeypatch
        )
        # qwen3_5 (dense VLM) declaring an MTP head under text_config.
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        _write_mtp_index(tmp_path, has_mtp=True)
        settings = types.SimpleNamespace(mtp_enabled=True)

        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        sanitize_mock.assert_called_once()
        runtime_mock.assert_called_once()
        attach_mock.assert_called_once_with(True)
        # Ordering matters: the dense runtime patch assumes sanitize was
        # already installed by apply_mlx_vlm_mtp_patch.
        assert calls == ["attach=True", "sanitize", "runtime"]

    def test_vlm_patches_applied_when_mtp_disabled_for_vlm(self, tmp_path, monkeypatch):
        # Issue #1404: persisted ``mtp.*`` weights must still get a binding
        # site on the LanguageModel tree when entering through VLMBatchedEngine
        # even with mtp_enabled=False. Otherwise mlx-vlm's strict load_weights
        # fails with "parameters not in model" and the engine falls back to
        # LLM, silently dropping vision.
        calls, sanitize_mock, runtime_mock, attach_mock = self._stub_patches(
            monkeypatch
        )
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        _write_mtp_index(tmp_path, has_mtp=True)
        settings = types.SimpleNamespace(mtp_enabled=False)

        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        sanitize_mock.assert_called_once()
        runtime_mock.assert_called_once()
        attach_mock.assert_called_once_with(True)
        assert calls == ["attach=True", "sanitize", "runtime"]

    def test_vlm_attach_disabled_when_config_declares_mtp_but_weights_missing(
        self, tmp_path, monkeypatch
    ):
        # Issue #1426: unsloth Qwen3.6 UD MLX builds declare
        # mtp_num_hidden_layers=1 in config.json but ship no mtp.* weights.
        # Attaching MTPModule there causes mlx-vlm strict load_weights to
        # fail with "Missing N parameters: language_model.mtp.*", the
        # engine falls back to LLM, and vision is silently dropped. The
        # dispatcher must flip set_mtp_attach_enabled(False) so the runtime
        # patch's __init__ wrap skips attachment for this load.
        calls, sanitize_mock, runtime_mock, attach_mock = self._stub_patches(
            monkeypatch
        )
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5_moe", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        _write_mtp_index(tmp_path, has_mtp=False)
        settings = types.SimpleNamespace(mtp_enabled=False)

        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        sanitize_mock.assert_called_once()
        # Runtime patch itself still applies (process-wide class wrap is
        # idempotent and harmless when there are no mtp.* weights to bind);
        # the gate is what prevents MTPModule attachment.
        runtime_mock.assert_called_once()
        attach_mock.assert_called_once_with(False)
        assert calls == ["attach=False", "sanitize", "runtime"]

    def test_vlm_patches_skipped_when_not_for_vlm(self, tmp_path, monkeypatch):
        # BatchedEngine / DFlashEngine / LLM loader paths must NOT touch
        # mlx-vlm classes even when the model declares MTP heads. for_vlm
        # defaults to False so they pass through without invoking mlx-vlm
        # patches.
        calls, sanitize_mock, runtime_mock, attach_mock = self._stub_patches(
            monkeypatch
        )
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=True)

        maybe_apply_pre_load_patches(path, model_settings=settings)

        sanitize_mock.assert_not_called()
        runtime_mock.assert_not_called()
        assert calls == []

    def test_qwen36_moe_vlm_sanitize_when_no_mtp_heads(self, tmp_path, monkeypatch):
        # mlx-lm Qwen3.6 MoE VLMs without MTP heads still need the mlx-vlm
        # sanitize replacement so pre-converted switch_mlp weights load.
        # Runtime MTP patch must NOT run — there is no mtp.* tree to bind.
        calls, sanitize_mock, runtime_mock, attach_mock = self._stub_patches(
            monkeypatch
        )
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5_moe", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 0}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=False)

        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        sanitize_mock.assert_called_once()
        runtime_mock.assert_not_called()
        assert calls == ["sanitize"]

    def test_qwen36_moe_vlm_sanitize_skipped_without_for_vlm(
        self, tmp_path, monkeypatch
    ):
        calls, sanitize_mock, runtime_mock, attach_mock = self._stub_patches(
            monkeypatch
        )
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5_moe", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 0}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=False)

        maybe_apply_pre_load_patches(path, model_settings=settings)

        sanitize_mock.assert_not_called()
        runtime_mock.assert_not_called()
        assert calls == []


class TestCheckpointHasMtpWeights:
    """``_checkpoint_has_mtp_weights`` decides whether the mlx-vlm runtime
    patch attaches ``MTPModule`` at load time. The scan must:

    - return True when ``model.safetensors.index.json`` declares any key
      under the ``(language_model.|model.)?mtp.`` prefix family;
    - return False when no MTP-prefixed key is found;
    - return False on missing / unreadable inputs (callers treat that as
      "no MTP weights" — the conservative choice).
    """

    def _write_index(self, tmp_path, weight_map: dict) -> None:
        import json as _json

        (tmp_path / "model.safetensors.index.json").write_text(
            _json.dumps({"metadata": {}, "weight_map": weight_map})
        )

    def test_returns_true_when_index_has_language_model_mtp(self, tmp_path):
        self._write_index(
            tmp_path,
            {
                "language_model.model.embed_tokens.weight": "model.safetensors",
                "language_model.mtp.fc.weight": "model.safetensors",
            },
        )
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is True


    def test_returns_true_when_index_has_bare_mtp(self, tmp_path):
        self._write_index(
            tmp_path,
            {"mtp.layers.0.self_attn.q_proj.weight": "model.safetensors"},
        )
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is True

    def test_returns_true_when_index_has_model_language_model_mtp(self, tmp_path):
        # mlx-vlm HF-source layout before sanitize-time remap (oQ writes this).
        self._write_index(
            tmp_path,
            {"model.language_model.mtp.norm.weight": "model.safetensors"},
        )
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is True

    def test_returns_false_when_index_lacks_mtp(self, tmp_path):
        # Unsloth Qwen3.6 UD MLX layout: vision_tower + language_model.model.*
        # but no language_model.mtp.* keys despite mtp_num_hidden_layers > 0
        # in config.json (issue #1426).
        self._write_index(
            tmp_path,
            {
                "language_model.model.embed_tokens.weight": "model.safetensors",
                "vision_tower.blocks.0.attn.proj.weight": "model.safetensors",
            },
        )
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is False

    def test_returns_true_for_nextn_layout(self, tmp_path):
        # DeepSeek-V3-style checkpoints (GLM-5.2) keep the MTP head as an
        # extra decoder layer past num_hidden_layers, not under mtp.*.
        import json as _json

        (tmp_path / "config.json").write_text(
            _json.dumps({"num_hidden_layers": 78, "num_nextn_predict_layers": 1})
        )
        self._write_index(
            tmp_path,
            {
                "model.layers.0.self_attn.q_a_proj.weight": "model.safetensors",
                "model.layers.78.eh_proj.weight": "model.safetensors",
            },
        )
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is True

    def test_returns_false_for_nextn_config_without_weights(self, tmp_path):
        # Config declares nextn layers but the checkpoint stripped them.
        import json as _json

        (tmp_path / "config.json").write_text(
            _json.dumps({"num_hidden_layers": 78, "num_nextn_predict_layers": 1})
        )
        self._write_index(
            tmp_path,
            {"model.layers.0.self_attn.q_a_proj.weight": "model.safetensors"},
        )
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is False

    def test_returns_false_for_empty_dir(self, tmp_path):
        # No index, no shards — caller treats as "no MTP weights".
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is False

    def test_returns_false_for_nonexistent_path(self, tmp_path):
        assert (
            model_loading._checkpoint_has_mtp_weights(str(tmp_path / "does-not-exist"))
            is False
        )

    def test_returns_false_on_malformed_index(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text("{not valid")
        # Falls through to safetensors-header scan; no shards exist, so
        # the helper conservatively returns False.
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is False

    def test_returns_true_when_later_shard_has_mtp(self, tmp_path):
        from safetensors.numpy import save_file
        import numpy as np

        save_file(
            {
                "language_model.model.embed_tokens.weight": np.zeros(
                    (1,), dtype=np.float16
                )
            },
            str(tmp_path / "model-00001-of-00002.safetensors"),
        )
        save_file(
            {"language_model.mtp.fc.weight": np.zeros((1,), dtype=np.float16)},
            str(tmp_path / "model-00002-of-00002.safetensors"),
        )

        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is True


class TestExpandPerLayerQuantKeys:
    """expand_per_layer_quant_keys adds runtime module-tree key variants."""

    def test_adds_language_model_prefix_for_bare_key(self):
        cfg = {
            "quantization": {
                "bits": 6,
                "group_size": 64,
                "lm_head": {"bits": 8, "group_size": 64},
            }
        }

        model_loading.expand_per_layer_quant_keys(cfg)

        assert cfg["quantization"]["language_model.lm_head"] == {
            "bits": 8,
            "group_size": 64,
        }

    def test_adds_swapped_prefix_variant_for_model_language_model_key(self):
        key = "model.language_model.layers.0.linear_attn.in_proj_qkv"
        cfg = {
            "quantization": {
                "bits": 6,
                "group_size": 64,
                key: {"bits": 8, "group_size": 64},
            }
        }

        model_loading.expand_per_layer_quant_keys(cfg)

        swapped = "language_model.model.layers.0.linear_attn.in_proj_qkv"
        assert swapped in cfg["quantization"]
        assert cfg["quantization"][swapped]["bits"] == 8
