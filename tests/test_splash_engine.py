# SPDX-License-Identifier: Apache-2.0
"""Tests for the Splash external engine and its pool, settings and API wiring."""

import asyncio
import contextlib
import copy
import fcntl
import json
import os
import re
import signal
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from tokenizers import pre_tokenizers
from transformers import AutoTokenizer

import omlx.server as srv
from omlx.admin import routes as admin_routes
from omlx.admin.benchmark import (
    _CHAT_CALIBRATION_TOKENS,
    BenchmarkRequest,
    BenchmarkRun,
    _ChatBenchmark,
    run_benchmark,
)
from omlx.admin.external_api import StreamStats
from omlx.api.openai_models import (
    ChatCompletionRequest,
    ResponseFormat,
    ResponseFormatJsonSchema,
)
from omlx.api.responses_utils import ResponseStore
from omlx.engine import splash as splash_mod
from omlx.engine.base import GenerationOutput
from omlx.engine.splash import (
    SplashBinary,
    SplashEngine,
    SplashOptions,
    SplashTarget,
    find_splash_binary,
    resolve_splash_target,
    splash_effort,
    splash_memory_estimate,
    splash_messages,
    splash_repo_id,
    stage_model,
)
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.exceptions import InvalidRequestError, ModelLoadingError
from omlx.model_discovery import discover_models
from omlx.model_settings import ModelSettings
from omlx.settings import GlobalSettings
from omlx.utils.proc_memory import list_process_group_pids

QWEN38_TEXT = dict(splash_mod.SPLASH_FAMILIES["Qwen3.8-27B"])
FLASH_NEXT_TEXT = {
    "model_type": "qwen4_exp_text",
    "max_position_embeddings": 262144,
    "hidden_size": 2560,
    "num_hidden_layers": 48,
    "vocab_size": 248320,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "num_experts": 512,
    "num_experts_per_tok": 10,
}
REPO_ID = splash_repo_id("Qwen3.8-27B-4bit")
MLX_4BIT = {"mode": "affine", "bits": 4, "group_size": 64}
GIB = 1024**3


def _write_model(path: Path, text_config=None, quantization=None) -> Path:
    path.mkdir(parents=True)
    config = {
        "model_type": "qwen3_5",
        "text_config": QWEN38_TEXT if text_config is None else text_config,
        "quantization": MLX_4BIT if quantization is None else quantization,
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer.json").write_text("{}")
    (path / "tokenizer_config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"0" * 1024)
    return path


def _write_draft(path: Path, text=None) -> Path:
    text = QWEN38_TEXT if text is None else text
    path.mkdir(parents=True)
    config = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_size": text["hidden_size"],
        "vocab_size": text["vocab_size"],
        "num_target_layers": text["num_hidden_layers"],
        "dflash_config": {"block_size": 8},
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors").write_bytes(b"0" * 64)
    return path


def _add_optiq_vision(model: Path) -> dict:
    """OptiQ's layout: the vision tower in optiq/, listed by the index, and no
    processor file. Returns the index's weight map."""
    config = json.loads((model / "config.json").read_text())
    vision = {"patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2}
    (model / "config.json").write_text(json.dumps({**config, "vision_config": vision}))
    (model / "optiq").mkdir()
    (model / "optiq" / "optiq_vision.safetensors").write_bytes(b"0" * 512)
    weight_map = {
        "lm_head.weight": "model.safetensors",
        "vision_tower.blocks.0.attn.qkv.weight": "optiq/optiq_vision.safetensors",
    }
    index = model / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}))
    return weight_map


def _fake_launcher(path: Path, version: str = "1.3.1", body: str = "exit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = --version ]; then echo "Splash {version}"; exit 0; fi\n' + body
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def launchers(tmp_path, monkeypatch):
    """Install fake launchers by name on an M3 Mac; nothing else is found."""
    installed: dict[str, Path] = {}
    monkeypatch.setattr(splash_mod, "_find_launcher", installed.get)
    monkeypatch.setattr(splash_mod, "_is_m1_or_m2", lambda: False)

    def install(name="splash", version="1.3.1", body="exit 0\n"):
        installed[name] = _fake_launcher(tmp_path / "bin" / name, version, body)
        return installed[name]

    return install


def _binary(version="1.3.1", path="/opt/homebrew/bin/splash") -> SplashBinary:
    return SplashBinary(Path(path), version, auto_detected=True)


def _engine(tmp_path, version="1.3.1", **options) -> SplashEngine:
    model = tmp_path / "models" / "Qwen3.8-27B-4bit"
    target = SplashTarget(_binary(version), "Qwen3.8-27B", tmp_path / "draft")
    return SplashEngine(
        model_name=str(model),
        model_id="Qwen3.8-27B-4bit",
        target=target,
        hub_dir=tmp_path / "hub",
        log_dir=tmp_path / "logs",
        options=SplashOptions(**options),
    )


def _disk_cache(root):
    folder = root / ("a" * 32)
    folder.mkdir(parents=True)
    files = [
        folder / name
        for name in ("kv.slots", "kv.records", "state.slots", "state.records")
    ]
    for path in files:
        path.write_bytes(b"cached")
    (folder / "lock").touch()
    return files


class TestDiskCache:
    def test_clear_removes_cache_data_and_keeps_lock_and_other_files(self, tmp_path):
        root = tmp_path / "splash"
        files = _disk_cache(root / "model")
        folder = files[0].parent
        (folder / "notes.txt").write_text("keep")
        lock_inode = (folder / "lock").stat().st_ino
        sparse = files[0]
        with sparse.open("wb") as stream:
            stream.truncate(1024**3)
        expected = sum(path.stat().st_blocks * 512 for path in files)

        assert splash_mod.scan_ssd_cache(root) == (4, expected)
        assert splash_mod.scan_ssd_cache(root, clear=True) == (4, expected)
        assert not any(path.exists() for path in files)
        assert (folder / "lock").stat().st_ino == lock_inode
        assert (folder / "notes.txt").read_text() == "keep"
        assert splash_mod.scan_ssd_cache(root) == (0, 0)

    def test_clear_refuses_a_locked_namespace(self, tmp_path):
        root = tmp_path / "splash"
        files = _disk_cache(root / "model")
        with (files[0].parent / "lock").open("r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(RuntimeError, match="in use by another process"):
                splash_mod.scan_ssd_cache(root, clear=True)
        assert all(path.exists() for path in files)

    @pytest.mark.parametrize("level", ["root", "model", "namespace", "file"])
    def test_scan_and_clear_preserve_symlink_targets(self, tmp_path, level):
        root = tmp_path / "splash"
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "kv.slots"
        target.write_text("keep")
        link = root
        if level != "root":
            root.mkdir()
            link = root / "model"
        if level in ("namespace", "file"):
            link.mkdir()
            link /= "a" * 32
        if level == "file":
            link.mkdir()
            link /= "kv.slots"
        link.symlink_to(target if level == "file" else outside)

        if level == "root":
            with pytest.raises(OSError):
                splash_mod.scan_ssd_cache(root, clear=True)
        else:
            assert splash_mod.scan_ssd_cache(root) == (0, 0)
            assert splash_mod.scan_ssd_cache(root, clear=True) == (0, 0)
        assert target.read_text() == "keep"

    @pytest.mark.parametrize("custom", [False, True])
    async def test_api_clears_unloaded_splash_cache_and_usage(
        self, tmp_path, monkeypatch, custom
    ):
        settings = GlobalSettings(base_path=tmp_path)
        if custom:
            settings.cache.ssd_cache_dir = str(tmp_path / "custom-cache")
        root = settings.cache.get_ssd_cache_dir(tmp_path)
        files = _disk_cache(root / "splash" / "model")
        pool = EnginePool()
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: settings)
        monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
        monkeypatch.setattr(
            admin_routes, "_clear_cold_remote_cluster_cache_roots", lambda _: (0, 0)
        )

        before = admin_routes._build_runtime_cache_observability(settings)
        assert before["total_num_files"] == 4
        assert before["total_size_bytes"] > 0
        with patch.dict(
            srv.app.dependency_overrides, {admin_routes.require_admin: lambda: True}
        ):
            response = TestClient(srv.app).post("/admin/api/ssd-cache/clear")
        assert response.status_code == 200
        assert response.json()["total_deleted"] == 4
        assert not any(path.exists() for path in files)
        after = admin_routes._build_runtime_cache_observability(settings)
        assert after["total_num_files"] == after["total_size_bytes"] == 0

    def test_usage_counts_cold_cache_without_counting_a_loaded_cache_twice(
        self, tmp_path, monkeypatch
    ):
        settings = GlobalSettings(base_path=tmp_path)
        root = tmp_path / "cache" / "splash"
        _disk_cache(root / "loaded")
        cold = _disk_cache(root / "unloaded")
        pool, entry = _pool(tmp_path)
        engine = _engine(tmp_path, cache_dir=root / "loaded", cache_disk_bytes=GIB)
        entry.engine = engine
        engine._status = {"disk": {"used_bytes": 123, "capacity_bytes": GIB}}
        monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
        stats = admin_routes._build_runtime_cache_observability(settings)
        assert stats["total_num_files"] == 5
        assert stats["total_size_bytes"] == 123 + sum(
            path.stat().st_blocks * 512 for path in cold
        )

    async def test_clear_also_reaches_a_loaded_models_previous_cache_root(
        self, tmp_path
    ):
        root = tmp_path / "cache" / "splash"
        old_root = tmp_path / "old-cache" / "splash"
        files = _disk_cache(root / "unloaded") + _disk_cache(old_root / "loaded")
        pool, entry = _pool(tmp_path)
        engine = _engine(tmp_path, cache_dir=old_root / "loaded", cache_disk_bytes=GIB)
        entry.engine = engine
        engine.stop = AsyncMock()
        assert await pool.clear_splash_ssd_cache(root) == 8
        assert not any(path.exists() for path in files)

    async def test_api_unloads_idle_splash_before_deleting_its_cache(
        self, tmp_path, monkeypatch
    ):
        settings = GlobalSettings(base_path=tmp_path)
        root = tmp_path / "cache" / "splash"
        files = _disk_cache(root / "model")
        pool, entry = _pool(tmp_path)
        engine = _engine(tmp_path, cache_dir=root / "model", cache_disk_bytes=GIB)
        entry.engine = engine
        entry.is_pinned = True

        async def stop():
            assert all(path.exists() for path in files)
            assert (
                pool._acquire_loaded_engine(entry.model_id, False, True, None) is None
            )

        engine.stop = AsyncMock(side_effect=stop)
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: settings)
        monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
        monkeypatch.setattr(
            admin_routes, "_clear_cold_remote_cluster_cache_roots", lambda _: (0, 0)
        )
        response = await admin_routes.clear_ssd_cache(is_admin=True)

        assert response["total_deleted"] == 4
        engine.stop.assert_awaited_once()
        assert entry.engine is None
        assert entry.is_pinned
        assert not any(path.exists() for path in files)

    @pytest.mark.parametrize("busy", ["lease", "request", "loading", "pending"])
    async def test_api_refuses_to_clear_busy_splash(self, tmp_path, monkeypatch, busy):
        settings = GlobalSettings(base_path=tmp_path)
        root = tmp_path / "cache" / "splash"
        files = _disk_cache(root / "model")
        pool, entry = _pool(tmp_path)
        engine = _engine(tmp_path, cache_dir=root / "model", cache_disk_bytes=GIB)
        entry.engine = engine
        engine.stop = AsyncMock()
        entry.in_use = int(busy == "lease")
        engine._active = int(busy == "request")
        entry.is_loading = busy == "loading"
        entry.pending_unload_reason = "manual unload" if busy == "pending" else None
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: settings)
        monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
        with pytest.raises(HTTPException) as error:
            await admin_routes.clear_ssd_cache(is_admin=True)
        assert error.value.status_code == 409
        engine.stop.assert_not_awaited()
        assert all(path.exists() for path in files)


# -- binary ---------------------------------------------------------------------


class TestBinary:
    @pytest.mark.parametrize(
        "m1_or_m2, expected", [(False, "splash"), (True, "splash-m1")]
    )
    def test_automatic_choice_follows_the_chip(
        self, launchers, monkeypatch, m1_or_m2, expected
    ):
        launchers("splash")
        launchers("splash-m1", version="1.3.0-m1.1")
        monkeypatch.setattr(splash_mod, "_is_m1_or_m2", lambda: m1_or_m2)

        binary, reason = find_splash_binary(None)

        assert reason == ""
        assert binary.path.name == expected
        assert binary.auto_detected is True

    def test_falls_back_to_the_other_build(self, launchers):
        launchers("splash-m1", version="1.3.0-m1.1")

        binary, _ = find_splash_binary(None)

        assert binary.path.name == "splash-m1"
        assert binary.label == "Splash M1"
        assert binary.loads_mixed_quantization is False

    def test_configured_path_wins_over_detection(self, launchers, tmp_path):
        launchers("splash")
        custom = _fake_launcher(tmp_path / "src" / "splash", version="1.3.1")

        binary, _ = find_splash_binary(str(custom))

        assert binary.path == custom
        assert binary.auto_detected is False

    def test_failed_version_read_is_retried(self, launchers, monkeypatch):
        launcher = launchers("splash")
        answers = iter([None, "1.3.1"])
        calls = []

        def read(path):
            calls.append(path)
            return next(answers)

        monkeypatch.setattr(splash_mod, "_read_version", read)
        monkeypatch.setattr(splash_mod, "VERSION_RETRY_SECONDS", 0)
        splash_mod._versions.clear()

        assert find_splash_binary(None)[0] is None
        assert find_splash_binary(None)[0].path == launcher
        assert find_splash_binary(None)[0].path == launcher
        assert len(calls) == 2

    def test_unusable_binaries_explain_why(self, launchers, tmp_path):
        assert find_splash_binary(None) == (None, splash_mod.SPLASH_INSTALL_HINT)

        launchers("splash", version="1.2.1")
        binary, reason = find_splash_binary(None)
        assert binary is None and "too old" in reason

        binary, reason = find_splash_binary(str(tmp_path / "missing"))
        assert binary is None and "not an executable" in reason


# -- compatibility ----------------------------------------------------------------


class TestCompatibility:

    def test_flash_next_mlx_checkpoint_explains_gguf_requirement(
        self, tmp_path, launchers
    ):
        launchers("splash", version="1.4.0")
        model = _write_model(tmp_path / "Flash-Next-MLX", FLASH_NEXT_TEXT)

        target, reason = resolve_splash_target(model, None, [])

        assert target is None
        assert "Qwen3.8-Flash-Next needs a GGUF" in reason
        assert splash_mod.splash_draft_to_download(model, None, []) is None

    def test_supported_checkpoint_resolves_with_its_draft(self, tmp_path, launchers):
        launchers("splash")
        model = _write_model(tmp_path / "Qwen3.8-27B-4bit")
        other = _write_draft(
            tmp_path / "other-draft", text={**QWEN38_TEXT, "hidden_size": 1}
        )
        draft = _write_draft(tmp_path / "z-lab" / "Qwen3.8-27B-DFlash2")

        target, reason = resolve_splash_target(model, None, [str(other), str(draft)])

        assert reason == ""
        assert target.family == "Qwen3.8-27B"
        assert target.draft_path == draft

    @pytest.mark.parametrize(
        "text_config, quantization, match",
        [
            ({**QWEN38_TEXT, "hidden_size": 4096}, None, "Splash serves"),
            (None, {"bits": 8, "group_size": 64, "lm_head": {"bits": 8}}, "4-bit"),
        ],
    )
    def test_unsupported_checkpoints_explain_why(
        self, tmp_path, launchers, text_config, quantization, match
    ):
        launchers("splash", version="1.3.0")
        model = _write_model(tmp_path / "m", text_config, quantization)
        draft = _write_draft(tmp_path / "draft")

        target, reason = resolve_splash_target(model, None, [str(draft)])

        assert target is None and match in reason

    def test_mixed_bits_need_splash_1_3_1(self, tmp_path, launchers):
        quant = {**MLX_4BIT, "language_model.model.layers.0.mlp.down_proj": {"bits": 5}}
        model = _write_model(tmp_path / "Qwen3.8-27B-oQ4e", quantization=quant)
        draft = _write_draft(tmp_path / "draft")

        launchers("splash", version="1.3.0")
        assert resolve_splash_target(model, None, [str(draft)])[0] is None
        launchers("splash", version="1.3.1")
        assert resolve_splash_target(model, None, [str(draft)])[0] is not None

    @pytest.mark.parametrize(
        "quantization_config",
        [
            {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "a": {"quant_algo": "NVFP4"},
                    "b": {"quant_algo": "FP8"},
                },
            },
            {
                "quant_method": "compressed-tensors",
                "format": "mixed-precision",
                "config_groups": {
                    "group_0": {"format": "float-quantized"},
                    "group_1": {"format": "nvfp4-pack-quantized"},
                },
            },
        ],
    )
    def test_nvfp4_and_fp8_checkpoints_need_splash_1_3_2(
        self, tmp_path, launchers, quantization_config
    ):
        model = _write_model(tmp_path / "Qwen3.8-27B-NVFP4")
        config = json.loads((model / "config.json").read_text())
        del config["quantization"]
        config["quantization_config"] = quantization_config
        (model / "config.json").write_text(json.dumps(config))
        draft = _write_draft(tmp_path / "draft")

        launchers("splash", version="1.3.1")
        _, reason = resolve_splash_target(model, None, [str(draft)])
        assert "Splash 1.3.2 also serves NVFP4 and FP8" in reason
        launchers("splash", version="1.3.2")
        target, reason = resolve_splash_target(model, None, [str(draft)])
        assert target is not None and reason == ""

        # Other schemes of the same tools are not formats Splash reads.
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "config_groups": {"g": {"format": "pack-quantized"}},
        }
        (model / "config.json").write_text(json.dumps(config))
        _, reason = resolve_splash_target(model, None, [str(draft)])
        assert "MLX, NVFP4 or FP8" in reason

    def test_missing_files_and_draft_explain_why(self, tmp_path, launchers):
        launchers("splash")
        model = _write_model(tmp_path / "m")

        _, reason = resolve_splash_target(model, None, [])
        assert "DFlash2 draft" in reason and "Qwen3.8-27B-DFlash2" in reason

        (model / "tokenizer.json").unlink()
        _, reason = resolve_splash_target(model, None, [])
        assert "tokenizer.json" in reason

    def test_qwen35_122b_checkpoint_needs_splash_1_4_1_and_its_draft(
        self, tmp_path, launchers
    ):
        text = dict(splash_mod.SPLASH_FAMILIES["Qwen3.5-122B-A10B"])
        model = _write_model(tmp_path / "Ornith-1.5-122B-MLX-4bit", text)
        draft = _write_draft(tmp_path / "incoai" / "Ornith-1.5-122B-DFlash2", text)

        launchers("splash", version="1.4.0")
        _, reason = resolve_splash_target(model, None, [str(draft)])
        assert reason == "Qwen3.5-122B-A10B needs Splash 1.4.1 or later"

        launchers("splash", version="1.4.1")
        assert (
            splash_mod.splash_draft_to_download(model, None, [])
            == "incoai/Ornith-1.5-122B-DFlash2"
        )
        target, reason = resolve_splash_target(model, None, [str(draft)])
        assert reason == ""
        assert target.draft_path == draft

    def test_flint_mlx_checkpoint_explains_gguf_requirement(self, tmp_path, launchers):
        launchers("splash", version="1.4.1")
        # Flint's own config.json is flat: no text_config.
        model = _write_model(tmp_path / "flint-4bit")
        config = {**splash_mod.SPLASH_FAMILIES["Flint"], "quantization": MLX_4BIT}
        (model / "config.json").write_text(json.dumps(config))

        _, reason = resolve_splash_target(model, None, [])

        assert reason == "Flint needs a GGUF; Splash cannot load its MLX checkpoints"

    def test_pruned_mlx_checkpoint_explains_the_expert_count(self, tmp_path, launchers):
        launchers("splash", version="1.4.1")
        text = {**splash_mod.SPLASH_FAMILIES["Qwen3.6-35B-A3B"], "num_experts": 128}
        model = _write_model(tmp_path / "Qwen3.6-35B-A3B-REAP-4bit", text)

        _, reason = resolve_splash_target(model, None, [])

        assert reason == (
            "Splash serves Qwen3.6-35B-A3B MLX checkpoints with all 256 routed experts"
        )


# -- staging --------------------------------------------------------------------------

# Splash's own check of a --model value (server/serve_options.py).
SPLASH_REPO_ID = re.compile(
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?/"
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?"
)


def _qwen38_gguf(vocab: list[str]) -> dict:
    return {
        "general.architecture": "qwen35",
        "qwen35.block_count": 64,
        "qwen35.context_length": 262144,
        "qwen35.embedding_length": 5120,
        "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.full_attention_interval": 4,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": vocab,
        "tokenizer.ggml.token_type": [3 if t.startswith("<|") else 1 for t in vocab],
        "tokenizer.ggml.merges": [],
        "tokenizer.ggml.eos_token_id": vocab.index("<|im_end|>"),
        "tokenizer.chat_template": (
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
            "<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        ),
    }


class TestGguf:

    @pytest.mark.parametrize("layout", ["single", "split", "variant"])
    def test_flash_next_gguf_discovery_draft_and_engine(
        self, tmp_path, monkeypatch, launchers, write_gguf, global_settings, layout
    ):
        vocab = sorted(pre_tokenizers.ByteLevel.alphabet()) + [
            "<|im_start|>",
            "<|im_end|>",
        ]
        repo = tmp_path / "models" / "ISTA-DASLab" / "Flash-Next-GGUF"
        metadata = {
            key: value
            for key, value in _qwen38_gguf(vocab).items()
            if key.startswith("tokenizer.")
        }
        metadata.update(
            {
                "general.architecture": "qwen4exp",
                "qwen4exp.block_count": 49,
                "qwen4exp.nextn_predict_layers": 1,
                "qwen4exp.context_length": 262144,
                "qwen4exp.embedding_length": 2560,
                "qwen4exp.attention.head_count": 24,
                "qwen4exp.attention.head_count_kv": 2,
                "qwen4exp.attention.key_length": 256,
                "qwen4exp.full_attention_interval": 4,
                "qwen4exp.expert_count": 512,
                "qwen4exp.expert_used_count": 10,
            }
        )
        name = "Flash-Next-Q2_0"
        source_folder = repo / "Q2_0" if layout == "variant" else repo
        first = f"{name}.gguf" if layout == "single" else f"{name}-00001-of-00002.gguf"
        write_gguf(source_folder / first, metadata)
        if layout != "single":
            write_gguf(source_folder / f"{name}-00002-of-00002.gguf", {})
        models = discover_models(tmp_path / "models", tmp_path / "gguf")
        model = models[name]
        text = {**FLASH_NEXT_TEXT, "vocab_size": len(vocab)}
        folder = Path(model.model_path)
        config = json.loads((folder / "config.json").read_text())
        assert config["model_type"] == "qwen4_exp"
        assert config["text_config"] == {**text, "full_attention_interval": 4}
        assert model.model_format == "gguf"
        family = splash_mod.SPLASH_FAMILIES["Qwen3.8-Flash-Next"]
        monkeypatch.setitem(family, "vocab_size", len(vocab))

        launchers("splash", version="1.3.2")
        target, reason = resolve_splash_target(folder, None, [])
        assert target is None
        assert "Splash 1.4.0 or later" in reason
        assert splash_mod.splash_draft_to_download(folder, None, []) is None

        launchers("splash", version="1.4.0")
        wrong_draft = _write_draft(tmp_path / "27B-draft")
        assert (
            splash_mod.splash_draft_to_download(folder, None, [str(wrong_draft)])
            == "incoai/Qwen3.8-Flash-Next-DFlash2"
        )
        _, reason = resolve_splash_target(folder, None, [str(wrong_draft)])
        assert "incoai/Qwen3.8-Flash-Next-DFlash2" in reason
        draft = _write_draft(
            tmp_path / "models" / "incoai" / "Qwen3.8-Flash-Next-DFlash2", text
        )
        target, reason = resolve_splash_target(
            folder, None, [str(wrong_draft), str(draft)]
        )
        assert reason == ""
        assert target.family == "Qwen3.8-Flash-Next"
        assert target.draft_path == draft

        tokenizer = AutoTokenizer.from_pretrained(folder)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        assert tokenizer.decode(tokenizer.encode(prompt)) == prompt
        staged = stage_model(folder, "omlx/Flash-Next", tmp_path / "hub")
        for part in config["omlx_gguf"]["parts"]:
            assert (staged / part).resolve() == (
                source_folder / Path(part).name
            ).resolve()

        pool = EnginePool()
        pool.discover_models([str(tmp_path / "models")])
        assert pool._entries[draft.name].is_helper
        engine = pool._external_engine_for(
            pool._entries[name], ModelSettings(), model.engine_type
        )
        assert isinstance(engine, SplashEngine)
        assert engine._target.draft_path == draft
        command = engine._serve_command(9000)
        model_arg = command[command.index("--model") + 1]
        assert model_arg == engine._repo_id + (":split" if layout != "single" else "")
        if layout != "single":
            assert command[command.index("--served-model-name") + 1] == engine._repo_id
        assert "--offline" in command
        assert "--language-only" in command
        assert command[command.index("--draft-model") + 1] == str(draft.resolve())
        assert engine._options.max_memory_bytes == splash_memory_estimate(
            folder, draft, 262144, "int8"
        )

    def test_qwen_gguf_serves_on_splash_with_a_derived_tokenizer(
        self, tmp_path, monkeypatch, launchers, write_gguf
    ):
        # A small vocabulary keeps the derived tokenizer cheap to build.
        family = splash_mod.SPLASH_FAMILIES["Qwen3.8-27B"]
        monkeypatch.setitem(family, "vocab_size", 300)
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        specials = ["<|im_start|>", "<|im_end|>"]
        vocab = alphabet + specials + [f"t{i}" for i in range(300 - 258)]
        launchers("splash", version="1.3.2")
        repo = tmp_path / "models" / "unsloth" / "Qwen3.8-27B-GGUF"
        metadata = _qwen38_gguf(vocab)
        write_gguf(repo / "Qwen3.8-27B-UD-Q4_K_M.gguf", metadata)
        write_gguf(repo / "Qwen3.8-27B-BF16-00001-of-00002.gguf", metadata)
        write_gguf(repo / "Qwen3.8-27B-BF16-00002-of-00002.gguf", {})
        projector = {"general.architecture": "clip"}
        write_gguf(repo / "mmproj-BF16.gguf", projector, size=512)
        draft = _write_draft(tmp_path / "draft", {**QWEN38_TEXT, "vocab_size": 300})

        models = discover_models(tmp_path / "models", tmp_path / "gguf")
        folder = models["Qwen3.8-27B-UD-Q4_K_M"].model_path
        target, reason = resolve_splash_target(folder, None, [str(draft)])

        assert reason == ""
        assert target.family == "Qwen3.8-27B"
        gguf_bytes = (repo / "Qwen3.8-27B-UD-Q4_K_M.gguf").stat().st_size
        assert splash_mod._weight_bytes(Path(folder)) == gguf_bytes + 512
        split = models["Qwen3.8-27B-BF16"].model_path
        assert resolve_splash_target(split, None, [str(draft)]) == (
            None,
            "Split GGUFs need Splash 1.4.0 or later",
        )

        # The tokenizer comes from the GGUF header, chat template included.
        tokenizer = AutoTokenizer.from_pretrained(folder)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        assert prompt == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        assert tokenizer.decode(tokenizer.encode(prompt)) == prompt
        assert tokenizer.encode("<|im_end|>") == [vocab.index("<|im_end|>")]
        assert tokenizer.eos_token == "<|im_end|>"

    def test_flint_gguf_serves_text_only_on_splash_1_4_1(
        self, tmp_path, monkeypatch, launchers, write_gguf
    ):
        vocab = sorted(pre_tokenizers.ByteLevel.alphabet()) + ["<|end_of_text|>"]
        monkeypatch.setitem(
            splash_mod.SPLASH_FAMILIES["Flint"], "vocab_size", len(vocab)
        )
        # Layers 0, 1, 48 and every fourth from 5 attend every key; the rest
        # slide over 513 tokens.
        sliding = [2 <= layer < 48 and layer % 4 != 1 for layer in range(49)]
        metadata = {
            "general.architecture": "flint",
            "flint.block_count": 49,
            "flint.context_length": 262144,
            "flint.embedding_length": 2048,
            "flint.attention.head_count": 32,
            "flint.attention.head_count_kv": 4,
            "flint.attention.key_length": 128,
            "flint.expert_count": 256,
            "flint.expert_used_count": 6,
            "flint.attention.sliding_window": 513,
            "flint.attention.sliding_window_pattern": sliding,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "flint",
            "tokenizer.ggml.tokens": vocab,
            "tokenizer.ggml.token_type": [1] * (len(vocab) - 1) + [3],
            "tokenizer.ggml.merges": [],
        }
        repo = tmp_path / "models" / "unsloth" / "flint-GGUF"
        write_gguf(repo / "flint-UD-Q4_K_XL.gguf", metadata, size=4096)
        # A projector beside it does not give Flint a vision tower.
        write_gguf(repo / "mmproj-BF16.gguf", {"general.architecture": "clip"}, 512)
        text = {**splash_mod.SPLASH_FAMILIES["Flint"], "vocab_size": len(vocab)}
        draft = _write_draft(tmp_path / "models" / "incoai" / "Flint-DFlash2", text)
        models = discover_models(tmp_path / "models", tmp_path / "gguf")
        folder = Path(models["flint-UD-Q4_K_XL"].model_path)

        launchers("splash", version="1.4.0")
        _, reason = resolve_splash_target(folder, None, [str(draft)])
        assert reason == "Flint needs Splash 1.4.1 or later"
        launchers("splash", version="1.4.1")
        target, reason = resolve_splash_target(folder, None, [str(draft)])
        assert reason == ""

        engine = SplashEngine(
            model_name=str(folder),
            model_id="flint-UD-Q4_K_XL",
            target=target,
            hub_dir=tmp_path / "hub",
            log_dir=tmp_path / "logs",
            options=SplashOptions(),
        )
        assert "--language-only" in engine._serve_command(9000)
        gguf_bytes = (repo / "flint-UD-Q4_K_XL.gguf").stat().st_size
        assert splash_mod._weight_bytes(folder) == gguf_bytes
        # Only the 14 full-attention layers' KV grows with the context.
        text_config = json.loads((folder / "config.json").read_text())["text_config"]
        kv = splash_mod._kv_bytes_per_token(text_config, "bf16")
        assert kv == 14 * 2 * 4 * 128 * 2

        # Flint's own tokenizer splits digits in runs of up to three and does
        # not NFC-normalize text.
        backend = AutoTokenizer.from_pretrained(folder).backend_tokenizer
        assert backend.normalizer is None
        pieces = backend.pre_tokenizer.pre_tokenize_str("1234567")
        assert [piece for piece, _ in pieces] == ["123", "456", "7"]

    def test_pruned_moe_gguf_serves_on_splash_1_4_1(
        self, tmp_path, monkeypatch, launchers, write_gguf
    ):
        vocab = sorted(pre_tokenizers.ByteLevel.alphabet()) + ["<|im_end|>"]
        family = splash_mod.SPLASH_FAMILIES["Qwen3.8-Flash-Next"]
        monkeypatch.setitem(family, "vocab_size", len(vocab))

        def flash_next(experts: int) -> dict:
            return {
                "general.architecture": "qwen4exp",
                "qwen4exp.block_count": 48,
                "qwen4exp.context_length": 262144,
                "qwen4exp.embedding_length": 2560,
                "qwen4exp.attention.head_count": 24,
                "qwen4exp.attention.head_count_kv": 2,
                "qwen4exp.attention.key_length": 256,
                "qwen4exp.full_attention_interval": 4,
                "qwen4exp.expert_count": experts,
                "qwen4exp.expert_used_count": 10,
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.ggml.pre": "qwen35",
                "tokenizer.ggml.tokens": vocab,
                "tokenizer.ggml.token_type": [1] * (len(vocab) - 1) + [3],
                "tokenizer.ggml.merges": [],
            }

        # ISTA's Coder release keeps 256 of Flash-Next's 512 routed experts.
        repo = tmp_path / "models" / "ISTA-DASLab" / "Flash-Next-Coder-GGUF"
        write_gguf(repo / "Flash-Next-Coder-IQ1_M.gguf", flash_next(256))
        write_gguf(repo / "Flash-Next-Odd-Q2_0.gguf", flash_next(260))
        draft = _write_draft(
            tmp_path / "draft", {**FLASH_NEXT_TEXT, "vocab_size": len(vocab)}
        )
        models = discover_models(tmp_path / "models", tmp_path / "gguf")
        pruned = models["Flash-Next-Coder-IQ1_M"].model_path
        odd = models["Flash-Next-Odd-Q2_0"].model_path

        launchers("splash", version="1.4.0")
        assert resolve_splash_target(pruned, None, [str(draft)]) == (
            None,
            "GGUFs pruned to fewer routed experts need Splash 1.4.1 or later",
        )
        launchers("splash", version="1.4.1")
        target, reason = resolve_splash_target(pruned, None, [str(draft)])
        assert reason == ""
        assert target.family == "Qwen3.8-Flash-Next"
        _, reason = resolve_splash_target(odd, None, [str(draft)])
        assert reason == (
            "Splash serves Qwen3.8-Flash-Next GGUFs with a multiple of 8 routed "
            "experts up to 512, not 260"
        )


class TestStaging:
    def test_alike_model_ids_stage_apart(self):
        assert splash_repo_id("org/model") != splash_repo_id("org-model")

    @pytest.mark.parametrize(
        "model_id",
        [
            "Qwen3.8-27B-4bit",
            "mlx-community/Qwen3.8-27B-4bit",
            "-a--b..c.-",
            "x" * 120,
            "model.git",
        ],
    )
    def test_repo_id_passes_splash_model_id_check(self, model_id):
        repo_id = splash_repo_id(model_id)

        assert SPLASH_REPO_ID.fullmatch(repo_id)
        assert "--" not in repo_id and ".." not in repo_id
        assert not repo_id.endswith(".git")
        assert repo_id.startswith("omlx/")

    def test_stage_model_links_files_and_points_refs_main(self, tmp_path):
        model = _write_model(tmp_path / "models" / "Qwen3.8-27B-4bit")
        (model / ".cache" / "huggingface").mkdir(parents=True)
        (model / ".cache" / "huggingface" / "x.metadata").write_text("abc")

        snapshot = stage_model(model, "omlx/Qwen3.8-27B-4bit", tmp_path / "hub")

        repo_dir = tmp_path / "hub" / "models--omlx--Qwen3.8-27B-4bit"
        assert (repo_dir / "refs" / "main").read_text() == snapshot.name
        assert re.fullmatch(r"[0-9a-f]{40}", snapshot.name)
        linked = sorted(p.name for p in snapshot.iterdir())
        assert linked == sorted(p.name for p in model.iterdir() if p.is_file())
        assert (snapshot / "config.json").resolve() == (model / "config.json").resolve()

    def test_changed_files_get_a_new_snapshot(self, tmp_path):
        model = _write_model(tmp_path / "m")
        first = stage_model(model, "omlx/m", tmp_path / "hub")

        (model / "model.safetensors").write_bytes(b"1" * 2048)
        second = stage_model(model, "omlx/m", tmp_path / "hub")

        assert second.name != first.name
        assert not first.exists()

    def test_hub_blob_links_keep_their_blob_target(self, tmp_path):
        # A model inside the user's Hub cache: Splash reuses blob names
        # instead of hashing files whose link resolves into blobs/.
        blobs = tmp_path / "cache" / "blobs"
        blobs.mkdir(parents=True)
        source = _write_model(tmp_path / "source")
        snapshot = tmp_path / "cache" / "snapshots" / "rev"
        snapshot.mkdir(parents=True)
        for path in source.iterdir():
            blob = blobs / f"{path.name}-sha"
            path.rename(blob)
            (snapshot / path.name).symlink_to(blob)

        staged = stage_model(snapshot, "omlx/m", tmp_path / "hub")

        assert (staged / "config.json").resolve().parent == blobs.resolve()

    def test_optiq_layout_stages_as_splash_reads_it(self, tmp_path):
        # Splash refuses index shards outside the snapshot root and a vision
        # checkpoint without a processor file.
        model = _write_model(tmp_path / "m")
        weight_map = _add_optiq_vision(model)

        snapshot = stage_model(model, "omlx/m", tmp_path / "hub")

        index = snapshot / "model.safetensors.index.json"
        assert not index.is_symlink()
        assert set(json.loads(index.read_text())["weight_map"].values()) == {
            "model.safetensors",
            "optiq-optiq_vision.safetensors",
        }
        sidecar = (model / "optiq" / "optiq_vision.safetensors").resolve()
        assert (snapshot / "optiq-optiq_vision.safetensors").resolve() == sidecar
        assert json.loads((snapshot / "preprocessor_config.json").read_text()) == {
            "patch_size": 16,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
        model_index = json.loads((model / "model.safetensors.index.json").read_text())
        assert model_index["weight_map"] == weight_map


# -- request translation -----------------------------------------------------------


class TestRequestTranslation:
    @pytest.mark.parametrize(
        "kwargs, effort",
        [
            ({"enable_thinking": False}, "none"),
            ({"reasoning_effort": "high"}, "high"),
            ({"reasoning_effort": "Off"}, "none"),
            ({"reasoning_effort": "moderate"}, "medium"),
            ({"reasoning_effort": "turbo"}, None),
            ({}, None),
            (None, None),
        ],
    )
    def test_effort_maps_to_splash_levels(self, kwargs, effort):
        assert splash_effort(kwargs) == effort

    def test_tool_call_arguments_become_json_strings(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "f", "arguments": {"a": 1}}}
                ],
            },
        ]

        converted = splash_messages(messages)

        call = converted[1]["tool_calls"][0]
        assert call["type"] == "function"
        assert json.loads(call["function"]["arguments"]) == {"a": 1}
        assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"a": 1}

    def test_env_resolves_offline_from_the_private_cache(self, monkeypatch):
        monkeypatch.setenv("PYTHONHOME", "/bundle")
        monkeypatch.setenv("PYTHONPATH", "/bundle/lib")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        env = splash_mod._splash_env("secret", Path("/omlx/splash/hub"))

        assert env["SPLASH_API_KEY"] == "secret"
        assert env["HF_HUB_CACHE"] == "/omlx/splash/hub"
        assert env["HF_HUB_OFFLINE"] == "1"
        assert "PYTHONHOME" not in env and "PYTHONPATH" not in env
        assert env["PATH"].split(":") == [
            *splash_mod._BREW_BIN_DIRS,
            "/usr/bin",
            "/bin",
        ]

    def test_serve_command_maps_omlx_settings(self, tmp_path):
        engine = _engine(
            tmp_path,
            max_context=10**6,
            max_memory_bytes=24 * GIB,
            kv_format="bf16",
            ane_prefill=False,
            cache_disk_bytes=5 * GIB,
            cache_dir=tmp_path / "ssd",
        )

        command = engine._serve_command(9000)

        def value(flag):
            return command[command.index(flag) + 1]

        assert command[:4] == [
            "/opt/homebrew/bin/splash",
            "serve",
            "--model",
            REPO_ID,
        ]
        assert {
            "--offline",
            "--language-only",
            "--no-webui",
            "--persistent-cache",
            "--disable-ane",
        } <= set(command)
        assert value("--draft-model") == str((tmp_path / "draft").resolve())
        assert value("--idle-release") == "off"
        assert value("--kv-format") == "bf16"
        assert value("--max-context") == str(splash_mod.SPLASH_MAX_CONTEXT)
        assert value("--max-memory") == str(24 * GIB)
        assert value("--max-cache-disk") == str(5 * GIB)
        assert value("--cache-dir") == str(tmp_path / "ssd")

    def test_serve_command_leaves_unset_limits_to_splash(self, tmp_path):
        command = _engine(tmp_path)._serve_command(9000)

        for flag in (
            "--max-context",
            "--max-memory",
            "--max-cache-disk",
            "--disable-ane",
        ):
            assert flag not in command

    def test_request_body_passes_sampling_through(self, tmp_path):
        body = _engine(tmp_path)._request_body(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=100,
            min_p=0.05,
            repetition_penalty=1.1,
            presence_penalty=0.5,
            kwargs={"response_format": {"type": "json_object"}, "seed": 7},
        )

        assert body["top_k"] == 100 and body["min_p"] == 0.05
        assert body["repetition_penalty"] == 1.1 and body["presence_penalty"] == 0.5
        assert body["response_format"] == {"type": "json_object"}
        assert body["seed"] == 7
        assert "stop" not in body

    def _body_with(self, tmp_path, version, kwargs):
        return _engine(tmp_path, version=version)._request_body(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            kwargs=kwargs,
        )

    def test_request_body_passes_the_thinking_budget(self, tmp_path):
        body = self._body_with(tmp_path, "1.3.2", {"thinking_budget": 4096})

        assert body["thinking_token_budget"] == 4096

    def test_request_body_leaves_an_unset_budget_to_splash(self, tmp_path):
        body = self._body_with(tmp_path, "1.3.2", {"thinking_budget": None})

        assert "thinking_token_budget" not in body

    def test_splash_before_1_3_2_gets_no_budget_field(self, tmp_path):
        # thinking_token_budget arrived in Splash 1.3.2.
        body = self._body_with(tmp_path, "1.3.1", {"thinking_budget": 4096})

        assert "thinking_token_budget" not in body


# -- streaming -------------------------------------------------------------------


def _sse(*chunks) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def _engine_with_transport(tmp_path, handler) -> SplashEngine:
    engine = _engine(tmp_path)
    engine._client = httpx.AsyncClient(
        base_url="http://splash", transport=httpx.MockTransport(handler)
    )
    return engine


class TestStreaming:
    async def test_reasoning_content_and_tool_calls(self, tmp_path):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            call_start = {
                "index": 0,
                "id": "call_1",
                "function": {"name": "get", "arguments": '{"a"'},
            }
            call_end = {"index": 0, "function": {"arguments": ": 1}"}}
            body = _sse(
                {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
                {"choices": [{"delta": {"content": "Hi"}}]},
                {"choices": [{"delta": {"tool_calls": [call_start]}}]},
                {
                    "choices": [
                        {
                            "delta": {"tool_calls": [call_end]},
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                        "prompt_tokens_details": {"cached_tokens": 4},
                    },
                },
            )
            return httpx.Response(200, content=body)

        engine = _engine_with_transport(tmp_path, handler)
        outputs = [
            o
            async for o in engine.stream_chat(
                [{"role": "user", "content": "hi"}],
                chat_template_kwargs={"reasoning_effort": "high"},
                tools=[{"type": "function", "function": {"name": "get"}}],
            )
        ]

        final = outputs[-1]
        assert final.finished
        assert final.text == "<think>hmm</think>Hi"
        assert final.finish_reason == "tool_calls"
        assert final.tool_calls == [
            {"id": "call_1", "name": "get", "arguments": '{"a": 1}'}
        ]
        assert (final.prompt_tokens, final.completion_tokens, final.cached_tokens) == (
            7,
            3,
            4,
        )
        assert all(not o.finished for o in outputs[:-1])
        assert outputs[0].first_token_at is not None
        body = seen["body"]
        assert body["model"] == REPO_ID
        assert body["reasoning_effort"] == "high"
        assert body["stream"] is True and body["return_progress"] is True
        assert engine.has_active_requests() is False

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"preserve_thinking": True}, True),
            ({"preserve_thinking": False}, False),
            ({"reasoning_effort": "high"}, None),
            (None, None),
        ],
    )
    async def test_preserve_thinking_is_forwarded(self, tmp_path, kwargs, expected):
        # Qwen 3.6+ templates drop reasoning from turns before the latest user
        # message unless preserve_thinking is set, which defeats prefix caching.
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200, content=_sse({"choices": [{"delta": {"content": "x"}}]})
            )

        engine = _engine_with_transport(tmp_path, handler)
        async for _ in engine.stream_chat(
            [{"role": "user", "content": "hi"}], chat_template_kwargs=kwargs
        ):
            pass
        assert seen["body"].get("preserve_thinking") is expected

    async def test_unfinished_reasoning_is_closed(self, tmp_path):
        def handler(request):
            return httpx.Response(
                200, content=_sse({"choices": [{"delta": {"reasoning_content": "x"}}]})
            )

        engine = _engine_with_transport(tmp_path, handler)
        final = await engine.chat([{"role": "user", "content": "hi"}])

        assert final.text == "<think>x</think>"
        assert final.new_text == final.text

    async def test_client_errors_become_invalid_requests(self, tmp_path):
        def handler(request):
            return httpx.Response(400, json={"error": {"message": "bad top_p"}})

        engine = _engine_with_transport(tmp_path, handler)

        with pytest.raises(InvalidRequestError, match="bad top_p"):
            await engine.chat([{"role": "user", "content": "hi"}])
        assert engine.has_active_requests() is False

    async def test_server_errors_raise(self, tmp_path):
        engine = _engine_with_transport(
            tmp_path, lambda request: httpx.Response(503, text="engine failure")
        )

        with pytest.raises(RuntimeError, match="503"):
            await engine.chat([{"role": "user", "content": "hi"}])

    async def test_live_requests_during_stream_and_on_error(self, tmp_path):
        checked = {"prefill": False, "generating": False}

        class StreamingBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                progress = {
                    "total": 100,
                    "cache": 20,
                    "processed": 60,
                    "time_ms": 200.0,
                }
                yield f"data: {json.dumps({'choices': [], 'prompt_progress': progress})}\n\n".encode()
                live = engine.get_live_requests()
                assert live["generating"] == []
                row = live["prefilling"][0]
                assert row["request_id"] == "req-1"
                assert (row["total"], row["processed"]) == (100, 60)
                assert row["speed"] == (60 - 20) / 0.2
                assert row["eta"] == (100 - 60) / 200.0
                checked["prefill"] = True
                chunk = {"choices": [{"delta": {"content": "Hello world"}}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"

        engine = _engine_with_transport(
            tmp_path, lambda request: httpx.Response(200, stream=StreamingBody())
        )
        async for out in engine.stream_chat(
            [{"role": "user", "content": "hi"}], max_tokens=128, _request_id="req-1"
        ):
            if not out.finished:
                live = engine.get_live_requests()
                assert live["prefilling"] == []
                row = live["generating"][0]
                assert row["generated_tokens"] > 0
                assert (row["prompt_tokens"], row["max_tokens"]) == (100, 128)
                checked["generating"] = True

        assert checked == {"prefill": True, "generating": True}
        assert engine.get_live_requests() == {"prefilling": [], "generating": []}

        engine._client = httpx.AsyncClient(
            base_url="http://splash",
            transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")),
        )
        with pytest.raises(RuntimeError):
            async for _ in engine.stream_chat([{"role": "user", "content": "hi"}]):
                pass
        assert engine.get_live_requests() == {"prefilling": [], "generating": []}

    async def test_plain_completions_are_rejected(self, tmp_path):
        with pytest.raises(InvalidRequestError, match="chat requests only"):
            await _engine(tmp_path).generate("Once upon a time")


# -- process lifecycle -------------------------------------------------------------

# A stand-in for `splash serve`: resolves --model offline from the private
# cache like the real launcher, then serves /v1/models and /status to requests
# that carry the key from SPLASH_API_KEY.
FAKE_SERVER = textwrap.dedent("""\
    import json, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    args = sys.argv[1:]
    model, port = args[args.index("--model") + 1], int(args[args.index("--port") + 1])
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    repo = os.path.join(os.environ["HF_HUB_CACHE"], "models--" + model.replace("/", "--"))
    revision = open(os.path.join(repo, "refs", "main")).read()
    assert os.path.isfile(os.path.join(repo, "snapshots", revision, "config.json"))
    key = os.environ["SPLASH_API_KEY"]
    status = {
        "maximum_context_tokens": 32768,
        "memory_plan": {"budget": {"hard_budget_bytes": 7 * 1024**3}},
    }
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer " + key:
                self.send_response(401); self.end_headers(); return
            data = {"data": [{"id": model}]} if self.path == "/v1/models" else status
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        def log_message(self, *a):
            pass
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """)


@pytest.fixture
def lifecycle(tmp_path, monkeypatch, launchers):
    monkeypatch.setattr(splash_mod, "STARTUP_POLL_SECONDS", 0.05)
    monkeypatch.setattr(
        SplashEngine,
        "_load_tokenizer",
        lambda self: setattr(self, "_tokenizer", MagicMock()),
    )
    _write_model(tmp_path / "models" / "Qwen3.8-27B-4bit")

    def install(body: str) -> SplashEngine:
        launcher = launchers("splash", body=body)
        engine = _engine(tmp_path)
        engine._target = SplashTarget(
            SplashBinary(launcher, "1.3.1", True), "Qwen3.8-27B", tmp_path / "draft"
        )
        return engine

    return install


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except (ProcessLookupError, PermissionError):
        # macOS reports EPERM for a group that only holds unreaped zombies.
        return False
    return True


class TestLifecycle:
    async def test_start_serves_the_staged_model_and_stop_ends_the_group(
        self, tmp_path, lifecycle
    ):
        script = tmp_path / "server.py"
        script.write_text(FAKE_SERVER)
        engine = lifecycle(f'exec "{sys.executable}" "{script}" "$@"\n')

        await engine.start()
        group = engine._process.pid
        try:
            assert engine.get_stats()["running"] is True
            assert engine.reserved_bytes() == 7 * GIB
            # The context Splash fit to its budget, for the server's limit.
            assert engine.max_context_tokens == 32768
            # Request paths call start() again on a loaded engine.
            await engine.start()
            assert engine._process.pid == group
        finally:
            await engine.stop()

        assert not _group_alive(group)
        assert engine.get_stats()["running"] is False

    async def test_exit_during_start_reports_the_log(self, tmp_path, lifecycle):
        engine = lifecycle(
            'echo "Installing collected packages: numpy"\n'
            'echo "error: the engine\'s device check failed" >&2\nexit 1\n'
        )

        with pytest.raises(RuntimeError, match="device check failed") as error:
            await engine.start()
        assert "Installing" not in str(error.value)
        assert engine._process is None

    async def test_stop_waits_for_splash_and_kills_a_stuck_one(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(splash_mod, "STOP_TIMEOUT_SECONDS", 1)
        # A Splash stand-in that outlives SIGTERM; the watchdog shell does not.
        stuck = [
            sys.executable,
            "-c",
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "while True: time.sleep(0.1)",
        ]
        engine = _engine(tmp_path)
        engine._process = subprocess.Popen(
            splash_mod._watchdog_command(stuck), start_new_session=True
        )
        group = engine._process.pid
        await asyncio.sleep(0.5)

        await engine.stop()

        assert not _group_alive(group)

    async def test_stop_signals_a_process_that_missed_the_first_sigterm(
        self, tmp_path, caplog
    ):
        # Stands in for a process the watchdog forks while SIGTERM is in
        # flight: the first signal never reaches it, a second one ends it.
        ready = tmp_path / "ready"
        late = [
            sys.executable,
            "-c",
            "import pathlib, signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda *_: signal.signal(signal.SIGTERM, signal.SIG_DFL))\n"
            "pathlib.Path(sys.argv[1]).touch()\n"
            "while True: time.sleep(0.1)",
            str(ready),
        ]
        engine = _engine(tmp_path)
        engine._process = subprocess.Popen(
            splash_mod._watchdog_command(late), start_new_session=True
        )
        group = engine._process.pid
        try:
            for _ in range(100):
                if ready.exists():
                    break
                await asyncio.sleep(0.1)
            assert ready.exists()

            await asyncio.wait_for(engine.stop(), timeout=10)
        finally:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(group, signal.SIGKILL)

        assert not _group_alive(group)
        assert "ignored SIGTERM" not in caplog.text

    async def test_dead_splash_is_reported_and_not_relaunched(self, tmp_path):
        engine = _engine(tmp_path)
        engine._process = subprocess.Popen(["/bin/sh", "-c", "exit 3"])
        engine._process.wait()

        assert "exited with code 3" in engine.runtime_failed_reason
        with pytest.raises(RuntimeError, match="exited with code 3"):
            await engine.start()

    def test_watchdog_stops_splash_when_its_parent_dies(self):
        # The watchdog polls its parent's PID; a PID that is gone must end
        # the command it supervises.
        dead_parent = os.fork()
        if dead_parent == 0:
            os._exit(0)
        os.waitpid(dead_parent, 0)
        command = splash_mod._watchdog_command(["/bin/sleep", "60"])
        command[command.index(str(os.getpid()))] = str(dead_parent)
        process = subprocess.Popen(command, start_new_session=True)
        try:
            assert process.wait(timeout=10) is not None
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
        assert not _group_alive(process.pid)


# -- engine pool -----------------------------------------------------------------------


@pytest.fixture
def global_settings(tmp_path, monkeypatch):
    settings = GlobalSettings(base_path=tmp_path / "omlx")
    monkeypatch.setattr("omlx.engine_pool.get_global_settings", lambda: settings)
    return settings


def _pool(tmp_path, with_draft=True) -> tuple[EnginePool, EngineEntry]:
    pool = EnginePool()
    model = _write_model(tmp_path / "models" / "Qwen3.8-27B-4bit")
    entry = EngineEntry(
        model_id="Qwen3.8-27B-4bit",
        model_path=str(model),
        model_type="vlm",
        engine_type="vlm",
        estimated_size=16 * GIB,
        model_context_length=262144,
    )
    pool._entries[entry.model_id] = entry
    if with_draft:
        draft = _write_draft(tmp_path / "models" / "z-lab" / "Qwen3.8-27B-DFlash2")
        pool._entries["Qwen3.8-27B-DFlash2"] = EngineEntry(
            model_id="Qwen3.8-27B-DFlash2",
            model_path=str(draft),
            model_type="llm",
            engine_type="batched",
            estimated_size=GIB,
            is_helper=True,
        )
    return pool, entry


class TestEnginePool:

    async def test_cancelled_start_stops_splash_without_waiting_for_readiness(
        self, tmp_path, lifecycle, monkeypatch
    ):
        engine = lifecycle("exec sleep 60\n")
        pool, entry = _pool(tmp_path / "pool")
        pool._get_final_ceiling = lambda: 0
        monkeypatch.setattr(pool, "_external_engine_for", lambda *args: engine)
        waiting = asyncio.Event()
        finish_readiness = asyncio.Event()

        async def wait_until_ready():
            waiting.set()
            await finish_readiness.wait()

        monkeypatch.setattr(engine, "_wait_until_ready", wait_until_ready)
        loading = asyncio.create_task(pool.get_engine(entry.model_id))
        try:
            await asyncio.wait_for(waiting.wait(), timeout=5)
            group = engine._process.pid
            loading.cancel()
            done, _ = await asyncio.wait({loading}, timeout=5)
            assert loading in done
            with pytest.raises(asyncio.CancelledError):
                await loading
            assert not _group_alive(group)
            assert not pool._lock.locked()
            assert not entry.is_loading
            assert entry.engine is None
            assert entry.model_id not in pool._starting_external
        finally:
            finish_readiness.set()
            await asyncio.gather(loading, return_exceptions=True)
            await engine.stop()

    def test_models_without_the_setting_stay_in_process(self, tmp_path):
        pool, entry = _pool(tmp_path)

        assert pool._external_engine_for(entry, None, "vlm") is None
        assert pool._external_engine_for(entry, ModelSettings(), "vlm") is None

    def test_selected_engine_that_cannot_serve_fails_the_load(
        self, tmp_path, launchers, global_settings
    ):
        launchers("splash")
        pool, entry = _pool(tmp_path, with_draft=False)

        with pytest.raises(ModelLoadingError, match="Unknown external engine"):
            pool._external_engine_for(entry, ModelSettings(external_engine="x"), "vlm")
        with pytest.raises(ModelLoadingError, match="DFlash2 draft"):
            pool._external_engine_for(
                entry, ModelSettings(external_engine="splash"), "vlm"
            )

    def test_a_gguf_runs_on_an_external_engine_without_the_setting(
        self, tmp_path, launchers, global_settings
    ):
        pool, entry = _pool(tmp_path)
        entry.model_format = "gguf"

        # No in-process fallback, even with no engine to run it.
        with pytest.raises(ModelLoadingError, match="Splash is not installed"):
            pool._external_engine_for(entry, None, "vlm")
        launchers("splash")
        engine = pool._external_engine_for(entry, ModelSettings(), "vlm")
        assert isinstance(engine, SplashEngine)

    async def test_gguf_over_the_ceiling_reports_why_splash_cannot_serve(
        self, tmp_path, launchers, global_settings
    ):
        launchers("splash")
        pool, entry = _pool(tmp_path, with_draft=False)
        entry.model_format = "gguf"
        entry.estimated_size = 55 * GIB
        pool._get_final_ceiling = lambda: 32 * GIB

        with pytest.raises(ModelLoadingError, match="DFlash2 draft"):
            await pool.get_engine(entry.model_id)

    def test_options_follow_omlx_settings(self, tmp_path, launchers, global_settings):
        launchers("splash")
        pool, entry = _pool(tmp_path)
        pool._get_final_ceiling = lambda: 200 * GIB
        pool._current_model_memory = 40 * GIB
        cfg = pool._scheduler_config
        cfg.paged_ssd_cache_dir = str(tmp_path / "ssd")
        cfg.paged_ssd_cache_max_size = 50 * GIB
        cfg.hot_cache_only = False
        global_settings.integrations.splash_kv_format = "bf16"
        settings = ModelSettings(external_engine="splash", max_context_window=65536)

        engine = pool._external_engine_for(entry, settings, "vlm")

        options = engine._options
        assert options.max_context == 65536
        # Empty limit: enough for the 64K window, well inside the room.
        assert options.max_memory_bytes == splash_memory_estimate(
            entry.model_path, engine._target.draft_path, 65536, "bf16"
        )
        assert options.kv_format == "bf16"
        assert options.cache_disk_bytes == 50 * GIB
        assert options.cache_dir == tmp_path / "ssd" / "splash" / REPO_ID.split("/")[1]
        assert engine._target.draft_path.name == "Qwen3.8-27B-DFlash2"
        assert engine._hub_dir == global_settings.base_path / "splash" / "hub"

        global_settings.integrations.splash_max_memory_gb = 24
        global_settings.integrations.splash_ssd_cache_gb = 0
        global_settings.sampling.max_context_window_policy = 131072
        engine = pool._external_engine_for(
            entry, ModelSettings(external_engine="splash"), "vlm"
        )
        assert engine._options.max_memory_bytes == 24 * GIB
        assert engine._options.cache_disk_bytes == 0
        # No explicit limit: Splash fits the context to its budget.
        assert engine._options.max_context is None

        # The ceiling's room still bounds a larger limit.
        global_settings.integrations.splash_max_memory_gb = 180
        engine = pool._external_engine_for(
            entry, ModelSettings(external_engine="splash"), "vlm"
        )
        assert engine._options.max_memory_bytes == 160 * GIB

        cfg.hot_cache_only = True
        global_settings.integrations.splash_ssd_cache_gb = None
        engine = pool._external_engine_for(
            entry, ModelSettings(external_engine="splash"), "vlm"
        )
        assert engine._options.cache_disk_bytes == 0

    def test_budget_stays_under_the_admission_target(
        self, tmp_path, launchers, global_settings
    ):
        launchers("splash")
        pool, entry = _pool(tmp_path)
        pool._get_final_ceiling = lambda: 200 * GIB
        pool._get_admission_soft_target = lambda: 100 * GIB
        global_settings.integrations.splash_max_memory_gb = 64
        settings = ModelSettings(external_engine="splash")

        # Admission asks for what Splash wants, up to the soft target.
        assert pool._entry_runtime_resident_size(entry, settings) == 64 * GIB
        global_settings.integrations.splash_max_memory_gb = 150
        assert pool._entry_runtime_resident_size(entry, settings) == 100 * GIB

        pool._current_model_memory = 90 * GIB
        engine = pool._external_engine_for(entry, settings, "vlm")
        assert engine._options.max_memory_bytes == 10 * GIB
        pool._current_model_memory = 100 * GIB
        with pytest.raises(ModelLoadingError, match="leaves no memory"):
            pool._external_engine_for(entry, settings, "vlm")

    def test_limit_the_shrunk_budget_cannot_hold_is_left_to_splash(
        self, tmp_path, launchers, global_settings
    ):
        launchers("splash")
        pool, entry = _pool(tmp_path)
        pool._get_final_ceiling = lambda: 40 * GIB
        settings = ModelSettings(external_engine="splash", max_context_window=262144)
        engine = pool._external_engine_for(entry, settings, "vlm")
        draft = engine._target.draft_path
        wanted = splash_memory_estimate(entry.model_path, draft, 262144, "int8")

        # Room for the whole window: the limit goes to Splash.
        assert engine._options.max_memory_bytes == wanted
        assert engine._options.max_context == 262144

        # Other models take memory, and the room left cannot hold 256K:
        # Splash would refuse --max-context 262144, so it fits the context.
        pool._current_model_memory = 40 * GIB - (wanted - 1)
        engine = pool._external_engine_for(entry, settings, "vlm")
        assert engine._options.max_memory_bytes == wanted - 1
        assert engine._options.max_context is None

        # A smaller limit the shrunk room still holds goes to Splash.
        small = ModelSettings(external_engine="splash", max_context_window=32768)
        engine = pool._external_engine_for(entry, small, "vlm")
        assert engine._options.max_context == 32768

        # A configured budget too small for the limit leaves it to Splash too.
        pool._current_model_memory = 0
        global_settings.integrations.splash_max_memory_gb = 4
        engine = pool._external_engine_for(entry, settings, "vlm")
        assert engine._options.max_memory_bytes == 4 * GIB
        assert engine._options.max_context is None

    async def test_load_counts_the_planned_budget_and_unload_skips_mlx_barrier(
        self, tmp_path, launchers, global_settings
    ):
        launchers("splash")
        pool, entry = _pool(tmp_path)
        pool._get_final_ceiling = lambda: 0
        manager = MagicMock()
        manager.get_settings.return_value = ModelSettings(external_engine="splash")
        pool._settings_manager = manager

        async def fake_start(self):
            self._tokenizer = MagicMock()
            self._status = {"memory_plan": {"budget": {"hard_budget_bytes": 30 * GIB}}}

        with (
            patch.object(SplashEngine, "start", fake_start),
            patch.object(SplashEngine, "stop", AsyncMock()) as stop,
        ):
            await pool._load_engine(entry.model_id)
            assert isinstance(entry.engine, SplashEngine)
            assert pool._current_model_memory == 30 * GIB
            assert entry.actual_size == 30 * GIB

            with patch("omlx.engine_pool.mx") as mx:
                await pool._unload_engine(entry.model_id)

        stop.assert_awaited_once()
        mx.get_active_memory.assert_not_called()
        assert entry.engine is None
        assert pool._current_model_memory == 0

    def test_runtime_signature_reloads_on_engine_change_only(self, tmp_path):
        pool, entry = _pool(tmp_path)

        mlx = pool._engine_runtime_signature(entry.model_id, ModelSettings())
        splash = pool._engine_runtime_signature(
            entry.model_id, ModelSettings(external_engine="splash")
        )
        # An exposed profile overriding the context must not restart Splash.
        longer = pool._engine_runtime_signature(
            entry.model_id,
            ModelSettings(external_engine="splash", max_context_window=65536),
        )

        assert mlx != splash
        assert splash == longer

    def test_status_reports_out_of_process_memory(self, tmp_path):
        pool, entry = _pool(tmp_path)
        engine = _engine(tmp_path)
        engine.process_memory_bytes = lambda: 20 * GIB
        entry.engine = engine
        pool._entries["mlx"] = EngineEntry(
            model_id="mlx",
            model_path="/tmp/mlx",
            model_type="llm",
            engine_type="batched",
            estimated_size=GIB,
            engine=MagicMock(),
        )

        assert pool.external_process_memory_bytes() == 20 * GIB
        starting = _engine(tmp_path)
        starting.process_memory_bytes = lambda: 5 * GIB
        pool._starting_external["other"] = starting
        assert pool.external_process_memory_bytes() == 25 * GIB
        del pool._starting_external["other"]
        rows = {m["id"]: m for m in pool.get_status()["models"]}
        assert rows[entry.model_id]["actual_size"] == 20 * GIB
        assert rows[entry.model_id]["external_engine"] == "splash"
        assert rows["mlx"]["external_engine"] is None


# -- admin API -------------------------------------------------------------------------


async def _post_global(settings: GlobalSettings, **fields) -> dict:
    with patch.object(admin_routes, "_get_global_settings", lambda: settings):
        return await admin_routes.update_global_settings(
            request=admin_routes.GlobalSettingsRequest(**fields), is_admin=True
        )


async def _put_model(pool, settings: ModelSettings, **fields) -> ModelSettings:
    manager = MagicMock()
    manager.get_settings.return_value = copy.deepcopy(settings)
    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch.object(admin_routes, "_get_settings_manager", return_value=manager),
        patch.object(admin_routes, "_get_server_state", return_value=MagicMock()),
    ):
        await admin_routes.update_model_settings(
            "Qwen3.8-27B-4bit",
            admin_routes.ModelSettingsRequest(**fields),
            is_admin=True,
        )
    return manager.set_settings.call_args.args[1]


class TestAdminAPI:
    async def test_splash_path_is_checked_only_when_it_changes(
        self, tmp_path, launchers
    ):
        settings = GlobalSettings(base_path=tmp_path)

        with pytest.raises(HTTPException) as error:
            await _post_global(settings, splash_path=str(tmp_path / "missing"))
        assert error.value.status_code == 400
        assert settings.integrations.splash_path is None

        # A launcher removed after it was saved must not block other saves,
        # because the web UI resends every integration field.
        settings.integrations.splash_path = "/removed/splash"
        await _post_global(
            settings, splash_path="/removed/splash", markitdown_enabled=False
        )
        assert settings.integrations.markitdown_enabled is False

        custom = launchers("splash")
        await _post_global(settings, splash_path=f"  {custom}  ")
        assert settings.integrations.splash_path == str(custom)
        await _post_global(settings, splash_path="")
        assert settings.integrations.splash_path is None

    async def test_splash_options_are_validated_and_persisted(self, tmp_path):
        settings = GlobalSettings(base_path=tmp_path)

        result = await _post_global(
            settings,
            splash_kv_format="bf16",
            splash_max_memory_gb=24,
            splash_ssd_cache_gb=0,
            splash_ane_prefill=False,
        )

        assert "integrations" in result["runtime_applied"]
        restored = GlobalSettings.load(base_path=tmp_path).integrations
        assert (restored.splash_kv_format, restored.splash_max_memory_gb) == (
            "bf16",
            24,
        )
        assert (restored.splash_ssd_cache_gb, restored.splash_ane_prefill) == (0, False)
        for fields in (
            {"splash_kv_format": "fp8"},
            {"splash_max_memory_gb": 0},
            {"splash_ssd_cache_gb": -1},
        ):
            with pytest.raises(HTTPException):
                await _post_global(settings, **fields)

    def test_global_settings_report_the_detected_launcher(self, tmp_path, launchers):
        launcher = launchers("splash")

        def pool_not_ready():
            raise HTTPException(status_code=503, detail="Server not initialized")

        with patch.object(admin_routes, "_get_engine_pool", pool_not_ready):
            response = admin_routes._global_settings_response(
                GlobalSettings(base_path=tmp_path)
            )

        status = response["external_engines"]["splash"]
        assert status["available"] is True
        assert status["path"] == str(launcher)
        assert (status["version"], status["auto_detected"]) == ("1.3.1", True)
        assert response["integrations"]["splash_kv_format"] == "int8"

    async def test_model_external_engine_is_validated(
        self, tmp_path, launchers, global_settings
    ):
        launchers("splash")
        pool, _ = _pool(tmp_path)

        saved = await _put_model(pool, ModelSettings(), external_engine="splash")
        assert saved.external_engine == "splash"
        saved = await _put_model(pool, saved, external_engine=None)
        assert saved.external_engine is None

        with pytest.raises(HTTPException, match="Unknown external engine"):
            await _put_model(pool, ModelSettings(), external_engine="llama")
        saved = await _put_model(pool, saved, external_engine="splash")
        del pool._entries["Qwen3.8-27B-DFlash2"]
        with pytest.raises(HTTPException, match="DFlash2 draft"):
            await _put_model(pool, ModelSettings(), external_engine="splash")
        # The web modal resends the engine with every save; a draft removed
        # after it was chosen must not block an unrelated change.
        saved = await _put_model(pool, saved, external_engine="splash", temperature=0.5)
        assert (saved.external_engine, saved.temperature) == ("splash", 0.5)

    def test_model_list_reports_engine_compatibility(
        self, tmp_path, launchers, global_settings
    ):
        launchers("splash")
        pool, entry = _pool(tmp_path)

        rows = {
            info["id"]: admin_routes._external_engines_for_model(pool, info, None)
            for info in pool.get_status()["models"]
        }

        assert rows[entry.model_id] == [
            {
                "id": "splash",
                "label": "Splash",
                "compatible": True,
                "reason": "",
                "draft_repo": None,
            }
        ]
        assert rows["Qwen3.8-27B-DFlash2"] == []

        # Without its draft the model names the repo that provides it.
        del pool._entries["Qwen3.8-27B-DFlash2"]
        info = pool.get_status()["models"][0]
        (row,) = admin_routes._external_engines_for_model(pool, info, None)
        assert not row["compatible"]
        assert row["draft_repo"] == "incoai/Qwen3.8-27B-DFlash2"

        # A draft would not help a checkpoint Splash refuses for another reason.
        bf16 = _write_model(tmp_path / "bf16")
        config = json.loads((bf16 / "config.json").read_text())
        del config["quantization"]
        (bf16 / "config.json").write_text(json.dumps(config))
        entry.model_path = str(bf16)
        (row,) = admin_routes._external_engines_for_model(pool, info, None)
        assert row["draft_repo"] is None


# -- native response_format ---------------------------------------------------


class TestNativeResponseFormat:
    def test_pydantic_json_schema_goes_out_in_wire_form(self):
        engine = MagicMock(supports_native_response_format=True)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        rf = ResponseFormat(
            type="json_schema",
            json_schema=ResponseFormatJsonSchema(
                name="person", schema=schema, strict=True
            ),
        )

        wire = srv._native_response_format(engine, rf)

        assert wire["json_schema"]["schema"] == schema
        assert "schema_" not in wire["json_schema"]
        assert wire["json_schema"]["strict"] is True

    def test_only_native_engines_without_structured_outputs_take_it(self):
        rf = ResponseFormat(type="json_object")
        native = MagicMock(supports_native_response_format=True)

        assert srv._native_response_format(MagicMock(spec=[]), rf) is None
        assert srv._native_response_format(native, None) is None
        assert (
            srv._native_response_format(
                native, rf, structured_outputs={"json_schema": {}}
            )
            is None
        )

    @pytest.fixture
    def serve_engine(self, monkeypatch):
        def install(engine):
            engine.model_type = "qwen3_5"
            engine.is_diffusion_model = False
            engine.start = AsyncMock()
            engine.count_chat_tokens.return_value = 10
            pool = MagicMock()
            pool.preload_pinned_models = AsyncMock()
            pool.check_ttl_expirations = AsyncMock()
            pool.shutdown = AsyncMock()
            pool.get_entry.return_value = SimpleNamespace(
                config_model_type="qwen3_5", preserve_thinking_default=False
            )
            monkeypatch.setattr(srv._server_state, "engine_pool", pool)
            monkeypatch.setattr(srv._server_state, "responses_store", ResponseStore())
            monkeypatch.setattr(
                srv, "get_engine_for_model", AsyncMock(return_value=engine)
            )
            monkeypatch.setattr(srv, "resolve_model_id", lambda name: name)
            monkeypatch.setattr(srv, "validate_context_window", lambda *a, **k: None)
            monkeypatch.setattr(
                srv,
                "get_model_settings_for_request",
                lambda name: ModelSettings(cache_reasoning_output=False),
            )
            monkeypatch.setitem(
                srv.app.dependency_overrides, srv.verify_inference_api_key, lambda: True
            )
            return TestClient(srv.app, raise_server_exceptions=False)

        return install

    @pytest.mark.parametrize("native", [True, False])
    def test_chat_completions(self, serve_engine, native):
        captured = {}

        async def preflight(*args, **kwargs):
            captured.update(kwargs)
            raise HTTPException(status_code=418, detail="Captured")

        engine = MagicMock(supports_native_response_format=native)
        engine.preflight_chat = AsyncMock(side_effect=preflight)
        with serve_engine(engine) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "response_format": {"type": "json_object"},
                },
            )

        assert response.status_code == 418
        if native:
            assert captured["response_format"] == {"type": "json_object"}
            assert "Warning" not in response.headers
        else:
            assert "response_format" not in captured

    @pytest.mark.parametrize("stream", [True, False])
    def test_responses_api(self, serve_engine, stream):
        captured = {}
        output = GenerationOutput(
            text='{"name": "test"}',
            new_text='{"name": "test"}',
            prompt_tokens=4,
            completion_tokens=6,
            finish_reason="stop",
            finished=True,
        )

        async def stream_chat(messages, **kwargs):
            captured.update(messages=messages, kwargs=kwargs)
            yield output

        async def chat(messages, **kwargs):
            captured.update(messages=messages, kwargs=kwargs)
            return output

        engine = MagicMock(supports_native_response_format=True, tokenizer=None)
        engine.preflight_chat = AsyncMock(return_value=None)
        engine.stream_chat = stream_chat
        engine.chat = AsyncMock(side_effect=chat)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        with serve_engine(engine) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "m",
                    "input": "hi",
                    "stream": stream,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "user_schema",
                            "schema": schema,
                            "strict": True,
                        }
                    },
                },
            )

        assert response.status_code == 200
        assert "Warning" not in response.headers
        rf = captured["kwargs"]["response_format"]
        assert "schema" in (rf.get("json_schema") or rf)
        # Splash enforces the format; no JSON instruction is injected.
        assert [m["content"] for m in captured["messages"]] == ["hi"]


# -- memory -------------------------------------------------------------------------


class TestMemoryEstimate:
    def test_one_context_window(self, tmp_path):
        text = {
            **QWEN38_TEXT,
            "full_attention_interval": 4,
            "layer_types": ["linear_attention"] * 3 * 16 + ["full_attention"] * 16,
        }
        model = _write_model(tmp_path / "m", text_config=text)
        draft = _write_draft(tmp_path / "draft")
        fixed = 1024 + int(64 / 3) + 3 * GIB

        # 16 attention layers x K/V x 4 heads x 256 values, plus int8 scales:
        # 33,280 bytes a token, Splash's 1040 KiB page of 32 tokens.
        int8 = splash_memory_estimate(model, draft, 262144, "int8")
        assert int8 - fixed == 33_280 * 262144
        bf16 = splash_memory_estimate(model, draft, 65536, "bf16")
        assert bf16 - fixed == 65_536 * 65536
        # Splash caps context at 256K, and no limit means the whole window.
        assert splash_memory_estimate(model, draft, None, "int8") == int8
        assert splash_memory_estimate(model, draft, 10**6, "int8") == int8

    def test_counts_a_vision_tower_in_a_subdirectory(self, tmp_path):
        model = _write_model(tmp_path / "m")
        draft = _write_draft(tmp_path / "draft")
        before = splash_memory_estimate(model, draft, 1024, "int8")

        _add_optiq_vision(model)

        assert splash_memory_estimate(model, draft, 1024, "int8") == before + 512


class TestProcessMemory:
    def test_list_process_group_pids(self):
        assert os.getpid() in list_process_group_pids(os.getpgid(0))
        assert list_process_group_pids(-1) == []
        assert list_process_group_pids(99999999) == []

    def test_memory_takes_splash_metal_accounting_over_footprint(
        self, tmp_path, monkeypatch
    ):
        model = _write_model(tmp_path / "models" / "Qwen3.8-27B-4bit")
        engine = _engine(tmp_path)
        assert engine.process_memory_bytes() == 0
        engine._process = MagicMock(pid=4200, **{"poll.return_value": None})
        monkeypatch.setattr(
            splash_mod, "list_process_group_pids", lambda pgid: [4200, 4201]
        )
        monkeypatch.setattr(splash_mod, "get_phys_footprint", lambda pid: 100)

        # Before the first status snapshot, the weights on disk stand in.
        weights = sum(f.stat().st_size for f in model.iterdir())
        assert engine.process_memory_bytes() == weights

        engine._status = {
            "memory_actual": {"current_bytes": 5 * GIB, "peak_bytes": 6 * GIB},
            "memory_plan": {"budget": {"hard_budget_bytes": 9 * GIB}},
        }
        assert engine.process_memory_bytes() == 5 * GIB
        assert engine.reserved_bytes() == 9 * GIB

    def test_benchmark_endpoint_requires_a_running_engine(self, tmp_path):
        engine = _engine(tmp_path)
        with pytest.raises(RuntimeError, match="not started"):
            engine.benchmark_endpoint()

        engine._process = MagicMock(pid=4200, **{"poll.return_value": None})
        engine._base_url = "http://127.0.0.1:8123"
        engine._api_key = "secret"
        assert engine.benchmark_endpoint() == {
            "base_url": "http://127.0.0.1:8123/v1",
            "api_key": "secret",
            "model": REPO_ID,
            "extra_body": {"reasoning_effort": "none"},
        }


# -- benchmark ------------------------------------------------------------------------


def _bench_engine(prompt_tokens=10_000) -> MagicMock:
    engine = MagicMock()
    engine.benchmark_endpoint.return_value = {
        "base_url": "http://127.0.0.1:8123/v1",
        "api_key": "key",
        "model": "splash-model",
        "extra_body": {"reasoning_effort": "none"},
    }
    engine.process_memory_bytes.return_value = 999 * 1024**2
    engine.count_chat_tokens.return_value = 40
    engine.tokenizer.encode.side_effect = lambda text, add_special_tokens=False: list(
        range(prompt_tokens)
    )
    engine.tokenizer.decode.side_effect = lambda ids: f"tokens:{len(ids)}"
    return engine


class TestSplashBenchmark:
    async def test_warmup_calibrates_template_overhead(self):
        bench = _ChatBenchmark(_bench_engine(), "code_python")
        stats = MagicMock(spec=StreamStats, prompt_tokens=_CHAT_CALIBRATION_TOKENS - 40)
        bench.client.stream_chat_completion = AsyncMock(return_value=stats)

        await bench.warmup(32)

        # The engine's template is 40 tokens shorter than the tokenizer's.
        assert bench.overhead == 0
        assert bench.prompt(100) == "tokens:100"

    async def test_run_benchmark_against_splash_endpoint(self, monkeypatch):
        stats = StreamStats(
            prompt_tokens=1000,
            completion_tokens=100,
            cached_tokens=0,
            start_time=0.0,
            first_content_time=0.5,
            last_content_time=1.5,
            end_time=1.5,
            text="benchmark output",
            content_observed=True,
        )
        client = MagicMock()
        client.stream_chat_completion = AsyncMock(return_value=stats)
        client.aclose = AsyncMock()
        monkeypatch.setattr(
            "omlx.admin.benchmark.ExternalAPIClient", lambda config: client
        )
        pool = MagicMock()
        pool.get_loaded_model_ids.return_value = []
        pool.get_engine = AsyncMock(return_value=_bench_engine(2000))
        pool._unload_engine = AsyncMock()
        pool.get_entry.return_value = SimpleNamespace(
            model_type="qwen3_5", model_settings=None, model_path=Path("/tmp/model")
        )
        run = BenchmarkRun(
            bench_id="splash-run-1",
            request=BenchmarkRequest(
                model_id="splash-model", prompt_lengths=[1024], batch_sizes=[2]
            ),
        )

        await run_benchmark(run, pool)

        client.aclose.assert_awaited_once()
        results = {r["test_type"]: r for r in run.results}
        assert (results["single"]["requested_pp"], results["single"]["pp"]) == (
            1024,
            1000,
        )
        assert results["batch"]["batch_size"] == 2
        assert results["batch"]["peak_memory_bytes"] == 999 * 1024**2
        # Benchmark requests bypass stream_chat() but keep the model busy.
        engine = pool.get_engine.return_value
        engine.outside_requests.assert_any_call(1)
        engine.outside_requests.assert_any_call(2)
        assert run.upload_state["skipped_reason"] == "splash"
        skipped = next(e for e in run.events if e["type"] == "upload_skipped")
        assert skipped["reason"] == "splash"


# -- cache observability ---------------------------------------------------------------


class TestSplashRuntimeCache:
    async def test_status_poller_and_stop_cancellation(self, tmp_path, monkeypatch):
        monkeypatch.setattr(splash_mod, "STATUS_POLL_SECONDS", 0.01)
        payload = {"kv": {"block_tokens": 32, "pages_cache": 15}}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/status":
                return httpx.Response(200, json=payload)
            return httpx.Response(404)

        engine = _engine_with_transport(tmp_path, handler)
        engine._status_task = asyncio.create_task(engine._poll_status())
        await asyncio.sleep(0.05)

        assert engine._status == payload
        await engine.stop()
        assert engine._status_task is None

    def test_cache_dashboard_lists_splash_next_to_mlx(self, tmp_path):
        settings = SimpleNamespace(
            base_path=tmp_path,
            cache=SimpleNamespace(
                ssd_cache_max_size="auto",
                get_ssd_cache_dir=lambda base_path: tmp_path / "ssd_cache",
                is_ssd_cache_dir_unavailable=lambda: False,
                get_ssd_cache_max_size_bytes=lambda base_path: 0,
            ),
        )

        class FakeMlxEngine:
            scheduler = None

            def get_runtime_cache_stats(self):
                return {
                    "block_size": 512,
                    "indexed_blocks": 25,
                    "ssd_cache": {"num_files": 2, "total_size_bytes": 300},
                    "cache_rates": {"cumulative": {"prefix_hits": 10}},
                }

        splash = _engine(tmp_path)
        splash._status = {
            "kv": {"block_tokens": 32, "pages_cache": 40},
            "disk": {
                "capacity_bytes": 2000,
                "used_bytes": 500,
                "kv_demotions": 3,
                "kv_restores": 2,
            },
            "cache": {
                "hits": 15,
                "cold_misses": 5,
                "reused_tokens": 1200,
                "kv_hit_tokens": 800,
                "kv_disk_hit_tokens": 400,
            },
            "state": {"entries": 6, "bytes": 24000, "evictions": 1},
        }
        pool = SimpleNamespace(
            _entries={
                "mlx-model": SimpleNamespace(engine=FakeMlxEngine()),
                "splash-model": SimpleNamespace(engine=splash),
            },
            get_status=lambda: {
                "models": [
                    {"id": "mlx-model", "loaded": True},
                    {"id": "splash-model", "loaded": True},
                ]
            },
        )

        with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
            payload = admin_routes._build_runtime_cache_observability(settings)

        rows = {m["id"]: m for m in payload["models"]}
        assert "splash_cache" not in rows["mlx-model"]
        splash_row = rows["splash-model"]
        assert splash_row["cache_tier"] == "splash"
        assert splash_row["splash_cache"] == {
            "reused_tokens": 1200,
            "kv_hit_tokens": 800,
            "kv_disk_hit_tokens": 400,
            "state_entries": 6,
            "state_bytes": 24000,
        }
        assert (splash_row["total_size_bytes"], splash_row["num_files"]) == (500, 1)
        assert (splash_row["block_size"], splash_row["indexed_blocks"]) == (32, 40)


# -- vision ----------------------------------------------------------------------

VISION_CONFIG = {"model_type": "qwen3_5", "depth": 27}
IMAGE = {
    "type": "image_url",
    "image_url": {"url": "data:image/png;base64," + "A" * 4000},
}


def _vision_engine(tmp_path) -> SplashEngine:
    model = _write_model(tmp_path / "models" / "Qwen3.8-27B-4bit")
    config = json.loads((model / "config.json").read_text())
    (model / "config.json").write_text(
        json.dumps({**config, "vision_config": VISION_CONFIG})
    )
    return _engine(tmp_path)


class TestVision:
    def test_text_only_checkpoint_stays_language_only(self, tmp_path):
        _write_model(tmp_path / "models" / "Qwen3.8-27B-4bit")
        engine = _engine(tmp_path)

        assert not engine.accepts_image_input
        assert "--language-only" in engine._serve_command(9000)
        assert not srv._engine_accepts_images(engine)

    def test_vision_checkpoint_loads_the_vision_tower(self, tmp_path):
        engine = _vision_engine(tmp_path)

        assert engine.accepts_image_input
        assert "--language-only" not in engine._serve_command(9000)
        assert srv._engine_accepts_images(engine)

    def test_chat_request_keeps_image_parts_for_splash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            srv, "get_engine_pool", lambda: SimpleNamespace(get_entry=lambda _: None)
        )
        engine = _vision_engine(tmp_path)
        request = ChatCompletionRequest(
            model="Qwen3.8-27B-4bit",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is this?"}, IMAGE],
                }
            ],
        )

        extracted = srv._extract_chat_messages(request, engine, None, None)

        parts = extracted.messages[0]["content"]
        assert isinstance(parts, list)
        assert [p["type"] for p in parts] == ["text", "image_url"]

    async def test_image_parts_reach_splash(self, tmp_path):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200, content=_sse({"choices": [{"delta": {"content": "A cat"}}]})
            )

        _vision_engine(tmp_path)
        engine = _engine_with_transport(tmp_path, handler)
        message = {"role": "user", "content": [{"type": "text", "text": "?"}, IMAGE]}

        outputs = [o async for o in engine.stream_chat([message])]

        assert outputs[-1].text == "A cat"
        assert seen["body"]["messages"] == [message]

    async def test_audio_and_video_are_refused_before_splash(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("the request must not reach Splash")

        engine = _engine_with_transport(tmp_path, handler)
        for part in (
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}},
            {"type": "input_audio", "input_audio": {"data": "AA", "format": "wav"}},
        ):
            message = {"role": "user", "content": [part]}
            with pytest.raises(InvalidRequestError, match="text and image input only"):
                [o async for o in engine.stream_chat([message])]

    def test_token_count_fallback_skips_image_payloads(self, tmp_path):
        engine = _vision_engine(tmp_path)

        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise ValueError("no template")

            def encode(self, text, **kwargs):
                return list(text)

        engine._tokenizer = Tokenizer()
        message = {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}, IMAGE],
        }

        assert engine.count_chat_tokens([message]) == len("hello")

    async def test_responses_input_image_reaches_splash_as_image_url(self, tmp_path):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200, content=_sse({"choices": [{"delta": {"content": "ok"}}]})
            )

        _vision_engine(tmp_path)
        engine = _engine_with_transport(tmp_path, handler)
        url = "data:image/png;base64,AAAA"
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "?"},
                {"type": "input_image", "image_url": url, "detail": "auto"},
            ],
        }

        [o async for o in engine.stream_chat([message])]

        assert seen["body"]["messages"][0]["content"] == [
            {"type": "text", "text": "?"},
            {"type": "image_url", "image_url": {"url": url, "detail": "auto"}},
        ]


# -- later system messages (#4697) -------------------------------------------------


class TestLaterSystem:
    @pytest.mark.parametrize(
        ("later_system", "expected"),
        [
            ("native", True),
            ("patched", True),
            ("unsupported", False),
            ({"default": "patched", "tool_use": "native"}, None),
        ],
    )
    async def test_status_sets_the_tokenizer_capability(
        self, tmp_path, later_system, expected
    ):
        status = {"chat_template": {"later_system": later_system}}
        engine = _engine_with_transport(
            tmp_path, lambda request: httpx.Response(200, json=status)
        )
        engine._tokenizer = SimpleNamespace()

        await engine._refresh_status()

        assert engine._tokenizer._omlx_supports_mid_system_messages is expected

    def test_token_count_folds_later_system_messages(self, tmp_path):
        class StrictTemplate:
            def apply_chat_template(self, messages, **kwargs):
                if any(m["role"] == "system" for m in messages[1:]):
                    raise ValueError("System message must be at the beginning.")
                return list("".join(f"<{m['role']}>{m['content']}" for m in messages))

        engine = _engine(tmp_path)
        engine._tokenizer = StrictTemplate()
        messages = [
            {"role": "system", "content": "lead"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "note"},
        ]

        assert engine.count_chat_tokens(messages) == len("<system>lead\n\nnote<user>hi")
