import json
import struct
from pathlib import Path

from omlx.model_discovery import estimate_model_size


def test_exl3_size_reserves_optional_mtp(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "expert_quant": {"format": "exl3"}})
    )
    header = json.dumps(
        {
            "weight": {"data_offsets": [0, 100]},
            "language_model.mtp.weight": {"data_offsets": [100, 150]},
        }
    ).encode()
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", len(header)) + header + bytes(150))
    assert estimate_model_size(tmp_path) == int(150 * 1.05)
    assert estimate_model_size(tmp_path, include_mtp=False) == int(100 * 1.05)
    (tmp_path / "config.json").write_text('{"model_type": "other"}')
    assert estimate_model_size(tmp_path) == int(shard.stat().st_size * 1.05)


def test_exl3_size_invalid_header_preserves_conservative_estimate(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "expert_quant": {"format": "exl3"}})
    )
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", 100000000) + bytes(100))
    assert estimate_model_size(tmp_path) == int(shard.stat().st_size * 1.05)


def test_runtime_admission_and_settle_follow_native_mtp_setting(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch
    from omlx.engine_pool import EngineEntry
    from tests.test_engine_pool import _make_pool

    test_exl3_size_reserves_optional_mtp(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "expert_quant": {"format": "exl3"}})
    )
    entry = EngineEntry(
        model_id="packed",
        model_path=str(tmp_path),
        model_type="vlm",
        engine_type="vlm",
        config_model_type="qwen4_exp",
        estimated_size=int(150 * 1.05),
    )
    pool = _make_pool()
    with patch.object(
        pool, "_qwen4_ple_offload_status", return_value=(False, False, None)
    ):
        assert pool._entry_runtime_resident_size(
            entry, SimpleNamespace(mtp_enabled=False)
        ) == int(100 * 1.05)
        assert pool._entry_runtime_resident_size(
            entry, SimpleNamespace(mtp_enabled=True)
        ) == int(150 * 1.05)
