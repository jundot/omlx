import json
import struct
from pathlib import Path
from omlx.model_discovery import estimate_model_size


def test_exl3_size_excludes_only_mtp(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "expert_quant": {"format": "exl3"}}))
    header = json.dumps({"weight": {"data_offsets": [0, 100]}, "language_model.mtp.weight": {"data_offsets": [100, 150]}}).encode()
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", len(header)) + header + bytes(150))
    assert estimate_model_size(tmp_path) == 105
    (tmp_path / "config.json").write_text('{"model_type": "other"}')
    assert estimate_model_size(tmp_path) == int(shard.stat().st_size * 1.05)


def test_exl3_size_invalid_header_preserves_conservative_estimate(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "expert_quant": {"format": "exl3"}}))
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", 100000000) + bytes(100))
    assert estimate_model_size(tmp_path) == int(shard.stat().st_size * 1.05)
