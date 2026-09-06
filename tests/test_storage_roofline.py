# SPDX-License-Identifier: Apache-2.0
"""Storage roofline: prediction math, header profiling, tiny measurement."""

import json
import struct
from pathlib import Path

from omlx.utils.storage_roofline import (
    MoEStepProfile,
    StorageMeasurement,
    build_report,
    measure_storage,
    moe_step_profile,
    predict_roofline,
    volume_info_for,
)


def _meas(rand_Bps: float = 3_000_000_000.0) -> StorageMeasurement:
    return StorageMeasurement(
        volume_mount="/tmp", file_bytes=1024**3,
        seq_read_Bps=4_000_000_000.0, rand_read_Bps=rand_Bps,
        rand_iops=1500.0, rand_lat_ms_p50=0.7, rand_lat_ms_p90=0.8,
        rand_lat_ms_p99=0.9, rand_lat_ms_max=1.0,
        samples=256, cache_clean=True, method="F_NOCACHE",
    )


def _prof(bytes_per_step: int = 1_000_000_000) -> MoEStepProfile:
    return MoEStepProfile(
        model_dir="/tmp/m", model_type="qwen4_exp_text", supported=True,
        num_moe_layers=48, routed_total_per_layer=512, top_k=10,
        bytes_per_step=bytes_per_step,
    )


def test_predict_base_ceiling_math():
    pred = predict_roofline(_prof(), _meas(), tok_per_cycle=1.0)
    assert pred.ceiling_base_tok_s == 3.0  # 3 GB/s / 1 GB per step
    assert not pred.mtp_profitable
    assert pred.margin_tok_per_cycle == 1.0 - 2.3


def test_predict_mtp_verdict_flips_on_ratio():
    # Same SSD, same model: only the byte ratio decides.
    lose = predict_roofline(_prof(), _meas(), tok_per_cycle=1.79, verify_byte_mult=2.3)
    assert not lose.mtp_profitable
    assert lose.ceiling_mtp_tok_s < lose.ceiling_base_tok_s
    win = predict_roofline(_prof(), _meas(), tok_per_cycle=2.6, verify_byte_mult=2.3)
    assert win.mtp_profitable
    assert win.ceiling_mtp_tok_s > win.ceiling_base_tok_s


def _write_fake_model(tmp_path: Path) -> Path:
    """2 layers x 2 routed experts, 3x [4,8] F32 tensors each (=384 B/expert)."""
    mdir = tmp_path / "fake-moe"
    mdir.mkdir()
    (mdir / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "num_hidden_layers": 2,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
    }))
    tensors: dict[str, bytes] = {}
    for layer in range(2):
        for exp in range(2):
            for w in ("w1", "w2", "w3"):
                tensors[f"model.layers.{layer}.mlp.experts.{exp}.{w}"] = bytes(128)
    header = {}
    off = 0
    blob = b""
    for k, data in tensors.items():
        header[k] = {"dtype": "F32", "shape": [4, 8], "data_offsets": [off, off + len(data)]}
        off += len(data)
        blob += data
    hdr_json = json.dumps(header).encode()
    with open(mdir / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hdr_json)))
        f.write(hdr_json)
        f.write(blob)
    return mdir


def test_profile_from_headers(tmp_path):
    prof = moe_step_profile(_write_fake_model(tmp_path))
    assert prof.supported, prof.reason
    assert prof.num_moe_layers == 2
    assert prof.routed_total_per_layer == 2
    assert prof.top_k == 1
    # 2 layers x top1 x 384 B routed-active per layer
    assert prof.bytes_per_step == 2 * 384


def test_profile_missing_topk_unsupported(tmp_path):
    mdir = _write_fake_model(tmp_path)
    cfg = json.loads((mdir / "config.json").read_text())
    del cfg["num_experts_per_tok"]
    (mdir / "config.json").write_text(json.dumps(cfg))
    prof = moe_step_profile(mdir)
    assert not prof.supported
    assert "num_experts_per_tok" in prof.reason


def test_measure_tiny_cold(tmp_path):
    meas = measure_storage(tmp_path, file_gb=0.002, read_mb=1, samples=8,
                           include_write=True)
    assert meas.samples == 8
    assert meas.seq_read_Bps > 0
    assert meas.rand_read_Bps > 0
    assert meas.rand_iops > 0
    assert meas.rand_lat_ms_max >= meas.rand_lat_ms_p50 > 0
    assert meas.write_Bps > 0
    # Scratch cleaned up.
    assert list(tmp_path.glob(".omlx_storage_roofline.bin")) == []


def test_volume_info_tmp(tmp_path):
    vol = volume_info_for(tmp_path)
    assert vol.total_bytes > 0
    assert vol.free_bytes > 0


def test_build_report_calibration():
    rep = build_report(volume_info_for("/tmp"), _meas(), _prof(),
                       predict_roofline(_prof(), _meas()),
                       measured_base_tok_s=1.5)
    assert rep["calibration"]["efficiency"] == 0.5  # 1.5 measured / 3.0 ceiling
