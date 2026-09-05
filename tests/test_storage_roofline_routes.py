# SPDX-License-Identifier: Apache-2.0
"""Route-level tests for the storage-roofline bench endpoints."""

import json
import struct
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import routes as admin_routes
from omlx.admin import storage_bench
from omlx.utils.storage_roofline import StorageMeasurement


class _FakeEntry:
    def __init__(self, model_type="llm", model_path="/fake"):
        self.model_type = model_type
        self.model_path = model_path


class _FakePool:
    def __init__(self, model_path="/fake"):
        self._entries = {"local-model": _FakeEntry(model_path=model_path)}

    def get_entry(self, model_id):
        return self._entries.get(model_id)


def _write_fake_model(tmp_path: Path) -> str:
    mdir = tmp_path / "fake-moe"
    mdir.mkdir()
    (mdir / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "num_hidden_layers": 2,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
    }))
    header, off, blob = {}, 0, b""
    for layer in range(2):
        for exp in range(2):
            for w in ("w1", "w2", "w3"):
                k = f"model.layers.{layer}.mlp.experts.{exp}.{w}"
                header[k] = {"dtype": "F32", "shape": [4, 8],
                             "data_offsets": [off, off + 128]}
                off += 128
                blob += bytes(128)
    hdr_json = json.dumps(header).encode()
    with open(mdir / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hdr_json)))
        f.write(hdr_json)
        f.write(blob)
    return str(mdir)


@pytest.fixture
def client(tmp_path, monkeypatch):
    pool = _FakePool(model_path=_write_fake_model(tmp_path))
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)

    async def _fake_require_admin():
        return True

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = _fake_require_admin
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_storage_jobs():
    storage_bench._jobs.clear()
    yield
    storage_bench._jobs.clear()


def _tiny_measurement() -> dict:
    return asdict(StorageMeasurement(
        volume_mount="/tmp", file_bytes=1024**3,
        seq_read_Bps=4_000_000_000.0, rand_read_Bps=3_000_000_000.0,
        rand_iops=1500.0, rand_lat_ms_p50=0.7, rand_lat_ms_p90=0.8,
        rand_lat_ms_p99=0.9, rand_lat_ms_max=1.0,
        samples=256, cache_clean=True, method="F_NOCACHE",
    ))


class TestStorageStart:
    def test_unknown_model_404(self, client):
        r = client.post("/admin/api/bench/storage/start",
                        json={"model_id": "nope"})
        assert r.status_code == 404

    def test_start_and_poll(self, client):
        with patch.object(storage_bench, "run_storage_benchmark",
                          lambda job: None):
            r = client.post("/admin/api/bench/storage/start",
                            json={"model_id": "local-model", "file_gb": 0.5})
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        got = client.get(f"/admin/api/bench/storage/{job_id}/results")
        assert got.status_code == 200
        assert got.json()["status"] == "running"

    def test_second_start_409_while_running(self, client):
        with patch.object(storage_bench, "run_storage_benchmark",
                          lambda job: None):
            r = client.post("/admin/api/bench/storage/start",
                            json={"model_id": "local-model"})
            assert r.status_code == 200
            r2 = client.post("/admin/api/bench/storage/start",
                             json={"model_id": "local-model"})
        assert r2.status_code == 409

    def test_throughput_active_409(self, client):
        with patch("omlx.admin.benchmark.get_active_run",
                   return_value=object()):
            r = client.post("/admin/api/bench/storage/start",
                            json={"model_id": "local-model"})
        assert r.status_code == 409

    def test_unknown_job_404(self, client):
        r = client.get("/admin/api/bench/storage/nope/results")
        assert r.status_code == 404


class TestStoragePredict:
    def test_no_measurement_404(self, client):
        # latest_saved_report() would find real reports on a dev machine;
        # isolate the no-measurement branch explicitly.
        with patch.object(storage_bench, "latest_measurement", return_value=None):
            r = client.get("/admin/api/bench/storage/predict",
                           params={"model_id": "local-model"})
        assert r.status_code == 404

    def test_predict_from_completed_job(self, client):
        job = storage_bench.StorageBenchJob(
            job_id="storage_test", status="completed",
            request=storage_bench.StorageBenchRequest(model_id="local-model"),
            target_dir="/tmp",
            report={"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "measurement": _tiny_measurement()},
        )
        storage_bench._jobs[job.job_id] = job
        r = client.get("/admin/api/bench/storage/predict",
                       params={"model_id": "local-model",
                               "tok_per_cycle": 1.5, "verify_mult": 2.3})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["profile"]["bytes_per_step"] == 2 * 384
        assert body["prediction"]["mtp_profitable"] is False
        assert "tok/cycle" in body["prediction"]["explanation"]

    def test_predict_unknown_model_404(self, client):
        r = client.get("/admin/api/bench/storage/predict",
                       params={"model_id": "nope"})
        assert r.status_code == 404


class TestStorageAutoParams:
    def test_auto_params_empty_when_nothing_derived(self, client):
        import omlx.utils.storage_roofline as sr

        # No auto file on disk (fresh tmp results dir) and no usable runs.
        with patch.object(sr, "load_auto_params", return_value=None), \
             patch.object(sr, "_bench_results_dirs", return_value=[]), \
             patch.object(sr, "derive_verify_mult", return_value=None), \
             patch.object(sr, "derive_tok_per_cycle", return_value=None), \
             patch.object(sr, "derive_bytes_per_token_base", return_value=None):
            r = client.get("/admin/api/bench/storage/auto-params",
                           params={"model_id": "local-model"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["available"] is False

    def test_auto_params_returns_derived(self, client):
        import omlx.utils.storage_roofline as sr

        auto = {
            "version": 1,
            "model_dir": "/fake",
            "derived_at": "2026-09-04T09:00:00",
            "tok_per_cycle": 1.79,
            "verify_byte_mult": 2.31,
            "bytes_per_token_base": 12345.6,
            "source": {},
        }
        with patch.object(sr, "load_auto_params", return_value=auto):
            r = client.get("/admin/api/bench/storage/auto-params",
                           params={"model_id": "local-model"})
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert body["tok_per_cycle"] == 1.79
        assert body["verify_byte_mult"] == 2.31

    def test_auto_params_unknown_model_404(self, client):
        r = client.get("/admin/api/bench/storage/auto-params",
                       params={"model_id": "nope"})
        assert r.status_code == 404

    def test_predict_uses_auto_when_params_omitted(self, client):
        import omlx.utils.storage_roofline as sr

        job = storage_bench.StorageBenchJob(
            job_id="storage_auto", status="completed",
            request=storage_bench.StorageBenchRequest(model_id="local-model"),
            target_dir="/tmp",
            report={"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "measurement": _tiny_measurement()},
        )
        storage_bench._jobs[job.job_id] = job
        auto = {
            "version": 1,
            "tok_per_cycle": 1.79,
            "verify_byte_mult": 2.31,
            "bytes_per_token_base": None,
            "source": {},
        }
        with patch.object(sr, "load_auto_params", return_value=auto):
            r = client.get("/admin/api/bench/storage/predict",
                           params={"model_id": "local-model"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["params_source"] == "auto"
        assert body["prediction"]["tok_per_cycle"] == 1.79
        assert body["prediction"]["verify_byte_mult"] == 2.31

    def test_predict_params_source_default_when_no_auto(self, client):
        import omlx.utils.storage_roofline as sr

        job = storage_bench.StorageBenchJob(
            job_id="storage_def", status="completed",
            request=storage_bench.StorageBenchRequest(model_id="local-model"),
            target_dir="/tmp",
            report={"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "measurement": _tiny_measurement()},
        )
        storage_bench._jobs[job.job_id] = job
        with patch.object(sr, "load_auto_params", return_value=None):
            r = client.get("/admin/api/bench/storage/predict",
                           params={"model_id": "local-model"})
        assert r.status_code == 200
        body = r.json()
        assert body["params_source"] == "default"
        assert body["prediction"]["tok_per_cycle"] == 1.0

    def test_predict_explicit_overrides_auto(self, client):
        import omlx.utils.storage_roofline as sr

        job = storage_bench.StorageBenchJob(
            job_id="storage_expl", status="completed",
            request=storage_bench.StorageBenchRequest(model_id="local-model"),
            target_dir="/tmp",
            report={"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "measurement": _tiny_measurement()},
        )
        storage_bench._jobs[job.job_id] = job
        auto = {"tok_per_cycle": 1.79, "verify_byte_mult": 2.31}
        with patch.object(sr, "load_auto_params", return_value=auto):
            r = client.get("/admin/api/bench/storage/predict",
                           params={"model_id": "local-model",
                                   "tok_per_cycle": 2.6, "verify_mult": 2.0})
        assert r.status_code == 200
        body = r.json()
        assert body["params_source"] == "explicit"
        assert body["prediction"]["tok_per_cycle"] == 2.6
        assert body["prediction"]["verify_byte_mult"] == 2.0
        # Auto rides along for inspection either way.
        assert body["params_auto"]["tok_per_cycle"] == 1.79

class TestStorageHistory:
    def test_history_empty_ok(self, client):
        import omlx.utils.storage_roofline as sr

        with patch.object(sr, "list_reports", return_value=[]):
            r = client.get("/admin/api/bench/storage/history")
        assert r.status_code == 200
        assert r.json() == {"reports": []}

    def test_history_lists_reports(self, client):
        import omlx.utils.storage_roofline as sr

        entries = [
            {"timestamp": "2026-09-04T10:00:00", "path": "a.json",
             "volume_media": "YSSDHB", "seq_read_GBps": 2.4,
             "rand_read_GBps": 3.0, "bytes_per_step_MB": 1292,
             "ceiling_base_tok_s": 2.27, "ceiling_effective_tok_s": None,
             "cache_clean": True},
            {"timestamp": "2026-09-01T10:00:00", "path": "b.json",
             "volume_media": "YSSDHB", "seq_read_GBps": 2.2,
             "rand_read_GBps": 2.8, "bytes_per_step_MB": 1292,
             "ceiling_base_tok_s": 2.11, "ceiling_effective_tok_s": None,
             "cache_clean": False},
        ]
        with patch.object(sr, "list_reports", return_value=entries):
            r = client.get("/admin/api/bench/storage/history")
        assert r.status_code == 200
        assert len(r.json()["reports"]) == 2
        assert r.json()["reports"][0]["rand_read_GBps"] == 3.0

    def test_list_reports_reads_disk(self, tmp_path, monkeypatch):
        import omlx.utils.storage_roofline as sr

        monkeypatch.setattr(sr, "_results_dir", lambda: tmp_path)
        # Two reports; newest mtime first.
        for i, rand in enumerate([1.0e9, 3.0e9]):
            rep = {
                "timestamp": f"2026-09-0{i + 1}T10:00:00",
                "volume": {"media_name": "SSD", "mount": "/"},
                "measurement": {"seq_read_Bps": 2e9, "rand_read_Bps": rand,
                                "cache_clean": True},
                "prediction": {"bytes_per_step": 1292 * 1048576,
                               "ceiling_base_tok_s": 2.3,
                               "ceiling_effective_tok_s": None},
            }
            import time as _t
            (tmp_path / f"rep{i}.json").write_text(json.dumps(rep))
            _t.sleep(0.05)
        out = sr.list_reports()
        assert len(out) == 2
        assert out[0]["rand_read_GBps"] == 3.0
        assert out[0]["bytes_per_step_MB"] == 1292
