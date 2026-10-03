# SPDX-License-Identifier: Apache-2.0
"""The MTP max draft depth is rank 0's on every rank of a pipeline.

``qwen3_5`` gets a depth ceiling of 4 where the M5 packed verify kernels exist and
3 elsewhere, chosen per host. A cluster mixing chips then disagrees on the length
of the first verify window and hangs. The cluster worker's install step adopts
rank 0's ceiling before the model is built.
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _depths_after_install(tmp_path, depths: str):
    hostfile = tmp_path / "ring.json"
    hostfile.write_text(
        json.dumps([[f"127.0.0.1:{_free_port()}"], [f"127.0.0.1:{_free_port()}"]])
    )
    worker = Path(__file__).with_name("mtp_depth_worker.py")
    procs = []
    for rank in (0, 1):
        env = dict(
            os.environ,
            MLX_RANK=str(rank),
            MLX_HOSTFILE=str(hostfile),
            DEPTHS=depths,
        )
        procs.append(
            subprocess.Popen(
                [sys.executable, str(worker)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    results = {}
    for rank, proc in enumerate(procs):
        try:
            out, _ = proc.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            for p in procs:
                p.kill()
            raise
        assert proc.returncode == 0, f"rank {rank} failed:\n{out[-2000:]}"
        line = [ln for ln in out.splitlines() if ln.startswith("{")][-1]
        results[rank] = json.loads(line)
    return results


@pytest.mark.parametrize("depths", ["4,3", "3,4"])
def test_every_rank_adopts_rank_zeros_max_depth(tmp_path, depths):
    pytest.importorskip("mlx.core")
    r = _depths_after_install(tmp_path, depths)
    rank_zero = int(depths.split(",")[0])
    assert r[0]["depth"] == r[1]["depth"] == rank_zero
    assert r[0]["fixed"] is r[1]["fixed"] is False


def test_a_single_rank_keeps_its_own_depth(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from omlx.patches import mlx_lm_mtp
    from omlx.patches.mlx_lm_mtp import distributed_sync

    group = type("Group", (), {"rank": lambda s: 0, "size": lambda s: 1})()
    monkeypatch.setattr(mx.distributed, "init", lambda *a, **k: group)
    monkeypatch.setattr(mlx_lm_mtp, "_MTP_DEPTH", 3)
    monkeypatch.setattr(mlx_lm_mtp, "_MTP_DEPTH_FIXED", False)
    assert distributed_sync.sync_max_depth() == (3, False)
