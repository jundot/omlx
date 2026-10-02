# SPDX-License-Identifier: Apache-2.0
"""Multi-row Lightning MTP must decide identically on every rank of a pipeline.

With 2+ rows in flight the draft depth, standard-vs-MTP mode and parking come from
``BatchPolicy``, whose only non-deterministic input is each rank's own measured
cycle time. Rank 1's MTP cycles being 1.3x slower made rank 0 park MTP while rank 1
did not; the ranks then verified windows of different length and the group died in
the collective. ``distributed_sync.install_batch_policy`` feeds every rank rank 0's
cycle time.
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


def _run_two_ranks(tmp_path, *, sync: bool, slow_rank: int = 1):
    hostfile = tmp_path / "ring.json"
    hostfile.write_text(
        json.dumps([[f"127.0.0.1:{_free_port()}"], [f"127.0.0.1:{_free_port()}"]])
    )
    worker = Path(__file__).with_name("batch_policy_worker.py")
    procs = []
    for rank in (0, 1):
        env = dict(
            os.environ,
            MLX_RANK=str(rank),
            MLX_HOSTFILE=str(hostfile),
            SYNC="1" if sync else "0",
            SKEW_RANK=str(slow_rank),
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
        results[rank] = json.loads(line)["decisions"]
    return results


def test_a_1_3x_slower_rank_parks_mtp_on_its_own_without_the_sync(tmp_path):
    pytest.importorskip("mlx.core")
    r = _run_two_ranks(tmp_path, sync=False)
    assert r[0] != r[1]
    # rank 0 keeps drafting while rank 1 parks: the split that killed both ranks
    assert ["park", 0] not in r[0]
    assert ["park", 0] in r[1]


def test_batch_policy_decisions_are_identical_on_every_rank_with_the_sync(tmp_path):
    pytest.importorskip("mlx.core")
    r = _run_two_ranks(tmp_path, sync=True)
    assert r[0] == r[1]
    assert any(mode == "mtp" for mode, _ in r[0])
    assert ["park", 0] not in r[0]


def test_the_sync_follows_rank_zero_when_rank_zero_is_the_slow_one(tmp_path):
    pytest.importorskip("mlx.core")
    r = _run_two_ranks(tmp_path, sync=True, slow_rank=0)
    assert r[0] == r[1]
    # rank 0's clock decides: it parks MTP, so the fast rank 1 parks with it
    assert ["park", 0] in r[1]
