# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-family (dense hybrid) pipeline parallelism.

Two defects kept a ``qwen3_5`` checkpoint (e.g. Qwen3.8-27B-oQ4e-mtp) from being
split across two Macs:

1. the planner's vision guard refused a text-only mlx-lm load (#3662);
2. the MTP patch replaced ``Qwen3_5TextModel.__call__`` with a single-host body that
   iterates ``None`` layers and has no stage exchange (#3518).
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from omlx.cluster import planner

# -- 1. planner: vision guard -------------------------------------------------------


def test_qwen35_dense_with_vision_config_is_pipelinable(monkeypatch):
    monkeypatch.setattr(
        planner, "_model_source", lambda mt: "class M(PipelineMixin): ..."
    )
    config = {"model_type": "qwen3_5", "vision_config": {"depth": 27}}
    assert planner._supports_pipeline(config) is True


def test_other_vision_checkpoints_keep_the_vision_guard(monkeypatch):
    monkeypatch.setattr(
        planner, "_model_source", lambda mt: "def pipeline(self, group): ..."
    )
    config = {"model_type": "qwen3_5_moe", "vision_config": {"depth": 24}}
    assert planner._supports_pipeline(config) is False


# -- 2. the forward: two real ranks over the loopback ring ---------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_pipelined_qwen35_forward_matches_single_process(tmp_path):
    pytest.importorskip("mlx_lm.models.qwen3_5")
    hostfile = tmp_path / "ring.json"
    hostfile.write_text(
        json.dumps([[f"127.0.0.1:{_free_port()}"], [f"127.0.0.1:{_free_port()}"]])
    )
    worker = Path(__file__).with_name("qwen35_pipeline_worker.py")
    procs = []
    for rank in (0, 1):
        env = dict(os.environ, MLX_RANK=str(rank), MLX_HOSTFILE=str(hostfile))
        procs.append(
            subprocess.Popen(
                [sys.executable, str(worker)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    outputs = []
    for proc in procs:
        try:
            out, _ = proc.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            for p in procs:
                p.kill()
            raise
        outputs.append((proc.returncode, out))
    for rank, (code, out) in enumerate(outputs):
        assert code == 0, f"rank {rank} failed:\n{out[-2000:]}"
        assert '"ok": true' in out
