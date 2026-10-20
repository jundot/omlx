# SPDX-License-Identifier: Apache-2.0
"""Two real MLX ring ranks decode through an RDMA stage edge and match a single-process run."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from rdma_loopback import LoopbackLink

pytestmark = pytest.mark.integration
_TESTS = Path(__file__).resolve().parent
_NEW_TOKENS = 12


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.skipif(
    shutil.which("mlx.launch", path=os.path.dirname(sys.executable)) is None,
    reason="mlx.launch is not installed",
)
def test_pipeline_tokens_match_with_the_stage_edge_on_rdma():
    link = LoopbackLink(request_bytes=64 * 1024, reply_bytes=64 * 1024)
    try:
        environment = {
            **os.environ,
            "RDMA_TEST_LINK": link.name,
            "RDMA_TEST_SOCKET": link.socket_path,
            "RDMA_TEST_MAILBOX": link.mailbox_path,
            "RDMA_TEST_TOKENS": str(_NEW_TOKENS),
            "PYTHONPATH": os.pathsep.join(
                [str(_TESTS), os.environ.get("PYTHONPATH", "")]
            ),
        }
        launcher = os.path.join(os.path.dirname(sys.executable), "mlx.launch")
        completed = subprocess.run(
            [
                launcher,
                "-n",
                "2",
                "--backend",
                "ring",
                "--starting-port",
                str(_free_port()),
                sys.executable,
                str(_TESTS / "rdma_pipeline_worker.py"),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=240,
        )
    finally:
        link.close()
    results = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith('{"rank"')
    ]
    assert len(results) == 2, completed.stdout + completed.stderr
    for result in results:
        assert result["stage_links_active"] and result["matches"], result
        assert result["sampling_rank_only"] and result["prefill_overlap"], result
    # Every generated token crossed the stage edge, so the link carried at least that many replies.
    assert link.replies >= _NEW_TOKENS
