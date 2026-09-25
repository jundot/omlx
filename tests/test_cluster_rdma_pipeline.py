# SPDX-License-Identifier: Apache-2.0
"""Real MLX ring ranks decode through RDMA stage edges and match a single-process run."""

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
@pytest.mark.parametrize("ranks", [2, 3])
def test_pipeline_tokens_match_with_every_edge_and_the_tokens_on_rdma(ranks):
    links = [
        LoopbackLink(request_bytes=64 * 1024, reply_bytes=64 * 1024)
        for _ in range(ranks - 1)
    ]
    try:
        edges = [
            {
                "name": link.name,
                "socket": link.socket_path,
                "mailbox": link.mailbox_path,
            }
            for link in links
        ]
        environment = {
            **os.environ,
            "RDMA_TEST_LINKS": json.dumps(edges),
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
                str(ranks),
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
        for link in links:
            link.close()
    results = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith('{"rank"')
    ]
    assert len(results) == ranks, completed.stdout + completed.stderr
    for result in results:
        assert result["stage_links_active"] and result["matches"], result
        assert result["sampling_rank_only"] and result["prefill_overlap"], result
        assert result["token_relay"]["active"], result
        # No decode step used the ring for its tokens.
        assert result["ring_sums"] == 0, result
    # Every generated token crossed every edge, so each link carried at least that many replies.
    assert all(link.replies >= _NEW_TOKENS for link in links)
