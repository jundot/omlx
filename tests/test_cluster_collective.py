# SPDX-License-Identifier: Apache-2.0
"""Tests for the local MLX collective diagnostic."""

import json
import subprocess

import pytest

from omlx.cluster.collective import (
    CollectiveSmokeError,
    _run_local_minimax_decode_smoke,
    run_local_collective_smoke,
    run_local_pipeline_smoke,
)


def test_local_collective_smoke_validates_both_ranks():
    def runner(argv, *, timeout):
        assert "--backend" in argv
        assert "ring" in argv
        assert "--repeat-hosts" in argv
        assert timeout == 4.0
        records = [
            {
                "type": "collective_result",
                "backend": "ring",
                "rank": rank,
                "size": 2,
                "input": rank + 1,
                "sum": 3,
            }
            for rank in (0, 1)
        ]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="\n".join(json.dumps(record) for record in records),
            stderr="",
        )

    result = run_local_collective_smoke(
        timeout=4.0,
        runner=runner,
        starting_port=43000,
    )

    assert result["ok"] is True
    assert result["backend"] == "ring"
    assert result["loopback_only"] is True
    assert [record["rank"] for record in result["ranks"]] == [0, 1]


def test_local_collective_smoke_rejects_missing_rank():
    def runner(argv, *, timeout):
        record = {
            "type": "collective_result",
            "rank": 0,
            "size": 2,
            "sum": 3,
        }
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(record),
            stderr="rank 1 failed",
        )

    with pytest.raises(CollectiveSmokeError, match="each rank"):
        run_local_collective_smoke(
            runner=runner,
            starting_port=43000,
        )


def test_local_collective_smoke_rejects_invalid_port():
    with pytest.raises(ValueError, match="starting_port"):
        run_local_collective_smoke(starting_port=65535)


def test_local_pipeline_smoke_validates_unequal_nemotron_ranks():
    def runner(argv, *, timeout):
        assert "omlx.cluster.pipeline_smoke_worker" in argv
        assert timeout == 7.0
        records = [
            {
                "type": "pipeline_result",
                "model_type": "nemotron_h",
                "rank": rank,
                "size": 2,
                "start_layer": 2 if rank == 0 else 0,
                "end_layer": 4 if rank == 0 else 2,
                "local_layer_count": 2,
                "local_cache_count": 1,
                "output_shape": [1, 3, 32],
                "checksum": 1.25,
            }
            for rank in (0, 1)
        ]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="\n".join(json.dumps(record) for record in records),
            stderr="",
        )

    result = run_local_pipeline_smoke(
        timeout=7.0,
        runner=runner,
        starting_port=43000,
    )

    assert result["ok"] is True
    assert result["model_type"] == "nemotron_h"
    assert [record["start_layer"] for record in result["ranks"]] == [2, 0]


def test_local_pipeline_smoke_rejects_divergent_outputs():
    def runner(argv, *, timeout):
        records = [
            {
                "type": "pipeline_result",
                "model_type": "nemotron_h",
                "rank": rank,
                "size": 2,
                "local_layer_count": 2,
                "local_cache_count": 1,
                "output_shape": [1, 3, 32],
                "checksum": float(rank),
            }
            for rank in (0, 1)
        ]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="\n".join(json.dumps(record) for record in records),
            stderr="",
        )

    with pytest.raises(CollectiveSmokeError, match="checksums differ"):
        run_local_pipeline_smoke(runner=runner, starting_port=43000)


def test_local_minimax_decode_smoke_validates_real_rank_roles():
    def runner(argv, *, timeout):
        assert "omlx.cluster.minimax_decode_smoke_worker" in argv
        assert timeout == 9.0
        records = [
            {
                "type": "minimax_decode_result",
                "model_type": "minimax_m3_vl",
                "rank": rank,
                "size": 2,
                "steps": 3,
                "skip_logits": rank != 0,
                "local_layer_count": 2,
                "local_cache_count": 2,
                "logprobs_width": 128,
                "next_token": 17,
            }
            for rank in (0, 1)
        ]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="\n".join(json.dumps(record) for record in records),
            stderr="",
        )

    result = _run_local_minimax_decode_smoke(
        timeout=9.0,
        runner=runner,
        starting_port=43000,
    )

    assert result["ok"] is True
    assert result["steps"] == 3
    assert result["ranks"][0]["skip_logits"] is False
    assert result["ranks"][1]["skip_logits"] is True
