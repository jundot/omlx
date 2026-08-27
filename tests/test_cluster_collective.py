# SPDX-License-Identifier: Apache-2.0
"""Tests for the local MLX collective diagnostic."""

import json
import os
import subprocess

import pytest

from omlx.cluster.collective import (
    CollectiveSmokeError,
    _run_local_minimax_decode_smoke,
    run_local_collective_smoke,
    run_local_generation_wedge_smoke,
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


def test_generation_wedge_smoke_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        run_local_generation_wedge_smoke(mode="exploded", state_dir=tmp_path)


def test_generation_wedge_smoke_rejects_invalid_port(tmp_path):
    with pytest.raises(ValueError, match="starting_port"):
        run_local_generation_wedge_smoke(
            mode="caught", state_dir=tmp_path, starting_port=65535
        )


def test_generation_wedge_smoke_reports_a_clean_stop_as_not_wedged(tmp_path):
    def runner(argv, *, timeout):
        assert "omlx.cluster.generation_wedge_worker" in argv
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    result = run_local_generation_wedge_smoke(
        mode="caught", state_dir=tmp_path, runner=runner, starting_port=43000
    )

    assert result == {
        "mode": "caught",
        "wedged": False,
        "ok": True,
        "elapsed_seconds": pytest.approx(0.0, abs=5.0),
        "returncode": 0,
    }


def test_generation_wedge_smoke_reports_a_launcher_timeout_as_wedged(tmp_path):
    def runner(argv, *, timeout):
        raise CollectiveSmokeError("MLX collective did not finish within 6.00s")

    result = run_local_generation_wedge_smoke(
        mode="killed", state_dir=tmp_path, runner=runner, starting_port=43000
    )

    assert result["wedged"] is True
    assert result["ok"] is False
    assert result["returncode"] is None


def test_generation_wedge_smoke_configures_the_worker_through_the_environment(
    tmp_path,
):
    """The worker reads its config from the environment, not argv (simpler
    than threading extra args through mlx's own launcher CLI) -- confirm the
    right variables reach the runner's environment at call time and are
    fully restored afterward, since this mutates process-global state."""

    os.environ["OMLX_WEDGE_MODE"] = "leftover-from-a-previous-run"
    seen = {}

    def runner(argv, *, timeout):
        seen.update(
            {key: os.environ.get(key) for key in os.environ if key.startswith("OMLX_WEDGE_")}
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    run_local_generation_wedge_smoke(
        mode="killed",
        state_dir=tmp_path,
        deployment_id="probe-run",
        steps=7,
        fatal_step=3,
        fatal_rank=0,
        runner=runner,
        starting_port=43000,
    )

    assert seen == {
        "OMLX_WEDGE_MODE": "killed",
        "OMLX_WEDGE_STEPS": "7",
        "OMLX_WEDGE_FATAL_STEP": "3",
        "OMLX_WEDGE_FATAL_RANK": "0",
        "OMLX_WEDGE_STATE_DIR": str(tmp_path),
        "OMLX_WEDGE_DEPLOYMENT_ID": "probe-run",
    }
    # The pre-existing (unrelated) value is restored, not left as "killed".
    assert os.environ.pop("OMLX_WEDGE_MODE") == "leftover-from-a-previous-run"


def test_generation_wedge_smoke_leaves_no_env_vars_behind_when_none_preexisted(
    tmp_path,
):
    for key in list(os.environ):
        if key.startswith("OMLX_WEDGE_"):
            del os.environ[key]

    def runner(argv, *, timeout):
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    run_local_generation_wedge_smoke(
        mode="caught", state_dir=tmp_path, runner=runner, starting_port=43000
    )

    assert not any(key.startswith("OMLX_WEDGE_") for key in os.environ)
