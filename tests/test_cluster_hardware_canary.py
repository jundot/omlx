# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the opt-in Apple-silicon hardware canary."""

from unittest.mock import Mock

import pytest

from omlx.cluster import hardware_canary


def test_hardware_canary_composes_all_five_checks(monkeypatch):
    monkeypatch.setattr(hardware_canary, "_require_metal", lambda: object())
    worker = Mock(return_value={"ok": True})
    collective = Mock(return_value={"ok": True})
    pipeline = Mock(return_value={"ok": True})
    vision = Mock(return_value={"ok": True})
    crash = Mock(return_value={"ok": True})

    result = hardware_canary.run_local_hardware_canary(
        timeout=12,
        allocation_mib=64,
        worker_runner=worker,
        collective_runner=collective,
        pipeline_runner=pipeline,
        vision_runner=vision,
        crash_runner=crash,
    )

    assert result["ok"] is True
    assert set(result["checks"]) == {
        "worker",
        "collective",
        "pipeline",
        "vision_prefill",
        "crash_recovery",
    }
    worker.assert_called_once_with(timeout=5.0)
    collective.assert_called_once_with(timeout=12)
    pipeline.assert_called_once_with(timeout=12)
    vision.assert_called_once_with()
    crash.assert_called_once_with(allocation_mib=64, timeout=12)


def test_hardware_canary_rejects_unsafe_allocation_before_running_checks(monkeypatch):
    require_metal = Mock()
    monkeypatch.setattr(hardware_canary, "_require_metal", require_metal)

    with pytest.raises(ValueError, match="between 16 and 512"):
        hardware_canary.run_local_hardware_canary(allocation_mib=15)

    require_metal.assert_not_called()


@pytest.mark.parametrize("allocation_mib", [0, 15, 513])
def test_crash_canary_rejects_unsafe_allocation_sizes(monkeypatch, allocation_mib):
    monkeypatch.setattr(hardware_canary, "_require_metal", lambda: object())

    with pytest.raises(ValueError, match="between 16 and 512"):
        hardware_canary._crash_recovery_canary(
            allocation_mib=allocation_mib,
            timeout=1,
        )


def test_canary_deployment_is_small_and_loopback_only():
    deployment = hardware_canary._canary_deployment()

    assert deployment.world_size == 2
    assert {host.ssh for host in deployment.hosts} == {"127.0.0.1"}
    assert (
        max(item.planned_weight_bytes for item in deployment.assignments) < 16 * 1024**2
    )
