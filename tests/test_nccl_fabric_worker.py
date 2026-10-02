# SPDX-License-Identifier: Apache-2.0
"""Fail-closed parsing and dual-rail pinning for the NCCL fabric probe."""

from __future__ import annotations

import argparse
import os

import pytest

from omlx.cluster.nccl_fabric_worker import (
    _configure_nccl_fabric_environment,
    _rank_device_lists,
)


def test_rank_device_lists_accept_two_dgx_spark_rails():
    assert _rank_device_lists(
        '[["enp1s0f1np1","enP2p1s0f1np1"],'
        '["enp1s0f1np1","enP2p1s0f1np1"]]'
    ) == (
        ("enp1s0f1np1", "enP2p1s0f1np1"),
        ("enp1s0f1np1", "enP2p1s0f1np1"),
    )


def test_rank_device_lists_reject_shell_syntax():
    with pytest.raises(argparse.ArgumentTypeError, match="invalid name"):
        _rank_device_lists('[["eth0;false"],["eth0"]]')


def test_nccl_environment_pins_socket_and_rdma_rails(monkeypatch):
    for key in ("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_DISABLE"):
        monkeypatch.delenv(key, raising=False)
    args = argparse.Namespace(
        interfaces=(
            ("enp1s0f1np1", "enP2p1s0f1np1"),
            ("other0", "other1"),
        ),
        rdma_devices=(
            ("rocep1s0f1", "roceP2p1s0f1"),
            ("other_hca0", "other_hca1"),
        ),
    )

    _configure_nccl_fabric_environment(args, 0)

    assert os.environ["NCCL_SOCKET_IFNAME"] == (
        "=enp1s0f1np1,enP2p1s0f1np1"
    )
    assert os.environ["NCCL_IB_HCA"] == (
        "=rocep1s0f1,roceP2p1s0f1"
    )
    assert os.environ["NCCL_IB_DISABLE"] == "0"
