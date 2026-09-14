# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest

from omlx.cluster.jaccl_lease import (
    JacclCommunicatorBusyError,
    acquire_jaccl_communicator_lease,
)


def test_jaccl_lease_is_exclusive_and_recoverable(tmp_path):
    first = acquire_jaccl_communicator_lease(
        deployment_id="first",
        state_dir=tmp_path,
    )
    try:
        with pytest.raises(JacclCommunicatorBusyError, match="deployment first"):
            acquire_jaccl_communicator_lease(
                deployment_id="second",
                state_dir=tmp_path,
            )
    finally:
        first.close()

    second = acquire_jaccl_communicator_lease(
        deployment_id="second",
        state_dir=tmp_path,
    )
    second.close()


def test_jaccl_lease_writes_bounded_private_diagnostics(tmp_path):
    lease = acquire_jaccl_communicator_lease(
        deployment_id="diagnostic",
        state_dir=tmp_path,
    )
    try:
        payload = json.loads(lease.path.read_text())
        assert payload["deployment_id"] == "diagnostic"
        assert payload["pid"] == os.getpid()
        assert lease.path.stat().st_mode & 0o777 == 0o600
        assert lease.path.stat().st_size < 4096
    finally:
        lease.close()
