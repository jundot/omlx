# SPDX-License-Identifier: Apache-2.0
"""Tests for fatal process-exit helpers."""

import subprocess
import sys
from unittest.mock import patch

from omlx.utils.fatal import FATAL_EXIT_CODE, fatal_exit


def test_fatal_exit_dumps_traceback_and_exits():
    with (
        patch("omlx.utils.fatal.faulthandler.dump_traceback") as dump_traceback,
        patch("omlx.utils.fatal.os._exit") as exit_process,
    ):
        fatal_exit("fatal test")

    dump_traceback.assert_called_once_with(all_threads=True)
    exit_process.assert_called_once_with(FATAL_EXIT_CODE)


def test_gpu_submissions_ignored_exits_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from omlx.utils.fatal import exit_if_gpu_submissions_ignored; "
            "exit_if_gpu_submissions_ignored(RuntimeError("
            "'kIOGPUCommandBufferCallbackErrorSubmissionsIgnored'))",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == FATAL_EXIT_CODE
    assert "Metal GPU submissions disabled" in result.stderr
