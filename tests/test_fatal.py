# SPDX-License-Identifier: Apache-2.0
"""Tests for fatal process-exit helpers."""

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
