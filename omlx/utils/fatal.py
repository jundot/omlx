# SPDX-License-Identifier: Apache-2.0
"""Fatal process-exit helpers."""

from __future__ import annotations

import faulthandler
import logging
import os
from typing import NoReturn

logger = logging.getLogger(__name__)

FATAL_EXIT_CODE = 70
FATAL_TEARDOWN_TIMEOUT_S = 60.0


def fatal_exit(reason: str, exit_code: int = FATAL_EXIT_CODE) -> NoReturn:
    """Terminate immediately so an external supervisor can restart cleanly."""
    logger.critical(
        "%s; exiting process so the supervisor can restart with a clean state",
        reason,
    )
    try:
        faulthandler.dump_traceback(all_threads=True)
    except Exception:
        logger.exception("Failed to dump thread tracebacks before fatal exit")
    os._exit(exit_code)
