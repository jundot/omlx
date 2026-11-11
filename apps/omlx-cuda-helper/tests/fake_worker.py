# SPDX-License-Identifier: Apache-2.0
"""Runs a stage worker that knows the fake family (tests only)."""

import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent)]

from omlx_cuda_helper import families, worker  # noqa: E402

families.FAMILIES["fake"] = families.Family(contract=7, module="fake_family")
sys.exit(worker.main())
