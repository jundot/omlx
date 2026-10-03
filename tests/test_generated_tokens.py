# SPDX-License-Identifier: Apache-2.0
"""tokens.json is the single source; the generated artifact must match it."""

import subprocess
import sys
from pathlib import Path


def test_generated_artifacts_match_the_token_file():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "omlx" / "admin" / "build_tokens.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
