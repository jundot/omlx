"""Attentive correction must match AngelSpec's own module, not a look-alike.

The oracle runs in two processes on purpose: torch and mlx cannot share one
interpreter here (their numpy builds disagree). The torch side imports
``angelspec.models.draft.dsv41_attentive_correction`` from the reference checkout
and writes parameters, inputs and its output to an .npz; the MLX side rebuilds
the same weights and compares slot by slot.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
REPO = HERE.parent.parent
REFERENCE = Path("/tmp/angelspec")
TORCH_VENV = Path(os.environ.get("TMPDIR", "/tmp")) / "torch-parity311"


def _mlx_python() -> str:
    return sys.executable


@pytest.mark.skipif(not (REFERENCE / "angelspec").exists(), reason="AngelSpec reference checkout is absent")
@pytest.mark.skipif(not (TORCH_VENV / "bin" / "python").exists(), reason="torch venv for the oracle is absent")
def test_attentive_correction_matches_angelspec(tmp_path):
    npz = tmp_path / "oracle.npz"
    producer = subprocess.run(
        [str(TORCH_VENV / "bin" / "python"), str(HERE / "attentive_correction_oracle_torch.py"), "--out", str(npz)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "", "PYTHONHOME": ""},
    )
    assert producer.returncode == 0, producer.stderr[-2000:]
    assert npz.exists()

    compare = subprocess.run(
        [_mlx_python(), str(HERE / "attentive_correction_oracle_mlx.py"), "--npz", str(npz)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONHOME": ""},
    )
    assert compare.returncode == 0, compare.stderr[-2000:]
    report = eval(compare.stdout.strip().splitlines()[-1])
    assert report["max_abs_error"] < 1e-5, report
