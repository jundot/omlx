# SPDX-License-Identifier: Apache-2.0
"""The planning surface must run without a GPU library.

``families/__init__`` documents that a family's ``__init__`` needs no GPU
library and that ``planner()`` returns the family's GPU-free part, and the
daemon runs ``planner(model_type).plan(...)`` in-process. An ``mlx`` import
reached from there therefore takes session start down on a box whose MLX is
missing or broken. The rest of this suite runs where MLX *is* installed, so a
top-level ``import mlx.core`` is invisible to it: this file is the only test
that can see one.

Each probe blocks ``mlx`` with a meta-path finder that raises
``ModuleNotFoundError`` — what CPython raises once every finder has failed —
rather than the ``sys.modules["mlx"] = None`` trick, which models a different
failure. The finder **records every attempted name**, and the probes assert the
record is empty afterwards: the property under test is "no ``mlx`` import was
attempted", not "no ``mlx`` import succeeded", because a fallback that swallows
an ``ImportError`` passes the second and still depends on failing.

Every probe carries a positive control, and it asserts the blocked *name*, not
just that something raised: a control that accepts any ``ModuleNotFoundError``
would still print "blocked" after ``oqh`` were renamed, and the proof that the
blocker reaches transitive imports would disappear silently.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Recorded by tests/fixtures/record_prefill_contract.py in the oMLX tree, the
# same fixture test_qwen3_5_moe.py replays.
PLAN_FIXTURE = (
    Path(__file__).parents[3]
    / "tests"
    / "fixtures"
    / "prefill_contract"
    / "qwen3_5_moe"
    / "model"
)

BLOCKER = """
import importlib.abc
import sys

ATTEMPTS = []


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            ATTEMPTS.append(fullname)
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


sys.meta_path.insert(0, Block())


def assert_blocked():
    import importlib

    try:
        importlib.import_module("mlx.core")
    except ModuleNotFoundError as exc:
        print("control-blocked:", exc.name)
        ATTEMPTS.clear()
        return
    print("control: FAILED, mlx imported while blocked")
    raise SystemExit(3)


def assert_no_attempts():
    if ATTEMPTS:
        print("mlx-attempts:", ",".join(ATTEMPTS))
        raise SystemExit(5)
    print("mlx-attempts: none")
"""

PLANNER_PROBE = BLOCKER + """
assert_blocked()

import importlib

families = importlib.import_module("omlx_cuda_helper.families")
print("families-imported:", len(families.FAMILIES))

for model_type in sorted(families.FAMILIES):
    module = families.planner(model_type)
    print(f"planner({model_type}): {module.__name__}")

assert_no_attempts()
"""

PLAN_PROBE = BLOCKER + """
assert_blocked()

import sys
from pathlib import Path

from omlx_cuda_helper import families

model_dir = Path(sys.argv[1])
shared, sizes = families.planner("qwen3_5_moe").plan(str(model_dir))
print("plan-shared:", shared)
print("plan-sizes:", sizes)

assert_no_attempts()
"""

OQH_PROBE = BLOCKER + """
assert_blocked()

import importlib

try:
    importlib.import_module("omlx_cuda_helper.oqh")
except ModuleNotFoundError as exc:
    print("oqh-blocked-by:", exc.name)
else:
    print("oqh: FAILED, imported while mlx was blocked")
    raise SystemExit(4)
"""


def _blocked_name(stdout: str, label: str) -> str:
    """The module name the probe reported, so a test can assert on it."""
    for line in stdout.splitlines():
        if line.startswith(f"{label}: "):
            return line.split(": ", 1)[1].strip()
    raise AssertionError(f"no {label} line in:\n{stdout}")


def _assert_control(stdout: str) -> None:
    name = _blocked_name(stdout, "control-blocked")
    # CPython reports the top-level name, so "mlx" is the expected answer and a
    # submodule name is acceptable; anything else means the block stopped
    # reaching mlx while the probe still called itself blocked.
    assert name == "mlx" or name.startswith(
        "mlx."
    ), f"control blocked {name!r}, not mlx"


def _run(
    tmp_path: Path, name: str, source: str, *args: str
) -> subprocess.CompletedProcess:
    script = tmp_path / f"{name}.py"
    script.write_text(source)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PACKAGE_ROOT)
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=120,
    )


def test_the_blocker_blocks_a_module_that_needs_mlx(tmp_path: Path) -> None:
    """The positive control: oqh legitimately needs mlx, and by that name."""
    result = _run(tmp_path, "oqh", OQH_PROBE)

    _assert_control(result.stdout)
    # The name, not merely "something raised": a renamed module would otherwise
    # satisfy this control while proving nothing.
    assert _blocked_name(result.stdout, "oqh-blocked-by") == _blocked_name(
        result.stdout, "control-blocked"
    ), (result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_families_planner_imports_without_mlx(tmp_path: Path) -> None:
    result = _run(tmp_path, "planner", PLANNER_PROBE)

    _assert_control(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "mlx-attempts: none" in result.stdout
    # Driving the registry rather than a hardcoded pair: a new family that
    # imports mlx from its __init__ is covered the day it is registered.
    for model_type in (
        "deepseek_v41",
        "glm5_next",
        "inkling2",
        "minimax_m3_vl",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen4_exp",
    ):
        assert (
            f"planner({model_type}): omlx_cuda_helper.families.{model_type}"
            in result.stdout
        )


@pytest.mark.skipif(
    not PLAN_FIXTURE.exists(),
    reason="the oMLX contract fixture is not next to this tree",
)
def test_plan_runs_without_mlx(tmp_path: Path) -> None:
    """The daemon's own call, not just the import: ``plan()`` on a real fixture."""
    result = _run(tmp_path, "plan", PLAN_PROBE, str(PLAN_FIXTURE))

    _assert_control(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    shared = int(_blocked_name(result.stdout, "plan-shared"))
    assert shared > 0
    assert "mlx-attempts: none" in result.stdout
