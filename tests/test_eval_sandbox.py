# SPDX-License-Identifier: Apache-2.0
"""Code benchmarks must not use preexec_fn (unsafe in the threaded server)."""

import resource
import subprocess

import pytest

from omlx.eval import humaneval, livecodebench, mbpp
from omlx.eval.sandbox import limited_python_command


def _run(script, tmp_path, stdin=""):
    path = tmp_path / "script.py"
    path.write_text(script)
    return subprocess.run(
        limited_python_command(str(path), 512 * 1024 * 1024, 7),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_child_applies_cpu_limit(tmp_path):
    result = _run(
        "import resource\nprint(resource.getrlimit(resource.RLIMIT_CPU)[0])\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "7"
    # The parent is untouched.
    assert resource.getrlimit(resource.RLIMIT_CPU)[0] != 7


def test_script_runs_as_main_with_stdin(tmp_path):
    result = _run(
        "import sys\nif __name__ == '__main__':\n    print(sys.stdin.read().upper())\n",
        tmp_path,
        stdin="abc",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ABC"


def test_failure_reports_script_line(tmp_path):
    result = _run("x = 1\nassert x == 2\n", tmp_path)
    assert result.returncode != 0
    assert 'script.py", line 2' in result.stderr
    assert "AssertionError" in result.stderr


@pytest.mark.parametrize("module", [humaneval, mbpp, livecodebench])
def test_benchmarks_do_not_use_preexec_fn(module, monkeypatch):
    calls = []
    real_run = subprocess.run

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", spy)
    if module is livecodebench:
        stdout, ok, err = module._execute_code("print(input())", "hi\n")
        assert (stdout.strip(), ok) == ("hi", True), err
    elif module is humaneval:
        ok, err = module._execute_with_tests(
            "def f():\n    return 1\n",
            "def check(c):\n    assert c() == 1\n",
            "f",
        )
        assert ok, err
    else:
        ok, err = module._execute_with_tests(
            "def f():\n    return 1\n", ["assert f() == 1"]
        )
        assert ok, err
    assert calls and all("preexec_fn" not in kw for kw in calls)
