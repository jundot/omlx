# SPDX-License-Identifier: Apache-2.0
"""The peer-discoverable interpreter shim (#2680).

``discover_remote_python_executable`` probes ``~/.omlx/bin/omlx-cluster-python``
before anything else, but nothing ever created that file.  A packaged-app peer
therefore failed every candidate and was reported as "worker runtime is not
installed" while the app sat in /Applications.  These tests pin the contract of
the writer that now ships it on every install mode.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

from omlx.cluster.worker_shim import CLUSTER_PYTHON_SHIM, ensure_cluster_python_shim


def test_shim_is_written_executable_at_the_probed_candidate_path(tmp_path):
    written = ensure_cluster_python_shim(home=tmp_path)

    assert written == tmp_path / ".omlx" / "bin" / "omlx-cluster-python"
    assert written.is_file()
    assert os.access(written, os.X_OK)
    # The discovery candidate list spells this exact path.
    assert CLUSTER_PYTHON_SHIM == "~/.omlx/bin/omlx-cluster-python"


def test_shim_reproduces_the_bundled_interpreter_environment(tmp_path):
    """The bundled app's env is what makes ``import omlx`` work at all."""

    written = ensure_cluster_python_shim(
        home=tmp_path,
        executable="/Applications/oMLX.app/Contents/Resources/Python/"
        "cpython-3.11/bin/python3.11",
        environ={
            "PYTHONHOME": "/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11",
            "PYTHONPATH": "/Applications/oMLX.app/Contents/Resources",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMLX_BASE_PATH": "/tmp/custom omlx",
        },
    )
    script = written.read_text(encoding="utf-8")

    assert script.startswith("#!/bin/sh\n")
    assert (
        "export PYTHONHOME=/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11"
        in script
    )
    assert "export PYTHONPATH=/Applications/oMLX.app/Contents/Resources" in script
    assert "export PYTHONDONTWRITEBYTECODE=1" in script
    # A path with a space must survive the shell, so this one is quoted.
    assert "export OMLX_BASE_PATH='/tmp/custom omlx'" in script
    assert script.rstrip().endswith(
        'exec /Applications/oMLX.app/Contents/Resources/Python/'
        'cpython-3.11/bin/python3.11 "$@"'
    )


def test_shim_forwards_arguments_to_the_interpreter_verbatim(tmp_path):
    """It must behave as an interpreter: both ``-c`` and ``-m`` reach python."""

    fake = tmp_path / "fake-python"
    fake.write_text(
        '#!/bin/sh\nprintf "HOME=%s\\nARGS=%s\\n" "$PYTHONHOME" "$*"\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)

    written = ensure_cluster_python_shim(
        home=tmp_path,
        executable=str(fake),
        environ={"PYTHONHOME": "/opt/py home"},
    )
    completed = subprocess.run(
        [str(written), "-m", "omlx.cli", "cluster", "status", "--json"],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    )

    assert completed.stdout.splitlines() == [
        "HOME=/opt/py home",
        "ARGS=-m omlx.cli cluster status --json",
    ]


def test_shim_omits_unset_interpreter_variables(tmp_path):
    """A pip/source install has no PYTHONHOME — exporting an empty one breaks it."""

    written = ensure_cluster_python_shim(
        home=tmp_path,
        executable="/usr/local/bin/python3.11",
        environ={},
    )
    script = written.read_text(encoding="utf-8")

    assert "PYTHONHOME" not in script
    assert "PYTHONPATH" not in script
    assert script.rstrip().endswith('exec /usr/local/bin/python3.11 "$@"')


def test_rewrite_is_atomic_and_skipped_when_unchanged(tmp_path):
    """A live SSH probe may be executing the shim; never truncate it in place."""

    first = ensure_cluster_python_shim(home=tmp_path, executable=sys.executable)
    stamp = first.stat().st_mtime_ns

    second = ensure_cluster_python_shim(home=tmp_path, executable=sys.executable)

    assert second == first
    assert second.stat().st_mtime_ns == stamp
    assert list(second.parent.iterdir()) == [second]


def test_writer_never_raises_when_the_home_directory_is_unwritable(tmp_path, caplog):
    """Shim installation is best effort — it must not break server startup.

    Best effort is not the same as silent: a peer that cannot be discovered
    later is far cheaper to diagnose with this line in the log.
    """

    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="omlx.cluster.worker_shim"):
        assert ensure_cluster_python_shim(home=blocked) is None

    assert any(
        CLUSTER_PYTHON_SHIM in record.message and record.levelno >= logging.WARNING
        for record in caplog.records
    ), caplog.text


def test_a_refused_executable_says_why(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG, logger="omlx.cluster.worker_shim"):
        assert ensure_cluster_python_shim(home=tmp_path, executable="python3") is None

    assert any("not absolute" in record.message for record in caplog.records), caplog.text


def test_a_successful_publish_records_what_it_pointed_at(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="omlx.cluster.worker_shim"):
        ensure_cluster_python_shim(home=tmp_path, executable="/usr/bin/python3")

    assert any("/usr/bin/python3" in record.message for record in caplog.records), (
        caplog.text
    )


def test_executable_that_is_not_an_absolute_path_is_refused(tmp_path):
    assert ensure_cluster_python_shim(home=tmp_path, executable="python3") is None
    assert not (tmp_path / ".omlx" / "bin").exists()


def test_shim_survives_a_stale_file_at_the_target_path(tmp_path):
    bin_dir = tmp_path / ".omlx" / "bin"
    bin_dir.mkdir(parents=True)
    stale = bin_dir / "omlx-cluster-python"
    stale.write_text("#!/bin/sh\nexec /gone/python \"$@\"\n", encoding="utf-8")
    stale.chmod(0o644)

    written = ensure_cluster_python_shim(home=tmp_path, executable="/usr/bin/python3")

    assert written == stale
    assert "/usr/bin/python3" in stale.read_text(encoding="utf-8")
    assert os.access(stale, os.X_OK)


def test_default_home_and_environment_come_from_the_running_server(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    written = ensure_cluster_python_shim()

    assert written == tmp_path / ".omlx" / "bin" / "omlx-cluster-python"
    assert sys.executable in written.read_text(encoding="utf-8")
