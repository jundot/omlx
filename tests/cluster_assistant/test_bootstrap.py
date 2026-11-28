"""Bootstrap checks for the Cluster Assistant workspace.

These do not test product behavior. They prove that the project skeleton and
the assistant package are wired correctly before the election implementation
starts.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_project_handoff_files_exist() -> None:
    assert (ROOT / "CLUSTER-ASSISTANT.md").is_file()


def test_assistant_package_imports() -> None:
    from omlx.cluster import assistant

    assert assistant.__doc__
    assert "coordinator election" in assistant.__doc__.lower()
