"""Bootstrap checks for the Cluster Assistant workspace.

These do not test product behavior. They prove that the project skeleton and
the assistant package are wired correctly before the election implementation
starts.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_project_handoff_files_exist() -> None:
    assert (ROOT / "AGENTS.md").is_file()
    assert (ROOT / "docs" / "cluster-assistant" / "PROJECT-BRIEF.md").is_file()
    assert (ROOT / "docs" / "cluster-assistant" / "README.md").is_file()


def test_assistant_package_imports() -> None:
    from omlx.cluster import assistant

    assert assistant.__doc__
    assert "coordinator election" in assistant.__doc__.lower()
