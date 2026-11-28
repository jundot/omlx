"""Bootstrap checks for the Cluster Assistant package.

These do not test product behavior; they prove the assistant package is
importable and documented before the election implementation starts.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_assistant_package_imports() -> None:
    from omlx.cluster import assistant

    assert assistant.__doc__
    assert "coordinator election" in assistant.__doc__.lower()
