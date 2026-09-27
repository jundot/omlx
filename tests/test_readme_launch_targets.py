# SPDX-License-Identifier: Apache-2.0
"""The launch-target lists in the READMEs stay in step.

``README.md`` names the tools ``omlx launch`` can start in two places: the
welcome paragraph and the admin-dashboard paragraph. Each translation of
those two sentences has to name the same tools — a target that only lands
in English is a documentation bug the translated editions never see.

The two paragraphs are told apart by ``Pi``: the admin-dashboard list has
it, the welcome list does not, in English and in every translation.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READMES = sorted(ROOT.glob("README*.md"))


def _target_lines(path: Path) -> tuple[str, str]:
    """``(welcome line, dashboard line)`` — the two launch-target lists."""
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if "OpenClaw" in line
    ]
    assert len(lines) == 2, f"{path.name}: expected two target lists, found {len(lines)}"
    dashboard = [line for line in lines if "Pi" in line]
    welcome = [line for line in lines if "Pi" not in line]
    assert len(dashboard) == 1, f"{path.name}: expected exactly one list naming Pi"
    assert len(welcome) == 1, f"{path.name}: expected exactly one list without Pi"
    return welcome[0], dashboard[0]


def _split_names(text: str) -> list[str]:
    """The English list prose (``A, B, or C`` / ``A, B, and C``) as names."""
    text = text.replace(", or ", ", ").replace(", and ", ", ")
    return [name.strip() for name in text.split(",")]


def _english_targets() -> tuple[list[str], list[str]]:
    """The target names, parsed out of the English paragraphs."""
    welcome, dashboard = _target_lines(ROOT / "README.md")
    assert "To connect " in welcome, "the English welcome list moved"
    assert "Set up " in dashboard, "the English dashboard list moved"
    welcome_list = welcome.split("To connect ", 1)[1].split(", see ", 1)[0]
    dashboard_list = dashboard.split("Set up ", 1)[1].split(" directly from ", 1)[0]
    return _split_names(welcome_list), _split_names(dashboard_list)


def test_every_readme_names_every_launch_target():
    welcome_targets, dashboard_targets = _english_targets()
    # The parse has to keep finding a target, or this test checks nothing.
    assert "DeepSeek Harness" in welcome_targets
    assert "DeepSeek Harness" in dashboard_targets
    assert "Pi" in dashboard_targets and "Pi" not in welcome_targets

    assert len(READMES) >= 5, "the translated READMEs are gone"
    for path in READMES:
        welcome, dashboard = _target_lines(path)
        for target in welcome_targets:
            assert target in welcome, f"{path.name} drops {target!r} from its welcome list"
        for target in dashboard_targets:
            assert (
                target in dashboard
            ), f"{path.name} drops {target!r} from its dashboard list"
