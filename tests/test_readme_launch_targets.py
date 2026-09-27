# SPDX-License-Identifier: Apache-2.0
"""Each translation's launch-target lists stay in step.

``README.md`` names the tools ``omlx launch`` can start twice: in the welcome
sentence and in the admin-dashboard sentence. Every translation has to name
the same tools in the same two sentences — a target that only lands in
English is a documentation bug the translated editions never see.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READMES = sorted(ROOT.glob("README*.md"))
# The welcome sentence names every target but Pi (it is dashboard-only).
WELCOME = (
    "OpenClaw",
    "OpenCode",
    "Codex",
    "Hermes Agent",
    "Copilot",
    "DeepSeek Harness",
)
DASHBOARD = WELCOME + ("Pi",)


def test_every_readme_names_every_launch_target():
    assert len(READMES) >= 5, "the translated READMEs are gone"
    for path in READMES:
        rows = path.read_text(encoding="utf-8").splitlines()
        lines = [row for row in rows if "OpenClaw" in row]
        assert len(lines) == 2, f"{path.name}: expected welcome and dashboard lists"
        for want, line in zip((WELCOME, DASHBOARD), lines):
            for target in want:
                assert target in line, f"{path.name} drops {target!r}"
        assert "Pi" not in lines[0], f"{path.name} lists Pi in its welcome sentence"
