"""Packaging invariants for the privileged (SMJobBless) helper.

SMJobBless has strict, easy-to-break requirements (ServiceManagement.h):

* the helper's Info.plist is read from ``__TEXT,__info_plist`` and must carry
  ``CFBundleInfoDictionaryVersion`` 6.0, a ``CFBundleIdentifier`` equal to the
  tool's file name and exactly one ``SMAuthorizedClients`` requirement;
* the launchd plist is read from ``__TEXT,__launchd_plist`` — not from
  ``Contents/Library/LaunchDaemons`` — and carries only its ``Label``, because
  SMJobBless fills in ``ProgramArguments`` itself;
* the tool lives at ``Contents/Library/LaunchServices/<label>`` and the label is
  also the key in the app's ``SMPrivilegedExecutables`` dictionary.

These tests run offline and need no certificate: they pin the plists, the build
script and the Swift constants to each other. The actual bless (one admin
prompt) still needs a signed app and a click in the Performance screen.

The signing team is **not** in the repository. Both requirements carry the
``$(OMLX_TEAM_ID)`` marker; Xcode substitutes it in the app's Info.plist and
``build.sh`` substitutes it in the helper plists. The team-dependent assertions
read ``OMLX_TEAM_ID`` (or a built stage) and skip when neither is available, so
a contributor without an Apple team can still run this suite.
"""

import os
import plistlib
import re
import subprocess
from pathlib import Path

import pytest

MAC_APP = Path(__file__).resolve().parents[2]
REPO = MAC_APP.parents[1]

LABEL = "app.omlx.helper"
TEAM_ENV = "OMLX_TEAM_ID"
TEAM_MARKER = "$(OMLX_TEAM_ID)"
HELPER_BIN = f"Contents/Library/LaunchServices/{LABEL}"
REQUIREMENT_TEAM = re.compile(r'certificate leaf\[subject\.OU\] = "([^"]*)"')


def _plist(path: Path) -> dict:
    with path.open("rb") as handle:
        return plistlib.load(handle)


def _user_requirement() -> str:
    """The helper requirement as checked by the system before blessing."""
    return _plist(MAC_APP / "Resources" / "Info.plist")["SMPrivilegedExecutables"][
        LABEL
    ]


def _client_requirement() -> str:
    """The app requirement the installer helper evaluates against the app."""
    clients = _plist(MAC_APP / "Helper" / "Info.plist")["SMAuthorizedClients"]
    assert isinstance(clients, list) and len(clients) == 1
    return clients[0]


def _render_requirement(template: str, team: str) -> str:
    """Mirror build.sh: replace the team marker with a concrete team id."""
    return template.replace(TEAM_MARKER, team)


def _built_stage_team() -> str | None:
    """Read the team out of an already-built stage, if one exists."""
    for relative in (
        "build/Stage/oMLX.app",
        "build/SlimStage/oMLX.app",
        "build/Build/Products/Release/oMLX.app",
        "build/Build/Products/Debug/oMLX.app",
    ):
        info = MAC_APP / relative / "Contents" / "Info.plist"
        if not info.exists():
            continue
        try:
            requirement = _plist(info)["SMPrivilegedExecutables"][LABEL]
        except (KeyError, plistlib.InvalidFileException):
            continue
        match = REQUIREMENT_TEAM.search(requirement)
        if match and match.group(1):
            return match.group(1)
    return None


def _team() -> str:
    team = os.environ.get(TEAM_ENV, "").strip()
    if not team:
        team = _built_stage_team() or ""
    if not team:
        pytest.skip(
            f"set {TEAM_ENV}=<apple team id> (or build a signed stage) to check "
            "the SMJobBless team"
        )
    return team


def _helper_swift() -> str:
    return (MAC_APP / "Helper" / "main.swift").read_text()


def _performance_vm_swift() -> str:
    return (
        MAC_APP / "Sources/AppView/ViewModels/PerformanceScreenVM.swift"
    ).read_text()


def _build_script() -> str:
    return (MAC_APP / "Scripts" / "build.sh").read_text()


def _package_script() -> str:
    return (MAC_APP / "Scripts" / "package_dmg.sh").read_text()


def test_requirements_carry_the_team_marker_not_a_committed_team():
    """The repository carries no signing identity; both sides use the marker."""
    for requirement in (_user_requirement(), _client_requirement()):
        assert TEAM_MARKER in requirement, requirement
        match = REQUIREMENT_TEAM.search(requirement)
        assert match, requirement
        # The marker is the only allowed value: a literal team id here would
        # make every other contributor override a file under version control.
        assert match.group(1) == TEAM_MARKER


def test_app_and_helper_requirements_name_the_same_team():
    team = _team()
    privileged = _render_requirement(_user_requirement(), team)
    clients = _render_requirement(_client_requirement(), team)

    assert list(_plist(MAC_APP / "Resources" / "Info.plist")["SMPrivilegedExecutables"]) == [
        LABEL
    ]
    assert f'identifier "{LABEL}"' in privileged
    assert f'"{team}"' in privileged
    assert 'identifier "app.omlx"' in clients
    assert f'"{team}"' in clients

    # Same requirement body on both sides: only the identifier differs.
    assert privileged.replace(f'"{LABEL}"', '"X"') == clients.replace(
        '"app.omlx"', '"X"'
    ), "app and helper must pin the same trust structure"

    # SMJobBless rejects the tool without this exact version, and the
    # identifier must equal the tool's file name.
    helper_info = _plist(MAC_APP / "Helper" / "Info.plist")
    assert helper_info["CFBundleInfoDictionaryVersion"] == "6.0"
    assert helper_info["CFBundleIdentifier"] == LABEL


def test_rendered_requirements_are_valid_code_requirements(tmp_path):
    """The substitution must leave a string the requirement parser accepts."""
    csreq = Path("/usr/bin/csreq")
    if not csreq.exists():
        pytest.skip("csreq is not available to validate requirement syntax")

    team = _team()
    for name, requirement in (
        ("helper", _render_requirement(_user_requirement(), team)),
        ("app", _render_requirement(_client_requirement(), team)),
    ):
        path = tmp_path / f"{name}.requirement"
        path.write_text(requirement + "\n")
        result = subprocess.run(
            [str(csreq), "-v", "-r", str(path), "-t"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{name}: {result.stderr or result.stdout}"
        assert team in result.stdout


# Developer ID Certification Authority and Developer ID Application leaf.
DEVELOPER_ID_FIELDS = (
    "certificate 1[field.1.2.840.113635.100.6.2.6]",
    "certificate leaf[field.1.2.840.113635.100.6.1.13]",
)


def test_requirements_pin_developer_id_not_any_team_certificate():
    """A plain Apple Development certificate shares the team OU, so the
    requirement must also pin the Developer ID CA and leaf markers — those two
    extensions are what an Apple Development certificate does not carry."""
    for requirement in (_user_requirement(), _client_requirement()):
        for field in DEVELOPER_ID_FIELDS:
            assert field in requirement
        assert "anchor apple generic" in requirement
        # The team alone is not the requirement: something must stand between
        # the anchor and the OU clause, or any team certificate would match.
        assert requirement.count("certificate") >= 3


def test_bless_plist_declares_the_mach_service_but_no_program():
    bless = _plist(MAC_APP / "Helper" / f"{LABEL}.bless.plist")
    assert bless["Label"] == LABEL
    # Without MachServices launchd never creates the bootstrap name the helper's
    # NSXPCListener checks in with, and the app's XPC call cannot connect.
    assert bless["MachServices"] == {LABEL: True}
    assert bless["RunAtLoad"] is False
    # Apple's contract: SMJobBless owns Program/ProgramArguments.
    assert "Program" not in bless
    assert "ProgramArguments" not in bless


def test_bundled_launchdaemon_plist_targets_the_bundle_program():
    daemon = _plist(MAC_APP / "Helper" / f"{LABEL}.plist")
    assert daemon["Label"] == LABEL
    # The SMAppService path must not point at /Library/PrivilegedHelperTools:
    # that layout belongs to SMJobBless, which never installs it there.
    assert daemon["BundleProgram"] == HELPER_BIN
    assert daemon["MachServices"] == {LABEL: True}
    assert "ProgramArguments" not in daemon
    assert "Program" not in daemon


def test_build_script_embeds_both_plist_sections():
    script = _build_script()
    assert "__info_plist" in script
    assert "__launchd_plist" in script
    assert f"{LABEL}.bless.plist" in script
    assert f'cp "$RENDERED_HELPER_PLISTS/{LABEL}.plist"' in script
    assert "Contents/Library/LaunchDaemons" in script


def test_build_script_renders_the_team_into_the_embedded_plists():
    """The helper plists bypass Xcode, so build.sh must substitute the team."""
    script = _build_script()
    assert 'OMLX_TEAM_ID="${OMLX_TEAM_ID:-}"' in script
    # A 10-character Apple team id: the value is pasted into a sed script and
    # into a code requirement, so anything else is rejected up front.
    assert "[A-Z0-9]{10}" in script
    assert "$(OMLX_TEAM_ID)" in script
    # The embedded sections must be the rendered copies, never the sources.
    assert '-Xlinker "$RENDERED_HELPER_PLISTS/Info.plist"' in script
    assert '-Xlinker "$RENDERED_HELPER_PLISTS/app.omlx.helper.bless.plist"' in script
    assert '-Xlinker "$HELPER_DIR/Info.plist"' not in script


def test_swift_identity_matches_the_plists_and_the_ledger():
    swift = _performance_vm_swift()
    assert f'helperID = "{LABEL}"' in swift
    assert f'installedPath = "/Library/PrivilegedHelperTools/{LABEL}"' in swift

    version = re.search(r"helperBundleVersion = (\d+)", swift)
    assert version, "PrivilegedHelper.helperBundleVersion is missing"
    helper_info = _plist(MAC_APP / "Helper" / "Info.plist")
    assert int(helper_info["CFBundleVersion"]) == int(version.group(1))


def test_helper_runs_one_fixed_command_without_a_shell():
    source = _helper_swift()
    assert '"/usr/sbin/sysctl"' in source
    assert "iogpu.wired_limit_mb=" in source
    # No shell: the validated integer is an argument, never a command string.
    assert "do shell script" not in source
    assert "NSAppleScript" not in source
    assert '"/bin/sh"' not in source


def test_wired_limit_range_is_identical_on_both_sides():
    bounds = re.compile(r"mb > 0, mb <= ([\d_]+)")
    helper = bounds.search(_helper_swift())
    app = bounds.search(_performance_vm_swift())
    assert helper and app, "both sides must validate the wired limit"
    assert helper.group(1) == app.group(1)
    assert int(helper.group(1).replace("_", "")) == 1_048_576


def test_package_script_signs_helper_then_app_then_image():
    script = _package_script()
    helper_sign = script.index('--sign "$SIGN_IDENTITY" "$HELPER"')
    app_sign = script.index('echo "signing $APP_PATH"')
    dmg_sign = script.index('--sign "$SIGN_IDENTITY" "$DMG"')
    assert helper_sign < app_sign < dmg_sign
    assert "--options runtime" in script
    assert "--timestamp" in script
    # Credentials come from the environment; nothing is read from a file and no
    # secret is embedded. The example identity in the usage comment is a
    # placeholder, not a real team.
    assert 'SIGN_IDENTITY="${SIGN_IDENTITY:-}"' in script
    assert 'NOTARY_PROFILE="${NOTARY_PROFILE:-}"' in script
    assert "app-specific-password" not in script
    assert ".p8" not in script


def test_package_script_refuses_a_stage_built_for_another_team():
    """Signing with an identity whose team the staged requirement does not name
    would fail at bless time, after the user approved the admin dialog."""
    script = _package_script()
    assert "SMPrivilegedExecutables" in script
    assert "OMLX_TEAM_ID" in script
    assert "IDENTITY_TEAM" in script


@pytest.mark.parametrize(
    "relative",
    [
        "Helper/Info.plist",
        f"Helper/{LABEL}.bless.plist",
        f"Helper/{LABEL}.plist",
        "Resources/Info.plist",
    ],
)
def test_plists_parse(relative):
    assert _plist(MAC_APP / relative)