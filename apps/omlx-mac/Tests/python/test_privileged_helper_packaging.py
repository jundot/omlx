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
"""

import plistlib
import re
from pathlib import Path

import pytest

MAC_APP = Path(__file__).resolve().parents[2]
REPO = MAC_APP.parents[1]

LABEL = "app.omlx.helper"
TEAM = "YB894QRG4W"
HELPER_BIN = f"Contents/Library/LaunchServices/{LABEL}"


def _plist(path: Path) -> dict:
    with path.open("rb") as handle:
        return plistlib.load(handle)


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


def test_app_and_helper_requirements_name_the_same_team():
    app_info = _plist(MAC_APP / "Resources" / "Info.plist")
    helper_info = _plist(MAC_APP / "Helper" / "Info.plist")

    privileged = app_info["SMPrivilegedExecutables"]
    assert list(privileged) == [LABEL]
    assert f'identifier "{LABEL}"' in privileged[LABEL]
    assert f'"{TEAM}"' in privileged[LABEL]

    # SMJobBless rejects the tool without this exact version, and the
    # identifier must equal the tool's file name.
    assert helper_info["CFBundleInfoDictionaryVersion"] == "6.0"
    assert helper_info["CFBundleIdentifier"] == LABEL

    clients = helper_info["SMAuthorizedClients"]
    assert isinstance(clients, list) and len(clients) == 1
    assert 'identifier "app.omlx"' in clients[0]
    assert f'"{TEAM}"' in clients[0]


# Developer ID Certification Authority and Developer ID Application leaf.
DEVELOPER_ID_FIELDS = (
    "certificate 1[field.1.2.840.113635.100.6.2.6]",
    "certificate leaf[field.1.2.840.113635.100.6.1.13]",
)


def test_requirements_pin_developer_id_not_any_team_certificate():
    """A plain Apple Development certificate shares the team OU, so the
    requirement must also pin the Developer ID CA and leaf markers."""
    privileged = _plist(MAC_APP / "Resources" / "Info.plist")["SMPrivilegedExecutables"]
    clients = _plist(MAC_APP / "Helper" / "Info.plist")["SMAuthorizedClients"]

    for requirement in [privileged[LABEL], clients[0]]:
        for field in DEVELOPER_ID_FIELDS:
            assert field in requirement
        assert "anchor apple generic" in requirement


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
    assert f'cp "$HELPER_DIR/{LABEL}.plist"' in script
    assert "Contents/Library/LaunchDaemons" in script


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
    helper_sign = script.index(f"{LABEL}\"")
    app_sign = script.index('echo "signing $APP_PATH"')
    dmg_sign = script.index('--sign "$SIGN_IDENTITY" "$DMG"')
    assert helper_sign < app_sign < dmg_sign
    assert "--options runtime" in script
    assert "--timestamp" in script
    # Credentials come from the environment; nothing is read from a file and no
    # secret is embedded. The team ID in the usage comment is not a secret.
    assert 'SIGN_IDENTITY="${SIGN_IDENTITY:-}"' in script
    assert 'NOTARY_PROFILE="${NOTARY_PROFILE:-}"' in script
    assert "app-specific-password" not in script
    assert ".p8" not in script


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