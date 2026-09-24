# SPDX-License-Identifier: Apache-2.0
"""Detect installed git-pinned dependencies that drifted from omlx's pins.

pip treats a ``pkg @ git+...@SHA`` requirement as satisfied whenever the
installed version number matches, even when the installed commit differs.
``pip install -e .`` after a ``git pull`` therefore keeps whatever commit
was already present for any pin that moved without a version bump, and the
resulting skew produces failures that look like omlx bugs (stale model
implementations shadowing vendored compat modules, fixes that "landed" but
never arrived). Detect confirmed drift at startup and log exactly how to
fix it.

Kept as a leaf module so the comparison logic stays unit-testable without
the CLI surface.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Full 40-hex commit at the end of the URL (fragment stripped first). Pins
# by branch or tag have no stable commit to compare against, so they are
# not checked.
_PINNED_COMMIT_RE = re.compile(r"@([0-9a-f]{40})$")


@dataclass(frozen=True)
class PinDrift:
    """One git-pinned dependency whose installed commit differs from the pin."""

    name: str
    # Original requirement string, reusable verbatim as a pip argument.
    requirement: str
    pinned_commit: str
    installed_commit: str


def find_git_pin_drift(
    requirements: Iterable[str],
    read_direct_url: Callable[[str], str | None],
) -> list[PinDrift]:
    """Compare git-pinned requirements against installed direct_url.json data.

    ``read_direct_url`` maps a distribution name to the text of its
    ``direct_url.json``, or None when the package or the file is missing.

    Environment markers are deliberately not evaluated: a marker only
    decides whether pip installs the package (e.g. the audio extra). If the
    package is present it came from some pin and is worth checking; if it
    is absent the None lookup skips it.

    Only a confirmed vcs commit mismatch is reported. Anything unverifiable
    is skipped so index installs (no direct_url.json) and Homebrew resource
    builds (archive_info, not vcs_info) never produce false positives.
    """
    from packaging.requirements import InvalidRequirement, Requirement

    drift: list[PinDrift] = []
    for raw in requirements:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        url = req.url or ""
        if not url.startswith("git+"):
            continue
        pinned_match = _PINNED_COMMIT_RE.search(url.split("#", 1)[0])
        if pinned_match is None:
            continue
        pinned = pinned_match.group(1)

        text = read_direct_url(req.name)
        if text is None:
            continue
        try:
            direct_url = json.loads(text)
        except (TypeError, ValueError):
            continue
        if not isinstance(direct_url, dict):
            continue
        vcs_info = direct_url.get("vcs_info")
        if not isinstance(vcs_info, dict):
            continue
        installed = str(vcs_info.get("commit_id") or "")
        if installed and installed.lower() != pinned.lower():
            drift.append(
                PinDrift(
                    name=req.name,
                    requirement=raw,
                    pinned_commit=pinned,
                    installed_commit=installed,
                )
            )
    return drift


def _read_installed_direct_url(name: str) -> str | None:
    from importlib import metadata

    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    # read_text returns None when the file does not exist (index installs).
    return dist.read_text("direct_url.json")


def log_git_pin_drift(dist_name: str = "omlx") -> list[PinDrift]:
    """Log one warning per drifted git pin.

    Never raises: this is a startup diagnostic and must not take the
    server down, whatever the metadata on this machine looks like.
    """
    try:
        from importlib import metadata

        requirements = metadata.requires(dist_name) or []
        drift = find_git_pin_drift(requirements, _read_installed_direct_url)
    except Exception as exc:  # noqa: BLE001
        logger.debug("git pin drift check failed: %s", exc)
        return []
    for item in drift:
        logger.warning(
            "%s is installed at commit %.12s but omlx pins %.12s. pip skips "
            "moved git pins when the version number is unchanged, so "
            "'pip install -e .' does not correct this. Fix with: "
            'pip install --force-reinstall --no-deps "%s"',
            item.name,
            item.installed_commit,
            item.pinned_commit,
            item.requirement,
        )
    return drift
