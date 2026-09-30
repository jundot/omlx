# SPDX-License-Identifier: Apache-2.0
"""Locate a macOS app bundle by the CFBundleIdentifier its Info.plist declares."""

from __future__ import annotations

import plistlib
from pathlib import Path


def find_mac_app(
    roots: tuple[Path, ...], names: tuple[str, ...], bundle_id: str
) -> Path | None:
    """First bundle under ``roots`` named ``names`` declaring ``bundle_id``.

    Matches on the identifier, not the folder name, so renamed bundles still
    resolve; an unreadable plist is skipped.
    """
    for root in roots:
        for name in names:
            bundle = root / name
            plist_path = bundle / "Contents" / "Info.plist"
            if not plist_path.is_file():
                continue
            try:
                with plist_path.open("rb") as f:
                    info = plistlib.load(f)
            except (OSError, plistlib.InvalidFileException):
                continue
            if info.get("CFBundleIdentifier") == bundle_id:
                return bundle
    return None
