"""GitHub release selection helpers used by the menubar app's update check.

Kept separate from app.py so the logic is unit-testable without pulling in
the PyObjC stack.
"""

from typing import Any

from packaging.version import InvalidVersion, Version


def select_latest_stable_release(
    releases: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the highest stable release from a GitHub /releases response.

    Don't trust the GitHub `prerelease` flag alone. Historically dev/rc tags
    have been published with that flag unset, which makes /releases/latest
    return them as if they were stable. Filter via PEP 440 too, and skip
    drafts and unparseable tags.
    """
    best_release: dict[str, Any] | None = None
    best_version: Version | None = None

    for release in releases:
        if release.get("draft") or release.get("prerelease"):
            continue
        tag = release.get("tag_name")
        if not tag:
            continue
        try:
            version = Version(tag.lstrip("v"))
        except InvalidVersion:
            continue
        if version.is_prerelease:
            continue
        if best_version is None or version > best_version:
            best_version = version
            best_release = release

    return best_release
