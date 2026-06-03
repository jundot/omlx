"""GitHub release selection helpers used by the admin panel's release picker.

Kept as a leaf module so the PEP 440 / channel-filter logic stays
unit-testable without pulling in the admin route surface.
"""

from typing import Any

from packaging.version import InvalidVersion, Version

UPDATE_CHANNEL_STABLE = "stable"
UPDATE_CHANNEL_RELEASE_CANDIDATE = "release_candidate"
UPDATE_CHANNEL_DEV = "dev"
UPDATE_CHANNELS = {
    UPDATE_CHANNEL_STABLE,
    UPDATE_CHANNEL_RELEASE_CANDIDATE,
    UPDATE_CHANNEL_DEV,
}


def normalize_update_channel(channel: str | None) -> str:
    """Normalize app update channel names shared with the Swift app."""
    if channel == "release_candidate" or channel == "beta":
        return UPDATE_CHANNEL_RELEASE_CANDIDATE
    if channel == "dev" or channel == "nightly":
        return UPDATE_CHANNEL_DEV
    return UPDATE_CHANNEL_STABLE


def select_latest_stable_release(
    releases: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the highest stable release from a GitHub /releases response.

    Don't trust the GitHub `prerelease` flag alone. Historically dev/rc tags
    have been published with that flag unset, which makes /releases/latest
    return them as if they were stable. Filter via PEP 440 too, and skip
    drafts and unparseable tags.
    """
    return select_latest_release(releases, channel=UPDATE_CHANNEL_STABLE)


def select_latest_release(
    releases: list[dict[str, Any]],
    channel: str | None = UPDATE_CHANNEL_STABLE,
) -> dict[str, Any] | None:
    """Pick the highest release allowed by the update channel.

    Stable accepts final releases only. Release Candidate accepts final and rc
    tags, but not dev/alpha/beta tags. Dev accepts any parseable non-draft tag.
    """
    normalized_channel = normalize_update_channel(channel)
    best_release: dict[str, Any] | None = None
    best_version: Version | None = None

    for release in releases:
        if release.get("draft"):
            continue
        tag = release.get("tag_name")
        if not tag:
            continue
        try:
            version = Version(tag.lstrip("v"))
        except InvalidVersion:
            continue

        if not _release_allowed_for_channel(release, version, normalized_channel):
            continue

        if best_version is None or version > best_version:
            best_version = version
            best_release = release

    return best_release


def _release_allowed_for_channel(
    release: dict[str, Any],
    version: Version,
    channel: str,
) -> bool:
    if channel == UPDATE_CHANNEL_DEV:
        return True
    if channel == UPDATE_CHANNEL_RELEASE_CANDIDATE:
        return not version.is_devrelease and not _is_alpha_beta(version)
    return not release.get("prerelease") and not version.is_prerelease


def _is_alpha_beta(version: Version) -> bool:
    pre = version.pre
    return pre is not None and pre[0] in {"a", "b"}
