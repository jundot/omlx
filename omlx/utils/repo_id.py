# SPDX-License-Identifier: Apache-2.0
"""Validation for HuggingFace / ModelScope style repository IDs.

Repo ids are joined onto the local model directory by the downloaders
(``model_dir / repo_id``), so each segment must be a single path component.
A bare "contains one slash" check accepts values like ``../..`` that escape
the model directory; ``validate_repo_id`` enforces per-segment rules instead.
"""

from __future__ import annotations

import re

# Mirrors huggingface_hub's repo id rules: each segment starts with an
# alphanumeric and may contain word characters, dots, and dashes.
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_MAX_SEGMENT_LEN = 96


def validate_repo_id(repo_id: str, label: str = "repository ID") -> str | None:
    """Return a human-readable error for an invalid repo id, None if valid.

    Accepts exactly two non-empty segments ("owner/model"), each a single
    safe path component (no ``.``, ``..``, separators, or leading dash).
    ``label`` names the id in the error message (e.g. "model ID").
    """
    repo_id = repo_id.strip()
    parts = repo_id.split("/")
    if len(parts) != 2:
        return f"Invalid {label}: '{repo_id}'. " f"Expected format: 'owner/model'"
    for part in parts:
        if not part or part in (".", ".."):
            return (
                f"Invalid {label}: '{repo_id}'. "
                f"Expected format: 'owner/model' with non-empty, "
                "non-relative segments"
            )
        if len(part) > _MAX_SEGMENT_LEN or not _SEGMENT_RE.match(part):
            return (
                f"Invalid {label}: '{repo_id}'. Segments may contain "
                "alphanumerics, dots, underscores, and dashes, and must "
                "start with an alphanumeric"
            )
    return None
