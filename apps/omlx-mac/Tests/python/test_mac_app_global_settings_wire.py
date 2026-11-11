# SPDX-License-Identifier: Apache-2.0
"""Wire contract between the macOS app and ``/admin/api/global-settings``."""

import re
from pathlib import Path

from omlx.admin.routes import GlobalSettingsRequest

PATCH_DTO = (
    Path(__file__).resolve().parents[2] / "Sources/Net/DTO/GlobalSettingsDTO.swift"
)
PATCH_MEMBER = re.compile(r"^\s+var ([A-Za-z][A-Za-z0-9]*): .* = nil$", re.M)


def test_every_patch_field_is_accepted_by_the_server():
    # GlobalSettingsRequest ignores unknown keys, so a stale app field returns
    # 200 and never persists. The patch encodes with .convertToSnakeCase.
    body = PATCH_DTO.read_text().split("struct GlobalSettingsPatch", 1)[1]
    members = PATCH_MEMBER.findall(body)
    assert members
    accepted = set(GlobalSettingsRequest.model_fields)
    unknown = [
        m for m in members if re.sub(r"(?<!^)(?=[A-Z])", "_", m).lower() not in accepted
    ]
    assert not unknown
