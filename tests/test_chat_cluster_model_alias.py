# SPDX-License-Identifier: Apache-2.0
"""Static contract checks for Chat's cluster model-ID migration."""

from pathlib import Path


CHAT_TEMPLATE = (
    Path(__file__).parents[1] / "omlx" / "admin" / "templates" / "chat.html"
)


def test_chat_maps_active_deployment_ids_to_public_model_ids():
    source = CHAT_TEMPLATE.read_text(encoding="utf-8")

    assert "fetch('/admin/api/cluster/deployments'" in source
    assert "gatewayByPath.get(String(deployment.model || ''))" in source
    assert "this.aliasToGateway[deployment.deployment_id] = gatewayId" in source


def test_chat_migrates_saved_cluster_model_handles():
    source = CHAT_TEMPLATE.read_text(encoding="utf-8")

    assert "const gatewayId = this.resolveGatewayModelId(chat.model)" in source
    assert "chat.model = gatewayId" in source
    assert "const gatewayId = this.resolveGatewayModelId(session.model)" in source
