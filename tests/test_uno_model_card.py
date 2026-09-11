# SPDX-License-Identifier: Apache-2.0
"""Optional adapter verification must not hide an otherwise usable model card."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from huggingface_hub.utils import GatedRepoError

from omlx.admin.hf_downloader import HFDownloader


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "gated", "network", "json", "shape"])
async def test_verification_failure_preserves_metadata_and_retry_recovers(
    tmp_path, monkeypatch, caplog, failure
):
    from omlx.admin import hf_downloader as hf

    info = SimpleNamespace(
        id="example/K2-Uno",
        downloads=12,
        likes=2,
        tags=["uno"],
        pipeline_tag="text-generation",
        safetensors=None,
        card_data=None,
        created_at=None,
        last_modified=None,
        siblings=[
            SimpleNamespace(rfilename=f, size=1000)
            for f in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "README.md",
            )
        ],
    )
    api = Mock()
    api.model_info.return_value = info
    monkeypatch.setattr(hf, "_get_hf_api", lambda: (api, "https://huggingface.co"))
    config, readme = tmp_path / "adapter_config.json", tmp_path / "README.md"
    readme.write_text("# Retained model card")
    config.write_text("broken" if failure == "json" else "[]")
    errors = {
        "timeout": TimeoutError("simulated timeout"),
        "gated": GatedRepoError(
            "access denied",
            response=httpx.Response(
                403,
                request=httpx.Request("GET", "https://huggingface.co/example/K2-Uno"),
            ),
        ),
        "network": httpx.ConnectError("simulated connection failure"),
    }
    broken = True

    def download(*, filename, **kwargs):
        if filename == "README.md":
            return str(readme)
        if broken and failure in errors:
            raise errors[failure]
        return str(config)

    monkeypatch.setattr(hf, "hf_hub_download", download)
    result = await HFDownloader.get_model_info(info.id)
    assert result["is_adapter"] is True and result["is_uno_adapter"] is None
    assert result["uno_adapter_error"]
    assert result["model_card"] == "# Retained model card"
    assert len(result["files"]) == 3 and result["downloads"] == 12
    assert "example/K2-Uno" in caplog.text
    broken = False
    config.write_text(
        json.dumps(
            {"peft_type": "LORA", "base_model_name_or_path": "IFM/K2-Horizon-7B"}
        )
    )
    retried = await HFDownloader.get_model_info(info.id)
    assert retried["is_uno_adapter"] is True and retried["uno_adapter_error"] is None
    assert retried["model_card"] == result["model_card"]


@pytest.mark.asyncio
async def test_primary_metadata_failure_is_not_misreported_as_unverified(monkeypatch):
    from omlx.admin import hf_downloader as hf

    api = Mock()
    api.model_info.side_effect = TimeoutError("metadata unavailable")
    monkeypatch.setattr(hf, "_get_hf_api", lambda: (api, "https://huggingface.co"))
    with pytest.raises(TimeoutError, match="metadata unavailable"):
        await HFDownloader.get_model_info("example/K2-Uno")
