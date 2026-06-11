# SPDX-License-Identifier: Apache-2.0
"""Tests for image-model pool rejection (mirrors test_video_pool_and_lease
Part A and TestUpscalerPoolRejection).

Image models are job-managed via POST /v1/images and must never be
pool-loaded: EnginePool.get_engine rejects model_type "image" with
ModelTypeNotLoadableError BEFORE the memory-admission loop (a misrouted
chat request must not evict resident LLM engines), and _load_engine carries
a defense-in-depth arm for any other caller. EngineEntry copies
image_pipeline / image_alias from DiscoveredModel and get_status() emits
both so the chat UI can route image models.
"""

import time

import pytest

from omlx.engine_pool import EngineEntry, EnginePool
from omlx.exceptions import EnginePoolError, ModelTypeNotLoadableError

GB = 1024**3

_IMAGE_COMPONENTS = ("transformer", "text_encoder", "vae")


class FakeLLMEngine:
    """Loaded-engine stand-in that records eviction attempts."""

    def __init__(self):
        self.stop_called = False

    def has_active_requests(self) -> bool:
        return False

    async def stop(self) -> None:
        self.stop_called = True


def _make_pool_with_image_and_llm():
    pool = EnginePool(scheduler_config=None)
    fake_engine = FakeLLMEngine()
    pool._entries["llm-id"] = EngineEntry(
        model_id="llm-id",
        model_path="/nonexistent/llm-id",
        model_type="llm",
        engine_type="batched",
        estimated_size=4 * GB,
        engine=fake_engine,
        last_access=time.time(),
    )
    pool._entries["image-id"] = EngineEntry(
        model_id="image-id",
        model_path="/nonexistent/image-id",
        model_type="image",
        engine_type="image",
        estimated_size=42 * GB,
        image_pipeline="t2i",
        image_alias="z-image-turbo",
    )
    return pool, fake_engine


class TestImagePoolRejection:
    def test_model_type_not_loadable_is_engine_pool_error(self):
        assert issubclass(ModelTypeNotLoadableError, EnginePoolError)
        exc = ModelTypeNotLoadableError("image-id", "image")
        assert exc.model_id == "image-id"
        assert exc.model_type == "image"
        assert "/v1/images" in str(exc)

    def test_model_type_map_has_image_engine(self):
        assert EnginePool._MODEL_TYPE_TO_ENGINE["image"] == "image"

    async def test_get_engine_rejects_image_before_admission(self, monkeypatch):
        """Image rejection fires before admission: no LLM eviction happens.

        Memory is mocked so that, had the 42GB image entry reached the
        admission loop, projected (20 + 42 GB) > ceiling (50 GB) would
        have evicted the idle llm entry. The rejection must fire first.
        """
        pool, fake_engine = _make_pool_with_image_and_llm()
        pool._get_final_ceiling = lambda: 50 * GB
        monkeypatch.setattr(
            "omlx.engine_pool.get_phys_footprint", lambda pid=None: 20 * GB
        )

        with pytest.raises(ModelTypeNotLoadableError) as excinfo:
            await pool.get_engine("image-id")

        assert excinfo.value.model_id == "image-id"
        assert excinfo.value.model_type == "image"
        assert "/v1/images" in str(excinfo.value)
        # The resident llm engine must be untouched -- not stopped, not
        # unloaded.
        assert pool._entries["llm-id"].engine is fake_engine
        assert fake_engine.stop_called is False

    async def test_load_engine_defensive_arm_rejects_image(self):
        """_load_engine itself refuses image entries (defense in depth for
        callers that bypass get_engine) -- an mflux dir must never fall
        into BatchedEngine."""
        pool, _ = _make_pool_with_image_and_llm()

        with pytest.raises(ModelTypeNotLoadableError) as excinfo:
            await pool._load_engine("image-id")

        assert excinfo.value.model_type == "image"
        assert pool._entries["image-id"].engine is None


class TestImagePoolStatus:
    """get_status() emits image_pipeline + image_alias so /v1/models/status
    consumers (chat UI routing, auto-pick) can tell t2i / edit apart."""

    def test_get_status_emits_image_fields(self):
        pool, _ = _make_pool_with_image_and_llm()

        models = {m["id"]: m for m in pool.get_status()["models"]}

        assert models["image-id"]["image_pipeline"] == "t2i"
        assert models["image-id"]["image_alias"] == "z-image-turbo"
        # Non-image entries carry the empty defaults
        assert models["llm-id"]["image_pipeline"] == ""
        assert models["llm-id"]["image_alias"] == ""

    def test_discover_propagates_image_fields_to_status(self, tmp_path):
        """End to end: DiscoveredModel.image_pipeline/image_alias ->
        EngineEntry -> get_status() wire fields."""
        model_dir = tmp_path / "AbstractFramework" / "Qwen-Image-Edit-2511-4bit"
        for comp in _IMAGE_COMPONENTS:
            (model_dir / comp).mkdir(parents=True)
            (model_dir / comp / "model.safetensors.index.json").write_text("{}")
            (model_dir / comp / "0.safetensors").write_bytes(b"\0" * 64)

        pool = EnginePool(scheduler_config=None)
        pool.discover_models(str(tmp_path))

        models = {m["id"]: m for m in pool.get_status()["models"]}
        entry = models["Qwen-Image-Edit-2511-4bit"]
        assert entry["model_type"] == "image"
        assert entry["engine_type"] == "image"
        assert entry["image_pipeline"] == "edit"
        assert entry["image_alias"] == "qwen-image-edit-2511"

    def test_engine_entry_copies_image_fields_from_discovery(self, tmp_path):
        """EngineEntry construction copies both image fields verbatim."""
        model_dir = tmp_path / "Z-Image-Turbo-4bit"
        for comp in _IMAGE_COMPONENTS:
            (model_dir / comp).mkdir(parents=True)
            (model_dir / comp / "model.safetensors.index.json").write_text("{}")
            (model_dir / comp / "0.safetensors").write_bytes(b"\0" * 64)

        pool = EnginePool(scheduler_config=None)
        pool.discover_models(str(tmp_path))

        entry = pool._entries["Z-Image-Turbo-4bit"]
        assert entry.model_type == "image"
        assert entry.engine_type == "image"
        assert entry.image_pipeline == "t2i"
        assert entry.image_alias == "z-image-turbo"
