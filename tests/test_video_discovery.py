# SPDX-License-Identifier: Apache-2.0
"""Tests for video (diffusers-layout) model discovery.

Covers the discovery-layer changes from docs/video-generation-engine-spec.md
section 4.1: model_index.json as a model root, WanPipeline -> "video",
unknown-pipeline skip, no phantom component entries from org-folder descent,
and regression guards for the existing config.json (LLM) path.
"""

import json
from pathlib import Path

import pytest

from omlx.model_discovery import (
    VIDEO_PIPELINE_CLASSES,
    _is_model_dir,
    detect_model_type,
    discover_models,
    estimate_model_size,
    read_model_index_pipeline_class,
    read_video_pipeline_kind,
)

# Component subdirs of a Wan2.2-style diffusers repo. Each carries its own
# config.json + weights, so pre-fix org-folder descent would have registered
# them as phantom standalone "llm" models.
_WAN_COMPONENTS = ("transformer", "transformer_2", "vae", "text_encoder")


def make_diffusers_dir(
    parent: Path,
    name: str = "Wan2.2-T2V-A14B",
    class_name: str | None = "WanPipeline",
    component_weight_bytes: int = 1024,
    index_extra: dict | None = None,
) -> Path:
    """Create a diffusers-layout model dir with fake component weights."""
    model_dir = parent / name
    model_dir.mkdir(parents=True)

    index: dict = {"_diffusers_version": "0.35.0"}
    if class_name is not None:
        index["_class_name"] = class_name
    if index_extra:
        index.update(index_extra)
    (model_dir / "model_index.json").write_text(json.dumps(index))

    for comp in _WAN_COMPONENTS:
        comp_dir = model_dir / comp
        comp_dir.mkdir()
        # Component config.json is registerable on its own (would detect as
        # "llm") -- exactly what made the phantom-entry bug dangerous.
        (comp_dir / "config.json").write_text(
            json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
        )
        (comp_dir / "model.safetensors").write_bytes(b"\0" * component_weight_bytes)

    return model_dir


def make_llm_dir(parent: Path, name: str = "llama-3b", weight_bytes: int = 512) -> Path:
    """Create a plain transformers-layout LLM model dir."""
    model_dir = parent / name
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    )
    (model_dir / "model.safetensors").write_bytes(b"\0" * weight_bytes)
    return model_dir


class TestReadModelIndexPipelineClass:
    """Unit tests for read_model_index_pipeline_class."""

    def test_valid_wan_pipeline(self, tmp_path):
        (tmp_path / "model_index.json").write_text(
            json.dumps({"_class_name": "WanPipeline", "_diffusers_version": "0.35.0"})
        )
        assert read_model_index_pipeline_class(tmp_path) == "WanPipeline"

    def test_unknown_pipeline_class_still_returned(self, tmp_path):
        """The reader returns the raw class; the allowlist filter lives elsewhere."""
        (tmp_path / "model_index.json").write_text(
            json.dumps({"_class_name": "FluxPipeline"})
        )
        assert read_model_index_pipeline_class(tmp_path) == "FluxPipeline"

    def test_missing_model_index(self, tmp_path):
        assert read_model_index_pipeline_class(tmp_path) is None

    def test_missing_class_name_key(self, tmp_path):
        (tmp_path / "model_index.json").write_text(
            json.dumps({"_diffusers_version": "0.35.0"})
        )
        assert read_model_index_pipeline_class(tmp_path) is None

    def test_non_string_class_name(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": 123}))
        assert read_model_index_pipeline_class(tmp_path) is None

    def test_invalid_json(self, tmp_path):
        (tmp_path / "model_index.json").write_text("{not valid json")
        assert read_model_index_pipeline_class(tmp_path) is None

    def test_wan_pipeline_in_allowlist(self):
        assert "WanPipeline" in VIDEO_PIPELINE_CLASSES

    def test_wan_image_to_video_pipeline_in_allowlist(self):
        assert "WanImageToVideoPipeline" in VIDEO_PIPELINE_CLASSES


class TestReadVideoPipelineKind:
    """Unit tests for read_video_pipeline_kind (t2v / i2v / ti2v)."""

    @staticmethod
    def _write_index(path: Path, payload: dict) -> None:
        (path / "model_index.json").write_text(json.dumps(payload))

    def test_image_to_video_pipeline_is_i2v(self, tmp_path):
        self._write_index(tmp_path, {"_class_name": "WanImageToVideoPipeline"})
        assert read_video_pipeline_kind(tmp_path) == "i2v"

    def test_wan_pipeline_with_transformer_2_is_t2v(self, tmp_path):
        # T2V-A14B declares a real transformer_2 component entry
        self._write_index(tmp_path, {
            "_class_name": "WanPipeline",
            "transformer_2": ["diffusers", "WanTransformer3DModel"],
        })
        assert read_video_pipeline_kind(tmp_path) == "t2v"

    def test_wan_pipeline_without_transformer_2_is_ti2v(self, tmp_path):
        self._write_index(tmp_path, {"_class_name": "WanPipeline"})
        assert read_video_pipeline_kind(tmp_path) == "ti2v"

    def test_wan_pipeline_with_null_transformer_2_is_ti2v(self, tmp_path):
        # TI2V-5B declares "transformer_2": [null, null]
        self._write_index(tmp_path, {
            "_class_name": "WanPipeline",
            "transformer_2": [None, None],
        })
        assert read_video_pipeline_kind(tmp_path) == "ti2v"

    def test_missing_model_index_is_empty(self, tmp_path):
        assert read_video_pipeline_kind(tmp_path) == ""

    def test_unknown_class_is_empty(self, tmp_path):
        self._write_index(tmp_path, {"_class_name": "FluxPipeline"})
        assert read_video_pipeline_kind(tmp_path) == ""

    def test_invalid_json_is_empty(self, tmp_path):
        (tmp_path / "model_index.json").write_text("{not valid json")
        assert read_video_pipeline_kind(tmp_path) == ""


class TestDetectModelTypeVideo:
    """Tests for detect_model_type video branch + LLM regression."""

    def test_wan_pipeline_dir_is_video(self, tmp_path):
        model_dir = make_diffusers_dir(tmp_path)
        assert detect_model_type(model_dir) == "video"

    def test_wan_image_to_video_pipeline_dir_is_video(self, tmp_path):
        model_dir = make_diffusers_dir(
            tmp_path, name="Wan2.2-I2V-A14B",
            class_name="WanImageToVideoPipeline",
        )
        assert detect_model_type(model_dir) == "video"

    def test_wan_pipeline_index_alone_is_video(self, tmp_path):
        """model_index.json alone (no components yet) already types as video."""
        (tmp_path / "model_index.json").write_text(
            json.dumps({"_class_name": "WanPipeline"})
        )
        assert detect_model_type(tmp_path) == "video"

    def test_unknown_pipeline_falls_through_to_llm(self, tmp_path):
        """Unknown pipeline class does not type as video. detect_model_type
        falls back to "llm" (the skip happens in _register_model)."""
        model_dir = make_diffusers_dir(tmp_path, class_name="FluxPipeline")
        assert detect_model_type(model_dir) == "llm"

    def test_config_json_llm_unchanged(self, tmp_path):
        """Regression guard: plain config.json LLM detection is unaffected."""
        model_dir = make_llm_dir(tmp_path)
        assert detect_model_type(model_dir) == "llm"

    def test_video_branch_runs_before_missing_config_fallback(self, tmp_path):
        """A WanPipeline dir has no root config.json; without the video branch
        the missing-config early-exit would have returned "llm"."""
        model_dir = make_diffusers_dir(tmp_path)
        assert not (model_dir / "config.json").exists()
        assert detect_model_type(model_dir) == "video"


class TestIsModelDir:
    """_is_model_dir accepts model_index.json as a model root."""

    def test_model_index_json_is_model_root(self, tmp_path):
        (tmp_path / "model_index.json").write_text(
            json.dumps({"_class_name": "WanPipeline"})
        )
        assert _is_model_dir(tmp_path) is True

    def test_config_json_is_model_root(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert _is_model_dir(tmp_path) is True

    def test_empty_dir_is_not_model_root(self, tmp_path):
        assert _is_model_dir(tmp_path) is False

    def test_adapter_wins_over_model_index(self, tmp_path):
        """adapter_config.json + model_index.json -> adapter check wins."""
        (tmp_path / "model_index.json").write_text(
            json.dumps({"_class_name": "WanPipeline"})
        )
        (tmp_path / "adapter_config.json").write_text("{}")
        assert _is_model_dir(tmp_path) is False


class TestDiscoverVideoOwnerRepoLayout:
    """Owner/repo (organized two-level) layout."""

    def test_single_video_entry_no_phantoms(self, tmp_path):
        make_diffusers_dir(tmp_path / "Wan-AI", name="Wan2.2-T2V-A14B")

        models = discover_models(tmp_path)

        assert set(models.keys()) == {"Wan2.2-T2V-A14B"}
        entry = models["Wan2.2-T2V-A14B"]
        assert entry.model_type == "video"
        assert entry.engine_type == "video"
        assert entry.config_model_type == "WanPipeline"
        # No phantom component entries
        for comp in _WAN_COMPONENTS:
            assert comp not in models

    def test_entry_paths_and_size(self, tmp_path):
        model_dir = make_diffusers_dir(
            tmp_path / "Wan-AI", component_weight_bytes=1000
        )

        models = discover_models(tmp_path)
        entry = models["Wan2.2-T2V-A14B"]
        assert Path(entry.model_path) == model_dir
        # 4 components x 1000 bytes, 5% runtime overhead
        assert entry.estimated_size == int(4 * 1000 * 1.05)

    def test_video_alongside_llm_in_same_org(self, tmp_path):
        org = tmp_path / "Wan-AI"
        make_diffusers_dir(org)
        make_llm_dir(org, name="some-llm")

        models = discover_models(tmp_path)
        assert set(models.keys()) == {"Wan2.2-T2V-A14B", "some-llm"}
        assert models["Wan2.2-T2V-A14B"].model_type == "video"
        assert models["some-llm"].model_type == "llm"


class TestDiscoverVideoFlatLayout:
    """Flat layout: the diffusers dir sits directly under model_dir.

    This is the org-folder-descent fix: pre-fix, a dir without root
    config.json was treated as an organization folder and its component
    subdirs (transformer/, vae/, ...) were registered as phantom llm models.
    """

    def test_single_video_entry_no_phantoms(self, tmp_path):
        make_diffusers_dir(tmp_path, name="Wan2.2-T2V-A14B")

        models = discover_models(tmp_path)

        assert set(models.keys()) == {"Wan2.2-T2V-A14B"}
        entry = models["Wan2.2-T2V-A14B"]
        assert entry.model_type == "video"
        assert entry.engine_type == "video"
        assert entry.config_model_type == "WanPipeline"
        for comp in _WAN_COMPONENTS:
            assert comp not in models

    def test_flat_and_owner_repo_give_same_result(self, tmp_path):
        flat_root = tmp_path / "flat"
        flat_root.mkdir()
        make_diffusers_dir(flat_root)

        org_root = tmp_path / "org"
        org_root.mkdir()
        make_diffusers_dir(org_root / "Wan-AI")

        flat = discover_models(flat_root)
        org = discover_models(org_root)

        assert set(flat.keys()) == set(org.keys()) == {"Wan2.2-T2V-A14B"}
        for key in ("model_type", "engine_type", "config_model_type", "estimated_size"):
            assert getattr(flat["Wan2.2-T2V-A14B"], key) == getattr(
                org["Wan2.2-T2V-A14B"], key
            )


class TestVideoPipelineKindViaDiscovery:
    """DiscoveredModel.video_pipeline is set at registration for video
    models (i2v / t2v / ti2v) and stays empty for everything else."""

    def test_i2v_repo_discovers_with_i2v_kind(self, tmp_path):
        make_diffusers_dir(
            tmp_path, name="Wan2.2-I2V-A14B",
            class_name="WanImageToVideoPipeline",
        )

        models = discover_models(tmp_path)

        assert set(models.keys()) == {"Wan2.2-I2V-A14B"}
        entry = models["Wan2.2-I2V-A14B"]
        assert entry.model_type == "video"
        assert entry.engine_type == "video"
        assert entry.config_model_type == "WanImageToVideoPipeline"
        assert entry.video_pipeline == "i2v"

    def test_t2v_repo_discovers_with_t2v_kind(self, tmp_path):
        make_diffusers_dir(
            tmp_path,
            name="Wan2.2-T2V-A14B",
            index_extra={
                "transformer_2": ["diffusers", "WanTransformer3DModel"]
            },
        )

        models = discover_models(tmp_path)
        assert models["Wan2.2-T2V-A14B"].video_pipeline == "t2v"

    def test_ti2v_repo_discovers_with_ti2v_kind(self, tmp_path):
        # No transformer_2 entry in model_index.json -> TI2V-5B shape
        make_diffusers_dir(tmp_path, name="Wan2.2-TI2V-5B")

        models = discover_models(tmp_path)
        assert models["Wan2.2-TI2V-5B"].video_pipeline == "ti2v"

    def test_llm_entry_has_empty_video_pipeline(self, tmp_path):
        make_llm_dir(tmp_path)

        models = discover_models(tmp_path)
        assert models["llama-3b"].video_pipeline == ""


class TestUnknownPipelineSkipped:
    """Unknown diffusers pipelines are skipped at registration -- no entry,
    no phantom component entries."""

    def test_flux_pipeline_flat_not_registered(self, tmp_path):
        make_diffusers_dir(tmp_path, name="FLUX.2-dev", class_name="FluxPipeline")

        models = discover_models(tmp_path)
        assert models == {}

    def test_unknown_i2v_pipeline_still_not_registered(self, tmp_path):
        """Regression: widening the allowlist to WanImageToVideoPipeline must
        not let other image-to-video pipelines through."""
        make_diffusers_dir(
            tmp_path, name="CogVideoX1.5-5B-I2V",
            class_name="CogVideoXImageToVideoPipeline",
        )

        models = discover_models(tmp_path)
        assert models == {}

    def test_flux_pipeline_owner_repo_not_registered(self, tmp_path):
        make_diffusers_dir(
            tmp_path / "black-forest-labs", name="FLUX.2-dev", class_name="FluxPipeline"
        )

        models = discover_models(tmp_path)
        assert models == {}

    def test_flux_skip_logs_warning(self, tmp_path, caplog):
        make_diffusers_dir(tmp_path, name="FLUX.2-dev", class_name="FluxPipeline")

        with caplog.at_level("WARNING", logger="omlx.model_discovery"):
            discover_models(tmp_path)

        assert any(
            "FluxPipeline" in rec.message and "FLUX.2-dev" in rec.message
            for rec in caplog.records
        )

    def test_flux_does_not_block_sibling_models(self, tmp_path):
        make_diffusers_dir(tmp_path, name="FLUX.2-dev", class_name="FluxPipeline")
        make_llm_dir(tmp_path, name="llama-3b")
        make_diffusers_dir(tmp_path, name="Wan2.2-T2V-A14B")

        models = discover_models(tmp_path)
        assert set(models.keys()) == {"Wan2.2-T2V-A14B", "llama-3b"}


class TestMalformedModelIndex:
    """model_index.json with no _class_name or broken JSON: never video,
    discovery does not crash."""

    def test_missing_class_name_not_video(self, tmp_path):
        make_diffusers_dir(tmp_path, name="no-class-name", class_name=None)

        models = discover_models(tmp_path)

        # A model_index.json without a readable _class_name (and no root
        # config.json) is skipped entirely: registering it would produce
        # an unloadable llm entry. No phantoms either.
        assert "no-class-name" not in models
        for comp in _WAN_COMPONENTS:
            assert comp not in models

    def test_invalid_json_not_video(self, tmp_path):
        model_dir = make_diffusers_dir(tmp_path, name="bad-json", class_name=None)
        (model_dir / "model_index.json").write_text("{definitely not json")

        models = discover_models(tmp_path)

        for entry in models.values():
            assert entry.model_type != "video"
        for comp in _WAN_COMPONENTS:
            assert comp not in models

    def test_malformed_index_without_weights_not_registered(self, tmp_path):
        """No weights anywhere -> estimate_model_size raises -> entry dropped
        gracefully (no exception escapes discover_models)."""
        model_dir = tmp_path / "empty-index"
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text(json.dumps({"foo": "bar"}))

        models = discover_models(tmp_path)
        assert models == {}


class TestAdapterExclusion:
    """adapter_config.json wins over model_index.json."""

    def test_adapter_with_model_index_excluded_flat(self, tmp_path):
        model_dir = make_diffusers_dir(tmp_path, name="wan-lora")
        (model_dir / "adapter_config.json").write_text("{}")

        models = discover_models(tmp_path)

        assert "wan-lora" not in models
        # Adapter dirs are skipped wholesale -- no descent, no phantoms.
        for comp in _WAN_COMPONENTS:
            assert comp not in models

    def test_adapter_with_model_index_excluded_in_org(self, tmp_path):
        model_dir = make_diffusers_dir(tmp_path / "Wan-AI", name="wan-lora")
        (model_dir / "adapter_config.json").write_text("{}")

        models = discover_models(tmp_path)
        assert models == {}


class TestEstimateModelSizeDiffusers:
    """estimate_model_size sums recursive **/*.safetensors for diffusers
    layouts (no root-level weight files)."""

    def test_recursive_sum_with_overhead(self, tmp_path):
        model_dir = tmp_path / "Wan2.2-T2V-A14B"
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text(
            json.dumps({"_class_name": "WanPipeline"})
        )
        sizes = {
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": 3000,
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": 2000,
            "transformer_2/diffusion_pytorch_model.safetensors": 1500,
            "vae/diffusion_pytorch_model.safetensors": 700,
            "text_encoder/model.safetensors": 300,
        }
        for rel, size in sizes.items():
            f = model_dir / rel
            f.parent.mkdir(exist_ok=True)
            f.write_bytes(b"\0" * size)

        expected = int(sum(sizes.values()) * 1.05)
        assert estimate_model_size(model_dir) == expected

    def test_no_weights_raises(self, tmp_path):
        model_dir = tmp_path / "wan-empty"
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text(
            json.dumps({"_class_name": "WanPipeline"})
        )
        with pytest.raises(ValueError):
            estimate_model_size(model_dir)


class TestLLMRegressionGuard:
    """Normal LLM dirs (config.json) still discover exactly as before."""

    def test_flat_llm(self, tmp_path):
        make_llm_dir(tmp_path, name="llama-3b", weight_bytes=2048)

        models = discover_models(tmp_path)

        assert set(models.keys()) == {"llama-3b"}
        entry = models["llama-3b"]
        assert entry.model_type == "llm"
        assert entry.engine_type == "batched"
        assert entry.config_model_type == "llama"
        assert entry.estimated_size == int(2048 * 1.05)

    def test_org_folder_llm_descent_still_works(self, tmp_path):
        org = tmp_path / "mlx-community"
        make_llm_dir(org, name="llama-3b")
        make_llm_dir(org, name="qwen-7b")

        models = discover_models(tmp_path)
        assert set(models.keys()) == {"llama-3b", "qwen-7b"}
        assert all(m.model_type == "llm" for m in models.values())

    def test_mixed_llm_and_video(self, tmp_path):
        make_llm_dir(tmp_path, name="llama-3b")
        make_diffusers_dir(tmp_path / "Wan-AI")

        models = discover_models(tmp_path)
        assert set(models.keys()) == {"llama-3b", "Wan2.2-T2V-A14B"}
        assert models["llama-3b"].model_type == "llm"
        assert models["llama-3b"].engine_type == "batched"
        assert models["Wan2.2-T2V-A14B"].model_type == "video"
        assert models["Wan2.2-T2V-A14B"].engine_type == "video"
