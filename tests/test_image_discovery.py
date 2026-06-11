# SPDX-License-Identifier: Apache-2.0
"""Tests for image (mlx-gen / mflux saved-format) model discovery.

Covers the discovery-layer image arms: is_image_model_dir layout acceptance
and rejections (incl. the partial-Wan shield and the upscaler-arm fallback),
read_image_model_kind name heuristics, detect_model_type "image",
_is_model_dir acceptance, full discover_models registration (owner-repo and
flat layouts, no phantom component entries, unknown-name skip), recursive
size estimation, and regression guards for LLM and video_upscaler dirs.
"""

import json
from pathlib import Path

import pytest

from omlx.model_discovery import (
    _is_model_dir,
    detect_model_type,
    discover_models,
    estimate_model_size,
    is_image_model_dir,
    is_video_upscaler_dir,
    read_image_model_kind,
)

# Component subdirs of an mlx-gen saved-format image repo. Each holds a
# safetensors index + shard files; none has a config.json of its own.
_IMAGE_COMPONENTS = ("transformer", "text_encoder", "vae")


def make_image_dir(
    parent: Path,
    name: str = "Z-Image-Turbo-4bit",
    components: tuple[str, ...] = _IMAGE_COMPONENTS,
    component_weight_bytes: int = 1000,
) -> Path:
    """Create an mflux saved-format image model dir with fake shards.

    Shards must be non-empty: estimate_model_size raises on a zero total
    and _register_model would drop the entry.
    """
    model_dir = parent / name
    model_dir.mkdir(parents=True)
    for comp in components:
        comp_dir = model_dir / comp
        comp_dir.mkdir()
        (comp_dir / "model.safetensors.index.json").write_text("{}")
        (comp_dir / "0.safetensors").write_bytes(b"\0" * component_weight_bytes)
    return model_dir


def make_upscaler_dir(parent: Path, name: str = "seedvr2-3b-8bit") -> Path:
    """SeedVR2-style upscaler layout: transformer/ + vae/, NO text_encoder/."""
    return make_image_dir(parent, name=name, components=("transformer", "vae"))


def make_llm_dir(parent: Path, name: str = "llama-3b", weight_bytes: int = 512) -> Path:
    model_dir = parent / name
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    )
    (model_dir / "model.safetensors").write_bytes(b"\0" * weight_bytes)
    return model_dir


class TestIsImageModelDir:
    """Layout acceptance + every rejection arm."""

    def test_full_layout_accepted(self, tmp_path):
        d = make_image_dir(tmp_path)
        assert is_image_model_dir(d) is True

    def test_root_config_json_rejected(self, tmp_path):
        d = make_image_dir(tmp_path)
        (d / "config.json").write_text("{}")
        assert is_image_model_dir(d) is False

    def test_model_index_json_rejected(self, tmp_path):
        # A Wan diffusers dir has model_index.json; never an image model
        d = make_image_dir(tmp_path)
        (d / "model_index.json").write_text(
            json.dumps({"_class_name": "WanPipeline"})
        )
        assert is_image_model_dir(d) is False

    def test_scheduler_dir_rejected(self, tmp_path):
        # Partial Wan download shield: scheduler/ disqualifies the dir
        d = make_image_dir(tmp_path)
        (d / "scheduler").mkdir()
        assert is_image_model_dir(d) is False

    def test_transformer_2_dir_rejected(self, tmp_path):
        # Partial Wan download shield: transformer_2/ disqualifies the dir
        d = make_image_dir(tmp_path)
        (d / "transformer_2").mkdir()
        assert is_image_model_dir(d) is False

    def test_missing_text_encoder_falls_to_upscaler_arm(self, tmp_path):
        """transformer/ + vae/ without text_encoder/ is the SeedVR2 shape:
        NOT an image model, IS a video upscaler."""
        d = make_upscaler_dir(tmp_path)
        assert is_image_model_dir(d) is False
        assert is_video_upscaler_dir(d) is True
        assert detect_model_type(d) == "video_upscaler"

    def test_missing_vae_rejected_by_both_arms(self, tmp_path):
        d = make_image_dir(tmp_path, components=("transformer", "text_encoder"))
        assert is_image_model_dir(d) is False
        assert is_video_upscaler_dir(d) is False
        assert _is_model_dir(d) is False

    def test_missing_transformer_rejected(self, tmp_path):
        d = make_image_dir(tmp_path, components=("text_encoder", "vae"))
        assert is_image_model_dir(d) is False

    def test_missing_component_index_rejected(self, tmp_path):
        # Aborted download: the dir exists but an index has not landed yet
        d = make_image_dir(tmp_path)
        (d / "text_encoder" / "model.safetensors.index.json").unlink()
        assert is_image_model_dir(d) is False


class TestReadImageModelKind:
    """Directory-name heuristics for every known mflux alias."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Qwen-Image-Edit-2511-4bit", ("edit", "qwen-image-edit-2511")),
            ("Qwen-Image-Edit-2509-4bit", ("edit", "qwen-image-edit-2509")),
            ("Qwen-Image-Edit-4bit", ("edit", "qwen-image-edit")),
            ("Z-Image-Turbo-4bit", ("t2i", "z-image-turbo")),
            ("z-image-base-4bit", ("t2i", "z-image")),
            ("Qwen-Image-4bit", ("t2i", "qwen-image")),
        ],
    )
    def test_known_aliases(self, name, expected):
        assert read_image_model_kind(Path("/models") / name) == expected

    def test_2511_marker_wins_over_plain_edit(self):
        # The dated markers must be checked before the bare edit alias
        kind, alias = read_image_model_kind(Path("qwen-image-edit-2511"))
        assert (kind, alias) == ("edit", "qwen-image-edit-2511")

    def test_case_insensitive(self):
        assert read_image_model_kind(Path("QWEN-IMAGE-EDIT-2509")) == (
            "edit", "qwen-image-edit-2509",
        )

    def test_unknown_name_is_empty(self):
        assert read_image_model_kind(Path("FLUX.2-klein-saved")) == ("", "")

    def test_llm_name_is_empty(self):
        assert read_image_model_kind(Path("llama-3b")) == ("", "")


class TestDetectModelTypeImage:
    def test_full_layout_is_image(self, tmp_path):
        d = make_image_dir(tmp_path)
        assert detect_model_type(d) == "image"

    def test_image_branch_runs_before_missing_config_fallback(self, tmp_path):
        """Image dirs have no root config.json; without the image branch the
        missing-config early-exit would have returned "llm"."""
        d = make_image_dir(tmp_path)
        assert not (d / "config.json").exists()
        assert detect_model_type(d) == "image"

    def test_config_json_llm_unchanged(self, tmp_path):
        """Regression guard: plain config.json LLM detection is unaffected."""
        d = make_llm_dir(tmp_path)
        assert detect_model_type(d) == "llm"

    def test_upscaler_layout_is_not_image(self, tmp_path):
        d = make_upscaler_dir(tmp_path)
        assert detect_model_type(d) == "video_upscaler"

    def test_image_layout_is_not_upscaler(self, tmp_path):
        # Mutual exclusion the other way: text_encoder/ excludes the
        # upscaler arm
        d = make_image_dir(tmp_path)
        assert is_video_upscaler_dir(d) is False
        assert detect_model_type(d) == "image"


class TestIsModelDirImage:
    def test_image_root_accepted(self, tmp_path):
        d = make_image_dir(tmp_path)
        assert _is_model_dir(d) is True

    def test_adapter_wins_over_image_layout(self, tmp_path):
        d = make_image_dir(tmp_path)
        (d / "adapter_config.json").write_text("{}")
        assert _is_model_dir(d) is False

    def test_partial_image_dir_not_model_root(self, tmp_path):
        d = make_image_dir(tmp_path, components=("text_encoder",))
        assert _is_model_dir(d) is False


class TestDiscoverImageOwnerRepoLayout:
    """Owner/repo (AbstractFramework/<repo>) two-level layout."""

    def test_single_image_entry_no_phantoms(self, tmp_path):
        make_image_dir(tmp_path / "AbstractFramework")

        models = discover_models(tmp_path)

        assert set(models.keys()) == {"Z-Image-Turbo-4bit"}
        entry = models["Z-Image-Turbo-4bit"]
        assert entry.model_type == "image"
        assert entry.engine_type == "image"
        assert entry.image_pipeline == "t2i"
        assert entry.image_alias == "z-image-turbo"
        # config_model_type surfaces the alias (no config.json to read)
        assert entry.config_model_type == "z-image-turbo"
        # No phantom component entries from org-folder descent
        for comp in _IMAGE_COMPONENTS:
            assert comp not in models

    def test_entry_paths_and_size(self, tmp_path):
        model_dir = make_image_dir(
            tmp_path / "AbstractFramework", component_weight_bytes=1000
        )

        models = discover_models(tmp_path)
        entry = models["Z-Image-Turbo-4bit"]
        assert Path(entry.model_path) == model_dir
        # 3 components x 1000 bytes via the recursive glob, 5% overhead
        assert entry.estimated_size == int(3 * 1000 * 1.05)

    def test_edit_model_registers_with_edit_pipeline(self, tmp_path):
        make_image_dir(
            tmp_path / "AbstractFramework", name="Qwen-Image-Edit-2511-4bit"
        )

        models = discover_models(tmp_path)
        entry = models["Qwen-Image-Edit-2511-4bit"]
        assert entry.model_type == "image"
        assert entry.image_pipeline == "edit"
        assert entry.image_alias == "qwen-image-edit-2511"
        assert entry.config_model_type == "qwen-image-edit-2511"


class TestDiscoverImageFlatLayout:
    """Flat layout: the image dir sits directly under model_dir."""

    def test_single_image_entry_no_phantoms(self, tmp_path):
        make_image_dir(tmp_path)

        models = discover_models(tmp_path)

        assert set(models.keys()) == {"Z-Image-Turbo-4bit"}
        entry = models["Z-Image-Turbo-4bit"]
        assert entry.model_type == "image"
        assert entry.engine_type == "image"
        assert entry.image_pipeline == "t2i"
        assert entry.image_alias == "z-image-turbo"
        assert entry.config_model_type == "z-image-turbo"
        for comp in _IMAGE_COMPONENTS:
            assert comp not in models

    def test_flat_and_owner_repo_give_same_result(self, tmp_path):
        flat_root = tmp_path / "flat"
        flat_root.mkdir()
        make_image_dir(flat_root)

        org_root = tmp_path / "org"
        org_root.mkdir()
        make_image_dir(org_root / "AbstractFramework")

        flat = discover_models(flat_root)
        org = discover_models(org_root)

        assert set(flat.keys()) == set(org.keys()) == {"Z-Image-Turbo-4bit"}
        for key in (
            "model_type", "engine_type", "image_pipeline", "image_alias",
            "config_model_type", "estimated_size",
        ):
            assert getattr(flat["Z-Image-Turbo-4bit"], key) == getattr(
                org["Z-Image-Turbo-4bit"], key
            )

    def test_image_alongside_llm(self, tmp_path):
        make_image_dir(tmp_path)
        make_llm_dir(tmp_path, name="llama-3b")

        models = discover_models(tmp_path)
        assert set(models.keys()) == {"Z-Image-Turbo-4bit", "llama-3b"}
        assert models["Z-Image-Turbo-4bit"].model_type == "image"
        assert models["llama-3b"].model_type == "llm"


class TestUnknownImageNameSkipped:
    """Image-layout dirs whose name maps to no mflux alias are skipped at
    registration -- no entry, no phantom component entries."""

    def test_unknown_name_flat_not_registered(self, tmp_path):
        make_image_dir(tmp_path, name="FLUX.2-klein-saved")

        models = discover_models(tmp_path)
        assert models == {}
        for comp in _IMAGE_COMPONENTS:
            assert comp not in models

    def test_unknown_name_owner_repo_not_registered(self, tmp_path):
        make_image_dir(tmp_path / "black-forest-labs", name="FLUX.2-klein-saved")

        models = discover_models(tmp_path)
        assert models == {}

    def test_skip_logs_warning(self, tmp_path, caplog):
        make_image_dir(tmp_path, name="FLUX.2-klein-saved")

        with caplog.at_level("WARNING", logger="omlx.model_discovery"):
            discover_models(tmp_path)

        assert any(
            "FLUX.2-klein-saved" in rec.message
            and "no mflux alias" in rec.message
            for rec in caplog.records
        )

    def test_unknown_name_does_not_block_siblings(self, tmp_path):
        make_image_dir(tmp_path, name="FLUX.2-klein-saved")
        make_image_dir(tmp_path, name="Z-Image-Turbo-4bit")
        make_llm_dir(tmp_path, name="llama-3b")

        models = discover_models(tmp_path)
        assert set(models.keys()) == {"Z-Image-Turbo-4bit", "llama-3b"}


class TestEstimateImageModelSize:
    """estimate_model_size sums recursive **/*.safetensors for image
    layouts (no root-level weight files, no root safetensors index)."""

    def test_recursive_sum_with_overhead(self, tmp_path):
        model_dir = tmp_path / "Z-Image-Turbo-4bit"
        model_dir.mkdir()
        sizes = {
            "transformer/model-00001-of-00002.safetensors": 3000,
            "transformer/model-00002-of-00002.safetensors": 2000,
            "text_encoder/model.safetensors": 700,
            "vae/model.safetensors": 300,
        }
        for rel, size in sizes.items():
            f = model_dir / rel
            f.parent.mkdir(exist_ok=True)
            f.write_bytes(b"\0" * size)
        for comp in _IMAGE_COMPONENTS:
            (model_dir / comp).mkdir(exist_ok=True)
            (model_dir / comp / "model.safetensors.index.json").write_text("{}")

        expected = int(sum(sizes.values()) * 1.05)
        assert estimate_model_size(model_dir) == expected

    def test_no_weights_raises(self, tmp_path):
        model_dir = tmp_path / "image-empty"
        model_dir.mkdir()
        for comp in _IMAGE_COMPONENTS:
            (model_dir / comp).mkdir()
            (model_dir / comp / "model.safetensors.index.json").write_text("{}")
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
        # LLM entries carry empty image fields
        assert entry.image_pipeline == ""
        assert entry.image_alias == ""

    def test_org_folder_llm_descent_still_works(self, tmp_path):
        org = tmp_path / "mlx-community"
        make_llm_dir(org, name="llama-3b")
        make_llm_dir(org, name="qwen-7b")

        models = discover_models(tmp_path)
        assert set(models.keys()) == {"llama-3b", "qwen-7b"}
        assert all(m.model_type == "llm" for m in models.values())


class TestUpscalerMutualExclusion:
    """video_upscaler dirs still discover as before -- the image arm must
    not capture them, and image dirs must not register as upscalers."""

    def test_upscaler_discovers_as_video_upscaler(self, tmp_path):
        make_upscaler_dir(tmp_path / "AbstractFramework")

        models = discover_models(tmp_path)
        assert "seedvr2-3b-8bit" in models
        entry = models["seedvr2-3b-8bit"]
        assert entry.model_type == "video_upscaler"
        assert entry.engine_type == "video_upscaler"
        assert entry.estimated_size > 0
        assert entry.image_pipeline == ""
        assert entry.image_alias == ""

    def test_image_discovers_as_image_not_upscaler(self, tmp_path):
        make_image_dir(tmp_path / "AbstractFramework")

        models = discover_models(tmp_path)
        entry = models["Z-Image-Turbo-4bit"]
        assert entry.model_type == "image"

    def test_mixed_image_and_upscaler_in_same_org(self, tmp_path):
        org = tmp_path / "AbstractFramework"
        make_image_dir(org)
        make_upscaler_dir(org)

        models = discover_models(tmp_path)
        assert set(models.keys()) == {"Z-Image-Turbo-4bit", "seedvr2-3b-8bit"}
        assert models["Z-Image-Turbo-4bit"].model_type == "image"
        assert models["seedvr2-3b-8bit"].model_type == "video_upscaler"
