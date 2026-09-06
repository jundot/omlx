# SPDX-License-Identifier: Apache-2.0
"""Tests for extending the Qwen3.5/3.8 prefill accelerations to DFlash targets.

These cover the install contract only -- no model is loaded. The patch functions
themselves are exercised by tests/test_qwen35_q4_mlp.py,
tests/test_qwen35_fa256_attention.py and tests/test_qwen35_ane_prefill.py.
"""

from types import SimpleNamespace

import pytest

from omlx.patches import dflash_qwen35_accel as accel

# mlx-vlm-bound patches that a DFlash target must never install: dflash drives
# Qwen targets through mlx-lm, so these would mutate VLM globals with no path to
# execute. Guarded here because an earlier revision installed two of them.
VLM_ONLY_PATCHES = (
    ("omlx.patches.qwen35_gdn_prework", "apply_qwen35_gdn_prework_patch"),
    ("omlx.patches.qwen35_gdn_chunked", "apply_qwen35_gdn_prefill_patch"),
    ("omlx.patches.qwen35_verify_sdpa_split", "apply_qwen35_verify_sdpa_split_patch"),
    ("omlx.patches.qwen35_ragged_decode", "apply_qwen35_ragged_decode_patch"),
)


@pytest.fixture
def stub_patches(monkeypatch):
    """Replace the class-level patch entry points with recorders."""
    import omlx.patches.qwen35_fa256_attention as fa256
    import omlx.patches.qwen35_q4_mlp as q4

    calls: list[str] = []

    def recorder(name):
        def _fn(*args, **kwargs):
            calls.append(name)
            return True

        return _fn

    monkeypatch.setattr(q4, "apply_qwen35_q4_mlp_patch", recorder("q4_mlp"))
    monkeypatch.setattr(
        q4, "apply_qwen35_q4_lm_prefill_linear_patch", recorder("q4_lm_prefill_linear")
    )
    monkeypatch.setattr(
        fa256, "apply_qwen35_fa256_attention_patch", recorder("fa256_attention")
    )
    return calls


@pytest.fixture
def forbid_vlm_patches(monkeypatch):
    """Make any mlx-vlm-only patch call fail the test."""
    import importlib

    for module_name, attr in VLM_ONLY_PATCHES:
        module = importlib.import_module(module_name)

        def _boom(*args, _attr=attr, **kwargs):
            raise AssertionError(f"DFlash must not install {_attr}")

        monkeypatch.setattr(module, attr, _boom, raising=False)


class TestAccelEnabled:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("OMLX_DFLASH_QWEN35_ACCEL", raising=False)
        assert accel.accel_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "off", "OFF"])
    def test_kill_switch(self, monkeypatch, value):
        monkeypatch.setenv("OMLX_DFLASH_QWEN35_ACCEL", value)
        assert accel.accel_enabled() is False


class TestClassPatchInstall:
    def test_installs_only_mlx_lm_patches(
        self, monkeypatch, stub_patches, forbid_vlm_patches
    ):
        monkeypatch.delenv("OMLX_DFLASH_QWEN35_ACCEL", raising=False)
        result = accel.install_dflash_qwen35_class_patches(SimpleNamespace())
        assert result["enabled"] is True
        assert set(result["class_patches"]) == {
            "q4_mlp",
            "q4_lm_prefill_linear",
            "fa256_attention",
        }
        assert all(result["class_patches"].values())
        assert stub_patches == ["q4_mlp", "q4_lm_prefill_linear", "fa256_attention"]

    def test_kill_switch_installs_nothing(self, monkeypatch, stub_patches):
        monkeypatch.setenv("OMLX_DFLASH_QWEN35_ACCEL", "0")
        result = accel.install_dflash_qwen35_class_patches(SimpleNamespace())
        assert result == {"enabled": False}
        assert stub_patches == []

    def test_per_model_settings_disable_individual_patches(
        self, monkeypatch, stub_patches
    ):
        monkeypatch.delenv("OMLX_DFLASH_QWEN35_ACCEL", raising=False)
        settings = SimpleNamespace(
            qwen35_q4_mlp_prefill_enabled=False, fa256_steel_prefill_enabled=True
        )
        result = accel.install_dflash_qwen35_class_patches(settings)
        assert set(result["class_patches"]) == {"fa256_attention"}
        assert stub_patches == ["fa256_attention"]

    def test_settings_default_to_enabled_like_batched_engines(
        self, monkeypatch, stub_patches
    ):
        """Absent attributes mean enabled, matching BatchedEngine's getattr default."""
        monkeypatch.delenv("OMLX_DFLASH_QWEN35_ACCEL", raising=False)
        result = accel.install_dflash_qwen35_class_patches(None)
        assert set(result["class_patches"]) == {
            "q4_mlp",
            "q4_lm_prefill_linear",
            "fa256_attention",
        }

    def test_patch_failure_is_reported_not_raised(self, monkeypatch, stub_patches):
        import omlx.patches.qwen35_fa256_attention as fa256

        monkeypatch.delenv("OMLX_DFLASH_QWEN35_ACCEL", raising=False)

        def _raise(*args, **kwargs):
            raise RuntimeError("no metal here")

        monkeypatch.setattr(fa256, "apply_qwen35_fa256_attention_patch", _raise)
        result = accel.install_dflash_qwen35_class_patches(SimpleNamespace())
        assert result["class_patches"]["fa256_attention"] is False
        assert result["class_patches"]["q4_mlp"] is True


class TestAneEnable:
    def _stub_ane(self, monkeypatch, returns=64):
        import omlx.patches.qwen35_ane_prefill as ane

        seen: dict = {}

        def _enable(model, **kwargs):
            seen["model"] = model
            seen.update(kwargs)
            return returns

        monkeypatch.setattr(ane, "enable_qwen35_ane_prefill", _enable)
        return seen

    def test_not_called_when_setting_disabled(self, monkeypatch):
        monkeypatch.delenv("OMLX_DFLASH_QWEN35_ACCEL", raising=False)
        seen = self._stub_ane(monkeypatch)
        model = SimpleNamespace()
        result = accel.enable_dflash_qwen35_ane(
            model, SimpleNamespace(qwen35_ane_prefill_enabled=False)
        )
        assert result["ane_mlp_layers"] == 0
        assert seen == {}

    def test_forwards_configured_split(self, monkeypatch):
        monkeypatch.delenv("OMLX_DFLASH_QWEN35_ACCEL", raising=False)
        seen = self._stub_ane(monkeypatch)
        model = SimpleNamespace(_omlx_ane_gdn_prefill_count=48)
        settings = SimpleNamespace(
            qwen35_ane_prefill_enabled=True,
            qwen35_ane_prefill_sequence_length=2048,
            qwen35_ane_prefill_fraction=0.6,
            qwen35_ane_prefill_max_layers=64,
            qwen35_ane_prefill_gdn=True,
            qwen35_ane_prefill_gdn_fraction=0.6,
            qwen35_ane_prefill_gdn_max_layers=48,
            qwen35_ane_prefill_dual_ane=False,
        )
        result = accel.enable_dflash_qwen35_ane(model, settings, prefill_step_size=2048)
        assert result["ane_mlp_layers"] == 64
        assert result["ane_gdn_layers"] == 48
        assert seen["model"] is model
        assert seen["sequence_length"] == 2048
        assert seen["fraction"] == pytest.approx(0.6)
        assert seen["gdn_fraction"] == pytest.approx(0.6)
        assert seen["dual_ane"] is False

    def test_warns_when_shape_exceeds_prefill_step(self, monkeypatch, caplog):
        monkeypatch.delenv("OMLX_DFLASH_QWEN35_ACCEL", raising=False)
        self._stub_ane(monkeypatch)
        settings = SimpleNamespace(
            qwen35_ane_prefill_enabled=True, qwen35_ane_prefill_sequence_length=4096
        )
        with caplog.at_level("WARNING"):
            accel.enable_dflash_qwen35_ane(
                SimpleNamespace(), settings, prefill_step_size=2048
            )
        assert any("cannot tile" in r.getMessage() for r in caplog.records)

    def test_warns_that_ane_route_is_approximate(self, monkeypatch, caplog):
        monkeypatch.delenv("OMLX_DFLASH_QWEN35_ACCEL", raising=False)
        self._stub_ane(monkeypatch)
        settings = SimpleNamespace(qwen35_ane_prefill_enabled=True)
        with caplog.at_level("WARNING"):
            accel.enable_dflash_qwen35_ane(SimpleNamespace(), settings)
        assert any("approximate INT8" in r.getMessage() for r in caplog.records)

    def test_kill_switch_skips_ane(self, monkeypatch):
        monkeypatch.setenv("OMLX_DFLASH_QWEN35_ACCEL", "0")
        seen = self._stub_ane(monkeypatch)
        result = accel.enable_dflash_qwen35_ane(
            SimpleNamespace(), SimpleNamespace(qwen35_ane_prefill_enabled=True)
        )
        assert result == {"enabled": False}
        assert seen == {}

    def test_enable_failure_is_contained(self, monkeypatch):
        import omlx.patches.qwen35_ane_prefill as ane

        monkeypatch.delenv("OMLX_DFLASH_QWEN35_ACCEL", raising=False)

        def _raise(*args, **kwargs):
            raise RuntimeError("no ANE runtime")

        monkeypatch.setattr(ane, "enable_qwen35_ane_prefill", _raise)
        result = accel.enable_dflash_qwen35_ane(
            SimpleNamespace(), SimpleNamespace(qwen35_ane_prefill_enabled=True)
        )
        assert result["ane_mlp_layers"] == 0
