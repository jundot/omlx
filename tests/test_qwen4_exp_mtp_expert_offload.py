# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next (qwen4_exp): native Lightning MTP + MoE expert offload.

The backbone experts stream from the checkpoint while the embedded native MTP
head (``mtp.*``) stays resident, mirroring the DeepSeek V4.1 / glm5_next
pairing.
"""

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

E, D, INTER, GROUP = 32, 64, 32, 32


def test_family_gate_admits_qwen4_exp_mtp_with_offload():
    from omlx.model_settings import validate_moe_expert_offload

    settings = {"moe_expert_offload_enabled": True, "mtp_enabled": True}
    validate_moe_expert_offload(settings, model_type="qwen4_exp")
    validate_moe_expert_offload(settings, model_type="qwen4-exp")
    # Families without a resident-head path stay rejected.
    with pytest.raises(ValueError, match="MoE expert offload cannot"):
        validate_moe_expert_offload(settings, model_type="qwen3_5_moe")
    # DFlash and VLM MTP have no offload-aware draft path, qwen4_exp included.
    for key in ("dflash_enabled", "vlm_mtp_enabled"):
        with pytest.raises(ValueError, match="MoE expert offload cannot"):
            validate_moe_expert_offload(
                {"moe_expert_offload_enabled": True, key: True},
                model_type="qwen4_exp",
            )


def _make_glu(seed):
    mx.random.seed(seed)
    glu = SwitchGLU(D, INTER, E)
    nn.quantize(glu, group_size=GROUP, bits=4)
    return glu


def _glu_tensors(glu, prefix):
    out = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(glu, proj)
        for field in ("weight", "scales", "biases"):
            if lin.get(field) is not None:
                out[f"{prefix}.{proj}.{field}"] = lin[field]
    return out


class _Mlp(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.switch_mlp = glu


class _Layer(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.mlp = _Mlp(glu)


class _Head(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.layers = [_Layer(glu)]


class _Model(nn.Module):
    """Backbone ``layers.*.mlp.switch_mlp`` plus a native ``mtp`` head."""

    def __init__(self, backbone, head):
        super().__init__()
        self.layers = [_Layer(g) for g in backbone]
        self.mtp = _Head(head)


def _build(tmp_path):
    backbone = [_make_glu(0), _make_glu(1)]
    head = _make_glu(2)
    tensors = {}
    for i, g in enumerate(backbone):
        tensors.update(_glu_tensors(g, f"layers.{i}.mlp.switch_mlp"))
    tensors.update(_glu_tensors(head, "mtp.layers.0.mlp.switch_mlp"))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    return _Model(backbone, head), head


@pytest.mark.parametrize("mtp_resident", [False, True])
def test_native_mtp_head_stays_resident(tmp_path, mtp_resident):
    from omlx.patches.moe_expert_offload import (
        OffloadSwitchGLU,
        apply_moe_expert_offload,
    )

    model, head = _build(tmp_path)
    wrapped = apply_moe_expert_offload(
        model, tmp_path, resident_fraction=0.25, mtp_resident=mtp_resident
    )
    assert all(isinstance(l.mlp.switch_mlp, OffloadSwitchGLU) for l in model.layers)
    head_mlp = model.mtp.layers[0].mlp.switch_mlp
    if mtp_resident:
        assert wrapped == 2
        assert head_mlp is head and not isinstance(head_mlp, OffloadSwitchGLU)
    else:
        assert wrapped == 3
        assert isinstance(head_mlp, OffloadSwitchGLU)


def test_mtp_path_matcher():
    from omlx.patches.moe_expert_offload import _is_mtp_path

    assert _is_mtp_path("mtp.layers.0.mlp.switch_mlp")
    assert _is_mtp_path("language_model.mtp.layers.0.mlp.switch_mlp")
    assert not _is_mtp_path("layers.3.mlp.switch_mlp")
    assert not _is_mtp_path("language_model.model.layers.0.mlp.switch_mlp")
    assert not _is_mtp_path("mtp_proj.switch_mlp")


def test_admission_and_wrapper_share_the_mtp_matcher(tmp_path, monkeypatch):
    """Admission must price the head by the same rule the wrapper skips it."""
    from omlx.patches import moe_expert_offload as moe

    _build(tmp_path)
    full = sum(f.stat().st_size for f in tmp_path.glob("*.safetensors"))
    seen = []
    real = moe._is_mtp_path

    def spy(name):
        seen.append(name)
        return real(name)

    monkeypatch.setattr(moe, "_is_mtp_path", spy)
    streamed = moe.estimate_offload_admission_bytes(tmp_path, full, 0.25)
    resident = moe.estimate_offload_admission_bytes(
        tmp_path, full, 0.25, mtp_resident=True
    )
    assert any(n.startswith("mtp.layers.0.mlp.switch_mlp.") for n in seen)
    assert resident > streamed
