"""LoRA adapters: PEFT and mlx formats, refusals and compatibility."""

import json

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from omlx.cache.paged_ssd_cache import numerics_revision_for_model  # noqa: E402
from omlx.lora_adapters import (  # noqa: E402
    AdapterError,
    LoRALinear,
    adapter_info,
    apply_adapter,
    compatibility_problem,
)

DIMS, HIDDEN, RANK = 64, 128, 4


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(DIMS, DIMS, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.QuantizedLinear(HIDDEN, DIMS, bias=False)
        self.gate = nn.Linear(DIMS, 8, bias=False)  # A MoE router


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _MLP()


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_Layer(), _Layer()]


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.set_dtype(mx.bfloat16)

    @property
    def layers(self):
        return self.model.layers


def _factors():
    mx.random.seed(0)
    return {
        "layers.1.self_attn.q_proj": (
            mx.random.normal((RANK, DIMS)) * 0.1,
            mx.random.normal((DIMS, RANK)) * 0.1,
        ),
        "layers.0.mlp.down_proj": (
            mx.random.normal((RANK, HIDDEN)) * 0.1,
            mx.random.normal((DIMS, RANK)) * 0.1,
        ),
    }


def _write(path, fmt, factors, **config):
    """Write factors given as PEFT [r, in] / [out, r] in the requested format."""
    path.mkdir()
    if fmt == "peft":
        tensors = {}
        for name, (a, b) in factors.items():
            key = f"base_model.model.model.{name}"
            tensors[f"{key}.lora_A.weight"] = a
            tensors[f"{key}.lora_B.weight"] = b
        weights = "adapter_model.safetensors"
        config = {"peft_type": "LORA", "r": RANK, "lora_alpha": 8, **config}
    else:
        tensors = {}
        for name, (a, b) in factors.items():
            tensors[f"model.{name}.lora_a"] = a.T
            tensors[f"model.{name}.lora_b"] = b.T
        weights = "adapters.safetensors"
        params = {"rank": RANK, "scale": 0.5, "dropout": 0.0}
        config = {"fine_tune_type": "lora", "lora_parameters": params, **config}
    (path / "adapter_config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(path / weights), tensors)
    return path


def _rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.abs(a - b).max() / mx.abs(b).max()).item()


@pytest.mark.parametrize("fmt", ["peft", "mlx"])
def test_adapter_adds_the_scaled_update(tmp_path, fmt):
    # PEFT: alpha / r = 2, and alpha_pattern gives down_proj 4 / r = 1.
    factors = _factors()
    path = _write(tmp_path / "a", fmt, factors, alpha_pattern={"down_proj": 4})
    scales = {"peft": (2.0, 1.0), "mlx": (0.5, 0.5)}[fmt]
    model = _Model()
    q = model.layers[1].self_attn.q_proj
    down = model.layers[0].mlp.down_proj
    x = mx.random.normal((1, 3, DIMS)).astype(mx.bfloat16)
    h = mx.random.normal((1, 3, HIDDEN)).astype(mx.bfloat16)
    (qa, qb), (da, db) = factors.values()
    want_q = q(x) + scales[0] * (x @ qa.T) @ qb.T
    want_down = down(h) + scales[1] * (h @ da.T) @ db.T

    assert apply_adapter(model, path) == 2
    assert isinstance(model.layers[1].self_attn.q_proj, LoRALinear)
    assert model.layers[1].self_attn.q_proj.lora_a.dtype == mx.bfloat16
    assert _rel(model.layers[1].self_attn.q_proj(x), want_q) < 0.02
    assert _rel(model.layers[0].mlp.down_proj(h), want_down) < 0.02
    # The SSD cache must not reuse KV computed without this adapter.
    assert numerics_revision_for_model(model).startswith("lora-")


def test_refused_adapter_leaves_the_model_unchanged(tmp_path):
    factors = _factors()
    a, b = factors["layers.0.mlp.down_proj"]
    factors["layers.0.mlp.down_proj"] = (a[:, :96], b)
    model = _Model()

    with pytest.raises(AdapterError, match=r"layers.0.mlp.down_proj is 128 -> 64"):
        apply_adapter(model, _write(tmp_path / "a", "peft", factors))
    # Decode paths read router weights directly, so a wrapper would break them.
    router = {"layers.0.mlp.gate": (a[:, :DIMS], b[:8])}
    with pytest.raises(AdapterError, match="MoE router"):
        apply_adapter(model, _write(tmp_path / "b", "peft", router))
    assert isinstance(model.layers[1].self_attn.q_proj, nn.Linear)
    assert numerics_revision_for_model(model) is None


def test_adapter_info_and_compatibility(tmp_path):
    info = adapter_info(
        _write(tmp_path / "a", "peft", _factors(), base_model_name_or_path="org/base")
    )
    assert (info.format, info.base_model, info.rank, info.num_layers) == (
        "peft",
        "org/base",
        RANK,
        2,
    )
    assert info.targets == ("down_proj", "q_proj")
    assert info.resident_bytes == 2 * RANK * (2 * DIMS + HIDDEN + DIMS)
    assert info.problem is None
    assert (
        compatibility_problem(info, {"num_hidden_layers": 2, "hidden_size": DIMS})
        is None
    )
    assert "has 1" in compatibility_problem(
        info, {"text_config": {"num_hidden_layers": 1, "hidden_size": DIMS}}
    )
    assert "width 32" in compatibility_problem(
        info, {"num_hidden_layers": 2, "hidden_size": 32}
    )
    assert "does not describe" in compatibility_problem(info, {})
    # A malformed header is a problem to show, not an error for the listing.
    header = b'["not", "a", "dict"]'
    (tmp_path / "a" / "adapter_model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header
    )
    assert "malformed" in adapter_info(tmp_path / "a").problem


@pytest.mark.parametrize(
    "config, extra, problem",
    [
        ({"use_dora": True}, None, "DoRA"),
        ({"modules_to_save": ["lm_head"]}, None, "modules_to_save"),
        ({"init_lora_weights": "pissa_niter_4"}, None, "residual base"),
        ({"loftq_config": {"loftq_bits": 4}}, None, "LoftQ"),
        ({}, "lm_head", "decoder layers"),
        ({}, "visual.layers.0.qkv", "decoder layers"),
    ],
)
def test_unsupported_adapters_report_why(tmp_path, config, extra, problem):
    factors = _factors()
    if extra:
        factors = {extra: factors["layers.1.self_attn.q_proj"]}
    info = adapter_info(_write(tmp_path / "a", "peft", factors, **config))
    assert problem in info.problem
    with pytest.raises(AdapterError):
        apply_adapter(_Model(), tmp_path / "a")
