import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

torch = None


def _linear(x, w):
    return x @ w.transpose(-1, -2)


def _norm(x, w, eps):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def _rope(x, positions, rope_dim, theta, inverse=False):
    inv = theta ** (-torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    angles = positions.float().unsqueeze(-1) * inv
    cos, sin = angles.cos(), angles.sin()
    if inverse:
        sin = -sin
    if x.ndim == positions.ndim + 2:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:].reshape(*x.shape[:-1], rope_dim // 2, 2)
    real, imag = tail[..., 0], tail[..., 1]
    rotated = torch.stack((real * cos - imag * sin, real * sin + imag * cos), -1).reshape(*x.shape[:-1], rope_dim)
    return torch.cat((head, rotated), -1)


def _mixes(h, fn, scale, base, cfg):
    n = h.shape[-2]
    flat = h.flatten(-2).float()
    mixes = _linear(flat, fn.float()) * torch.rsqrt((flat * flat).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    pre = torch.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + cfg.hc_eps
    post = 2 * torch.sigmoid(mixes[..., n : 2 * n] * scale[1] + base[n : 2 * n])
    comb = (mixes[..., 2 * n :] * scale[2] + base[2 * n :]).reshape(*mixes.shape[:-1], n, n)
    comb = torch.softmax(comb, -1) + cfg.hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + cfg.hc_eps)
    for _ in range(cfg.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + cfg.hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + cfg.hc_eps)
    return pre, post, comb


def _pre(h, pre):
    return (h.float() * pre.unsqueeze(-1)).sum(-2).to(h.dtype)


def _post(x, residual, post, comb):
    mixed = torch.einsum("btij,btid->btjd", comb, residual.float())
    return (post.unsqueeze(-1) * x.float().unsqueeze(-2) + mixed).to(x.dtype)


def _attn(x, start, history, w, cfg):
    b, t, _ = x.shape
    pos = torch.arange(start, start + t).view(1, t)
    q = _linear(_norm(_linear(x, w["wq_a.weight"]), w["q_norm.weight"], cfg.rms_norm_eps), w["wq_b.weight"]).reshape(b, t, cfg.num_attention_heads, cfg.head_dim)
    q = _rope(q, pos, cfg.qk_rope_head_dim, cfg.rope_theta)
    block = _rope(_norm(_linear(x, w["wkv.weight"]), w["kv_norm.weight"], cfg.rms_norm_eps), pos, cfg.qk_rope_head_dim, cfg.rope_theta)
    context = history.latents[:, None, :, None, :].expand(b, t, history.latents.shape[1], cfg.num_attention_heads, cfg.head_dim)
    own = block[:, None, :, None, :].expand(b, t, t, cfg.num_attention_heads, cfg.head_dim)
    keys = torch.cat((context, own), 2)
    valid = torch.cat((history.valid[:, None, :].expand(b, t, -1), torch.ones((b, t, t), dtype=torch.bool)), -1)
    scores = (q[:, :, None] * keys).sum(-1) * cfg.head_dim ** -0.5
    scores = scores.masked_fill(~valid[..., None], -torch.inf)
    sink = w["attn_sink"].reshape(1, 1, cfg.num_attention_heads, 1).expand(b, t, -1, -1)
    probs = torch.softmax(torch.cat((scores.permute(0, 1, 3, 2), sink), -1), -1)[..., :-1].permute(0, 1, 3, 2)
    out = (probs[..., None] * keys).sum(2)
    out = _rope(out, pos, cfg.qk_rope_head_dim, cfg.rope_theta, inverse=True)
    grouped = out.reshape(b, t, cfg.o_groups, cfg.num_attention_heads * cfg.head_dim // cfg.o_groups)
    a = w["wo_a.weight"].reshape(cfg.o_groups, cfg.o_lora_rank, -1)
    return _linear(torch.einsum("btgd,grd->btgr", grouped, a).reshape(b, t, -1), w["wo_b.weight"])


def _swiglu(gate, up, limit):
    return torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)


def _moe(x, w, cfg):
    logits = _linear(x.float(), w["ffn.gate.weight"].float()) + w["ffn.gate.bias"].float()
    scores = torch.sqrt(torch.nn.functional.softplus(logits))
    picked = torch.argsort(scores + w["ffn.gate.bias_vl"].float(), -1)[..., -cfg.num_experts_per_tok :]
    chosen = torch.gather(scores, -1, picked)
    chosen = chosen / chosen.sum(-1, keepdim=True)
    out = torch.zeros_like(x)
    for i in range(cfg.num_experts):
        gate, up = _linear(x, w[f"ffn.experts.{i}.w1.weight"]), _linear(x, w[f"ffn.experts.{i}.w3.weight"])
        value = _linear(_swiglu(gate, up, cfg.swiglu_limit), w[f"ffn.experts.{i}.w2.weight"])
        out = out + ((picked == i).to(x.dtype) * chosen.to(x.dtype)).sum(-1, keepdim=True) * value
    gate, up = _linear(x, w["ffn.shared_experts.w1.weight"]), _linear(x, w["ffn.shared_experts.w3.weight"])
    return out * cfg.router_scaling_factor + _linear(_swiglu(gate, up, cfg.swiglu_limit), w["ffn.shared_experts.w2.weight"])


def _reference(x, start, history, w, cfg):
    h = x[:, :, None, :].expand(-1, -1, cfg.hc_mult, -1).clone()
    ap, ao, ac = _mixes(h, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"], cfg)
    h = _post(_attn(_norm(_pre(h, ap), w["attn_norm.weight"], cfg.rms_norm_eps), start, history, w, cfg), h, ao, ac)
    fp, fo, fc = _mixes(h, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"], cfg)
    h = _post(_moe(_norm(_pre(h, fp), w["ffn_norm.weight"], cfg.rms_norm_eps), w, cfg), h, fo, fc)
    return _pre(h, fp)


def _weights(cfg):
    rng = np.random.default_rng(0)

    def sample(*shape):
        return rng.standard_normal(shape, dtype=np.float32) * np.float32(0.08)

    d, h, hd, q, groups, rank, inter = cfg.dim, cfg.num_attention_heads, cfg.head_dim, cfg.q_lora_rank, cfg.o_groups, cfg.o_lora_rank, cfg.moe_intermediate_size
    values = {
        "wq_a.weight": sample(q, d), "q_norm.weight": 1 + sample(q), "wq_b.weight": sample(h * hd, q),
        "wkv.weight": sample(hd, d), "kv_norm.weight": 1 + sample(hd), "wo_a.weight": sample(groups * rank, h * hd // groups),
        "wo_b.weight": sample(d, groups * rank), "attn_sink": sample(h), "attn_norm.weight": 1 + sample(d), "ffn_norm.weight": 1 + sample(d),
        "hc_attn_fn": sample((2 + cfg.hc_mult) * cfg.hc_mult, cfg.hc_mult * d), "hc_attn_base": sample((2 + cfg.hc_mult) * cfg.hc_mult),
        "hc_attn_scale": 1 + sample(3), "hc_ffn_fn": sample((2 + cfg.hc_mult) * cfg.hc_mult, cfg.hc_mult * d),
        "hc_ffn_base": sample((2 + cfg.hc_mult) * cfg.hc_mult), "hc_ffn_scale": 1 + sample(3),
        "ffn.gate.weight": sample(cfg.num_experts, d), "ffn.gate.bias": sample(cfg.num_experts), "ffn.gate.bias_vl": sample(cfg.num_experts),
    }
    for i in range(cfg.num_experts):
        values.update({f"ffn.experts.{i}.w1.weight": sample(inter, d), f"ffn.experts.{i}.w2.weight": sample(d, inter), f"ffn.experts.{i}.w3.weight": sample(inter, d)})
    values.update({"ffn.shared_experts.w1.weight": sample(inter, d), "ffn.shared_experts.w2.weight": sample(d, inter), "ffn.shared_experts.w3.weight": sample(inter, d)})
    return values, rng


def _config():
    return SimpleNamespace(dim=64, num_attention_heads=4, head_dim=16, qk_rope_head_dim=8, q_lora_rank=24, o_groups=2, o_lora_rank=12, moe_intermediate_size=32, rms_norm_eps=1e-5, rope_theta=10000.0, hc_mult=2, hc_eps=1e-6, hc_sinkhorn_iters=4, num_experts=8, num_experts_per_tok=2, route_norm=True, router_scaling_factor=1.5, moe_router_score_func="sqrtsoftplus", swiglu_limit=0.25)


def _produce(path):
    global torch
    import torch as real_torch

    torch = real_torch
    cfg = _config()
    values, rng = _weights(cfg)
    x = rng.standard_normal((1, 6, 64), dtype=np.float32)
    latents = rng.standard_normal((1, 5, 16), dtype=np.float32)
    valid = np.array([[True, False, True, True, False]])
    w = {name: torch.from_numpy(value) for name, value in values.items()}
    history = SimpleNamespace(latents=torch.from_numpy(latents), valid=torch.from_numpy(valid))
    out = _reference(torch.from_numpy(x), 19, history, w, cfg).detach().numpy()
    np.savez(path, x=x, start_position=np.array(19, np.int32), indices=np.full((1, 5), -1, np.int32), valid=valid, latents=latents, out=out, **values)


def _compare(path):
    import mlx.core as mx

    sys.path.insert(0, str(Path(__file__).parents[2]))
    from omlx.patches.deepseek_v41_dfly2.dfly2_context import DFly2LayerContext
    from omlx.patches.deepseek_v41_dfly2.dfly2_layer import make_draft_layer

    archive = np.load(path)
    cfg = _config()
    reserved = {"x", "start_position", "indices", "valid", "latents", "out"}
    weights = {name: mx.array(archive[name]) for name in archive.files if name not in reserved}
    history = DFly2LayerContext(mx.array(archive["indices"]), mx.array(archive["valid"]), mx.array(archive["latents"]))
    actual = make_draft_layer(weights, cfg)(mx.array(archive["x"]), int(archive["start_position"]), history)
    mx.eval(actual)
    actual = np.asarray(actual)
    expected = archive["out"]
    max_abs = float(np.max(np.abs(actual - expected)))
    max_rel = float(np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-8)))
    print(json.dumps({"max_abs_error": max_abs, "max_rel_error": max_rel}))
    if not np.allclose(actual, expected, rtol=1e-4, atol=1e-5):
        raise AssertionError(f"DFly2 MLX parity failed: max_abs={max_abs}, max_rel={max_rel}")


def test_dfly2_layer_matches_torch_reference():
    import pytest

    torch_python = Path(tempfile.gettempdir()) / "torch-parity311" / "bin" / "python"
    if not torch_python.exists():
        pytest.skip(f"real torch parity venv is absent: {torch_python}")
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "dfly2-layer-parity.npz"
        producer_env = os.environ.copy()
        producer_env.pop("PYTHONPATH", None)
        producer_env.pop("PYTHONHOME", None)
        subprocess.run([str(torch_python), str(Path(__file__)), "--torch-producer", str(archive)], check=True, env=producer_env, capture_output=True, text=True)
        compare_env = os.environ.copy()
        compare_env["PYTHONPATH"] = str(Path(__file__).parents[2])
        result = subprocess.run([sys.executable, str(Path(__file__)), "--mlx-compare", str(archive)], check=True, env=compare_env, capture_output=True, text=True)
        metrics = json.loads(result.stdout)
    assert metrics["max_abs_error"] < 1e-4
    assert metrics["max_rel_error"] < 1e-4


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_dfly2_layer_parity.py --torch-producer|--mlx-compare FILE.npz")
    if sys.argv[1] == "--torch-producer":
        _produce(sys.argv[2])
    elif sys.argv[1] == "--mlx-compare":
        _compare(sys.argv[2])
    else:
        raise SystemExit(f"unknown mode: {sys.argv[1]}")
