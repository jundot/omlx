"""Record the prefill/decode and DSpark arrays of deepseek_v41_expected.npz from DeepSeek's
official reference (inference/ of a DeepSeek-V4.1-Flash checkpoint repo) on CPU in fp32.

The TileLang kernels the fp32 model reaches (act_quant, fp4_act_quant, sparse_attn,
hc_split_sinkhorn) are replaced by torch ports that follow kernel.py operation by operation;
weights come from the same per-name generator as tests/test_deepseek_v41.py. Needs torch,
ml_dtypes and sympy (imported by the reference engram.py).

  python record_deepseek_v41_expected.py <inference dir> record <out.npz>
  python record_deepseek_v41_expected.py <inference dir> check <old.npz>
      check: act_quant(o) removed, i.e. the df42c109 model (the 1017 Indexer owns_k guard
      equals the republish workaround used for that recording); compares with <old.npz>.
"""
import sys
import types
import zlib

import ml_dtypes
import numpy as np
import torch

torch.set_default_dtype(torch.float32)
REF, MODE = sys.argv[1], sys.argv[2]


# ---- exact CPU ports of the kernel.py TileLang kernels (fp32 compute) --------------------
def _round_scale(amax, inv):
    v = (amax.float() * torch.tensor(inv, dtype=torch.float32)).contiguous()
    bits = v.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int64)
    return ((e + 127) << 23).to(torch.int32).view(torch.float32)


def _np_cast(x, dt):
    return torch.from_numpy(x.detach().cpu().numpy().astype(np.float32).astype(dt).astype(np.float32))


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    n = x.size(-1)
    xf = x.float().unflatten(-1, (n // block_size, block_size))
    amax = xf.abs().amax(-1).clamp_min(1e-4)
    s = _round_scale(amax, 1 / 448.0) if scale_fmt is not None else amax * torch.tensor(1 / 448.0, dtype=torch.float32)
    q = _np_cast((xf / s.unsqueeze(-1)).clamp(-448.0, 448.0), ml_dtypes.float8_e4m3fn)
    if inplace:
        x.copy_((q * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), s


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    n = x.size(-1)
    xf = x.float().unflatten(-1, (n // block_size, block_size))
    amax = xf.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2.0**-9)
        s = _np_cast(amax / 6.0, ml_dtypes.float8_e4m3fn)
    else:
        amax = amax.clamp_min(6 * 2.0**-126)
        s = _round_scale(amax, 1 / 6.0)
    q = _np_cast((xf / s.unsqueeze(-1)).clamp(-6.0, 6.0), ml_dtypes.float4_e2m1fn)
    assert inplace
    x.copy_((q * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
    return x


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.py sparse_attn_kernel: 64-key blocks, running max from -1e30, P cast to the
    tensor dtype (fp32 here), sink added after the last block."""
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    out = torch.empty_like(q)
    for bi in range(b):
        for mi in range(m):
            qf = q[bi, mi].float()
            acc = torch.zeros(h, d)
            se = torch.zeros(h)
            mx_ = torch.full((h,), -1e30)
            for t in range(0, topk, 64):
                idx = topk_idxs[bi, mi, t:t + 64]
                if idx.numel() < 64:
                    idx = torch.cat([idx, idx.new_full((64 - idx.numel(),), -1)])
                valid = idx != -1
                kvb = torch.where(valid[:, None], kv[bi, idx.clamp_min(0)].float(), torch.zeros(()))
                s = qf @ kvb.T
                s = torch.where(valid[None], s, torch.tensor(-float("inf")))
                s = s * softmax_scale
                prev = mx_
                mx_ = torch.maximum(prev, s.amax(-1))
                sc = torch.exp(prev - mx_)
                p = torch.exp(s - mx_[:, None])
                se = se * sc + p.sum(-1)
                acc = acc * sc[:, None] + p.to(q.dtype).float() @ kvb
            se = se + torch.exp(attn_sink.float() - mx_)
            out[bi, mi] = (acc / se[:, None]).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.exp(comb - comb.amax(-1, keepdim=True))
    comb = comb / comb.sum(-1, keepdim=True) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _unused(*a, **k):
    raise RuntimeError("quantized GEMMs are not used with fp32 weights")


kernel = types.ModuleType("kernel")
kernel.act_quant, kernel.fp4_act_quant, kernel.sparse_attn = act_quant, fp4_act_quant, sparse_attn
kernel.hc_split_sinkhorn, kernel.fp8_gemm, kernel.fp4_gemm = hc_split_sinkhorn, _unused, _unused
sys.modules["kernel"] = kernel
sys.path.insert(0, REF)
source = open(f"{REF}/model.py").read()
if MODE == "check":
    lines = [l for l in source.splitlines() if l.strip() != "act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)"]
    assert len(lines) == len(source.splitlines()) - 2, "expected two act_quant(o) lines"
    source = "\n".join(lines)
model_mod = types.ModuleType("model")
model_mod.__file__ = f"{REF}/model.py"
sys.modules["model"] = model_mod
exec(compile(source, f"{REF}/model.py", "exec"), model_mod.__dict__)
M = model_mod


def tiny(**kwargs):
    values = dict(vocab_size=64, dim=32, moe_inter_dim=32, n_layers=5, n_heads=2, q_lora_rank=32,
                  head_dim=32, rope_head_dim=8, o_groups=2, o_lora_rank=16, index_n_heads=8,
                  index_head_dim=32, index_topk=3, n_routed_experts=4, n_activated_experts=2,
                  window_size=4, dtype="bf16", expert_dtype=None, n_mtp_layers=0, max_seq_len=64,
                  max_batch_size=2, hc_sinkhorn_iters=3, candidate_source_layer=3,
                  candidate_block_size=2, candidate_topk_blocks=2, index_source_layers=(1, 3, 4))
    values.update(kwargs)
    return M.ModelArgs(**values)


def build(args, seed=341, scale=0.08):
    model = M.Transformer(args).float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            rng = np.random.default_rng(seed + zlib.crc32(name.encode()))
            data = rng.normal(0, scale, tuple(p.shape)).astype(np.float32)
            if "norm.weight" in name:
                data += 1
            if "weights_proj.weight" in name:
                data = np.abs(data)
            p.copy_(torch.from_numpy(data))
    return model


# Input ids of test_official_prefill_decode. With seed 16 every act_quant(o) input of these
# runs lies at least 500 ulps from an E4M3 rounding boundary on the GPU; seed 5 put one 7 ulps
# away at length 17, so matching it depended on the GPU's attention summation order, which
# the reference leaves free. The df42c109 arrays that check mode compares with used seed 5.
IDS_SEED = 5 if MODE == "check" else 16
out = {}
for length in (1, 3, 8, 17):
    model = build(tiny())
    ids = torch.from_numpy(np.random.default_rng(IDS_SEED).integers(3, 60, (1, length + 3), dtype=np.int32).astype(np.int64))
    _, logits, _ = model(ids[:, :length], 0)
    out[f"prefill{length}_0"] = logits.float().numpy()
    for index, position in enumerate(range(length, length + 3), 1):
        _, logits, _ = model(ids[:, position:position + 1], position)
        out[f"prefill{length}_{index}"] = logits.float().numpy()
args = tiny(n_mtp_layers=3, dspark_block_size=3, dspark_noise_token_id=2, dspark_target_layer_ids=(2, 3, 4),
            dspark_n_routed_experts=2, dspark_n_activated_experts=1, dspark_markov_rank=32,
            compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0), temperature=0)
model = build(args)
index = 0
for start, ids in [(0, [3, 4, 5, 6, 7]), (5, [8]), (6, [9])]:
    _, logits, hidden = model(torch.tensor([ids]), start)
    out[f"dspark_{index}"] = hidden.float().numpy()
    out[f"dspark_{index + 1}"] = logits.float().numpy()
    index += 2
    actual = model.forward_spec(torch.tensor([[10]]), hidden, start)
    if start:
        out[f"dspark_{index}"] = actual[0].numpy()
        out[f"dspark_{index + 1}"] = actual[1].float().numpy()
        out[f"dspark_{index + 2}"] = actual[2].float().numpy()
        index += 3

if MODE == "check":
    old = np.load(sys.argv[3])
    for k in sorted(out):
        a, e = out[k], old[k]
        if a.shape != e.shape:
            print(f"{k:12} SHAPE {a.shape} vs {e.shape}"); continue
        d = np.abs(a.astype(np.float64) - e)
        print(f"{k:12} max|d| {d.max():.3e}  max rel {(d / np.maximum(np.abs(e), 1e-6)).max():.3e}  "
              f"allclose(2e-5, 2e-4) {np.allclose(a, e, atol=2e-5, rtol=2e-4)}")
else:
    np.savez(sys.argv[3], **out)
    print("recorded", len(out), "arrays to", sys.argv[3])
