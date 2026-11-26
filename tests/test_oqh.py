"""oQh format: kernels, quantizer, planner pricing, load hook and producer fusion."""

import gc
import json

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx_lm.utils import load_model  # noqa: E402

from omlx.custom_kernels.oqh import fast  # noqa: E402
from omlx.lora_adapters import LoRALinear, apply_adapter  # noqa: E402
from omlx.oqh import format as fmt  # noqa: E402
from omlx.oqh import quantize, targets  # noqa: E402
from omlx.oqh.loading import install_load_hook  # noqa: E402

pytestmark = pytest.mark.skipif(not fast.has_native(), reason="oQh kernels not built")


def _rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.abs(a - b).max() / mx.abs(b).max()).item()


def _signs(key, n):
    return mx.array(fmt.make_signs(key, n)).astype(mx.float32)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_matmuls_match_the_dequantized_weight(bits):
    rows, cols, experts = 64, 512, 3
    mx.random.seed(bits)
    sg = _signs("w", cols)
    gram = mx.eye(cols)[None]
    w = mx.random.normal((rows, cols)) * 0.05
    packed, rscale, _, _ = quantize.quantize(w, gram, bits, sg, 512)
    ref_w = quantize.dequantize(packed, rscale, bits, sg, 512)
    x = mx.random.normal((5, cols)).astype(mx.bfloat16)
    xr = quantize.rotate(x, sg, 512).astype(mx.bfloat16)
    ref = x.astype(mx.float32) @ ref_w.T
    assert _rel(fast.qmv(xr, packed, rscale, bits), ref) < 0.02
    assert (
        _rel(fast.qmm(mx.concatenate([xr] * 4, 0), packed, rscale, bits)[:5], ref)
        < 0.02
    )

    we = mx.random.normal((experts, rows, cols)) * 0.05
    pe, re, _, _ = quantize.quantize(
        we, mx.broadcast_to(gram, (experts, cols, cols)), bits, sg, 512
    )
    idx = mx.array([2, 0])
    got = fast.gather_qmv(xr[:1], pe, re, idx, bits)
    for i, e in enumerate((2, 0)):
        ref_e = (
            x[:1].astype(mx.float32)
            @ quantize.dequantize(pe[e], re[e], bits, sg, 512).T
        )
        assert _rel(got[i : i + 1], ref_e) < 0.02

    # Tensor-parallel slices hand over strided row scales.
    strided = mx.stack([rscale, rscale], -1).astype(mx.bfloat16)[:, 0]
    assert mx.array_equal(
        fast.qmv(xr, packed, strided, bits),
        fast.qmv(xr, packed, rscale.astype(mx.bfloat16), bits),
    )

    # Batched rows share each weight block and keep the one-row result, also in a
    # partly filled last row group and for gate | up pairs.
    xs = quantize.rotate(mx.random.normal((11, cols)), sg, 512).astype(mx.bfloat16)
    for n in (3, 11):
        got = fast.qmv(xs[:n], packed, rscale, bits)
        pair = fast.qmv(xs[:n], packed, rscale, bits, packed, rscale)
        for i in range(n):
            one = fast.qmv(xs[i : i + 1], packed, rscale, bits)
            assert mx.array_equal(got[i : i + 1], one)
            assert mx.array_equal(pair[i : i + 1], mx.concatenate([one, one], -1))

    # A gate | up pair in one call gives both products bit for bit.
    up, up_scale = packed[::-1], rscale[::-1]
    single = [fast.qmv(xr, packed, rscale, bits), fast.qmv(xr, up, up_scale, bits)]
    paired = fast.qmv(xr, packed, rscale, bits, up, up_scale)
    assert mx.array_equal(paired, mx.concatenate(single, -1))
    single = [fast.gather_qmv(xr[:1], pe, re, idx, bits)] * 2
    paired = fast.gather_qmv(xr[:1], pe, re, idx, bits, pe, re)
    assert mx.array_equal(paired, mx.concatenate(single, -1))


def test_decode_batches_take_the_batched_affine_kernels():
    from omlx.oqh.modules import OqhLinear
    from omlx.patches import batched_decode_qmm

    batched_decode_qmm.apply_batched_decode_qmm_patch()
    n = k = 1024
    mx.random.seed(5)
    m = OqhLinear(k, n, 2, 256)
    m.weight = mx.random.randint(0, 2**32 - 1, (n, k * 2 // 32)).astype(mx.uint32)
    m.rscale = (mx.random.uniform(shape=(n,)) * 0.02).astype(mx.bfloat16)
    m.oqh_signs = _signs("b", k).reshape(-1, 256).astype(mx.bfloat16)
    x = mx.random.normal((8, 1, k)).astype(mx.bfloat16)
    got = m.matmul_rotated(x)
    # Each row agrees with the one-row GEMV up to rounding.
    for i in range(8):
        assert _rel(got[i], m.matmul_rotated(x[i : i + 1])[0]) < 0.01
    assert any(key[1:4] == (8, k, n) for key in batched_decode_qmm._CHECKED)


def _switch(experts, rows, cols, key, seed, group_size=None):
    """oQh switch linear holding quantized random experts, row or group scales."""
    from omlx.oqh.modules import OqhAffineSwitchLinear, OqhSwitchLinear

    mx.random.seed(seed)
    block = min(cols, fmt.MAX_BLOCK)
    sg = _signs(key, cols)
    w = mx.random.normal((experts, rows, cols)) * 0.05
    gram = mx.broadcast_to(mx.eye(cols), (experts, cols, cols))
    # bf16 scales, as checkpoints store them.
    if group_size:
        packed, scales, biases, _, _ = quantize.quantize_affine(
            w, gram, 4, sg, block, group_size
        )
        m = OqhAffineSwitchLinear(cols, rows, experts, 4, group_size, block)
        m.weight = packed
        m.oqh_scales = scales.astype(mx.bfloat16)
        m.oqh_biases = biases.astype(mx.bfloat16)
    else:
        packed, rscale, _, _ = quantize.quantize(w, gram, 4, sg, block)
        m = OqhSwitchLinear(cols, rows, experts, 4, block)
        m.weight, m.rscale = packed, rscale.astype(mx.bfloat16)
    m.oqh_signs = sg.reshape(-1, block).astype(mx.bfloat16)
    return m


def test_gather_gemv_rows_match_one_row_decode():
    m = _switch(8, 64, 512, "s", 0)
    x = mx.random.normal((3, 1, 1, 1, 512)).astype(mx.bfloat16)
    idx = mx.array([[[1, 5]], [[5, 2]], [[7, 0]]])
    batched = m.matmul_rotated(x, idx)
    for b in range(3):
        assert mx.array_equal(
            batched[b : b + 1], m.matmul_rotated(x[b : b + 1], idx[b : b + 1])
        )
    # One stream verifying three positions.
    verify = m.matmul_rotated(x.reshape(1, 3, 1, 1, 512), idx.reshape(1, 3, 2))
    assert mx.array_equal(verify.reshape(batched.shape), batched)


def test_fused_switch_glu_serves_mlx_vlm_and_weighted_sum_callers():
    from mlx_vlm.models.switch_layers import SwitchGLU

    from omlx.oqh import runtime
    from omlx.patches import qwen35_moe_weighted_sum as weighted

    glu = SwitchGLU(512, 256, 8)
    glu.gate_proj = _switch(8, 256, 512, "in", 1)
    glu.up_proj = _switch(8, 256, 512, "in", 2)
    glu.down_proj = _switch(8, 512, 256, "down", 3)
    x = mx.random.normal((1, 40, 512)).astype(mx.bfloat16)
    # Four routes: the native weighted sum supports top-k 4, 6, 8 and 10.
    inds = mx.random.randint(0, 8, (1, 40, 4))
    scores = mx.softmax(mx.random.normal((1, 40, 4)), axis=-1)
    shared = mx.random.normal((1, 40, 512)).astype(mx.bfloat16)
    want = glu(x, inds, scores, shared).astype(mx.float32)
    assert runtime._install_mlp(glu) == [glu.gate_proj, glu.up_proj]
    # mlx-vlm passes the router weights and shared experts to SwitchGLU.
    assert _rel(glu(x, inds, scores, shared), want) < 0.03
    kernel = weighted._native_weighted_sum()
    if kernel is not None:
        got = weighted._native_switch_weighted_sum(glu, x, inds, scores, kernel)
        assert _rel(got + shared, want) < 0.03


@pytest.mark.parametrize("group_size", [None, 128], ids=["row", "group"])
def test_tensor_parallel_split_keeps_blocks_and_row_scales(group_size):
    from types import SimpleNamespace

    from mlx.nn.layers.distributed import shard_inplace
    from mlx_lm.models.switch_layers import SwitchGLU

    from omlx.cluster.tensor_strategies import _glm5_columns, _glm5_rows
    from omlx.oqh import runtime

    def glu():
        m = SwitchGLU(512, 256, 8)
        m.gate_proj = _switch(8, 256, 512, "in", 1, group_size)
        m.up_proj = _switch(8, 256, 512, "in", 2, group_size)
        # 256 inputs split over two ranks: 128-wide blocks.
        m.down_proj = _switch(8, 512, 256, "down", 3, group_size)
        m.down_proj.oqh_signs = m.down_proj["oqh_signs"].reshape(2, 128)
        runtime._install_mlp(m)
        return m

    x = mx.random.normal((1, 5, 512)).astype(mx.bfloat16)
    inds = mx.random.randint(0, 8, (1, 5, 2))
    want = glu()(x, inds).astype(mx.float32)
    parts = []
    for rank in range(2):
        m = glu()
        group = SimpleNamespace(rank=lambda r=rank: r, size=lambda: 2)
        shard_inplace(m.gate_proj, _glm5_rows, group=group)
        shard_inplace(m.up_proj, _glm5_rows, group=group)
        shard_inplace(m.down_proj, _glm5_columns, group=group)
        assert m.gate_proj.output_dims == m.up_proj.output_dims == 128
        if group_size:
            # Group scales split with the rows of gate / up and the inputs of down.
            assert m.gate_proj["oqh_scales"].shape == (8, 128, 4)
            assert m.down_proj["oqh_biases"].shape == (8, 512, 1)
        else:
            assert m.down_proj["rscale"].shape == (8, 512)
        assert m.down_proj["oqh_signs"].shape == (1, 128)
        parts.append(m(x, inds).astype(mx.float32))
    assert _rel(parts[0] + parts[1], want) < 0.02


def test_planner_limits_oqh_tensor_parallel():
    from omlx.cluster import planner

    oqh = {"version": 1, "tensors": {}}
    glm = {"model_type": "glm5_next", fmt.CONFIG_KEY: oqh}
    assert planner._supports_tensor_parallel(glm)
    assert fmt.TP_BLOCK_RANKS in planner._tensor_parallel_divisors(glm)
    # Qwen3.5 tensor parallelism swaps linears for MLX's sharded layer types.
    assert not planner._supports_tensor_parallel(
        {"model_type": "qwen3_5", fmt.CONFIG_KEY: oqh}
    )


def test_quantize_beats_row_rounding_under_the_gram():
    rows, cols = 64, 512
    mx.random.seed(0)
    x = mx.random.normal((1024, cols)) * (1 + 3 * (mx.arange(cols) % 7 == 0))
    from omlx import oq

    gram = oq._shrunk_gram(x.T @ x)
    w = mx.random.normal((rows, cols)) * 0.05
    _, _, before, after = quantize.quantize(w, gram, 3, _signs("w", cols), 512)
    assert after < 0.95 * before


@pytest.mark.parametrize("block", [128, 256])
def test_producers_match_the_reference(block):
    dim = 1024
    mx.random.seed(block)
    x = mx.random.normal((3, dim)).astype(mx.bfloat16)
    r = mx.random.normal((3, dim)).astype(mx.bfloat16)
    w = (1 + 0.2 * mx.random.normal((dim,))).astype(mx.bfloat16)
    sg = _signs("n", dim).astype(mx.bfloat16)
    # The norm's plain output is what MLX's rms_norm returns; it rotates in 256 blocks.
    base = mx.fast.rms_norm(x, w, 1e-6)
    rot, plain = fast.rmsnorm_had(x, w, sg, 1e-6)
    assert mx.array_equal(plain, base)
    assert _rel(rot, quantize.rotate(base, sg, 256)) < 0.01
    assert _rel(fast.rotate(x, sg, block), quantize.rotate(x, sg, block)) < 0.01

    a, b = x.astype(mx.float32), r.astype(mx.float32)
    got = fast.act_split_had(mx.concatenate([x, r], -1), sg, block)
    assert _rel(got, quantize.rotate(nn.silu(a) * b, sg, block)) < 0.01
    # Prefill passes gate and up as two inputs: the same values.
    assert mx.array_equal(fast.act_pair_had(x, r, sg, block), got)
    # GLM-5.3's clamped SwiGLU: gate capped at the limit, up clipped to it.
    big = mx.concatenate([x, r], -1) * 8
    clamped = nn.silu(mx.minimum(8 * a, 10.0)) * mx.clip(8 * b, -10.0, 10.0)
    got = fast.act_split_had(big, sg, block, limit=10.0)
    assert _rel(got, quantize.rotate(clamped, sg, block)) < 0.01
    assert mx.array_equal(fast.act_pair_had(8 * x, 8 * r, sg, block, 10.0), got)
    heads = x.reshape(3, dim // 128, 128)
    z = r.reshape(3, dim // 128, 128)
    wg = w[:128]
    ref = nn.silu(z.astype(mx.float32)) * mx.fast.rms_norm(heads, wg, 1e-6).astype(
        mx.float32
    )
    got, _ = fast.gnorm_had(heads, z, wg, sg, 1e-6, block)
    assert (
        _rel(got.reshape(3, dim), quantize.rotate(ref.reshape(3, dim), sg, block))
        < 0.01
    )


def test_memo_lives_as_long_as_the_plain_array():
    from omlx.oqh import modules

    plain, rotated = mx.ones((2, 256)), mx.zeros((2, 256))
    modules.remember(plain, rotated, 7)
    assert modules._recall(plain, 7) is rotated
    del plain
    gc.collect()
    assert not any(k == 7 for _, _, k in modules._MEMO.entries)


def test_targets_and_row_parallel_blocks():
    assert targets.QWEN.is_target(
        "language_model.model.layers.3.mlp.switch_mlp.down_proj"
    )
    assert not targets.QWEN.is_target("language_model.model.layers.3.mlp.gate")
    assert not targets.QWEN.is_target("mtp.layers.0.mlp.down_proj")
    assert not targets.QWEN.is_target("model.layers.0.linear_attn.in_proj_b", (4, 2048))
    # 768 = 3 x 256 inputs: a TP2 rank holds 384 of the down projection's, so its
    # blocks narrow to 128; gate reads the whole input in 256-wide blocks.
    assert targets.QWEN.rotation("model.layers.0.mlp.down_proj", 768) == (
        "layers.0.mlp.down_proj",
        128,
    )
    assert targets.QWEN.rotation("model.layers.0.mlp.gate_proj", 768) == (
        "layers.0.post",
        256,
    )
    assert targets.QWEN.rotation("model.layers.0.self_attn.q_proj", 5120)[0] == (
        targets.QWEN.rotation("model.layers.0.self_attn.v_proj", 5120)[0]
    )


def test_oqh_refuses_unsupported_models_and_levels(tmp_path):
    from omlx import oq

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(ValueError, match="oQh does not support"):
        oq.quantize_oq_streaming(str(tmp_path), str(tmp_path / "out"), 3, hadamard=True)
    # DeepSeek V4.1 oQ4 keeps the source FP4 experts, so oQh needs a lower level.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    with pytest.raises(ValueError, match="below oQ4"):
        oq.quantize_oq_streaming(str(tmp_path), str(tmp_path / "out"), 4, hadamard=True)


def test_reinvest_keeps_the_affine_size_and_widths():
    shapes = {
        f"model.layers.{i}.{name}": shape
        for i in range(2)
        for name, shape in (
            ("mlp.gate_proj", (1024, 512)),
            ("mlp.up_proj", (1024, 512)),
            ("mlp.down_proj", (512, 1024)),
            ("self_attn.q_proj", (512, 512)),
        )
    }
    base = {"bits": 3, "group_size": 64, "mode": "affine"}
    planned = {
        p: dict(base) for p, sh in shapes.items() if targets.QWEN.is_target(p, sh)
    }
    planned["model.layers.1.self_attn.q_proj"]["bits"] = 5
    plan = targets.reinvest(shapes, planned, {"0": 1.0, "1": 2.0})
    assert all(plan[p] >= planned[p]["bits"] for p in plan)
    affine = sum(targets._affine_bytes(shapes[p], planned[p]) for p in plan)
    assert sum(targets.price(shapes[p], plan[p]) for p in plan) <= affine
    # The narrowest targets get the freed bytes first, down projections first; the
    # q_proj of layer 0 then takes a second bit before any 3-bit gate / up fits.
    assert [plan[f"model.layers.{i}.mlp.down_proj"] for i in (0, 1)] == [4, 4]
    assert plan["model.layers.0.self_attn.q_proj"] == 5
    assert plan["model.layers.0.mlp.gate_proj"] == 3
    # From oQ4, Qwen and GLM keep group scales: their bytes leave less to reinvest.
    assert targets.QWEN.group_size(4) == fmt.AFFINE_GROUP_SIZE
    assert targets.QWEN.group_size(3.5) is None
    assert targets.GLM5_NEXT.group_size(4) == fmt.AFFINE_GROUP_SIZE
    gs = targets.QWEN.group_size(4)
    grouped = targets.reinvest(shapes, planned, {"0": 1.0, "1": 2.0}, gs)
    assert sum(targets.price(shapes[p], grouped[p], gs) for p in plan) <= affine
    assert sum(grouped.values()) < sum(plan.values())


def test_reinvest_raises_gate_and_up_together():
    shapes = {
        "model.layers.0.mlp.gate_proj": (1024, 512),
        "model.layers.0.mlp.up_proj": (1024, 512),
        "model.layers.0.mlp.down_proj": (512, 1024),
    }
    planned = {p: {"bits": 3, "group_size": 64, "mode": "affine"} for p in shapes}
    planned["model.layers.0.mlp.down_proj"]["bits"] = 5
    # The slack buys one more bit for gate or up alone, not for both.
    plan = targets.reinvest(shapes, planned, {})
    assert (
        plan["model.layers.0.mlp.gate_proj"] == plan["model.layers.0.mlp.up_proj"] == 3
    )
    assert plan["model.layers.0.mlp.down_proj"] == 6


def _tiny_qwen(tmp_path, moe: bool, group_size=None):
    """A tiny Qwen3.5 checkpoint whose decoder linears are oQh, and its float twin."""
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3_5, qwen3_5_moe

    config = {
        "model_type": "qwen3_5_moe" if moe else "qwen3_5",
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "linear_num_value_heads": 8,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "vocab_size": 64,
        "full_attention_interval": 2,
        "max_position_embeddings": 512,
        "tie_word_embeddings": False,
    }
    if moe:
        config.update(
            num_experts=4,
            num_experts_per_tok=2,
            shared_expert_intermediate_size=256,
            moe_intermediate_size=256,
        )
    module = qwen3_5_moe if moe else qwen3_5
    mx.random.seed(3)
    model = module.Model(module.ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    weights = dict(tree_flatten(model.parameters()))
    twin = dict(weights)
    entries = {}
    for key in list(weights):
        base = key.removesuffix(".weight")
        if not key.endswith(".weight") or not targets.QWEN.is_target(
            base, weights[key].shape
        ):
            continue
        w = weights.pop(key).astype(mx.float32)
        signs_key, block = targets.QWEN.rotation(base, int(w.shape[-1]))
        sg = _signs(signs_key, int(w.shape[-1]))
        gram = mx.broadcast_to(
            mx.eye(w.shape[-1]), (w.shape[0] if w.ndim == 3 else 1, *w.shape[-1:] * 2)
        )
        if group_size:
            packed, scales, biases, _, _ = quantize.quantize_affine(
                w, gram, 4, sg, block, group_size
            )
            scales, biases = scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)
            weights[f"{base}.oqh_scales"] = scales
            weights[f"{base}.oqh_biases"] = biases
            deq = quantize.dequantize_affine(packed, scales, biases, 4, sg, block)
        else:
            packed, rscale, _, _ = quantize.quantize(w, gram, 4, sg, block)
            weights[f"{base}.rscale"] = rscale
            deq = quantize.dequantize(packed, rscale, 4, sg, block)
        weights[key] = packed
        weights[f"{base}.oqh_signs"] = sg.reshape(-1, block).astype(mx.bfloat16)
        twin[key] = deq.astype(mx.bfloat16)
        entries[base] = fmt.config_entry(4, block, group_size)
    path = tmp_path / ("moe" if moe else "dense")
    path.mkdir()
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    config[fmt.CONFIG_KEY] = {"version": fmt.FORMAT_VERSION, "tensors": entries}
    (path / "config.json").write_text(json.dumps(config))
    ref = module.Model(module.ModelArgs.from_dict(config))
    ref.load_weights(list(twin.items()))
    return path, ref, len(entries)


@pytest.mark.parametrize("group_size", [None, 128], ids=["row", "group"])
@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_load_hook_serves_the_checkpoint_with_fusion(tmp_path, moe, group_size):
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    from omlx.oqh.loading import install_load_hook
    from omlx.oqh.modules import OQH_MODULES

    path, ref, count = _tiny_qwen(tmp_path, moe, group_size)
    install_load_hook()
    model, _ = load_model(path)
    loaded = [m for _, m in model.named_modules() if isinstance(m, OQH_MODULES)]
    assert len(loaded) >= count - (2 if moe else 0)
    fused = {type(m).__name__ for _, m in model.named_modules()}
    assert "OqhRMSNorm" in fused
    # The GDN gated norm hands the rotated input to out_proj.
    assert any(n.startswith("Oqh") and n.endswith("RMSNormGated") for n in fused)

    ids = mx.array([[1, 5, 9, 2, 7, 3, 11, 4]])
    want = ref(ids).astype(mx.float32)
    got = model(ids).astype(mx.float32)
    # Stock affine matmuls drift from a float twin by 0.025-0.04 on this model
    # even without rotation; each group-scale linear alone matches within 0.6%.
    tol = 0.03 if group_size is None else 0.06
    assert _rel(got, want) < tol

    cache = make_prompt_cache(model)
    model(ids[:, :6], cache=cache)
    step = mx.concatenate(
        [model(ids[:, i : i + 1], cache=cache) for i in (6, 7)], axis=1
    )
    assert _rel(step.astype(mx.float32), want[:, 6:]) < tol

    # A caller that bypasses the decoder layer still gets plain-basis results.
    layer = model.model.layers[0]
    h = mx.random.normal((1, 3, 256)).astype(mx.bfloat16)
    ref_layer = ref.model.layers[0]
    assert _rel(layer.mlp(h), ref_layer.mlp(h)) < 0.03

    # Row-exact MTP verify feeds every row through the one-token arithmetic.
    from mlx_vlm.models.qwen3_5.speculative_verifier import (
        Qwen3_5BatchInvariantForward,
    )

    from omlx.oqh.modules import exact_rows

    assert getattr(Qwen3_5BatchInvariantForward.__call__, "_omlx_oqh", False)
    mlp = layer.mlp.shared_expert if moe else layer.mlp
    with exact_rows():
        rows = Qwen3_5BatchInvariantForward._feed_forward(None, mlp, h)
    for i in range(3):
        assert mx.array_equal(rows[:, i : i + 1], mlp(h[:, i : i + 1]))
    # Outside it, three rows take the stock matmul: equal up to rounding.
    assert _rel(mlp(h), rows) < 0.03


def test_lora_adapter_runs_on_fused_oqh_layers(tmp_path):
    """Wrapped oQh projections leave the fused MLP and match the float twin."""
    path, ref, _ = _tiny_qwen(tmp_path, False)
    install_load_hook()
    model, _ = load_model(path)
    assert type(model.model.layers[0].mlp).__name__.startswith("Oqh")

    mx.random.seed(5)
    tensors = {}
    twin = dict(ref.model.layers[0].named_modules())
    for name in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        out_dims, in_dims = twin[name].weight.shape
        key = f"base_model.model.model.layers.0.{name}"
        tensors[f"{key}.lora_A.weight"] = mx.random.normal((8, in_dims)) * 0.05
        tensors[f"{key}.lora_B.weight"] = mx.random.normal((out_dims, 8)) * 0.05
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8, "lora_alpha": 32})
    )
    mx.save_safetensors(str(adapter / "adapter_model.safetensors"), tensors)

    ids = mx.array([[1, 5, 9, 2, 7, 3, 11, 4]])
    base = ref(ids).astype(mx.float32)
    assert apply_adapter(model, adapter) == apply_adapter(ref, adapter) == 3
    mlp = model.model.layers[0].mlp
    assert not type(mlp).__name__.startswith("Oqh")
    assert isinstance(mlp.gate_proj, LoRALinear)
    want = ref(ids).astype(mx.float32)
    assert _rel(want, base) > 0.05
    assert _rel(model(ids).astype(mx.float32), want) < 0.03


def _tiny_qwen4_exp(tmp_path, monkeypatch):
    """A tiny Qwen4-Exp source whose experts the oQh kernels can serve."""
    from dataclasses import asdict

    from mlx.utils import tree_flatten

    from omlx import oq
    from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config

    # _tiny_config installs the vendored Qwen4-Exp model first.
    cfg = _tiny_config()
    from mlx_vlm.models.qwen4_exp.language import configure_ple_runtime
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    cfg.text_config.hidden_size = 256
    # 384 = 3 x 128: the down projection rotates 128-wide blocks, as in
    # Qwen3.8-Flash-Next (640 = 5 x 128).
    cfg.text_config.moe_intermediate_size = 384
    cfg.text_config.linear_key_head_dim = 32
    cfg.text_config.linear_value_head_dim = 32
    source = tmp_path / "source"
    source.mkdir()
    configure_ple_runtime(source, mode="resident")
    mx.random.seed(7)
    model = Model(cfg)
    model.eval()
    weights = {
        k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v
        for k, v in tree_flatten(model.parameters())
    }
    config = asdict(cfg)
    config["architectures"] = ["Qwen4ExpForConditionalGeneration"]
    (source / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    tokens = mx.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], dtype=mx.int32)
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
    monkeypatch.setattr("mlx_lm.tokenizer_utils.load", lambda *a, **kw: object())
    return source, tokens


@pytest.mark.parametrize("level", [3.5, 4])
def test_qwen4_exp_experts_round_trip(tmp_path, monkeypatch, level):
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen3_5.speculative_verifier import (
        Qwen3_5BatchInvariantForward,
    )
    from mlx_vlm.utils import load_model

    from omlx import oq
    from omlx.oqh.loading import install_load_hook
    from omlx.patches.mlx_vlm_qwen4_exp_compat import configure_qwen4_exp_runtime

    source, tokens = _tiny_qwen4_exp(tmp_path, monkeypatch)
    out = tmp_path / "oqh"
    oq.quantize_oq_streaming(
        str(source),
        str(out),
        level,
        hadamard=True,
        stream_calibration=True,
        imatrix_num_samples=2,
        imatrix_seq_length=6,
        sensitivity_map_override={0: 1, 1: 1},
    )
    config = json.loads((out / "config.json").read_text())
    entries = config[fmt.CONFIG_KEY]["tensors"]
    # Only the routed experts: the trunk keeps its affine fast paths.
    assert {p.rsplit(".", 2)[-2] + "." + p.rsplit(".", 1)[-1] for p in entries} == {
        "switch_mlp.gate_proj",
        "switch_mlp.up_proj",
        "switch_mlp.down_proj",
    }
    assert {e["block"] for p, e in entries.items() if p.endswith("down_proj")} == {128}
    # oQ4 keeps group scales on the rotated experts.
    want_group = fmt.AFFINE_GROUP_SIZE if level >= 4 else None
    assert {e.get("group_size") for e in entries.values()} == {want_group}

    # Twin: the same checkpoint with the experts stored as dequantized BF16.
    twin = tmp_path / "twin"
    twin.mkdir()
    weights = {}
    for f in out.glob("*.safetensors"):
        weights.update(mx.load(str(f)))
    for path, e in entries.items():
        signs = weights.pop(f"{path}.oqh_signs").reshape(-1).astype(mx.float32)
        packed = weights[f"{path}.weight"]
        if "group_size" in e:
            scales = weights.pop(f"{path}.oqh_scales")
            biases = weights.pop(f"{path}.oqh_biases")
            w = quantize.dequantize_affine(
                packed, scales, biases, e["bits"], signs, e["block"]
            )
        else:
            rscale = weights.pop(f"{path}.rscale")
            w = quantize.dequantize(packed, rscale, e["bits"], signs, e["block"])
        weights[f"{path}.weight"] = w.astype(mx.bfloat16)
    mx.save_safetensors(str(twin / "model.safetensors"), weights)
    del config[fmt.CONFIG_KEY]
    (twin / "config.json").write_text(json.dumps(config))

    install_load_hook()
    models = {}
    for name, path in (("oqh", out), ("twin", twin)):
        configure_qwen4_exp_runtime(path, mode="resident", mtp_enabled=False)
        models[name] = load_model(path, lazy=False, strict=True)
    switches = [
        m for _, m in models["oqh"].named_modules() if hasattr(m, "oqh_experts")
    ]
    assert len(switches) == 2
    assert not any("rscale" in k for k, _ in tree_flatten(models["twin"].parameters()))
    want = models["twin"](tokens).logits.astype(mx.float32)
    assert _rel(models["oqh"](tokens).logits, want) < 0.03

    # Row-exact MTP verify runs the experts with the one-token decode arithmetic.
    switch = switches[0]
    x = mx.random.normal((1, 3, 256)).astype(mx.bfloat16)
    inds = mx.array([[[0, 3], [1, 2], [3, 1]]])
    verify = Qwen3_5BatchInvariantForward._switch_glu(None, switch, x, inds)
    for i in range(3):
        one = switch(x[:, i : i + 1], inds[:, i : i + 1])
        assert mx.array_equal(verify[:, i : i + 1], one)


def _tiny_glm5_next(tmp_path, monkeypatch):
    """A tiny GLM-5.3 source with one sparse layer the oQh kernels can serve."""
    from mlx.utils import tree_flatten

    from omlx import oq
    from omlx.patches import mlx_vlm_glm5_next_compat
    from tests.test_mlx_vlm_glm5_next_compat import _tiny_config

    mlx_vlm_glm5_next_compat.apply_mlx_vlm_glm5_next_compat_patch()
    from mlx_vlm.models.glm5_next import Model

    cfg = _tiny_config()
    text = cfg.text_config
    text.hidden_size = 256
    text.moe_intermediate_size = 256
    text.n_routed_experts = 4
    text.n_shared_experts = 1
    text.first_k_dense_replace = 1
    text.mlp_layer_types = ["dense", "sparse"]
    text.index_topk = 2048
    source = tmp_path / "source"
    source.mkdir()
    mx.random.seed(11)
    model = Model(cfg)
    weights = {
        k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v
        for k, v in tree_flatten(model.parameters())
    }
    config = dict(vars(text))
    config["linear_attn_config"] = {
        "num_heads": text.linear_num_heads,
        "head_dim": text.linear_head_dim,
        "short_conv_kernel_size": text.linear_conv_kernel_dim,
        "gate_lower_bound": text.linear_lower_bound,
    }
    payload = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForCausalLM"],
        "text_config": config,
    }
    (source / "config.json").write_text(json.dumps(payload))
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    tokens = mx.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], dtype=mx.int32)
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
    monkeypatch.setattr("mlx_lm.tokenizer_utils.load", lambda *a, **kw: object())
    return source, tokens


@pytest.mark.parametrize("level", [3.5, 4], ids=["row", "group"])
def test_glm5_next_experts_round_trip(tmp_path, monkeypatch, level):
    from mlx.utils import tree_flatten
    from mlx_vlm.utils import load_model

    from omlx import oq
    from omlx.oqh.loading import install_load_hook

    source, tokens = _tiny_glm5_next(tmp_path, monkeypatch)
    out = tmp_path / "oqh"
    oq.quantize_oq_streaming(
        str(source),
        str(out),
        level,
        hadamard=True,
        imatrix_num_samples=2,
        imatrix_seq_length=6,
        sensitivity_map_override={0: 1, 1: 1},
    )
    config = json.loads((out / "config.json").read_text())
    entries = config[fmt.CONFIG_KEY]["tensors"]
    assert {p.split(".mlp.")[-1] for p in entries} == {
        "switch_mlp.gate_proj",
        "switch_mlp.up_proj",
        "switch_mlp.down_proj",
    }
    # TP2 splits the down projection's 256 inputs: 128-wide blocks.
    assert {e["block"] for p, e in entries.items() if p.endswith("down_proj")} == {128}
    want_group = fmt.AFFINE_GROUP_SIZE if level >= fmt.AFFINE_LEVEL else None
    assert {e.get("group_size") for e in entries.values()} == {want_group}

    twin = tmp_path / "twin"
    twin.mkdir()
    weights = {}
    for f in out.glob("*.safetensors"):
        weights.update(mx.load(str(f)))
    for path, e in entries.items():
        signs = weights.pop(f"{path}.oqh_signs").reshape(-1).astype(mx.float32)
        if "group_size" in e:
            w = quantize.dequantize_affine(
                weights[f"{path}.weight"],
                weights.pop(f"{path}.oqh_scales"),
                weights.pop(f"{path}.oqh_biases"),
                e["bits"],
                signs,
                e["block"],
            )
        else:
            w = quantize.dequantize(
                weights[f"{path}.weight"],
                weights.pop(f"{path}.rscale"),
                e["bits"],
                signs,
                e["block"],
            )
        weights[f"{path}.weight"] = w.astype(mx.bfloat16)
    mx.save_safetensors(str(twin / "model.safetensors"), weights)
    del config[fmt.CONFIG_KEY]
    (twin / "config.json").write_text(json.dumps(config))

    install_load_hook()
    oqh_model = load_model(out, lazy=False, strict=True)
    twin_model = load_model(twin, lazy=False, strict=True)
    switches = [m for _, m in oqh_model.named_modules() if hasattr(m, "oqh_experts")]
    assert len(switches) == 1
    assert not any("rscale" in k for k, _ in tree_flatten(twin_model.parameters()))
    want = twin_model(tokens).logits.astype(mx.float32)
    assert _rel(oqh_model(tokens).logits, want) < 0.03

    # GLM passes its routing scores; the decode rows match one-token calls.
    switch = switches[0]
    x = mx.random.normal((1, 3, 256)).astype(mx.bfloat16)
    inds = mx.array([[[0, 3], [1, 2], [3, 1]]])
    scores = mx.full((1, 3, 2), 0.5, dtype=mx.float32)
    rows = switch(x, inds, scores=scores, weighted_sum=True)
    for i in range(3):
        one = switch(x[:, i : i + 1], inds[:, i : i + 1], scores=scores[:, i : i + 1])
        assert mx.array_equal(rows[:, i : i + 1], one)


@pytest.mark.parametrize("group_size", [None, 128], ids=["row", "group"])
def test_glm5_next_fused_decode_kernels_read_oqh_experts(group_size):
    from omlx.oqh import runtime
    from omlx.patches import mlx_vlm_glm5_next_compat
    from tests.test_mlx_vlm_glm5_next_compat import _tiny_config

    mlx_vlm_glm5_next_compat.apply_mlx_vlm_glm5_next_compat_patch()
    from mlx_vlm.models.glm5_next import language

    # The fused kernels need 512-aligned inputs at 4 bits; the router kernel
    # needs 16 or more experts and fewer than 16 inputs per expert.
    text = _tiny_config().text_config
    text.hidden_size = 512
    text.moe_intermediate_size = 512
    text.n_routed_experts = 64
    text.n_shared_experts = 1
    text.num_experts_per_tok = 2
    text.swiglu_limit = 10.0
    moe = language.Glm5NextMoE(text)
    moe.switch_mlp.gate_proj = _switch(64, 512, 512, "in", 1, group_size)
    moe.switch_mlp.up_proj = _switch(64, 512, 512, "in", 2, group_size)
    moe.switch_mlp.down_proj = _switch(64, 512, 512, "down", 3, group_size)
    moe.shared_experts.set_dtype(mx.bfloat16)
    nn.quantize(moe.shared_experts, group_size=64, bits=4)
    moe.gate.weight = mx.random.normal(moe.gate.weight.shape)
    runtime._install_mlp(moe.switch_mlp)
    mx.eval(moe.parameters())

    def reference(x):
        indices, scores = moe.gate(x)
        y = moe.switch_mlp(x, indices)
        y = (y * scores[..., None].astype(y.dtype)).sum(axis=-2)
        return (y + moe.shared_experts(x)).astype(mx.float32)

    x = (mx.random.normal((1, 1, 512)) * 4).astype(mx.bfloat16)
    fused = moe._decode_select(x)
    assert fused is not None
    assert _rel(fused, reference(x)) < 0.03
    rows = (mx.random.normal((1, 3, 512)) * 4).astype(mx.bfloat16)
    indices, scores = moe.gate(rows)
    verify = moe._decode_experts(rows, indices, scores)
    assert verify is not None
    assert _rel(verify, reference(rows)) < 0.03


def _tiny_hy_v4(tmp_path, monkeypatch, kind):
    """A tiny Hy4 or Hy4-flash source whose routed experts oQh can serve.

    All experts are routed per token so tiny rounding differences cannot flip
    a selection, and the index top-k exceeds the prompt.
    """
    from mlx.utils import tree_flatten

    from omlx import oq
    from omlx.patches import mlx_lm_mtp
    from omlx.patches.hy_v4 import apply_patch
    from tests import test_hy_v4

    apply_patch()
    # oQ's calibration loader leaves the process-wide MTP flag as the
    # pre-load patches set it; keep that out of later tests.
    monkeypatch.setattr(mlx_lm_mtp, "_MTP_ACTIVE", mlx_lm_mtp.is_mtp_active())
    cfg = test_hy_v4._hy4_config() if kind == "hy4" else test_hy_v4._flash_config()
    text = cfg["text_config"]
    text.update(
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        n_routed_experts=4,
        num_experts_per_tok=4,
        index_topk=64,
    )
    if kind == "flash":
        # Latent experts narrower than the norm in front of them.
        text.update(hidden_size=512, routed_expert_hidden_size=256)
    cfg = test_hy_v4._with_dfly(cfg)
    model = test_hy_v4._model(cfg)
    weights = {
        k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v
        for k, v in tree_flatten(model.parameters())
    }
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    tokens = mx.array([[10 + (7 * i) % 250 for i in range(32)]], dtype=mx.int32)
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
    monkeypatch.setattr("mlx_lm.tokenizer_utils.load", lambda *a, **kw: object())
    return source, tokens


@pytest.mark.parametrize("kind", ["hy4", "flash"])
def test_hy_v4_experts_round_trip(tmp_path, monkeypatch, kind):
    from mlx.utils import tree_flatten
    from mlx_vlm.utils import load_model

    from omlx import oq
    from omlx.oqh import runtime

    source, tokens = _tiny_hy_v4(tmp_path, monkeypatch, kind)
    out = tmp_path / "oqh"
    oq.quantize_oq_streaming(
        str(source),
        str(out),
        4,
        hadamard=True,
        imatrix_num_samples=1,
        imatrix_seq_length=32,
        sensitivity_map_override={i: 1 for i in range(6)},
        preserve_mtp=True,
    )
    config = json.loads((out / "config.json").read_text())
    entries = config[fmt.CONFIG_KEY]["tensors"]
    assert {p.split(".mlp.")[-1] for p in entries} == {
        "switch_mlp.gate_proj",
        "switch_mlp.up_proj",
        "switch_mlp.down_proj",
    }
    assert config["dfly_num_hidden_layers"] == 2

    twin = tmp_path / "twin"
    twin.mkdir()
    weights = {}
    for f in out.glob("*.safetensors"):
        weights.update(mx.load(str(f)))
    for path, e in entries.items():
        signs = weights.pop(f"{path}.oqh_signs").reshape(-1).astype(mx.float32)
        w = quantize.dequantize_affine(
            weights[f"{path}.weight"],
            weights.pop(f"{path}.oqh_scales"),
            weights.pop(f"{path}.oqh_biases"),
            e["bits"],
            signs,
            e["block"],
        )
        weights[f"{path}.weight"] = w.astype(mx.bfloat16)
    mx.save_safetensors(str(twin / "model.safetensors"), weights)
    del config[fmt.CONFIG_KEY]
    (twin / "config.json").write_text(json.dumps(config))

    install_load_hook()
    oqh_model = load_model(out, lazy=False, strict=True)
    twin_model = load_model(twin, lazy=False, strict=True)
    switches = [m for _, m in oqh_model.named_modules() if hasattr(m, "oqh_experts")]
    assert len(switches) == 5
    assert not any("rscale" in k for k, _ in tree_flatten(twin_model.parameters()))
    # The flash experts read a latent projection, so no norm may rotate for them.
    norms = [layer.post_attention_layernorm for layer in oqh_model.layers]
    fused = any(isinstance(n, runtime._DualNorm) for n in norms)
    assert fused == (kind == "hy4")
    want = twin_model(tokens).logits.astype(mx.float32)
    assert _rel(oqh_model(tokens).logits, want) < 0.03


def test_deepseek_v41_oqh_experts_export_and_load(tmp_path, monkeypatch):
    from omlx import oq
    from omlx.oqh.modules import OqhSwitchLinear
    from omlx.patches.deepseek_v41.loading import load
    from omlx.patches.deepseek_v41.oq import quantize as quantize_v41
    from tests.test_deepseek_v41 import write_checkpoint

    source, _ = write_checkpoint(tmp_path, vision=False, dim=256, moe_inter_dim=256)
    tokens = mx.array([[3, 4, 5, 6, 7, 8, 9, 10], [9, 8, 7, 6, 5, 4, 3, 11]])
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
    out = tmp_path / "oqh"
    quantize_v41(
        source,
        out,
        oq_level=3,
        enhanced=True,
        hadamard=True,
        target_bpw=32,
        hard_cap_bpw=32,
        imatrix_num_samples=2,
        imatrix_seq_length=8,
        sensitivity_map_override={i: 1.0 for i in range(5)},
    )
    config = json.loads((out / "config.json").read_text())
    entries = config[fmt.CONFIG_KEY]["tensors"]
    # Only the routed experts; the trunk and shared expert keep their oQ format.
    assert {n.rsplit(".", 1)[-1] for n in entries} == {"w1", "w2", "w3"}
    assert all(".ffn.experts." in n for n in entries)
    assert not set(entries) & set(config["omlx_deepseek_v41"]["quantized_modules"])

    # Twin: the same export with the experts stored as dequantized BF16.
    twin = tmp_path / "twin"
    twin.mkdir()
    weights = {}
    for f in out.glob("*.safetensors"):
        weights.update(mx.load(str(f)))
    for path, e in entries.items():
        signs = weights.pop(f"{path}.oqh_signs").reshape(-1).astype(mx.float32)
        w = quantize.dequantize(
            weights[f"{path}.weight"],
            weights.pop(f"{path}.rscale"),
            e["bits"],
            signs,
            e["block"],
        )
        weights[f"{path}.weight"] = w.astype(mx.bfloat16)
    mx.save_safetensors(str(twin / "model.safetensors"), weights)
    for f in out.iterdir():
        if f.suffix != ".safetensors" and f.name not in (
            "config.json",
            "model.safetensors.index.json",
        ):
            (twin / f.name).write_bytes(f.read_bytes())
    del config[fmt.CONFIG_KEY]
    (twin / "config.json").write_text(json.dumps(config))
    (twin / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )

    model, _ = load(out)
    reference, _ = load(twin)
    try:
        experts = model.language_model.layers[0].ffn.experts
        assert isinstance(experts.w1, OqhSwitchLinear)
        # Same routes on both sides: routing flips in a tiny random model
        # would hide the expert arithmetic.
        x = mx.random.normal((1, 4, 1, 1, 256))
        idx = mx.array([[[0, 1], [2, 3], [1, 2], [3, 0]]])
        weights = mx.full((1, 4, 2), 0.5)
        want = reference.language_model.layers[0].ffn.experts(x, idx, weights)
        assert _rel(experts(x, idx, weights), want) < 0.02
        assert mx.isfinite(model(mx.array([[3, 4, 5, 6, 7]]))).all().item()
    finally:
        model.close()
        reference.close()


def test_deepseek_v41_oqh_stages_match_one_forward(tmp_path, monkeypatch):
    """Rows carried across stages give the experts of one forward over all layers."""
    from omlx import oq
    from omlx.patches.deepseek_v41.oqh import write_experts
    from tests.test_deepseek_v41 import write_checkpoint

    source, _ = write_checkpoint(tmp_path, vision=False, dim=256, moe_inter_dim=256)
    tokens = mx.array([[3, 4, 5, 6, 7, 8, 9, 10], [9, 8, 7, 6, 5, 4, 3, 11]])
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
    widths = {
        f"language_model.layers.{i}.ffn.experts.{part}": 3
        for i in range(5)
        for part in ("w1", "w3", "w2")
    }

    class Writer:
        def __init__(self):
            self.values = {}

        def add(self, values):
            self.values.update(values)

    outputs = []
    # One layer per stage carries every row's stream four times; five per stage
    # is a single forward.
    for per_pass in (1, 5):
        writer = Writer()
        write_experts(
            source,
            writer,
            widths,
            num_samples=2,
            seq_length=8,
            layers_per_pass=per_pass,
        )
        outputs.append(writer.values)
    assert outputs[0].keys() == outputs[1].keys()
    for key, value in outputs[0].items():
        assert mx.array_equal(value, outputs[1][key]), key


def test_deepseek_v41_tensor_split_keeps_oqh_blocks():
    from omlx.patches.deepseek_v41.config import TensorSplit
    from omlx.patches.deepseek_v41.loading import tensor_slice

    up = "language_model.layers.0.ffn.experts.w1"
    down = "language_model.layers.0.ffn.experts.w2"
    rscale = mx.arange(4 * 8).reshape(4, 8)
    signs = mx.arange(8 * 128).reshape(8, 128)
    # w1 / w3 split their rows: row scales follow, input signs stay whole.
    assert tensor_slice(f"{up}.rscale", rscale, 1, 2).tolist() == rscale[:, 4:].tolist()
    assert tensor_slice(f"{up}.oqh_signs", signs, 1, 2) is signs
    # w2 splits its inputs: whole blocks of signs, every row scale.
    assert tensor_slice(f"{down}.rscale", rscale, 1, 2) is rscale
    assert tensor_slice(f"{down}.oqh_signs", signs, 1, 2).tolist() == signs[4:].tolist()
    # An uneven split moves whole blocks; one finer than a block is refused.
    split = TensorSplit((1, 1), (3, 1))
    assert (
        tensor_slice(f"{up}.rscale", rscale, 1, 2, split).tolist()
        == rscale[:, 6:].tolist()
    )
    assert (
        tensor_slice(f"{down}.oqh_signs", signs, 0, 2, split).tolist()
        == signs[:6].tolist()
    )
    with pytest.raises(ValueError):
        tensor_slice(f"{down}.oqh_signs", signs, 0, 2, TensorSplit((1, 1), (15, 1)))
