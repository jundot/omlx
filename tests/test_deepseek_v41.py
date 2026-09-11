"""Small-weight regression checks against recorded official outputs.

Expected values come from the independent PyTorch reference with CPU kernel
substitutions. See fixtures/deepseek_v41_expected.md for provenance and limits.
"""

import sys
import types
import zlib
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.language import LanguageModel


def tiny(**kwargs):
    values = dict(
        vocab_size=64,
        dim=32,
        moe_inter_dim=32,
        n_layers=5,
        n_heads=2,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=8,
        o_groups=2,
        o_lora_rank=16,
        index_n_heads=8,
        index_head_dim=32,
        index_topk=3,
        n_routed_experts=4,
        n_activated_experts=2,
        window_size=4,
        dtype="bf16",
        expert_dtype=None,
        n_mtp_layers=0,
        max_seq_len=64,
        max_batch_size=2,
        hc_sinkhorn_iters=3,
        candidate_source_layer=3,
        candidate_block_size=2,
        candidate_topk_blocks=2,
        index_source_layers=(1, 3, 4),
    )
    values.update(kwargs)
    return ModelConfig(**values)


def cpu_kernels(torch):
    mod = types.ModuleType("kernel")

    def act_quant(x, block_size=32, scale_fmt=None, scale_dtype=None, inplace=False):
        y = x.float().unflatten(-1, (-1, block_size))
        scale = (
            (y.abs().amax(-1, keepdim=True).clamp_min(448 * 2**-126) / 448)
            .log2()
            .ceil()
            .exp2()
        )
        out = (
            ((y / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale)
            .flatten(-2)
            .to(x.dtype)
        )
        if inplace:
            x.copy_(out)
        return out

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
        y = x.float().unflatten(-1, (-1, block_size))
        e4 = scale_dtype == torch.float8_e4m3fn
        scale = (
            y.abs().amax(-1, keepdim=True).clamp_min(6 * (2**-9 if e4 else 2**-126)) / 6
        )
        scale = (
            scale.to(torch.float8_e4m3fn).float() if e4 else scale.log2().ceil().exp2()
        )
        table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6.0])
        a = (y / scale).abs().clamp_max(6)
        # Independent nearest-code selection, preferring even codes on ties.
        order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
        index = (a[..., None] - table[order]).abs().argmin(-1)
        result = (torch.sign(y) * table[order[index]] * scale).flatten(-2).to(x.dtype)
        if inplace:
            x.copy_(result)
        return result

    def sparse_attn(q, kv, sink, indices, scale):
        selected = kv[0, indices[0].clamp_min(0).long()][None].float()
        scores = torch.einsum("bshd,bskd->bshk", q.float(), selected) * scale
        scores = scores.masked_fill(indices[:, :, None] < 0, -torch.inf)
        sink = sink[None, None, :, None].expand(*scores.shape[:-1], 1)
        probabilities = torch.cat([scores, sink], -1).softmax(-1)[..., :-1]
        return torch.einsum("bshk,bskd->bshd", probabilities, selected).to(q.dtype)

    def hc_split_sinkhorn(mixes, scale, base, n, iterations, eps):
        pre = (mixes[..., :n] * scale[0] + base[:n]).sigmoid() + eps
        post = 2 * (mixes[..., n : 2 * n] * scale[1] + base[n : 2 * n]).sigmoid()
        comb = (mixes[..., 2 * n :] * scale[2] + base[2 * n :]).unflatten(
            -1, (n, n)
        ).softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(iterations - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    mod.act_quant, mod.fp4_act_quant = act_quant, fp4_act_quant
    mod.sparse_attn, mod.hc_split_sinkhorn = sparse_attn, hc_split_sinkhorn

    def unavailable(*args, **kwargs):
        raise AssertionError("This reference test uses float weights, not CUDA GEMM")

    mod.fp4_gemm = mod.fp8_gemm = unavailable
    return mod


@pytest.fixture
def expected():
    with np.load(Path(__file__).parent / "fixtures/deepseek_v41_expected.npz") as data:
        return dict(data)


def load_reference_weights(model, seed=341, scale=0.08, vision=False):
    from mlx.utils import tree_flatten

    def values(name, shape):
        rng = np.random.default_rng(seed + zlib.crc32(name.encode()))
        data = rng.normal(0, scale, shape).astype(np.float32)
        if (
            ("norm" in name and name.endswith("weight"))
            if vision
            else "norm.weight" in name
        ):
            data += 1
        if "weights_proj.weight" in name:
            data = np.abs(data)
        return data

    weights = []
    for name, parameter in tree_flatten(model.parameters()):
        original = name.replace(".ffn.", ".mlp.") if vision else name
        if ".experts." in original and parameter.ndim == 3:
            prefix, suffix = original.split(".experts.")
            data = np.stack(
                [
                    values(f"{prefix}.experts.{expert}.{suffix}", parameter.shape[1:])
                    for expert in range(parameter.shape[0])
                ]
            )
        else:
            data = values(original, parameter.shape)
        weights.append((name, mx.array(data)))
    model.load_weights(weights, strict=True)


@pytest.mark.parametrize("length", [1, 3, 8, 17])
def test_official_prefill_decode(expected, length):
    model = LanguageModel(tiny())
    load_reference_weights(model)
    ids = np.random.default_rng(5).integers(3, 60, (1, length + 3), dtype=np.int32)
    cache = model.make_cache()
    actual = np.asarray(model(mx.array(ids[:, :length]), cache=cache))[:, -1]
    np.testing.assert_allclose(
        actual, expected[f"prefill{length}_0"], atol=2e-5, rtol=2e-4
    )
    for index, position in enumerate(range(length, length + 3), 1):
        actual = np.asarray(
            model(mx.array(ids[:, position : position + 1]), cache=cache)
        )[:, -1]
        np.testing.assert_allclose(
            actual, expected[f"prefill{length}_{index}"], atol=3e-5, rtol=3e-4
        )


def test_chunk_boundaries_and_late_join():
    mx.random.seed(32)
    model = LanguageModel(tiny())
    a, b = mx.array([[4, 5, 6, 7, 8, 9, 10]]), mx.array([[12, 13, 14]])
    ca, cb = model.make_cache(), model.make_cache()
    expected = model(a)
    actual = mx.concatenate(
        [
            model(a[:, :3], cache=ca),
            model(a[:, 3:5], cache=ca),
            model(a[:, 5:], cache=ca),
        ],
        1,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    model(b, cache=cb)
    merged = [DeepseekV41Cache.merge([x, y]) for x, y in zip(ca, cb)]
    next_ids = mx.array([[15], [16]])
    result = model(next_ids, cache=merged)
    solo_a = model(mx.concatenate([a, next_ids[:1]], 1))[:, -1:]
    solo_b = model(mx.concatenate([b, next_ids[1:]], 1))[:, -1:]
    np.testing.assert_allclose(result, mx.concatenate([solo_a, solo_b], 0), atol=1e-5)
    for c in merged:
        c.filter(mx.array([1]))
    result = model(mx.array([[17]]), cache=merged)
    np.testing.assert_allclose(
        result, model(mx.array([[12, 13, 14, 16, 17]]))[:, -1:], atol=1e-5
    )


def test_left_padding():
    model = LanguageModel(tiny())
    cache = model.make_cache()
    for c in cache:
        c.left_padding = mx.array([0, 3])
    x = mx.array([[3, 4, 5, 6, 7], [0, 0, 0, 8, 9]])
    actual = model(x, cache=cache)
    np.testing.assert_allclose(actual[1:2, -1:], model(x[1:2, -2:])[:, -1:], atol=1e-5)
    assert cache[0].offset.tolist() == [5, 2]


def test_engram_hash_history_and_image_boundary(expected):
    from omlx.patches.deepseek_v41.engram import NgramHash

    c = tiny(
        engram_layer_ids=(1,),
        engram_num_embeddings=(72,),
        engram_vocab_size=5,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=64,
    )
    hasher = NgramHash(c, np.arange(64))
    ids = np.array([[3, 4, 5, 6, 7, 8, 9]])
    image = np.array([[False, False, True, True, False, False, False]])
    first, history = hasher(ids[:, :3], image_mask=image[:, :3])
    second, history = hasher(ids[:, 3:], history, image[:, 3:])
    np.testing.assert_array_equal(
        np.concatenate([first, second], 1), expected["hash_0"]
    )
    np.testing.assert_array_equal(history, [[7, 8, 9]])
    with pytest.raises(ValueError, match="vocabulary"):
        NgramHash(c, np.arange(63))


def test_vision_and_aligner_reference(expected):
    from omlx.patches.deepseek_v41.vision import Aligner, ViT

    c = tiny(
        vision_n_layers=2,
        vision_dim=32,
        vision_n_heads=4,
        vision_inter_dim=32,
        vision_patch_size=2,
    )
    rng = np.random.default_rng(7)
    for index, cls in enumerate([ViT, Aligner]):
        model = cls(c)
        load_reference_weights(model, seed=7, scale=0.1, vision=True)
        x = rng.normal(0, 1, (20, 3, 2, 2) if cls is ViT else (20, 32)).astype(
            np.float32
        )
        np.testing.assert_allclose(
            model(mx.array(x), 4, 5), expected[f"vision_{index}"], atol=2e-6, rtol=1e-4
        )


@pytest.mark.parametrize(
    "shape", [(120, 50), (30, 190), (1, 2000), (2000, 1), (12, 12)]
)
def test_image_processor_matches_reference(expected, shape):
    from PIL import Image

    from omlx.patches.deepseek_v41.processing import image_patches

    c = tiny(
        vision_n_layers=1,
        vision_patch_size=2,
        vision_min_pixels=64,
        vision_max_n_token=32,
    )
    actual, ah, aw, types_ = image_patches(Image.new("RGB", shape, (19, 101, 220)), c)
    if shape in ((1, 2000), (2000, 1)):
        assert actual.shape[0] == ah * aw and len(types_) <= c.vision_max_n_token
        return
    key = f"image{shape[0]}x{shape[1]}"
    np.testing.assert_array_equal([ah, aw], expected[f"{key}_grid"])
    np.testing.assert_array_equal(actual.astype(mx.float32), expected[f"{key}_patch"])
    np.testing.assert_array_equal(types_, expected[f"{key}_types"])


def test_fp8_and_fp4_roundtrip():
    from omlx.patches.deepseek_v41.quantization import quantize_activation

    torch = pytest.importorskip("torch")
    kernel = cpu_kernels(torch)
    rng = np.random.default_rng(8)
    raw = np.concatenate([rng.normal(0, 4, (4, 32)), np.zeros((1, 32))]).astype(
        np.float32
    )
    np.testing.assert_array_equal(
        quantize_activation(mx.array(raw)),
        kernel.act_quant(torch.tensor(raw), inplace=True).numpy(),
    )
    for group, scale in [(32, None), (16, torch.float8_e4m3fn)]:
        expected = kernel.fp4_act_quant(torch.tensor(raw), group, True, scale)
        np.testing.assert_array_equal(
            quantize_activation(mx.array(raw), 4, group, scale is not None),
            expected.numpy(),
        )


def test_mmap_engram_selected_rows_and_close(tmp_path):
    from omlx.patches.deepseek_v41.storage import DiskEngramEmbedding

    path = tmp_path / "table.safetensors"
    # Byte-compatible E4M3 data and E8M0 scales represented by source dtypes.
    values = mx.arange(8 * 32, dtype=mx.float32).reshape(8, 32) / 32
    mx.save_safetensors(str(path), {"weight": values.astype(mx.bfloat16)})
    table = DiskEngramEmbedding(path, "weight", None)
    actual = table(mx.array([[6, 1, 6]]))
    table.close()
    np.testing.assert_array_equal(
        actual.astype(mx.float32), values[mx.array([[6, 1, 6]])]
    )
    with pytest.raises(RuntimeError, match="closed"):
        table(mx.array([0]))


def test_cache_handler_restores_type_and_continuation():
    from omlx.cache.type_registry import CacheTypeRegistry
    from omlx.patches.deepseek_v41 import apply_patch

    apply_patch()
    model = LanguageModel(tiny())
    cache = model.make_cache()
    model(mx.array([[3, 4, 5, 6, 7]]), cache=cache)
    restored = []
    for item in cache:
        handler = CacheTypeRegistry.get_handler_by_class_name(type(item).__name__)
        state = handler.serialize_state(item)
        restored.append(handler.deserialize_state(state, item.meta_state))
    assert all(isinstance(c, DeepseekV41Cache) for c in restored)
    np.testing.assert_allclose(
        model(mx.array([[9]]), cache=restored),
        model(mx.array([[3, 4, 5, 6, 7, 9]]))[:, -1:],
        atol=1e-5,
    )


def write_checkpoint(tmp_path, vision=True, **config_overrides):
    import json
    import re

    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from omlx.patches.deepseek_v41.model import Model

    c = tiny(
        vision_n_layers=1 if vision else 0,
        vision_dim=32,
        vision_n_heads=4,
        vision_inter_dim=32,
        vision_patch_size=2,
        vision_min_pixels=64,
        vision_max_n_token=32,
        image_token_id=63,
        **config_overrides,
    )
    model = Model(c)
    source = tmp_path / "source"
    source.mkdir()
    weights = {}
    for name, value in tree_flatten(model.parameters()):
        name = name.removeprefix("language_model.")
        match = re.match(r"((?:layers|mtp)\.\d+\.ffn\.experts)\.(w[123])\.weight", name)
        if match:
            for e in range(value.shape[0]):
                weights[f"{match[1]}.{e}.{match[2]}.weight"] = value[e]
        else:
            weights[
                name.replace(".ffn.", ".mlp.") if name.startswith("vision.") else name
            ] = value
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    config = {
        "model_type": "deepseek_v41",
        "text_config": asdict(c),
        "image_token_id": 63,
    }
    if vision:
        config["vision_config"] = {
            "num_hidden_layers": 1,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "patch_size": 2,
            "max_image_tokens": 32,
            "min_pixels": 64,
        }
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    special = [
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<pad>",
        "hello",
        "world",
        "<unk>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<｜System｜>",
        "<think>",
        "</think>",
    ]
    vocabulary = (
        special + [f"t{i}" for i in range(len(special), 63)] + ["<｜deepseek_image｜>"]
    )
    backend = Tokenizer(
        models.WordLevel(dict(zip(vocabulary, range(64))), unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=special[0],
        eos_token=special[1],
        pad_token="<pad>",
        unk_token="<unk>",
        additional_special_tokens=special[6:] + [vocabulary[-1]],
    )
    tokenizer.save_pretrained(source)
    return source, model


def test_dspark_matches_official_single_pass_reference(expected):
    c = tiny(
        preserve_mtp=True,
        n_mtp_layers=3,
        dspark_block_size=3,
        dspark_noise_token_id=2,
        dspark_target_layer_ids=(2, 3, 4),
        dspark_n_routed_experts=2,
        dspark_n_activated_experts=1,
        dspark_markov_rank=32,
        compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0),
        temperature=0,
    )
    model = LanguageModel(c)
    load_reference_weights(model)
    target, draft = model.make_cache(), model.make_dspark_cache()
    index = 0
    for start, ids in [(0, [3, 4, 5, 6, 7]), (5, [8]), (6, [9])]:
        logits, hidden = model(mx.array([ids]), cache=target, return_dspark_hidden=True)
        np.testing.assert_allclose(
            hidden, expected[f"dspark_{index}"], atol=2e-5, rtol=2e-4
        )
        np.testing.assert_allclose(
            logits[:, -1], expected[f"dspark_{index + 1}"], atol=2e-5, rtol=2e-4
        )
        index += 2
        actual = model.forward_spec(mx.array([[10]]), hidden, draft)
        if start == 0:
            assert actual is None
        else:
            np.testing.assert_array_equal(actual[0], expected[f"dspark_{index}"])
            for component in [1, 2]:
                np.testing.assert_allclose(
                    actual[component],
                    expected[f"dspark_{index + component}"],
                    atol=3e-5,
                    rtol=3e-4,
                )
            index += 3
        assert all(item.offset == start + len(ids) for item in draft)


@pytest.mark.parametrize("preserve", [False, True])
def test_convert_preserves_complete_dspark(tmp_path, preserve):
    import json

    from mlx.utils import tree_flatten

    from omlx.patches.deepseek_v41.convert import convert
    from omlx.patches.deepseek_v41.loading import load

    source, before = write_checkpoint(
        tmp_path,
        preserve_mtp=True,
        n_mtp_layers=3,
        dspark_block_size=3,
        dspark_noise_token_id=2,
        dspark_target_layer_ids=(2, 3, 4),
        dspark_n_routed_experts=2,
        dspark_n_activated_experts=1,
        dspark_markov_rank=32,
        compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0),
    )
    target = tmp_path / "converted"
    convert(source, target, preserve_mtp=preserve)
    after, _ = load(target)
    direct, _ = load(source, preserve_mtp=preserve)
    try:
        expected = dict(tree_flatten(before.parameters()))
        for loaded in (after, direct):
            values = dict(tree_flatten(loaded.parameters()))
            assert any(".mtp." in name for name in values) == preserve
            for name, value in values.items():
                np.testing.assert_array_equal(value, expected[name])
            assert loaded.config.preserve_mtp == preserve
        config = json.loads((target / "config.json").read_text())
        assert config["omlx_deepseek_v41"]["preserve_mtp"] == preserve
    finally:
        after.close()
        direct.close()


def test_conversion_loading_and_image_path(tmp_path):
    from PIL import Image

    from omlx.models.vlm import VLMModelAdapter
    from omlx.patches.deepseek_v41.convert import convert
    from omlx.patches.deepseek_v41.loading import load

    source, before = write_checkpoint(tmp_path)
    target = tmp_path / "converted"
    convert(source, target)
    model, processor = load(target)
    images = [Image.new("RGB", (6, 8), "red"), Image.new("RGB", (8, 6), "blue")]
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image", "url": "first"},
                    {"type": "text", "text": "world"},
                    {"type": "image", "url": "second"},
                ],
            }
        ],
        enable_thinking=False,
    )
    inputs = processor(text=[prompt], images=images)
    features = model.get_input_embeddings(**inputs)
    expected = before(**inputs)
    adapter = VLMModelAdapter(model)
    actual = adapter(
        inputs["input_ids"],
        inputs_embeds=features.inputs_embeds,
        cache=adapter.make_cache(),
    )
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    model.close()


def test_batch_generator_ragged_prefill_and_admission():
    from mlx_lm.generate import BatchGenerator

    model = LanguageModel(tiny())
    generator = BatchGenerator(
        model,
        max_tokens=3,
        prefill_batch_size=2,
        prefill_step_size=3,
        sampler=lambda logits: mx.argmax(logits, -1),
    )
    uids = generator.insert([[3, 4, 5, 6, 7], [8, 9]])
    result = {uid: [] for uid in uids}
    while len(result[uids[0]]) < 1:
        for response in generator.next()[1]:
            result[response.uid].append(response.token)
    late = generator.insert([[10, 11, 12]])[0]
    result[late] = []
    for _ in range(20):
        prefill, outputs = generator.next()
        if not outputs and not prefill:
            break
        for response in outputs:
            result[response.uid].append(response.token)
    for uid, prompt in zip([*uids, late], [[3, 4, 5, 6, 7], [8, 9], [10, 11, 12]]):
        expected = []
        for _ in range(3):
            token = int(mx.argmax(model(mx.array([prompt + expected]))[0, -1]).item())
            expected.append(token)
        assert result[uid] == expected


def test_dsml_and_reasoning_encoding(tmp_path):
    from omlx.api.tool_calling import ToolCallStreamFilter, parse_tool_calls
    from omlx.patches.deepseek_v41.convert import convert
    from omlx.patches.deepseek_v41.loading import load

    source, _ = write_checkpoint(tmp_path, vision=False)
    convert(source, tmp_path / "mlx")
    model, processor = load(tmp_path / "mlx")
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": "hello"}], reasoning_effort=33
    )
    assert "Reasoning Effort: 33" in prompt and prompt.endswith("<think>")
    text = '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="weather">\n<｜DSML｜ parameter name="city" string="true">Seoul</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    visible, calls = parse_tool_calls(text, processor.tokenizer)
    assert len(calls) == 1 and calls[0].function.name == "weather"
    assert "Seoul" in calls[0].function.arguments
    stream_filter = ToolCallStreamFilter(processor.tokenizer)
    output = "".join(stream_filter.feed(char) for char in text) + stream_filter.finish()
    assert "<｜DSML｜" not in output
    model.close()


def test_mxfp_repacking_has_exact_source_values():
    from omlx.patches.deepseek_v41.convert import repack_weight

    torch = pytest.importorskip("torch")
    raw = np.arange(128, dtype=np.uint8).reshape(4, 32)
    scale = np.full((1, 1), 125, np.uint8)
    values, spec = repack_weight(raw, "F8_E4M3", scale, "F8_E8M0")
    actual = mx.dequantize(values["weight"], values["scales"], group_size=32, **spec)
    expected = torch.from_numpy(raw).view(torch.float8_e4m3fn).float().numpy() * 0.25
    # Code 127 is NaN and is outside a finite converted checkpoint.
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32))[:3], expected[:3]
    )
    raw4 = np.arange(64, dtype=np.uint8).reshape(4, 16)
    values, spec = repack_weight(raw4, "I8", np.full((4, 1), 127, np.uint8), "F8_E8M0")
    actual = mx.dequantize(values["weight"], values["scales"], group_size=32, **spec)
    table = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    expected = np.stack([table[raw4 & 15], table[raw4 >> 4]], -1).reshape(4, 32)
    np.testing.assert_array_equal(actual.astype(mx.float32), expected)


def test_engram_gate_matches_official(expected):
    from omlx.patches.deepseek_v41.engram import Engram

    c = tiny(
        engram_layer_ids=(1,),
        engram_num_embeddings=(72,),
        engram_vocab_size=5,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
    )
    model = Engram(c, 0)
    load_reference_weights(model, seed=41, scale=0.05)
    rng = np.random.default_rng(41)
    h = rng.normal(0, 1, (1, 3, 4, 32)).astype(np.float32)
    ids = rng.integers(0, 72, (1, 3, 6), dtype=np.int32)
    mask = np.array([[False, True, False]])
    actual = model(mx.array(h), ids, mx.array(mask))
    np.testing.assert_allclose(actual, expected["engram_0"], atol=2e-6, rtol=1e-5)
    np.testing.assert_array_equal(actual[:, 1], h[:, 1])


async def _run_vlm_engine(tmp_path, direct=False):
    import asyncio

    from PIL import Image

    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.patches.deepseek_v41.convert import convert
    from omlx.scheduler import SchedulerConfig

    source, _ = write_checkpoint(tmp_path)
    target = source if direct else tmp_path / "mlx"
    if not direct:
        convert(source, target)
    engine = VLMBatchedEngine(
        str(target),
        enable_thinking=False,
        scheduler_config=SchedulerConfig(prefill_step_size=4),
    )
    try:
        await engine.start()
        text = await asyncio.wait_for(
            engine.generate("hello world", max_tokens=3, temperature=0), 30
        )
        assert text.finished and text.completion_tokens > 0
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image", "url": "prepared"},
                    {"type": "text", "text": "world"},
                ],
            }
        ]
        inputs = engine._prepare_vision_inputs(
            messages, [Image.new("RGB", (6, 8), "red")]
        )
        ids, embeds, extras = inputs[:3]
        assert len(ids) == embeds.shape[1] and len(ids) > 3
        outputs = []
        async for chunk in engine.stream_generate(
            ids,
            max_tokens=3,
            temperature=0,
            vlm_inputs_embeds=embeds,
            vlm_extra_kwargs=extras,
        ):
            outputs.append(chunk)
        assert outputs and outputs[-1].finished
    finally:
        await engine.stop()


def raw_safetensors(path, tensors):
    import json
    import struct

    header, payload, offset = {}, [], 0
    for name, (value, dtype) in tensors.items():
        data = np.ascontiguousarray(value).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(value.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payload.append(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payload))


def test_mmap_fp8_rows_use_per_channel_scales(tmp_path):
    from omlx.patches.deepseek_v41.storage import DiskEngramEmbedding

    torch = pytest.importorskip("torch")
    raw = np.arange(5 * 64, dtype=np.uint8).reshape(5, 64) % 120
    scales = np.arange(10, dtype=np.uint8).reshape(5, 2) + 120
    path = tmp_path / "engram.safetensors"
    raw_safetensors(path, {"weight": (raw, "F8_E4M3"), "scale": (scales, "F8_E8M0")})
    table = DiskEngramEmbedding(path, "weight", "scale")
    actual = table(mx.array([[4, 0, 2, 4]]))
    table.close()
    selected = torch.from_numpy(raw[[4, 0, 2, 4]]).view(torch.float8_e4m3fn).float()
    scale = torch.from_numpy(np.exp2(scales[[4, 0, 2, 4]].astype(np.float32) - 127))
    expected = (
        (selected.reshape(4, 2, 32) * scale[..., None])
        .reshape(1, 4, 64)
        .to(torch.bfloat16)
        .float()
        .numpy()
    )
    np.testing.assert_array_equal(actual.astype(mx.float32), expected)


def test_quantized_conversion_and_loaded_projection(tmp_path):
    import json

    from omlx.patches.deepseek_v41.convert import convert
    from omlx.patches.deepseek_v41.loading import load
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection

    torch = pytest.importorskip("torch")
    source, _ = write_checkpoint(tmp_path, vision=False)
    name = "layers.0.attn.wq_a"
    fp8 = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32) % 120
    scale = np.full((1, 1), 124, np.uint8)
    path = source / "quant.safetensors"
    raw_safetensors(
        path, {name + ".weight": (fp8, "F8_E4M3"), name + ".scale": (scale, "F8_E8M0")}
    )
    data = mx.load(str(source / "model.safetensors"))
    data.pop(name + ".weight")
    mx.eval(data)
    mx.save_safetensors(str(source / "model.safetensors"), data)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    index["weight_map"].update(
        {name + ".weight": "quant.safetensors", name + ".scale": "quant.safetensors"}
    )
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    convert(source, tmp_path / "converted")
    model, _ = load(tmp_path / "converted")
    projection = model.language_model.layers[0].attn.wq_a
    assert isinstance(projection, QuantizedProjection)
    values = np.random.default_rng(14).normal(0, 0.1, (2, 32)).astype(np.float32)
    x = cpu_kernels(torch).act_quant(torch.tensor(values), inplace=True)
    weight = torch.from_numpy(fp8).view(torch.float8_e4m3fn).float() * 0.125
    expected = (x @ weight.T).numpy()
    np.testing.assert_allclose(
        projection(mx.array(values)), expected, atol=2e-6, rtol=1e-5
    )
    model.close()


@pytest.mark.parametrize("direct", [False, True])
def test_real_vlm_engine_text_and_images(tmp_path, direct):
    import subprocess

    script = (
        "import asyncio,sys; from pathlib import Path; "
        "sys.path.insert(0,sys.argv[1]); import test_deepseek_v41 as t; "
        "asyncio.run(t._run_vlm_engine(Path(sys.argv[2]), sys.argv[3] == '1'))"
    )
    # Engine startup installs process-wide Metal routes. Keep those real hooks
    # in a subprocess so unrelated estimator unit tests retain their fixtures.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).parent),
            str(tmp_path),
            "1" if direct else "0",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "bits,group,e4", [(8, 32, False), (4, 32, False), (4, 16, True)]
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_packed_cache_arithmetic_and_bytes(bits, group, e4, dtype):
    from omlx.patches.deepseek_v41.quantization import (
        pack_activation,
        quantize_activation,
        unpack_activation,
    )

    raw = np.random.default_rng(18).normal(0, 4, (2, 7, 512)).astype(np.float32)
    raw[0, 0] = 0
    x = mx.array(raw).astype(dtype)
    packed = pack_activation(x, bits, group, e4)
    assert packed.dtype == mx.uint8
    assert packed.nbytes == 2 * 7 * (512 * bits // 8 + 512 // group)
    np.testing.assert_array_equal(
        unpack_activation(packed, bits, group, e4, dtype).astype(mx.float32),
        quantize_activation(x, bits, group, e4).astype(mx.float32),
    )


def test_persistent_cache_is_packed_and_decode_only_unpacks_selected_rows(monkeypatch):
    c = tiny(index_topk=2)
    model = LanguageModel(c)
    cache = model.make_cache()
    mx.eval(model(mx.array([[3] * 32]), cache=cache))
    for state in cache:
        for slot, width in [(1, 33), (2, 18), (3, 17)]:
            assert state[slot].dtype == mx.uint8
            assert state[slot].shape[-1] == width
    from omlx.patches.deepseek_v41 import quantization

    def no_expansion(*args, **kwargs):
        raise AssertionError("Production attention must decode inside Metal")

    monkeypatch.setattr(quantization, "unpack_activation", no_expansion)
    actual = model(mx.array([[4]]), cache=cache)
    mx.eval(actual)
    assert any(state[2].shape[1] > c.index_topk for state in cache)
    np.testing.assert_allclose(
        actual, model(mx.array([[3] * 32 + [4]]))[:, -1:], atol=1e-5
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize(
    "bits,group,e4m3", [(8, 32, False), (4, 32, False), (4, 16, True)]
)
def test_compiled_activation_is_exact(dtype, bits, group, e4m3):
    from omlx.patches.deepseek_v41.quantization import (
        _quantize_activation,
        quantize_activation,
    )

    mx.random.seed(49)
    values = mx.random.normal((2, 19, 128)) * mx.power(
        2.0, mx.arange(-9, 10)[None, :, None]
    )
    for x in [values, mx.zeros_like(values), values * 1e-20, values * 1e-38]:
        x = x.astype(dtype)
        expected = _quantize_activation(x, bits, group, e4m3)
        actual = quantize_activation(x, bits, group, e4m3)
        np.testing.assert_array_equal(
            actual.astype(mx.float32), expected.astype(mx.float32)
        )


@pytest.mark.parametrize("quantized", [False, True])
def test_sorted_prefill_preserves_weighted_expert_output(quantized):
    from omlx.patches.deepseek_v41.language import MoE
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection

    mx.random.seed(1204)
    layer = MoE(tiny())
    if quantized:
        for name in ("w1", "w2", "w3"):
            projection = getattr(layer.experts, name)
            weight, scales = mx.quantize(
                projection.weight.astype(mx.bfloat16),
                bits=4,
                group_size=32,
                mode="mxfp4",
            )
            setattr(
                layer.experts, name, QuantizedProjection(weight, scales, 4, "mxfp4")
            )
    for length in (1, 6, 32, 65):
        x = mx.random.normal((2, length, 32)).astype(
            mx.bfloat16 if quantized else mx.float32
        )
        indices, weights = layer.gate(x, None)
        routed = layer.experts(x[..., None, None, :], indices, weights).squeeze(-2)
        expected = (
            routed.astype(mx.float32).sum(-2)
            + layer.shared_experts(x).astype(mx.float32)
        ).astype(x.dtype)
        actual = layer(x, None)
        # Sorted GEMM may change FP32 reduction order relative to vector QMM.
        np.testing.assert_allclose(
            actual.astype(mx.float32),
            expected.astype(mx.float32),
            rtol=1e-5,
            atol=2e-7,
        )
        mx.eval(actual)
        repeated = layer(x, None)
        mx.eval(repeated)
        np.testing.assert_array_equal(
            actual.astype(mx.float32), repeated.astype(mx.float32)
        )


def test_load_preserves_bf16_head_without_changing_prefill_logits(tmp_path):
    from omlx.patches.deepseek_v41.convert import convert
    from omlx.patches.deepseek_v41.loading import load

    source, _ = write_checkpoint(tmp_path, vision=False)
    filename = source / "model.safetensors"
    weights = mx.load(str(filename))
    weights["head.weight"] = weights["head.weight"].astype(mx.bfloat16)
    mx.eval(weights)
    mx.save_safetensors(str(filename), weights)
    expected = weights["head.weight"].astype(mx.float32)
    target = tmp_path / "converted"
    convert(source, target)
    for path in (source, target):
        model, _ = load(path)
        try:
            head = model.language_model.head
            assert head.weight.dtype == mx.bfloat16
            np.testing.assert_array_equal(head.weight.astype(mx.float32), expected)
            ids = mx.array([[3, 4, 5]])
            actual = model(ids)
            mx.eval(actual)
            head.weight = head.weight.astype(mx.float32)
            reference = model(ids)
            np.testing.assert_array_equal(actual, reference)
        finally:
            model.close()
