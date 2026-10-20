# SPDX-License-Identifier: Apache-2.0
"""Single-prompt, layer-streamed FP32 conditioning for the native v1 bundle.

The language-model head is preserved by conversion but is never used here.
Streaming bounds resident encoder weights and trades throughput for memory.
"""

import inspect
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from omlx.image_bundle import MAX_METADATA, PIPELINE, SIDECAR, read_json
from omlx.image_inspect import RECIPE, contained, tensor_header

PREFIX = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
TEMPLATE = PREFIX + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
MAX_TOKENS = 512


def native_bundle(bundle):
    root = Path(bundle).resolve(strict=True)
    manifest = read_json(contained(root, SIDECAR))
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("pipeline") != PIPELINE
        or manifest.get("layout") != "native-bf16-f32-v1"
        or manifest.get("recipe") != RECIPE
    ):
        raise ValueError("unsupported native bundle contract")
    components = manifest.get("components", {})
    if (
        not isinstance(components, dict)
        or any(
            not isinstance(components.get(name), dict)
            or components[name].get("path") != name
            for name in ("text_encoder", "transformer", "vae")
        )
        or manifest.get("processor") != "processor"
    ):
        raise ValueError("runtime v1 requires canonical component paths")
    return root


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise InterruptedError("image generation cancelled")


def prompt_tokens(bundle: Path, prompt: str):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if len(prompt.encode("utf-8")) > 32768:
        raise ValueError("prompt exceeds 32768 bytes")
    tokenizer = Tokenizer.from_file(str(contained(bundle, "processor/tokenizer.json")))
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False).ids
    ids = tokenizer.encode(TEMPLATE.format(prompt), add_special_tokens=False).ids
    if len(prefix) != 14 or ids[: len(prefix)] != prefix:
        raise ValueError("tokenizer does not implement the conditioning recipe")
    if len(ids) > MAX_TOKENS:
        raise ValueError(
            f"prompt exceeds {MAX_TOKENS} encoded tokens; truncation is disabled"
        )
    if not all(0 <= value < 151936 for value in ids):
        raise ValueError("tokenizer ids exceed the encoder vocabulary")
    return ids, len(prefix)


class EncoderWeights:
    def __init__(self, bundle):
        self.root = Path(bundle).resolve(strict=True)
        self.directory = contained(self.root, "text_encoder")
        self.index = read_json(
            contained(self.root, "text_encoder/model.safetensors.index.json")
        )["weight_map"]
        if not isinstance(self.index, dict):
            raise ValueError("invalid encoder index")
        self.headers = {}
        for shard in set(self.index.values()):
            path = contained(self.root, str(Path("text_encoder") / shard))
            if not path.is_relative_to(self.directory):
                raise ValueError("encoder shard escapes component")
            values, _, _, _ = tensor_header(path, False)
            # Offset metadata remains local; no tensor payload is read here.
            with path.open("rb") as stream:
                length = int.from_bytes(stream.read(8), "little")
                if not 2 <= length <= MAX_METADATA:
                    raise ValueError("encoder header changed during inspection")
                header = json.loads(stream.read(length))
            self.headers[shard] = path, length + 8, header
            for key in values:
                if self.index.get(key) != shard:
                    raise ValueError(f"encoder index mismatch: {key}")
        expected = encoder_shapes()
        if set(self.index) != set(expected):
            raise ValueError("encoder tensor names do not match native layout v1")
        for name, (shape, dtype) in expected.items():
            entry = self.headers[self.index[name]][2].get(name)
            if entry is None or entry["shape"] != shape or entry["dtype"] != dtype:
                raise ValueError(f"encoder tensor shape/dtype mismatch: {name}")

    def array(self, name):
        path, base, header = self.headers[self.index[name]]
        entry = header[name]
        dtype = "<u2" if entry["dtype"] == "BF16" else "<f4"
        return (
            np.memmap(
                path,
                mode="r",
                dtype=dtype,
                offset=base + entry["data_offsets"][0],
                shape=tuple(entry["shape"]),
            ),
            entry["dtype"],
        )

    def weight(self, name):
        value, dtype = self.array(name)
        result = mx.array(np.array(value, copy=True))
        return (result.view(mx.bfloat16) if dtype == "BF16" else result).astype(
            mx.float32
        )


def encoder_shapes():
    shapes = {
        "embed_tokens.weight": ([151936, 4096], "BF16"),
        "lm_head.weight": ([151936, 4096], "BF16"),
        "norm.weight": ([4096], "F32"),
    }
    layer = {
        "input_layernorm.weight": ([4096], "F32"),
        "post_attention_layernorm.weight": ([4096], "F32"),
        "self_attn.q_norm.weight": ([128], "F32"),
        "self_attn.k_norm.weight": ([128], "F32"),
        "self_attn.q_proj.weight": ([4096, 4096], "BF16"),
        "self_attn.k_proj.weight": ([1024, 4096], "BF16"),
        "self_attn.v_proj.weight": ([1024, 4096], "BF16"),
        "self_attn.o_proj.weight": ([4096, 4096], "BF16"),
        "mlp.gate_proj.weight": ([12288, 4096], "BF16"),
        "mlp.up_proj.weight": ([12288, 4096], "BF16"),
        "mlp.down_proj.weight": ([4096, 12288], "BF16"),
    }
    for i in range(36):
        shapes.update({f"layers.{i}.{k}": v for k, v in layer.items()})
    return shapes


def encode_prompt_steps(bundle, prompt, *, cancel=None, on_layer=None):
    from mlx_vlm.models.qwen3_vl.config import TextConfig
    from mlx_vlm.models.qwen3_vl.language import Qwen3VLDecoderLayer

    root = native_bundle(bundle)
    check_cancel(cancel)
    ids, drop = prompt_tokens(root, prompt)
    config = read_json(contained(root, "text_encoder/config.json"))
    expected = {
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000,
        "hidden_act": "silu",
        "attention_bias": False,
    }
    if any(config.get(k) != value for k, value in expected.items()):
        raise ValueError("unsupported encoder architecture")
    if config.get("rope_scaling") != {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    }:
        raise ValueError("unsupported encoder rotary configuration")
    args = TextConfig(
        **{
            k: v
            for k, v in config.items()
            if k in inspect.signature(TextConfig).parameters
        }
    )
    weights = EncoderWeights(root)
    embedding, dtype = weights.array("embed_tokens.weight")
    h = (
        mx.array(np.array(embedding[np.array(ids)[None]], copy=True))
        .view(mx.bfloat16)
        .astype(mx.float32)
    )
    del embedding
    n = len(ids)
    mask = mx.array(
        np.where(np.arange(n)[:, None] >= np.arange(n)[None, :], 0.0, -np.inf).astype(
            np.float32
        )
    )[None, None]
    positions = mx.broadcast_to(mx.arange(n)[None, None], (3, 1, n))
    for i in range(36):
        check_cancel(cancel)
        layer = Qwen3VLDecoderLayer(args, i)
        prefix = f"layers.{i}."
        layer.load_weights(
            [
                (key[len(prefix) :], weights.weight(key))
                for key in weights.index
                if key.startswith(prefix)
            ],
            strict=True,
        )
        h = layer(h, mask=mask, position_ids=positions)
        mx.eval(h)
        if on_layer is not None:
            on_layer(i, h)
        del layer
        yield ("encoding", i + 1, 36)
    check_cancel(cancel)
    h = mx.fast.rms_norm(h, weights.weight("norm.weight"), 1e-6)[:, drop:]
    mx.eval(h)
    if not bool(mx.all(mx.isfinite(h)).item()):
        raise ValueError("non-finite prompt conditioning")
    return h


def run_steps(steps):
    try:
        while True:
            next(steps)
    except StopIteration as stop:
        return stop.value
    finally:
        steps.close()


def encode_prompt(bundle, prompt, *, cancel=None, on_layer=None):
    return run_steps(
        encode_prompt_steps(bundle, prompt, cancel=cancel, on_layer=on_layer)
    )
