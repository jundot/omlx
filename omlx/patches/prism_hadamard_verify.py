# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from functools import wraps
from types import ModuleType

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from ..oqh.modules import _recall, remember
from ..oqh.runtime import _KEYS, _rebase
from . import qwen35_gdn_verify_fused, qwen35_verify_qmm
from .mlx_vlm_mtp import qwen35_verify_attention, qwen35_verify_linear

_prism_mod: ModuleType | None
_q35_lang: ModuleType | None
_verifier_mod: ModuleType | None
_verify_linear_mod: ModuleType | None
try:
    from mlx_vlm.models.prism_hadamard_qwen35 import (
        prism_hadamard_qwen35 as _loaded_prism_mod,
    )
    from mlx_vlm.models.qwen3_5 import language as _loaded_q35_lang
    from mlx_vlm.models.qwen3_5 import speculative_verifier as _loaded_verifier_mod
    from mlx_vlm.speculative.ops import linear as _loaded_verify_linear_mod

    _prism_mod = _loaded_prism_mod
    _q35_lang = _loaded_q35_lang
    _verifier_mod = _loaded_verifier_mod
    _verify_linear_mod = _loaded_verify_linear_mod
except ImportError:
    _prism_mod = None
    _q35_lang = None
    _verifier_mod = None
    _verify_linear_mod = None

logger = logging.getLogger(__name__)

PRISM_HADAMARD_MODEL_TYPE = "prism_hadamard_qwen35"
_SUPPORTED_BLOCKS = (512, 1024, 2048, 4096)
_EVAL_CHUNK = 64
_FWHT_KERNELS: dict[tuple[int, bool], Callable[..., list[mx.array]]] = {}


def _fwht_source(block: int, inverse: bool) -> str:
    threads = 512 if block == 4096 else 256
    vals = block // threads
    pairs = block // (2 * threads)
    scale = f"{1.0 / math.sqrt(block):.9e}f"
    inv_flag = 1 if inverse else 0
    return f"""
        constexpr int BLOCK = {block};
        constexpr int THREADS = {threads};
        constexpr int VALS = {vals};
        constexpr int PAIRS = {pairs};
        constexpr int BPR = K / BLOCK;
        constexpr int INVERSE = {inv_flag};
        constexpr float SCALE = {scale};

        uint tid = thread_position_in_threadgroup.x;
        uint lane = tid & 31u;
        uint blk = threadgroup_position_in_grid.x;
        const device T* xp = x + long(blk) * BLOCK;
        const device float* sp = signs + long(blk % BPR) * BLOCK;
        device T* yp = y + long(blk) * BLOCK;
        threadgroup float buf[BLOCK];

        int sg_base = int(tid >> 5) * (VALS * 32);
        for (int c = 0; c < VALS; ++c) {{
            int idx = sg_base + c * 32 + int(lane);
            float v = float(xp[idx]);
            if (!INVERSE) {{
                v *= sp[idx];
            }}
            for (ushort s = 1; s < 32; s <<= 1) {{
                float partner = simd_shuffle_xor(v, s);
                v = ((lane & s) == 0) ? (v + partner) : (partner - v);
            }}
            buf[idx] = v;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int step = 32; step < BLOCK; step <<= 1) {{
            if (step * 2 < BLOCK) {{
                for (int p = 0; p < PAIRS; ++p) {{
                    int pair_idx = p * THREADS + int(tid);
                    int lo = ((pair_idx & ~(step - 1)) << 1) | (pair_idx & (step - 1));
                    int hi = lo | step;
                    float a = buf[lo];
                    float b = buf[hi];
                    buf[lo] = a + b;
                    buf[hi] = a - b;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }} else {{
                for (int p = 0; p < PAIRS; ++p) {{
                    int pair_idx = p * THREADS + int(tid);
                    int lo = ((pair_idx & ~(step - 1)) << 1) | (pair_idx & (step - 1));
                    int hi = lo | step;
                    float a = buf[lo];
                    float b = buf[hi];
                    float y0 = (a + b) * SCALE;
                    float y1 = (a - b) * SCALE;
                    if (INVERSE) {{
                        y0 *= sp[lo];
                        y1 *= sp[hi];
                    }}
                    yp[lo] = T(y0);
                    yp[hi] = T(y1);
                }}
            }}
        }}
    """


def signed_fwht(
    x: mx.array,
    signs: mx.array,
    block: int,
    *,
    inverse: bool = False,
) -> mx.array:
    """Compute the normalized signed blockwise Walsh-Hadamard transform of ``x``."""
    if block == 0 or x.size == 0:
        return x
    width = int(x.shape[-1])
    if signs.shape != (width,):
        raise ValueError(
            f"signed_fwht expects signs of shape ({width},), got {signs.shape}"
        )
    if (
        mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and x.dtype in (mx.float16, mx.bfloat16)
        and signs.dtype == mx.float32
        and block in _SUPPORTED_BLOCKS
        and width % block == 0
    ):
        key = (block, inverse)
        kernel = _FWHT_KERNELS.get(key)
        if kernel is None:
            built = mx.fast.metal_kernel(
                name=f"omlx_signed_fwht_b{block}_{'inv' if inverse else 'fwd'}",
                input_names=["x", "signs"],
                output_names=["y"],
                source=_fwht_source(block, inverse),
            )
            assert callable(built)
            kernel = built
            _FWHT_KERNELS[key] = kernel
        threads = 512 if block == 4096 else 256
        blocks = x.size // block
        (out,) = kernel(
            inputs=[x, signs],
            template=[("T", x.dtype), ("K", width)],
            grid=(threads * blocks, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )
        return out
    orig_shape, orig_dtype = x.shape, x.dtype
    xf = x.astype(mx.float32)
    if not inverse:
        xf = xf * signs
    xf = mx.hadamard_transform(
        xf.reshape(-1, block), scale=1.0 / math.sqrt(block)
    ).reshape(orig_shape)
    if inverse:
        xf = xf * signs
    return xf.astype(orig_dtype)


def _recast_trunk_params_fp16(language_model: nn.Module) -> int:
    flat_items = tree_flatten(language_model.parameters(), destination={})
    assert isinstance(flat_items, dict)
    pending = [
        (key, value)
        for key, value in flat_items.items()
        if isinstance(key, str)
        and isinstance(value, mx.array)
        and value.dtype == mx.float32
        and not (key.endswith((".signs", ".A_log")) or key == "A_log")
    ]
    count = len(pending)
    if count == 0:
        return 0
    fp16_max = float(mx.finfo(mx.float16).max)
    for start in range(0, count, _EVAL_CHUNK):
        bounds = [
            mx.max(mx.abs(value)) for _, value in pending[start : start + _EVAL_CHUNK]
        ]
        overflow = mx.any(mx.stack(bounds) > fp16_max)
        mx.eval(overflow)
        if bool(overflow.item()):
            raise ValueError(
                "a prism_hadamard_qwen35 float32 parameter exceeds float16 range"
            )
    while pending:
        chunk = pending[:_EVAL_CHUNK]
        pending = pending[_EVAL_CHUNK:]
        converted = [(key, value.astype(mx.float16)) for key, value in chunk]
        mx.eval([value for _, value in converted])
        language_model.update(tree_unflatten(converted))
    return count


class _FastHadamardLinear(nn.Module):
    block: int
    bits: int
    group_size: int
    mode: str
    _hadamard_key: int | None = None
    _hadamard_affine: nn.QuantizedLinear | None = None

    def _affine(self) -> nn.QuantizedLinear:
        ql = self._hadamard_affine
        w = self["weight"]
        s = self["scales"]
        b = self["biases"]
        if ql is None or ql["weight"] is not w or ql["scales"] is not s:
            ql = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
            nn.Module.__init__(ql)
            ql.weight, ql.scales, ql.biases = w, s, b
            ql.group_size, ql.bits, ql.mode = self.group_size, self.bits, "affine"
            object.__setattr__(self, "_hadamard_affine", ql)
        return ql

    def rotated_input(self, x: mx.array) -> mx.array:
        if not self.block:
            return x
        key = self._hadamard_key
        if key is None:
            return signed_fwht(x, self["signs"], self.block)
        rot = _recall(x, key)
        if rot is None:
            rot = signed_fwht(x, self["signs"], self.block)
            remember(x, rot, key)
        return rot

    def __call__(self, x: mx.array) -> mx.array:
        if x.dtype == mx.float32:
            return self(x.astype(self["scales"].dtype)).astype(mx.float32)
        rot = self.rotated_input(x)
        ql = self._affine()
        if qwen35_verify_qmm.is_row_exact_armed() and _verify_linear_mod is not None:
            out: mx.array = _verify_linear_mod._target_verify_linear(ql, rot)
            return out
        res: mx.array = ql(rot)
        return res


class _FastHadamardEmbedding(_FastHadamardLinear):
    def __call__(self, indices: mx.array) -> mx.array:
        shape = indices.shape
        flat = indices.reshape(-1)
        out = (
            mx.dequantize(
                self["weight"][flat],
                self["scales"][flat],
                self["biases"][flat],
                group_size=self.group_size,
                bits=self.bits,
            )
            .reshape(*shape, -1)
            .astype(mx.float16)
        )
        if self.block:
            out = signed_fwht(out, self["signs"], self.block, inverse=True)
        return out

    def as_linear(self, x: mx.array) -> mx.array:
        return _FastHadamardLinear.__call__(self, x)


def _same_rotation(a: _FastHadamardLinear, b: _FastHadamardLinear) -> bool:
    if a.block != b.block:
        return False
    if a.block == 0:
        return True
    return bool(mx.array_equal(a["signs"], b["signs"]).item())


def _bind_sibling_group(readers: Sequence[_FastHadamardLinear]) -> int:
    for reader in readers:
        object.__setattr__(reader, "_hadamard_key", None)
    groups: list[list[_FastHadamardLinear]] = []
    for reader in readers:
        for group in groups:
            if _same_rotation(group[0], reader):
                group.append(reader)
                break
        else:
            groups.append([reader])
    bound = 0
    for group in groups:
        if len(group) < 2:
            continue
        key = next(_KEYS)
        for reader in group:
            object.__setattr__(reader, "_hadamard_key", key)
        bound += 1
    return bound


_MIXER_PROJECTIONS = (
    "in_proj_qkv",
    "in_proj_z",
    "q_proj",
    "k_proj",
    "v_proj",
)


def install(model: nn.Module) -> tuple[int, int]:
    """Install float16 trunk casts, rotation memoization, and affine views on ``model``."""
    if _prism_mod is None:
        return 0, 0

    apply_prism_hadamard_patch()
    language_model = getattr(model, "language_model", model)
    if not isinstance(language_model, nn.Module):
        return 0, 0

    recast = _recast_trunk_params_fp16(language_model)
    for _, module in language_model.named_modules():
        if isinstance(module, _prism_mod.HadamardQuantizedEmbedding):
            _rebase(module, _FastHadamardEmbedding)
        elif isinstance(module, _prism_mod.HadamardQuantizedLinear):
            _rebase(module, _FastHadamardLinear)

    inner = getattr(language_model, "model", None)
    layers = getattr(inner, "layers", ())
    groups = 0
    for layer in layers:
        mixer = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
        if mixer is not None:
            readers = [
                proj
                for name in _MIXER_PROJECTIONS
                if isinstance(proj := getattr(mixer, name, None), _FastHadamardLinear)
            ]
            groups += _bind_sibling_group(readers)
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            mlp_readers = [
                proj
                for name in ("gate_proj", "up_proj")
                if isinstance(proj := getattr(mlp, name, None), _FastHadamardLinear)
            ]
            groups += _bind_sibling_group(mlp_readers)

    object.__setattr__(model, "_omlx_fp16_recast", True)
    object.__setattr__(language_model, "_omlx_fp16_recast", True)
    object.__setattr__(model, "_omlx_prism_hadamard_installed", True)
    if recast:
        logger.info(
            "prism_hadamard_verify installed (%d trunk tensors -> fp16, %d sibling rotation groups)",
            recast,
            groups,
        )
    return recast, groups


def _patch_verifier_hadamard() -> None:
    if _verifier_mod is None or _q35_lang is None:
        return
    verifier_cls = _verifier_mod.Qwen3_5BatchInvariantForward
    if getattr(verifier_cls, "_omlx_prism_hadamard", False):
        return

    orig_linears = verifier_cls._linears
    orig_feed_forward = verifier_cls._feed_forward
    orig_embed_linear = verifier_cls._embedding_as_linear
    swiglu_fn = _q35_lang.swiglu

    @wraps(orig_linears)
    def _linears(self, linears, x):
        items = tuple(linears)
        if len(items) < 2:
            return orig_linears(self, items, x)
        if x.dtype == mx.float32 and any(
            isinstance(item, _FastHadamardLinear) for item in items
        ):
            first_had = next(
                item for item in items if isinstance(item, _FastHadamardLinear)
            )
            casted = _linears(self, items, x.astype(first_had["scales"].dtype))
            return tuple(out.astype(mx.float32) for out in casted)
        results: list[mx.array] = []
        idx = 0
        while idx < len(items):
            cur = items[idx]
            if isinstance(cur, _FastHadamardLinear) and cur._hadamard_key is not None:
                end = idx + 1
                while (
                    end < len(items)
                    and isinstance(items[end], _FastHadamardLinear)
                    and items[end]._hadamard_key == cur._hadamard_key
                ):
                    end += 1
                if end - idx >= 2:
                    rot = cur.rotated_input(x)
                    affines = tuple(lin._affine() for lin in items[idx:end])
                    results.extend(orig_linears(self, affines, rot))
                    idx = end
                    continue
            next_idx = idx + 1
            while next_idx < len(items) and not (
                isinstance(items[next_idx], _FastHadamardLinear)
                and items[next_idx]._hadamard_key is not None
            ):
                next_idx += 1
            results.extend(orig_linears(self, items[idx:next_idx], x))
            idx = next_idx
        return tuple(results)

    @wraps(orig_feed_forward)
    def _feed_forward(self, feed_forward, x):
        gate = getattr(feed_forward, "gate_proj", None)
        up = getattr(feed_forward, "up_proj", None)
        down = getattr(feed_forward, "down_proj", None)
        if (
            down is None
            or hasattr(feed_forward, "switch_mlp")
            or not isinstance(gate, _FastHadamardLinear)
            or not isinstance(up, _FastHadamardLinear)
            or gate._hadamard_key is None
            or gate._hadamard_key != up._hadamard_key
        ):
            return orig_feed_forward(self, feed_forward, x)
        if x.dtype == mx.float32:
            return _feed_forward(
                self, feed_forward, x.astype(gate["scales"].dtype)
            ).astype(mx.float32)
        rot = gate.rotated_input(x)
        gate_ql = gate._affine()
        up_ql = up._affine()
        if mx.metal.is_available() and qwen35_verify_qmm.sg8_swiglu_eligible(
            gate_ql, up_ql, rot
        ):
            batch, length, width = rot.shape
            hidden = qwen35_verify_qmm.vk_swiglu_sg8(
                rot.reshape(batch * length, width),
                gate_ql,
                up_ql,
                qwen35_verify_qmm.group_sums_for(rot),
            ).reshape(batch, length, -1)
            return self._linear(down, hidden)
        gate_out, up_out = orig_linears(self, (gate_ql, up_ql), rot)
        return self._linear(down, swiglu_fn(gate_out, up_out))

    @wraps(orig_embed_linear)
    def _embedding_as_linear(self, embedding, x):
        if isinstance(embedding, _FastHadamardEmbedding):
            return embedding.as_linear(x)
        return orig_embed_linear(self, embedding, x)

    verifier_cls._linears = _linears
    verifier_cls._feed_forward = _feed_forward
    verifier_cls._embedding_as_linear = _embedding_as_linear
    verifier_cls._omlx_prism_hadamard = True


def _patch_prism_model_load_weights() -> None:
    if _prism_mod is None:
        return
    cls = _prism_mod.Model
    if getattr(cls, "_omlx_prism_load_weights_patched", False):
        return
    orig_load_weights = cls.load_weights

    @wraps(orig_load_weights)
    def load_weights(self, file_or_weights, strict: bool = True):
        result = orig_load_weights(self, file_or_weights, strict=strict)
        install(self)
        return result

    cls.load_weights = load_weights
    cls._omlx_prism_load_weights_patched = True


def apply_prism_hadamard_patch() -> bool:
    """Install verify routing and load hooks for ``prism_hadamard_qwen35``."""
    if _prism_mod is None:
        return False
    qwen35_verify_qmm.apply_verify_qmm_patch()
    qwen35_verify_linear.apply()
    if _q35_lang is not None:
        qwen35_verify_attention.apply(_q35_lang)
    qwen35_gdn_verify_fused.apply_arrays_cache_replay_patch()
    _patch_verifier_hadamard()
    _patch_prism_model_load_weights()
    return True
