# SPDX-License-Identifier: Apache-2.0
#
# Single-barrier 1024-wide Walsh-Hadamard butterfly adapted from MTPLX
# (mtplx/kernels/hadamard_rotate.py, Copyright 2026 Youssof Altoukhi,
# Apache-2.0, https://github.com/youssofal/mtplx).

from __future__ import annotations

import logging
import math
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from types import ModuleType

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from ..oqh.modules import _recall, remember
from ..oqh.runtime import _KEYS, _rebase
from . import (
    q2_verify_qmv,
    qwen35_gdn_prework,
    qwen35_gdn_verify_fused,
    qwen35_verify_qmm,
    qwen35_verify_sdpa_split,
)
from .mlx_vlm_mtp import qwen35_verify_attention, qwen35_verify_linear

_prism_mod: ModuleType | None
_q35_lang: ModuleType | None
_verifier_mod: ModuleType | None
try:
    from mlx_vlm.models.prism_hadamard_qwen35 import (
        prism_hadamard_qwen35 as _loaded_prism_mod,
    )
    from mlx_vlm.models.qwen3_5 import language as _loaded_q35_lang
    from mlx_vlm.models.qwen3_5 import speculative_verifier as _loaded_verifier_mod

    _prism_mod = _loaded_prism_mod
    _q35_lang = _loaded_q35_lang
    _verifier_mod = _loaded_verifier_mod
except ImportError:
    _prism_mod = None
    _q35_lang = None
    _verifier_mod = None

logger = logging.getLogger(__name__)

PRISM_HADAMARD_MODEL_TYPE = "prism_hadamard_qwen35"
_SUPPORTED_BLOCKS = (512, 1024, 2048, 4096)
_SEAM_BLOCK = 1024
_NORM_ROTATE_MAX_ROWS = 16
_EVAL_CHUNK = 64
_FWHT_KERNELS: dict[tuple[int, bool], Callable[..., list[mx.array]]] = {}
_NORM_ROTATE_KERNELS: dict[tuple[bool, bool], Callable[..., list[mx.array]]] = {}
_ZERO_SCALARS: dict[mx.Dtype, mx.array] = {}
_ELIDED_TOKENS: dict[int, weakref.ReferenceType[mx.array]] = {}


def _zero_scalar(dtype: mx.Dtype) -> mx.array:
    scalar = _ZERO_SCALARS.get(dtype)
    if scalar is None:
        scalar = mx.zeros((), dtype=dtype)
        mx.eval(scalar)
        _ZERO_SCALARS[dtype] = scalar
    return scalar


def _mark_elided(token: mx.array) -> None:
    tid = id(token)

    def _drop(_ref: object, key: int = tid) -> None:
        _ELIDED_TOKENS.pop(key, None)

    _ELIDED_TOKENS[tid] = weakref.ref(token, _drop)


def _is_elided(x: mx.array) -> bool:
    ref = _ELIDED_TOKENS.get(id(x))
    return ref is not None and ref() is x


def _fwht_1024_butterfly() -> str:
    return """
        for (int h = 1; h < 8; h <<= 1) {
            for (int p = 0; p < 8; ++p) {
                if ((p & h) == 0) {
                    float a = v[p];
                    float b = v[p + h];
                    v[p] = a + b;
                    v[p + h] = a - b;
                }
            }
        }
        for (ushort m = 1; m < 32; m <<= 1) {
            bool upper = (lane & m) != 0;
            for (int p = 0; p < 8; ++p) {
                float partner = simd_shuffle_xor(v[p], m);
                v[p] = upper ? (partner - v[p]) : (v[p] + partner);
            }
        }
        threadgroup float tg[1024];
        for (int p = 0; p < 8; ++p) {
            tg[int(tid) * 8 + p] = v[p];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    """


def _fwht_1024_epilogue(out_base: str, *, inverse: bool = False) -> str:
    sign_mul = "out_v *= sp[idx];" if inverse else ""
    return f"""
        for (int g = int(tid); g < 256; g += 128) {{
            float w0 = tg[g], w1 = tg[g + 256], w2 = tg[g + 512], w3 = tg[g + 768];
            float a0 = w0 + w1, a1 = w0 - w1, a2 = w2 + w3, a3 = w2 - w3;
            float r[4] = {{a0 + a2, a1 + a3, a0 - a2, a1 - a3}};
            for (int q = 0; q < 4; ++q) {{
                int idx = g + 256 * q;
                float out_v = r[q] * SCALE;
                {sign_mul}
                ({out_base})[idx] = static_cast<T>(out_v);
            }}
        }}
    """


def _fwht_1024_source(inverse: bool) -> str:
    inv_flag = 1 if inverse else 0
    return f"""
        constexpr int BLOCK = 1024;
        constexpr int BPR = K / BLOCK;
        constexpr int INVERSE = {inv_flag};
        constexpr float SCALE = 0.03125f;

        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint blk = threadgroup_position_in_grid.x;
        const device T* xp = x + long(blk) * BLOCK;
        const device float* sp = signs + long(blk % BPR) * BLOCK;
        device T* yp = y + long(blk) * BLOCK;
        int e0 = int(tid) * 8;

        float v[8];
        for (int p = 0; p < 8; ++p) {{
            float val = float(xp[e0 + p]);
            v[p] = INVERSE ? val : val * sp[e0 + p];
        }}
        """ + _fwht_1024_butterfly() + _fwht_1024_epilogue("yp", inverse=inverse)


def _fwht_source(block: int, inverse: bool) -> str:
    if block == 1024:
        return _fwht_1024_source(inverse)
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


# Copyright (c) 2026 David Dalcu. The block-parallel norm_rotate kernel below
# is adapted from ddalcu/mlx-serve's MIT-licensed
# src/kernels/quant/norm_rotate.metal.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
def _norm_rotate_source(has_add: bool, write_normed: bool) -> str:
    b_ptr = "        const device T* br = b + long(row) * D;\n" if has_add else ""
    rb_decl = (
        "        long rb = long(row) * D + cb;\n" if (has_add or write_normed) else ""
    )
    if has_add:
        load_sum = """
            vec<T, 4> vb0 = *((const device vec<T, 4>*)(br + col));
            vec<T, 4> vb1 = *((const device vec<T, 4>*)(br + col + 4));
            T cur[8] = {
                T(va0[0] + vb0[0]), T(va0[1] + vb0[1]), T(va0[2] + vb0[2]), T(va0[3] + vb0[3]),
                T(va1[0] + vb1[0]), T(va1[1] + vb1[1]), T(va1[2] + vb1[2]), T(va1[3] + vb1[3]),
            };
        """
        write_sum = """
        #pragma clang loop unroll(full)
        for (int p = 0; p < 8; ++p) { s_out[rb + p] = own[p]; }
        """
    else:
        load_sum = """
            T cur[8] = {va0[0], va0[1], va0[2], va0[3], va1[0], va1[1], va1[2], va1[3]};
        """
        write_sum = ""
    if write_normed:
        norm_loop = """
        #pragma clang loop unroll(full)
        for (int p = 0; p < 8; ++p) {
            T nv = w[cb + p] * static_cast<T>(float(own[p]) * inv);
            n_out[rb + p] = nv;
            v[p] = float(nv) * float(signs[cb + p]);
        }
        """
    else:
        norm_loop = """
        #pragma clang loop unroll(full)
        for (int p = 0; p < 8; ++p) {
            v[p] = float(w[cb + p] * static_cast<T>(float(own[p]) * inv));
        }
        """
    return (
        f"""
        constexpr int BLK = 1024;
        constexpr int NB = D / BLK;
        constexpr float SCALE = 0.03125f;

        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint blk = threadgroup_position_in_grid.x;
        uint row = threadgroup_position_in_grid.y;

        const device T* ar = a + long(row) * D;
{b_ptr}        T own[8];
        float ss = 0.0f;
        for (int k = 0; k < NB; ++k) {{
            int col = k * BLK + int(tid) * 8;
            vec<T, 4> va0 = *((const device vec<T, 4>*)(ar + col));
            vec<T, 4> va1 = *((const device vec<T, 4>*)(ar + col + 4));
{load_sum}            #pragma clang loop unroll(full)
            for (int p = 0; p < 8; ++p) {{
                float vf = float(cur[p]);
                ss += vf * vf;
                if (k == int(blk)) {{
                    own[p] = cur[p];
                }}
            }}
        }}
        ss = simd_sum(ss);
        threadgroup float red[4];
        if (lane == 0) {{
            red[sg] = ss;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = metal::precise::rsqrt(
            ((red[0] + red[1]) + (red[2] + red[3])) / float(D) + eps[0]
        );

        int cb = int(blk) * BLK + int(tid) * 8;
{rb_decl}{write_sum}        float v[8];
{norm_loop}        """
        + _fwht_1024_butterfly()
        + _fwht_1024_epilogue("rot_out + long(row) * D + int(blk) * BLK")
    )


def _norm_rotate_kernel(
    has_add: bool, write_normed: bool
) -> Callable[..., list[mx.array]]:
    key = (has_add, write_normed)
    kernel = _NORM_ROTATE_KERNELS.get(key)
    if kernel is None:
        inputs = ["a", "b"] if has_add else ["a"]
        inputs += ["w", "signs", "eps"] if write_normed else ["w", "eps"]
        outputs = [
            name
            for flag, name in (
                (has_add, "s_out"),
                (write_normed, "n_out"),
                (True, "rot_out"),
            )
            if flag
        ]
        built = mx.fast.metal_kernel(
            name=f"omlx_norm_rotate_b1024_a{int(has_add)}_w{int(write_normed)}",
            input_names=inputs,
            output_names=outputs,
            source=_norm_rotate_source(has_add, write_normed),
        )
        assert callable(built)
        kernel = built
        _NORM_ROTATE_KERNELS[key] = kernel
    return kernel


def norm_rotate(
    a: mx.array,
    b: mx.array | None,
    weight: mx.array,
    signs: mx.array | None,
    eps: mx.array,
    *,
    write_normed: bool,
) -> tuple[mx.array, mx.array | None, mx.array]:
    """Fuse optional residual add, RMSNorm, and 1024-block FWHT on Metal."""
    if a.ndim < 1 or a.size == 0 or eps.dtype != mx.float32 or eps.size < 1:
        raise ValueError("norm_rotate incompatible shapes or dtypes")
    width = int(a.shape[-1])
    if (
        width < _SEAM_BLOCK
        or width % _SEAM_BLOCK != 0
        or a.dtype not in (mx.float16, mx.bfloat16)
        or weight.shape != (width,)
        or weight.dtype != a.dtype
        or (b is not None and (b.shape != a.shape or b.dtype != a.dtype))
        or (
            write_normed
            and (
                signs is None
                or signs.shape != (width,)
                or signs.dtype not in (mx.float16, mx.bfloat16, mx.float32)
            )
        )
    ):
        raise ValueError("norm_rotate incompatible shapes or dtypes")
    rows = a.size // width
    has_add = b is not None
    args: list[mx.array] = [a, b, weight] if b is not None else [a, weight]
    if write_normed:
        assert signs is not None
        args.append(signs)
    args.append(eps)
    n_outs = 1 + int(has_add) + int(write_normed)
    outs = _norm_rotate_kernel(has_add, write_normed)(
        inputs=args,
        template=[("T", a.dtype), ("D", width)],
        grid=(128 * (width // _SEAM_BLOCK), rows, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[a.shape] * n_outs,
        output_dtypes=[a.dtype] * n_outs,
    )
    s_out = outs[0] if has_add else a
    n_out = outs[int(has_add)] if write_normed else None
    return s_out, n_out, outs[-1]


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
        q2_verify_qmv.metal_ready()
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
        threads = 128 if block == 1024 else (512 if block == 4096 else 256)
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
        stacked = mx.stack(
            [mx.max(mx.abs(value)) for _, value in pending[start : start + _EVAL_CHUNK]]
        )
        invalid = mx.any(~mx.isfinite(stacked) | (stacked > fp16_max))
        mx.eval(invalid)
        if bool(invalid.item()):
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
    _hadamard_group: tuple[int, mx.array, int] | None = None
    _hadamard_affine: nn.QuantizedLinear | None = None
    _neg_scale_pair: tuple[mx.array, mx.array] | None = None

    @property
    def _hadamard_key(self) -> int | None:
        grp = self._hadamard_group
        if grp is None or self.block != grp[2]:
            return None
        if grp[2] != 0 and self["signs"] is not grp[1]:
            return None
        return grp[0]

    def _is_neg_scale(self, scales: mx.array, biases: mx.array) -> bool:
        pair = self._neg_scale_pair
        return pair is not None and pair[0] is scales and pair[1] is biases

    def _affine(self) -> nn.QuantizedLinear:
        ql = self._hadamard_affine
        w = self["weight"]
        s = self["scales"]
        b = self["biases"]
        if (
            ql is None
            or ql["weight"] is not w
            or ql["scales"] is not s
            or ql["biases"] is not b
            or ql.bits != self.bits
            or ql.group_size != self.group_size
        ):
            ql = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
            nn.Module.__init__(ql)
            ql.weight, ql.scales, ql.biases = w, s, b
            ql.group_size, ql.bits, ql.mode = self.group_size, self.bits, "affine"
            q2_verify_qmv.mark_ternary(ql, is_ternary=self._is_neg_scale(s, b))
            object.__setattr__(self, "_hadamard_affine", ql)
        return ql

    def rotated_input(self, x: mx.array) -> mx.array:
        if not self.block:
            return x
        key = self._hadamard_key
        if key is not None:
            rot = _recall(x, key)
            if rot is not None:
                return rot
        if _is_elided(x):
            raise RuntimeError("elided Hadamard seam norm missed _MEMO")
        rot = signed_fwht(x, self["signs"], self.block)
        if key is not None:
            remember(x, rot, key)
        return rot

    def __call__(self, x: mx.array) -> mx.array:
        if x.dtype == mx.float32:
            return self(x.astype(self["scales"].dtype)).astype(mx.float32)
        res: mx.array = self._affine()(self.rotated_input(x))
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


@dataclass(frozen=True)
class _HadamardSeamSpec:
    layer_ref: weakref.ReferenceType[nn.Module]
    owner_attr: str
    norm: nn.RMSNorm
    norm_weight: mx.array
    owner: nn.Module
    reader_attrs: tuple[str, ...]
    readers: tuple[_FastHadamardLinear, ...]
    signs_refs: tuple[mx.array, ...]
    width: int
    key: int
    write_normed: bool
    w_eff: mx.array
    signs_t: mx.array | None
    eps_arr: mx.array

    def __call__(
        self, a: mx.array, b: mx.array | None
    ) -> tuple[mx.array, mx.array] | None:
        layer = self.layer_ref()
        if (
            layer is None
            or getattr(layer, self.owner_attr, None) is not self.owner
            or hasattr(self.owner, "switch_mlp")
            or (
                self.owner_attr == "mlp"
                and getattr(self.owner, "down_proj", None) is None
            )
            or not q2_verify_qmv.metal_ready()
            or a.dtype not in (mx.float16, mx.bfloat16)
            or a.shape[-1] != self.width
            or self.norm.weight is not self.norm_weight
            or a.dtype != self.w_eff.dtype
            or (b is not None and (b.dtype != a.dtype or b.shape != a.shape))
        ):
            return None
        rows = a.size // self.width
        if not (1 <= rows <= _NORM_ROTATE_MAX_ROWS):
            return None
        for attr, reader, signs_ref in zip(
            self.reader_attrs, self.readers, self.signs_refs
        ):
            if (
                getattr(self.owner, attr, None) is not reader
                or reader.block != _SEAM_BLOCK
                or reader._hadamard_key != self.key
                or reader["signs"] is not signs_ref
            ):
                return None
        s_out, n_out, rot_out = norm_rotate(
            a,
            b,
            self.w_eff,
            self.signs_t,
            self.eps_arr,
            write_normed=self.write_normed,
        )
        if self.write_normed:
            assert n_out is not None
            remember(n_out, rot_out, self.key)
            return s_out, n_out
        # mx.broadcast_to wraps the cached scalar in a fresh weakrefable view without allocating rot_out.
        token = mx.broadcast_to(_zero_scalar(rot_out.dtype), rot_out.shape)
        _mark_elided(token)
        remember(token, rot_out, self.key)
        return s_out, token


def _bind_sibling_groups(sites: Sequence[Sequence[_FastHadamardLinear]]) -> int:
    for site in sites:
        for reader in site:
            object.__setattr__(reader, "_hadamard_group", None)
    pair_checks: list[tuple[int, int, int, mx.array]] = []
    for s_idx, site in enumerate(sites):
        for i in range(len(site)):
            ri = site[i]
            if ri.block == 0:
                continue
            si = ri["signs"]
            for j in range(i + 1, len(site)):
                rj = site[j]
                if (
                    rj.block == ri.block
                    and rj["signs"] is not si
                    and rj["signs"].shape == si.shape
                ):
                    pair_checks.append((s_idx, i, j, mx.array_equal(si, rj["signs"])))
    equal_pairs: set[tuple[int, int, int]] = set()
    for start in range(0, len(pair_checks), _EVAL_CHUNK):
        chunk = pair_checks[start : start + _EVAL_CHUNK]
        mx.eval([chk for _, _, _, chk in chunk])
        for s_idx, i, j, chk in chunk:
            if bool(chk.item()):
                equal_pairs.add((s_idx, i, j))
    bound = 0
    for s_idx, site in enumerate(sites):
        if len(site) < 2:
            continue
        groups: list[list[tuple[int, _FastHadamardLinear]]] = []
        for idx, reader in enumerate(site):
            for group in groups:
                rep_idx, rep = group[0]
                if rep.block == reader.block and (
                    rep.block == 0
                    or rep["signs"] is reader["signs"]
                    or (s_idx, rep_idx, idx) in equal_pairs
                ):
                    group.append((idx, reader))
                    break
            else:
                groups.append([(idx, reader)])
        for group in groups:
            if len(group) < 2:
                continue
            key = next(_KEYS)
            for _, reader in group:
                signs_ref = (
                    reader["signs"] if reader.block else _zero_scalar(mx.float32)
                )
                object.__setattr__(
                    reader, "_hadamard_group", (key, signs_ref, reader.block)
                )
            bound += 1
    return bound


def _mark_ternary_linears(language_model: nn.Module) -> None:
    candidates: list[_FastHadamardLinear] = []
    for _, module in language_model.named_modules():
        if isinstance(module, _FastHadamardLinear):
            scales = module["scales"]
            biases = module["biases"]
            if (
                isinstance(scales, mx.array)
                and isinstance(biases, mx.array)
                and scales.shape == biases.shape
                and scales.dtype == biases.dtype
            ):
                candidates.append(module)
            else:
                object.__setattr__(module, "_neg_scale_pair", None)
    for start in range(0, len(candidates), _EVAL_CHUNK):
        chunk = candidates[start : start + _EVAL_CHUNK]
        checks = [mx.array_equal(m["biases"], -m["scales"]) for m in chunk]
        mx.eval(checks)
        for mod, ok in zip(chunk, checks):
            flag = bool(ok.item())
            pair = (mod["scales"], mod["biases"]) if flag else None
            object.__setattr__(mod, "_neg_scale_pair", pair)
            if mod._hadamard_affine is not None:
                q2_verify_qmv.mark_ternary(mod._hadamard_affine, is_ternary=flag)


_SeamEntry = tuple[
    weakref.ReferenceType[nn.Module],
    str,
    nn.RMSNorm,
    nn.Module,
    tuple[str, ...],
    tuple[_FastHadamardLinear, ...],
    bool,
]


def _seam_candidate(
    layer: nn.Module,
    owner_attr: str,
    norm: object,
    owner: object,
    attrs: tuple[str, ...],
    write_normed: bool,
) -> _SeamEntry | None:
    if (
        type(norm) is not nn.RMSNorm
        or not isinstance(owner, nn.Module)
        or norm.weight.dtype not in (mx.float16, mx.bfloat16)
    ):
        return None
    width = int(norm.weight.shape[0])
    if width < _SEAM_BLOCK or width % _SEAM_BLOCK != 0:
        return None
    readers: list[_FastHadamardLinear] = []
    for attr in attrs:
        proj = getattr(owner, attr, None)
        if not isinstance(proj, _FastHadamardLinear) or proj.block != _SEAM_BLOCK:
            return None
        readers.append(proj)
    first = readers[0]
    key = first._hadamard_key
    if (
        key is None
        or first["signs"].shape != (width,)
        or any(r._hadamard_key != key for r in readers[1:])
    ):
        return None
    return (
        weakref.ref(layer),
        owner_attr,
        norm,
        owner,
        attrs,
        tuple(readers),
        write_normed,
    )


def _bind_layer_seams(layers: Sequence[nn.Module]) -> None:
    pending: list[_SeamEntry] = []
    for layer in layers:
        for attr in ("input_layernorm", "post_attention_layernorm"):
            if isinstance(norm_mod := getattr(layer, attr, None), nn.Module):
                object.__setattr__(norm_mod, "_omlx_seam_norm", None)
        is_linear = bool(getattr(layer, "is_linear", False))
        mixer_attr = "linear_attn" if is_linear else "self_attn"
        mixer = getattr(layer, mixer_attr, None)
        mixer_attrs = (
            ("in_proj_qkv", "in_proj_z")
            if is_linear
            else ("q_proj", "k_proj", "v_proj")
        )
        if (
            in_entry := _seam_candidate(
                layer,
                mixer_attr,
                getattr(layer, "input_layernorm", None),
                mixer,
                mixer_attrs,
                is_linear,
            )
        ) is not None:
            pending.append(in_entry)
        mlp = getattr(layer, "mlp", None)
        if (
            not hasattr(mlp, "switch_mlp")
            and getattr(mlp, "down_proj", None) is not None
            and (
                post_entry := _seam_candidate(
                    layer,
                    "mlp",
                    getattr(layer, "post_attention_layernorm", None),
                    mlp,
                    ("gate_proj", "up_proj"),
                    False,
                )
            )
            is not None
        ):
            pending.append(post_entry)

    if not pending:
        return
    unit_checks = [
        mx.all(mx.abs(readers[0]["signs"]) == 1.0)
        for _, _, _, _, _, readers, _ in pending
    ]
    mx.eval(unit_checks)
    built_tensors: list[mx.array] = []
    specs: list[tuple[nn.RMSNorm, _HadamardSeamSpec]] = []
    for (
        layer_ref,
        owner_attr,
        norm,
        owner,
        attrs,
        readers,
        write_normed,
    ), unit_ok in zip(pending, unit_checks):
        if not bool(unit_ok.item()):
            continue
        signs = readers[0]["signs"]
        key = readers[0]._hadamard_key
        assert key is not None
        dtype = norm.weight.dtype
        eps_arr = mx.array([norm.eps], dtype=mx.float32)
        if write_normed:
            w_eff = norm.weight
            signs_cast = signs.astype(dtype)
            signs_t: mx.array | None = signs_cast
            built_tensors.extend((eps_arr, signs_cast))
        else:
            w_eff = (norm.weight.astype(mx.float32) * signs).astype(dtype)
            signs_t = None
            built_tensors.extend((eps_arr, w_eff))
        specs.append(
            (
                norm,
                _HadamardSeamSpec(
                    layer_ref,
                    owner_attr,
                    norm,
                    norm.weight,
                    owner,
                    attrs,
                    readers,
                    tuple(r["signs"] for r in readers),
                    int(norm.weight.shape[0]),
                    key,
                    write_normed,
                    w_eff,
                    signs_t,
                    eps_arr,
                ),
            )
        )
    if built_tensors:
        mx.eval(built_tensors)
    for norm, spec in specs:
        object.__setattr__(norm, "_omlx_seam_norm", spec)


_MIXER_PROJECTIONS = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "q_proj",
    "k_proj",
    "v_proj",
)


def install(model: nn.Module) -> None:
    """Install float16 trunk casts, rotation memoization, and affine views on ``model``."""
    if _prism_mod is None:
        return

    apply_prism_hadamard_patch()
    language_model = getattr(model, "language_model", model)
    assert isinstance(language_model, nn.Module)

    recast = _recast_trunk_params_fp16(language_model)
    for _, module in language_model.named_modules():
        if isinstance(module, _prism_mod.HadamardQuantizedEmbedding):
            _rebase(module, _FastHadamardEmbedding)
        elif isinstance(module, _prism_mod.HadamardQuantizedLinear):
            _rebase(module, _FastHadamardLinear)

    _mark_ternary_linears(language_model)

    inner = getattr(language_model, "model", None)
    layers = getattr(inner, "layers", ())
    sites: list[list[_FastHadamardLinear]] = []
    for layer in layers:
        mixer = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
        if mixer is not None:
            sites.append(
                [
                    proj
                    for name in _MIXER_PROJECTIONS
                    if isinstance(
                        proj := getattr(mixer, name, None), _FastHadamardLinear
                    )
                ]
            )
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            sites.append(
                [
                    proj
                    for name in ("gate_proj", "up_proj")
                    if isinstance(proj := getattr(mlp, name, None), _FastHadamardLinear)
                ]
            )
    groups = _bind_sibling_groups(sites)
    _bind_layer_seams(layers)

    object.__setattr__(model, "_omlx_fp16_recast", True)
    object.__setattr__(language_model, "_omlx_fp16_recast", True)
    object.__setattr__(model, "_omlx_prism_hadamard_installed", True)
    object.__setattr__(language_model, "_omlx_prism_hadamard_installed", True)
    if recast:
        logger.info(
            "prism_hadamard_verify installed (%d trunk tensors -> fp16, %d sibling rotation groups)",
            recast,
            groups,
        )


def _patch_verifier_hadamard() -> None:
    if _verifier_mod is None or _q35_lang is None:
        return
    verifier_cls = _verifier_mod.Qwen3_5BatchInvariantForward
    if getattr(verifier_cls, "_omlx_prism_hadamard", False):
        return

    orig_verify = verifier_cls.verify
    orig_linears = verifier_cls._linears
    orig_feed_forward = verifier_cls._feed_forward
    orig_embed_linear = verifier_cls._embedding_as_linear
    swiglu_fn = _q35_lang.swiglu

    @wraps(orig_verify)
    def verify(self, language_model, inputs, *, cache=None, **kwargs):
        auto_arm = (
            getattr(language_model, "_omlx_prism_hadamard_installed", False)
            and not qwen35_verify_qmm._is_armed()
            and not qwen35_verify_qmm.is_row_exact_armed()
        )
        if not auto_arm:
            return orig_verify(self, language_model, inputs, cache=cache, **kwargs)
        qwen35_verify_qmm.set_verify_qmm_armed(True)
        try:
            return orig_verify(self, language_model, inputs, cache=cache, **kwargs)
        finally:
            qwen35_verify_qmm.set_verify_qmm_armed(False)

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
        hidden = qwen35_verify_qmm.verify_swiglu(gate_ql, up_ql, rot)
        if hidden is not None:
            return self._linear(down, hidden)
        gate_out, up_out = orig_linears(self, (gate_ql, up_ql), rot)
        return self._linear(down, swiglu_fn(gate_out, up_out))

    @wraps(orig_embed_linear)
    def _embedding_as_linear(self, embedding, x):
        if isinstance(embedding, _FastHadamardEmbedding):
            return embedding.as_linear(x)
        return orig_embed_linear(self, embedding, x)

    verifier_cls.verify = verify
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
    own_load_weights = cls.__dict__.get("load_weights")

    @wraps(cls.load_weights)
    def load_weights(self, file_or_weights, strict: bool = True):
        if (
            own_load_weights is not None
            and own_load_weights is not nn.Module.load_weights
        ):
            result = own_load_weights(self, file_or_weights, strict=strict)
        else:
            result = nn.Module.load_weights(self, file_or_weights, strict=strict)
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
    qwen35_verify_sdpa_split.apply_qwen35_verify_sdpa_split_patch()
    qwen35_gdn_prework.apply_qwen35_gdn_prework_patch()
    qwen35_gdn_verify_fused.apply_arrays_cache_replay_patch()
    _patch_verifier_hadamard()
    _patch_prism_model_load_weights()
    return True
