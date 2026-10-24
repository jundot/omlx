"""Self-contained packed JANGH2 Metal inference (opt-in, inference only).

Unlike the reference backend, no selected dense weight matrices or host reads of
routing IDs are constructed. Kernels consume the existing uint32 bitstreams and
float16 row scales. All accumulation and intermediate activations are float32.
Factories (not tensor graphs) are cached per thread; no Metal stream is captured.

GPU compilation/performance must be qualified on Apple Silicon. The reference
backend remains independent and unchanged. No custom C++ build is required.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import math
import threading
from typing import Any, Mapping

PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')
_LOCAL = threading.local()


@dataclass(frozen=True)
class KernelSpec:
    input_dims: int
    hidden_dims: int
    experts: int
    bits: tuple[int, int, int]
    rotations: tuple[str, str, str]
    levels: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    gate_rows: int = 4
    down_rows: int = 4
    sort_threshold: int = 64

    def __post_init__(self):
        for name in ('input_dims', 'hidden_dims', 'experts'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.input_dims % 32 or self.hidden_dims % 32:
            raise ValueError('Metal JANGH2 requires D and hidden dimensions divisible by 32')
        if len(self.bits) != 3 or any(type(b) is not int or b not in (2, 3, 4) for b in self.bits):
            raise ValueError('Expected three supported bit widths (2, 3, 4)')
        if len(self.rotations) != 3 or any(r not in ('none', 'hadamard32') for r in self.rotations):
            raise ValueError('Unknown JANGH2 rotation')
        if len(self.levels) != 3:
            raise ValueError('Three explicit codebooks are required')
        for b, table in zip(self.bits, self.levels):
            if len(table) != 1 << b or not all(math.isfinite(v) for v in table):
                raise ValueError('Invalid codebook length or non-finite value')
        if self.gate_rows not in (2, 4, 8) or self.down_rows not in (2, 4, 8):
            raise ValueError('Rows per simdgroup must be 2, 4 or 8')
        if type(self.sort_threshold) is not int or self.sort_threshold < 1:
            raise ValueError('sort_threshold must be a positive integer')


@dataclass(frozen=True)
class RoutePlan:
    routes: int
    top_k: int
    output_shape: tuple[int, ...]
    sorted: bool
    gate_grid: tuple[int, int, int]
    gate_threads: tuple[int, int, int]
    down_grid: tuple[int, int, int]
    down_threads: tuple[int, int, int]
    intermediate_elements: int


def route_plan(x_shape, indices_shape, spec: KernelSpec) -> RoutePlan:
    x_shape, indices_shape = tuple(x_shape), tuple(indices_shape)
    if (len(x_shape) < 2 or len(x_shape) != len(indices_shape)
            or x_shape[:-1] != indices_shape[:-1]
            or x_shape[-1] != spec.input_dims or indices_shape[-1] < 1
            or any(d < 0 for d in x_shape + indices_shape)):
        raise ValueError('Token dimensions / expert routes do not match the kernel contract')
    routes = math.prod(indices_shape)
    # Output Hadamard needs one full block of 32 hidden rows in shared memory.
    rotated = spec.rotations[2] == 'hadamard32'
    groups = 32 // spec.gate_rows if rotated else 2
    tile = groups * spec.gate_rows
    gate_threads = (groups * 32, 1, 1)
    down_threads = (64, 1, 1)
    return RoutePlan(routes, indices_shape[-1], (*indices_shape, spec.input_dims),
                     routes >= spec.sort_threshold,
                     (math.ceil(spec.hidden_dims / tile) * gate_threads[0], routes, 1),
                     gate_threads,
                     (math.ceil(spec.input_dims / (2 * spec.down_rows)) * 64, routes, 1),
                     down_threads, routes * spec.hidden_dims)


@dataclass(frozen=True)
class KernelDefinition:
    name: str
    inputs: tuple[str, ...]
    source: str
    header: str


def _float_literal(value: float) -> str:
    # Nine significant digits round-trip float32. Tables come from validated
    # checkpoint metadata; never replace them with a presumed affine codebook.
    text = format(float(value), '.9g')
    if '.' not in text and 'e' not in text.lower():
        text += '.0'
    return text + 'f'


def _table_function(name: str, levels: tuple[float, ...]) -> str:
    def select(values, bit):
        if len(values) == 1:
            return _float_literal(values[0])
        mid = len(values) // 2
        return f'((q & {1 << bit}u) ? {select(values[mid:], bit - 1)} : {select(values[:mid], bit - 1)})'
    body = select(levels, int(math.log2(len(levels))) - 1)
    return f'inline float {name}(uint q) {{ return {body}; }}\n'


_READ_HEADER = r'''
template <int B>
inline uint jangh_read(const device uint* row, uint column) {
    const uint position = column * B;
    const uint word = position >> 5;
    const uint shift = position & 31u;
    uint code = row[word] >> shift;
    // Widths are multiples of 32: a final-element spill cannot overrun a row.
    if (shift + B > 32u) code |= row[word + 1] << (32u - shift);
    return code & ((1u << B) - 1u);
}
'''


def _definition(prefix, inputs, source, header):
    digest = hashlib.sha256((header + source).encode()).hexdigest()[:12]
    return KernelDefinition(f'jangh2_{prefix}_{digest}', tuple(inputs), source, header)


@functools.lru_cache(maxsize=256)
def build_gate_up(spec: KernelSpec, top_k: int, sorted_routes: bool) -> KernelDefinition:
    if type(top_k) is not int or top_k < 1:
        raise ValueError('top_k must be positive')
    d, h, e = spec.input_dims, spec.hidden_dims, spec.experts
    bg, bu, _ = spec.bits
    rps = spec.gate_rows
    rotate_hidden = spec.rotations[2] == 'hadamard32'
    groups = 32 // rps if rotate_hidden else 2
    tile = groups * rps
    row_expr = 'token_rows[route]' if sorted_routes else f'route / {top_k}u'
    shared_decl = 'threadgroup float hidden_block[32];' if rotate_hidden else ''
    write = (f'if (lane == 0u) hidden_block[sg * {rps}u + r] = value;'
             if rotate_hidden else f'if (lane == 0u && row < {h}u) out[route * {h}u + row] = value;')
    rotate_epilogue = ''
    if rotate_hidden:
        rotate_epilogue = f'''
    // All threads participate; no early return may bypass this barrier.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {{
        float value = hidden_block[lane];
        for (ushort step = 1; step < 32; step <<= 1) {{
            float other = simd_shuffle_xor(value, step);
            value = (lane & step) ? other - value : value + other;
        }}
        uint row = tile_base + lane;
        if (row < {h}u) out[route * {h}u + row] = value * 0.1767766952966369f;
    }}
'''
    source = f'''
    const uint lane = thread_position_in_threadgroup.x & 31u;
    const uint sg = thread_position_in_threadgroup.x >> 5;
    const uint route = threadgroup_position_in_grid.y;
    const uint token = {row_expr};
    const uint expert = ids[route];
    const bool invalid = expert >= {e}u;
    const uint safe_expert = invalid ? 0u : expert;
    const uint tile_base = threadgroup_position_in_grid.x * {tile}u;
    const float NAN_VALUE = as_type<float>(0x7fc00000u);
    {shared_decl}
    float accg[{rps}] = {{0.0f}};
    float accu[{rps}] = {{0.0f}};
    // Match the FP32 reference GEMV: four consecutive columns per lane,
    // with row scales rounded into each weight before the multiply-add.
    for (uint block = 0; block < {d}u; block += 128u) {{
        const uint col = block + lane * 4u;
        float vxg[4], vxu[4];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4u; ++j) {{
            vxg[j] = col + j < {d}u ? xg[token * {d}u + col + j] : 0.0f;
            vxu[j] = col + j < {d}u ? xu[token * {d}u + col + j] : 0.0f;
        }}
        #pragma clang loop unroll(full)
        for (uint r = 0; r < {rps}u; ++r) {{
            const uint row = tile_base + sg * {rps}u + r;
            if (row < {h}u && col < {d}u) {{
                const device uint* pg = wg + (ulong(safe_expert) * {h}u + row) * {d * bg // 32}u;
                const device uint* pu = wu + (ulong(safe_expert) * {h}u + row) * {d * bu // 32}u;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4u; ++j) {{
                    if (col + j < {d}u) {{
                        float wg_value = jangh_level_g(jangh_read<{bg}>(pg, col+j)) * float(sg_weights[ulong(safe_expert) * {h}u + row]);
                        float wu_value = jangh_level_u(jangh_read<{bu}>(pu, col+j)) * float(su_weights[ulong(safe_expert) * {h}u + row]);
                        accg[r] = fma(wg_value, vxg[j], accg[r]);
                        accu[r] = fma(wu_value, vxu[j], accu[r]);
                    }}
                }}
            }}
        }}
    }}
    #pragma clang loop unroll(full)
    for (uint r = 0; r < {rps}u; ++r) {{
        const uint row = tile_base + sg * {rps}u + r;
        // SIMD reduction runs in every lane, including padded output rows.
        float g = accg[r];
        float u = accu[r];
        // A simd_sum may choose a different addition tree. Tiny changes can
        // change later MoE routes, so retain the reference's reduction order.
        for (ushort step = 16; step > 0; step >>= 1) {{
            g += simd_shuffle_down(g, step);
            u += simd_shuffle_down(u, step);
        }}
        float value = 0.0f;
        if (row < {h}u) {{
            float ex = metal::precise::exp(-abs(g));
            float sigmoid = g >= 0.0f ? 1.0f / (1.0f + ex) : ex / (1.0f + ex);
            value = invalid ? NAN_VALUE : (g * sigmoid) * u;
        }}
        {write}
    }}
    {rotate_epilogue}
'''
    header = _READ_HEADER + _table_function('jangh_level_g', spec.levels[0]) + _table_function('jangh_level_u', spec.levels[1])
    return _definition('gate_up', ('xg', 'xu', 'wg', 'sg_weights', 'wu', 'su_weights', 'ids', 'token_rows'), source, header)


@functools.lru_cache(maxsize=128)
def build_down(spec: KernelSpec) -> KernelDefinition:
    n, k, e, bits = spec.input_dims, spec.hidden_dims, spec.experts, spec.bits[2]
    rps = spec.down_rows
    source = f'''
    const uint lane = thread_position_in_threadgroup.x & 31u;
    const uint sg = thread_position_in_threadgroup.x >> 5;
    const uint route = threadgroup_position_in_grid.y;
    const uint expert = ids[route];
    const bool invalid = expert >= {e}u;
    const uint safe_expert = invalid ? 0u : expert;
    const uint tile_base = threadgroup_position_in_grid.x * {2 * rps}u;
    const float NAN_VALUE = as_type<float>(0x7fc00000u);
    float acc[{rps}] = {{0.0f}};
    for (uint block = 0; block < {k}u; block += 128u) {{
        uint col = block + lane * 4u;
        float vx[4];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4u; ++j)
            vx[j] = col+j < {k}u ? hidden[route*{k}u+col+j] : 0.0f;
        #pragma clang loop unroll(full)
        for (uint r = 0; r < {rps}u; ++r) {{
            uint row = tile_base + sg*{rps}u + r;
            if (row < {n}u && col < {k}u) {{
                const device uint* p = wd + (ulong(safe_expert)*{n}u+row)*{k * bits // 32}u;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4u; ++j)
                    if (col+j < {k}u)
                        acc[r] = fma(jangh_level_d(jangh_read<{bits}>(p,col+j)) * float(sd_weights[ulong(safe_expert)*{n}u+row]), vx[j], acc[r]);
            }}
        }}
    }}
    #pragma clang loop unroll(full)
    for (uint r = 0; r < {rps}u; ++r) {{
        uint row = tile_base + sg*{rps}u + r;
        float value = acc[r];
        for (ushort step = 16; step > 0; step >>= 1)
            value += simd_shuffle_down(value, step);
        if (lane == 0u && row < {n}u) {{
            out[route*{n}u+row] = OutType(invalid ? NAN_VALUE : value);
        }}
    }}
'''
    return _definition('down', ('hidden', 'wd', 'sd_weights', 'ids'), source,
                       _READ_HEADER + _table_function('jangh_level_d', spec.levels[2]))


def _kernel_for(mx, definition: KernelDefinition):
    cache = getattr(_LOCAL, 'factories', None)
    if cache is None:
        cache = _LOCAL.factories = {}
    key = (id(mx), definition.name)
    if key not in cache:
        factory = getattr(getattr(mx, 'fast', None), 'metal_kernel', None)
        if not callable(factory):
            raise RuntimeError('The metal backend needs mlx.core.fast.metal_kernel; use reference-fast to roll back')
        cache[key] = factory(name=definition.name, input_names=list(definition.inputs),
                             output_names=['out'], source=definition.source,
                             header=definition.header, ensure_row_contiguous=True)
    return cache[key]


def forward(x: Any, indices: Any, projections: Mapping[str, Any], spec: KernelSpec, mx):
    """Build a lazy fused GPU graph. Intentionally no eval/item/CPU route copy.

    Router IDs generated by the model are already bounded. For defensive custom
    calls, invalid IDs are guarded in each kernel and produce NaNs rather than
    out-of-range memory reads. check_performance.py tests this contract on Metal.
    """
    plan = route_plan(x.shape, indices.shape, spec)
    if str(indices.dtype).split('.')[-1] not in ('int32', 'uint32', 'int64', 'uint64'):
        raise TypeError('Expert IDs must be integer arrays')
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise TypeError('Metal JANGH2 supports float16, bfloat16 and float32 activations')
    if set(projections) != set(PROJECTIONS):
        raise ValueError('A complete packed gate/up/down triplet is required')
    for name, bits, n, k in zip(PROJECTIONS, spec.bits,
                               (spec.hidden_dims,spec.hidden_dims,spec.input_dims),
                               (spec.input_dims,spec.input_dims,spec.hidden_dims)):
        packed, scales = projections[name]
        if (tuple(packed.shape) != (spec.experts,n,k*bits//32) or packed.dtype != mx.uint32
                or tuple(scales.shape) != (spec.experts,n) or scales.dtype != mx.float16):
            raise ValueError(f'Packed shape or dtype mismatch at {name}')
    if not plan.routes:
        return mx.zeros(plan.output_shape, dtype=x.dtype)
    xf = x.reshape(-1, spec.input_dims).astype(mx.float32)
    flat = indices.reshape(-1)
    if str(indices.dtype).split('.')[-1] in ('int64','uint64'):
        # Avoid truncating a 64-bit invalid index into a valid uint32 expert.
        flat = mx.where((flat >= 0) & (flat < spec.experts),flat,spec.experts)
    flat_ids = flat.astype(mx.uint32)
    if plan.sorted:
        order = mx.argsort(flat_ids)
        ids = flat_ids[order]
        token_rows = (order // plan.top_k).astype(mx.uint32)
        inverse = mx.argsort(order)
    else:
        ids, token_rows, inverse = flat_ids, flat_ids, None
    rotated = None
    if 'hadamard32' in spec.rotations[:2]:
        rotated = mx.hadamard_transform(xf.reshape(-1,32)).reshape(xf.shape)
    xg = rotated if spec.rotations[0] == 'hadamard32' else xf
    xu = rotated if spec.rotations[1] == 'hadamard32' else xf
    pg, sg = projections['gate_proj']; pu, su = projections['up_proj']; pd, sd = projections['down_proj']
    gu = _kernel_for(mx, build_gate_up(spec, plan.top_k, plan.sorted))
    hidden = gu(inputs=[xg,xu,pg,sg,pu,su,ids,token_rows],
                output_shapes=[(plan.routes,spec.hidden_dims)], output_dtypes=[mx.float32],
                grid=plan.gate_grid, threadgroup=plan.gate_threads)[0]
    # Gate/up epilogue already rotates the hidden vector if down requires it.
    down = _kernel_for(mx, build_down(spec))
    result = down(inputs=[hidden,pd,sd,ids], template=[('OutType',x.dtype)],
                  output_shapes=[(plan.routes,spec.input_dims)], output_dtypes=[x.dtype],
                  grid=plan.down_grid, threadgroup=plan.down_threads)[0]
    if plan.sorted:
        result = result[inverse]
    return result.reshape(plan.output_shape)
