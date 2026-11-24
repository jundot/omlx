# SPDX-License-Identifier: Apache-2.0
"""Short Hy4 verification with cache mutation outside compiled layer graphs."""

import logging
from functools import cache

import mlx.core as mx
from mlx_lm.models.cache import RotatingKVCache

from ...cache._rotating_subclass import PrefillReadyRotatingKVCache
from ..mlx_lm_mtp.cache_rollback import stage_functional_rotating_update
from .layers import hc_expand

logger = logging.getLogger(__name__)


def _mix_weight(pre):
    return pre._fused.get(pre.hc_fn)


def _input_modules(layer):
    attn = layer.self_attn
    return (
        layer.hc_attn_layer.hc_pre,
        layer.input_layernorm,
        attn.linear_gate,
        attn.q_a_proj,
        attn.q_a_layernorm,
        attn.q_b_proj,
        attn.kv_a_proj_with_mqa,
        attn.kv_a_layernorm,
    )


def _output_modules(layer, sliding):
    attn = layer.self_attn
    modules = (
        attn.o_proj,
        layer.hc_mlp_layer.hc_pre,
        layer.post_attention_layernorm,
        layer.mlp,
    )
    if "post_proj_norm" in attn:
        modules += (attn.post_proj_norm,)
    if sliding:
        modules += (attn.embed_q, attn.unembed_out)
    return modules


def _project(layer, streams, weight):
    x, post = layer.hc_attn_layer.hc_pre._mix(streams, weight)
    x = layer.input_layernorm(x)
    return x, post, layer.self_attn._project_inputs(x)


def layer_output(attention, pre, norm, mlp, values, gate, streams, post, weight):
    streams = hc_expand(attention._project_output(values, gate), streams, post)
    x, post = pre._mix(streams, weight)
    return hc_expand(mlp(norm(x)), streams, post)


def _finish(layer, values, gate, streams, post, weight):
    return layer_output(
        layer.self_attn,
        layer.hc_mlp_layer.hc_pre,
        layer.post_attention_layernorm,
        layer.mlp,
        values,
        gate,
        streams,
        post,
        weight,
    )


_APPEND = r"""
    constexpr uint KV = 64, PE = 8, PAST = 511;
    const uint index = thread_position_in_grid.x;
    if (index >= (PAST + ROWS) * (KV + PE)) return;
    const uint row = index / (KV + PE);
    const uint col = index % (KV + PE);
    const uint old_row = (row + start[0]) % OLD_ROWS;
    if (col < KV) {
        auto dst = reinterpret_cast<device uint4*>(out_keys);
        auto old = reinterpret_cast<const device uint4*>(old_keys);
        auto add = reinterpret_cast<const device uint4*>(new_keys);
        dst[row * KV + col] = row < PAST
          ? old[old_row * KV + col] : add[(row - PAST) * KV + col];
    } else {
        auto dst = reinterpret_cast<device uint4*>(out_values);
        auto old = reinterpret_cast<const device uint4*>(old_values);
        auto add = reinterpret_cast<const device uint4*>(new_values);
        dst[row * PE + col - KV] = row < PAST
          ? old[old_row * PE + col - KV] : add[(row - PAST) * PE + col - KV];
    }
"""


@cache
def _append_kernel():
    return mx.fast.metal_kernel(
        name="omlx_hy4_verify_sliding_append",
        input_names=["old_keys", "old_values", "new_keys", "new_values", "start"],
        output_names=["out_keys", "out_values"],
        source=_APPEND,
    )


def _append(old_keys, old_values, keys, values, start):
    rows = keys.shape[2]
    total = 511 + rows
    return _append_kernel()(
        inputs=[old_keys, old_values, keys, values, start],
        template=[("ROWS", rows), ("OLD_ROWS", old_keys.shape[2])],
        grid=(total * 72, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 1, total, 512), (1, 1, total, 64)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )


def _sliding_layout(window):
    if (
        type(window) in (RotatingKVCache, PrefillReadyRotatingKVCache)
        and getattr(window, "_omlx_mtp_undo_attached", False)
        and window.keep == 0
        and window.max_size == 512
        and window.keys is not None
        and window.values is not None
        and window.keys.shape[:2] == (1, 1)
        and window.keys.shape[-1] == 512
        and window.keys.shape[2] >= 512
        and window.offset >= window.keys.shape[2]
        and 0 <= window._idx <= window.keys.shape[2]
        and window.values.shape == (1, 1, window.keys.shape[2], 64)
        and window.keys.dtype == window.values.dtype == mx.bfloat16
    ):
        return window.offset, window._idx, window.keys.shape[2]
    return None


def _build_layer(layer, following, sliding):
    attn = layer.self_attn
    modules = _output_modules(layer, sliding)
    if following is not None:
        modules += _input_modules(following)

    def finish(values, gate, streams, post, weight, next_weight):
        streams = _finish(layer, values, gate, streams, post, weight)
        prepared = (
            () if following is None else _project(following, streams, next_weight)
        )
        return streams, prepared

    def finish_sliding(
        gate, streams, post, weight, next_weight, q, qp, kv, kp, mask, sinks
    ):
        values = attn._verify(q, qp, kv, kp, None, mask, sinks)
        return finish(values, gate, streams, post, weight, next_weight)

    return mx.compile(
        finish_sliding if sliding else finish, inputs=[m.state for m in modules]
    )


def _build_sliding(block, following):
    modules = [m for layer in block for m in _output_modules(layer, True)]
    for layer in block[1:]:
        modules.extend(_input_modules(layer))
    if following is not None:
        modules.extend(_input_modules(following))

    def forward(prepared, streams, mask, windows, start, offset, sinks, tails, heads):
        states = []
        for index, layer in enumerate(block):
            _, post, (gate, _, q, qp, k, kp) = prepared
            attn = layer.self_attn
            qp, kp = attn.rope(qp, offset), attn.rope(kp, offset)
            keys, values = _append(*windows[index], k, kp, start)
            states.append((keys, values, k, kp))
            attended = attn._verify(q, qp, keys, values, None, mask, sinks[index])
            streams = _finish(layer, attended, gate, streams, post, tails[index])
            if index + 1 < len(block):
                prepared = _project(block[index + 1], streams, heads[index])
        prepared = () if following is None else _project(following, streams, heads[-1])
        return streams, prepared, tuple(states)

    return mx.compile(forward, inputs=[m.state for m in modules])


class LayerStack:
    """The text model owns its graphs. Cache arrays are explicit inputs."""

    def __init__(self, layers, hidden):
        self.layers = tuple(layers)
        self.ready = tuple(layer._can_compile_moe(hidden) for layer in layers)
        self.following = tuple(
            layer if ready and "engram_layer" not in layer else None
            for layer, ready in zip(self.layers[1:], self.ready[1:])
        ) + (None,)
        self.graphs = {}

    def matches(self, layers):
        return len(layers) == len(self.layers) and all(
            layer is saved for layer, saved in zip(layers, self.layers)
        )

    def _sliding_end(self, i, windows, mask, positions):
        layer = self.layers[i].self_attn
        if (
            layer.kind != "sliding"
            or layer.index_module is not None
            or layer.rank != 512
            or layer.rope_dim != 64
            or layer.window != 512
            or not isinstance(mask, mx.array)
            or (positions is not None and positions.general)
        ):
            return i + 1
        layout = _sliding_layout(windows[i])
        if layout is None:
            return i + 1
        end = i + 1
        while end < min(i + 3, len(self.layers)):
            successor = self.following[end - 1]
            if successor is None:
                break
            attn = successor.self_attn
            if (
                attn.kind != "sliding"
                or attn.index_module is not None
                or attn.rank != 512
                or attn.rope_dim != 64
                or attn.window != 512
                or _sliding_layout(windows[end]) != layout
            ):
                break
            end += 1
        return end

    def _sliding(self, i, end, prepared, streams, windows, mask, positions):
        block = self.layers[i:end]
        following = self.following[end - 1]
        key = (i, end, True)
        graph = self.graphs.get(key)
        if graph is None:
            graph = self.graphs[key] = _build_sliding(block, following)
            logger.debug("Hy4 short verify: compile sliding layers %d–%d", i, end - 1)
        windows = windows[i:end]
        window = windows[0]
        size = window.keys.shape[2]
        start = mx.array([(window._idx + size - 511) % size], dtype=mx.uint32)
        offset = (
            mx.array([window.offset], dtype=mx.int32)
            if positions is None
            else positions.offset
        )
        tail_weights = tuple(_mix_weight(layer.hc_mlp_layer.hc_pre) for layer in block)
        head_layers = block[1:] if following is None else (*block[1:], following)
        head_weights = tuple(
            _mix_weight(layer.hc_attn_layer.hc_pre) for layer in head_layers
        )
        sinks = tuple(
            layer.self_attn.learnable_sink_param.astype(mx.bfloat16) for layer in block
        )
        streams, prepared, states = graph(
            prepared,
            streams,
            mask,
            tuple((w.keys, w.values) for w in windows),
            start,
            offset,
            sinks,
            tail_weights,
            head_weights,
        )
        for layer, window, (keys, values, added_keys, added_values) in zip(
            block, windows, states
        ):
            stage_functional_rotating_update(
                window, added_keys, added_values, decode_consistent=False
            )
            window.keys, window.values = keys, values
            window.offset += added_keys.shape[2]
            window._idx = keys.shape[2]
            if layer.self_attn._capture_topk:
                layer.self_attn._captured_topk = None
        return streams, prepared

    def _layer(self, i, prepared, streams, window, mask, prev_topk, positions):
        layer = self.layers[i]
        attn = layer.self_attn
        x, post, projected = prepared
        kv, topk, gate, q, qp, k, kp, sinks = attn._prepare_cache(
            x, projected, mask, window, prev_topk, positions
        )
        keys, values = kv.update_and_fetch(k, kp)
        sliding = (
            attn.kind == "sliding"
            and topk is None
            and isinstance(mask, mx.array)
            and 512 <= keys.shape[2] < 512 + x.shape[1]
        )
        following = self.following[i]
        key = (i, i + 1, sliding)
        graph = self.graphs.get(key)
        if graph is None:
            graph = self.graphs[key] = _build_layer(layer, following, sliding)
        weight = _mix_weight(layer.hc_mlp_layer.hc_pre)
        next_weight = (
            weight if following is None else _mix_weight(following.hc_attn_layer.hc_pre)
        )
        args = gate, streams, post, weight, next_weight
        if sliding:
            streams, prepared = graph(*args, q, qp, keys, values, mask, sinks)
        else:
            attended = attn._verify(q, qp, keys, values, topk, mask, sinks)
            streams, prepared = graph(attended, *args)
        return streams, prepared, topk

    def __call__(self, streams, windows, masks, engram_inputs, positions):
        prev_topk = None
        prepared = ()
        i = 0
        while i < len(self.layers):
            layer = self.layers[i]
            kind = layer.self_attn.kind
            mask = masks.get(kind)
            if self.ready[i]:
                if not prepared:
                    if "engram_layer" in layer and engram_inputs is not None:
                        streams = streams + layer.engram_layer(streams, *engram_inputs)
                    prepared = _project(
                        layer, streams, _mix_weight(layer.hc_attn_layer.hc_pre)
                    )
                end = self._sliding_end(i, windows, mask, positions)
                if end > i + 1:
                    streams, prepared = self._sliding(
                        i, end, prepared, streams, windows, mask, positions
                    )
                    i = end
                    continue
                streams, prepared, topk = self._layer(
                    i, prepared, streams, windows[i], mask, prev_topk, positions
                )
            else:
                streams, topk = layer(
                    streams, mask, windows[i], prev_topk, engram_inputs, positions
                )
                prepared = ()
            if kind == "sparse":
                prev_topk = topk
            i += 1
        return streams
