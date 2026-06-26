from functools import partial
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx_lm.models.cache import BatchKVCache, dynamic_roll
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import ModelConfig, TextConfig

_MSA_SPARSE_DECODE_DEFAULT_MAX_DENSITY = 0.5


def _is_bool_mask(mask: mx.array) -> bool:
    return mask.dtype == mx.bool_


def _is_integer_mask(mask: mx.array) -> bool:
    return mask.dtype in {
        mx.int8,
        mx.int16,
        mx.int32,
        mx.int64,
        mx.uint8,
        mx.uint16,
        mx.uint32,
        mx.uint64,
    }


def _is_empty_kv_state(state) -> bool:
    return state is None or (
        isinstance(state, (tuple, list))
        and len(state) >= 2
        and state[0] is None
        and state[1] is None
    )


def _switch_gather_sort(x: mx.array, indices: mx.array):
    *_, num_selected = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // num_selected], indices[order], inv_order


def _switch_scatter_unsort(x: mx.array, inv_order: mx.array, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


@mx.compile
def _minimax_moe_select(
    gates: mx.array,
    correction_bias: mx.array,
    k: int,
    routed_scaling_factor: float,
    scoring_func: str,
):
    gates = gates.astype(mx.float32)
    if scoring_func == "sigmoid":
        scores = mx.sigmoid(gates)
    else:
        scores = mx.softmax(gates, axis=-1, precise=True)

    biased_scores = scores + correction_bias
    inds = mx.argpartition(-biased_scores, kth=k - 1, axis=-1)[..., :k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    return inds, weights * routed_scaling_factor


@mx.compile
def _build_sparse_causal_mask_compiled(
    idx_queries: mx.array,
    idx_keys: mx.array,
    q_start: int,
    scale: float,
    block_size: int,
    sparse_topk_blocks: int,
    sparse_init_blocks: int,
    sparse_local_blocks: int,
):
    B, H_idx, L, _ = idx_queries.shape
    total_len = idx_keys.shape[2]
    neg = mx.array(-float("inf"), dtype=mx.float32)

    scores = mx.matmul(
        idx_queries.astype(mx.float32),
        idx_keys.astype(mx.float32).swapaxes(-1, -2),
    )
    scores = scores * scale

    qpos = mx.arange(q_start, q_start + L)
    kpos = mx.arange(total_len)
    causal = kpos[None, :] <= qpos[:, None]
    scores = mx.where(causal[None, None], scores, neg)

    num_blocks = (total_len + block_size - 1) // block_size
    pad = num_blocks * block_size - total_len
    if pad:
        pad_values = mx.full(
            (*scores.shape[:-1], pad), -float("inf"), dtype=scores.dtype
        )
        scores = mx.concatenate([scores, pad_values], axis=-1)

    blocks = mx.arange(num_blocks)
    cur_block = qpos // block_size
    causal_block = blocks[None, :] <= cur_block[:, None]
    valid_blocks = causal_block[None, None]

    scores = scores.reshape(B, H_idx, L, num_blocks, block_size)
    block_scores = mx.max(scores, axis=-1)
    block_scores = mx.max(block_scores, axis=1)
    block_scores = mx.where(block_scores == block_scores, block_scores, neg)

    init_blocks = blocks[None, :] < sparse_init_blocks
    if sparse_local_blocks > 0:
        local_start = mx.maximum(cur_block - sparse_local_blocks + 1, 0)
        local_blocks = (blocks[None, :] >= local_start[:, None]) & (
            blocks[None, :] <= cur_block[:, None]
        )
    else:
        local_blocks = mx.zeros((L, num_blocks), dtype=mx.bool_)

    valid_blocks = mx.broadcast_to(valid_blocks[:, 0], block_scores.shape)
    selected_scores = mx.where(valid_blocks, block_scores, neg)
    forced_init_blocks = (init_blocks & causal_block)[None] & valid_blocks
    forced_local_blocks = (local_blocks & causal_block)[None] & valid_blocks
    selected_scores = mx.where(
        forced_init_blocks,
        mx.array(1e30, dtype=selected_scores.dtype),
        selected_scores,
    )
    selected_scores = mx.where(
        forced_local_blocks,
        mx.array(1e29, dtype=selected_scores.dtype),
        selected_scores,
    )

    topk_idx = mx.argpartition(-selected_scores, kth=sparse_topk_blocks - 1, axis=-1)[
        ..., :sparse_topk_blocks
    ]
    topk_valid = mx.take_along_axis(valid_blocks, topk_idx, axis=-1)

    block_selected = mx.any(topk_idx[..., None] == blocks, axis=-2)
    block_selected = block_selected & valid_blocks

    key_blocks = (kpos // block_size).astype(mx.int32)
    key_blocks = mx.broadcast_to(key_blocks[None, None, :], (B, L, total_len))
    key_selected = mx.take_along_axis(block_selected, key_blocks, axis=-1)
    key_selected = key_selected & causal[None]

    sparse_mask = key_selected[:, None]
    return sparse_mask, topk_idx[:, None], topk_valid[:, None]


@partial(mx.compile, shapeless=True)
def _swiglu_oai(
    x_linear,
    x_glu,
    alpha: float = 1.702,
    limit: float = 7.0,
    beta: float = 1.0,
):
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
    return x_glu * mx.sigmoid(alpha * x_glu) * (x_linear + beta)


class MiniMaxSwiGLUOAI(nn.Module):
    def __init__(
        self,
        alpha: float = 1.702,
        limit: float = 7.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.limit = limit
        self.beta = beta

    def __call__(self, x, gate):
        return _swiglu_oai(x, gate, self.alpha, self.limit, self.beta)


class MiniMaxRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6, gemma: bool = True):
        super().__init__()
        self.weight = mx.zeros((dims,)) if gemma else mx.ones((dims,))
        self.eps = eps
        self.gemma = gemma

    def __call__(self, x):
        weight = self.weight + 1 if self.gemma else self.weight
        return mx.fast.rms_norm(x, weight.astype(mx.float32), self.eps).astype(x.dtype)


class MiniMaxM3KVCache:
    step = KVCache.step

    def __init__(self):
        self.kv_cache = KVCache()
        self.index_keys = None
        self.index_offset = 0

    @property
    def offset(self):
        return self.kv_cache.offset

    @offset.setter
    def offset(self, value):
        self.kv_cache.offset = int(value)
        self.index_offset = int(value)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        return self.kv_cache.update_and_fetch(keys, values)

    def to_batch(self, left_padding):
        batch_cache = MiniMaxM3BatchKVCache(left_padding)
        left_padding = mx.array(left_padding)

        if self.kv_cache.empty() and self.index_keys is None:
            return batch_cache

        if left_padding.size != 1:
            raise ValueError(
                "Cannot convert a warm MiniMax M3 cache to a multi-row batch "
                f"cache with left padding {left_padding.tolist()}."
            )

        pad = int(left_padding.item())
        keys, values = self.kv_cache.state
        if pad:
            keys = mx.pad(keys, [(0, 0), (0, 0), (pad, 0), (0, 0)])
            values = mx.pad(values, [(0, 0), (0, 0), (pad, 0), (0, 0)])
        batch_cache.kv_cache.state = (
            keys,
            values,
            mx.array([self.offset], dtype=mx.int32),
            left_padding.astype(mx.int32),
        )

        if self.index_keys is not None:
            index_keys = self.index_keys[..., : self.index_offset, :]
            if pad:
                index_keys = mx.pad(index_keys, [(0, 0), (0, 0), (pad, 0), (0, 0)])
            batch_cache.index_keys = index_keys
            batch_cache.index_offset = index_keys.shape[2]

        return batch_cache

    def update_index_and_fetch(self, keys: mx.array):
        prev = self.index_offset
        incoming = keys.shape[2]
        if self.index_keys is None or (prev + incoming) > self.index_keys.shape[2]:
            B, n_heads, _, head_dim = keys.shape
            n_steps = (self.step + incoming - 1) // self.step
            new_shape = (B, n_heads, n_steps * self.step, head_dim)
            new_keys = mx.zeros(new_shape, keys.dtype)
            if self.index_keys is not None:
                if prev % self.step != 0:
                    self.index_keys = self.index_keys[..., :prev, :]
                self.index_keys = mx.concatenate([self.index_keys, new_keys], axis=2)
            else:
                self.index_keys = new_keys

        self.index_offset += incoming
        self.index_keys[..., prev : self.index_offset, :] = keys
        return self.index_keys[..., : self.index_offset, :]

    def make_mask(self, *args, **kwargs):
        return self.kv_cache.make_mask(*args, **kwargs)

    def size(self):
        return self.kv_cache.size()

    def empty(self):
        return self.kv_cache.empty()

    def is_trimmable(self):
        return True

    def trim(self, n):
        trimmed = self.kv_cache.trim(n)
        self.index_offset = max(0, self.index_offset - trimmed)
        return trimmed

    @classmethod
    def merge(cls, caches, prefix_lens=None):
        return MiniMaxM3BatchKVCache.merge(caches, prefix_lens=prefix_lens)

    @property
    def state(self):
        kv_state = None if self.kv_cache.empty() else self.kv_cache.state
        index_state = (
            None
            if self.index_keys is None
            else self.index_keys[..., : self.index_offset, :]
        )
        return kv_state, index_state

    @state.setter
    def state(self, value):
        kv_state, index_state = value
        self.kv_cache = KVCache()
        if not _is_empty_kv_state(kv_state):
            self.kv_cache.state = kv_state
        self.index_keys = index_state
        self.index_offset = 0 if index_state is None else index_state.shape[2]

    @property
    def meta_state(self):
        return str(self.index_offset)

    @meta_state.setter
    def meta_state(self, value):
        self.index_offset = int(value) if value else 0

    @property
    def nbytes(self):
        index_nbytes = 0 if self.index_keys is None else self.index_keys.nbytes
        return self.kv_cache.nbytes + index_nbytes


class MiniMaxM3BatchKVCache:
    step = BatchKVCache.step

    def __init__(self, left_padding):
        self.kv_cache = BatchKVCache(left_padding)
        self.index_keys = None
        self.index_offset = 0

    @property
    def offset(self):
        return self.kv_cache.offset

    @property
    def left_padding(self):
        return self.kv_cache.left_padding

    @property
    def _idx(self):
        return self.kv_cache._idx

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        return self.kv_cache.update_and_fetch(keys, values)

    def update_index_and_fetch(self, keys: mx.array):
        prev = self.index_offset
        incoming = keys.shape[2]
        if self.index_keys is None or (prev + incoming) > self.index_keys.shape[2]:
            B, n_heads, _, head_dim = keys.shape
            n_steps = (self.step + incoming - 1) // self.step
            new_shape = (B, n_heads, n_steps * self.step, head_dim)
            new_keys = mx.zeros(new_shape, keys.dtype)
            if self.index_keys is not None:
                if prev % self.step != 0:
                    self.index_keys = self.index_keys[..., :prev, :]
                self.index_keys = mx.concatenate([self.index_keys, new_keys], axis=2)
            else:
                self.index_keys = new_keys

        self.index_offset += incoming
        self.index_keys[..., prev : self.index_offset, :] = keys
        return self.index_keys[..., : self.index_offset, :]

    def prepare(self, **kwargs):
        self.kv_cache.prepare(**kwargs)

    def finalize(self):
        right_padding = getattr(self.kv_cache, "_right_padding", None)
        self.kv_cache.finalize()
        if right_padding is not None and self.index_keys is not None:
            self.index_keys = dynamic_roll(
                self.index_keys, right_padding[:, None], axis=2
            )

    def make_mask(self, *args, **kwargs):
        return self.kv_cache.make_mask(*args, **kwargs)

    def filter(self, batch_indices):
        min_left_pad = self.left_padding[batch_indices].min().item()
        self.kv_cache.filter(batch_indices)
        if self.index_keys is not None:
            self.index_keys = self.index_keys[batch_indices]
            if min_left_pad > 0:
                self.index_keys = self.index_keys[..., min_left_pad:, :]
                self.index_offset -= min_left_pad

    def extend(self, other):
        if not isinstance(other, MiniMaxM3BatchKVCache):
            raise TypeError(f"Cannot extend MiniMaxM3BatchKVCache with {type(other)}")
        self.kv_cache.extend(other.kv_cache)
        self._extend_index_keys(other)

    def _extend_index_keys(self, other):
        if self.index_keys is None and other.index_keys is None:
            return

        max_idx = max(self.index_offset, other.index_offset)
        self_len = 0 if self.index_keys is None else self.index_keys.shape[2]
        other_len = 0 if other.index_keys is None else other.index_keys.shape[2]
        max_size = max(self_len, other_len)
        ref = self.index_keys if self.index_keys is not None else other.index_keys
        _, heads, _, dims = ref.shape

        def pad(cache):
            keys = cache.index_keys
            if keys is None:
                batch = cache.offset.shape[0]
                keys = mx.zeros((batch, heads, 0, dims), dtype=ref.dtype)
            left = max_idx - cache.index_offset
            right = max_size - keys.shape[2] - left
            if right < 0:
                keys = keys[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                keys = mx.pad(keys, [(0, 0), (0, 0), (left, right), (0, 0)])
            return keys

        self.index_keys = mx.concatenate([pad(self), pad(other)], axis=0)
        self.index_offset = max_idx

    def extract(self, idx):
        cache = MiniMaxM3KVCache()
        cache.kv_cache = self.kv_cache.extract(idx)
        padding = self.left_padding[idx].item()
        if self.index_keys is not None:
            cache.index_keys = mx.contiguous(
                self.index_keys[idx : idx + 1, :, padding : self.index_offset]
            )
            cache.index_offset = cache.index_keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches, prefix_lens=None):
        caches = list(caches)
        out = cls([0] * len(caches))
        if not caches:
            return out

        out.kv_cache = BatchKVCache.merge([cache.kv_cache for cache in caches])
        index_states = [
            (
                None
                if cache.index_keys is None
                or int(getattr(cache, "index_offset", 0) or 0) <= 0
                else cache.index_keys[..., : int(cache.index_offset), :]
            )
            for cache in caches
        ]
        sample = next((state for state in index_states if state is not None), None)
        if sample is None:
            return out

        max_idx = max(
            [int(getattr(out.kv_cache, "_idx", 0) or 0)]
            + [int(v) for v in (prefix_lens or [])]
            + [int(getattr(cache, "index_offset", 0) or 0) for cache in caches]
        )
        _, heads, _, dims = sample.shape
        rows = []
        for cache, state in zip(caches, index_states):
            index_len = int(getattr(cache, "index_offset", 0) or 0)
            if state is None:
                state = mx.zeros((1, heads, 0, dims), dtype=sample.dtype)
                index_len = 0
            left = max_idx - index_len
            right = max_idx - state.shape[2] - left
            if right < 0:
                state = state[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                state = mx.pad(state, [(0, 0), (0, 0), (left, right), (0, 0)])
            rows.append(state)

        out.index_keys = mx.concatenate(rows, axis=0)
        out.index_offset = max_idx
        return out

    def size(self):
        return self.kv_cache.size()

    def empty(self):
        return self.kv_cache.empty()

    def is_trimmable(self):
        return self.kv_cache.is_trimmable()

    def trim(self, n):
        trimmed = self.kv_cache.trim(n)
        self.index_offset = max(0, self.index_offset - trimmed)
        return trimmed

    @property
    def state(self):
        kv_state = (
            (None, None, self.kv_cache.offset, self.kv_cache.left_padding)
            if self.kv_cache.empty()
            else self.kv_cache.state
        )
        index_state = (
            None
            if self.index_keys is None
            else self.index_keys[..., : self.index_offset, :]
        )
        return kv_state, index_state

    @state.setter
    def state(self, value):
        kv_state, index_state = value
        if _is_empty_kv_state(kv_state):
            left_padding = [0] if kv_state is None or len(kv_state) < 4 else kv_state[3]
            self.kv_cache = BatchKVCache(left_padding)
            if kv_state is not None and len(kv_state) >= 3 and kv_state[2] is not None:
                self.kv_cache.offset = kv_state[2]
        else:
            self.kv_cache.state = kv_state
        self.index_keys = index_state
        self.index_offset = 0 if index_state is None else index_state.shape[2]

    @property
    def meta_state(self):
        return str(self.index_offset)

    @meta_state.setter
    def meta_state(self, value):
        self.index_offset = int(value) if value else 0

    @property
    def nbytes(self):
        index_nbytes = 0 if self.index_keys is None else self.index_keys.nbytes
        return self.kv_cache.nbytes + index_nbytes


class MiniMaxMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        alpha: float = 1.702,
        limit: float = 7.0,
        beta: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = MiniMaxSwiGLUOAI(alpha, limit, beta)

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x), self.gate_proj(x)))


class MiniMaxPackedSwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation: MiniMaxSwiGLUOAI,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_up_proj = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=bias
        )
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _switch_gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)
        gate, up = mx.split(gate_up, 2, axis=-1)
        x = self.down_proj(
            self.activation(up, gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _switch_scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class MiniMaxAttention(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.use_qk_norm = args.use_qk_norm

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        if self.use_qk_norm:
            self.q_norm = MiniMaxRMSNorm(
                self.head_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )
            self.k_norm = MiniMaxRMSNorm(
                self.head_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )

        self.has_sparse_index = args.has_sparse_index(layer_idx)
        if self.has_sparse_index:
            sparse_config = args.sparse_attention_config
            self.sparse_block_size = sparse_config.get("sparse_block_size", 128)
            self.sparse_topk_blocks = sparse_config.get("sparse_topk_blocks", 16)
            self.sparse_init_blocks = sparse_config.get("sparse_init_block", 0)
            self.sparse_local_blocks = sparse_config.get("sparse_local_block", 1)
            self.sparse_score_type = sparse_config.get("sparse_score_type", "max")
            self.index_dim = sparse_config.get("sparse_index_dim", self.head_dim)
            self.index_heads = sparse_config.get("sparse_num_index_heads", 4)
            self.index_q_proj = nn.Linear(
                args.hidden_size, self.index_heads * self.index_dim, bias=False
            )
            self.index_k_proj = nn.Linear(args.hidden_size, self.index_dim, bias=False)
            self.index_q_norm = MiniMaxRMSNorm(
                self.index_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )
            self.index_k_norm = MiniMaxRMSNorm(
                self.index_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )

        self.rope = initialize_rope(
            args.rotary_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    @staticmethod
    def _sparse_query_positions(
        B: int,
        L: int,
        q_start: int,
        q_positions: Optional[mx.array] = None,
    ):
        if q_positions is None:
            positions = mx.arange(q_start, q_start + L)
            return mx.broadcast_to(positions[None, :], (B, L))

        positions = q_positions
        if positions.ndim == 0:
            positions = positions.reshape(1, 1)
        elif positions.ndim == 1:
            if positions.shape[0] == B and L == 1:
                positions = positions[:, None]
            else:
                positions = positions[None, :]
        elif positions.ndim == 3:
            positions = positions[0]

        if positions.shape[-1] != L:
            positions = positions[..., -L:]
        if positions.shape[0] == 1 and B != 1:
            positions = mx.broadcast_to(positions, (B, L))
        elif positions.shape[0] != B:
            positions = positions.reshape(B, L)
        return positions

    def _build_sparse_mask(
        self,
        idx_queries: mx.array,
        idx_keys: mx.array,
        q_start: int,
        mask: Optional[mx.array] = None,
        return_block_indices: bool = False,
        build_token_mask: bool = True,
        q_positions: Optional[mx.array] = None,
    ):
        B, H_idx, L, _ = idx_queries.shape
        total_len = idx_keys.shape[2]
        if (
            build_token_mask
            and q_positions is None
            and (mask is None or isinstance(mask, str))
            and self.sparse_score_type == "max"
            and (total_len + self.sparse_block_size - 1) // self.sparse_block_size
            >= self.sparse_topk_blocks
        ):
            if self.num_attention_heads % H_idx != 0:
                raise ValueError(
                    "MiniMax M3 sparse index heads must divide attention heads: "
                    f"{H_idx} index heads, {self.num_attention_heads} attention heads."
                )
            sparse_mask, topk_idx, topk_valid = _build_sparse_causal_mask_compiled(
                idx_queries,
                idx_keys,
                q_start,
                self.scale,
                self.sparse_block_size,
                self.sparse_topk_blocks,
                self.sparse_init_blocks,
                self.sparse_local_blocks,
            )
            if return_block_indices:
                return sparse_mask, topk_idx, topk_valid
            return sparse_mask

        scale = self.scale
        neg = mx.array(-float("inf"), dtype=mx.float32)

        scores = mx.matmul(
            idx_queries.astype(mx.float32),
            idx_keys.astype(mx.float32).swapaxes(-1, -2),
        )
        scores = scores * scale

        qpos = self._sparse_query_positions(B, L, q_start, q_positions)
        kpos = mx.arange(total_len)
        causal = kpos[None, None, :] <= qpos[:, :, None]
        scores = mx.where(causal[:, None], scores, neg)
        valid = self._selection_valid_mask(mask, B, H_idx, L, total_len)
        if valid is not None:
            scores = mx.where(valid, scores, neg)

        block_size = self.sparse_block_size
        num_blocks = (total_len + block_size - 1) // block_size
        pad = num_blocks * block_size - total_len
        if pad:
            pad_values = mx.full(
                (*scores.shape[:-1], pad), -float("inf"), dtype=scores.dtype
            )
            scores = mx.concatenate([scores, pad_values], axis=-1)
            if valid is not None:
                valid_pad = mx.zeros((*valid.shape[:-1], pad), dtype=mx.bool_)
                valid = mx.concatenate([valid, valid_pad], axis=-1)

        blocks = mx.arange(num_blocks)
        cur_block = qpos // block_size
        causal_block = blocks[None, None, :] <= cur_block[:, :, None]

        scores = scores.reshape(B, H_idx, L, num_blocks, block_size)
        if valid is not None:
            valid_blocks = mx.any(
                valid.reshape(B, valid.shape[1], L, num_blocks, block_size),
                axis=-1,
            )
        else:
            valid_blocks = causal_block[:, None]

        if self.sparse_score_type == "lse":
            block_scores = mx.logsumexp(scores, axis=-1)
        else:
            block_scores = mx.max(scores, axis=-1)
        block_scores = mx.max(block_scores, axis=1)
        block_scores = mx.where(block_scores == block_scores, block_scores, neg)
        if valid_blocks.shape[1] == 1:
            valid_blocks = valid_blocks[:, 0]
        else:
            valid_blocks = mx.any(valid_blocks, axis=1)
        valid_blocks = valid_blocks & causal_block

        init_blocks = blocks[None, None, :] < self.sparse_init_blocks
        if self.sparse_local_blocks > 0:
            local_start = mx.maximum(cur_block - self.sparse_local_blocks + 1, 0)
            local_blocks = (blocks[None, None, :] >= local_start[:, :, None]) & (
                blocks[None, None, :] <= cur_block[:, :, None]
            )
        else:
            local_blocks = mx.zeros((B, L, num_blocks), dtype=mx.bool_)

        selected_scores = mx.where(valid_blocks, block_scores, neg)
        forced_init_blocks = (init_blocks & causal_block) & valid_blocks
        forced_local_blocks = (local_blocks & causal_block) & valid_blocks
        selected_scores = mx.where(
            forced_init_blocks,
            mx.array(1e30, dtype=selected_scores.dtype),
            selected_scores,
        )
        selected_scores = mx.where(
            forced_local_blocks,
            mx.array(1e29, dtype=selected_scores.dtype),
            selected_scores,
        )

        k_eff = min(self.sparse_topk_blocks, num_blocks)
        topk_idx = mx.argpartition(-selected_scores, kth=k_eff - 1, axis=-1)[
            ..., :k_eff
        ]
        topk_valid = mx.take_along_axis(valid_blocks, topk_idx, axis=-1)
        if not build_token_mask:
            return None, topk_idx[:, None], topk_valid[:, None]

        block_selected = mx.any(topk_idx[..., None] == blocks, axis=-2)
        block_selected = block_selected & valid_blocks

        key_blocks = (kpos // block_size).astype(mx.int32)
        key_blocks = mx.broadcast_to(key_blocks[None, None, :], (B, L, total_len))
        key_selected = mx.take_along_axis(block_selected, key_blocks, axis=-1)
        key_selected = key_selected & causal

        sparse_mask = key_selected[:, None]
        sparse_mask = self._merge_sparse_mask(sparse_mask, mask)
        if return_block_indices:
            return sparse_mask, topk_idx[:, None], topk_valid[:, None]
        return sparse_mask

    def _build_sparse_decode_indices(
        self,
        idx_queries: mx.array,
        idx_keys: mx.array,
        q_start: int,
        q_positions: Optional[mx.array] = None,
    ):
        B, H_idx, L, _ = idx_queries.shape
        total_len = idx_keys.shape[2]
        if B != 1 or L != 1:
            return None
        explicit_positions = q_positions is not None
        if not explicit_positions and q_start + 1 != total_len:
            return None

        block_size = self.sparse_block_size
        num_blocks = (total_len + block_size - 1) // block_size
        pad = num_blocks * block_size - total_len
        neg = mx.array(-float("inf"), dtype=mx.float32)
        qpos = None

        scores = mx.matmul(
            idx_queries.astype(mx.float32),
            idx_keys.astype(mx.float32).swapaxes(-1, -2),
        )
        scores = scores * self.scale
        if explicit_positions:
            qpos = self._sparse_query_positions(B, L, q_start, q_positions)
            kpos = mx.arange(total_len)
            causal = kpos[None, None, None, :] <= qpos[:, None, :, None]
            scores = mx.where(causal, scores, neg)
        if pad:
            pad_values = mx.full(
                (*scores.shape[:-1], pad), -float("inf"), dtype=scores.dtype
            )
            scores = mx.concatenate([scores, pad_values], axis=-1)

        scores = scores.reshape(B, H_idx, L, num_blocks, block_size)
        if self.sparse_score_type == "lse":
            block_scores = mx.logsumexp(scores, axis=-1)
        else:
            block_scores = mx.max(scores, axis=-1)
        block_scores = mx.max(block_scores, axis=1, keepdims=True)
        selected_scores = mx.where(block_scores == block_scores, block_scores, neg)

        blocks = mx.arange(num_blocks)
        if explicit_positions:
            cur_block = qpos // block_size
            valid_blocks = blocks[None, None, None, :] <= cur_block[:, None, :, None]
            selected_scores = mx.where(valid_blocks, selected_scores, neg)
        else:
            valid_blocks = None

        if self.sparse_init_blocks > 0:
            init_blocks = blocks < self.sparse_init_blocks
            if valid_blocks is not None:
                init_blocks = init_blocks[None, None, None, :] & valid_blocks
            else:
                init_blocks = init_blocks[None, None, None, :]
            selected_scores = mx.where(
                init_blocks,
                mx.array(1e30, dtype=selected_scores.dtype),
                selected_scores,
            )

        if self.sparse_local_blocks > 0:
            if explicit_positions:
                cur_block = qpos // block_size
                local_start = mx.maximum(cur_block - self.sparse_local_blocks + 1, 0)
                local_blocks = (
                    blocks[None, None, None, :] >= local_start[:, None, :, None]
                ) & (blocks[None, None, None, :] <= cur_block[:, None, :, None])
                if valid_blocks is not None:
                    local_blocks = local_blocks & valid_blocks
            else:
                cur_block = q_start // block_size
                local_start = max(cur_block - self.sparse_local_blocks + 1, 0)
                local_blocks = (blocks >= local_start) & (blocks <= cur_block)
                local_blocks = local_blocks[None, None, None, :]
            selected_scores = mx.where(
                local_blocks,
                mx.array(1e29, dtype=selected_scores.dtype),
                selected_scores,
            )

        k_eff = min(self.sparse_topk_blocks, num_blocks)
        topk_idx = mx.argpartition(-selected_scores, kth=k_eff - 1, axis=-1)[
            ..., :k_eff
        ]
        if valid_blocks is None:
            topk_valid = mx.ones(topk_idx.shape, dtype=mx.bool_)
        else:
            topk_valid = mx.take_along_axis(valid_blocks, topk_idx, axis=-1)
        return topk_idx, topk_valid

    @staticmethod
    def _expand_2d_mask(mask: mx.array, B: int, L: int, total_len: int):
        if mask.shape[-1] != total_len:
            mask = mask[..., :total_len]

        if mask.shape[0] == B:
            return mask[:, None, None, :]
        if mask.shape[0] == L:
            return mask[None, None, :, :]
        return mask[None, None, :, :]

    @staticmethod
    def _normalize_attention_mask(
        mask: Optional[mx.array],
        B: int,
        L: int,
        total_len: int,
        *,
        causal: bool = False,
    ):
        if mask is None or isinstance(mask, str):
            return mask
        if _is_integer_mask(mask):
            mask = mask.astype(mx.bool_)
        if mask.ndim == 2:
            mask = MiniMaxAttention._expand_2d_mask(mask, B, L, total_len)
        elif mask.ndim == 3:
            if mask.shape[-1] != total_len:
                mask = mask[..., :total_len]
            if mask.shape[0] == B:
                mask = mask[:, None, :, :]
            else:
                mask = mask[None, :, :, :]
        elif mask.shape[-1] != total_len:
            mask = mask[..., :total_len]
        if causal:
            mask = MiniMaxAttention._merge_causal_mask(mask, L, total_len)
        return mask

    @staticmethod
    def _merge_causal_mask(mask: mx.array, L: int, total_len: int):
        qpos = mx.arange(total_len - L, total_len)
        kpos = mx.arange(total_len)
        causal_mask = kpos[None, :] <= qpos[:, None]
        causal_mask = causal_mask[None, None, :, :]
        if _is_bool_mask(mask):
            return mask & causal_mask
        causal_bias = mx.where(
            causal_mask,
            mx.array(0.0, dtype=mask.dtype),
            mx.array(-float("inf"), dtype=mask.dtype),
        )
        return mask + causal_bias

    @staticmethod
    def _selection_valid_mask(
        mask: Optional[mx.array], B: int, H_idx: int, L: int, total_len: int
    ):
        valid = MiniMaxAttention._normalize_attention_mask(mask, B, L, total_len)
        if valid is None or isinstance(valid, str):
            return None

        if not _is_bool_mask(valid):
            valid = valid.astype(mx.float32) > mx.array(-1e20, dtype=mx.float32)

        if valid.shape[1] != 1 and valid.shape[1] != H_idx:
            valid = mx.any(valid, axis=1, keepdims=True)

        heads = H_idx if valid.shape[1] == H_idx else 1
        valid = mx.broadcast_to(valid, (B, heads, L, total_len))
        return valid

    @staticmethod
    def _merge_sparse_mask(sparse_mask: mx.array, mask: Optional[mx.array]):
        B, _, L, total_len = sparse_mask.shape
        mask = MiniMaxAttention._normalize_attention_mask(mask, B, L, total_len)
        if mask is None or isinstance(mask, str):
            return sparse_mask
        if _is_bool_mask(mask):
            return sparse_mask & mask
        sparse_bias = mx.where(
            sparse_mask,
            mx.array(0.0, dtype=mask.dtype),
            mx.array(-float("inf"), dtype=mask.dtype),
        )
        return sparse_bias + mask

    def _can_use_sparse_decode_attention(
        self,
        queries: mx.array,
        keys: mx.array,
        original_mask: Optional[mx.array],
    ) -> bool:
        B, H, L, _ = queries.shape
        _, K, total_len, _ = keys.shape
        num_blocks = (total_len + self.sparse_block_size - 1) // self.sparse_block_size
        selected_len = min(self.sparse_topk_blocks, num_blocks) * self.sparse_block_size
        sparse_density = selected_len / total_len if total_len else 1.0
        return (
            original_mask is None
            and B == 1
            and L == 1
            and self.index_heads == K
            and H % K == 0
            and selected_len < total_len
            and sparse_density <= _MSA_SPARSE_DECODE_DEFAULT_MAX_DENSITY
        )

    def _sparse_block_offsets(self, dtype):
        cache = getattr(self, "_minimax_m3_sparse_block_offsets_cache", None)
        if (
            cache is None
            or cache[0] != dtype
            or cache[1].shape[0] != self.sparse_block_size
        ):
            offsets = mx.arange(self.sparse_block_size, dtype=dtype)
            mx.eval(offsets)
            object.__setattr__(
                self, "_minimax_m3_sparse_block_offsets_cache", (dtype, offsets)
            )
            return offsets
        return cache[1]

    def _sparse_decode_attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        topk_idx: mx.array,
        topk_valid: mx.array,
        q_start: int,
        *,
        topk_all_valid: bool = False,
        q_positions: Optional[mx.array] = None,
    ):
        B, H, L, D = queries.shape
        _, K, total_len, _ = keys.shape
        index_heads = topk_idx.shape[1]
        if B != 1 or L != 1 or index_heads not in (1, K) or H % K != 0:
            return None

        selected_len = topk_idx.shape[-1] * self.sparse_block_size
        if selected_len >= total_len:
            return None

        block_offsets = self._sparse_block_offsets(topk_idx.dtype)
        positions = topk_idx[..., None] * self.sparse_block_size + block_offsets
        positions = positions.reshape(B, index_heads, L, selected_len)
        if index_heads == 1 and K != 1:
            positions = mx.broadcast_to(positions, (B, K, L, selected_len))

        if topk_all_valid and q_positions is None and q_start + L == total_len:
            valid = positions < total_len
            positions = mx.minimum(
                positions, mx.array(total_len - 1, dtype=positions.dtype)
            )
        else:
            valid = mx.broadcast_to(
                topk_valid[..., None],
                topk_idx.shape + (self.sparse_block_size,),
            )
            valid = valid.reshape(B, index_heads, L, selected_len)
            if index_heads == 1 and K != 1:
                valid = mx.broadcast_to(valid, (B, K, L, selected_len))
            qpos = self._sparse_query_positions(B, L, q_start, q_positions).astype(
                positions.dtype
            )
            valid = (
                valid
                & (positions < total_len)
                & (positions <= qpos[:, None, :, None])
            )
            positions = mx.where(
                valid, positions, mx.array(total_len, dtype=positions.dtype)
            )
            valid = positions < total_len
            positions = mx.minimum(
                positions, mx.array(total_len - 1, dtype=positions.dtype)
            )

        gather_idx = positions[:, :, 0, :, None]
        gather_idx = mx.broadcast_to(gather_idx, (B, K, selected_len, D))
        compact_keys = mx.take_along_axis(keys, gather_idx, axis=2)
        compact_values = mx.take_along_axis(values, gather_idx, axis=2)

        repeat = H // K
        compact_mask = mx.repeat(valid, repeat, axis=1)
        return scaled_dot_product_attention(
            queries,
            compact_keys,
            compact_values,
            cache=None,
            scale=self.scale,
            mask=compact_mask,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        offset = 0 if cache is None else cache.offset
        rope_offset = offset
        if position_ids is not None:
            if position_ids.ndim == 2:
                rope_offset = position_ids[:, 0]
            else:
                rope_offset = position_ids[0] if position_ids.ndim > 0 else position_ids
        sparse_q_start = 0 if cache is None else getattr(cache, "index_offset", offset)
        sparse_q_positions = position_ids if position_ids is not None else None
        original_mask = mask
        use_sparse_mask = self.has_sparse_index and (
            cache is None or hasattr(cache, "update_index_and_fetch")
        )

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = queries.reshape(B, L, self.num_attention_heads, self.head_dim)
        keys = keys.reshape(B, L, self.num_key_value_heads, self.head_dim)
        values = values.reshape(B, L, self.num_key_value_heads, self.head_dim)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        if use_sparse_mask:
            idx_queries = self.index_q_proj(x)
            idx_keys = self.index_k_proj(x)
            idx_queries = idx_queries.reshape(B, L, self.index_heads, self.index_dim)
            idx_keys = idx_keys.reshape(B, L, 1, self.index_dim)
            idx_queries = self.index_q_norm(idx_queries).transpose(0, 2, 1, 3)
            idx_keys = self.index_k_norm(idx_keys).transpose(0, 2, 1, 3)
            idx_queries = self.rope(idx_queries, offset=rope_offset)
            idx_keys = self.rope(idx_keys, offset=rope_offset)
        else:
            idx_queries = None
            idx_keys = None

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=rope_offset)
            keys = self.rope(keys, offset=rope_offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries, offset=rope_offset)
            keys = self.rope(keys, offset=rope_offset)

        if use_sparse_mask:
            if cache is not None and hasattr(cache, "update_index_and_fetch"):
                idx_keys = cache.update_index_and_fetch(idx_keys)
            mask = self._normalize_attention_mask(
                mask, B, L, keys.shape[2], causal=sparse_q_positions is None
            )
            if idx_keys.shape[2] > self.sparse_block_size * self.sparse_topk_blocks:
                compact_candidate = self._can_use_sparse_decode_attention(
                    queries, keys, original_mask
                )
                sparse_mask = None
                decode_indices = None
                if compact_candidate and mask is None:
                    decode_indices = self._build_sparse_decode_indices(
                        idx_queries,
                        idx_keys,
                        int(sparse_q_start),
                        q_positions=sparse_q_positions,
                    )
                if decode_indices is None:
                    sparse_mask, topk_idx, topk_valid = self._build_sparse_mask(
                        idx_queries,
                        idx_keys,
                        int(sparse_q_start),
                        mask,
                        return_block_indices=True,
                        build_token_mask=not compact_candidate,
                        q_positions=sparse_q_positions,
                    )
                else:
                    topk_idx, topk_valid = decode_indices
                if compact_candidate:
                    output = self._sparse_decode_attention(
                        queries,
                        keys,
                        values,
                        topk_idx,
                        topk_valid,
                        int(sparse_q_start),
                        topk_all_valid=(
                            decode_indices is not None and sparse_q_positions is None
                        ),
                        q_positions=sparse_q_positions,
                    )
                    if output is not None:
                        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                        return self.o_proj(output)
                    if sparse_mask is None:
                        sparse_mask = self._build_sparse_mask(
                            idx_queries,
                            idx_keys,
                            int(sparse_q_start),
                            mask,
                            q_positions=sparse_q_positions,
                        )
                mask = sparse_mask
        else:
            mask = self._normalize_attention_mask(
                mask, B, L, keys.shape[2], causal=True
            )

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MiniMaxSparseMoeBlock(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor
        self.scoring_func = args.scoring_func
        self.shared_expert_index = args.num_local_experts
        self.pack_shared_expert = (
            args.n_shared_experts == 1
            and args.shared_intermediate_size == args.intermediate_size
        )

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        activation = MiniMaxSwiGLUOAI(
            args.swiglu_alpha,
            args.swiglu_limit,
            args.swiglu_beta,
        )
        if self.pack_shared_expert:
            self.switch_mlp = MiniMaxPackedSwitchGLU(
                args.hidden_size,
                args.intermediate_size,
                args.num_local_experts + 1,
                activation=activation,
            )
        else:
            self.switch_mlp = SwitchGLU(
                args.hidden_size,
                args.intermediate_size,
                args.num_local_experts,
                activation=activation,
            )
        self.shared_experts = (
            MiniMaxMLP(
                args.hidden_size,
                args.shared_intermediate_size,
                args.swiglu_alpha,
                args.swiglu_limit,
                args.swiglu_beta,
                bias=False,
            )
            if args.n_shared_experts and not self.pack_shared_expert
            else None
        )
        self.e_score_correction_bias = (
            mx.zeros((args.num_local_experts,)) if args.use_routing_bias else None
        )
        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        gates = self.gate(x.astype(mx.float32))
        if self.e_score_correction_bias is not None:
            inds, scores = _minimax_moe_select(
                gates,
                self.e_score_correction_bias,
                self.num_experts_per_tok,
                self.routed_scaling_factor,
                self.scoring_func,
            )
            scores = scores.astype(x.dtype)
        else:
            if self.scoring_func == "sigmoid":
                scores = mx.sigmoid(gates)
            else:
                scores = mx.softmax(gates, axis=-1, precise=True)

            k = self.num_experts_per_tok
            inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
            scores = mx.take_along_axis(scores, inds, axis=-1)
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
            scores = (scores * self.routed_scaling_factor).astype(x.dtype)
        if self.pack_shared_expert:
            shared_inds = mx.full(
                (*inds.shape[:-1], 1), self.shared_expert_index, dtype=inds.dtype
            )
            shared_scores = mx.ones((*scores.shape[:-1], 1), dtype=scores.dtype)
            inds = mx.concatenate([inds, shared_inds], axis=-1)
            scores = mx.concatenate([scores, shared_scores], axis=-1)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class MiniMaxDecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MiniMaxAttention(args, layer_idx)
        self.input_layernorm = MiniMaxRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
        )
        self.post_attention_layernorm = MiniMaxRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
        )
        self.is_moe_layer = args.is_moe_layer(layer_idx)
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxSparseMoeBlock(args)
        else:
            self.mlp = MiniMaxMLP(
                args.hidden_size,
                args.dense_intermediate_size,
                args.swiglu_alpha,
                args.swiglu_limit,
                args.swiglu_beta,
                bias=False,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        h = x + self.self_attn(
            self.input_layernorm(x), mask, cache, position_ids=position_ids
        )
        mlp = self.block_sparse_moe if self.is_moe_layer else self.mlp
        return h + mlp(self.post_attention_layernorm(h))


class MiniMaxM3Model(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MiniMaxDecoderLayer(args=args, layer_idx=layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = MiniMaxRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
        )

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        capture_layer_ids: Optional[List[int]] = None,
        hidden_sink: Optional[list] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            mask = create_attention_mask(
                h, cache[0] if cache and cache[0] is not None else cache
            )

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for idx, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, c, position_ids=position_ids)
            if hidden_sink is not None and idx in capture_set:
                hidden_sink.append(h)

        if hidden_sink is not None and not capture_set:
            hidden_sink.append(h)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: Optional[ModelConfig] = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = MiniMaxM3Model(args)
        self._position_ids = None
        self._rope_deltas = None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        del image_grid_thw, video_grid_thw
        batch_size, seq_length = input_ids.shape
        positions = mx.arange(seq_length, dtype=mx.int32)
        positions = mx.broadcast_to(positions[None, :], (batch_size, seq_length))

        if attention_mask is not None:
            seen_token = mx.cumsum(attention_mask.astype(mx.int32), axis=-1) > 0
            left_padding = seq_length - mx.sum(
                seen_token.astype(mx.int32), axis=-1
            )
            positions = positions - left_padding.astype(positions.dtype)[:, None]

        rope_deltas = mx.zeros((batch_size, 1), dtype=positions.dtype)
        return positions, rope_deltas

    @staticmethod
    def _first_cache(cache):
        if not cache:
            return None
        for entry in cache:
            if entry is not None:
                return entry
        return None

    @staticmethod
    def _position_ids_from_cache_offsets(
        offsets: mx.array,
        seq_length: int,
    ) -> mx.array:
        offsets = mx.maximum(offsets, 0).astype(mx.int32)
        positions = mx.arange(seq_length, dtype=mx.int32)
        return offsets[:, None] + positions[None, :]

    @staticmethod
    def _scalar_cache_offset(cache_offset) -> int:
        if isinstance(cache_offset, mx.array):
            if cache_offset.ndim == 0:
                return int(cache_offset.item())
            return 0
        return int(cache_offset)

    def _slice_cached_position_ids(
        self,
        batch_size: int,
        seq_length: int,
        cache_offset: int,
    ):
        cached = self._position_ids
        if (
            cached is not None
            and cached.ndim == 2
            and cached.shape[0] == batch_size
            and cached.shape[-1] >= cache_offset + seq_length
        ):
            return cached[:, cache_offset : cache_offset + seq_length]
        return None

    @staticmethod
    def _offset_position_ids(cache_offset, batch_size: int, seq_length: int):
        if isinstance(cache_offset, mx.array):
            if cache_offset.ndim > 0:
                return LanguageModel._position_ids_from_cache_offsets(
                    cache_offset[:batch_size], seq_length
                )
            cache_offset = int(cache_offset.item())
        cache_offset = int(cache_offset)
        base = mx.arange(cache_offset, cache_offset + seq_length, dtype=mx.int32)
        return mx.broadcast_to(base[None, :], (batch_size, seq_length))

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        capture_layer_ids = kwargs.pop("capture_layer_ids", None)
        return_hidden = kwargs.pop("return_hidden", False)
        return_shared_kv = kwargs.pop("return_shared_kv", False)
        skip_logits = kwargs.pop("skip_logits", False)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        kwargs.pop("visual_pos_masks", None)
        rope_deltas_kw = kwargs.pop("rope_deltas", None)
        if (
            mask is None
            and attention_mask is not None
            and attention_mask.shape[-1] == inputs.shape[-1]
        ):
            mask = attention_mask

        if (
            position_ids is None
            and inputs.shape[0] > 1
        ):
            first_cache = self._first_cache(cache)
            cache_offset = 0
            cache_offsets = None
            if first_cache is not None:
                cache_offset = first_cache._idx if hasattr(first_cache, "_idx") else 0
                offset = getattr(first_cache, "offset", None)
                if isinstance(offset, mx.array) and offset.ndim > 0:
                    cache_offsets = offset
                elif offset is not None and not hasattr(first_cache, "_idx"):
                    cache_offset = offset

            scalar_cache_offset = self._scalar_cache_offset(cache_offset)
            if mask is None and first_cache is not None and scalar_cache_offset == 0:
                left_padding = getattr(first_cache, "left_padding", None)
                if (
                    isinstance(left_padding, mx.array)
                    and left_padding.ndim > 0
                    and left_padding.size >= inputs.shape[0]
                ):
                    positions = mx.arange(inputs.shape[-1], dtype=mx.int32)[None, :]
                    mask = positions >= left_padding[: inputs.shape[0], None]

            rope_mask = mask
            if mask is not None and mask.shape[-1] != inputs.shape[-1]:
                rope_mask = None
            if rope_mask is not None and rope_mask.ndim != 2:
                rope_mask = None

            batch_size, seq_length = inputs.shape
            needs_rope_index = (
                cache is None
                or first_cache is None
                or self._rope_deltas is None
                or (cache_offsets is None and scalar_cache_offset == 0)
            )
            if needs_rope_index:
                position_ids = None
                if first_cache is not None:
                    position_ids = self._slice_cached_position_ids(
                        batch_size, seq_length, scalar_cache_offset
                    )
                if position_ids is None:
                    position_ids, rope_deltas = self.get_rope_index(
                        inputs,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        attention_mask=rope_mask,
                    )
                    self._position_ids = position_ids
                    self._rope_deltas = rope_deltas
                elif self._rope_deltas is None:
                    self._rope_deltas = mx.zeros(
                        (batch_size, 1), dtype=position_ids.dtype
                    )
            else:
                rope_deltas = (
                    rope_deltas_kw if rope_deltas_kw is not None else self._rope_deltas
                )
                if cache_offsets is not None:
                    delta = mx.maximum(cache_offsets[:batch_size], 0).astype(mx.int32)
                    if rope_deltas is not None:
                        delta = delta + rope_deltas[:batch_size].squeeze(-1)
                    position_ids = self._position_ids_from_cache_offsets(
                        delta, seq_length
                    )
                else:
                    delta = scalar_cache_offset
                    if rope_deltas is not None:
                        delta = delta + int(rope_deltas.reshape(-1)[0].item())
                    position_ids = self._offset_position_ids(
                        delta, batch_size, seq_length
                    )
        hidden_sink = kwargs.pop(
            "hidden_sink",
            [] if capture_layer_ids is not None or return_hidden else None,
        )

        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
            position_ids=position_ids,
        )

        logits = self.logits_from_hidden(out) if skip_logits is False else None
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            gdn_states=None,
            shared_kv_states={} if return_shared_kv else None,
        )

    def logits_from_hidden(self, hidden: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self.logits_from_hidden(self.model.norm(hidden))

    def rollback_speculative_cache(
        self,
        caches: List[Any],
        gdn_states: Any,
        accepted: Any,
        block_size: int,
    ) -> int:
        del gdn_states
        if isinstance(accepted, int):
            accepted = mx.array([accepted])
        elif not isinstance(accepted, mx.array):
            accepted = mx.array(accepted)
        if accepted.ndim == 0:
            accepted = accepted.reshape(1)

        max_a = int(accepted.max().item())
        n = max_a + 1
        trim = int(block_size) - n
        is_batch = accepted.size > 1
        valid_ends = accepted + 1

        for cache in caches:
            if cache is None:
                continue

            if trim > 0 and hasattr(cache, "trim"):
                cache.trim(trim)

            if not is_batch or not hasattr(cache, "_idx"):
                continue

            kv_len = int(cache._idx)
            verify_start = kv_len - n
            if verify_start < 0:
                continue
            valid_ends_list = [int(v) for v in valid_ends.tolist()]

            kv_cache = getattr(cache, "kv_cache", cache)
            if getattr(kv_cache, "keys", None) is not None:
                for bi, valid_end in enumerate(valid_ends_list):
                    start = verify_start + valid_end
                    if start < kv_len:
                        kv_cache.keys[bi, :, start:kv_len, :] = 0
                        kv_cache.values[bi, :, start:kv_len, :] = 0

            index_keys = getattr(cache, "index_keys", None)
            if index_keys is not None:
                idx_len = int(getattr(cache, "index_offset", kv_len))
                idx_verify_start = idx_len - n
                if idx_verify_start < 0:
                    continue
                for bi, valid_end in enumerate(valid_ends_list):
                    start = idx_verify_start + valid_end
                    if start < idx_len:
                        cache.index_keys[bi, :, start:idx_len, :] = 0
        return max_a

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        return [
            MiniMaxM3KVCache() if layer.self_attn.has_sparse_index else KVCache()
            for layer in self.layers
        ]

    @property
    def cast_predicate(self):
        def predicate(k):
            keep_fp32 = "e_score_correction_bias" in k or k.endswith(
                "block_sparse_moe.gate.weight"
            )
            return not keep_fp32

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("block_sparse_moe.gate"):
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return True

        return predicate

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        n = group.size()

        for layer in self.layers:
            if (
                layer.self_attn.has_sparse_index
                and layer.self_attn.index_heads % n != 0
            ):
                raise ValueError(
                    "MiniMax M3 sparse index heads must be divisible by "
                    f"the sharding group size: {layer.self_attn.index_heads} "
                    f"heads, group size {n}."
                )
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, "all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, "all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.num_attention_heads //= n
            layer.self_attn.num_key_value_heads //= n
            if layer.self_attn.has_sparse_index:
                layer.self_attn.index_q_proj = shard_linear(
                    layer.self_attn.index_q_proj,
                    "all-to-sharded",
                    group=group,
                )
                layer.self_attn.index_heads //= n

            if not layer.is_moe_layer:
                continue
            if layer.block_sparse_moe.pack_shared_expert:
                shard_inplace(
                    layer.block_sparse_moe.switch_mlp.gate_up_proj,
                    "all-to-sharded",
                    group=group,
                )
            else:
                shard_inplace(
                    layer.block_sparse_moe.switch_mlp.gate_proj,
                    "all-to-sharded",
                    group=group,
                )
                shard_inplace(
                    layer.block_sparse_moe.switch_mlp.up_proj,
                    "all-to-sharded",
                    group=group,
                )
            shard_inplace(
                layer.block_sparse_moe.switch_mlp.down_proj,
                "sharded-to-all",
                group=group,
            )
            layer.block_sparse_moe.sharding_group = group
