# SPDX-License-Identifier: Apache-2.0
"""Text-only HY V4 Flash Engram residual (Transformers c2ec8505).

The checkpoint's 2-D BF16 embedding is intentionally *not* an MLX parameter.
Only its selected rows cross from the table provider into MLX. The hash runs
on the host: it needs exact int64 arithmetic, and the consumer is a host-side
row gather anyway.
"""

from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class TableProvider(Protocol):
    def gather(self, block: int, row_ids: np.ndarray) -> mx.array:
        """Read layer-local rows; return BF16 of shape (*row_ids.shape, head_dim)."""
        ...


# The Engram's nn.RMSNorm layers use eps=None in HF, i.e. *opmath* epsilon
# for FP32/BF16. This is not the decoder's configurable rms_norm_eps.
_RMS_EPS = float(np.finfo(np.float32).eps)


@dataclass
class _HashState:
    source: mx.array
    lookup: np.ndarray
    multipliers: np.ndarray


def _fp8_round_trip(x: mx.array) -> mx.array:
    """Read as scaled E4M3FN per *head vector*, with HF's dtype boundaries."""
    scale = mx.maximum(mx.max(mx.abs(x), axis=-1, keepdims=True), 1e-12) / 448.0
    packed = mx.to_fp8(mx.clip(x / scale, -448.0, 448.0))
    restored = mx.from_fp8(packed, x.dtype) * scale
    return x + (restored - x)


class _AffineRMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = mx.ones((width,))

    def __call__(self, x):
        # PyTorch applies the learned gain in opmath precision *before* the
        # cast back. Native BF16 MLX RMSNorm rounds the normalized value first.
        return mx.fast.rms_norm(
            x.astype(mx.float32), self.weight.astype(mx.float32), _RMS_EPS
        ).astype(x.dtype)


class EngramLayer(nn.Module):
    def __init__(self, args, block: int):
        super().__init__()
        self.block = block  # model.layers.N is zero-based; the hash seed is not.
        self.layer_id = block + 1
        self.hidden_size = args.hidden_size
        self.hc_mult = args.hc_mult
        self.max_order = args.engram_max_ngram_size
        self.heads_per_order = args.engram_n_head_per_ngram
        self.pad_id = args.engram_pad_id
        self.fp8_embedding = args.engram_fp8_embedding
        self.seed = args.engram_seed
        lookup_size = getattr(args, "engram_lookup_table_size", 120818)
        if not (0 <= self.pad_id < lookup_size):
            raise ValueError("engram_pad_id must be covered by the frozen lookup table")
        orders = args.engram_layer_head_vocab_sizes[str(self.layer_id)]
        if (
            self.max_order < 2
            or self.heads_per_order < 1
            or len(orders) != self.max_order - 1
            or any(len(order) != self.heads_per_order for order in orders)
            or any(type(p) is not int or p < 1 for order in orders for p in order)
            or args.engram_n_embed_per_ngram < 1
            or args.engram_n_embed_per_ngram % self.heads_per_order
        ):
            raise ValueError("invalid Engram head layout")
        self._moduli = np.array(orders, dtype=np.int64)
        self._offsets = np.cumsum(np.r_[np.int64(0), self._moduli.ravel()[:-1]])
        self.head_dim = args.engram_n_embed_per_ngram // self.heads_per_order
        self.num_heads = self._moduli.size
        width = self.num_heads * self.head_dim

        # These names match the original non-table checkpoint tensors exactly.
        self.lookup_table = mx.arange(lookup_size, dtype=mx.int32)
        self.lookup_per_head_norms = [
            _AffineRMSNorm(self.head_dim) for _ in range(self.num_heads)
        ]
        self.value_proj = nn.Linear(width, self.hidden_size)
        self.key_projs = [
            nn.Linear(width, self.hidden_size) for _ in range(self.hc_mult)
        ]
        self.norm_key = [_AffineRMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        self.norm_query = [
            _AffineRMSNorm(self.hidden_size) for _ in range(self.hc_mult)
        ]
        self.unified_pre_gamma = mx.ones((self.hidden_size,))
        self.unified_per_hc_scales = [
            mx.zeros((self.hidden_size,)) for _ in range(self.hc_mult)
        ]
        self._hash_state = None
        self._tables = None

    def bind_tables(self, provider: TableProvider):
        if not callable(getattr(provider, "gather", None)):
            raise TypeError(
                "Engram table provider must implement gather(block, row_ids)"
            )
        if hasattr(provider, "shape_for_block") and provider.shape_for_block(
            self.block
        ) != (int(self._moduli.sum()), self.head_dim):
            raise ValueError("Engram head layout and provider table shape disagree")
        self._tables = provider

    def _hash_inputs(self) -> _HashState:
        # Derive the PCG64 draw from the *loaded* normalization map, not the
        # number of tokenizer rows. Invalidate it if load_weights replaces it.
        source = self.lookup_table
        state = self._hash_state
        if state is None or state.source is not source:
            if (
                source.ndim != 1
                or source.dtype != mx.int32
                or source.size <= self.pad_id
            ):
                raise ValueError("invalid Engram I32 token lookup table")
            lookup = np.asarray(source)
            if lookup.min() < 0:
                raise ValueError("negative compressed Engram token ID")
            compressed_size = int(lookup.max()) + 1
            bound = max(1, int(np.iinfo(np.int64).max) // compressed_size // 2)
            rng = np.random.default_rng(self.seed + 10007 * self.layer_id)
            draws = rng.integers(0, bound, size=self.max_order, dtype=np.int64)
            state = _HashState(source, lookup, draws * 2 + 1)
            self._hash_state = state
        return state

    def hash_ngrams(self, ngram_tokens) -> np.ndarray:
        """Return one *within-head* hash per order/head (no layer or head base)."""
        tokens = np.asarray(ngram_tokens)
        if (
            tokens.ndim != 3
            or tokens.shape[-1] != self.max_order
            or tokens.dtype.kind not in "ui"
        ):
            raise ValueError(
                "expected integer [batch, tokens, max_ngram_size] Engram inputs"
            )
        if tokens.dtype.kind == "i" and (tokens < 0).any():
            raise ValueError("negative token ID in Engram inputs")
        state = self._hash_inputs()
        # Resized/embedding-only ids beyond the normalization map read as PAD.
        # Cast before selecting the pad id (which may not fit narrow token
        # dtypes); UINT64 ids past INT64_MAX also fall back to pad.
        host_ids = tokens.astype(np.int64, copy=False)
        ids = np.where(
            (host_ids >= 0) & (host_ids < len(state.lookup)), host_ids, self.pad_id
        )
        mixed = state.lookup[ids] * state.multipliers
        key = mixed[..., 0]
        heads = []
        for i, moduli in enumerate(self._moduli, start=1):
            key = key ^ mixed[..., i]
            heads.append(key[..., None] % moduli)
        return np.concatenate(heads, axis=-1)

    def __call__(self, streams: mx.array, ngram_tokens: mx.array) -> mx.array:
        """Compute the residual only; the decoder adds it to its input once."""
        if self._tables is None:
            raise RuntimeError("bind a verified Engram table provider before inference")
        if streams.shape != (*ngram_tokens.shape[:2], self.hc_mult, self.hidden_size):
            raise ValueError("Engram hidden streams and n-gram positions disagree")
        if hasattr(ngram_tokens, "tokens"):
            from .prepared_engram import PreparedEngrams

            if not isinstance(ngram_tokens, PreparedEngrams):
                raise TypeError("unrecognized prepared Engram input")
            x = ngram_tokens.take(self, streams)
            ngram_tokens = ngram_tokens.tokens
        else:
            rows = self.hash_ngrams(ngram_tokens) + self._offsets
            x = self._tables.gather(self.block, rows)
        if (
            not isinstance(x, mx.array)
            or x.dtype != mx.bfloat16
            or x.shape != (*ngram_tokens.shape[:2], self.num_heads, self.head_dim)
        ):
            raise ValueError(
                "Engram provider must return one original BF16 vector per requested row"
            )
        x = x.astype(streams.dtype)
        if self.fp8_embedding:
            x = _fp8_round_trip(x)
        # Normalize every head independently in one kernel, then concatenate
        # in n-gram-major, head-minor order for the shared/key projections.
        gains = mx.stack([n.weight for n in self.lookup_per_head_norms]).astype(
            mx.float32
        )
        normalized = mx.fast.rms_norm(x.astype(mx.float32), None, _RMS_EPS)
        x = (normalized * gains).astype(x.dtype).reshape(*x.shape[:2], -1)

        gates = []
        for i in range(self.hc_mult):
            key = self.norm_key[i](self.key_projs[i](x))
            query = self.norm_query[i](streams[:, :, i])
            product = key * query
            dot = product.astype(mx.float32).sum(axis=-1).astype(product.dtype)
            # HF scalar multiply uses the full-precision scalar, then casts.
            gate = (dot.astype(mx.float32) * self.hidden_size**-0.5).astype(dot.dtype)
            gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
            gates.append(mx.sigmoid(gate.astype(mx.float32)).astype(gate.dtype))
        gates = mx.stack(gates, axis=2)[..., None]
        signal = self.unified_pre_gamma * mx.fast.rms_norm(
            self.value_proj(x), None, _RMS_EPS
        )
        signal = signal[:, :, None, :]
        residual = gates * signal + mx.stack(self.unified_per_hc_scales) * signal
        return mx.where(ngram_tokens[..., 0, None, None] != self.pad_id, residual, 0.0)
