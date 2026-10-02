# SPDX-License-Identifier: MIT
"""Token-normalized Engram lookup with request-local lookback."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE" all hash the same way.
    Returns the lookup plus the size of the compressed vocab -- and that size matters beyond bounds
    checking, because every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # a private-use char, so a token that is exactly one space survives Strip() instead of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # the raw Rust tokenizer, matching what training decodes with (no clean_up_tokenization_spaces)
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def _prime(n):
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


class NgramHash:
    """CPU int64 hashing matches the official NumPy RNG, not Qwen's SplitMix64.

    Only the last n-1 compressed token ids live in the request cache. Image
    positions stop lookback; they are not removed from the token timeline.
    """

    def __init__(self, config, token_map):
        self.config = config
        self.token_map = np.asarray(token_map, dtype=np.int64)
        vocab = int(self.token_map.max()) + 1
        if vocab != config.engram_compressed_vocab_size:
            raise ValueError(f"Engram compressed vocabulary mismatch: {vocab}")
        self.pad_id = int(self.token_map[config.engram_pad_id])
        primes, multipliers, seen = [], [], set()
        for layer_id in config.engram_layer_ids:
            groups = []
            for _ in range(config.engram_max_ngram_size - 1):
                current, group = config.engram_vocab_size - 1, []
                for _ in range(config.engram_n_heads):
                    current += 1
                    while not _prime(current) or current in seen:
                        current += 1
                    seen.add(current)
                    group.append(current)
                groups.append(group)
            primes.append(groups)
            rng = np.random.default_rng(10007 * layer_id)
            bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
            multipliers.append(
                rng.integers(0, bound, config.engram_max_ngram_size, dtype=np.int64) * 2
                + 1
            )
        self.primes = np.array(primes, dtype=np.int64)
        self.multipliers = np.array(multipliers, dtype=np.int64)
        flat = self.primes.reshape(len(primes), -1)
        self.offsets = np.cumsum(
            np.concatenate(
                [np.zeros((len(primes), 1), dtype=np.int64), flat[:, :-1]], -1
            ),
            -1,
        )
        if list(flat.sum(-1)) != list(config.engram_num_embeddings):
            raise ValueError("Engram table rows do not match the official prime layout")

    def __call__(self, ids, history=None, image_mask=None):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.min() < 0 or ids.max() >= len(self.token_map):
            raise ValueError("Token id outside Engram tokenizer vocabulary")
        tokens = self.token_map[ids]
        if image_mask is not None:
            tokens = np.where(np.asarray(image_mask), -1, tokens)
        depth = self.config.engram_max_ngram_size - 1
        if history is None:
            history = np.full((len(ids), depth), -1, dtype=np.int64)
        joined = np.concatenate([np.asarray(history), tokens], axis=1)
        positions = np.arange(tokens.shape[1]) + depth
        blocked = np.zeros_like(tokens, dtype=bool)
        lookback = []
        for shift in range(depth + 1):
            source = joined[:, positions - shift]
            blocked |= source == -1
            lookback.append(np.where(blocked, self.pad_id, source))
        product = np.stack(lookback, -1)[:, :, None, :] * self.multipliers
        rolling, hashes = product[..., 0], []
        for shift in range(1, depth + 1):
            rolling = np.bitwise_xor(rolling, product[..., shift])
            hashes.append(rolling[..., None] % self.primes[:, shift - 1])
        return np.concatenate(hashes, -1) + self.offsets, joined[:, -depth:]


class Engram(nn.Module):
    def __init__(self, config, table_index):
        super().__init__()
        self.embed = nn.Embedding(
            config.engram_num_embeddings[table_index], config.engram_head_dim
        )
        self.wkv = nn.Linear(
            (config.engram_max_ngram_size - 1)
            * config.engram_n_heads
            * config.engram_head_dim,
            config.dim * (config.hc_mult + 1),
            bias=False,
        )
        self.q_weight = mx.ones((config.hc_mult, config.dim))
        self.k_weight = mx.ones((config.hc_mult, config.dim))
        self._config = config

    def __call__(self, h, ids, image_mask=None):
        c = self._config
        kv = self.wkv(self.embed(mx.array(ids)).flatten(-2))
        key, value = mx.split(kv, [c.dim * c.hc_mult], axis=-1)
        key = key.astype(mx.float32).reshape(*h.shape)
        x = h.astype(mx.float32)
        inv = mx.rsqrt(mx.mean(x * x, -1) + c.norm_eps) * mx.rsqrt(
            mx.mean(key * key, -1) + c.norm_eps
        )
        dot = mx.sum(x * self.q_weight * self.k_weight * key, -1) * inv * c.dim**-0.5
        gate = mx.sigmoid(mx.sign(dot) * mx.sqrt(mx.maximum(mx.abs(dot), 1e-6)))
        # copysign(+sqrt, 0) is positive in the official reference.
        gate = mx.where(dot == 0, mx.sigmoid(mx.array(0.001)), gate)
        if image_mask is not None:
            gate = mx.where(image_mask[..., None], 0, gate)
        return (x + gate[..., None] * value.astype(mx.float32)[..., None, :]).astype(
            h.dtype
        )
