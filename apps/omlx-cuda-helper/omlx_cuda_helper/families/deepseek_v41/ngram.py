# SPDX-License-Identifier: Apache-2.0
"""Engram n-gram hashing and the table rows the host sends for a chunk.

Ported from oMLX (omlx/patches/deepseek_v41/engram.py and storage.py). The
hashes run over token ids that are compressed with the checkpoint's
tokenizer, so tokens that normalize alike share an id.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Regex, Tokenizer, normalizers

from ...tensors import unpack
from ...wire import HostArray
from . import quant
from .config import Config


def compressed_token_map(model_dir: str | Path) -> np.ndarray:
    """Compressed id of every token id of the checkpoint's tokenizer."""
    # A private-use character keeps a token that is one space from stripping
    # to the empty string.
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
    backend = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    size = backend.get_vocab_size(with_added_tokens=True)
    known: dict[str, int] = {}
    lookup = np.zeros(size, dtype=np.int64)
    for token_id in range(size):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # Part of a UTF-8 sequence: keyed by its raw form.
            key = backend.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        lookup[token_id] = known.setdefault(key, len(known))
    return lookup


def _prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


class NgramHash:
    """Table row ids of each position's n-grams, for every Engram layer.

    int64 arithmetic wraps as in the reference. Only the last n-1 compressed
    ids of the prompt are kept as history.
    """

    def __init__(self, config: Config, token_map: np.ndarray):
        self.token_map = np.asarray(token_map, dtype=np.int64)
        vocab = int(self.token_map.max()) + 1
        if vocab != config.engram_compressed_vocab_size:
            raise ValueError(f"Engram compressed vocabulary mismatch: {vocab}")
        self.depth = config.engram_max_ngram_size - 1
        self.pad_id = int(self.token_map[config.engram_pad_id])
        primes, multipliers, seen = [], [], set()
        for layer in config.engram_layer_ids:
            groups = []
            for _ in range(self.depth):
                current, group = config.engram_vocab_size - 1, []
                for _ in range(config.engram_n_heads):
                    current += 1
                    while not _prime(current) or current in seen:
                        current += 1
                    seen.add(current)
                    group.append(current)
                groups.append(group)
            primes.append(groups)
            rng = np.random.default_rng(10007 * layer)
            bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
            multipliers.append(
                rng.integers(0, bound, self.depth + 1, dtype=np.int64) * 2 + 1
            )
        self.primes = np.array(primes, dtype=np.int64)
        self.multipliers = np.array(multipliers, dtype=np.int64)
        flat = self.primes.reshape(len(primes), -1)
        self.offsets = np.cumsum(
            np.concatenate([np.zeros((len(primes), 1), np.int64), flat[:, :-1]], -1),
            -1,
        )
        if list(flat.sum(-1)) != list(config.engram_num_embeddings):
            raise ValueError("Engram table rows do not match the prime layout")

    def __call__(self, ids: np.ndarray, history: np.ndarray | None):
        """(row ids [1, L, tables, heads * (n-1)], history after the chunk)."""
        ids = np.asarray(ids, dtype=np.int64).reshape(1, -1)
        if ids.min() < 0 or ids.max() >= len(self.token_map):
            raise ValueError("Token id outside the Engram tokenizer vocabulary")
        tokens = self.token_map[ids]
        if history is None:
            history = np.full((1, self.depth), -1, dtype=np.int64)
        joined = np.concatenate([np.asarray(history, dtype=np.int64), tokens], axis=1)
        positions = np.arange(tokens.shape[1]) + self.depth
        blocked = np.zeros_like(tokens, dtype=bool)
        lookback = []
        for shift in range(self.depth + 1):
            source = joined[:, positions - shift]
            # A missing token stops the lookback at that position.
            blocked |= source == -1
            lookback.append(np.where(blocked, self.pad_id, source))
        product = np.stack(lookback, -1)[:, :, None, :] * self.multipliers
        rolling, hashes = product[..., 0], []
        for shift in range(1, self.depth + 1):
            rolling = np.bitwise_xor(rolling, product[..., shift])
            hashes.append(rolling[..., None] % self.primes[:, shift - 1])
        return np.concatenate(hashes, -1) + self.offsets, joined[:, -self.depth :]


def dedupe(ids: np.ndarray):
    """(unique id tuples, position -> tuple) as the host sends its rows.

    Short chunks and chunks without a repeated tuple keep the position order
    and have no inverse.
    """
    if ids.shape[0] * ids.shape[1] <= 8:
        return ids, None
    flat = ids.reshape(-1, ids.shape[-1])
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    if unique.shape[0] == flat.shape[0]:
        return ids, None
    return unique, inverse.reshape(ids.shape[:-1])


def _decode(value: HostArray) -> mx.array:
    """An array of a table file, named by its safetensors dtype."""
    raw, dtype = value
    if dtype == "BF16":
        return mx.array(np.ascontiguousarray(raw).view(np.uint16)).view(mx.bfloat16)
    if dtype.startswith("F8_E4M3"):
        return quant.from_fp8(mx.array(raw))
    if dtype.startswith("F8_E8M0"):
        if np.any(raw == 255):
            raise ValueError("NaN E8M0 scale in the Engram rows")
        return mx.array(np.exp2(raw.astype(np.float32) - 127))
    return mx.array(raw)


def host_rows(inputs: dict[str, HostArray], layer: int, ids: np.ndarray) -> mx.array:
    """bfloat16 table rows of ``ids`` from the arrays the host sent."""
    name = f"e{layer}."
    if name + "ids" not in inputs:
        raise RuntimeError(f"No Engram rows for layer {layer}")
    if not np.array_equal(inputs[name + "ids"][0], ids):
        raise RuntimeError("Engram rows were sent for other ids")
    bits, group_size, scaled = (int(v) for v in inputs[name + "q"][0])
    data = [inputs[f"{name}{n}"] for n in range(3) if f"{name}{n}" in inputs]
    values = _decode(data[0])
    if bits:
        # One rounding from fp32, as the host's dequantize gives.
        quant = {"group_size": group_size, "bits": bits}
        values = unpack(values, _decode(data[1]), _decode(data[2]), quant)
    elif scaled:
        scales = _decode(data[1])
        values = (values.reshape(values.shape[0], -1, 32) * scales[..., None]).reshape(
            values.shape
        )
    return values.reshape(*ids.shape[:-1], -1).astype(mx.bfloat16)
