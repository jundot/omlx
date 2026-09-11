# SPDX-License-Identifier: Apache-2.0
"""Engram conditional memory for DeepSeek-V4.1.

Official tables are hundreds of millions of rows (see ``engram_num_embeddings``).

Hot-row RAM cache lives in ``engram_cache.py`` (``OMLX_DSV41_ENGRAM_CACHE_GB`` / ``QuantHotRowCache``).

Modes via ``OMLX_DSV41_ENGRAM``:
- ``stub`` / ``lazy`` / ``off`` (default): no Engram layers (ModelArgs clears ids).
- ``mmap`` / ``ssd``: keep layer hooks; embed tables stay on SSD via numpy.memmap;
  only gathered rows are promoted to MLX arrays. Small params (wkv/q/k) load normally.
- ``full``: allocate resident Embedding tables (not recommended on 512GB hosts).

Optional ``OMLX_DSV41_ENGRAM_DIR`` points at a metadata dir (default
``~/llm/DeepSeek-V4.1-Flash-engram``) with ``meta.json`` + ``token_map.npy``.

Mmap gather: unique-index SSD dequant + bf16 upload. Threaded row fetches,
optional madvise prefetch, a tiny per-layer dequant LRU
(``OMLX_DSV41_ENGRAM_ROW_BUF``, default 8192), and an optional multi-GB
**quantized** hot-row cache (``OMLX_DSV41_ENGRAM_CACHE_GB``, default 50)
that keeps FP8 E4M3 + UE8M0 scales in RAM and evicts oldest (LRU) or CLOCK
when full — ~2× denser than a bf16 hot cache.
"""
from __future__ import annotations

import json
import logging
import mmap as mmap_mod
import os
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from omlx.patches.deepseek_v41.engram_cache import (
    QuantHotRowCache,
    _quant_cache_rows_per_layer,
    engram_cache_gb,
    engram_cache_policy,
    engram_cache_stats,
)
from omlx.patches.deepseek_v41.engram_fp8 import (
    _e4m3_lut,
    _e4m3_to_float32,
    _ue8m0_to_float32,
)

logger = logging.getLogger(__name__)

_DEAD = -1


def engram_mode() -> str:
    return os.environ.get("OMLX_DSV41_ENGRAM", "stub").strip().lower()


def engram_tables_enabled() -> bool:
    return engram_mode() in ("mmap", "ssd", "full")


def engram_is_mmap() -> bool:
    return engram_mode() in ("mmap", "ssd")


def engram_dir() -> Path:
    raw = os.environ.get("OMLX_DSV41_ENGRAM_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "llm" / "DeepSeek-V4.1-Flash-engram"


def engram_gather_threads() -> int:
    """Worker threads for SSD row gathers (default 16). 1 disables threading."""
    raw = os.environ.get("OMLX_DSV41_ENGRAM_GATHER_THREADS", "16").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def engram_row_buf() -> int:
    """Tiny per-layer working-set row buffer (default 8192). 0 disables.

    Not the old multi-GB HotRowCache — only a small LRU of recently gathered
    rows to accelerate decode without pinning tens of GB of RAM.
    """
    raw = os.environ.get("OMLX_DSV41_ENGRAM_ROW_BUF", "8192").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 8192


def engram_prefetch() -> bool:
    raw = os.environ.get("OMLX_DSV41_ENGRAM_PREFETCH", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def find_next_prime(start: int, seen_primes: set) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def compute_hash_multipliers(
    layer_ids: Tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> np.ndarray:
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(values * 2 + 1)
    return np.stack(rows, axis=0)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables (mirrors official EngramLayout)."""

    max_ngram_size: int
    layer_ids: Tuple[int, ...]
    num_embeddings: Tuple[int, ...]
    primes: Tuple[Tuple[Tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args) -> Optional["EngramLayout"]:
        layer_ids = tuple(getattr(args, "engram_layer_ids", ()) or ())
        if not layer_ids:
            return None
        nums = tuple(int(x) for x in (getattr(args, "engram_num_embeddings", ()) or ()))
        if len(nums) != len(layer_ids):
            raise ValueError(
                f"engram_num_embeddings length {len(nums)} != "
                f"engram_layer_ids length {len(layer_ids)}"
            )
        max_ngram_size = int(args.engram_max_ngram_size)
        n_heads = int(args.engram_n_heads)
        vocab_size = int(getattr(args, "engram_vocab_size", 0) or 0)
        primes_list: List[Tuple[Tuple[int, ...], ...]] = []
        seen: set = set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes = []
                current = vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes_list.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=nums,
            primes=tuple(primes_list),
            n_heads=n_heads,
            head_dim=int(args.engram_head_dim),
        )

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads


def _row_storage_dtype():
    """Prefer bfloat16 (matches MLX); fall back to float16."""
    try:
        import ml_dtypes

        return ml_dtypes.bfloat16
    except Exception:
        return np.float16


class _TinyRowLRU:
    """Fixed-capacity LRU of dequantized rows (bf16/f16). Decode working set only."""

    def __init__(self, capacity: int, head_dim: int):
        self.capacity = max(0, int(capacity))
        self.head_dim = int(head_dim)
        self.enabled = self.capacity > 0
        self.dtype = _row_storage_dtype()
        self._map: "OrderedDict[int, int]" = OrderedDict()
        self._n = 0
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.data = np.empty((self.capacity, self.head_dim), dtype=self.dtype)
        else:
            self.data = None

    def get_many(self, keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(keys.shape[0])
        hit = np.zeros(n, dtype=bool)
        out = np.empty((n, self.head_dim), dtype=np.float32)
        if not self.enabled or n == 0:
            self.misses += n
            return hit, out
        m = self._map
        data = self.data
        hit_idx = []
        slots = []
        for i in range(n):
            k = int(keys[i])
            slot = m.get(k)
            if slot is None:
                self.misses += 1
                continue
            hit[i] = True
            hit_idx.append(i)
            slots.append(slot)
            self.hits += 1
            m.move_to_end(k)
        if slots:
            out[np.asarray(hit_idx, dtype=np.int64)] = np.asarray(
                data[np.asarray(slots, dtype=np.int64)], dtype=np.float32
            )
        return hit, out

    def put_many(self, keys: np.ndarray, values_f32: np.ndarray) -> None:
        if not self.enabled or keys.size == 0:
            return
        vals = values_f32.astype(self.dtype, copy=False)
        m = self._map
        data = self.data
        for i in range(int(keys.shape[0])):
            k = int(keys[i])
            slot = m.get(k)
            if slot is not None:
                data[slot] = vals[i]
                m.move_to_end(k)
                continue
            if self._n < self.capacity:
                slot = self._n
                self._n += 1
            else:
                _old_k, slot = m.popitem(last=False)
            m[k] = slot
            data[slot] = vals[i]



# Prefetch pool only (whole-layer background jobs). Per-gather I/O uses a
# short-lived executor so nested weight/scale/chunk work cannot deadlock.
_PREFETCH_POOL: Optional[ThreadPoolExecutor] = None


def _prefetch_pool() -> ThreadPoolExecutor:
    """Lazy singleton for whole-layer Engram prefetch jobs."""
    global _PREFETCH_POOL
    if _PREFETCH_POOL is None:
        _PREFETCH_POOL = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="engram-prefetch"
        )
    return _PREFETCH_POOL


# Module-level E4M3 LUT (built once).
def _threaded_memmap_gather(mm: np.memmap, idxs: np.ndarray, n_workers: int) -> np.ndarray:
    """Gather rows from memmap; use threads when idxs is large (SSD random I/O)."""
    n = int(idxs.shape[0])
    cols = int(mm.shape[1])
    if n == 0:
        return np.empty((0, cols), dtype=mm.dtype)
    if n_workers <= 1 or n < 256:
        return np.asarray(mm[idxs])
    out = np.empty((n, cols), dtype=mm.dtype)
    # Split into contiguous index-slices (idxs itself is usually sorted unique).
    bounds = np.linspace(0, n, num=n_workers + 1, dtype=np.int64)

    def _work(lo: int, hi: int) -> None:
        if hi <= lo:
            return
        out[lo:hi] = np.asarray(mm[idxs[lo:hi]])

    # Fresh executor per call: safe under nested prefetch → dequant → gather.
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [
            pool.submit(_work, int(bounds[i]), int(bounds[i + 1]))
            for i in range(n_workers)
        ]
        for f in futs:
            f.result()
    return out


def _madvise_rows(mm: np.memmap, idxs: np.ndarray, merge_gap: int = 65536) -> None:
    """Advise the kernel that sorted row ranges will be needed soon."""
    underlying = getattr(mm, "_mmap", None)
    if underlying is None or idxs.size == 0:
        return
    row_bytes = int(mm.shape[1]) * int(mm.dtype.itemsize)
    starts = idxs.astype(np.int64, copy=False) * row_bytes
    i = 0
    n = int(starts.shape[0])
    try:
        while i < n:
            s = int(starts[i])
            e = s + row_bytes
            j = i + 1
            while j < n and int(starts[j]) <= e + merge_gap:
                e = int(starts[j]) + row_bytes
                j += 1
            underlying.madvise(mmap_mod.MADV_WILLNEED, s, e - s)
            i = j
    except Exception:
        # Best-effort; platforms / numpy builds may lack madvise.
        return


class MmapEngramTable:
    """SSD-backed FP8 Engram embedding table (weight + UE8M0 scales)."""

    def __init__(
        self,
        weight_path: str,
        weight_offset: int,
        scale_path: str,
        scale_offset: int,
        num_embeddings: int,
        head_dim: int,
        block_size: int = 32,
        row_buf: int = 0,
        hot_cache: Optional["QuantHotRowCache"] = None,
    ):
        self.num_embeddings = int(num_embeddings)
        self.head_dim = int(head_dim)
        self.block_size = int(block_size)
        n_scales = self.head_dim // self.block_size
        self.weight_path = weight_path
        self.weight_offset = int(weight_offset)
        self.scale_path = scale_path
        self.scale_offset = int(scale_offset)
        self.weight = np.memmap(
            weight_path,
            dtype=np.uint8,
            mode="r",
            offset=int(weight_offset),
            shape=(self.num_embeddings, self.head_dim),
        )
        self.scale = np.memmap(
            scale_path,
            dtype=np.uint8,
            mode="r",
            offset=int(scale_offset),
            shape=(self.num_embeddings, n_scales),
        )
        self.hot_cache = hot_cache
        # Tiny dequant LRU is redundant when the multi-GB quantized cache is on.
        if hot_cache is not None and hot_cache.enabled:
            self.row_buf = None
        else:
            self.row_buf = _TinyRowLRU(row_buf, head_dim) if row_buf > 0 else None
        self._n_workers = engram_gather_threads()
        self._do_prefetch = engram_prefetch()
        self._prefetch_fut: Optional[Future] = None
        self._prefetch_uniq: Optional[np.ndarray] = None

    def bind_hot_cache(self, cache: Optional["QuantHotRowCache"]) -> None:
        self.hot_cache = cache
        if cache is not None and cache.enabled:
            self.row_buf = None

    def _gather_u8(self, mm: np.memmap, uniq: np.ndarray) -> np.ndarray:
        if self._do_prefetch and uniq.size >= 512:
            _madvise_rows(mm, uniq)
        return _threaded_memmap_gather(mm, uniq, self._n_workers)

    def _gather_quant_rows(self, flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SSD-gather quantized weight+scale for unique ids."""
        if flat.size == 0:
            n_scales = self.head_dim // self.block_size
            return (
                np.empty((0, self.head_dim), dtype=np.uint8),
                np.empty((0, n_scales), dtype=np.uint8),
            )
        n_workers = self._n_workers
        if n_workers > 1 and flat.size >= 256:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fw = pool.submit(self._gather_u8, self.weight, flat)
                fs = pool.submit(self._gather_u8, self.scale, flat)
                return fw.result(), fs.result()
        return self._gather_u8(self.weight, flat), self._gather_u8(self.scale, flat)

    @staticmethod
    def _dequant_u8(
        w: np.ndarray, s: np.ndarray, head_dim: int, block_size: int
    ) -> np.ndarray:
        w_f = _e4m3_lut()[w].reshape(w.shape[0], -1, block_size)
        s_f = _ue8m0_to_float32(s)[..., None]
        return (w_f * s_f).reshape(w.shape[0], head_dim).astype(np.float32, copy=False)

    def _dequant_rows(self, flat: np.ndarray) -> np.ndarray:
        """Dequant unique row indices -> float32 [M, head_dim]."""
        if flat.size == 0:
            return np.empty((0, self.head_dim), dtype=np.float32)
        w, s = self._gather_quant_rows(flat)
        return self._dequant_u8(w, s, self.head_dim, self.block_size)

    def prefetch_unique(self, uniq: np.ndarray) -> None:
        """Kick off background SSD gather+dequant (overlap with compute).

        When a quantized hot-cache is bound, the job also returns the FP8+scale
        rows so ``gather`` can insert them without a second SSD round-trip.
        """
        if not self._do_prefetch or uniq.size == 0:
            return
        self._prefetch_uniq = np.asarray(uniq, dtype=np.int64)
        ids = self._prefetch_uniq
        hot = self.hot_cache
        want_quant = hot is not None and hot.enabled

        def _job():
            if want_quant:
                w, s = self._gather_quant_rows(ids)
                rows = self._dequant_u8(w, s, self.head_dim, self.block_size)
                return rows, w, s
            return self._dequant_rows(ids), None, None

        self._prefetch_fut = _prefetch_pool().submit(_job)

    def _take_prefetch(
        self, uniq: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (dequant_rows, weight_u8, scale_u8); quant arrays may be None."""
        fut = self._prefetch_fut
        cached = self._prefetch_uniq
        self._prefetch_fut = None
        self._prefetch_uniq = None
        if fut is None or cached is None:
            return None, None, None
        if cached.shape != uniq.shape or not np.array_equal(cached, uniq):
            try:
                fut.result()
            except Exception:
                pass
            return None, None, None
        rows, w, s = fut.result()
        return rows, w, s

    def gather(self, indices: np.ndarray) -> mx.array:
        """Gather + dequant rows. ``indices`` any int shape -> bf16/f16 [..., head_dim].

        Order: quantized hot-cache hits first (RAM dequant), then prefetch/SSD
        for misses only. Prefetch must not bypass the hot-cache or warm reuse
        never gets credit and never skips I/O.
        """
        idx = np.asarray(indices, dtype=np.int64)
        orig_shape = idx.shape
        flat = np.clip(idx.reshape(-1), 0, self.num_embeddings - 1)
        if flat.size == 0:
            return mx.zeros((*orig_shape, self.head_dim), dtype=mx.float32)

        uniq, inv = np.unique(flat, return_inverse=True)
        hot = self.hot_cache
        buf = self.row_buf
        rows: Optional[np.ndarray] = None

        if hot is not None and hot.enabled:
            hit_mask, rows = hot.get_many(uniq)
            miss_pos = np.flatnonzero(~hit_mask)
            if miss_pos.size == 0:
                # Full hot hit — drop any pending prefetch result.
                self._take_prefetch(uniq)
            else:
                miss_keys = uniq[miss_pos]
                pref, pref_w, pref_s = self._take_prefetch(uniq)
                if pref is not None and pref_w is not None:
                    # Prefetch covered the full unique set; take miss slice only.
                    rows[miss_pos] = pref[miss_pos]
                    hot.put_many(miss_keys, pref_w[miss_pos], pref_s[miss_pos])
                else:
                    w, s = self._gather_quant_rows(miss_keys)
                    hot.put_many(miss_keys, w, s)
                    rows[miss_pos] = self._dequant_u8(
                        w, s, self.head_dim, self.block_size
                    )
        else:
            pref, pref_w, pref_s = self._take_prefetch(uniq)
            if pref is not None:
                rows = pref
                if buf is not None and buf.enabled:
                    buf.put_many(uniq, rows)
            elif buf is not None and buf.enabled:
                hit_mask, rows = buf.get_many(uniq)
                miss_pos = np.flatnonzero(~hit_mask)
                if miss_pos.size:
                    miss_keys = uniq[miss_pos]
                    dequant = self._dequant_rows(miss_keys)
                    rows[miss_pos] = dequant
                    buf.put_many(miss_keys, dequant)
            else:
                rows = self._dequant_rows(uniq)

        out = rows[inv].reshape(*orig_shape, self.head_dim)
        storage = _row_storage_dtype()
        if out.dtype != storage:
            out = out.astype(storage)
        return mx.array(out)


def load_mmap_tables_from_meta(
    layout: EngramLayout, meta_dir: Optional[Path] = None
) -> Dict[int, MmapEngramTable]:
    meta_dir = meta_dir or engram_dir()
    meta_path = meta_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    tables: Dict[int, MmapEngramTable] = {}
    block = int(meta.get("config", {}).get("fp8_block_size", 32))
    row_buf = engram_row_buf()
    n_layers = len(layout.layer_ids)
    rows_each = _quant_cache_rows_per_layer(n_layers, layout.head_dim, block)
    policy = engram_cache_policy()
    for i, layer_id in enumerate(layout.layer_ids):
        entry = meta["layers"][str(layer_id)]
        w, s = entry["weight"], entry["scale"]
        hot = None
        if rows_each > 0:
            hot = QuantHotRowCache(
                rows_each, layout.head_dim, block_size=block, policy=policy
            )
            logger.info(
                "Engram quant hot-cache layer %s: %s rows (%.2f GB, policy=%s, "
                "%s B/row fp8+scale)",
                layer_id,
                rows_each,
                rows_each * hot.bytes_per_row / (1024**3),
                policy,
                hot.bytes_per_row,
            )
        tables[layer_id] = MmapEngramTable(
            weight_path=w["file"],
            weight_offset=w["offset"],
            scale_path=s["file"],
            scale_offset=s["offset"],
            num_embeddings=layout.num_embeddings[i],
            head_dim=layout.head_dim,
            block_size=block,
            row_buf=row_buf,
            hot_cache=hot,
        )
        logger.info(
            "Engram mmap layer %s: %s rows from %s @%s (gather_threads=%s row_buf=%s hot=%s)",
            layer_id,
            layout.num_embeddings[i],
            w["file"],
            w["offset"],
            engram_gather_threads(),
            0 if hot is not None and hot.enabled else row_buf,
            "on" if hot is not None and hot.enabled else "off",
        )
    return tables


def prefetch_engram_layer(model: Any, layer_id: int, hash_ids: np.ndarray) -> None:
    """Background-dequant hash ids for one Engram layer (overlap with other layers)."""
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        layers = getattr(model, "layers", [])
    for layer in layers:
        eng = getattr(layer, "engram", None)
        if eng is None or int(eng.layer_id) != int(layer_id):
            continue
        table = getattr(eng, "_mmap_table", None)
        if table is None:
            return
        idx = np.asarray(hash_ids, dtype=np.int64)
        flat = np.clip(idx.reshape(-1), 0, table.num_embeddings - 1)
        uniq = np.unique(flat)
        table.prefetch_unique(uniq)
        return


class NgramHashState:
    """Maps each position to hash ids of n-grams ending there (official NgramHashState)."""

    def __init__(self, args, layout: EngramLayout, token_map: np.ndarray):
        self.layout = layout
        vocab_size = int(getattr(args, "engram_compressed_vocab_size", 0) or 0)
        if vocab_size and int(token_map.max()) + 1 > vocab_size:
            logger.warning(
                "token_map max+1=%s > engram_compressed_vocab_size=%s",
                int(token_map.max()) + 1,
                vocab_size,
            )
        if vocab_size and int(token_map.max()) + 1 != vocab_size:
            # Official asserts exact equality; we warn but continue if close.
            computed = int(token_map.max()) + 1
            # unique count may be lower than max+1 if sparse; multipliers use vocab_size.
            if abs(computed - vocab_size) > 1:
                logger.warning(
                    "token_map max+1=%s != engram_compressed_vocab_size=%s "
                    "(hash multipliers use expected)",
                    computed,
                    vocab_size,
                )
        use_vocab = vocab_size or int(token_map.max()) + 1
        pad_token_id = int(
            getattr(args, "engram_pad_token_id", getattr(args, "engram_pad_id", 2))
        )
        self.pad_id = int(token_map[pad_token_id])
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]).astype(np.int64) for sizes in flat]
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, use_vocab
        )
        # primes: [n_layers, max_ngram-1, n_heads]
        self.primes = np.asarray(layout.primes, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=np.int64)  # [n_layers, n_hash_cols]
        self.multipliers = multipliers  # [n_layers, max_ngram]
        self.token_map = np.asarray(token_map, dtype=np.int64)
        self._cache: Optional[np.ndarray] = None  # [B, max_seq]
        self._cache_batch = 0

    @classmethod
    def from_engram_dir(cls, args, layout: EngramLayout, meta_dir: Optional[Path] = None):
        meta_dir = meta_dir or engram_dir()
        meta_path = meta_dir / "meta.json"
        token_path = meta_dir / "token_map.npy"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            tp = meta.get("token_map")
            if tp:
                token_path = Path(tp)
        token_map = np.load(token_path)
        return cls(args, layout, token_map)

    def _ensure_cache(self, batch: int, need_len: int):
        if (
            self._cache is None
            or self._cache.shape[0] < batch
            or self._cache.shape[1] < need_len
        ):
            new_len = max(need_len, 4096 if self._cache is None else self._cache.shape[1])
            new_batch = max(batch, 1 if self._cache is None else self._cache.shape[0])
            cache = np.full((new_batch, new_len), _DEAD, dtype=np.int64)
            if self._cache is not None:
                b = min(self._cache.shape[0], new_batch)
                l = min(self._cache.shape[1], new_len)
                cache[:b, :l] = self._cache[:b, :l]
            self._cache = cache

    def forward(
        self,
        input_ids: np.ndarray,
        start_pos: int,
        token_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return hash ids [B, L, n_engram_layers, n_hash_cols] as int64 numpy."""
        input_ids = np.asarray(input_ids, dtype=np.int64)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        batch, seqlen = input_ids.shape
        self._ensure_cache(batch, start_pos + seqlen)
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = np.where(token_mask, compressed, _DEAD)
        self._cache[:batch, start_pos : start_pos + seqlen] = compressed

        positions = np.broadcast_to(
            np.arange(start_pos, start_pos + seqlen, dtype=np.int64)[None, :],
            (batch, seqlen),
        )
        blocked = np.zeros_like(positions, dtype=bool)
        tokens = []
        for shift in range(self.layout.max_ngram_size):
            src_pos = np.maximum(positions - shift, 0)
            # gather along seq
            source = np.take_along_axis(self._cache[:batch], src_pos, axis=1)
            blocked = blocked | (positions < shift) | (source == _DEAD)
            tokens.append(np.where(blocked, self.pad_id, source))
        tokens_a = np.stack(tokens, axis=-1)  # [B, L, max_ngram]

        # products: [B, L, n_layers, max_ngram]
        products = tokens_a[:, :, None, :] * self.multipliers[None, None, :, :]
        rolling = products[..., 0]
        hashes = []
        for i in range(1, self.layout.max_ngram_size):
            rolling = np.bitwise_xor(rolling, products[..., i])
            # primes[:, i-1] -> [n_layers, n_heads]
            mod = rolling[..., None] % self.primes[:, i - 1][None, None, :, :]
            hashes.append(mod)
        out = np.concatenate(hashes, axis=-1) + self.offsets[None, None, :, :]
        return out.astype(np.int64)


class Engram(nn.Module):
    """Per-layer Engram: hash-id embedding + gated residual into HC streams."""

    def __init__(self, args, layer_id: int, layout: EngramLayout):
        super().__init__()
        self.layer_id = layer_id
        self.layout = layout
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = int(args.hidden_size)
        self.hc_mult = int(args.hc_mult)
        self.head_dim = layout.head_dim
        self.n_hash_cols = layout.n_hash_cols
        self.eps = float(getattr(args, "rms_norm_eps", 1e-20))
        self.clamp_value = 1e-6
        mode = engram_mode()
        full_rows = int(layout.num_embeddings[self.layer_hash_index])
        self.full_num_embeddings = full_rows
        self.stub = False
        self._mmap_table: Optional[MmapEngramTable] = None

        if engram_is_mmap():
            # No resident Embedding — tables come from MmapEngramTable.
            logger.info(
                "deepseek_v41 Engram layer %s mmap mode (rows=%s); "
                "tables via OMLX_DSV41_ENGRAM_DIR",
                layer_id,
                full_rows,
            )
        elif mode == "full":
            self.embed = nn.Embedding(full_rows, layout.head_dim)
            logger.warning(
                "deepseek_v41 Engram layer %s FULL resident table (%s rows) — "
                "expects ~100GB+ unified memory per layer",
                layer_id,
                full_rows,
            )
        else:
            # Should not normally construct Engram in stub (ids cleared), but keep safe.
            rows = 1024
            self.embed = nn.Embedding(rows, layout.head_dim)
            self.stub = True
            logger.info(
                "deepseek_v41 Engram layer %s stub table %s rows (full=%s)",
                layer_id,
                rows,
                full_rows,
            )

        self.wkv = nn.Linear(
            self.n_hash_cols * layout.head_dim,
            self.dim * (self.hc_mult + 1),
            bias=False,
        )
        # Official: Parameter ones; loaded from checkpoint (bf16).
        self.q_weight = mx.ones((self.hc_mult, self.dim))
        self.k_weight = mx.ones((self.hc_mult, self.dim))

    def bind_mmap_table(self, table: MmapEngramTable) -> None:
        self._mmap_table = table

    def _embed_ids(self, hash_ids) -> mx.array:
        if self._mmap_table is not None:
            # Prefer ndarray (model forward already hashed on CPU) to avoid mx↔np roundtrip.
            if isinstance(hash_ids, np.ndarray):
                ids_np = np.asarray(hash_ids, dtype=np.int64)
            else:
                mx.eval(hash_ids)
                ids_np = np.asarray(hash_ids, dtype=np.int64)
            return self._mmap_table.gather(ids_np)
        embed = getattr(self, "embed", None)
        assert embed is not None
        if isinstance(hash_ids, np.ndarray):
            hash_ids = mx.array(hash_ids)
        if self.stub:
            ids = hash_ids % embed.weight.shape[0]
        else:
            ids = hash_ids
        return embed(ids)

    def __call__(
        self,
        x: mx.array,
        hash_ids=None,
        token_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """x: [B,L,hc,D]. Without hash_ids, returns x unchanged."""
        if hash_ids is None:
            return x
        gathered = self._embed_ids(hash_ids)  # [B,L,n_hash_cols,head_dim]
        b, l = gathered.shape[0], gathered.shape[1]
        flat = gathered.reshape(b, l, -1)
        if flat.dtype != x.dtype:
            flat = flat.astype(x.dtype)
        kv = self.wkv(flat)
        key = kv[..., : self.hc_mult * self.dim].astype(mx.float32).reshape(
            b, l, self.hc_mult, self.dim
        )
        value = kv[..., self.hc_mult * self.dim :]
        weight = self.q_weight.astype(mx.float32) * self.k_weight.astype(mx.float32)
        h = x.astype(mx.float32)
        rstd = mx.rsqrt(mx.mean(mx.square(h), axis=-1) + self.eps) * mx.rsqrt(
            mx.mean(mx.square(key), axis=-1) + self.eps
        )
        inv_sqrt_d = self.dim**-0.5
        dot = mx.sum(h * weight * key, axis=-1) * rstd * inv_sqrt_d
        mag = mx.sqrt(mx.maximum(mx.abs(dot), self.clamp_value))
        signed = mx.where(dot < 0, -mag, mag)
        gate = mx.sigmoid(signed)
        if token_mask is not None:
            gate = mx.where(token_mask[..., None], gate, mx.zeros_like(gate))
        out = h + gate[..., None] * value.astype(mx.float32)[..., None, :]
        return out.astype(x.dtype)


def bind_mmap_tables(model: Any, layout: Optional[EngramLayout] = None) -> int:
    """Attach MmapEngramTable to each Engram module. Returns number bound."""
    if not engram_is_mmap():
        return 0
    layout = layout or getattr(getattr(model, "model", model), "engram_layout", None)
    if layout is None:
        return 0
    tables = load_mmap_tables_from_meta(layout)
    bound = 0
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        layers = getattr(model, "layers", [])
    for layer in layers:
        eng = getattr(layer, "engram", None)
        if eng is None:
            continue
        table = tables.get(eng.layer_id)
        if table is not None:
            eng.bind_mmap_table(table)
            bound += 1
    return bound


def bind_engram_hash(model: Any, args=None) -> Optional[NgramHashState]:
    """Create NgramHashState from ENGRAM_DIR token_map and attach to model.model."""
    inner = getattr(model, "model", model)
    layout = getattr(inner, "engram_layout", None)
    if layout is None:
        return None
    args = args or getattr(model, "args", None) or getattr(inner, "args", None)
    state = NgramHashState.from_engram_dir(args, layout)
    inner.engram_hash = state
    bind_mmap_tables(model, layout)
    return state
