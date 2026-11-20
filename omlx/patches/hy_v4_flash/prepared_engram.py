# SPDX-License-Identifier: Apache-2.0
"""Opt-in Engram read-ahead for one forward, not a table or request cache.

As in SGLang FlashEngramHasher (ccb8d9cc) and vLLM FlashNgramHashState
(bb9b882a), plan all layer hashes from tokens/lookback before the decoder.
Use our Transformers c2ec8505 OOV/PAD and per-layer checkpoint-map semantics,
not the serving engines' sentinel, scheduler or CPU-pinned embedding formats.
Only original BF16 row bits enter MLX. The decoder still owns all FP8 and
hidden-dependent math; request history still belongs solely to Model.
"""

from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from threading import RLock

import mlx.core as mx
import numpy as np

from . import residency

# Hard aggregate *logical row payload* cap, not a process RSS guarantee. In
# addition to these rows, scratch is O(requested rows + ngram IDs) indices and
# at most one unique-row buffer / reconstructed-row copy per block. Never
# allocate from table size, other than the existing frozen token lookup maps.
MAX_PREPARED_BYTES = 16 << 20


def _compress(host_ids, lookup, pad):
    ids = np.where((host_ids >= 0) & (host_ids < len(lookup)), host_ids, pad)
    return lookup[ids]


def _rows(compressed, layer, multipliers):
    mixed = compressed * multipliers  # exact int64, layer-specific PCG64 draws
    key = mixed[..., 0]
    rows = np.empty((*key.shape, layer.num_heads), dtype=np.int64)
    start = 0
    for i, moduli in enumerate(layer._moduli, start=1):
        key = key ^ mixed[..., i]
        stop = start + len(moduli)
        np.remainder(key[..., None], moduli, out=rows[..., start:stop])
        start = stop
    rows += layer._offsets  # local to this block, ngram-major then head-minor
    return rows


def _publish(slot, tables, block, rows):
    try:
        bits = tables._read_rows(block, rows)
    except BaseException as exc:
        slot.set_exception(exc)
    else:
        slot.set_result(bits)


def _dispatch(pool, tables, indices, slots):
    """Hold one lease across workers. close/stats/gather cannot race their FDs.

    Publish the hot result independently: the consumer can submit first-layer
    work while the three late FDs are being read. No worker calls MLX. Drain
    even failed/unconsumed jobs *before* releasing a mapping or descriptor.
    """
    try:
        with tables._lock:
            tables._require_open()
            for b, rows in indices.items():
                spec = tables._files[b].spec
                if rows.size * spec.stride > residency._MAX_GATHER_BYTES:
                    raise ValueError(
                        "Engram gather exceeds per-call byte limit; chunk the prefill"
                    )
                if rows.min() < 0 or rows.max() >= spec.rows:
                    raise IndexError(f"Engram row outside block {b}")
            workers = []
            first = residency._BLOCKS[0]
            try:
                for b, rows in indices.items():
                    if b != first:
                        workers.append(pool.submit(_publish, slots[b], tables, b, rows))
                if first in indices:
                    _publish(slots[first], tables, first, indices[first])
            finally:
                wait(workers)
                errors = [s.exception() for s in slots.values() if s.done()]
                if any(isinstance(e, (OSError, ValueError)) for e in errors):
                    tables._release()  # same fail-closed policy as gather()
                for error in errors:
                    if error is not None:
                        raise error
    except BaseException as exc:
        for slot in slots.values():
            if not slot.done():  # e.g. provider closed before acquiring lease
                slot.set_exception(exc)
        raise


class PreparedEngrams:
    """Forward-local, single-use compact rows. No hidden states or table views."""

    def __init__(self, tokens, layers, provider, pending):
        self.tokens = tokens  # retain ORIGINAL IDs for the decoder's PAD mask
        self.shape = tokens.shape
        self._owners = {layer.block: (layer, layer.lookup_table) for layer in layers}
        self._provider = provider
        self._pending = pending.copy()
        self._prefix_submitted = False

    def take(self, layer, streams=None) -> mx.array:
        b = layer.block
        owner = self._owners.get(b)
        if (
            owner is None
            or layer is not owner[0]
            or layer.lookup_table is not owner[1]
            or layer._tables is not self._provider
            or b not in self._pending
        ):
            raise RuntimeError("no current prepared Engram rows for this layer")
        value = self._pending.pop(b)
        if isinstance(value, Future):
            # MLX is lazy. While actual preads are outstanding, submit the
            # already-built decoder prefix ONCE without waiting or copying any
            # hidden state to the host. Hashes never depend on this tensor.
            if (
                streams is not None
                and not self._prefix_submitted
                and (
                    not value.done()
                    or any(not job.done() for job in self._pending.values())
                )
            ):
                mx.async_eval(streams)
                self._prefix_submitted = True
            # Copies of selected row bits outlive the provider and this batch.
            value = mx.array(value.result()).view(mx.bfloat16)
        if (
            not isinstance(value, mx.array)
            or value.dtype != mx.bfloat16
            or value.shape != (*self.shape[:2], layer.num_heads, layer.head_dim)
        ):
            raise ValueError(
                "Engram provider must return one original BF16 vector per requested row"
            )
        return value

    def _expire(self):
        self._pending.clear()
        self._owners.clear()
        self._provider = self.tokens = None


class _Preparer:
    def __init__(self, model, max_bytes):
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_PREPARED_BYTES:
            raise ValueError(f"max_bytes must be in [1, {MAX_PREPARED_BYTES}]")
        self._layers = tuple(
            d.engram_layer for d in model.layers if d.engram_layer is not None
        )
        if not self._layers or len({layer.block for layer in self._layers}) != len(
            self._layers
        ):
            raise ValueError("expected distinct Engram blocks in decoder order")
        self._provider = self._layers[0]._tables
        if self._provider is None or any(
            layer._tables is not self._provider for layer in self._layers
        ):
            raise RuntimeError("bind one Engram table provider before installation")
        self._max_bytes = max_bytes
        self._gate = RLock()  # one in-flight batch per installation
        self._closed = self._active = False
        self._sources = ()
        self._refresh_maps()  # outside inference: group cold lookup evaluations
        # The dispatcher waits on the readers, so they must not share a pool.
        threaded = type(self._provider) is residency.EngramTables
        self._pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="hy4-engram-dispatch")
            if threaded
            else None
        )
        self._readers = (
            ThreadPoolExecutor(max_workers=3, thread_name_prefix="hy4-engram-read")
            if threaded
            else None
        )
        self._host_reads = self._peak_bytes = 0

    def _refresh_maps(self):
        sources = tuple(layer.lookup_table for layer in self._layers)
        if self._sources and all(a is b for a, b in zip(self._sources, sources)):
            return
        mx.eval(*sources)
        groups = []
        for layer in self._layers:
            state = layer._hash_inputs()  # baseline validation, loaded map, layer seed
            entry = (layer, state.multipliers)
            for lookup, pad, members in groups:
                if pad == layer.pad_id and np.array_equal(lookup, state.lookup):
                    members.append(entry)
                    break
            else:
                groups.append((state.lookup, layer.pad_id, [entry]))
        self._groups, self._sources = groups, sources

    def _plan(self, tokens):
        if (
            not isinstance(tokens, mx.array)
            or tokens.ndim != 3
            or min(tokens.shape) < 1
            or not mx.issubdtype(tokens.dtype, mx.integer)
            or any(layer.max_order != tokens.shape[-1] for layer in self._layers)
        ):
            raise ValueError(
                "expected nonempty integer [batch, tokens, max_ngram_size] Engram inputs"
            )
        if any(layer._tables is not self._provider for layer in self._layers):
            raise RuntimeError("Engram table binding changed; reinstall the preparer")
        positions = tokens.shape[0] * tokens.shape[1]
        sizes = [
            positions * layer.num_heads * layer.head_dim * 2 for layer in self._layers
        ]
        if sum(sizes) > self._max_bytes or max(sizes) > residency._MAX_GATHER_BYTES:
            raise ValueError(
                "prepared Engram rows exceed byte limit; chunk the prefill"
            )
        self._refresh_maps()  # re-group exactly if load_weights replaced a map
        host = np.asarray(tokens)  # the only per-forward MLX -> host Engram read
        self._host_reads += 1
        if host.dtype.kind == "i" and (host < 0).any():
            raise ValueError("negative token ID in Engram inputs")
        ids = host.astype(
            np.int64, copy=False
        )  # UINT64 overflow denotes OOV, not negative input
        indices = {}
        for lookup, pad, members in self._groups:
            compressed = _compress(ids, lookup, pad)
            for layer, multipliers in members:
                indices[layer.block] = _rows(compressed, layer, multipliers)
        self._peak_bytes = max(self._peak_bytes, sum(sizes))
        return indices

    @contextmanager
    def batch(self, tokens):
        with self._gate:
            if self._closed or self._active:
                raise RuntimeError("Engram preparer is closed or already active")
            self._active = True
            completion = prepared = None
            try:
                indices = self._plan(tokens)
                if self._pool is None:
                    # Unknown providers need not be thread-safe. Preserve their
                    # public API and decoder-order side effects on the MLX thread.
                    pending = {
                        layer.block: self._provider.gather(
                            layer.block, indices[layer.block]
                        )
                        for layer in self._layers
                    }
                else:
                    pending = {b: Future() for b in indices}
                    completion = self._pool.submit(
                        _dispatch, self._readers, self._provider, indices, pending
                    )
                prepared = PreparedEngrams(
                    tokens, self._layers, self._provider, pending
                )
                del pending, indices
                yield prepared
            finally:
                try:
                    if completion is not None:
                        completion.result()  # propagate failures even for a skipped block
                finally:
                    if prepared is not None:
                        prepared._expire()
                    self._active = False

    def stats(self):
        with self._gate:
            return dict(
                input_host_reads=self._host_reads,
                peak_batch_row_bytes=self._peak_bytes,
                max_batch_row_bytes=self._max_bytes,
                closed=self._closed,
            )

    def close(self):
        with self._gate:
            if self._active:
                raise RuntimeError("cannot close an active Engram batch")
            for pool in (self._pool, self._readers):
                if pool is not None:
                    pool.shutdown(wait=True)
            self._closed = True
            self._layers = self._groups = self._sources = ()
            self._pool = self._readers = self._provider = None


@contextmanager
def install_prepared_engram(model, *, max_bytes=MAX_PREPARED_BYTES):
    """Install on this model only, after weights and tables are bound.

    No MLX parameters, disk cache or global patches. Keep the provider open
    around the installation. Failure and normal exit both join the workers
    and restore the previous setting; provider ownership stays with the caller.
    """
    missing = object()
    previous = getattr(model, "_engram_preparer", missing)
    if previous is not missing and previous is not None:
        raise RuntimeError("Engram preparer already installed")
    preparer = _Preparer(model, max_bytes)
    try:
        model._engram_preparer = preparer
        yield preparer
    finally:
        preparer.close()
        if previous is missing:
            del model._engram_preparer
        else:
            model._engram_preparer = previous
