# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1: split at any layer up to the CED midpoint.

Besides ``(h, pre)``, layers after the cut read the shared state of the last
KV source before it: its compressed KV and index keys (rebuilt here from this
Mac's cache plus the chunk's new rows), the chunk's top-k ids and the
candidate blocks. Engram tables stay here: this Mac sends each chunk's rows
for the helper's Engram layers.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import mlx.core as mx
import numpy as np

from ...patches.deepseek_v41 import growth
from ...patches.deepseek_v41.loading import load
from ...patches.deepseek_v41.storage import decode_rows, dedupe_tuple_ids
from ..adapter import Arrays, Capture, HostArrays, StageAdapter, language_tensor

_SLOTS = 7
# Compressed KV and index keys grow with the prompt on KV-source layers; the
# helper sends only the rows a chunk adds.
_GROWING = (2, 3)
_HISTORY = 6


def _key(layer: int, slot: int) -> str:
    return f"{layer}.{slot}"


def _rows(item: Any, slot: int) -> int:
    value = item[slot]
    return 0 if value is None else int(value.shape[1])


_WARM = object()


class _HostRows:
    """A helper's Engram table: this Mac reads each chunk's rows and sends
    them with the chunk, so the helper holds no table and reads no SSD."""

    def __init__(self, dim: int):
        self.dim = dim
        self.rows: Any = None

    def put(self, inputs: HostArrays | None, layer: int) -> None:
        if inputs is None:
            self.rows = _WARM
            return
        name = f"e{layer}."
        if name + "ids" not in inputs:
            raise RuntimeError(f"No Engram rows for layer {layer}")
        bits, group_size, scaled = inputs[name + "q"][0].tolist()
        data = [inputs[f"{name}{n}"] for n in range(3) if f"{name}{n}" in inputs]
        self.rows = (inputs[name + "ids"][0], data, bits or None, group_size, scaled)

    def __call__(self, indices: mx.array) -> mx.array:
        host = np.asarray(indices).astype(np.int64)
        rows, self.rows = self.rows, None
        if rows is _WARM:
            return mx.zeros((*host.shape, self.dim), mx.bfloat16)
        if rows is None or not np.array_equal(rows[0], host):
            raise RuntimeError("Engram rows were sent for other ids")
        _, data, bits, group_size, scaled = rows
        return decode_rows(data, host.shape, bits, group_size, bool(scaled))


@contextmanager
def _stage_from(lm: Any, k: int):
    start = lm.start_idx
    lm.start_idx = k
    try:
        yield
    finally:
        lm.start_idx = start


class DeepseekV41Adapter(StageAdapter):
    model_types = ("deepseek_v41",)

    def layer_count(self, lm: Any) -> int:
        return lm._config.n_layers

    def max_split(self, lm: Any) -> int:
        # The CED midpoint holds the decoder replay input, so it stays here.
        return lm._config.n_layers // 2

    def layer_work(self, lm: Any) -> list[float]:
        c = lm._config
        work = [1.0] * c.n_layers
        if c.ced_prefill:
            # CED: past the midpoint only the window tail runs, and decoder
            # tails of chunks that another chunk follows are skipped.
            mid = c.n_layers // 2
            work = [1.0] * mid + [0.1] + [0.0] * (c.n_layers - mid - 1)
        # An Engram layer also projects its table rows; measured on M3 Ultra
        # and M5 Max it costs about two plain layers.
        for i in c.engram_layer_ids:
            work[i] *= 2
        return work

    def resident(self, tensor: str) -> bool:
        # Engram rows come from this Mac; draft heads and vision never load.
        return (
            ".engram.embed." not in tensor
            and ".mtp." not in tensor
            and language_tensor(tensor)
        )

    def keep_first(self, lm: Any, k: int) -> None:
        lm.start_idx, lm.end_idx = 0, k
        lm.layers = lm.layers[:k]
        for layer in lm.layers:
            if "engram" in layer:
                layer.engram.embed = _HostRows(lm._config.engram_head_dim)

    def load_first(self, path: str, k: int, a8_min_tokens: int | None = None) -> Any:
        model, _ = load(
            path,
            engram_ssd_offload=True,
            a8_prefill=a8_min_tokens is not None,
            a8_prefill_min_tokens=a8_min_tokens or 128,
            stage=lambda lm: self.keep_first(lm, k),
        )
        return model.language_model

    def prefetch(
        self, lm: Any, cache: list, current: Any, following: Any, start: int = 0
    ) -> None:
        with _stage_from(lm, start):
            lm.prefetch_engram(following, current, cache=cache)

    def job_feed(
        self, lm: Any, cache: list | None, tokens: np.ndarray, grid: list[int], k: int
    ) -> Callable[[int], HostArrays] | None:
        engram_ids = list(lm._config.engram_layer_ids)
        tables = {i: lm.layers[i].engram.embed for i in engram_ids if i < k}
        if not tables or lm._hasher is None:
            return None
        history = cache[0][_HISTORY] if cache else None
        if history is not None and history.size:
            history = np.asarray(history, dtype=np.int64)
        else:
            history = None
        tokens = np.asarray(tokens, dtype=np.int64).reshape(1, -1)
        starts = np.cumsum([0, *grid]).tolist()

        def feed(index: int) -> HostArrays:
            nonlocal history
            hashes, history = lm._hasher(
                tokens[:, starts[index] : starts[index + 1]], history
            )
            out: HostArrays = {}
            for i, table in tables.items():
                ids, _ = dedupe_tuple_ids(hashes[:, :, engram_ids.index(i)])
                data, (bits, group_size, scaled) = table.read_packed(ids)
                settings = np.array([bits or 0, group_size, scaled], dtype=np.int64)
                out[f"e{i}.ids"] = (ids, "int64")
                out[f"e{i}.q"] = (settings, "int64")
                for n, (raw, dtype) in enumerate(data):
                    out[f"e{i}.{n}"] = (raw, dtype)
            return out

        return feed

    def run_first(
        self,
        lm: Any,
        cache: list,
        tokens: mx.array,
        next_tokens: int,
        inputs: HostArrays | None = None,
        capture: Capture | None = None,
    ) -> tuple[Arrays, Arrays]:
        for i, layer in enumerate(lm.pipeline_layers):
            if "engram" in layer and isinstance(layer.engram.embed, _HostRows):
                layer.engram.embed.put(inputs, i)
        before = [{slot: _rows(item, slot) for slot in _GROWING} for item in cache]
        capture: dict[str, Any] = {}
        lm._omlx_prefill(
            tokens,
            cache=cache,
            next_prefill_tokens=next_tokens,
            skip_logits=True,
            _stage_capture=capture,
        )
        shared = capture["shared"]
        boundary = {"h": capture["h"], "pre": capture["pre"]}
        for name in ("idx", "candidates"):
            if name in shared:
                boundary[name] = shared[name]
        state = {}
        for i, item in enumerate(cache):
            for slot in range(_SLOTS):
                value = item[slot]
                if value is None:
                    continue
                if slot in _GROWING and item.compress_ratio:
                    value = value[:, before[i][slot] :]
                state[_key(i, slot)] = value
        mx.eval(list(boundary.values()), list(state.values()))
        return boundary, state

    def run_rest(
        self,
        prefill: Callable[..., Any],
        lm: Any,
        cache: list,
        tokens: mx.array,
        k: int,
        boundary: Arrays,
        state: Arrays,
        capture: Capture | None = None,
        **kwargs: Any,
    ) -> Any:
        shared = {
            name: boundary[name] for name in ("idx", "candidates") if name in boundary
        }
        sources = [i for i in lm._config.kv_source_layers if i < k]
        if sources:
            src = sources[-1]
            for slot, name in ((2, "kv"), (3, "index_k")):
                old, new = cache[src][slot], state[_key(src, slot)]
                shared[name] = new if old is None else mx.concatenate([old, new], 1)
        stage_input = {"h": boundary["h"], "pre": boundary["pre"], "shared": shared}
        # Engram hashes read the n-gram history from the stage's first layer;
        # the whole-model cache keeps it in layer 0.
        first = cache[k]
        history = first[_HISTORY]
        first[_HISTORY] = cache[0][_HISTORY]
        try:
            with _stage_from(lm, k):
                return prefill(
                    tokens, cache=cache[k:], _stage_input=stage_input, **kwargs
                )
        finally:
            first[_HISTORY] = (
                history if history is not None else mx.zeros((1, 0), mx.int64)
            )

    def apply_state(self, cache: list, k: int, state: Arrays) -> None:
        for i in range(k):
            item = cache[i]
            for slot in range(_SLOTS):
                value = state.get(_key(i, slot))
                if value is None:
                    continue
                if slot in _GROWING and item.compress_ratio:
                    previous = item[slot]
                    start = 0 if previous is None else int(previous.shape[1])
                    # Keep the in-place growth chain decode appends to.
                    item[slot] = growth.append(
                        item,
                        slot,
                        value if previous is None else previous,
                        value,
                        start,
                    )
                else:
                    item[slot] = value

    def prefix_state(self, cache: list, k: int) -> Arrays:
        state = {
            _key(i, slot): cache[i][slot]
            for i in range(k)
            for slot in range(_SLOTS)
            if cache[i][slot] is not None
        }
        mx.eval(list(state.values()))
        return state

    def load_prefix(self, cache: list, state: Arrays) -> None:
        for i, item in enumerate(cache):
            values = [state.get(_key(i, slot)) for slot in range(_SLOTS)]
            if any(value is not None for value in values):
                item.cache = values
