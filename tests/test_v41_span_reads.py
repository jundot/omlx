# SPDX-License-Identifier: Apache-2.0
"""V4.1 span reads — shared-backing-store demand path on synthetic checkpoints."""

import json
from pathlib import Path

import numpy as np
from streaming_fixtures import write_safetensors

from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore


def _make_checkpoint(root: Path, n_layers=2, n_experts=8, row_dim=4) -> dict:
    """Synthetic V4.1-style converted checkpoint."""
    rng = np.random.default_rng(0)
    tensors = {}
    expected = {}
    for layer in range(6, 6 + n_layers):
        for proj in ("w1", "w3", "w2"):
            for field in ("weight", "scales"):
                key = f"language_model.layers.{layer}.ffn.experts.{proj}.{field}"
                inner = row_dim if field == "weight" else 2
                arr = rng.integers(
                    0, 255, size=(n_experts, inner), dtype=np.uint8
                )
                tensors[key] = (arr, "U8")
                expected[key] = arr.tobytes()
    tensors["language_model.layers.0.self_attn.q.weight"] = (
        rng.integers(0, 255, size=(16, 16), dtype=np.uint8),
        "U8",
    )
    write_safetensors(root / "model-00001-of-00001.safetensors", tensors)
    index = {
        "weight_map": {
            k: "model-00001-of-00001.safetensors" for k in tensors
        }
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    (root / "config.json").write_text("{}")
    return expected


class _FakePlan:
    """Enough of ExpertOffloadPlan for _fetch_spans on a real
    ExpertBackingStore (the production storage path since V4-1)."""

    def __init__(self, root: Path):
        self._store = ExpertBackingStore(root)
        self._pool = None
        self.converted = {}  # marks the converted-checkpoint layout

    def fetch_span(self, prefix, proj, lo, hi):
        key = f"{prefix}.{proj}.weight"
        reader = self._store._reader_for_key(key, lo)
        rp = reader._rp_for(key)
        block = np.empty((hi - lo, rp.expert_bytes), dtype=np.uint8)
        assert self._store.read_expert_into(
            [(key, list(range(lo, hi)))], [block]
        )
        return {
            "weight": (
                block.view(rp.np_dtype).reshape(hi - lo, *rp.per_shape),
                reader.header[key]["dtype"],
            )
        }

    def _fetch_executor(self):
        from concurrent.futures import ThreadPoolExecutor

        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=4)
        return self._pool

    def close(self):
        self._store.close()


def _fake_slots(plan, prefix):
    from omlx.patches.deepseek_v41.moe_offload import _ExpertSlots
    from omlx.patches.expert_streaming.slot_cache import SlotArena

    slots = _ExpertSlots.__new__(_ExpertSlots)
    slots.plan = plan
    slots.prefix = prefix
    # Real arena instance (V4-2a): _fetch_spans only reads its staged
    # dict + counters; no projections needed since nothing grows.
    slots.arena = SlotArena(8, (), lambda p: {}, lambda p, v: None)
    slots.book = slots.arena.book
    slots.span_reads = 0
    slots.span_demand_rows = 0
    slots.span_phys_rows = 0
    slots.span_fallbacks = 0
    slots.staged_failures = 0
    return slots


def test_fetch_spans_identity_layout(tmp_path: Path, monkeypatch):
    """Identity layout: logical ids are physical rows — gap=1 merges only
    strictly-adjacent rows (zero overfetch); gap=0 never merges."""
    _make_checkpoint(tmp_path, n_layers=1, n_experts=8, row_dim=4)
    prefix = "language_model.layers.6.ffn.experts"
    plan = _FakePlan(tmp_path)

    # demand set {5, 6, 1}: sorted rows {1,5,6} -> spans {1},{5,6} at
    # the production default gap=1 (adjacent rows cost no dead bytes).
    fetch_list = [(5, 0, None), (6, 1, None), (1, 2, None)]
    slots = _fake_slots(plan, prefix)
    monkeypatch.setenv("OMLX_V41_SPAN_GAP", "1")
    via_gap1 = slots._fetch_spans(fetch_list)
    assert slots.span_reads == 2
    assert slots.span_demand_rows == 3
    assert slots.span_phys_rows == 3

    # gap=0 forces singleton spans — same payload, more commands.
    monkeypatch.setenv("OMLX_V41_SPAN_GAP", "0")
    slots0 = _fake_slots(plan, prefix)
    via_gap0 = slots0._fetch_spans(fetch_list)
    assert slots0.span_reads == 3
    for p1, p0 in zip(via_gap1, via_gap0):
        for proj in p1:
            np.testing.assert_array_equal(
                np.asarray(p1[proj]["weight"]),
                np.asarray(p0[proj]["weight"]),
            )

    # payload content: expert e serves row e of the source tensor.
    store = plan._store
    key = f"{prefix}.w1.weight"
    for e in (5, 6, 1):
        want = store.load_expert_slice(key, e)
        got = np.asarray(
            via_gap1[[f[0] for f in fetch_list].index(e)]["w1"]["weight"]
        )
        np.testing.assert_array_equal(got.reshape(-1)[:4], want.reshape(-1)[:4])
    plan.close()


def test_fetch_spans_falls_back_per_expert(tmp_path: Path, monkeypatch):
    """A failed span read must not kill the ensure: its rows degrade to
    the per-expert path and ``span_fallbacks`` records the event."""
    from omlx.patches.deepseek_v41.storage import decode_array

    _make_checkpoint(tmp_path, n_layers=1, n_experts=8, row_dim=4)
    prefix = "language_model.layers.6.ffn.experts"

    class _FailPlan(_FakePlan):
        def fetch_span(self, prefix, proj, lo, hi):
            raise OSError("span read failed")

        def fetch(self, prefix, proj, expert):
            key = f"{prefix}.{proj}.weight"
            row = self._store.load_expert_slice(key, int(expert))
            return {"weight": decode_array(row, "U8")}

    plan = _FailPlan(tmp_path)
    monkeypatch.setenv("OMLX_V41_SPAN_GAP", "1")
    slots = _fake_slots(plan, prefix)
    fetch_list = [(5, 0, None), (6, 1, None), (1, 2, None)]
    out = slots._fetch_spans(fetch_list)
    # Spans {1} and {5,6} both failed and fell back — one counter per
    # degraded span, payloads still aligned with fetch_list order.
    assert slots.span_fallbacks == 2
    assert slots.span_reads == 2
    assert slots.span_demand_rows == 3
    for (expert, _s, _v), payload in zip(fetch_list, out):
        want = plan._store.load_expert_slice(f"{prefix}.w1.weight", expert)
        np.testing.assert_array_equal(
            np.asarray(payload["w1"]["weight"]), want
        )
    plan.close()
