# SPDX-License-Identifier: Apache-2.0
"""Tests for the MoE offload timeline (omlx/patches/moe_offload_timeline.py)."""

import json
import time

import pytest

from omlx.patches import moe_offload_timeline as timeline

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

needs_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

E, D, INTER, K, GROUP = 32, 64, 32, 2, 32
FIELDS = {"span", "layer", "count", "start_ns", "end_ns"}


@pytest.fixture
def recording(tmp_path, monkeypatch):
    """Turn the timeline on for one test, writing to a temporary file."""
    path = tmp_path / "timeline.jsonl"
    monkeypatch.setattr(timeline, "ON", True)
    monkeypatch.setattr(timeline, "PATH", str(path))
    monkeypatch.setattr(timeline, "_events", [])
    monkeypatch.setattr(timeline, "_dropped", 0)
    return path


def _spans(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestFormat:
    def test_every_span_writes_the_fields_the_analysis_reads(self):
        """The field names are a file format: analysis tools read them."""
        for span in timeline.SPANS:
            line = timeline.json_line(span, 47, 3, 10, 25)
            assert line.endswith("}\n")
            assert json.loads(line) == {
                "span": span,
                "layer": 47,
                "count": 3,
                "start_ns": 10,
                "end_ns": 25,
            }

    def test_off_records_nothing_and_writes_no_file(self, tmp_path, monkeypatch):
        path = tmp_path / "timeline.jsonl"
        monkeypatch.setattr(timeline, "ON", False)
        monkeypatch.setattr(timeline, "PATH", str(path))
        assert timeline.flush() is None
        assert not path.exists()

    def test_flush_appends_and_reports_each_batch_once(self, recording):
        timeline.record("read", 3, 4096, timeline.now())
        assert timeline.flush() == f"1 spans, 0 dropped -> {recording}"
        assert timeline.flush() is None  # nothing new: no second report
        timeline.record("ensure", 3, 1, timeline.now())
        timeline.flush()
        assert [s["span"] for s in _spans(recording)] == ["read", "ensure"]

    def test_spans_past_the_capacity_are_counted_not_stored(
        self, recording, monkeypatch
    ):
        monkeypatch.setattr(timeline, "_CAPACITY", 2)
        for _ in range(5):
            timeline.record("read", 0, 1, timeline.now())
        assert timeline.flush() == f"2 spans, 3 dropped -> {recording}"

    def test_unwritable_path_is_reported_not_raised(self, recording, monkeypatch):
        monkeypatch.setattr(timeline, "PATH", str(recording.parent / "missing" / "t"))
        timeline.record("read", 0, 1, timeline.now())
        assert timeline.flush().startswith("could not write ")


def _wrapped(tmp_path, n_layers=2):
    from omlx.patches.moe_expert_offload import apply_moe_expert_offload

    class _Experts(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.switch_glu = glu

    class _Layer(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.experts = _Experts(glu)

    class _MiniMoE(nn.Module):
        def __init__(self, glus):
            super().__init__()
            self.layers = [_Layer(g) for g in glus]

    glus, tensors = [], {}
    for i in range(n_layers):
        mx.random.seed(i)
        glu = SwitchGLU(D, INTER, E)
        nn.quantize(glu, group_size=GROUP, bits=4)
        glus.append(glu)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(glu, proj)
            for field in ("weight", "scales", "biases"):
                if lin.get(field) is not None:
                    name = f"layers.{i}.experts.switch_glu.{proj}.{field}"
                    tensors[name] = lin[field]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    model = _MiniMoE(glus)
    assert apply_moe_expert_offload(model, tmp_path, 0.25) == n_layers
    return [layer.experts.switch_glu for layer in model.layers]


@needs_mlx
class TestOffloadSpans:
    def test_decode_step_records_its_steps_in_order(self, tmp_path, recording):
        """A decode step with misses: the routing readback, the reads and the
        slot writes all lie inside the ensure, and the ensure inside the layer."""
        first, second = _wrapped(tmp_path)
        x = mx.random.normal((1, 1, D))
        idx = mx.array([[[3, 9]]], dtype=mx.int32)
        mx.eval(first(x, idx), second(x, idx))
        timeline.flush()
        spans = _spans(recording)
        assert all(set(s) == FIELDS and s["start_ns"] <= s["end_ns"] for s in spans)
        assert {s["layer"] for s in spans} == {0, 1}  # the wrapped layers, in order

        mine = [s for s in spans if s["layer"] == 0]
        by = {name: [s for s in mine if s["span"] == name] for name in timeline.SPANS}
        (layer,), (routing,), (ensure,) = by["layer"], by["routing"], by["ensure"]
        assert (layer["count"], routing["count"], ensure["count"]) == (K, K, 2)
        assert sum(s["count"] for s in by["read"]) == first.cache.fetched_bytes
        assert len(by["install"]) == 2  # one per missing expert
        for inner in [routing, ensure, *by["install"]]:
            assert layer["start_ns"] <= inner["start_ns"]
            assert inner["end_ns"] <= layer["end_ns"]
        for read in by["read"]:  # issued by the ensure, on the IO pool
            assert ensure["start_ns"] <= read["start_ns"]
            assert read["end_ns"] <= ensure["end_ns"]

    def test_a_hit_records_an_ensure_without_reads(self, tmp_path, recording):
        (glu,) = _wrapped(tmp_path, n_layers=1)
        x = mx.random.normal((1, 1, D))
        idx = mx.array([[[3, 9]]], dtype=mx.int32)
        mx.eval(glu(x, idx))
        timeline.flush()
        mx.eval(glu(x, idx))
        timeline.flush()
        again = _spans(recording)[-3:]
        assert [(s["span"], s["count"]) for s in again] == [
            ("routing", K),
            ("ensure", 0),
            ("layer", K),
        ]

    def test_over_capacity_prefill_records_one_eval_per_chunk(
        self, tmp_path, recording
    ):
        (glu,) = _wrapped(tmp_path, n_layers=1)
        tokens = E  # routes to every expert: four chunks at 8 slots
        idx = mx.arange(tokens * K).reshape(1, tokens, K) % E
        mx.eval(glu(mx.random.normal((1, tokens, D)), idx.astype(mx.int32)))
        timeline.flush()
        spans = _spans(recording)
        evals = [s for s in spans if s["span"] == "eval"]
        assert len(evals) == E // glu.cache.capacity
        assert sum(s["count"] for s in evals) == tokens * K
        assert sum(s["count"] for s in spans if s["span"] == "ensure") == E

    def test_slow_reads_record_the_keepalive_wait(
        self, tmp_path, recording, monkeypatch
    ):
        import omlx.patches.moe_expert_offload as meo

        (glu,) = _wrapped(tmp_path, n_layers=1)
        x = mx.random.normal((1, 1, D))
        mx.eval(glu(x, mx.array([[[0, 1]]], dtype=mx.int32)))
        timeline.flush()
        read = meo.CheckpointExpertStore.read

        def slow_read(plan):
            time.sleep(0.02)
            return read(plan)

        monkeypatch.setattr(meo.CheckpointExpertStore, "read", staticmethod(slow_read))
        mx.eval(glu(x, mx.array([[[0, 9]]], dtype=mx.int32)))  # 0 hits, 9 misses
        timeline.flush()
        waits = [s for s in _spans(recording) if s["span"] == "keepalive"]
        assert len(waits) == 1
        assert waits[0]["end_ns"] - waits[0]["start_ns"] >= 10_000_000

    def test_off_leaves_the_reader_untimed(self, tmp_path, monkeypatch):
        import omlx.patches.moe_expert_offload as meo

        monkeypatch.setattr(timeline, "ON", False)
        (glu,) = _wrapped(tmp_path, n_layers=1)
        assert glu.cache._reader() is meo.CheckpointExpertStore.read
