# SPDX-License-Identifier: Apache-2.0
"""Fail-closed validation of the model stage loaded by a cluster rank."""

import contextlib
import json
import os
import signal
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

import omlx.cluster.inference_worker as inference_worker
from omlx.cluster.inference_worker import (
    _bind_generation_thread_stream,
    _configure_distributed_mtp,
    _configure_indexer_decode_owner,
    _configure_tensor_shard_weights,
    _cross_thread_generation_stream,
    _execution_settings,
    _install_distributed_model_protocol,
    _planned_prefill_shape_warmup_tokens,
    _run_prefill_shape_warmup,
    _server_arguments,
    _trace_collectives,
    _validate_loaded_stage,
    _validate_measured_weight_bytes,
    _wait_for_serve_release,
    _watch_launcher_parent,
    _write_cancel_request,
    build_parser,
)
from omlx.cluster.planner import PipelineAssignment

GiB = 1024**3


def test_ds4_mtp_caps_decode_concurrency_until_batched_verification(
    tmp_path,
    monkeypatch,
):
    model = tmp_path / "ds4"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"deepseek_v4"}')
    args = SimpleNamespace(
        model=str(model),
        decode_concurrency=16,
        prompt_concurrency=8,
        pipeline_microbatch_size=8,
    )

    monkeypatch.delenv("OMLX_DSV4_MTP_DECODE_CONCURRENCY", raising=False)
    assert inference_worker._apply_distributed_mtp_decode_concurrency(
        args,
        mtp_enabled=True,
    ) == 1
    assert args.decode_concurrency == 1
    assert args.prompt_concurrency == 1
    assert args.pipeline_microbatch_size == 1


def test_non_mtp_decode_concurrency_is_unchanged(tmp_path):
    model = tmp_path / "ds4"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"deepseek_v4"}')
    args = SimpleNamespace(
        model=str(model),
        decode_concurrency=16,
        prompt_concurrency=8,
        pipeline_microbatch_size=8,
    )

    assert inference_worker._apply_distributed_mtp_decode_concurrency(
        args,
        mtp_enabled=False,
    ) == 16
    assert args.decode_concurrency == 16
    assert args.prompt_concurrency == 8
    assert args.pipeline_microbatch_size == 8


def test_collective_trace_records_order_and_restores_mlx_functions(
    tmp_path, monkeypatch
):
    class Distributed:
        def all_sum(self, value, **_kwargs):
            return ("sum", value)

        def all_gather(self, value, **_kwargs):
            return ("gather", value)

    distributed = Distributed()
    mx = SimpleNamespace(distributed=distributed)
    group = SimpleNamespace(rank=lambda: 1)
    value = SimpleNamespace(shape=(2, 4096), dtype="bfloat16")
    original_sum = distributed.all_sum.__func__
    monkeypatch.setenv("OMLX_CLUSTER_TRACE_COLLECTIVES", "1")

    with _trace_collectives(
        mx,
        group,
        state_dir=tmp_path,
        deployment_id="trace-test",
    ) as path:
        assert distributed.all_sum(value) == ("sum", value)
        assert distributed.all_gather(value) == ("gather", value)

    assert distributed.all_sum.__func__ is original_sum
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(item["seq"], item["op"]) for item in records] == [
        (1, "all_sum"),
        (2, "all_gather"),
    ]
    assert records[0]["rank"] == 1
    assert records[0]["shape"] == [2, 4096]
    assert records[0]["dtype"] == "bfloat16"
    assert "test_collective_trace_records_order" in records[0]["caller"]


def test_worker_waits_for_matching_supervisor_serve_release(tmp_path):
    deployment_id = "cluster-test"
    plan_hash = "a" * 64
    marker = tmp_path / f"{deployment_id}-serve.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_id": deployment_id,
                "plan_hash": plan_hash,
                "world_size": 2,
            }
        )
    )

    _wait_for_serve_release(tmp_path, deployment_id, plan_hash, 2, timeout=0)


def test_worker_rejects_stale_supervisor_serve_release(tmp_path):
    deployment_id = "cluster-test"
    marker = tmp_path / f"{deployment_id}-serve.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_id": deployment_id,
                "plan_hash": "old",
                "world_size": 2,
            }
        )
    )

    with pytest.raises(TimeoutError, match="did not release"):
        _wait_for_serve_release(
            tmp_path,
            deployment_id,
            "new",
            2,
            timeout=0,
        )


def test_prefill_shape_warmup_requires_qualified_tp_metadata():
    qualified = {
        "deepseek_v4_adaptive_prefill": {
            "active": True,
            "shape_warmup_tokens": 1024,
        }
    }

    assert (
        _planned_prefill_shape_warmup_tokens(
            qualified,
            tensor_parallel_size=2,
        )
        == 1024
    )
    assert (
        _planned_prefill_shape_warmup_tokens(
            qualified,
            tensor_parallel_size=1,
        )
        == 0
    )
    assert (
        _planned_prefill_shape_warmup_tokens(
            {"deepseek_v4_adaptive_prefill": {"active": False}},
            tensor_parallel_size=2,
        )
        == 0
    )
    qualified["deepseek_v4_adaptive_prefill"]["shape_warmup_tokens"] = 8192
    assert (
        _planned_prefill_shape_warmup_tokens(
            qualified,
            tensor_parallel_size=2,
        )
        == 0
    )


def test_prefill_shape_warmup_evaluates_cache_then_releases_scratch():
    calls = []

    class FakeMX:
        int32 = "int32"

        @staticmethod
        def zeros(shape, *, dtype):
            calls.append(("zeros", shape, dtype))
            return "token-batch"

        @staticmethod
        def eval(*values):
            calls.append(("eval", values))

        @staticmethod
        def synchronize():
            calls.append(("synchronize",))

        @staticmethod
        def clear_cache():
            calls.append(("clear_cache",))

    class Model:
        def __call__(self, tokens, **kwargs):
            calls.append(("model", tokens, kwargs))
            return "hidden-output"

    caches = [SimpleNamespace(state="cache-0"), SimpleNamespace(state="cache-1")]

    def cache_factory(model, *, max_kv_size):
        calls.append(("cache_factory", model, max_kv_size))
        return caches

    ticks = iter((10.0, 10.25))
    model = Model()
    report = _run_prefill_shape_warmup(
        FakeMX,
        model,
        tokens=1024,
        max_kv_size=32768,
        cache_factory=cache_factory,
        clock=lambda: next(ticks),
    )

    assert report == {
        "active": True,
        "tokens": 1024,
        "elapsed_seconds": 0.25,
    }
    assert ("cache_factory", model, 32768) in calls
    assert ("zeros", (1, 1024), "int32") in calls
    assert (
        "model",
        "token-batch",
        {"cache": caches, "skip_lm_head": True},
    ) in calls
    assert ("eval", ("hidden-output", ["cache-0", "cache-1"])) in calls
    assert calls[-1] == ("clear_cache",)


def test_distributed_mtp_pins_one_signed_depth_on_every_rank(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4"})
    )
    monkeypatch.delenv("OMLX_MTP_FIXED_DEPTH", raising=False)

    fixed = _configure_distributed_mtp(
        tmp_path,
        enabled=True,
        depth=5,
        tensor_parallel_size=2,
    )

    assert fixed == 5
    assert os.environ["OMLX_MTP_FIXED_DEPTH"] == "5"


def test_distributed_mtp_can_qualify_coordinated_adaptive_depth(
    tmp_path, monkeypatch
):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4"})
    )
    monkeypatch.setenv("OMLX_MTP_FIXED_DEPTH", "5")
    monkeypatch.setenv("OMLX_MTP_DISTRIBUTED_ADAPTIVE_DEPTH", "1")

    fixed = _configure_distributed_mtp(
        tmp_path,
        enabled=True,
        depth=5,
        tensor_parallel_size=2,
    )

    assert fixed is None
    assert "OMLX_MTP_FIXED_DEPTH" not in os.environ


def test_distributed_mtp_can_qualify_lockstep_acceptance_depth(
    tmp_path, monkeypatch
):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4"})
    )
    monkeypatch.setenv("OMLX_MTP_FIXED_DEPTH", "5")
    monkeypatch.setenv("OMLX_MTP_DISTRIBUTED_LOCKSTEP_DEPTH", "1")

    fixed = _configure_distributed_mtp(
        tmp_path,
        enabled=True,
        depth=5,
        tensor_parallel_size=2,
    )

    assert fixed is None
    assert "OMLX_MTP_FIXED_DEPTH" not in os.environ


def test_distributed_mtp_refuses_unvalidated_model_family(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))

    with pytest.raises(RuntimeError, match="not validated"):
        _configure_distributed_mtp(
            tmp_path,
            enabled=True,
            depth=3,
            tensor_parallel_size=2,
        )


def test_signed_tensor_shard_weights_replace_a_stale_environment(monkeypatch, caplog):
    assignments = tuple(
        PipelineAssignment(
            node_id=f"node-{rank}",
            rank=rank,
            start_layer=0,
            end_layer=4,
            layer_weight_bytes=50 if rank == 0 else 30,
            fixed_weight_bytes=10,
            reserve_bytes=10,
            capacity_bytes=100,
            tensor_parallel_rank=rank,
            tensor_parallel_size=2,
            tensor_parallel_shard_weight=5 if rank == 0 else 3,
        )
        for rank in range(2)
    )
    monkeypatch.setenv("OMLX_TP_SHARD_WEIGHTS", "1,1")
    monkeypatch.setenv("OMLX_TP_NON_MOE_SHARD_WEIGHTS", "3,5")
    monkeypatch.setenv("OMLX_TP_MOE_SHARD_WEIGHTS", "4,4")

    assert _configure_tensor_shard_weights(
        assignments,
        rank=1,
        tensor_parallel_size=2,
    ) == (5, 3)
    assert os.environ["OMLX_TP_SHARD_WEIGHTS"] == "5,3"
    assert "OMLX_TP_NON_MOE_SHARD_WEIGHTS" not in os.environ
    assert "OMLX_TP_MOE_SHARD_WEIGHTS" not in os.environ
    assert "unsigned tensor-component shard overrides" in caplog.text
    assert "3,5" not in caplog.text
    assert "4,4" not in caplog.text


def test_non_tp_worker_clears_unsigned_component_shard_overrides(monkeypatch):
    monkeypatch.setenv("OMLX_TP_SHARD_WEIGHTS", "3,5")
    monkeypatch.setenv("OMLX_TP_NON_MOE_SHARD_WEIGHTS", "3,5")
    monkeypatch.setenv("OMLX_TP_MOE_SHARD_WEIGHTS", "4,4")

    assert _configure_tensor_shard_weights(
        (),
        rank=0,
        tensor_parallel_size=1,
    ) == (1,)
    assert "OMLX_TP_SHARD_WEIGHTS" not in os.environ
    assert "OMLX_TP_NON_MOE_SHARD_WEIGHTS" not in os.environ
    assert "OMLX_TP_MOE_SHARD_WEIGHTS" not in os.environ


def test_decode_indexer_owner_uses_the_fastest_measured_tp_rank(monkeypatch):
    profiles = [
        SimpleNamespace(decode_weight_bytes_per_second=15e9),
        SimpleNamespace(decode_weight_bytes_per_second=25e9),
    ]
    monkeypatch.setenv("OMLX_DSV4_INDEXER_DECODE_OWNER_RANK", "auto")

    assert _configure_indexer_decode_owner(
        profiles,
        rank=0,
        tensor_parallel_size=2,
    ) == 1
    assert os.environ["OMLX_DSV4_INDEXER_DECODE_OWNER_RANK"] == "1"


def test_decode_indexer_owner_has_an_operator_rollback(monkeypatch):
    monkeypatch.setenv("OMLX_DSV4_INDEXER_DECODE_OWNER_RANK", "off")

    assert _configure_indexer_decode_owner(
        (),
        rank=0,
        tensor_parallel_size=2,
    ) is None
    assert os.environ["OMLX_DSV4_INDEXER_DECODE_OWNER_RANK"] == "off"


def test_distributed_minimax_protocol_replaces_generic_tool_and_thinking_markers(
    tmp_path,
):
    model = tmp_path / "MiniMax-M3-4bit"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "minimax_m3_vl"}))

    class Tokenizer:
        @staticmethod
        def _tool_parser(*_args):
            return None

        _tool_call_start = "<tool_call>"
        _tool_call_end = "</tool_call>"
        _tool_call_start_tokens = (52,)
        _tool_call_end_tokens = (53,)
        _think_start = "<think>"
        _think_end = "</think>"
        _think_start_tokens = (54,)
        _think_end_tokens = (55,)

        @staticmethod
        def encode(text, add_special_tokens=False):
            assert add_special_tokens is False
            return {
                "]<]minimax[>[<tool_call>": [58, 52],
                "]<]minimax[>[</tool_call>": [58, 53],
                "<mm:think>": [59],
                "</mm:think>": [60],
            }[text]

    tokenizer = Tokenizer()
    assert _install_distributed_model_protocol(tokenizer, model) == "minimax_m3"
    assert tokenizer._tool_call_start == "]<]minimax[>[<tool_call>"
    assert tokenizer._tool_call_end == "]<]minimax[>[</tool_call>"
    assert tokenizer._tool_call_start_tokens == (58, 52)
    assert tokenizer._tool_call_end_tokens == (58, 53)
    assert tokenizer._think_start == "<mm:think>"
    assert tokenizer._think_end == "</mm:think>"
    assert tokenizer._think_start_tokens == (59,)
    assert tokenizer._think_end_tokens == (60,)

    ns = "]<]minimax[>["
    parsed = tokenizer._tool_parser(
        (f'{ns}<invoke name="get_weather">{ns}<city>Paris{ns}</city>{ns}</invoke>'),
        [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
    )
    assert parsed == {"name": "get_weather", "arguments": {"city": "Paris"}}


def test_distributed_protocol_leaves_other_model_families_untouched(tmp_path):
    model = tmp_path / "llama"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}))
    tokenizer = SimpleNamespace(_tool_call_start="<tool_call>")

    assert _install_distributed_model_protocol(tokenizer, model) == ""
    assert tokenizer._tool_call_start == "<tool_call>"


def test_eager_load_graph_is_visible_to_mlx_lm_generation_thread():
    """Regression for ``no Stream(gpu, N) in current thread`` on first prompt."""

    import mlx.core as mx

    previous = mx.default_stream(mx.default_device())
    stream = _cross_thread_generation_stream(mx)
    lazy_value = mx.arange(4) + 1
    observed: list[list[int]] = []

    class FakeResponseGenerator:
        def _generate(self):
            mx.eval(lazy_value)
            observed.append(lazy_value.tolist())

    original = FakeResponseGenerator._generate
    try:
        with _bind_generation_thread_stream(
            FakeResponseGenerator,
            mx,
            stream,
        ):
            worker = threading.Thread(target=FakeResponseGenerator()._generate)
            worker.start()
            worker.join()
    finally:
        mx.set_default_stream(previous)

    assert observed == [[1, 2, 3, 4]]
    assert FakeResponseGenerator._generate is original


def test_generation_thread_failure_forces_rank_exit(monkeypatch):
    import mlx.core as mx

    exits = []

    class FakeResponseGenerator:
        def _generate(self):
            raise RuntimeError("token collective corrupted")

    monkeypatch.setattr(
        inference_worker,
        "_fatal_generation_thread_exit",
        lambda: exits.append(inference_worker._FATAL_GENERATION_EXIT_CODE),
    )
    original = FakeResponseGenerator._generate
    with _bind_generation_thread_stream(
        FakeResponseGenerator,
        mx,
        mx.new_thread_unsafe_stream(mx.default_device()),
    ):
        with pytest.raises(RuntimeError, match="token collective corrupted"):
            FakeResponseGenerator()._generate()

    assert exits == [70]
    assert FakeResponseGenerator._generate is original


def _assignment() -> PipelineAssignment:
    return PipelineAssignment(
        node_id="studio",
        rank=0,
        start_layer=2,
        end_layer=5,
        layer_weight_bytes=30,
        fixed_weight_bytes=10,
        reserve_bytes=10,
        capacity_bytes=100,
    )


def _model(*, start=2, end=5, layers=None):
    if layers is None:
        layers = [None, None, object(), object(), object()]
    return SimpleNamespace(
        model=SimpleNamespace(
            start_idx=start,
            end_idx=end,
            layers=layers,
        )
    )


def test_loaded_stage_matches_approved_unequal_range():
    _validate_loaded_stage(_model(), _assignment())


def test_loaded_stage_rejects_model_specific_equal_split():
    with pytest.raises(RuntimeError, match="does not match the approved plan"):
        _validate_loaded_stage(_model(start=3), _assignment())


def test_loaded_stage_rejects_weights_outside_local_range():
    layers = [object(), None, object(), object(), object()]

    with pytest.raises(RuntimeError, match="weights before"):
        _validate_loaded_stage(_model(layers=layers), _assignment())


def test_loaded_stage_rejects_missing_local_layer():
    layers = [None, None, object(), None, object()]

    with pytest.raises(RuntimeError, match="missing weights"):
        _validate_loaded_stage(_model(layers=layers), _assignment())


def test_pure_tensor_stage_accepts_an_unset_pipeline_end_index():
    assignment = PipelineAssignment(
        node_id="studio",
        rank=0,
        start_layer=0,
        end_layer=3,
        layer_weight_bytes=30,
        fixed_weight_bytes=10,
        reserve_bytes=10,
        capacity_bytes=100,
        tensor_parallel_rank=0,
        tensor_parallel_size=2,
    )
    model = _model(start=0, end=None, layers=[object(), object(), object()])

    _validate_loaded_stage(model, assignment)


def test_launcher_watchdog_records_reason_and_exits_reparented_rank():
    updates: list[tuple[str, dict]] = []
    events: list[dict] = []
    exit_codes: list[int] = []
    releases: list[str] = []
    marker = SimpleNamespace(
        update=lambda phase, **extra: updates.append((phase, extra))
    )

    _watch_launcher_parent(
        42,
        marker,
        get_parent_pid=lambda: 1,
        wait=lambda _seconds: None,
        exit_process=exit_codes.append,
        emit_event=events.append,
        release_memory=releases.append,
    )

    assert updates == [
        (
            "launcher_lost",
            {
                "error": (
                    "rank launcher parent changed from 42 to 1; "
                    "the rank cannot safely continue"
                )
            },
        )
    ]
    assert events[0]["type"] == "launcher_lost"
    # The Metal release runs before the exit: os._exit skips every
    # finally/atexit handler, so a skipped release orphans wired memory.
    assert len(releases) == 1
    assert exit_codes == [1]


def test_worker_execution_contract_reaches_mlx_lm_and_runtime_optimizations():
    args = build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--backend",
            "ring",
            "--port",
            "32000",
            "--deployment-id",
            "test",
            "--plan-hash",
            "a" * 64,
            "--plan",
            "encoded",
            "--execution-profile",
            "throughput",
            "--auto-tune",
            "--decode-concurrency",
            "16",
            "--prompt-concurrency",
            "8",
            "--prefill-step-size",
            "4096",
            "--prompt-cache-size",
            "16",
            "--prompt-cache-bytes",
            "1048576",
            "--max-kv-size",
            "32768",
            "--pipeline-microbatch-size",
            "8",
            "--cache-affinity",
            "--sampling-rank-only",
            "--async-overlap",
            "--ring-connections-per-ip",
            "4",
            "--tuning-reason",
            "measured",
        ]
    )

    execution = _execution_settings(args)
    server = _server_arguments(args)

    assert execution.profile == "throughput"
    assert execution.pipeline_microbatch_size == 8
    assert execution.sampling_rank_only is True
    assert execution.ring_connections_per_ip == 4
    assert server.decode_concurrency == 16
    assert server.prefill_step_size == 4096
    assert server.max_kv_size == 32768
    assert server.pipeline is True

    assert _server_arguments(args, tensor_parallel_size=2).pipeline is False


def test_worker_execution_contract_parses_deepseek_ane_settings():
    args = build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--backend",
            "ring",
            "--port",
            "32000",
            "--deployment-id",
            "test",
            "--plan-hash",
            "a" * 64,
            "--plan",
            "encoded",
            "--prefill-step-size",
            "4096",
            "--deepseek-ane-prefill",
            "--deepseek-ane-sequence-length",
            "4096",
            "--deepseek-ane-down-fraction",
            "0.5",
            "--no-deepseek-ane-wo-a",
            "--no-deepseek-ane-cpu",
        ]
    )

    execution = _execution_settings(args)
    ane = execution.deepseek_ane_prefill
    assert ane.enabled is True
    assert ane.sequence_length == execution.prefill_step_size == 4096
    assert ane.down_fraction == 0.5
    assert ane.wo_a_enabled is False
    assert ane.cpu_enabled is False


def test_worker_enables_deepseek_ane_after_tensor_sharding(monkeypatch):
    from omlx.cluster.performance import DeepseekAnePrefillSettings
    from omlx.patches.deepseek_v4 import ane_prefill

    calls = []
    monkeypatch.setattr(
        ane_prefill,
        "enable_deepseek_v4_ane_prefill",
        lambda model, **kwargs: calls.append((model, kwargs)) or 17,
    )
    model = SimpleNamespace(args=SimpleNamespace(model_type="deepseek_v4"))
    settings = DeepseekAnePrefillSettings(
        enabled=True,
        sequence_length=4096,
        down_fraction=0.5,
        wo_a_enabled=False,
        cpu_enabled=False,
    )

    count = inference_worker._enable_distributed_deepseek_ane_prefill(
        model,
        settings,
    )

    assert count == 17
    assert calls[0][0] is model
    assert calls[0][1]["sequence_length"] == 4096
    assert calls[0][1]["wo_a_enabled"] is False
    assert calls[0][1]["cpu_fraction"] == 0.0


# ---------------------------------------------------------------------------
# Tensor-parallel sharding (B3)
#
# Sharding defers to the model's own shard(). A hand-rolled per-projection
# version died in q_norm on the first real two-Mac run, because Qwen3.5 shards a
# gated-delta-net conv1d and repeats KV heads — architecture-specific work a
# generic reimplementation cannot replicate.
# ---------------------------------------------------------------------------


def test_measured_weight_bytes_sums_resident_parameters():
    """Publishes what a rank actually holds, to compare against the plan."""

    class FakeArray:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    model = SimpleNamespace(
        parameters=lambda: {
            "layers": [{"q_proj": FakeArray(100)}, {"o_proj": FakeArray(50)}],
            "embed": FakeArray(25),
        }
    )
    assert inference_worker._measured_weight_bytes(model) == 175


def test_measured_weight_bytes_degrades_to_none():
    def explode():
        raise RuntimeError("no parameters")

    assert (
        inference_worker._measured_weight_bytes(SimpleNamespace(parameters=explode))
        is None
    )
    assert inference_worker._measured_weight_bytes(object()) is None


def test_measured_weight_guard_accepts_exact_tolerated_and_unavailable_values():
    assignment = _assignment()
    approved = assignment.fixed_weight_bytes + assignment.layer_weight_bytes
    tolerance = inference_worker._MEASURED_WEIGHT_ABSOLUTE_TOLERANCE_BYTES

    _validate_measured_weight_bytes(None, assignment)
    _validate_measured_weight_bytes(approved, assignment)
    _validate_measured_weight_bytes(approved + tolerance, assignment)


def test_measured_weight_guard_rejects_an_oversized_loaded_shard():
    assignment = _assignment()
    approved = assignment.fixed_weight_bytes + assignment.layer_weight_bytes
    tolerance = inference_worker._MEASURED_WEIGHT_ABSOLUTE_TOLERANCE_BYTES

    with pytest.raises(RuntimeError, match="more parameter memory.*approved shard"):
        _validate_measured_weight_bytes(approved + tolerance + 1, assignment)


# ---------------------------------------------------------------------------
# The marker recorded what a rank was planned to hold and never what it built.
# After a crash that made a stage pin that worked indistinguishable from one
# mlx-lm had silently replaced with an even split.
# ---------------------------------------------------------------------------


def test_the_stage_recorded_is_the_one_the_model_actually_built():
    assert inference_worker._loaded_stage(_model()) == {
        "loaded_start_layer": 2,
        "loaded_end_layer": 5,
        "loaded_layer_count": 3,
    }


def test_an_even_split_the_plan_never_asked_for_is_visible_afterwards():
    """The failure this exists for: 14 planned layers, 30 loaded."""

    split_evenly = _model(
        start=3,
        end=6,
        layers=[None, None, None, object(), object(), object()],
    )

    recorded = inference_worker._loaded_stage(split_evenly)

    assert recorded["loaded_start_layer"] == 3, "the plan asked for layer 2"
    assert recorded["loaded_end_layer"] == 6
    assert recorded["loaded_layer_count"] == 3


def test_a_model_that_cannot_be_read_records_nothing_rather_than_the_plan():
    assert inference_worker._loaded_stage(object()) == {
        "loaded_start_layer": None,
        "loaded_end_layer": None,
        "loaded_layer_count": None,
    }


def test_the_loaded_stage_survives_into_the_marker_a_reader_sees(tmp_path):
    marker = inference_worker.RuntimeMarker(
        state_dir=str(tmp_path),
        deployment_id="d",
        rank=0,
        world_size=2,
        model="org/model",
        backend="ring",
        plan_hash="a" * 64,
    )

    marker.update("loading", **inference_worker._loaded_stage(_model()))
    marker.update("ready", start_layer=2, end_layer=5)

    written = json.loads(marker.path.read_text())
    assert written["phase"] == "ready"
    assert written["loaded_start_layer"] == 2
    assert written["loaded_end_layer"] == 5
    assert written["loaded_layer_count"] == 3


# ---------------------------------------------------------------------------
# run_worker — the rank main path.
#
# Every defect below had a green unit test. They passed because they called the
# guarded function directly with an argument the launch path never supplies:
# ``check_rank_fits(role="workstation")`` when nothing passes a role,
# ``effective_stage`` after ``clear_assigned_stage()`` when the launch path
# guarantees the pin is set. ``run_worker`` itself had no test at all, so the
# order the guards run in — which is the entire bug in two of them — was never
# observed by anything.
#
# These drive ``run_worker`` end to end with fakes at the process boundaries
# (MLX's distributed init, mlx-lm's ModelProvider and server loop, the Metal
# wired limit) and let everything oMLX owns run for real: the argument parser,
# the runtime marker, the admission arithmetic, the stage pin and its guard.
# ---------------------------------------------------------------------------


def _uneven_plan(*, role: str = "workstation", tier: str = "balanced"):
    """The incident in ``_guard_effective_stage``'s own docstring.

    60 layers over two Macs. The plan gives the MacBook 14 of them to keep it
    usable; ``PipelineMixin``'s split, which is what actually runs unless the
    pin takes effect, would give it 30 — 122 GiB on a Mac that can admit 96.8.
    """

    return (
        PipelineAssignment(
            node_id="mbp",
            rank=0,
            start_layer=46,
            end_layer=60,
            layer_weight_bytes=14 * 4 * GiB,
            fixed_weight_bytes=2 * GiB,
            reserve_bytes=32 * GiB,
            capacity_bytes=107 * GiB,
            role=role,
            memory_guard_tier=tier,
        ),
        PipelineAssignment(
            node_id="studio",
            rank=1,
            start_layer=0,
            end_layer=46,
            layer_weight_bytes=46 * 4 * GiB,
            fixed_weight_bytes=2 * GiB,
            reserve_bytes=48 * GiB,
            capacity_bytes=460 * GiB,
            role="headless",
            memory_guard_tier=tier,
        ),
    )


def _even_plan():
    """A plan that matches the loader's own split, so the stage guard stays put."""

    return (
        PipelineAssignment(
            node_id="mbp",
            rank=0,
            start_layer=30,
            end_layer=60,
            layer_weight_bytes=30 * GiB,
            fixed_weight_bytes=GiB,
            reserve_bytes=8 * GiB,
            capacity_bytes=400 * GiB,
            role="headless",
        ),
        PipelineAssignment(
            node_id="studio",
            rank=1,
            start_layer=0,
            end_layer=30,
            layer_weight_bytes=30 * GiB,
            fixed_weight_bytes=GiB,
            reserve_bytes=8 * GiB,
            capacity_bytes=400 * GiB,
            role="headless",
        ),
    )


def _tensor_plan():
    """Two ranks holding the same full layer span as one TP stage."""

    return tuple(
        PipelineAssignment(
            node_id=node_id,
            rank=rank,
            start_layer=0,
            end_layer=60,
            layer_weight_bytes=30 * GiB,
            fixed_weight_bytes=GiB,
            reserve_bytes=8 * GiB,
            capacity_bytes=400 * GiB,
            role="headless",
            tensor_parallel_rank=rank,
            tensor_parallel_size=2,
        )
        for rank, node_id in enumerate(("mbp", "studio"))
    )


class _Group:
    def __init__(self, rank: int, size: int) -> None:
        self._rank, self._size = rank, size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _staged_model(assignment) -> Any:
    """A model shaped exactly as ``_validate_loaded_stage`` demands."""

    layers = [None] * assignment.start_layer + [
        object() for _ in range(assignment.layer_count)
    ]
    return SimpleNamespace(
        model=SimpleNamespace(
            start_idx=assignment.start_layer,
            end_idx=assignment.end_layer,
            layers=layers,
        )
    )


def _worker_argv(tmp_path, *, extra=()):
    return [
        "--model",
        "org/model",
        "--backend",
        "ring",
        "--port",
        "32000",
        "--deployment-id",
        "mini-abc123",
        "--plan-hash",
        "a" * 64,
        "--plan",
        "encoded-plan",
        "--peer-hosts",
        "127.0.0.1,studio.local",
        "--state-dir",
        str(tmp_path),
        *extra,
    ]


def _run_rank(
    monkeypatch,
    tmp_path,
    *,
    rank: int = 0,
    assignments=None,
    argv_extra=(),
    ceiling: int = 107 * GiB,
    wired_result=(0, None),
    assignment_honored: bool = False,
    optimizations: dict[str, Any] | None = None,
    tensor_parallel_size: int = 1,
):
    """Drive ``run_worker`` through its real argv, recording what it did.

    Returns a record of the observable decisions: the order guards ran in, the
    peer map the watchdog was handed, the arguments the admission checks
    received, and the marker as it stood while the rank was serving.
    """

    import mlx.core as mx
    import mlx_lm.server as mlx_server

    import omlx._torch_stub as torch_stub
    import omlx.patches.mlx_lm_pipeline_index as pipeline_index
    import omlx.process_memory_enforcer as enforcer
    import omlx.utils.model_loading as model_loading
    from omlx.patches.minimax_m3_mlx_lm import pipeline_patch

    assignments = assignments or _uneven_plan()
    assignment = assignments[rank]
    record: dict[str, Any] = {
        "order": [],
        "guard_calls": [],
        "stage_guard_calls": [],
        "wired_calls": [],
        "raw_set_wired_limit": [],
        "watchdog_hosts": None,
        "build_guard_kwargs": None,
        "marker_while_serving": None,
        "pin_at_load": None,
        "events": [],
    }

    # --- process boundaries: never reached in a test -----------------------
    monkeypatch.setattr(torch_stub, "install", lambda: None)
    monkeypatch.setattr(
        mx.distributed, "init", lambda **_kw: _Group(rank, len(assignments))
    )
    monkeypatch.setattr(
        mx,
        "set_wired_limit",
        lambda value: record["raw_set_wired_limit"].append(value),
    )
    monkeypatch.setattr(
        pipeline_index, "apply_mlx_lm_pipeline_index_patch", lambda: None
    )
    monkeypatch.setattr(
        model_loading,
        "maybe_apply_pre_load_patches",
        lambda _model, **_kwargs: None,
    )
    monkeypatch.setattr(
        inference_worker,
        "pipeline_assignment_is_honored",
        lambda _model: assignment_honored,
    )
    monkeypatch.setattr(inference_worker, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(
        inference_worker,
        "_wait_for_serve_release",
        lambda *_args, **_kwargs: record["order"].append("serve-release"),
    )
    monkeypatch.setattr(
        inference_worker,
        "_run_prefill_shape_warmup",
        lambda *_args, tokens, **_kwargs: record["order"].append("shape-warmup")
        or {"active": True, "tokens": tokens, "elapsed_seconds": 0.01},
    )

    def fake_emit_event(event):
        record["events"].append(dict(event))
        record["order"].append(f"event:{event.get('type')}")

    monkeypatch.setattr(inference_worker, "_emit_event", fake_emit_event)
    monkeypatch.setattr(
        inference_worker, "_start_launcher_watchdog", lambda _m, _pid, **_kw: None
    )

    def fake_wired(desired):
        record["order"].append("wired")
        record["wired_calls"].append(desired)
        return wired_result

    monkeypatch.setattr(enforcer, "_apply_metal_wired_limit", fake_wired)

    # --- mlx-lm ------------------------------------------------------------
    class FakeProvider:
        is_batchable = True

        def __init__(self, cli_args):
            self.cli_args = cli_args
            self.model = None
            # A real ModelProvider always publishes its tokenizer after
            # load_default(); the distributed protocol adapter is part of the
            # ready contract and must see that same object on every rank.
            self.tokenizer = SimpleNamespace()

        def load_default(self):
            record["order"].append("load")
            record["pin_at_load"] = pipeline_patch.assigned_stage()
            self.model = _staged_model(assignment)

    def fake_run(host, port, provider):
        record["order"].append("serve")
        record["serve_address"] = (host, port)
        written = list(tmp_path.glob("*.json"))
        record["marker_while_serving"] = json.loads(written[0].read_text())

    monkeypatch.setattr(mlx_server, "ModelProvider", FakeProvider)
    monkeypatch.setattr(mlx_server, "run", fake_run)

    # --- oMLX seams we want to observe rather than execute -----------------
    monkeypatch.setattr(
        inference_worker,
        "decode_worker_contract",
        lambda _plan: ("a" * 64, assignments, (), tensor_parallel_size),
    )
    # The fake plan above is not a real encoded contract, so the path_map
    # decoder it feeds must be faked alongside it (empty = legacy behavior).
    monkeypatch.setattr(
        inference_worker,
        "decode_worker_path_map",
        lambda _plan: {},
    )
    monkeypatch.setattr(
        inference_worker,
        "decode_worker_speculation",
        lambda _plan: (False, None),
    )

    def fake_guard_rank_load(item, *, rank, **kwargs):
        record["order"].append("guard")
        record["guard_calls"].append({"rank": rank, "node_id": item.node_id, **kwargs})
        return ceiling

    monkeypatch.setattr(inference_worker, "guard_rank_load", fake_guard_rank_load)

    def fake_check_rank_fits(required, **kwargs):
        record["order"].append("stage-guard")
        record["stage_guard_calls"].append({"required": required, **kwargs})
        return ceiling

    monkeypatch.setattr(inference_worker, "check_rank_fits", fake_check_rank_fits)

    @contextlib.contextmanager
    def fake_watch(budget, **_kwargs):
        record["watch_budget"] = budget
        yield None

    monkeypatch.setattr(inference_worker, "watch_rank_load", fake_watch)

    @contextlib.contextmanager
    def fake_pipeline_compat(_assignments):
        yield None

    monkeypatch.setattr(
        inference_worker, "install_pipeline_compatibility", fake_pipeline_compat
    )

    @contextlib.contextmanager
    def fake_optimizations(*_args, **_kwargs):
        yield optimizations or {"coalesced_batching": {"active": True}}

    monkeypatch.setattr(
        inference_worker, "install_runtime_optimizations", fake_optimizations
    )

    @contextlib.contextmanager
    def fake_telemetry(_marker, **_kwargs):
        yield SimpleNamespace()

    monkeypatch.setattr(inference_worker, "install_server_telemetry", fake_telemetry)

    def fake_build_guard(_model, **kwargs):
        record["build_guard_kwargs"] = kwargs
        return SimpleNamespace(active=False)

    monkeypatch.setattr(inference_worker, "build_guard", fake_build_guard)

    class SpyWatchdog:
        def __init__(self, hosts_by_rank, **kwargs):
            record["order"].append("watchdog")
            record["watchdog_hosts"] = dict(hosts_by_rank)
            record["watchdog_kwargs"] = kwargs

        def run(self):
            return None

    monkeypatch.setattr(inference_worker, "PeerWatchdog", SpyWatchdog)

    pipeline_patch.clear_assigned_stage()
    try:
        args = build_parser().parse_args(_worker_argv(tmp_path, extra=argv_extra))
        record["exit_code"] = inference_worker.run_worker(args)
        record["pin_after"] = pipeline_patch.assigned_stage()
    finally:
        pipeline_patch.clear_assigned_stage()
    return record


def test_rank_zero_ready_event_follows_collective_shape_warmup(
    monkeypatch,
    tmp_path,
):
    record = _run_rank(
        monkeypatch,
        tmp_path,
        rank=0,
        assignments=_tensor_plan(),
        tensor_parallel_size=2,
        optimizations={
            "deepseek_v4_adaptive_prefill": {
                "active": True,
                "shape_warmup_tokens": 1024,
            }
        },
    )

    order = record["order"]
    assert order.index("event:rank_ready") < order.index("serve-release")
    assert order.index("serve-release") < order.index("shape-warmup")
    assert order.index("shape-warmup") < order.index("event:ready")
    assert order.index("event:ready") < order.index("serve")


def test_worker_refuses_excess_resident_weights_before_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(
        inference_worker,
        "_measured_weight_bytes",
        lambda _model: 100 * GiB,
    )

    with pytest.raises(RuntimeError, match="more parameter memory"):
        _run_rank(monkeypatch, tmp_path)

    markers = list(tmp_path.glob("*.json"))
    assert len(markers) == 1
    failure = json.loads(markers[0].read_text())
    assert failure["phase"] == "failed"
    assert (
        "RuntimeError: loaded model retains more parameter memory" in (failure["error"])
    )


# --- finding 3: the rank is not its own peer -------------------------------


def test_a_rank_never_watches_itself(monkeypatch, tmp_path):
    """Rank 0's SSH target is loopback and its marker is on local disk.

    So its own entry in the peer map could only ever report "reachable, and
    this old" — and nothing refreshed that timestamp while idle. Sixty seconds
    after the last token every healthy rank independently declared itself lost
    and called ``os._exit(1)``, naming the local Mac in the message.
    """

    record = _run_rank(monkeypatch, tmp_path, rank=0)

    assert record["watchdog_hosts"] == {1: ("studio", "studio.local")}
    assert 0 not in record["watchdog_hosts"], "a rank must not be its own peer"


def test_remote_rank_relies_on_launcher_watchdog_instead_of_coordinator_loopback(
    monkeypatch, tmp_path
):
    """SSH targets are launcher-relative, so 127.0.0.1 is wrong on rank one.

    The first real two-Mac JACCL canary passed, then rank one looked for rank
    zero's marker on the Studio's own disk and killed the healthy job after two
    polls. Remote workers are already tied to their SSH launcher parent.
    """

    record = _run_rank(monkeypatch, tmp_path, rank=1)

    assert record["watchdog_hosts"] is None


def test_an_idle_deployment_survives_past_the_stale_window(monkeypatch, tmp_path):
    """The 60-second self-kill, driven through the map ``run_worker`` builds.

    A rank whose marker has not been touched for five minutes — an idle
    cluster between two turns of a conversation — must not be reported lost by
    its own watchdog.
    """

    from omlx.cluster.liveness import check_peers, raise_if_peer_lost

    record = _run_rank(monkeypatch, tmp_path, rank=0)
    idle_marker = dict(record["marker_while_serving"])
    idle_marker["updated_at"] = "2020-01-01T00:00:00+00:00"
    (tmp_path / "mini-abc123-rank-0.json").write_text(json.dumps(idle_marker))

    # The map contains only the remote peer: local idleness cannot kill itself.
    assert 0 not in record["watchdog_hosts"]

    # And the falsifier: the map this replaced did contain rank 0, whose stale
    # local marker would make a runtime watchdog fire.
    doomed = check_peers(
        {0: ("mbp", "127.0.0.1")},
        state_dir=str(tmp_path),
        deployment_id="mini-abc123",
        probe=lambda _target: True,
        require_heartbeat=True,
    )
    with pytest.raises(Exception, match="stopped reporting"):
        raise_if_peer_lost(doomed)


def test_the_peer_watchdog_is_armed_before_the_weights_are_read(monkeypatch, tmp_path):
    """A 300 GB model takes twenty minutes; nothing watched for those minutes."""

    record = _run_rank(monkeypatch, tmp_path)

    order = record["order"]
    assert order.index("watchdog") < order.index("load"), order


# --- finding 8: the stage guard must not consult the pin it checks ---------


def test_the_stage_guard_fires_in_the_order_the_launch_path_runs(monkeypatch, tmp_path):
    """The guard exists for "the pin did not take", so it cannot ask the pin.

    ``effective_stage`` short-circuits on ``_ASSIGNED_STAGE``, and the launch
    path sets that pin, so the guard compared the pin against the plan, found
    them equal by construction, and returned. It called ``check_rank_fits``
    zero times on every real launch.
    """

    record = _run_rank(monkeypatch, tmp_path, rank=0)

    assert len(record["stage_guard_calls"]) == 1, (
        "the loader's own split gives this rank 30 layers, not the planned 14"
    )
    call = record["stage_guard_calls"][0]
    # fixed 2 GiB + 30 layers x 4 GiB each: the 122 GiB from the incident.
    assert call["required"] == 122 * GiB
    assert call["rank"] == 0
    assert call["node_id"] == "mbp"


def test_the_stage_guard_admits_at_the_rank_role_not_the_headless_default(
    monkeypatch, tmp_path
):
    """Second defect in the same call: it passed no role, so 0.90 every time."""

    record = _run_rank(monkeypatch, tmp_path, rank=0)

    call = record["stage_guard_calls"][0]
    assert call["role"] == "workstation"
    assert call["memory_guard_tier"] == "balanced"


def test_a_proven_assignment_hook_uses_the_uneven_plan_without_refusal(
    monkeypatch, tmp_path
):
    """MiniMax and marked compatibility hooks load the plan, not the even split."""

    record = _run_rank(
        monkeypatch,
        tmp_path,
        rank=0,
        assignment_honored=True,
    )

    assert record["stage_guard_calls"] == []
    assert record["pin_at_load"] == (46, 60)


def test_the_stage_pin_still_reaches_the_loader(monkeypatch, tmp_path):
    """Asking what would happen unpinned must not disturb the pin itself."""

    record = _run_rank(monkeypatch, tmp_path, rank=0)

    assert record["pin_at_load"] == (46, 60)


def test_a_plan_the_loader_would_honour_anyway_is_not_second_guessed(
    monkeypatch, tmp_path
):
    record = _run_rank(monkeypatch, tmp_path, rank=0, assignments=_even_plan())

    assert record["stage_guard_calls"] == []


def test_the_stage_guard_answers_the_same_question_from_either_side_of_the_pin(
    monkeypatch,
):
    """Order-independence, so a future edit cannot silence it by moving a line.

    This is the exact state the old test suite could never produce: it only
    ever called ``effective_stage`` after ``clear_assigned_stage()``, which is
    the one state a real launch guarantees is impossible.
    """

    from omlx.patches.minimax_m3_mlx_lm import pipeline_patch

    assignments = _uneven_plan()
    calls: list[int] = []
    monkeypatch.setattr(
        inference_worker,
        "check_rank_fits",
        lambda required, **_kwargs: calls.append(required),
    )

    pipeline_patch.set_assigned_stage(46, 60)
    try:
        inference_worker._guard_effective_stage(
            assignments[0],
            assignments,
            rank=0,
            world_size=2,
            role="workstation",
        )
        assert calls == [122 * GiB], "the pin silenced the guard again"
        assert pipeline_patch.assigned_stage() == (46, 60), "the pin was not restored"
    finally:
        pipeline_patch.clear_assigned_stage()


# --- finding 9: the wired limit --------------------------------------------


def test_the_wired_limit_is_not_raised_before_anything_admits_the_rank(
    monkeypatch, tmp_path
):
    """It ran 36 lines before the guard, unconditionally, at Apple's maximum.

    Wired memory is not pageable, so this is the difference between "the model
    dies" and "the Mac cannot be typed on" — and single-node oMLX on the same
    machine leaves the limit alone entirely.
    """

    record = _run_rank(monkeypatch, tmp_path)

    order = record["order"]
    assert order.index("guard") < order.index("wired") < order.index("load"), order
    assert record["raw_set_wired_limit"] == [], (
        "the rank called mx.set_wired_limit directly, bypassing the sysctl-unset "
        "skip and the 5% OS margin the engine honours"
    )


def test_the_wired_limit_asks_for_the_admission_budget_not_apples_maximum(
    monkeypatch, tmp_path
):
    """The old ask was 107.5 GiB of a 128 GiB laptop, whatever the role said.

    Wiring memory this rank has already been refused permission to use protects
    nothing and costs the person at the keyboard everything. The number asked
    for is the one the role resolved to, so it moves when the role's fraction
    is retuned rather than being pinned here.
    """

    from omlx.cluster.memory_guard import admission_budget

    record = _run_rank(monkeypatch, tmp_path, ceiling=107 * GiB)

    expected = admission_budget(107 * GiB, role="workstation")
    assert record["wired_calls"] == [expected]
    assert record["watch_budget"] == expected, (
        "the load watcher and the wired limit must agree on the same budget"
    )
    assert expected < 107 * GiB, "a rank must not wire the whole ceiling"
    assert expected < admission_budget(107 * GiB, role="headless"), (
        "a Mac someone is working on must wire less than a headless one"
    )


def test_an_unmeasurable_host_leaves_the_allocator_alone(monkeypatch, tmp_path):
    """Ceiling 0 means no admission decision was made; wiring against it is a lie."""

    record = _run_rank(monkeypatch, tmp_path, ceiling=0)

    assert record["wired_calls"] == []
    assert record["raw_set_wired_limit"] == []


def test_a_host_without_the_enforcer_loads_rather_than_failing(monkeypatch):
    """A worker-only install has no ProcessMemoryEnforcer; that must not raise."""

    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "omlx.process_memory_enforcer":
            raise ImportError("no engine stack on this Mac")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert inference_worker._apply_rank_wired_limit(64 * GiB) == 0


# --- the role and the tier reach the rank ----------------------------------


def test_the_role_comes_from_the_plan_because_one_argv_serves_every_host(
    monkeypatch, tmp_path
):
    """mlx.launch runs one identical command line on every Mac.

    So ``--node-role`` cannot express "studio=headless, macbook=workstation",
    and every rank resolved to the 0.90 headless default no matter what the
    user chose. The plan is already per-rank and already hash-covered.
    """

    record = _run_rank(monkeypatch, tmp_path, rank=0)
    assert record["guard_calls"][0]["role"] == "workstation"

    on_the_studio = _run_rank(monkeypatch, tmp_path, rank=1)
    assert on_the_studio["guard_calls"][0]["role"] == "headless"


def test_the_flag_still_overrides_the_plan_for_a_hand_run_rank(monkeypatch, tmp_path):
    record = _run_rank(
        monkeypatch, tmp_path, rank=0, argv_extra=("--node-role", "headless")
    )

    assert record["guard_calls"][0]["role"] == "headless"


def test_a_plan_without_a_role_field_still_launches(monkeypatch, tmp_path):
    """Written with getattr so an older encoded plan cannot break the rank."""

    plain = (
        PipelineAssignment(
            node_id="mbp",
            rank=0,
            start_layer=30,
            end_layer=60,
            layer_weight_bytes=30 * GiB,
            fixed_weight_bytes=GiB,
            reserve_bytes=8 * GiB,
            capacity_bytes=400 * GiB,
        ),
        PipelineAssignment(
            node_id="studio",
            rank=1,
            start_layer=0,
            end_layer=30,
            layer_weight_bytes=30 * GiB,
            fixed_weight_bytes=GiB,
            reserve_bytes=8 * GiB,
            capacity_bytes=400 * GiB,
        ),
    )

    record = _run_rank(monkeypatch, tmp_path, rank=0, assignments=plain)

    assert record["exit_code"] == 0
    assert record["guard_calls"][0]["role"] == ""
    assert record["guard_calls"][0]["memory_guard_tier"] == "balanced"


def test_the_memory_guard_tier_reaches_both_gates(monkeypatch, tmp_path):
    """A user who capped this Mac at 40 GiB got a rank admitting at 107.5.

    The tier had no flag and no plan field, so every ceiling computation on a
    rank silently used "balanced" — and the dynamic reclaim ratio jumped 0.2
    to 0.5 with it.
    """

    record = _run_rank(
        monkeypatch,
        tmp_path,
        rank=0,
        assignments=_uneven_plan(tier="safe"),
    )

    assert record["guard_calls"][0]["memory_guard_tier"] == "safe"
    assert record["stage_guard_calls"][0]["memory_guard_tier"] == "safe"
    assert record["build_guard_kwargs"]["memory_guard_tier"] == "safe"


def test_the_tier_flag_overrides_the_plan(monkeypatch, tmp_path):
    record = _run_rank(
        monkeypatch,
        tmp_path,
        rank=0,
        assignments=_uneven_plan(tier="safe"),
        argv_extra=("--memory-guard-tier", "aggressive"),
    )

    assert record["guard_calls"][0]["memory_guard_tier"] == "aggressive"


def test_the_prefill_guard_is_sized_from_the_step_this_server_will_use(
    monkeypatch, tmp_path
):
    """The attention transient is set by the prefill chunk, not mlx-lm's default."""

    record = _run_rank(
        monkeypatch,
        tmp_path,
        rank=0,
        argv_extra=("--prefill-step-size", "4096"),
    )

    assert record["build_guard_kwargs"]["prefill_step_size"] == 4096


def test_the_rank_reaches_ready_and_records_what_bounded_it(monkeypatch, tmp_path):
    """The whole path, once, so the marker a reader sees is the one that ran."""

    from omlx.cluster.memory_guard import admission_budget

    record = _run_rank(monkeypatch, tmp_path, rank=0)

    marker = record["marker_while_serving"]
    assert record["exit_code"] == 0
    assert marker["phase"] == "ready"
    assert marker["node_role"] == "workstation"
    assert marker["memory_guard_tier"] == "balanced"
    assert marker["admission_ceiling_bytes"] == 107 * GiB
    assert marker["admission_budget_bytes"] == admission_budget(
        107 * GiB, role="workstation"
    )
    assert marker["start_layer"] == 46 and marker["end_layer"] == 60
    assert marker["loaded_start_layer"] == 46, "read off the model, not the plan"
    assert record["serve_address"] == ("127.0.0.1", 32000)


def test_the_marker_is_removed_when_the_rank_exits_cleanly(monkeypatch, tmp_path):
    _run_rank(monkeypatch, tmp_path, rank=0)

    assert list(tmp_path.glob("*.json")) == []


def test_launcher_watchdog_fires_on_a_stale_launcher_lease(tmp_path):
    updates: list[tuple[str, dict]] = []
    events: list[dict] = []
    exit_codes: list[int] = []
    aborts: list[str] = []
    releases: list[str] = []
    marker = SimpleNamespace(
        update=lambda phase, **extra: updates.append((phase, extra))
    )
    lease = tmp_path / "launcher-lease.json"
    lease.write_text("{}", encoding="utf-8")
    stale = time.time() - 120.0
    os.utime(lease, (stale, stale))

    _watch_launcher_parent(
        42,
        marker,
        watched_marker_path=lease,
        marker_stale_after=45.0,
        get_parent_pid=lambda: 42,
        wait=lambda _seconds: None,
        exit_process=exit_codes.append,
        emit_event=events.append,
        on_abort=aborts.append,
        release_memory=releases.append,
    )

    assert updates[0][0] == "launcher_lost"
    assert "stale" in updates[0][1]["error"]
    # The abort stage ran before the exit stage.
    assert len(aborts) == 1
    assert events[0]["type"] == "launcher_lost"
    assert len(releases) == 1
    assert exit_codes == [1]


def test_launcher_watchdog_ignores_a_fresh_lease(tmp_path):
    exit_codes: list[int] = []
    marker = SimpleNamespace(update=lambda phase, **extra: None)
    lease = tmp_path / "launcher-lease.json"
    lease.write_text("{}", encoding="utf-8")

    polls = [0]

    class StopLoop(Exception):
        pass

    def wait(_seconds):
        polls[0] += 1
        if polls[0] >= 3:
            raise StopLoop

    with pytest.raises(StopLoop):
        _watch_launcher_parent(
            42,
            marker,
            watched_marker_path=lease,
            marker_stale_after=45.0,
            get_parent_pid=lambda: 42,
            wait=wait,
            exit_process=exit_codes.append,
            emit_event=lambda _event: None,
        )

    assert exit_codes == []


def test_launcher_watchdog_ignores_a_lease_that_never_appeared(tmp_path):
    exit_codes: list[int] = []
    marker = SimpleNamespace(update=lambda phase, **extra: None)
    missing = tmp_path / "not-yet-created-lease.json"

    polls = [0]

    class StopLoop(Exception):
        pass

    def wait(_seconds):
        polls[0] += 1
        if polls[0] >= 3:
            raise StopLoop

    with pytest.raises(StopLoop):
        _watch_launcher_parent(
            42,
            marker,
            watched_marker_path=missing,
            marker_stale_after=45.0,
            get_parent_pid=lambda: 42,
            wait=wait,
            exit_process=exit_codes.append,
            emit_event=lambda _event: None,
        )

    assert exit_codes == []


def test_launcher_watchdog_fires_if_a_seen_lease_disappears(tmp_path):
    exit_codes: list[int] = []
    updates: list[tuple[str, dict]] = []
    marker = SimpleNamespace(
        update=lambda phase, **extra: updates.append((phase, extra))
    )
    lease = tmp_path / "launcher-lease.json"
    lease.write_text("{}", encoding="utf-8")
    polls = 0

    def wait(_seconds):
        nonlocal polls
        polls += 1
        if polls == 2:
            lease.unlink()

    _watch_launcher_parent(
        42,
        marker,
        watched_marker_path=lease,
        marker_stale_after=45.0,
        get_parent_pid=lambda: 42,
        wait=wait,
        exit_process=exit_codes.append,
        emit_event=lambda _event: None,
        release_memory=lambda _reason: None,
    )

    assert exit_codes == [1]
    assert "disappeared" in updates[0][1]["error"]


def test_launcher_watchdog_exits_when_dead_server_breaks_diagnostics(tmp_path):
    lease = tmp_path / "launcher-lease.json"
    lease.write_text("{}", encoding="utf-8")
    stale = time.time() - 120.0
    os.utime(lease, (stale, stale))
    exit_codes: list[int] = []
    releases: list[str] = []

    def broken(*_args, **_kwargs):
        raise BrokenPipeError("coordinator pipe is gone")

    _watch_launcher_parent(
        42,
        SimpleNamespace(update=broken),
        watched_marker_path=lease,
        marker_stale_after=45.0,
        get_parent_pid=lambda: 42,
        wait=lambda _seconds: None,
        exit_process=exit_codes.append,
        emit_event=broken,
        release_memory=releases.append,
    )

    assert len(releases) == 1
    assert exit_codes == [1]


def test_cancel_request_file_matches_the_telemetry_contract(tmp_path):
    _write_cancel_request(str(tmp_path), "dep-9", "peer watchdog test", plan_hash="e" * 64)

    payload = json.loads((tmp_path / "dep-9-cancel.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["deployment_id"] == "dep-9"
    assert payload["plan_hash"] == "e" * 64
    assert payload["scope"] == "all"
    assert isinstance(payload["epoch"], int) and payload["epoch"] > 0
    assert payload["reason"] == "peer watchdog test"


def test_cancel_request_epoch_advances_past_an_existing_clock_jump(tmp_path):
    path = tmp_path / "dep-9-cancel.json"
    path.write_text(json.dumps({"epoch": 9999999999999}), encoding="utf-8")

    _write_cancel_request(str(tmp_path), "dep-9", "new event", plan_hash="f" * 64)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["epoch"] == 10000000000000


def _fake_mlx_core(calls: list[str], *, with_metal: bool = True) -> SimpleNamespace:
    metal = (
        SimpleNamespace(clear_cache=lambda: calls.append("metal.clear_cache"))
        if with_metal
        else None
    )
    fake = SimpleNamespace(
        set_wired_limit=lambda limit: calls.append(f"set_wired_limit({limit})"),
        clear_cache=lambda: calls.append("clear_cache"),
    )
    if with_metal:
        fake.metal = metal
    return fake


def _install_fake_mlx_core(monkeypatch, fake_core: SimpleNamespace) -> None:
    """Make ``import mlx.core as mx`` bind to the fake without real Metal.

    ``import a.b as x`` resolves ``b`` as an attribute of package ``a``, so
    the fake package must expose ``core`` — a bare sys.modules entry is not
    enough.
    """
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_core))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)


def test_release_metal_memory_unwires_before_clearing(monkeypatch):
    calls: list[str] = []
    _install_fake_mlx_core(monkeypatch, _fake_mlx_core(calls))

    inference_worker._release_metal_memory("test")

    # Unwiring is the load-bearing call — the cache flush alone leaves the
    # wired model weights stranded when os._exit skips every cleanup handler.
    assert calls == [
        "set_wired_limit(0)",
        "clear_cache",
        "metal.clear_cache",
    ]


def test_release_metal_memory_tolerates_a_missing_metal_module(monkeypatch):
    calls: list[str] = []
    _install_fake_mlx_core(monkeypatch, _fake_mlx_core(calls, with_metal=False))

    inference_worker._release_metal_memory("test")

    assert calls == ["set_wired_limit(0)", "clear_cache"]


def test_release_metal_memory_survives_failures(monkeypatch):
    def fail(*_args):
        raise RuntimeError("wedged")

    fake = SimpleNamespace(
        set_wired_limit=fail,
        clear_cache=fail,
        metal=SimpleNamespace(clear_cache=fail),
    )
    _install_fake_mlx_core(monkeypatch, fake)

    # Must not raise: the force-exit behind this hook has to happen regardless.
    inference_worker._release_metal_memory("test")


def test_release_metal_memory_without_mlx_is_a_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)

    inference_worker._release_metal_memory("test")


def test_sigterm_handler_releases_metal_before_interrupt(monkeypatch):
    releases: list[str] = []
    monkeypatch.setattr(
        inference_worker, "_release_metal_memory", releases.append
    )
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)
    previous_alrm = signal.getsignal(signal.SIGALRM)
    try:
        inference_worker._install_signal_handlers()
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGALRM, previous_alrm)

    # The rank releases its own wired memory on SIGTERM, while it still can —
    # the coordinator's SIGKILL escalation behind it cannot unwire anything.
    assert len(releases) == 1
    assert "SIGTERM" in releases[0]


def test_peer_watchdog_on_lost_releases_metal_before_exit(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    class FakePeerWatchdog:
        def __init__(self, _hosts, **kwargs):
            captured.update(kwargs)

        def run(self):
            pass

    monkeypatch.setattr(inference_worker, "PeerWatchdog", FakePeerWatchdog)
    monkeypatch.setattr(inference_worker, "_emit_event", lambda _event: None)
    order: list[str] = []
    monkeypatch.setattr(
        inference_worker,
        "_release_metal_memory",
        lambda _reason: order.append("release"),
    )
    monkeypatch.setattr(os, "_exit", lambda code: order.append(f"exit:{code}"))
    assignments = [
        PipelineAssignment(
            node_id=node,
            rank=rank,
            start_layer=0,
            end_layer=1,
            layer_weight_bytes=10,
            fixed_weight_bytes=10,
            reserve_bytes=10,
            capacity_bytes=100,
        )
        for rank, node in enumerate(("local", "studio"))
    ]
    marker = SimpleNamespace(
        update=lambda phase, **extra: None,
        payload={"deployment_id": "dep-9", "plan_hash": "e" * 64},
    )

    watchdog = inference_worker._start_peer_watchdog(
        marker,
        assignments,
        ["127.0.0.1", "user@studio.local"],
        "dep-9",
        str(tmp_path),
        rank=0,
    )

    assert isinstance(watchdog, FakePeerWatchdog)
    captured["on_lost"]("peer vanished")
    assert order == ["release", "exit:1"]
class _ThinkTokenizer:
    think_end_id = 55
    think_start_id = 54
    unk_token_id = 0


def test_think_token_ids_resolves_from_tokenizer_attributes():
    end_ids, start_id = inference_worker._think_token_ids(_ThinkTokenizer())
    assert end_ids == [55]
    assert start_id == 54


def test_think_token_ids_falls_back_to_encoding_when_attributes_missing():
    class Tokenizer:
        unk_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return {"</think>": [7, 8]}[text]

    end_ids, start_id = inference_worker._think_token_ids(Tokenizer())
    assert end_ids == [7, 8]
    assert start_id is None


def test_think_token_ids_uses_think_end_string_when_id_missing():
    class Tokenizer:
        think_end = "<|/think|>"
        unk_token_id = 0

        @property
        def think_end_id(self):
            raise ValueError("multi-token sequence")

        def encode(self, text, add_special_tokens=False):
            return {"<|/think|>": [42]}[text]

    end_ids, _ = inference_worker._think_token_ids(Tokenizer())
    assert end_ids == [42]


def test_think_token_ids_retries_encode_without_add_special_tokens_kwarg():
    class Tokenizer:
        unk_token_id = 0

        @property
        def think_end_id(self):
            raise ValueError("multi-token sequence")

        def encode(self, text, add_special_tokens=False):
            if add_special_tokens is not False:
                raise TypeError("unexpected kwarg")
            return [9]

    end_ids, _ = inference_worker._think_token_ids(Tokenizer())
    assert end_ids == [9]


def test_think_token_ids_filters_non_integer_ids():
    class Tokenizer:
        unk_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [9, None, "x"]

    end_ids, _ = inference_worker._think_token_ids(Tokenizer())
    assert end_ids == [9]


def test_think_token_ids_returns_empty_when_unsupported():
    class Tokenizer:
        @property
        def think_end_id(self):
            raise ValueError("no thinking support")

        @property
        def think_start_id(self):
            raise ValueError("no thinking support")

        def encode(self, text, add_special_tokens=False):
            raise ValueError("no thinking support")

    end_ids, start_id = inference_worker._think_token_ids(Tokenizer())
    assert end_ids == []
    assert start_id is None


def test_install_thinking_budget_support_appends_processor_per_request():
    from omlx.api.thinking import ThinkingBudgetProcessor

    calls = []

    class FakeResponseGenerator:
        def _tokenize(self, tokenizer, request, args):
            prompt = getattr(request, "prompt_ids", [54])
            initial_state = getattr(request, "initial_state", "reasoning")
            return prompt, [prompt], ["assistant"], initial_state

    class FakeServer:
        ResponseGenerator = FakeResponseGenerator

        @staticmethod
        def _make_logits_processors(args):
            calls.append(args)
            return ["base-processor"]

    server = FakeServer()
    with inference_worker._install_thinking_budget_support(server, _ThinkTokenizer()):
        generator = server.ResponseGenerator()
        args = SimpleNamespace(chat_template_kwargs={"thinking_budget": 2048})
        generator._tokenize(
            _ThinkTokenizer(),
            SimpleNamespace(request_type="chat", initial_state="reasoning"),
            args,
        )
        processors = server._make_logits_processors(args)
        assert len(processors) == 2
        assert isinstance(processors[1], ThinkingBudgetProcessor)
        assert processors[1]._budget == 2048
        assert processors[1]._think_end_ids == [55]

        disabled = SimpleNamespace(
            chat_template_kwargs={"thinking_budget": 2048, "enable_thinking": False}
        )
        generator._tokenize(
            _ThinkTokenizer(),
            SimpleNamespace(request_type="chat", initial_state="reasoning"),
            disabled,
        )
        assert server._make_logits_processors(disabled) == ["base-processor"]

        inactive = SimpleNamespace(chat_template_kwargs={"thinking_budget": 2048})
        generator._tokenize(
            _ThinkTokenizer(),
            SimpleNamespace(request_type="chat", initial_state="normal"),
            inactive,
        )
        assert server._make_logits_processors(inactive) == ["base-processor"]

        text_completion = SimpleNamespace(
            chat_template_kwargs={"thinking_budget": 2048}
        )
        generator._tokenize(
            _ThinkTokenizer(),
            SimpleNamespace(
                request_type="text",
                prompt="<think>",
                prompt_ids=[54],
                initial_state="normal",
            ),
            text_completion,
        )
        assert len(server._make_logits_processors(text_completion)) == 2

    # The original factory is restored after the context exits.
    assert (
        calls and server._make_logits_processors is FakeServer._make_logits_processors
    )
    assert server.ResponseGenerator._tokenize is FakeResponseGenerator._tokenize


def test_thinking_close_pattern_and_utf8_piece_match_scheduler_rules():
    class Tokenizer:
        chat_template = r"{{ '\n</think>\n\n' }}"

        def encode(self, text, add_special_tokens=False):
            return {"\n": [10], "\n\n": [11, 12]}[text]

        def convert_ids_to_tokens(self, token_id):
            return {1: "<0xE3>", 2: "ordinary"}[token_id]

    tokenizer = Tokenizer()
    leading, trailing = inference_worker._thinking_close_pattern(tokenizer, "</think>")

    assert leading == [10]
    assert trailing == [11, 12]
    assert inference_worker._thinking_budget_token_to_piece(tokenizer, 1) == b"\xe3"
    assert inference_worker._thinking_budget_token_to_piece(tokenizer, 2) == "ordinary"


def test_install_thinking_budget_support_is_noop_without_think_tokens():
    class Tokenizer:
        @property
        def think_end_id(self):
            raise ValueError("no thinking support")

        def encode(self, text, add_special_tokens=False):
            raise ValueError("no thinking support")

    class FakeServer:
        @staticmethod
        def _make_logits_processors(args):
            return ["base-processor"]

    server = FakeServer()
    with inference_worker._install_thinking_budget_support(server, Tokenizer()):
        assert server._make_logits_processors(None) == ["base-processor"]
    assert server._make_logits_processors is FakeServer._make_logits_processors
