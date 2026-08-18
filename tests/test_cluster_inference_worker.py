# SPDX-License-Identifier: Apache-2.0
"""Fail-closed validation of the model stage loaded by a cluster rank."""

import contextlib
import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest

import omlx.cluster.inference_worker as inference_worker
from omlx.cluster.inference_worker import (
    _bind_generation_thread_stream,
    _cross_thread_generation_stream,
    _execution_settings,
    _install_distributed_model_protocol,
    _server_arguments,
    _validate_loaded_stage,
    _validate_measured_weight_bytes,
    _watch_launcher_parent,
    build_parser,
)
from omlx.cluster.planner import PipelineAssignment

GiB = 1024**3


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
        model_loading, "maybe_apply_pre_load_patches", lambda _model: None
    )
    monkeypatch.setattr(
        inference_worker,
        "pipeline_assignment_is_honored",
        lambda _model: assignment_honored,
    )
    monkeypatch.setattr(inference_worker, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(
        inference_worker, "_start_launcher_watchdog", lambda _m, _pid: None
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
        lambda _plan: ("a" * 64, assignments, (), 1),
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
        yield {"coalesced_batching": {"active": True}}

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
