# SPDX-License-Identifier: Apache-2.0
"""Planning happens on whichever Mac holds the whole model, not by hand."""

import json
import struct

import pytest

from omlx.cluster import planner, staging
from omlx.cluster.planner import (
    LOCAL_NODE,
    ModelLayout,
    NodeBudget,
    PlanningError,
    complete_model_layout,
    inspect_safetensors_layout,
    locate_model_layout,
    plan_unequal_pipeline,
    remote_model_layout,
)


def _write_shard(directory, name, layers):
    header = {}
    offset = 0
    for layer in layers:
        header[f"model.layers.{layer}.self_attn.q_proj.weight"] = {
            "dtype": "F16",
            "shape": [8, 8],
            "data_offsets": [offset, offset + 128],
        }
        offset += 128
    blob = json.dumps(header).encode()
    (directory / name).write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def _model(root, *, present_layers, declared_layers, index=False):
    """A model directory holding ``present_layers`` of a ``declared_layers`` model."""

    root.mkdir(parents=True, exist_ok=True)
    for layer in present_layers:
        _write_shard(root, f"model-{layer:05d}.safetensors", [layer])
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "num_hidden_layers": declared_layers,
                "hidden_size": 64,
                "num_attention_heads": 8,
            }
        )
    )
    if index:
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        f"model.layers.{layer}.self_attn.q_proj.weight": (
                            f"model-{layer:05d}.safetensors"
                        )
                        for layer in range(declared_layers)
                    }
                }
            )
        )
    return root


def _explode(*args, **kwargs):
    raise AssertionError("must not reach for a peer")


# ---------------------------------------------------------------------------
# A node holding one stage must not plan from it.
# ---------------------------------------------------------------------------


def test_a_node_holding_only_its_stage_is_refused(tmp_path):
    """Rank 0's shards read as a whole small model, which is the dangerous part."""

    root = _model(tmp_path / "m", present_layers=range(4), declared_layers=8)

    # Nothing in the files themselves says the other four layers exist.
    assert inspect_safetensors_layout(root).layer_count == 4

    with pytest.raises(PlanningError, match="4 of 8 layers"):
        complete_model_layout(root)


def test_a_missing_shard_is_refused(tmp_path):
    """The index names every file, so a stage-only node cannot even be read."""

    root = _model(tmp_path / "m", present_layers=range(4), declared_layers=8, index=True)

    with pytest.raises(PlanningError, match="weight file is missing"):
        complete_model_layout(root)


def test_a_complete_model_is_accepted(tmp_path):
    root = _model(tmp_path / "m", present_layers=range(8), declared_layers=8)

    assert complete_model_layout(root).layer_count == 8


def test_a_draft_head_past_the_declared_depth_is_not_a_partial_model(tmp_path):
    """MTP and EAGLE weights add layers the config never counted.

    The model is complete, so it must not be refused as a stage of itself —
    and the extra head must not be partitioned either, because the runtime
    model never instantiates it and a stage boundary over it fails to load.
    """

    root = _model(tmp_path / "m", present_layers=range(9), declared_layers=8)

    assert complete_model_layout(root).layer_count == 8


def test_a_model_whose_config_omits_its_depth_is_still_readable(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    _write_shard(root, "model-00000.safetensors", [0, 1])
    (root / "config.json").write_text(json.dumps({"model_type": "llama"}))

    assert complete_model_layout(root).layer_count == 2


# ---------------------------------------------------------------------------
# Choosing the node: ask, do not assume.
# ---------------------------------------------------------------------------


def test_the_local_node_plans_without_ssh_when_it_has_the_model(tmp_path, monkeypatch):
    root = _model(tmp_path / "m", present_layers=range(8), declared_layers=8)
    monkeypatch.setattr(planner, "remote_model_layout", _explode)

    holder = locate_model_layout(root, ["studio"])

    assert holder.node == LOCAL_NODE
    assert holder.is_local
    assert holder.layout.layer_count == 8


def test_the_peer_that_has_the_model_is_the_one_that_plans(tmp_path, monkeypatch):
    """The Mac being planned for holds one stage; the Studio holds the model."""

    stage_only = _model(tmp_path / "m", present_layers=range(4), declared_layers=8)
    asked = []

    def fake_remote(ssh_target, model_dir, **kwargs):
        asked.append((ssh_target, model_dir))
        return ModelLayout(
            source=str(model_dir),
            fixed_weight_bytes=1000,
            layer_weight_bytes=(100,) * 8,
        )

    monkeypatch.setattr(planner, "remote_model_layout", fake_remote)

    holder = locate_model_layout(stage_only, ["localhost", "studio"])

    assert holder.node == "studio", "and that is where staging pulls from"
    assert not holder.is_local
    assert holder.layout.layer_count == 8
    assert asked == [("studio", str(stage_only))], "no ssh to ourselves"


def test_peers_are_asked_in_order_until_one_has_the_model(tmp_path, monkeypatch):
    stage_only = _model(tmp_path / "m", present_layers=range(4), declared_layers=8)
    asked = []

    def fake_remote(ssh_target, model_dir, **kwargs):
        asked.append(ssh_target)
        if ssh_target != "studio":
            raise PlanningError(f"{ssh_target} holds 4 of 8 layers")
        return ModelLayout(
            source="/models/m", fixed_weight_bytes=0, layer_weight_bytes=(100,) * 8
        )

    monkeypatch.setattr(planner, "remote_model_layout", fake_remote)

    assert locate_model_layout(stage_only, ["mini", "studio", "mbp"]).node == "studio"
    assert asked == ["mini", "studio"], "the search stops at the first holder"


def test_every_node_and_its_reason_is_named_when_nobody_has_the_model(
    tmp_path, monkeypatch
):
    """Otherwise the operator is told only that planning failed, not where to look."""

    stage_only = _model(tmp_path / "m", present_layers=range(4), declared_layers=8)

    def fake_remote(ssh_target, model_dir, **kwargs):
        raise PlanningError("weight file is missing: model-00005.safetensors")

    monkeypatch.setattr(planner, "remote_model_layout", fake_remote)

    with pytest.raises(PlanningError) as excinfo:
        locate_model_layout(stage_only, ["studio"])

    message = str(excinfo.value)
    assert "local: " in message and "4 of 8 layers" in message
    assert "studio: " in message and "model-00005.safetensors" in message


def test_a_model_that_is_nowhere_is_reported_rather_than_crashing(tmp_path):
    with pytest.raises(PlanningError, match="no node holds a complete copy"):
        locate_model_layout(tmp_path / "absent", [])


# ---------------------------------------------------------------------------
# Carrying the answer back: same code on the peer, same plan either side.
# ---------------------------------------------------------------------------


def test_the_peer_runs_the_same_layout_code(monkeypatch):
    captured = {}

    def fake_run(ssh_target, snippet, argument, **kwargs):
        captured.update(target=ssh_target, snippet=snippet, argument=argument)
        return ModelLayout(
            source="/Users/omlx/.omlx/models/m",
            fixed_weight_bytes=2048,
            layer_weight_bytes=(100, 200, 300),
            tensor_count=9,
            supports_pipeline=True,
        ).to_dict()

    monkeypatch.setattr(planner, "run_remote_python", fake_run)

    layout = remote_model_layout("studio", "~/.omlx/models/m")

    assert captured["target"] == "studio"
    assert captured["argument"] == "~/.omlx/models/m"
    assert "complete_model_layout" in captured["snippet"], "the peer runs our checks too"
    assert layout.source == "/Users/omlx/.omlx/models/m"
    assert layout.layer_weight_bytes == (100, 200, 300)
    assert layout.supports_pipeline


def test_a_peer_that_cannot_read_the_model_fails_as_a_planning_error(monkeypatch):
    def fake_run(*args, **kwargs):
        raise RuntimeError("could not read the model layout on studio: no such file")

    monkeypatch.setattr(planner, "run_remote_python", fake_run)

    with pytest.raises(PlanningError, match="no such file"):
        remote_model_layout("studio", "/models/gone")


def test_a_peer_that_never_answers_fails_as_a_planning_error(monkeypatch):
    import subprocess

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired("ssh", 600)

    monkeypatch.setattr(planner, "run_remote_python", fake_run)

    with pytest.raises(PlanningError):
        remote_model_layout("studio", "/models/m")


def test_a_layout_survives_the_trip_between_nodes(tmp_path):
    root = _model(tmp_path / "m", present_layers=range(8), declared_layers=8)
    layout = complete_model_layout(root)

    assert ModelLayout.from_dict(json.loads(json.dumps(layout.to_dict()))) == layout


def test_the_plan_is_the_same_whichever_node_measured_the_model(tmp_path):
    """A remotely measured layout must not produce a different plan hash."""

    root = _model(tmp_path / "m", present_layers=range(8), declared_layers=8)
    local = complete_model_layout(root)
    carried = ModelLayout.from_dict(local.to_dict())
    nodes = [
        NodeBudget(node_id="studio", capacity_bytes=1 << 30, rank=0),
        NodeBudget(node_id="mbp", capacity_bytes=1 << 30, rank=1),
    ]

    assert (
        plan_unequal_pipeline(carried, nodes).plan_hash
        == plan_unequal_pipeline(local, nodes).plan_hash
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"fixed_weight_bytes": 1},
        {"fixed_weight_bytes": 1, "layer_weight_bytes": "eight"},
        {"fixed_weight_bytes": 1, "layer_weight_bytes": [1, None]},
        {"layer_weight_bytes": [1, 2]},
        {"fixed_weight_bytes": "lots", "layer_weight_bytes": [1, 2]},
    ],
)
def test_a_garbled_answer_is_refused_not_planned_on(payload):
    with pytest.raises(PlanningError):
        ModelLayout.from_dict(payload)


# ---------------------------------------------------------------------------
# The ssh hop itself.
# ---------------------------------------------------------------------------


class _Completed:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def test_the_peer_is_given_the_path_and_not_a_command(monkeypatch):
    """model_dir arrives from an API request; the peer's shell must not run it.

    Stands a local shell in for the peer's, so the assertion is what a shell
    does with the command rather than what it looks like.
    """

    import subprocess
    import sys

    hostile = "/models/$(echo substituted)/`echo backticked`/m"

    run = subprocess.run  # captured before the patch below replaces it

    def shell_as_peer(argv, **kwargs):
        return run(["sh", "-c", argv[-1]], capture_output=True, text=True, check=False)

    monkeypatch.setattr(staging.subprocess, "run", shell_as_peer)

    seen = staging.run_remote_python(
        "studio",
        "import json,sys;print(json.dumps(sys.argv[1]))",
        hostile,
        description="test",
        python_executable=sys.executable,
    )

    assert seen == hostile, "the peer must receive the path, not its output"


def test_the_remote_interpreter_path_still_expands_on_the_peer(monkeypatch):
    captured = {}

    def fake_subprocess_run(argv, **kwargs):
        captured["command"] = argv[-1]
        return _Completed(stdout="[]")

    monkeypatch.setattr(staging.subprocess, "run", fake_subprocess_run)

    staging.run_remote_python("studio", "print(1)", "/m", description="test")

    assert captured["command"].startswith("~/omlx-distributed/.venv/bin/python -c ")


def _plan_cli(model_root, *extra):
    import subprocess
    import sys

    return subprocess.run(
        [
            sys.executable, "-m", "omlx.cli", "cluster", "plan",
            "--model", str(model_root),
            "--node", "studio=8GiB",
            "--node", "mbp=4GiB",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_the_cli_plans_from_a_model_this_mac_holds(tmp_path):
    root = _model(tmp_path / "m", present_layers=range(8), declared_layers=8)

    result = _plan_cli(root, "--json")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model"]["layer_count"] == 8


def test_the_cli_asks_the_peers_it_was_given(monkeypatch, capsys):
    """--peer is what lets a stage-only Mac plan without a hand-carried JSON."""

    import argparse

    from omlx.cli import cluster_command

    asked = {}

    def fake_locate(model_path, hosts, **kwargs):
        asked["hosts"] = list(hosts)
        return planner.ModelHolder(
            node="studio",
            layout=ModelLayout(
                source="/models/m", fixed_weight_bytes=0, layer_weight_bytes=(100,) * 8
            ),
        )

    monkeypatch.setattr(planner, "locate_model_layout", fake_locate)

    code = cluster_command(
        argparse.Namespace(
            cluster_action="plan",
            model="~/.omlx/models/m",
            model_size=None,
            layers=8,
            node=["studio=8GiB", "mbp=4GiB"],
            reserve="0",
            peer=["studio", "mini"],
            json=False,
        )
    )

    assert code == 0
    assert asked["hosts"] == ["studio", "mini"]
    assert "Measured:    studio" in capsys.readouterr().out


def test_the_cli_refuses_to_plan_from_one_stage_and_says_who_to_ask(tmp_path):
    """The failure that forced planning to be done by hand on the Studio."""

    root = _model(tmp_path / "m", present_layers=range(4), declared_layers=8)

    result = _plan_cli(root)

    assert result.returncode == 2
    assert "4 of 8 layers" in result.stderr
    assert "--peer" in _plan_cli(root, "--help").stdout
