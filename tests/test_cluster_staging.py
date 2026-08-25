# SPDX-License-Identifier: Apache-2.0
"""Staging must send each Mac its layers' shards and nothing else."""

import json
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from omlx.cluster.staging import (
    _REMOTE_FILE_SIZES_SNIPPET,
    _REMOTE_INSTALL_SNIPPET,
    ShardInfo,
    index_shards,
    plan_cluster_staging,
    plan_staging,
    scp_copy,
    scp_push,
    shards_for_stage,
    sidecar_files,
    stage_manifest,
    validate_staged_model,
)


def _write_shard(directory, name, tensors, payload=b"\x00" * 32):
    header = {t: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for t in tensors}
    blob = json.dumps(header).encode()
    (directory / name).write_bytes(struct.pack("<Q", len(blob)) + blob + payload)


def _model(tmp_path, layers=8, per_file=2, with_index=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for start in range(0, layers, per_file):
        names = [
            f"model.layers.{i}.self_attn.q_proj.weight"
            for i in range(start, min(start + per_file, layers))
        ]
        fname = f"model-{start:05d}.safetensors"
        _write_shard(tmp_path, fname, names)
        for n in names:
            mapping[n] = fname
    _write_shard(tmp_path, "model-shared.safetensors", ["model.embed_tokens.weight"])
    mapping["model.embed_tokens.weight"] = "model-shared.safetensors"
    if with_index:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": mapping})
        )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "tokenizer.json").write_text("{}")
    return tmp_path


def test_indexes_shards_by_scanning_headers(tmp_path):
    """Mixed-precision exports often ship with no index.json — GLM-5.2 does not."""

    root = _model(tmp_path / "m", layers=8, per_file=2)
    shards = index_shards(root)

    assert len(shards) == 5  # 4 layer files + 1 shared
    layered = {s.name: sorted(s.layers) for s in shards if s.layers}
    assert layered["model-00000.safetensors"] == [0, 1]
    assert layered["model-00006.safetensors"] == [6, 7]
    shared = [s for s in shards if s.has_shared_tensors]
    assert [s.name for s in shared] == ["model-shared.safetensors"]


def test_uses_the_index_when_present(tmp_path):
    root = _model(tmp_path / "m", layers=8, per_file=2, with_index=True)
    shards = index_shards(root)
    assert sorted(next(s for s in shards if "00004" in s.name).layers) == [4, 5]


def test_a_stage_gets_only_its_shards_plus_shared(tmp_path):
    root = _model(tmp_path / "m", layers=8, per_file=2)
    shards = index_shards(root)

    tail = shards_for_stage(shards, 6, 8)
    names = {s.name for s in tail}
    assert "model-00006.safetensors" in names
    assert "model-shared.safetensors" in names, "embeddings/lm_head always travel"
    assert "model-00000.safetensors" not in names, "must not ship other stages' layers"


def test_plan_reports_the_saving(tmp_path):
    root = _model(tmp_path / "m", layers=8, per_file=2)
    plan = plan_staging(root, node_id="mbp", start_layer=6, end_layer=8)

    assert plan.node_id == "mbp"
    assert plan.required_bytes < plan.total_model_bytes
    assert plan.saved_bytes > 0
    assert plan.missing == plan.required, "nothing present yet, so everything is missing"
    assert plan.to_dict()["ready"] is False


def test_files_already_present_are_not_resent(tmp_path):
    root = _model(tmp_path / "m", layers=8, per_file=2)
    shards = {s.name: s.size_bytes for s in index_shards(root)}

    plan = plan_staging(root, node_id="mbp", start_layer=6, end_layer=8, present=shards)
    assert plan.missing == ()
    assert plan.missing_bytes == 0
    assert plan.to_dict()["ready"] is True


def test_remote_file_inventory_treats_missing_directory_as_empty(tmp_path):
    """A peer with no destination directory is the ordinary first staging run."""

    missing = tmp_path / "not-created"
    result = subprocess.run(
        [sys.executable, "-c", _REMOTE_FILE_SIZES_SNIPPET, str(missing)],
        capture_output=True,
        check=True,
        text=True,
    )

    assert json.loads(result.stdout) == {}


def test_a_truncated_file_is_resent(tmp_path):
    """Size mismatch means a half-finished transfer, not a staged file."""

    root = _model(tmp_path / "m", layers=8, per_file=2)
    present = {s.name: 1 for s in index_shards(root)}

    plan = plan_staging(root, node_id="mbp", start_layer=6, end_layer=8, present=present)
    assert plan.missing, "wrong-sized files must be re-staged"


def test_local_push_publishes_a_hidden_partial_file_atomically(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.safetensors").write_bytes(b"complete")
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    scp_push(
        destination_host="studio.local",
        source_dir=str(source),
        destination_dir="/models/example",
        filename="weights.safetensors",
    )

    scp_command = next(command for command in commands if command[0] == "scp")
    assert ".omlx-stage-" in scp_command[-1]
    assert scp_command[-1].rstrip("'").endswith(".part")
    install_command = commands[-1][-1]
    assert "os.replace" in install_command
    assert "/models/example/weights.safetensors" in install_command


def test_remote_pull_does_not_expose_the_final_name_until_scp_finishes(
    tmp_path, monkeypatch
):
    destination = tmp_path / "model"
    seen = []

    def fake_run(command, **_kwargs):
        seen.append(command)
        if command[0] == "scp":
            target = Path(command[-1])
            assert target.name.startswith(".omlx-stage-")
            assert target.suffix == ".part"
            assert not (destination / "weights.safetensors").exists()
            target.write_bytes(b"complete")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    scp_copy(
        source_host="studio.local",
        destination_host="127.0.0.1",
        source_dir="/models/example",
        destination_dir=str(destination),
        filename="weights.safetensors",
    )

    assert (destination / "weights.safetensors").read_bytes() == b"complete"
    assert not list(destination.glob(".omlx-stage-*.part"))


def test_remote_install_refuses_a_wrong_sized_partial_without_replacing_final(
    tmp_path,
):
    temporary = tmp_path / ".omlx-stage-test.part"
    final = tmp_path / "weights.safetensors"
    temporary.write_bytes(b"short")
    final.write_bytes(b"known-good")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _REMOTE_INSTALL_SNIPPET,
            str(temporary),
            str(final),
            "99",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert final.read_bytes() == b"known-good"
    assert temporary.read_bytes() == b"short"


def test_remote_install_atomically_replaces_final_after_size_validation(tmp_path):
    temporary = tmp_path / ".omlx-stage-test.part"
    final = tmp_path / "weights.safetensors"
    temporary.write_bytes(b"complete")
    final.write_bytes(b"old")

    subprocess.run(
        [
            sys.executable,
            "-c",
            _REMOTE_INSTALL_SNIPPET,
            str(temporary),
            str(final),
            str(len(b"complete")),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert final.read_bytes() == b"complete"
    assert not temporary.exists()


def test_rank_validation_allows_unrelated_indexed_shards_to_be_absent(tmp_path):
    root = _model(tmp_path / "m", layers=8, per_file=2, with_index=True)
    (root / "model-00000.safetensors").unlink()

    status = validate_staged_model(root, 4, 8)

    assert status["stage_ready"] is True
    assert status["missing_files"] == []
    assert "model-00000.safetensors" not in status["required_files"]


def test_rank_validation_rejects_a_missing_assigned_shard(tmp_path):
    root = _model(tmp_path / "m", layers=8, per_file=2, with_index=True)
    (root / "model-00006.safetensors").unlink()

    status = validate_staged_model(root, 4, 8)

    assert status["stage_ready"] is False
    assert status["missing_files"] == ["model-00006.safetensors"]


def test_cluster_plan_covers_every_layer_across_nodes(tmp_path):
    root = _model(tmp_path / "m", layers=8, per_file=2)

    class A:
        def __init__(self, node_id, s, e):
            self.node_id, self.start_layer, self.end_layer = node_id, s, e

    plans = plan_cluster_staging(root, [A("studio", 0, 6), A("mbp", 6, 8)])
    assert [p.node_id for p in plans] == ["studio", "mbp"]
    # Between them the nodes hold every layer file.
    covered = set()
    for p in plans:
        covered |= set(p.required)
    assert len([n for n in covered if "shared" not in n]) == 4
    # And the small stage really is smaller.
    assert plans[1].required_bytes < plans[0].required_bytes


def test_manifest_reads_the_exact_model_path_on_local_and_remote(tmp_path, monkeypatch):
    """Configured model directories must not be rewritten to ~/.omlx/models."""

    root = _model(tmp_path / "models with spaces" / "qwen", layers=4, per_file=2)

    class A:
        def __init__(self, node_id, start, end):
            self.node_id = node_id
            self.start_layer = start
            self.end_layer = end

    remote_calls = []
    present = {shard.name: shard.size_bytes for shard in index_shards(root)}

    def fake_remote(host, directory):
        remote_calls.append((host, directory))
        return present | {
            path.name: path.stat().st_size
            for path in root.iterdir()
            if path.is_file() and not path.name.endswith(".safetensors")
        }

    monkeypatch.setattr("omlx.cluster.staging.remote_file_sizes", fake_remote)
    monkeypatch.setattr(
        "omlx.cluster.staging.remote_model_dir", lambda _host, path: path
    )
    manifest = stage_manifest(
        root,
        [A("mbp", 0, 4), A("studio", 0, 4)],
        {"mbp": "127.0.0.1", "studio": "Studio.local"},
    )

    assert manifest["ready"] is True
    assert all(node["missing_files"] == 0 for node in manifest["nodes"])
    assert remote_calls == [("Studio.local", str(root))]


def test_manifest_is_not_ready_when_a_remote_sidecar_is_missing(
    tmp_path, monkeypatch
):
    root = _model(tmp_path / "m", layers=2, per_file=1)

    class A:
        node_id = "studio"
        start_layer = 0
        end_layer = 2

    weights = {shard.name: shard.size_bytes for shard in index_shards(root)}
    monkeypatch.setattr(
        "omlx.cluster.staging.remote_file_sizes",
        lambda host, directory: weights,
    )
    monkeypatch.setattr(
        "omlx.cluster.staging.remote_model_dir", lambda _host, path: path
    )

    manifest = stage_manifest(root, [A()], {"studio": "studio.local"})

    assert manifest["ready"] is False
    assert manifest["nodes"][0]["missing_files"] == 0
    assert manifest["nodes"][0]["missing_sidecars"] > 0
    assert manifest["total_missing_bytes"] > 0


def test_sidecars_are_identified(tmp_path):
    root = _model(tmp_path / "m")
    names = sidecar_files(root)
    assert "config.json" in names and "tokenizer.json" in names
    assert not any(n.endswith(".safetensors") for n in names)


def test_a_directory_without_weights_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no safetensors"):
        index_shards(tmp_path / "empty")


def test_shardinfo_is_hashable_for_set_operations():
    s = ShardInfo("a.safetensors", 10, frozenset({1}), False)
    assert len({s, s}) == 1


def test_remote_staging_pushes_weights_and_sidecars_to_the_named_node(
    tmp_path,
    monkeypatch,
):
    from omlx.cluster.staging import stage_remote_files

    root = _model(tmp_path / "source", layers=4, per_file=2)
    shards = index_shards(root)
    plan = plan_staging(
        root,
        node_id="studio",
        start_layer=2,
        end_layer=4,
        present={},
        shards=shards,
    )
    landed = {}
    pushes = []
    progress = []

    def fake_transfer(
        *,
        destination_host,
        source_dir,
        destination_dir,
        filename,
    ):
        pushes.append((destination_host, filename))
        landed[filename] = (root / filename).stat().st_size

    monkeypatch.setattr(
        "omlx.cluster.staging.check_disk_for_staging",
        lambda *args, **kwargs: 100 * 1024**3,
    )
    result = stage_remote_files(
        plan,
        model_path=root,
        destination_host="studio.local",
        sidecars=sidecar_files(root),
        transfer=fake_transfer,
        present_reader=lambda _host, _path: dict(landed),
        progress=lambda name, status, size: progress.append((name, status, size)),
    )

    assert result.ok
    assert pushes
    assert {host for host, _name in pushes} == {"studio.local"}
    copied = {name for _host, name in pushes}
    assert "config.json" in copied and "tokenizer.json" in copied
    assert "model-00002.safetensors" in copied
    assert "model-00000.safetensors" not in copied
    assert any(status == "copied" for _name, status, _size in progress)


def test_remote_staging_resumes_without_recopying_verified_files(
    tmp_path,
    monkeypatch,
):
    from omlx.cluster.staging import stage_remote_files

    root = _model(tmp_path / "source", layers=2, per_file=2)
    plan = plan_staging(root, node_id="studio", start_layer=0, end_layer=2)
    landed = {
        name: (root / name).stat().st_size
        for name in (*plan.required, *sidecar_files(root))
    }
    monkeypatch.setattr(
        "omlx.cluster.staging.check_disk_for_staging",
        lambda *args, **kwargs: 100 * 1024**3,
    )

    result = stage_remote_files(
        plan,
        model_path=root,
        destination_host="studio.local",
        sidecars=sidecar_files(root),
        transfer=lambda **_kwargs: pytest.fail("verified files must not be copied"),
        present_reader=lambda _host, _path: dict(landed),
    )

    assert result.ok
    assert result.copied == ()


def test_remote_staging_probes_the_peer_destination_dir_not_the_source(
    tmp_path,
    monkeypatch,
):
    # A cross-user cluster: the source lives under the coordinator's home, but
    # the peer holds its copy under a different $HOME. The present-file probe
    # and scp destination must use the peer path, otherwise the probe reads an
    # empty directory and re-copies the whole model.
    from omlx.cluster.staging import stage_remote_files

    root = _model(tmp_path / "source", layers=2, per_file=2)
    peer_dir = "/Users/peer/.omlx/models/m"
    plan = plan_staging(root, node_id="studio", start_layer=0, end_layer=2)
    landed = {
        name: (root / name).stat().st_size
        for name in (*plan.required, *sidecar_files(root))
    }
    seen_paths = []
    monkeypatch.setattr(
        "omlx.cluster.staging.check_disk_for_staging",
        lambda *args, **kwargs: 100 * 1024**3,
    )

    def present_reader(_host, path):
        seen_paths.append(path)
        return dict(landed)

    result = stage_remote_files(
        plan,
        model_path=root,
        destination_host="studio.local",
        destination_dir=peer_dir,
        sidecars=sidecar_files(root),
        transfer=lambda **_kwargs: pytest.fail(
            "files already on the peer must not be copied"
        ),
        present_reader=present_reader,
    )

    assert seen_paths == [peer_dir]
    assert result.ok
    assert result.copied == ()


def test_peer_owned_model_stages_from_the_holder_not_the_coordinator(monkeypatch):
    from omlx.cluster import staging
    from omlx.cluster.staging import StagingPlan, stage_files_from_source

    plan = StagingPlan(
        node_id="MacBook",
        start_layer=6,
        end_layer=8,
        required=("tail.safetensors",),
        missing=("tail.safetensors",),
        required_bytes=100,
        missing_bytes=100,
        total_model_bytes=1000,
    )
    landed = {}
    copies = []

    def transfer(**kwargs):
        copies.append(kwargs)
        landed[kwargs["filename"]] = 100

    monkeypatch.setattr(
        staging,
        "remote_file_sizes",
        lambda _host, _path: dict(landed),
    )
    monkeypatch.setattr(
        staging,
        "check_disk_for_staging",
        lambda *args, **kwargs: 100 * 1024**3,
    )
    resolved = []

    def fake_remote_model_dir(host, path, **_kwargs):
        resolved.append((host, path))
        return "/Users/peeruser/models/m"

    monkeypatch.setattr(staging, "remote_model_dir", fake_remote_model_dir)

    result = stage_files_from_source(
        plan,
        model_path="/models/m",
        source_host="studio",
        destination_host="macbook",
        expected_sizes={"tail.safetensors": 100},
        transfer=transfer,
    )

    assert result.ok
    assert copies[0]["source_host"] == "studio"
    assert copies[0]["destination_host"] == "macbook"
    # The scp source must be the path resolved in the SOURCE peer's own home,
    # not the coordinator's absolute form.
    assert resolved == [("studio", "/models/m")]
    assert copies[0]["source_dir"] == "/Users/peeruser/models/m"


def test_manifest_can_be_built_from_a_peer_shard_index(tmp_path, monkeypatch):
    from omlx.cluster import staging

    local = tmp_path / "model"
    local.mkdir()
    (local / "tail.safetensors").write_bytes(b"x" * 100)
    (local / "config.json").write_text("{}")
    shards = (
        ShardInfo("head.safetensors", 800, frozenset(range(6)), False),
        ShardInfo("tail.safetensors", 100, frozenset({6, 7}), False),
        ShardInfo("shared.safetensors", 50, frozenset(), True),
    )
    monkeypatch.setattr(
        staging,
        "remote_model_staging_inventory",
        lambda _host, _path, **_kwargs: (shards, {"config.json": 2}),
    )
    monkeypatch.setattr(
        staging, "remote_model_dir", lambda _host, path: path
    )
    monkeypatch.setattr(
        staging,
        "remote_file_sizes",
        lambda host, _path: (
            {
                "head.safetensors": 800,
                "tail.safetensors": 100,
                "shared.safetensors": 50,
                "config.json": 2,
            }
            if host == "studio"
            else {}
        ),
    )

    class Assignment:
        def __init__(self, node_id, start, end):
            self.node_id = node_id
            self.start_layer = start
            self.end_layer = end

    manifest = stage_manifest(
        local,
        [
            Assignment("MacBook", 6, 8),
            Assignment("Studio", 0, 6),
        ],
        {"MacBook": "127.0.0.1", "Studio": "studio"},
        source_host="studio",
    )

    assert manifest["source_host"] == "studio"
    assert manifest["nodes"][0]["missing_files"] == 1
    assert manifest["nodes"][1]["ready"] is True


def test_stage_route_runs_a_model_by_node_job_with_live_progress(
    tmp_path,
    monkeypatch,
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from omlx.cluster import routes
    from omlx.cluster.planner import ModelLayout
    from omlx.cluster.staging import StagingResult

    root = _model(tmp_path / "source", layers=4, per_file=2)
    app = FastAPI()
    app.include_router(routes.router)
    monkeypatch.setattr(routes, "remote_file_sizes", lambda _host, _path: {})
    monkeypatch.setattr(routes, "remote_model_dir", lambda _host, path: path)
    monkeypatch.setattr(
        routes,
        "complete_model_layout",
        lambda path: ModelLayout(
            source=path,
            fixed_weight_bytes=1,
            layer_weight_bytes=(10, 10, 10, 10),
            supports_pipeline=True,
        ),
    )

    def fake_stage(
        plan,
        *,
        model_path,
        source_host,
        destination_host,
        expected_sizes,
        destination_dir,
        parallel,
        progress,
    ):
        if not expected_sizes:
            return StagingResult(node_id=plan.node_id)
        filename = next(iter(expected_sizes))
        size = (root / filename).stat().st_size
        progress(filename, "copying", 0)
        progress(filename, "copied", size)
        return StagingResult(
            node_id=plan.node_id,
            copied=(filename,),
            bytes_copied=size,
            seconds=1.0,
        )

    monkeypatch.setattr(routes, "stage_files_from_source", fake_stage)
    nodes = [
        {"node_id": "local", "capacity_bytes": 1024**3},
        {"node_id": "studio", "capacity_bytes": 1024**3},
    ]
    with TestClient(app) as client:
        preview = client.post(
            "/admin/api/cluster/plan",
            json={"model_path": str(root), "nodes": nodes},
        )
        assert preview.status_code == 200, preview.text
        activation = {
            "model_path": str(root),
            "backend": "ring",
            "nodes": nodes,
            "hosts": [
                {
                    "node_id": "local",
                    "ssh": "127.0.0.1",
                    "ips": ["10.0.0.1"],
                },
                {
                    "node_id": "studio",
                    "ssh": "studio.local",
                    "ips": ["10.0.0.2"],
                },
            ],
            "preflight": True,
            "approved_placement": preview.json()["placement_signature"],
        }
        response = client.post(
            "/admin/api/cluster/stage",
            json={"activation": activation},
        )
        assert response.status_code == 200, response.text
        job = response.json()
        deadline = time.monotonic() + 2
        while job["status"] not in {"completed", "failed"}:
            assert time.monotonic() < deadline
            time.sleep(0.01)
            job = client.get(
                f"/admin/api/cluster/stage/{job['job_id']}"
            ).json()

    assert job["status"] == "completed", json.dumps(job, indent=2)
    assert job["ready"] is True
    assert job["nodes"]["local"]["status"] == "ready"
    assert job["nodes"]["studio"]["status"] == "ready"
    assert job["nodes"]["studio"]["files_completed"] == 1


# ---------------------------------------------------------------------------
# Disk: refuse before moving bytes, not after filling the volume.
# ---------------------------------------------------------------------------


def _plan_with_missing(files):
    from omlx.cluster.staging import StagingPlan

    missing_bytes = sum(files.values())
    names = tuple(files)
    return StagingPlan(
        node_id="studio",
        start_layer=0,
        end_layer=1,
        required=names,
        missing=names,
        required_bytes=missing_bytes,
        missing_bytes=missing_bytes,
        total_model_bytes=missing_bytes,
    )


def test_a_transfer_that_would_not_fit_is_refused_before_copying(tmp_path):
    from omlx.cluster.staging import InsufficientDiskError, check_disk_for_staging

    plan = _plan_with_missing({"a.safetensors": 1})
    object.__setattr__(plan, "missing_bytes", 100 * 1024**3)

    with pytest.raises(InsufficientDiskError) as excinfo:
        check_disk_for_staging(plan, tmp_path, free_bytes=50 * 1024**3)

    message = str(excinfo.value)
    assert "100.0 GiB" in message and "50.0 GiB" in message
    assert "Free up" in message, "must say how much short it is"


def test_system_headroom_is_kept_free(tmp_path):
    """Filling a volume to the brim destabilises macOS, not just the copy."""

    from omlx.cluster.staging import InsufficientDiskError, check_disk_for_staging

    plan = _plan_with_missing({"a.safetensors": 1})
    object.__setattr__(plan, "missing_bytes", 95 * 1024**3)

    # 100 GiB free, 95 GiB needed: fits arithmetically, refused for headroom.
    with pytest.raises(InsufficientDiskError):
        check_disk_for_staging(plan, tmp_path, free_bytes=100 * 1024**3)

    # Explicitly waiving headroom lets it through.
    assert check_disk_for_staging(
        plan, tmp_path, free_bytes=100 * 1024**3, headroom_bytes=0
    )


def test_a_fitting_transfer_is_allowed(tmp_path):
    from omlx.cluster.staging import check_disk_for_staging

    plan = _plan_with_missing({"a.safetensors": 1})
    object.__setattr__(plan, "missing_bytes", 10 * 1024**3)
    assert check_disk_for_staging(plan, tmp_path, free_bytes=200 * 1024**3)


def test_an_unmeasurable_filesystem_does_not_block(tmp_path):
    """Matches the memory guard: never block on a number we could not read."""

    from omlx.cluster.staging import check_disk_for_staging

    plan = _plan_with_missing({"a.safetensors": 1})
    object.__setattr__(plan, "missing_bytes", 500 * 1024**3)
    assert check_disk_for_staging(plan, tmp_path, free_bytes=0) == 0


def test_free_space_is_readable_for_a_path_that_does_not_exist_yet(tmp_path):
    """The destination directory is created by staging, so it may not exist."""

    from omlx.cluster.staging import free_disk_bytes

    assert free_disk_bytes(tmp_path / "not" / "created" / "yet") > 0
