# SPDX-License-Identifier: Apache-2.0
"""Which models this cluster can actually run — answered by the real planner."""

from __future__ import annotations

from pathlib import Path

from omlx.cluster.catalogue import (
    ModelFit,
    assess_model,
    catalogue_for_cluster,
    largest_context_that_fits,
)
from omlx.cluster.planner import ModelLayout, NodeBudget, synthetic_model_layout

GiB = 1024**3

# Qwen3-32B shaped: 8 KV heads x 128 dims x 2 bytes x (K and V).
KV_BYTES_PER_TOKEN_PER_LAYER = 8 * 128 * 2 * 2


def _nodes(*capacities_gib, reserve_gib=4):
    return [
        NodeBudget(
            node_id=f"node{index}",
            capacity_bytes=int(capacity * GiB),
            reserve_bytes=int(reserve_gib * GiB),
            rank=index,
        )
        for index, capacity in enumerate(capacities_gib)
    ]


def _model(size_gib, layers=48, kv=KV_BYTES_PER_TOKEN_PER_LAYER):
    """A layout with a real KV rate — the thing that makes context cost."""

    total = int(size_gib * GiB)
    base, remainder = divmod(total, layers)
    return ModelLayout(
        source="synthetic",
        fixed_weight_bytes=0,
        layer_weight_bytes=tuple(
            base + (1 if index < remainder else 0) for index in range(layers)
        ),
        kv_bytes_per_token_per_layer=kv,
        supports_tensor_parallel=True,
        supports_pipeline=True,
    )


# --- The basic question ----------------------------------------------------


def test_a_model_that_fits_one_node_does_not_ask_for_two():
    fit = assess_model(_model(20), _nodes(128, 128), model_id="small")
    assert fit.fits
    assert fit.nodes_required == 1
    assert fit.strategy == "single node"


def test_a_model_too_big_for_one_node_is_pipelined_across_two():
    fit = assess_model(_model(90), _nodes(64, 64), model_id="medium")
    assert fit.fits
    assert fit.nodes_required == 2
    assert fit.strategy == "pipeline"


def test_a_model_too_big_for_the_whole_cluster_is_refused_with_the_reason():
    fit = assess_model(_model(400), _nodes(64, 64), model_id="huge")
    assert not fit.fits
    assert fit.reason, "must say why, not just no"
    assert fit.strategy == ""


def test_a_refusal_reports_the_closest_pipeline_not_the_last_tensor_attempt():
    """The GUI must show the achievable shortfall, in GiB, not raw bytes."""

    fit = assess_model(_model(300), _nodes(60, 240), model_id="close")

    assert not fit.fits
    assert fit.closest_strategy == "pipeline"
    assert fit.closest_nodes_required == 2
    assert fit.shortfall_bytes == 8 * GiB
    assert "8.0 GiB more" in fit.reason
    assert "2-Mac pipeline" in fit.reason
    assert "additional bytes" not in fit.reason


def test_the_verdict_names_the_model_and_reads_like_a_sentence():
    fit = assess_model(_model(20), _nodes(128), model_id="qwen3-27b")
    assert fit.describe().startswith("qwen3-27b: fits on one node")
    assert "tokens of context" in fit.describe()


# --- Context is the part a weights-only answer gets wrong -------------------


def test_context_is_reported_not_just_whether_the_weights_load():
    fit = assess_model(_model(20), _nodes(128), model_id="small")
    assert fit.max_context_tokens >= 2048


def test_a_tighter_cluster_supports_less_context():
    """The same model on less memory must not claim the same context."""

    roomy = assess_model(_model(40), _nodes(256), model_id="m")
    tight = assess_model(_model(40), _nodes(60), model_id="m")
    assert roomy.fits and tight.fits
    assert tight.max_context_tokens < roomy.max_context_tokens


def test_context_never_exceeds_what_the_model_declares():
    fit = assess_model(
        _model(10), _nodes(128), model_id="short-ctx", declared_context_tokens=8192
    )
    assert fit.max_context_tokens <= 8192


def test_automatic_context_uses_a_nonstandard_native_model_ceiling():
    fit = assess_model(
        _model(10),
        _nodes(256),
        model_id="odd-context",
        declared_context_tokens=200_000,
    )
    assert fit.max_context_tokens == 200_000


def test_a_memory_limited_context_is_called_out():
    fit = assess_model(
        _model(40), _nodes(60), model_id="m", declared_context_tokens=262144
    )
    assert fit.fits
    assert fit.context_is_limited
    assert "model supports 262,144" in fit.describe()


def test_a_model_whose_weights_load_but_context_cannot_still_reports_zero():
    """Loading is not fitting; a model with no room for KV is not usable."""

    # Sized so the weights fit the 48 GiB usable budget with less spare than
    # the smallest context needs (2048 tokens x 48 layers x 4 KiB = 0.375 GiB).
    assert largest_context_that_fits(_model(47.9), _nodes(52)) == 0
    # And one that does leave room, so this is a threshold and not a constant.
    assert largest_context_that_fits(_model(47.0), _nodes(52)) >= 2048


# --- Strategy choice --------------------------------------------------------


def test_pipeline_is_preferred_to_tensor_parallel_at_equal_width():
    """PP loads faster, uses less memory, and tolerates a slower link."""

    fit = assess_model(_model(90), _nodes(64, 64), tensor_parallel_ok=True)
    assert fit.pipeline_stages == 2
    assert fit.tensor_parallel_size == 1


def test_a_model_that_cannot_shard_is_never_given_tensor_parallelism():
    fit = assess_model(_model(90), _nodes(64, 64), tensor_parallel_ok=False)
    assert fit.tensor_parallel_size == 1


def test_needing_every_node_is_stated_as_a_cost():
    fit = assess_model(_model(90), _nodes(64, 64))
    assert any("cannot run if one goes away" in w for w in fit.warnings)


def test_a_single_node_fit_carries_no_such_warning():
    assert not assess_model(_model(20), _nodes(128, 128)).warnings


# --- The catalogue ----------------------------------------------------------


def test_the_biggest_runnable_model_is_listed_first(tmp_path, monkeypatch):
    """What the cluster can run at its best is the thing being looked for."""

    def _fake(path, nodes, **_):
        sizes = {"a": 10, "b": 90, "c": 400}
        size = sizes[str(path)]
        return ModelFit(
            model_id=str(path),
            weight_bytes=size * GiB,
            fits=size < 100,
            reason="",
        )

    monkeypatch.setattr("omlx.cluster.catalogue.assess_model_path", _fake)
    catalogue = catalogue_for_cluster(["a", "b", "c"], _nodes(128))
    assert [fit.model_id for fit in catalogue] == ["b", "a", "c"]
    assert catalogue[-1].fits is False


def test_an_unreadable_model_is_reported_not_raised(tmp_path):
    from omlx.cluster.catalogue import assess_model_path

    fit = assess_model_path(tmp_path / "not-a-model", _nodes(128))
    assert not fit.fits
    assert "could not read" in fit.reason


def test_a_fit_serialises_for_the_interface():
    payload = assess_model(_model(20), _nodes(128), model_id="m").to_dict()
    assert payload["fits"] is True
    assert payload["strategy"] == "single node"
    assert payload["max_context_tokens"] > 0
    assert payload["summary"].startswith("m: fits")


# --- Planning before the download is a weaker claim, and says so ------------


def test_a_model_planned_from_its_size_alone_does_not_promise_a_context():
    """A synthetic layout knows no KV shape; claiming 524k would be invented."""

    layout = synthetic_model_layout(total_weight_bytes=20 * GiB, layer_count=48)
    fit = assess_model(layout, _nodes(128), model_id="not-downloaded")

    assert fit.fits
    assert fit.max_context_tokens == 0
    assert "context length unknown" in fit.describe()
    assert any("Download it" in w for w in fit.warnings)


# --- The endpoint -----------------------------------------------------------


CATALOGUE = "/admin/api/cluster/catalogue"


def _client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from omlx.cluster.routes import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _node_payload(capacity_gib, node_id="studio"):
    return {
        "node_id": node_id,
        "capacity_bytes": int(capacity_gib * GiB),
        "reserve_bytes": int(4 * GiB),
    }


def test_the_endpoint_needs_somewhere_to_look():
    response = _client().post(
        CATALOGUE, json={"nodes": [_node_payload(128)]}
    )
    assert response.status_code == 400
    assert "model_paths or model_dir" in response.json()["detail"]


def test_an_unreadable_directory_is_a_clear_error_not_a_crash(tmp_path):
    response = _client().post(
        CATALOGUE,
        json={
            "nodes": [_node_payload(128)],
            "model_dir": str(tmp_path / "does-not-exist"),
        },
    )
    assert response.status_code == 400
    assert "could not read" in response.json()["detail"]


def test_the_endpoint_reports_a_model_it_cannot_read(tmp_path):
    (tmp_path / "broken").mkdir()
    response = _client().post(
        CATALOGUE,
        json={"nodes": [_node_payload(128)], "model_dir": str(tmp_path)},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["node_count"] == 1
    assert body["runnable_count"] == 0
    assert body["largest_runnable"] is None
    assert len(body["models"]) == 1
    assert body["models"][0]["fits"] is False
    assert "could not read" in body["models"][0]["reason"]


def test_the_endpoint_answers_for_every_model_it_was_given(monkeypatch, tmp_path):
    def _fake(path, nodes, **_):
        size = {"big": 90, "small": 10}[Path(path).name]
        return ModelFit(
            model_id=Path(path).name,
            weight_bytes=int(size * GiB),
            fits=True,
            reason="",
            nodes_required=1,
            max_context_tokens=32768,
        )

    monkeypatch.setattr("omlx.cluster.catalogue.assess_model_path", _fake)
    response = _client().post(
        CATALOGUE,
        json={
            "nodes": [_node_payload(128)],
            "model_paths": [str(tmp_path / "small"), str(tmp_path / "big")],
        },
    )
    assert response.status_code == 200

    body = response.json()
    assert [model["model_id"] for model in body["models"]] == ["big", "small"]
    assert body["largest_runnable"]["model_id"] == "big"
    assert body["runnable_count"] == 2
    assert body["cluster_capacity_bytes"] == int(128 * GiB)


def test_a_subset_of_nodes_is_renumbered_for_the_planner():
    """Ranks are positional; a node that was rank 1 must not be planned as one."""

    both = _nodes(128, 128)
    fit = assess_model(_model(20), [both[1]], model_id="m")
    assert fit.fits, fit.reason


def test_a_narrower_split_renumbers_the_nodes_it_uses():
    """A 3-node cluster running a model on one node must still plan."""

    fit = assess_model(_model(20), _nodes(128, 128, 128), model_id="m")
    assert fit.fits
    assert fit.nodes_required == 1


# --- Capability, not just memory -------------------------------------------


def test_a_model_that_cannot_be_split_is_refused_however_well_it_fits():
    """The lesson from MiniMax-M3: fitting and being splittable are different.

    It was reported as fitting across two Macs on memory alone. That cost
    61.7 GiB of staging and two launches before mlx-lm raised "The model does
    not support pipelining but a pipeline_group was provided".
    """

    layout = _model(20, kv=KV_BYTES_PER_TOKEN_PER_LAYER)
    object.__setattr__(layout, "supports_pipeline", False)
    object.__setattr__(layout, "supports_tensor_parallel", False)

    # One node is fine — it only ever needed one.
    assert assess_model(layout, _nodes(128), model_id="m").fits

    # Two nodes is not, because it cannot be split at all.
    fit = assess_model(_model(200), _nodes(128, 128), model_id="big")
    object.__setattr__(fit, "splittable", False)

    big = _model(200)
    object.__setattr__(big, "supports_pipeline", False)
    object.__setattr__(big, "supports_tensor_parallel", False)
    refused = assess_model(big, _nodes(128, 128), model_id="big")
    assert not refused.fits
    assert "neither pipelining nor tensor parallelism" in refused.reason
    assert refused.splittable is False


def test_an_unsplittable_model_that_fits_the_larger_peer_says_so():
    layout = _model(64)
    object.__setattr__(layout, "supports_pipeline", False)
    object.__setattr__(layout, "supports_tensor_parallel", False)

    fit = assess_model(
        layout,
        _nodes(60, 256),
        model_id="studio-model",
        declared_context_tokens=262144,
    )

    assert fit.fits is False
    assert fit.failure_kind == "single_node_only"
    assert fit.standalone_node_id == "node1"
    assert fit.standalone_max_context_tokens == 262144
    assert "does fit on node1 by itself" in fit.reason
    assert fit.to_dict()["standalone_node_id"] == "node1"


def test_a_pipelinable_model_is_still_offered_across_nodes():
    fit = assess_model(_model(200), _nodes(128, 128), model_id="ok")
    assert fit.fits and fit.nodes_required == 2
    assert fit.splittable is True


def test_the_interface_can_grey_out_an_unsplittable_model():
    layout = _model(200)
    object.__setattr__(layout, "supports_pipeline", False)
    object.__setattr__(layout, "supports_tensor_parallel", False)
    payload = assess_model(layout, _nodes(128, 128), model_id="m").to_dict()
    assert payload["splittable"] is False
    assert payload["fits"] is False
