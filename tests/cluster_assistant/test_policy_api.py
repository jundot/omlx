# SPDX-License-Identifier: Apache-2.0
"""The plan/apply half of Phase A: the policy is visible and it is applied.

The solver tests live in ``test_policy.py``; these drive the real router so the
contract the card reads is the contract the backend produces.
"""

from __future__ import annotations

from types import SimpleNamespace

from assistant_harness import (
    PLAN_URL,
    take_measurement,
    take_plan,
)

from omlx.cluster.assistant import provisioning
from omlx.cluster.planner import ModelLayout

GIB = 1024**3


def table_layout(path: str) -> ModelLayout:
    """A checkpoint with an Engram table on every layer, still small enough."""

    return ModelLayout(
        source=path,
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=(2 * GIB,) * 20,
        supports_pipeline=True,
        tensor_parallel_ssd_layer_bytes=(512 * 1024**2,) * 20,
        table_offload_kind="engram",
        max_context_tokens=32768,
    )


def test_the_plan_reports_the_selected_macs_and_the_resolved_policy(api) -> None:
    client, _, _ = api()
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    assert plan["selected_node_ids"] == plan["rank_order"]
    assert plan["excluded_nodes"] == []
    policy = plan["resource_policy"]
    assert policy["context_tokens"] == plan["context_tokens"]
    assert policy["target_context_tokens"] == plan["context_tokens"]
    assert policy["admission"] == "p50"
    assert policy["table_offload"] == "none"


def context_limited_layout(path: str) -> ModelLayout:
    """The KV cache only fits at half the requested context on the small Mac."""

    return ModelLayout(
        source=path,
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=(2 * GIB,) * 20,
        supports_pipeline=True,
        kv_bytes_per_token_per_layer=1024 * 1024,
        max_context_tokens=65536,
    )


def non_pipeline_layout(path: str) -> ModelLayout:
    """An architecture mlx-lm cannot split into pipeline stages."""

    return ModelLayout(
        source=path,
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=(2 * GIB,) * 20,
        supports_pipeline=False,
    )


def test_a_non_pipeline_model_is_refused_before_the_coordinator(api, monkeypatch) -> None:
    client, _, _ = api()
    monkeypatch.setattr(
        provisioning, "inspect_safetensors_layout", non_pipeline_layout
    )
    measurement_id = take_measurement(client)

    response = client.post(
        PLAN_URL,
        json={
            "measurement_id": measurement_id,
            "model_path": "/models/qwen",
            "context_tokens": 8192,
            "mtp": False,
        },
    )

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "no_feasible_plan"
    assert body["action"] == "choose_pipeline_model"
    assert body["feasible"] is False
    assert body["coordinator_node"] == ""


def test_the_plan_exposes_a_speed_alternative_via_plan_tradeoffs() -> None:

    from omlx.cluster.assistant.models import _plan_tradeoffs

    def policy(ctx, moe):
        return SimpleNamespace(
            context_tokens=ctx,
            moe_expert_offload=moe,
            moe_resident_fraction=0.25 if moe else 1.0,
            target_context_tokens=1_048_576,
            table_offload="ple",
            table_kind="ple",
            memory_guard_tier=None,
            microbatch_size=1,
            admission="p50",
        )

    quality = SimpleNamespace(
        policy=policy(131072, True), election=SimpleNamespace(selected=None)
    )
    speed = SimpleNamespace(
        policy=policy(65536, False),
        election=SimpleNamespace(
            selected=None,
            plan_signature="a" * 32,
            coordinator_node="a",
            rank_order=("a", "b"),
        ),
    )
    tradeoffs = _plan_tradeoffs(quality, (speed,))
    assert [item.label for item in tradeoffs] == ["speed"]
    assert tradeoffs[0].context_tokens == 65536
    assert tradeoffs[0].resource_policy.moe_expert_offload is False
    assert tradeoffs[0].plan_signature == "a" * 32
    assert tradeoffs[0].against_context_tokens == 131072
    assert tradeoffs[0].against_moe_resident_fraction == 0.25

    # No tradeoff when the main plan already keeps experts resident, and an
    # alternative that still offloads experts is not a speed win.
    resident = SimpleNamespace(
        policy=policy(65536, False), election=SimpleNamespace(selected=None)
    )
    assert _plan_tradeoffs(resident, ()) == []
    assert _plan_tradeoffs(quality, (quality,)) == []
