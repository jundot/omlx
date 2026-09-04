# SPDX-License-Identifier: Apache-2.0
"""Failures must arrive as something a person can act on."""

import pytest

from omlx.cluster.guidance import explain


def test_missing_worker_runtime_is_explained_as_online_setup():
    guidance = explain(
        "cuda-worker-1 is online, but its oMLX worker runtime is not installed yet."
    )

    assert guidance.title == "The device is online but its worker runtime is missing"
    assert "SSH and hardware discovery succeeded" in guidance.explanation


def test_unverifiable_runtime_is_not_reported_as_a_missing_one():
    """#2680: a probe that could not run is not proof the runtime is absent."""

    unverified = explain("studio is online, but oMLX worker runtime could not be verified.")
    missing = explain("studio is online, but oMLX worker runtime is not installed.")

    assert unverified.title != missing.title
    assert "could not be checked" in unverified.title
    assert "install" not in unverified.title.lower()
    assert any("open omlx once" in step.lower() for step in unverified.steps)


@pytest.mark.parametrize(
    ("message", "expected_in_title"),
    [
        (
            "This is not the plan you approved — the budgets, roles or layer split changed.",
            "plan changed",
        ),
        ("weight file is missing: model-00002-of-00058.safetensors", "shard"),
        ("Host key verification failed.", "identity changed"),
        ("ssh: connect to host studio.local port 22: Connection refused", "reach"),
        ("Permission denied (publickey).", "rejected"),
        ("ssh: Could not resolve hostname studio.local", "resolve"),
        ("SSH command timed out after 30s", "responding"),
        ("model does not support tensor parallelism (no shard method): Gemma", "split"),
        (
            "no workable split for 2 nodes: hybrid shard does not fit node mac-1",
            "context",
        ),
        ("tensor_parallel_heads (33) is not divisible by tensor_parallel_size (2)", "split"),
        ("cluster registry is not configured", "isn't set up"),
        ("ssh-keygen failed: ", "SSH key"),
    ],
)
def test_known_failures_get_specific_guidance(message, expected_in_title):
    guidance = explain(message)
    assert expected_in_title.lower() in guidance.title.lower()
    assert guidance.steps, "guidance without a next step is not guidance"
    assert guidance.explanation


def test_unknown_failures_still_get_something_actionable():
    guidance = explain("kernel panic in the flux capacitor")
    assert guidance.steps
    assert guidance.title


def test_runtime_heartbeat_is_not_misreported_as_a_version_mismatch():
    guidance = explain(
        "Studio stopped publishing its runtime heartbeat. "
        "ValueError: quantized_matmul shapes are incompatible"
    )

    assert "different versions" not in guidance.title.lower()


def test_empty_and_none_are_safe():
    for value in (None, ""):
        guidance = explain(value)
        assert guidance.title and guidance.steps


def test_guidance_serialises_for_the_dashboard():
    payload = explain("Host key verification failed.").to_dict()
    assert set(payload) == {
        "title",
        "explanation",
        "steps",
        "doc_anchor",
        "command",
        "keygen_command",
        "code",
    }
    assert isinstance(payload["steps"], list)


def test_every_rule_carries_a_stable_code():
    from omlx.cluster.guidance import _FALLBACK, _RULES

    assert _FALLBACK.code == "unknown_failure"
    codes = [guidance.code for _pattern, guidance in _RULES]
    assert all(codes)
    assert len(set(codes)) == len(codes)


def test_first_seen_host_key_has_a_copyable_terminal_fallback():
    guidance = explain(
        "peer capability probe failed for clusteruser@studio: "
        "No ED25519 host key is known for studio and you have "
        "requested strict checking. Host key verification failed."
    )

    assert "isn't trusted yet" in guidance.title
    assert guidance.doc_anchor == "pairing"
    assert guidance.command == (
        "ssh-copy-id -i ~/.ssh/omlx_cluster.pub clusteruser@studio"
    )
    assert guidance.keygen_command.startswith("ssh-keygen -t ed25519")


def test_specific_rules_win_over_general_ones():
    """A publickey failure is a rejected login, not a generic timeout."""

    assert "rejected" in explain("Permission denied (publickey).").title.lower()
    # 'not found on this peer' must not be swallowed by the version rule.
    assert "model" in explain("/models/llama was not found on this peer").title.lower()
