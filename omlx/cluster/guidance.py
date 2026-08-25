# SPDX-License-Identifier: Apache-2.0
"""Turn cluster failures into something a person can act on.

Distributed setup fails in a handful of predictable ways — an SSH identity changed,
a model missing on the other Mac, versions out of step, a cable in the wrong
port. The underlying errors are accurate and unreadable. This maps them to a
title, a plain explanation, and concrete next steps.

Pure string handling, so it is testable and carries no import cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Guidance:
    """A readable failure with steps that resolve it.

    ``code`` is the stable machine key structured diagnostics (the incident
    feed) record next to the redacted message, so different failure funnels
    that map to the same guidance stay groupable.
    """

    title: str
    explanation: str
    steps: tuple[str, ...] = field(default_factory=tuple)
    doc_anchor: str | None = None
    command: str | None = None
    keygen_command: str | None = None
    code: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "explanation": self.explanation,
            "steps": list(self.steps),
            "doc_anchor": self.doc_anchor,
            "command": self.command,
            "keygen_command": self.keygen_command,
            "code": self.code,
        }


_SSH_TARGET = r"(?:[A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+"
_FAILED_FOR_TARGET = re.compile(
    rf"(?:probe|preflight|connection) failed for (?P<target>{_SSH_TARGET})(?=[:\s])",
    re.I,
)
_UNKNOWN_HOST_KEY = re.compile(
    rf"no (?:ED25519|ECDSA|RSA|DSA) host key is known for (?P<host>{_SSH_TARGET})",
    re.I,
)


def _first_seen_host_guidance(message: str) -> Guidance | None:
    """Attach a safe, copyable command to OpenSSH's first-seen-key error."""

    unknown = _UNKNOWN_HOST_KEY.search(message)
    if unknown is None:
        return None
    target_match = _FAILED_FOR_TARGET.search(message)
    target = (
        target_match.group("target")
        if target_match is not None
        else unknown.group("host")
    )
    return Guidance(
        "The other Mac isn't trusted yet",
        "SSH stopped before login because this Mac has not recorded the peer's "
        "host key. oMLX can set up its managed key, or Terminal can install it "
        "in one command.",
        (
            "Recommended: open SSH setup in oMLX and exchange the generated key.",
            "Terminal alternative: run the command below on this Mac, confirm "
            "the fingerprint, and enter the other Mac account's password.",
            "If the account names differ, change the target to user@host first, "
            "then retry the peer check.",
        ),
        "pairing",
        f"ssh-copy-id -i ~/.ssh/omlx_cluster.pub {target}",
        "ssh-keygen -t ed25519 -f ~/.ssh/omlx_cluster -N '' -C omlx-cluster",
        code="host_key_unknown",
    )


# Ordered: the first pattern that matches wins, so put specific before general.
_RULES: tuple[tuple[re.Pattern[str], Guidance], ...] = (
    (
        # Must precede the "not installed" rule below: a runtime we could not
        # reach is not a runtime we know to be absent (#2680).
        re.compile(
            r"worker runtime could not be (?:verified|run|checked)",
            re.I,
        ),
        Guidance(
            "The device is online but its worker runtime could not be checked",
            "SSH and hardware discovery succeeded and oMLX was found on the "
            "device, but nothing there could load the worker, so its runtime "
            "state is unknown rather than confirmed missing.",
            (
                "Open oMLX once on the named device — starting it publishes the "
                "helper (~/.omlx/bin/omlx-cluster-python) this Mac looks for.",
                "If the device runs oMLX from a pip or source install, start its "
                "oMLX server once so the same helper is written.",
                "Then return here and press Start again; oMLX re-checks the "
                "runtime, memory, and links automatically.",
            ),
            "worker-runtime",
            code="worker_runtime_unverified",
        ),
    ),
    (
        re.compile(r"oMLX worker runtime is not installed", re.I),
        Guidance(
            "The device is online but its worker runtime is missing",
            "SSH and hardware discovery succeeded, but this device cannot join "
            "an MLX collective until the matching oMLX worker build is installed.",
            (
                "Install or update oMLX on the named device to the same build as "
                "the coordinator.",
                "Return here and press Start again; oMLX will re-check the "
                "runtime, memory, and links automatically.",
            ),
            "worker-runtime",
            code="worker_runtime_missing",
        ),
    ),
    (
        # Before the generic version rule below, which would otherwise claim
        # the oMLX release differs when it is the interpreter that does (#2695).
        re.compile(r"python local=\S+ remote=", re.I),
        Guidance(
            "The two Macs are running different Python versions",
            "MLX ships a separate build for each Python version, so ranks on "
            "different ones load different binaries and can disagree in ways "
            "that produce wrong output rather than a clean error.",
            (
                "Run both Macs from the same oMLX install shape — either the "
                "packaged app on both, or the same Python version on both.",
                "If one Mac runs oMLX from source, recreate its environment on "
                "the Python version the other Mac reports, then reinstall.",
                "Re-run Set up cluster afterwards to confirm.",
            ),
            "worker-runtime",
            code="python_version_mismatch",
        ),
    ),
    (
        re.compile(
            r"weight file is missing|model stage is incomplete|missing .*\.safetensors",
            re.I,
        ),
        Guidance(
            "A model shard still needs preparing",
            "One Mac does not yet have a weight file required by its assigned "
            "layers, so oMLX kept the cluster stopped.",
            (
                "Retry Start — oMLX resumes staging and copies only the missing "
                "rank-local files.",
                "If it repeats, confirm the complete model still exists on the "
                "Mac named in the model card, then re-download that model if "
                "its source copy is incomplete.",
            ),
            "models",
            code="model_stage_incomplete",
        ),
    ),
    (
        re.compile(
            r"not the plan you approved|"
            r"request no longer matches the approved plan|"
            r"budgets, roles or layer split changed",
            re.I,
        ),
        Guidance(
            "The cluster plan changed before launch",
            "oMLX stopped safely because the layer placement it was about to "
            "launch differed from the one it had just checked and approved.",
            (
                "Retry Start — oMLX will rebuild and approve a fresh plan from "
                "the current memory budgets.",
                "If it repeats, open Details & recovery, reset any manual model "
                "split, and retry so automatic placement owns the full plan.",
            ),
            "plan-approval",
            code="plan_changed",
        ),
    ),
    (
        re.compile(
            r"remote host identification has changed|"
            r"offending .* host key|"
            r"host key verification failed",
            re.I,
        ),
        Guidance(
            "The other Mac's saved identity changed",
            "SSH refused the connection because this address now presents a "
            "different key. oMLX records new addresses automatically but does "
            "not overwrite an existing identity.",
            (
                "Confirm the address still belongs to the expected Mac.",
                "After verifying its fingerprint on that Mac, remove only the "
                "stale entry with ssh-keygen -R, then retry pairing.",
            ),
            "pairing",
            code="host_identity_changed",
        ),
    ),
    (
        re.compile(r"no matching host key", re.I),
        Guidance(
            "The Macs could not agree on an SSH host key",
            "The peer is reachable, but its SSH server did not offer a host-key "
            "algorithm this Mac accepts.",
            (
                "Enable Remote Login again on the peer to refresh its SSH host keys.",
                "Retry pairing after both Macs are updated to a current macOS release.",
            ),
            "pairing",
            code="no_matching_host_key",
        ),
    ),
    (
        re.compile(r"permission denied|publickey", re.I),
        Guidance(
            "The other Mac rejected the login",
            "SSH connected but the peer would not accept this Mac's key.",
            (
                "Generate a key here, then paste it on the peer using Pair.",
                "Check that the SSH username in the peer address is correct.",
            ),
            "pairing",
            code="login_rejected",
        ),
    ),
    (
        re.compile(r"could not resolve|name or service not known|nodename nor servname", re.I),
        Guidance(
            "That address doesn't resolve",
            "The peer's hostname could not be looked up from this Mac.",
            (
                "Use the Bonjour peer list above rather than typing an address.",
                "If you typed it, try the .local name or the Thunderbolt IP.",
                "Confirm both Macs are on the same network or cable.",
            ),
            code="address_unresolved",
        ),
    ),
    (
        re.compile(r"connection refused|no route to host|network is unreachable", re.I),
        Guidance(
            "Can't reach the other Mac",
            "The address resolved but nothing answered.",
            (
                "Check the Mac is awake — sleeping Macs drop the link.",
                "If you are using Thunderbolt, reseat the cable and re-run Detect link.",
                "Confirm Remote Login is enabled in System Settings → General → Sharing.",
            ),
            code="peer_unreachable",
        ),
    ),
    (
        re.compile(r"timed out|timeout", re.I),
        Guidance(
            "The other Mac stopped responding",
            "The connection opened but the peer did not finish in time.",
            (
                "Check the peer is not asleep or under heavy load.",
                "Retry — a first connection over a new link is often slower.",
            ),
            code="peer_timeout",
        ),
    ),
    (
        re.compile(r"does not implement tensor-parallel|no shard method", re.I),
        Guidance(
            "This model can't be split layer-wise",
            "Tensor parallelism needs the architecture to implement sharding in "
            "MLX-LM, and most do not. The model can still be split into pipeline "
            "stages across your Macs.",
            (
                "Continue — the cluster will use pipeline stages instead.",
                "For tensor parallelism, pick a Llama, Qwen, Mistral or DeepSeek "
                "family model.",
            ),
            "tensor-parallel",
            code="no_tensor_parallel_support",
        ),
    ),
    (
        re.compile(r"no workable split|does not fit node|do not fit", re.I),
        Guidance(
            "This model and context do not fit the current memory limits",
            "oMLX could not find a per-Mac split that holds both the model "
            "weights and the selected context cache.",
            (
                "Choose the next lower context option.",
                "Allow more memory on the Mac that is short, if it has headroom.",
                "Choose a smaller model or add another Mac.",
            ),
            code="no_workable_split",
        ),
    ),
    (
        re.compile(r"not divisible by tensor_parallel_size", re.I),
        Guidance(
            "This model can't be split that many ways",
            "Tensor parallelism divides the attention heads between Macs, and "
            "this model's head count does not divide evenly.",
            (
                "Let the automatic setup choose the split — it only offers "
                "degrees that divide evenly.",
            ),
            "tensor-parallel",
            code="heads_not_divisible",
        ),
    ),
    (
        re.compile(r"world size .* not divisible|is not a multiple", re.I),
        Guidance(
            "That number of Macs doesn't divide evenly",
            "The cluster is arranged as pipeline stages of equal tensor-parallel "
            "groups, so the node count must be a multiple of the split.",
            ("Use automatic setup, or pick a split that divides your node count.",),
            code="world_size_not_divisible",
        ),
    ),
    (
        re.compile(
            r"(?:mlx|omlx|python|runtime)?\s*version(?:s)?\s+"
            r"(?:do not match|mismatch|differ)|"
            r"(?:version|versions).*mismatch|mismatch.*(?:version|versions)",
            re.I,
        ),
        Guidance(
            "The two Macs are running different versions",
            "Ranks must run identical MLX and oMLX versions — a mismatch "
            "produces subtly wrong results rather than a clean error.",
            (
                "Update both Macs to the same oMLX release.",
                "Re-run Set up cluster afterwards to confirm.",
            ),
            code="version_mismatch",
        ),
    ),
    (
        re.compile(r"model.*not found|not found on this peer|missing the model", re.I),
        Guidance(
            "The other Mac doesn't have this model",
            "Every node loads its own shard from a local copy, so the model must "
            "exist at the same path on each Mac.",
            (
                "Download the same model on the peer Mac.",
                "Make sure the paths match exactly on both.",
            ),
            "models",
            code="model_missing_on_peer",
        ),
    ),
    (
        re.compile(r"registry is not configured", re.I),
        Guidance(
            "Clustering isn't set up on this Mac yet",
            "The cluster registry has not been initialised for this install.",
            ("Restart oMLX. If it persists, check the server logs for start-up errors.",),
            code="registry_not_configured",
        ),
    ),
    (
        re.compile(r"ssh-keygen (failed|is unavailable)", re.I),
        Guidance(
            "Couldn't create an SSH key",
            "The system's ssh-keygen could not run.",
            (
                "Confirm /usr/bin/ssh-keygen exists.",
                "If you manage your own keys, pair using an existing key instead.",
            ),
            code="ssh_keygen_failed",
        ),
    ),
)

_FALLBACK = Guidance(
    "Something went wrong setting up the cluster",
    "The step did not complete. The technical detail is shown below.",
    (
        "Retry — transient link and SSH failures are common on a first connection.",
        "If it repeats, run Detect link to check the connection between the Macs.",
    ),
    code="unknown_failure",
)


def explain(message: str | None) -> Guidance:
    """Map an error message to guidance, falling back to a generic-but-useful one.

    Never raises and never returns None: a failure the user cannot read is worse
    than a slightly generic one.
    """

    if not message:
        return _FALLBACK
    first_seen = _first_seen_host_guidance(message)
    if first_seen is not None:
        return first_seen
    for pattern, guidance in _RULES:
        if pattern.search(message):
            return guidance
    return _FALLBACK
