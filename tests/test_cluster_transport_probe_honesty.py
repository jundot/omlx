# SPDX-License-Identifier: Apache-2.0
"""A transport probe that could not run must not be reported as a measurement.

``detect_transports`` caught every probe exception, discarded the reason, and
synthesised an Ethernet link for every host pair. That is not a neutral
fallback: ``transports_are_fast_enough`` reads an empty result as "unknown" and
declines to assume the worst, but reads a named Ethernet link as a measured
reason to refuse tensor parallelism. Claiming the measurement we failed to take
is the more damaging of the two.
"""

from types import SimpleNamespace

from omlx.cluster import transport as transport_module


def _fake_config(*, raises: bool):
    """Stand in for ``mlx._distributed_utils.config``."""

    def extract_connectivity(hosts, verbose=False):
        if raises:
            raise RuntimeError("ssh: Permission denied (publickey)")
        return [], {}

    return SimpleNamespace(
        Host=lambda **kwargs: SimpleNamespace(**kwargs),
        extract_connectivity=extract_connectivity,
        make_connectivity_matrix=lambda hosts, index: [],
    )


def test_thunderbolt_probe_failure_is_not_reported_as_ethernet(monkeypatch):
    """An unreachable host yields no transport claim, not a slow-link claim."""

    monkeypatch.setattr(
        transport_module, "_import_mlx_config", lambda: _fake_config(raises=True)
    )
    monkeypatch.setattr(
        transport_module,
        "_rdma_available",
        lambda hosts, ssh_prefix="": (_ for _ in ()).throw(
            RuntimeError("ssh: Permission denied (publickey)")
        ),
    )

    transports = transport_module.detect_transports(["a", "b"])

    assert transports == (), (
        "a probe that raised must not synthesise a measured Ethernet link; "
        f"got {transports!r}"
    )


def test_completed_probe_with_no_thunderbolt_still_reports_ethernet(monkeypatch):
    """The honest Ethernet case must keep working: probe ran, found no TB."""

    monkeypatch.setattr(
        transport_module, "_import_mlx_config", lambda: _fake_config(raises=False)
    )
    monkeypatch.setattr(
        transport_module, "_rdma_available", lambda hosts, ssh_prefix="": False
    )

    transports = transport_module.detect_transports(["a", "b"])

    assert transports, "a completed probe that found no Thunderbolt reports Ethernet"
    assert {t.kind for t in transports} == {"ethernet"}
