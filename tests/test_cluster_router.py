# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the cluster router's resource-aware pick() logic.

These drive ClusterRouter.pick()/load()/score() directly with hand-set
snapshots -- no network, no running backends. They lock in the weighted
load balancing and the soft memory gate added on top of the MVP router.
"""
from __future__ import annotations

import json

import pytest

from omlx.cluster.router import Backend, ClusterRouter, Config, load_config


def _router(backends, **cfg_kw):
    cfg = Config(host="0.0.0.0", port=9000, router_api_key=None,
                 backends=backends, **cfg_kw)
    return ClusterRouter(cfg)


def _two(w2=1.0, w5=1.0, **cfg_kw):
    return _router(
        [Backend("m2", "http://m2:8000", "", weight=w2),
         Backend("m5", "http://m5:8000", "", weight=w5)],
        **cfg_kw,
    )


def _set(r, name, *, resident=(), cold=(), active=0, waiting=0, inflight=0,
         mem_used=0, mem_max=0, healthy=True):
    """resident = model hosted AND loaded; cold = hosted on disk, not loaded."""
    s = r.snap[name]
    s.healthy = healthy
    cat = {m: True for m in resident}
    cat.update({m: False for m in cold})
    s.catalog = cat
    s.active = active
    s.waiting = waiting
    s.mem_used = mem_used
    s.mem_max = mem_max
    r.inflight[name] = inflight


def test_eligible_filters_unhealthy_and_missing():
    r = _two()
    _set(r, "m2", resident=["a"], healthy=True)
    _set(r, "m5", resident=["b"], healthy=False)   # unhealthy host of "b"
    assert r.eligible("a") == ["m2"]
    assert r.eligible("b") == []                    # only host is unhealthy
    assert r.pick("b") is None
    assert r.pick("zzz") is None                    # hosted nowhere


def test_resident_preferred_over_idle_cold():
    r = _two()
    # m2 busy but resident; m5 idle but only has the model cold (on disk).
    _set(r, "m2", resident=["a"], inflight=5)
    _set(r, "m5", cold=["a"], inflight=0)
    assert r.pick("a") == "m2"


def test_cold_load_uses_least_loaded_when_none_resident():
    r = _two()
    _set(r, "m2", cold=["a"], inflight=3)
    _set(r, "m5", cold=["a"], inflight=1)
    assert r.pick("a") == "m5"


def test_weighted_load_prefers_faster_machine():
    # Equal raw depth, m5 twice the weight -> half the normalized load -> wins.
    r = _two(w2=1.0, w5=2.0)
    _set(r, "m2", resident=["a"], inflight=2)
    _set(r, "m5", resident=["a"], inflight=2)
    assert r.load("m2") == 2.0
    assert r.load("m5") == 1.0
    assert r.pick("a") == "m5"


def test_weighted_split_is_proportional():
    # Disable stickiness to isolate weight: simulate concurrent dispatch
    # (in-flight accumulates, never drains) and confirm the 2x-weight backend
    # ends up carrying ~2x the share.
    r = _two(w2=1.0, w5=2.0, affinity_hysteresis=0.0)
    _set(r, "m2", resident=["a"])
    _set(r, "m5", resident=["a"])
    counts = {"m2": 0, "m5": 0}
    for _ in range(12):
        c = r.pick("a")
        counts[c] += 1
        r.inflight[c] += 1            # this request stays in flight
    assert counts["m5"] > counts["m2"]
    assert abs(counts["m5"] - 2 * counts["m2"]) <= 3


def test_weighted_split_skews_even_at_default_hysteresis():
    # Stickiness (default hysteresis=1.0) mutes but does not erase the weight
    # skew: over a concurrent burst the 2x-weight backend still takes more.
    r = _two(w2=1.0, w5=2.0)               # affinity_hysteresis defaults to 1.0
    _set(r, "m2", resident=["a"])
    _set(r, "m5", resident=["a"])
    counts = {"m2": 0, "m5": 0}
    for _ in range(12):
        c = r.pick("a")
        counts[c] += 1
        r.inflight[c] += 1
    assert counts["m5"] > counts["m2"]


def test_memory_soft_gate_deprioritizes_pressured_backend():
    r = _two(mem_soft_floor=0.05)          # default mem_penalty
    # Both resident and idle, but m2 sits at 1% headroom -> penalty applies.
    _set(r, "m2", resident=["a"], mem_used=99, mem_max=100)
    _set(r, "m5", resident=["a"], mem_used=50, mem_max=100)
    assert r._mem_penalty("m2") > 0
    assert r._mem_penalty("m5") == 0
    assert r.pick("a") == "m5"


def test_memory_gate_is_soft_never_excludes():
    # Both resident but both near-OOM: pick still returns a backend, not None.
    r = _two(mem_soft_floor=0.05)
    _set(r, "m2", resident=["a"], mem_used=99, mem_max=100)
    _set(r, "m5", resident=["a"], mem_used=98, mem_max=100)
    assert r.pick("a") in ("m2", "m5")


def test_soft_gate_yields_to_pressured_backend_under_heavy_skew():
    # Genuinely soft: a near-OOM backend is still chosen when the only healthy
    # alternative is far busier than the penalty.
    r = _two()                              # default mem_penalty (10)
    _set(r, "m2", resident=["a"], inflight=15)               # load 15, no mem data
    _set(r, "m5", resident=["a"], mem_used=100, mem_max=100)  # idle but near-OOM
    assert 0 < r._mem_penalty("m5") < 15
    assert r.pick("a") == "m5"


def test_unknown_memory_is_not_penalized():
    # A backend that reports no mem_max (0) must not be treated as max-pressured.
    r = _two()
    _set(r, "m2", resident=["a"], mem_max=0)
    assert r._mem_penalty("m2") == 0.0


def test_sticky_affinity_holds_then_breaks():
    r = _two(affinity_hysteresis=1.0)
    _set(r, "m2", resident=["a"], inflight=1)       # load 1
    _set(r, "m5", resident=["a"], inflight=0)       # load 0
    r.affinity["a"] = "m2"
    assert r.pick("a") == "m2"                       # 1 <= 0 + 1 -> sticks
    _set(r, "m2", resident=["a"], inflight=5)        # now far busier
    r.affinity["a"] = "m2"
    assert r.pick("a") == "m5"                       # 5 > 0 + 1 -> breaks


def test_pick_model_less_uses_score():
    r = _two(w2=1.0, w5=2.0)
    _set(r, "m2", resident=["a"], inflight=2)        # load 2
    _set(r, "m5", resident=["b"], inflight=2)        # load 1
    assert r.pick_model_less() == "m5"


def test_load_config_parses_weight_and_mem(tmp_path, monkeypatch):
    cfg = {
        "listen": "0.0.0.0:9100",
        "affinity_hysteresis": 1.5,
        "mem_soft_floor": 0.1,
        "mem_penalty": 50.0,
        "backends": [
            {"name": "m2", "base_url": "http://m2:8000/", "api_key": "k", "weight": 1.0},
            {"name": "m5", "base_url": "http://m5:8000", "api_key": "k2", "weight": 1.5},
        ],
    }
    p = tmp_path / "cluster.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setenv("OMLX_CLUSTER_CONFIG", str(p))
    c = load_config()
    assert c.port == 9100
    assert c.affinity_hysteresis == 1.5
    assert c.mem_soft_floor == 0.1
    assert c.mem_penalty == 50.0
    assert {b.name: b.weight for b in c.backends} == {"m2": 1.0, "m5": 1.5}
    assert c.backends[0].base_url == "http://m2:8000"   # trailing slash stripped


def test_load_config_defaults_weight_to_one(tmp_path, monkeypatch):
    cfg = {"backends": [{"name": "m2", "base_url": "http://m2:8000", "api_key": "k"}]}
    p = tmp_path / "cluster.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setenv("OMLX_CLUSTER_CONFIG", str(p))
    c = load_config()
    assert c.backends[0].weight == 1.0


def test_load_config_rejects_nonpositive_weight(tmp_path, monkeypatch):
    cfg = {"backends": [
        {"name": "m2", "base_url": "http://m2:8000", "api_key": "k", "weight": 0},
    ]}
    p = tmp_path / "cluster.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setenv("OMLX_CLUSTER_CONFIG", str(p))
    with pytest.raises(SystemExit):
        load_config()
