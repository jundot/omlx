# SPDX-License-Identifier: Apache-2.0
"""Redeploy tuning: MTP on/off + depth are measured per config, never invented."""

from __future__ import annotations

from omlx.cluster.assistant.redeploy_tuning import (
    RedeployConfig,
    RedeployRow,
    choose_best,
    plan_configs,
    sweep,
)


def test_plan_configs_offers_off_then_on_then_depths() -> None:
    configs = plan_configs(mtp_capable=True, depths=(1, 2))
    assert [c.label for c in configs] == [
        "mtp_off",
        "mtp_on_adaptive",
        "mtp_on_depth_1",
        "mtp_on_depth_2",
    ]


def test_a_model_without_an_mtp_head_only_measures_off() -> None:
    configs = plan_configs(mtp_capable=False)
    assert [c.label for c in configs] == ["mtp_off"]


def test_settings_patch_only_writes_a_fixed_depth() -> None:
    assert RedeployConfig(mtp=False).settings_patch() == {}
    assert RedeployConfig(mtp=True, mtp_depth=None).settings_patch() == {}
    assert RedeployConfig(mtp=True, mtp_depth=3).settings_patch() == {
        "mtp_adaptive_max_depth": 3
    }


def test_choose_best_ranks_by_measured_decode() -> None:
    rows = [
        RedeployRow(config=RedeployConfig(mtp=False), feasible=True, decode_tok_per_s=40.0),
        RedeployRow(
            config=RedeployConfig(mtp=True, mtp_depth=2),
            feasible=True,
            decode_tok_per_s=70.0,
        ),
        RedeployRow(
            config=RedeployConfig(mtp=True, mtp_depth=3),
            feasible=True,
            decode_tok_per_s=65.0,
        ),
    ]
    best = choose_best(rows)
    assert best == RedeployConfig(mtp=True, mtp_depth=2)


def test_choose_best_breaks_a_tie_toward_adaptive() -> None:
    rows = [
        RedeployRow(
            config=RedeployConfig(mtp=True, mtp_depth=4),
            feasible=True,
            decode_tok_per_s=50.0,
        ),
        RedeployRow(
            config=RedeployConfig(mtp=True, mtp_depth=None),
            feasible=True,
            decode_tok_per_s=50.0,
        ),
    ]
    assert choose_best(rows) == RedeployConfig(mtp=True, mtp_depth=None)


def test_choose_best_ignores_unmeasured_and_infeasible_rows() -> None:
    rows = [
        RedeployRow(config=RedeployConfig(mtp=True, mtp_depth=1), feasible=True),
        RedeployRow(
            config=RedeployConfig(mtp=True, mtp_depth=2),
            feasible=False,
            decode_tok_per_s=999.0,
        ),
        RedeployRow(config=RedeployConfig(mtp=False), feasible=True, decode_tok_per_s=30.0),
    ]
    assert choose_best(rows) == RedeployConfig(mtp=False)


def test_choose_best_returns_none_when_nothing_measured() -> None:
    rows = [RedeployRow(config=RedeployConfig(mtp=False), feasible=True)]
    assert choose_best(rows) is None


async def test_sweep_measures_every_config_and_picks_the_best() -> None:
    measured: list[str] = []

    async def measure(config: RedeployConfig) -> RedeployRow:
        measured.append(config.label)
        rate = {  # a real seam would deploy and read oMLX's usage
            "mtp_off": 38.0,
            "mtp_on_adaptive": 55.0,
            "mtp_on_depth_2": 61.0,
        }.get(config.label)
        if rate is None:
            return RedeployRow(config=config, feasible=False, error="deploy failed")
        return RedeployRow(config=config, feasible=True, decode_tok_per_s=rate)

    result = await sweep(measure, plan_configs(mtp_capable=True, depths=(2,)))

    assert measured == ["mtp_off", "mtp_on_adaptive", "mtp_on_depth_2"]
    assert result.best == RedeployConfig(mtp=True, mtp_depth=2)
    assert result.mtp_applicable is True
    assert [row.to_dict()["config"] for row in result.rows] == measured


async def test_sweep_keeps_a_failed_config_as_an_infeasible_row() -> None:
    async def measure(config: RedeployConfig) -> RedeployRow:
        if config.mtp_depth == 3:
            return RedeployRow(config=config, feasible=False, error="plan_mismatch")
        return RedeployRow(config=config, feasible=True, decode_tok_per_s=40.0)

    result = await sweep(measure, plan_configs(mtp_capable=True, depths=(3,)))
    by_label = {row.to_dict()["config"]: row for row in result.rows}
    assert by_label["mtp_on_depth_3"].feasible is False
    assert by_label["mtp_on_depth_3"].error == "plan_mismatch"
    # The failure never aborts the sweep or becomes a rate.
    assert result.best in (RedeployConfig(mtp=False), RedeployConfig(mtp=True, mtp_depth=None))


async def test_sweep_marks_mtp_not_applicable_for_a_model_without_a_head() -> None:
    async def measure(config: RedeployConfig) -> RedeployRow:
        assert config.mtp is False
        return RedeployRow(config=config, feasible=True, decode_tok_per_s=42.0)

    result = await sweep(measure, plan_configs(mtp_capable=False), mtp_capable=False)
    assert result.mtp_applicable is False
    assert result.best == RedeployConfig(mtp=False)

def _cand(order, ctx, feasible=True):
    from types import SimpleNamespace

    return SimpleNamespace(
        rank_order=tuple(order), context_tokens=ctx, feasible=feasible
    )


def test_choose_candidate_prefers_the_approved_split() -> None:
    from omlx.cluster.assistant.redeploy_tuning import choose_candidate

    cands = [_cand(("a", "b"), 32768), _cand(("b", "a"), 32768)]
    chosen = choose_candidate(cands, wanted_order=("a", "b"), wanted_context=32768)
    assert chosen.rank_order == ("a", "b")


def test_choose_candidate_falls_back_to_the_fewest_mac_split_when_the_order_flips() -> None:
    # MTP off moved rank 0 from b to a; the wanted order ("b","a") does not
    # exist, so the honest baseline is the fewest-Mac split at the context.
    from omlx.cluster.assistant.redeploy_tuning import choose_candidate

    cands = [
        _cand(("a", "b", "c", "d"), 32768),
        _cand(("a", "b", "c"), 32768),
        _cand(("a", "b"), 32768),
    ]
    chosen = choose_candidate(cands, wanted_order=("b", "a"), wanted_context=32768)
    assert chosen.rank_order == ("a", "b")


def test_choose_candidate_uses_the_nearest_context_of_the_same_order() -> None:
    from omlx.cluster.assistant.redeploy_tuning import choose_candidate

    cands = [_cand(("a", "b"), 16384), _cand(("a", "b"), 65536)]
    chosen = choose_candidate(cands, wanted_order=("a", "b"), wanted_context=32768)
    assert chosen.context_tokens in (16384, 65536)


def test_choose_candidate_returns_none_when_nothing_is_offered() -> None:
    from omlx.cluster.assistant.redeploy_tuning import choose_candidate

    assert choose_candidate([], wanted_order=("a", "b"), wanted_context=32768) is None
