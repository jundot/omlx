# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the expert-streaming autotuner's pure logic (Fase H).

No model loads, no subprocesses: scoring, watchdog decisions, ceiling math,
candidate planning, and the per-model profile application path.
"""

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "bench"))

import autotune_expert_streaming as at  # noqa: E402

from omlx.model_settings import ModelSettings  # noqa: E402


class TestBalancedScore:
    def test_reference_scores_zero(self):
        assert at.balanced_score(100.0, 1.0, 100.0, 1.0) == pytest.approx(0.0)

    def test_faster_ttft_and_throughput_score_positive(self):
        # 2x faster TTFT, 2x throughput → 0.5*1 + 0.5*1 = 1.0
        assert at.balanced_score(50.0, 2.0, 100.0, 1.0) == pytest.approx(1.0)

    def test_half_weights(self):
        # TTFT 2x better, throughput unchanged → 0.5
        assert at.balanced_score(50.0, 1.0, 100.0, 1.0) == pytest.approx(0.5)
        # TTFT unchanged, throughput 2x better → 0.5
        assert at.balanced_score(100.0, 2.0, 100.0, 1.0) == pytest.approx(0.5)

    def test_regressions_score_negative(self):
        # TTFT 2x worse (−0.5) and throughput 2x worse (−0.5)
        assert at.balanced_score(200.0, 0.5, 100.0, 1.0) == pytest.approx(-0.5)

    def test_degenerate_inputs_are_minus_inf(self):
        assert at.balanced_score(0.0, 1.0, 100.0, 1.0) == float("-inf")
        assert at.balanced_score(100.0, 0.0, 100.0, 1.0) == float("-inf")
        assert at.balanced_score(100.0, 1.0, 0.0, 1.0) == float("-inf")


class TestTrialScore:
    def _result(self, ok=True, ttft=100.0, tok=1.0, swap=0.0):
        return at.TrialResult(
            cfg=at.Knobs(), context="2k", decode=32, ceiling_gib=20.0,
            status="ok" if ok else "failed", ttft_s=ttft, tok_s=tok,
            swap_growth_gib=swap,
        )

    def test_failed_trials_excluded(self):
        assert at.trial_score(self._result(ok=False), self._result()) == float("-inf")

    def test_swap_growth_penalized(self):
        clean = at.trial_score(self._result(ttft=50.0), self._result())
        dirty = at.trial_score(self._result(ttft=50.0, swap=1.0), self._result())
        assert clean == pytest.approx(0.5)
        assert dirty == pytest.approx(0.5 - 0.25)


class TestPriorUsable:
    def test_lru_capacity_gates_prior(self):
        from types import SimpleNamespace

        from omlx.patches.expert_streaming import _prior_usable

        assert _prior_usable(SimpleNamespace(capacity=1141)) is True
        assert _prior_usable(SimpleNamespace(capacity=0)) is False
        assert _prior_usable(None) is False
        assert _prior_usable(object()) is False


class TestWatchdog:
    POLICY = at.WatchdogPolicy(floor_available_gib=5.0, max_swap_growth_gib=2.0, floor_consecutive=2)

    def test_swap_growth_kills_immediately(self):
        reason, hits = at.watchdog_eval(20.0, 2.5, 0, self.POLICY)
        assert reason == "swap-growth"
        assert hits == 0

    def test_swap_growth_at_limit_passes(self):
        reason, _ = at.watchdog_eval(20.0, 2.0, 0, self.POLICY)
        assert reason is None

    def test_floor_needs_consecutive_hits(self):
        reason, hits = at.watchdog_eval(4.0, 0.0, 0, self.POLICY)
        assert reason is None and hits == 1
        reason, hits = at.watchdog_eval(4.0, 0.0, hits, self.POLICY)
        assert reason == "available-floor" and hits == 2

    def test_recovery_resets_counter(self):
        _, hits = at.watchdog_eval(4.0, 0.0, 0, self.POLICY)
        assert hits == 1
        reason, hits = at.watchdog_eval(8.0, 0.0, hits, self.POLICY)
        assert reason is None and hits == 0


class TestCeiling:
    def test_min_of_static_metal_and_available_minus_reserve(self):
        assert at.compute_ceiling_gib(42.0, 36.0, 30.0, 10.0) == pytest.approx(20.0)
        assert at.compute_ceiling_gib(18.0, 36.0, 40.0, 10.0) == pytest.approx(18.0)

    def test_metal_cap_zero_is_ignored(self):
        assert at.compute_ceiling_gib(18.0, 0.0, 40.0, 10.0) == pytest.approx(18.0)

    def test_floor_when_available_below_reserve(self):
        # Never hand the bench a ceiling smaller than the floor (skip rules
        # handle "machine too loaded" instead).
        assert at.compute_ceiling_gib(42.0, 0.0, 8.0, 10.0) == pytest.approx(6.0)


class TestDepthPruning:
    def test_near_saturated_disk_prunes_to_middle(self):
        assert at.prune_depth_candidates(4.0, 5.0, [8, 16, 32]) == [16]

    def test_slow_random_disk_keeps_sweep(self):
        assert at.prune_depth_candidates(0.5, 5.0, [8, 16, 32]) == [8, 16, 32]

    def test_missing_probe_keeps_sweep(self):
        assert at.prune_depth_candidates(None, None, [8, 16, 32]) == [8, 16, 32]


class TestKnobs:
    def test_env_mapping(self):
        env = at.Knobs(io_depth=32, coalesce=False, readahead=False, seed=True).env()
        assert env["OMLX_EXPERT_STREAMING_QD"] == "32"
        assert env["OMLX_EXPERT_STREAMING_COALESCE"] == "0"
        assert env["OMLX_EXPERT_STREAMING_RA"] == "0"
        assert env["OMLX_EXPERT_STREAMING_SEED"] == "1"
        assert env["OMLX_EXPERT_STREAMING_CACHE_PRIOR"] == "0.0"
        assert at.Knobs(prior=2.0).env()["OMLX_EXPERT_STREAMING_CACHE_PRIOR"] == "2.0"

    def test_profile_kwargs_matches_settings_fields(self):
        kw = at.Knobs(budget_gib=2.0, topk=None).profile_kwargs()
        s = ModelSettings(**kw)
        assert s.expert_streaming_budget_gib == 2.0
        assert s.expert_streaming_io_depth == 16
        assert s.expert_streaming_topk_threshold is None

    def test_label_is_stable_and_filesystem_safe(self):
        label = at.Knobs(budget_gib=4.0, topk=0.85).label()
        assert label == "b4_qd16_c1_ra1_s1_tk0.85"

    def test_prior_knob_label_profile_and_screen(self):
        assert at.Knobs(prior=1.0).label() == "b0_qd16_c1_ra1_s1_cp1"
        kw = at.Knobs(prior=1.0).profile_kwargs()
        s = ModelSettings(**kw)
        assert s.expert_streaming_cache_prior == 1.0
        kw0 = at.Knobs().profile_kwargs()
        assert ModelSettings(**kw0).expert_streaming_cache_prior is None
        base = at.Knobs()
        off = at.screen_candidates(
            base, budgets=[0.0], depths=[16], sweep_topk=False, sweep_prior=False
        )
        assert all(knob != "prior" for knob, _ in off)
        on = at.screen_candidates(
            base, budgets=[0.0], depths=[16], sweep_topk=False,
            sweep_prior=True, priors=[0.0, 1.0],
        )
        got = [(knob, cfg.prior) for knob, cfg in on if knob == "prior"]
        assert got == [("prior", 1.0)]

    def test_prior_arms_carried_on_positive_budget(self):
        """A prior arm on a zero-budget base is refused by the runtime (no
        LRU to rank with) and re-measures the base. Prior arms must ride a
        positive budget even when the sweep's budget list is [0]."""
        base = at.Knobs()
        on = at.screen_candidates(
            base, budgets=[0.0], depths=[16], sweep_topk=False,
            sweep_prior=True, priors=[1.0, 2.0],
            loaded_est_gib=1.0, available_gib=30.0, reserve_gib=4.0,
        )
        prior_arms = [cfg for knob, cfg in on if knob == "prior"]
        assert prior_arms, "prior arms disappeared from the sweep"
        for cfg in prior_arms:
            assert cfg.budget_gib > 0.0, (
                f"prior arm {cfg.label()} carries a dead budget {cfg.budget_gib}"
            )
        # Room (30 - 4 - 1 - 2 = 23 GiB) comfortably admits 1 GiB.
        assert all(cfg.budget_gib == 1.0 for cfg in prior_arms)

    def test_prior_arms_follow_room_when_tight(self):
        """With almost no room the carry budget shrinks below 1 GiB but the
        prior arm still rides a positive budget (clamped to the room)."""
        base = at.Knobs()
        on = at.screen_candidates(
            base, budgets=[0.0], depths=[16], sweep_topk=False,
            sweep_prior=True, priors=[2.0],
            loaded_est_gib=10.0, available_gib=14.0, reserve_gib=1.0,
        )
        prior_arms = [cfg for knob, cfg in on if knob == "prior"]
        assert prior_arms
        assert all(0.0 < cfg.budget_gib <= 1.0 for cfg in prior_arms)

    def test_prior_arms_survive_zero_room_as_no_candidates(self):
        """With literally no room for any positive budget the sweep skips
        prior arms rather than sending refused trials."""
        base = at.Knobs()
        on = at.screen_candidates(
            base, budgets=[0.0], depths=[16], sweep_topk=False,
            sweep_prior=True, priors=[2.0],
            loaded_est_gib=20.0, available_gib=14.0, reserve_gib=4.0,
        )
        assert not [cfg for knob, cfg in on if knob == "prior"]

    def test_prior_arms_untouched_when_base_already_positive(self):
        """A base that already rides a positive budget keeps it — the carry
        fix must not rewrite existing arms."""
        base = at.Knobs(budget_gib=2.0)
        on = at.screen_candidates(
            base, budgets=[0.0, 2.0], depths=[16], sweep_topk=False,
            sweep_prior=True, priors=[1.0],
        )
        prior_arms = [cfg for knob, cfg in on if knob == "prior"]
        assert prior_arms
        assert all(cfg.budget_gib == 2.0 for cfg in prior_arms)


class TestScreenCandidates:
    def test_ofat_candidates_carried_on_base(self):
        base = at.Knobs()
        cands = at.screen_candidates(
            base, budgets=[0.0, 2.0, 4.0], depths=[8, 16, 32], sweep_topk=False
        )
        # budget 2 alternatives + qd 2 + coalesce 1 + ra 1 + seed 1
        assert len(cands) == 7
        for _knob, cfg in cands:
            # every candidate differs from base in exactly one knob
            diffs = [
                field
                for field in ("budget_gib", "io_depth", "coalesce", "readahead", "seed")
                if getattr(cfg, field) != getattr(base, field)
            ]
            assert len(diffs) == 1

    def test_budget_filter_by_room(self):
        cands = at.screen_candidates(
            at.Knobs(), budgets=[0.0, 2.0, 4.0], depths=[16], sweep_topk=False,
            loaded_est_gib=20.0, available_gib=30.0, reserve_gib=10.0,
        )
        budgets_swept = [cfg.budget_gib for knob, cfg in cands if knob == "budget_gib"]
        # room = 30 − 10 − 20 − 2 < 2 → only budget 0 candidates… but budget 0
        # equals the base value so the knob is skipped entirely.
        assert budgets_swept == []

    def test_topk_only_with_flag(self):
        cands = at.screen_candidates(
            at.Knobs(), budgets=[0.0], depths=[16], sweep_topk=False
        )
        assert all(knob != "topk" for knob, _ in cands)
        cands = at.screen_candidates(
            at.Knobs(), budgets=[0.0], depths=[16], sweep_topk=True
        )
        assert any(knob == "topk" for knob, _ in cands)


class TestSelectBest:
    def test_picks_best_by_fake_evaluator(self):
        base = at.Knobs(budget_gib=0.0)
        cands = at.screen_candidates(
            base, budgets=[0.0, 2.0], depths=[16], sweep_topk=False
        )
        scores = {base: 0.0}
        for _knob, cfg in cands:
            scores[cfg] = 1.0 if cfg.budget_gib == 2.0 else -1.0
        best, score = at.select_best(base, cands, lambda cfg: scores[cfg], base_score=0.0)
        assert best.budget_gib == 2.0
        assert score == pytest.approx(1.0)

    def test_base_kept_when_nothing_beats_it(self):
        base = at.Knobs()
        cands = at.screen_candidates(
            base, budgets=[0.0, 4.0], depths=[16], sweep_topk=False
        )
        best, score = at.select_best(base, cands, lambda _cfg: -1.0, base_score=0.0)
        assert best == base
        assert score == pytest.approx(0.0)


class TestBenchCommand:
    def test_command_contains_trial_parameters(self, tmp_path):
        cmd = at.bench_command(
            at.Knobs(budget_gib=2.0, topk=0.85),
            python="py",
            bench_path=Path("bench/bench_expert_streaming.py"),
            model_key="qwen",
            context="2k",
            decode=32,
            ceiling_gib=20.0,
            min_free_gb=22.0,
            out_dir=tmp_path,
        )
        joined = " ".join(cmd)
        assert "--budget 2.0" in joined
        assert "--topk 0.85" in joined
        assert "--mem-ceiling-gib 20.0" in joined
        assert "--prompt-len 2k" in joined
        assert "--out-dir" in joined
        cmd_no_topk = at.bench_command(
            at.Knobs(), python="py", bench_path=Path("b"),
            model_key="qwen", context="2k", decode=32, ceiling_gib=20.0,
            min_free_gb=22.0, out_dir=tmp_path,
        )
        assert "--topk" not in " ".join(cmd_no_topk)


class TestApplyToProfile:
    def test_apply_writes_per_model_settings(self, tmp_path, monkeypatch):
        import omlx.settings as omlx_settings

        monkeypatch.setattr(omlx_settings, "resolve_default_base_path", lambda: tmp_path)
        written = at.apply_to_profile(
            "moe-model", at.Knobs(budget_gib=4.0, io_depth=8)
        )
        assert written == tmp_path / "model_settings.json"
        mgr = __import__("omlx.model_settings", fromlist=["ModelSettingsManager"]).ModelSettingsManager(tmp_path)
        s = mgr.get_settings("moe-model")
        assert s.expert_streaming_budget_gib == 4.0
        assert s.expert_streaming_io_depth == 8
        # untouched fields keep defaults
        assert s.expert_streaming_enabled is False

    def test_apply_persists_to_disk(self, tmp_path, monkeypatch):
        import omlx.settings as omlx_settings

        monkeypatch.setattr(omlx_settings, "resolve_default_base_path", lambda: tmp_path)
        at.apply_to_profile("moe-model", at.Knobs(io_depth=32))
        data = json.loads((tmp_path / "model_settings.json").read_text())
        assert data["models"]["moe-model"]["expert_streaming_io_depth"] == 32


class TestBuildRecommendation:
    def test_structure(self):
        machine = at.MachineProfile(
            total_gib=48.0, available_gib=30.0, swap_used_gib=1.0,
            static_ceiling_gib=42.0, metal_cap_gib=36.0, tier="balanced",
            checkpoint_gib=66.0, seq_gbps=5.0, rand_gbps=1.0,
        )
        winner = at.Knobs(budget_gib=2.0)
        rec = at.build_recommendation(
            model_key="qwen", model_id="Qwen3.8-Flash-Next-oQ4e-mtp",
            machine=machine, winner=winner, winner_score=0.25, trials=[],
            screen_context="2k", validate_context="8k", applied=True,
            notes=["n1"],
        )
        assert rec["objective"] == "balanced"
        assert rec["winner"]["expert_streaming_budget_gib"] == 2.0
        assert rec["winner"]["label"] == "b2_qd16_c1_ra1_s1"
        assert rec["applied_to_profile"] is True
        assert rec["machine"]["metal_cap_gib"] == 36.0
        assert rec["notes"] == ["n1"]
        assert rec["budget_knee_gib"] is None


class TestBudgetKnee:
    def test_empty_is_none(self):
        assert at.budget_knee_gib([]) is None

    def test_all_failed_is_none(self):
        assert at.budget_knee_gib([(0.0, float("-inf")), (2.0, float("-inf"))]) is None

    def test_best_at_zero_is_zero(self):
        # The LRU adds nothing: knee 0 caps budget_auto at page-cache.
        assert at.budget_knee_gib([(0.0, 0.0), (2.0, -0.1), (4.0, -0.2)]) == 0.0

    def test_smallest_budget_at_95pct(self):
        # best 1.0 at 8 GiB; 2 GiB already reaches 0.96 -> knee 2.
        pairs = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.96), (4.0, 0.99), (8.0, 1.0)]
        assert at.budget_knee_gib(pairs) == 2.0

    def test_plateau_picks_first(self):
        pairs = [(1.0, 0.5), (2.0, 0.5), (4.0, 0.5)]
        assert at.budget_knee_gib(pairs) == 1.0

    def test_write_budget_knee(self, tmp_path):
        dest = at.write_budget_knee(tmp_path, "qwen", 2.5)
        assert dest == tmp_path / ".omlx" / "expert_budget_knee.json"
        data = json.loads(dest.read_text())
        assert data["version"] == 1
        assert data["knee_gib"] == 2.5
        assert data["model"] == "qwen"


class TestColdTierSweep:
    def test_gated_off_without_tier(self):
        """No cold_tier arms when the model has no expert_cold/ dir."""
        cands = at.screen_candidates(
            at.Knobs(), budgets=[0.0], depths=[16], sweep_topk=False,
            cold_tier_available=False, hot_fractions=[0.25, 0.5],
        )
        assert all(knob != "cold_tier" for knob, _ in cands)

    def test_arms_need_explicit_sweep_flag(self):
        """--sweep-cold-tier + tier on disk → one arm per hot_fraction.
        Quality lever (non-bit-exact): the flag alone is not enough — it
        must never sweep automatically (project policy: defaults bit-exact)."""
        cands = at.screen_candidates(
            at.Knobs(), budgets=[0.0], depths=[16], sweep_topk=False,
            cold_tier_available=True, hot_fractions=[0.25, 0.5],
            sweep_cold_tier=True,
        )
        arms = [cfg for knob, cfg in cands if knob == "cold_tier"]
        assert len(arms) == 2
        # without the explicit flag: no cold_tier arms even with the dir present
        cands2 = at.screen_candidates(
            at.Knobs(), budgets=[0.0], depths=[16], sweep_topk=False,
            cold_tier_available=True, hot_fractions=[0.25, 0.5],
        )
        assert all(knob != "cold_tier" for knob, _ in cands2)
        assert all(cfg.cold_tier == "3" for cfg in arms)
        assert {cfg.hot_fraction for cfg in arms} == {0.25, 0.5}
        # profile_kwargs carries both knobs for --apply persistence
        kw = arms[0].profile_kwargs()
        assert kw["expert_streaming_cold_tier"] == "3"
        assert kw["expert_streaming_hot_fraction"] in (0.25, 0.5)

    def test_label_and_env(self):
        """The label names the tier + fraction; env() adds nothing new."""
        cfg = at.Knobs(cold_tier="3", hot_fraction=0.5)
        assert "ct3" in cfg.label() and "hf0.5" in cfg.label()
        assert "expert_streaming_cold_tier" in cfg.profile_kwargs()
