# SPDX-License-Identifier: Apache-2.0
"""Tests for the hardware-aware model recommendations."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import omlx.settings
from omlx.admin import model_recommender as rec
from omlx.admin.model_recommender import Build, Family, MemoryInputs

GiB = 1024**3


def make_family(
    name,
    total_b,
    builds,
    *,
    active_b=None,
    kv=64 * 1024,
    fixed=0,
    table=0,
    model_type="qwen3_5",
    downloads=100,
):
    return Family(
        name=name,
        model_type=model_type,
        total_params=int(total_b * 1e9),
        active_params=int((active_b or total_b) * 1e9),
        kv_bytes_per_token=kv,
        kv_fixed_bytes=fixed,
        max_context=262144,
        vision=True,
        audio=False,
        downloads=downloads,
        builds=tuple(
            Build(
                f"Jundot/{name}-oQ{level:g}e-mtp",
                float(level),
                False,
                True,
                int(gb * 1e9),
                table,
            )
            for level, gb in builds.items()
        ),
    )


# Sizes, parameter counts and downloads of the published builds.
QWEN_27B = make_family(
    "Qwen3.8-27B",
    27.8,
    {2.5: 12.65, 3: 13.79, 4: 16.97, 6: 23.72, 8: 30.0},
    downloads=15637,
)
GEMMA_31B = make_family(
    "gemma-4-31B-it",
    31.7,
    {3: 16.09, 3.5: 17.26, 4: 19.94, 6: 27.65, 8: 34.70},
    kv=80 * 1024,
    fixed=int(0.78 * GiB),
    model_type="gemma4",
    downloads=990,
)
QWEN_35B_A3B = make_family(
    "Qwen3.6-35B-A3B",
    36.0,
    {4: 21.61, 6: 30.12, 8: 38.6},
    active_b=4.7,
    kv=20 * 1024,
    model_type="qwen3_5_moe",
    downloads=3117,
)
FLASH_NEXT = make_family(
    "Qwen3.8-Flash-Next",
    125.8,
    {2.5: 82.9, 3.5: 96.23, 4: 106.29, 6: 150.55},
    active_b=7.4,
    kv=24 * 1024,
    table=int(32.0e9),
    model_type="qwen4_exp",
    downloads=20830,
)
GLM = make_family(
    "GLM-5.3-Flash",
    313.9,
    {2: 110.3, 2.5: 127.82, 4: 185.93},
    active_b=17.9,
    kv=11 * 1024,
    model_type="glm5_next",
    downloads=2467,
)
CATALOG = [QWEN_27B, GEMMA_31B, QWEN_35B_A3B, FLASH_NEXT, GLM]


def mac(monkeypatch, ram_gib, *, metal_gib=None, tier="balanced", guard=True):
    """MemoryInputs for a Mac whose Metal cap is macOS's default unless given."""
    ram = ram_gib * GiB
    monkeypatch.setattr(omlx.settings, "get_system_memory", lambda: ram)
    if metal_gib is None:
        metal_gib = ram_gib * (2 / 3 if ram_gib <= 32 else 0.75)
    return MemoryInputs(ram, tier, guard, 0, int(metal_gib * GiB))


def sections_for(monkeypatch, ram_gib, families=CATALOG, bandwidth=546, **kwargs):
    scenarios = rec.memory_scenarios(mac(monkeypatch, ram_gib, **kwargs))
    return rec.build_sections(families, scenarios, bandwidth, prefer_fp16=False)


def names(rows):
    return [row["repo_id"].split("/")[1] for row in rows]


class TestMemoryBudgets:
    def test_metal_cap_binds_until_the_wired_limit_is_raised(self, monkeypatch):
        scenarios = rec.memory_scenarios(mac(monkeypatch, 48, metal_gib=36))

        assert scenarios["now"].ceiling == 36 * GiB
        # The aggressive tier only shrinks the macOS reserve, which is not
        # the binding limit here.
        assert scenarios["memory_guard"].ceiling == 36 * GiB
        # The suggested sysctl value is whole MiB.
        balanced_static = 48 * GiB - int(0.08 * 48 * GiB)
        assert scenarios["wired_limit"].ceiling == balanced_static // 2**20 * 2**20
        # Raised wired limit keeps 5% of RAM for the OS.
        assert scenarios["both"].ceiling == (48 * GiB - 48 * GiB // 20) // 2**20 * 2**20
        now = scenarios["now"]
        assert now.usable == int(now.ceiling * 0.92) - 512 * 1024**2

    def test_custom_ceiling_caps_models_until_aggressive(self, monkeypatch):
        memory = mac(monkeypatch, 512, metal_gib=486)
        memory = MemoryInputs(
            memory.total_ram, "custom", True, 28 * GiB, memory.metal_cap
        )

        scenarios = rec.memory_scenarios(memory)

        assert scenarios["now"].ceiling == 28 * GiB
        assert scenarios["memory_guard"].ceiling == 486 * GiB

    def test_disabled_guard_ignores_the_metal_cap(self, monkeypatch):
        scenarios = rec.memory_scenarios(
            mac(monkeypatch, 48, metal_gib=36, guard=False)
        )
        assert scenarios["now"].ceiling == 48 * GiB - int(0.08 * 48 * GiB)


class TestEstimates:
    def test_ngram_table_moves_to_ssd_only_when_it_has_to(self, monkeypatch):
        build = next(b for b in FLASH_NEXT.builds if b.level == 4.0)

        small = rec.memory_scenarios(mac(monkeypatch, 128))["now"]
        on_ssd = rec.evaluate(FLASH_NEXT, build, small, 546)
        assert on_ssd is not None and on_ssd.table_on_ssd
        assert on_ssd.resident_bytes == int(
            (build.file_bytes - build.table_bytes) * 1.05
        )

        large = rec.memory_scenarios(mac(monkeypatch, 256, metal_gib=240))["now"]
        resident = rec.evaluate(FLASH_NEXT, build, large, 546)
        assert resident is not None and not resident.table_on_ssd
        assert resident.tokens_per_second > on_ssd.tokens_per_second * 2

    def test_build_that_cannot_hold_8k_tokens_is_dropped(self):
        build = next(b for b in QWEN_27B.builds if b.level == 4.0)
        resident = int(build.file_bytes * 1.05)

        def budget(tokens):
            usable = resident + tokens * QWEN_27B.kv_bytes_per_token
            return rec.Budget("balanced", 0, 64 * GiB, 0.92, usable)

        assert rec.evaluate(QWEN_27B, build, budget(4096), 273) is None
        fit = rec.evaluate(QWEN_27B, build, budget(16384), 273)
        assert fit is not None and fit.max_context == 16384 and not fit.comfortable

    def test_kv_cost_counts_global_layers_and_caps_sliding_ones(self):
        config = {
            "num_hidden_layers": 6,
            "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
            "num_key_value_heads": 8,
            "head_dim": 256,
            "num_global_key_value_heads": 2,
            "global_head_dim": 512,
            "sliding_window": 1024,
        }
        per_token, fixed = rec.kv_cache_cost(config)
        assert per_token == 2 * 512 * 4
        assert fixed == 5 * 1024 * 8 * 256 * 4

        shared = {**config, "num_kv_shared_layers": 2}
        assert rec.kv_cache_cost(shared) == (0, 4 * 1024 * 8 * 256 * 4)

        hybrid = {
            "text_config": {
                "num_hidden_layers": 8,
                "full_attention_interval": 4,
                "num_key_value_heads": 4,
                "head_dim": 256,
            }
        }
        assert rec.kv_cache_cost(hybrid) == (2 * 4 * 256 * 4, 0)

    def test_active_params_drop_unrouted_experts(self):
        config = {
            "num_hidden_layers": 4,
            "first_k_dense_replace": 1,
            "hidden_size": 100,
            "moe_intermediate_size": 10,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
        }
        routed = 3 * 8 * 3 * 100 * 10
        assert rec.active_params(config, 100_000) == 100_000 - routed * 3 // 4
        assert rec.active_params({"num_hidden_layers": 4}, 100_000) == 100_000


class TestSections:
    def test_similar_models_are_ordered_by_downloads(self):
        def fit(name, smarts, downloads):
            family = make_family(name, 10, {4: 5}, downloads=downloads)
            return rec.Fit(family, family.builds[0], 0, False, 65536, smarts, 10.0)

        ranked = rec.rank(
            [fit("a", 5.0, 10), fit("b", 4.9, 1000), fit("c", 4.0, 99999)]
        )
        assert [f.family.name for f in ranked] == ["b", "a", "c"]

    def test_best_match_on_a_48gb_mac(self, monkeypatch):
        sections = sections_for(monkeypatch, 48, bandwidth=273)

        assert names(sections["best"]) == [
            "Qwen3.8-27B-oQ4e-mtp",
            "gemma-4-31B-it-oQ4e-mtp",
            "Qwen3.6-35B-A3B-oQ4e-mtp",
        ]
        assert names(sections["precise"]) == [
            "Qwen3.8-27B-oQ8e-mtp",
            "gemma-4-31B-it-oQ6e-mtp",
        ]
        assert sections["bigger"] == []

    def test_smarter_and_faster_lower_bit_build_is_promoted(self, monkeypatch):
        sections = sections_for(monkeypatch, 96, bandwidth=400)

        lead = sections["best"][0]
        assert lead["repo_id"] == "Jundot/Qwen3.8-Flash-Next-oQ3.5e-mtp"
        assert lead["table_on_ssd"]

    def test_bigger_and_more_bits_are_separate_groups(self, monkeypatch):
        small = make_family("small-12B", 12.4, {4: 7.92, 8: 13.59}, kv=16 * 1024)
        big = make_family("big-27B", 27.8, {3: 13.79, 4: 16.97}, kv=16 * 1024)

        sections = sections_for(monkeypatch, 24, families=[small, big], bandwidth=273)

        assert names(sections["best"]) == ["small-12B-oQ4e-mtp"]
        assert names(sections["bigger"]) == ["big-27B-oQ3e-mtp"]
        assert names(sections["precise"]) == ["small-12B-oQ8e-mtp"]

    def test_faster_section_skips_squeezed_builds(self, monkeypatch):
        sections = sections_for(monkeypatch, 512, metal_gib=384, bandwidth=819)

        assert sections["faster"]
        assert all(row["level"] >= 4 for row in sections["faster"])
        shown = {row["name"] for key in ("best", "bigger") for row in sections[key]}
        assert not shown & {row["name"] for row in sections["faster"]}

    def test_unlock_names_the_setting_each_model_needs(self, monkeypatch):
        unlock = sections_for(monkeypatch, 128)["unlock"]

        glm = next(row for row in unlock if row["name"] == "GLM-5.3-Flash")
        assert glm["kind"] == "fits"
        assert glm["needs"] == ["wired_limit"]
        table = next(row for row in unlock if row["kind"] == "table_in_memory")
        assert table["repo_id"] == "Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"
        assert not table["table_on_ssd"]
        assert table["relative_speed"] > 2

    def test_unlock_skips_tables_that_cost_no_speed_on_ssd(self, monkeypatch):
        v41 = make_family(
            "DeepSeek-V4.1-Flash",
            566.6,
            {4: 432.0},
            active_b=31.5,
            kv=80 * 1024,
            table=int(122.9e9),
            model_type="deepseek_v41",
        )

        sections = sections_for(monkeypatch, 512, families=[v41], bandwidth=819)

        assert sections["best"][0]["table_on_ssd"]
        # No SSD slowdown is known for the Engram table, so nothing to unlock.
        assert sections["unlock"] == []

    def test_unlock_is_empty_on_the_highest_settings(self, monkeypatch):
        raised = (128 * GiB - 128 * GiB // 20) / GiB
        sections = sections_for(monkeypatch, 128, metal_gib=raised, tier="aggressive")
        assert sections["unlock"] == []

    def test_builds_for_chip_prefer_mtp_and_fp16_only_where_faster(self):
        builds = (
            Build("Jundot/x-oQ4e", 4.0, False, False, 1),
            Build("Jundot/x-oQ4e-mtp", 4.0, False, True, 1),
            Build("Jundot/x-oQ4e-fp16-mtp", 4.0, True, True, 1),
        )
        family = make_family("x", 1, {})
        family = Family(**{**family.__dict__, "builds": builds})

        assert (
            rec._builds_for_chip(family, prefer_fp16=False)[0].repo_id
            == "Jundot/x-oQ4e-mtp"
        )
        assert (
            rec._builds_for_chip(family, prefer_fp16=True)[0].repo_id
            == "Jundot/x-oQ4e-fp16-mtp"
        )


def row(
    repo_id,
    *,
    tags=(),
    pipeline="image-text-to-text",
    downloads=10,
    total=27e9,
    model_type="qwen3_5",
):
    return SimpleNamespace(
        id=repo_id,
        tags=list(tags),
        pipeline_tag=pipeline,
        downloads=downloads,
        sha="abc",
        config={"model_type": model_type},
        safetensors={"parameters": {"U32": int(total)}, "total": int(total)},
    )


class TestCatalog:
    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch):
        monkeypatch.setattr(rec, "_catalog", None)
        monkeypatch.setattr(rec, "_facts_cache", {})
        monkeypatch.setattr(rec, "_table_cache", {})
        monkeypatch.setattr(rec, "_get_hf_api", lambda: (object(), None))

    def _patch_hub(self, monkeypatch, rows, sizes, facts):
        monkeypatch.setattr(
            rec, "_list_models_stale_token_fallback", lambda api, kwargs: (rows, False)
        )
        monkeypatch.setattr(rec, "_blob_bytes_for_repos", AsyncMock(return_value=sizes))
        monkeypatch.setattr(
            rec,
            "_family_facts",
            lambda api, endpoint, repo_id, revision: facts[repo_id],
        )

    @pytest.mark.asyncio
    async def test_fetch_groups_builds_and_drops_decision_models(self, monkeypatch):
        dense = {"num_hidden_layers": 4, "num_key_value_heads": 2, "head_dim": 64}
        rows = [
            row("Jundot/Qwen3.8-27B-oQ4e-mtp", downloads=100),
            row("Jundot/Qwen3.8-27B-oQ8e-mtp", downloads=5),
            row("Jundot/clef-oQ4e", tags=["systemone"]),
            row("Jundot/embedder-oQ4e", pipeline="feature-extraction"),
            row("Jundot/not-an-oq-build"),
            row("Jundot/DeepSeek-V4.1-Flash-oQ4e-mtp", total=129e9),
        ]
        sizes = {
            "Jundot/Qwen3.8-27B-oQ4e-mtp": int(16.97e9),
            "Jundot/Qwen3.8-27B-oQ8e-mtp": 0,
            "Jundot/DeepSeek-V4.1-Flash-oQ4e-mtp": int(432e9),
        }
        report = {
            "engram": {"tensor_bytes": 122883656000},
            "remaining_weights": {"logical_parameters": 566591466194},
        }
        facts = {
            "Jundot/Qwen3.8-27B-oQ4e-mtp": {"config": dense, "model_type": "qwen3_5"},
            "Jundot/DeepSeek-V4.1-Flash-oQ4e-mtp": {
                "config": {
                    **dense,
                    "omlx_deepseek_v41": {"quantization_report": report},
                },
                "model_type": "deepseek_v41",
                "table_bytes": 122883656000,
                "total_params": 566591466194,
            },
        }
        self._patch_hub(monkeypatch, rows, sizes, facts)

        families, complete = await rec._fetch_families("Jundot")

        by_name = {f.name: f for f in families}
        assert set(by_name) == {"Qwen3.8-27B", "DeepSeek-V4.1-Flash"}
        qwen = by_name["Qwen3.8-27B"]
        assert qwen.downloads == 105
        # The build whose size did not resolve is left out until a retry.
        assert [b.level for b in qwen.builds] == [4.0]
        assert not complete
        v41 = by_name["DeepSeek-V4.1-Flash"]
        assert v41.total_params == 566591466194
        assert v41.builds[0].table_bytes == 122883656000

    @pytest.mark.asyncio
    async def test_qwen4_exp_reads_the_table_of_every_build(self, monkeypatch):
        rows = [
            row(f"Jundot/Qwen3.8-Flash-Next-oQ{level}e-mtp", model_type="qwen4_exp")
            for level in (4, 6, 8)
        ]
        rows[0].downloads = 100
        sizes = {r.id: int(150e9) for r in rows}
        facts = {
            rows[0].id: {"config": {"num_hidden_layers": 4}, "model_type": "qwen4_exp"}
        }
        self._patch_hub(monkeypatch, rows, sizes, facts)

        def table_bytes(api, endpoint, repo_id, revision):
            if "oQ6e" in repo_id:
                raise TimeoutError
            return int(57.6e9 if "oQ8e" in repo_id else 32.0e9)

        monkeypatch.setattr(rec, "_ngram_table_bytes", table_bytes)

        families, complete = await rec._fetch_families("Jundot")

        # oQ8 keeps an 8-bit table; a build whose table is unknown waits for a retry.
        tables = {b.level: b.table_bytes for b in families[0].builds}
        assert tables == {4.0: int(32.0e9), 8.0: int(57.6e9)}
        assert not complete

    @pytest.mark.asyncio
    async def test_failed_refresh_serves_the_last_catalog(self, monkeypatch):
        families = [QWEN_27B]
        monkeypatch.setattr(
            rec, "_fetch_families", AsyncMock(return_value=(families, True))
        )
        first, fetched_at, stale = await rec.load_families()
        assert first == families and not stale

        monkeypatch.setattr(rec, "_catalog", (fetched_at, time.time() - 1, families))
        monkeypatch.setattr(rec, "_fetch_families", AsyncMock(side_effect=TimeoutError))
        again, _, stale = await rec.load_families()
        assert again == families and stale
