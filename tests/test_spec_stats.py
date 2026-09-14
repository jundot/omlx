# SPDX-License-Identifier: Apache-2.0
"""Per-model speculation accumulator (plan v3, F3.2)."""

from types import SimpleNamespace

from omlx.patches.mlx_lm_mtp import spec_stats


def _stats(accepts, cycles, ngram=0, miss=0):
    return SimpleNamespace(
        init_emits=2, draft_emits=accepts, bonus_emits=0, verify_emits=cycles,
        accepts=accepts, cycles=cycles, depth_drafted=[cycles],
        ngram_cycles=ngram, ngram_misses=miss,
    )


def test_two_models_keep_separate_totals():
    spec_stats.reset()
    model_a, model_b = object(), object()
    ka = spec_stats.model_identity_keys(model_a)
    kb = spec_stats.model_identity_keys(model_b)
    spec_stats.record(_stats(accepts=10, cycles=5), "stop", keys=ka)
    spec_stats.record(_stats(accepts=2, cycles=8), "stop", keys=kb)

    a = spec_stats.get_speculation_stats(ka)
    b = spec_stats.get_speculation_stats(kb)
    assert a["totals"]["accepted_draft_tokens"] == 10
    assert b["totals"]["accepted_draft_tokens"] == 2
    assert a["totals"]["requests"] == 1 and b["totals"]["requests"] == 1

    # sem candidatos: agrega tudo (comportamento v2)
    both = spec_stats.get_speculation_stats()
    assert both["totals"]["accepted_draft_tokens"] == 12
    spec_stats.reset()


def test_wrapper_and_inner_model_share_an_entry():
    spec_stats.reset()
    inner = object()
    wrapper = SimpleNamespace(language_model=inner)
    spec_stats.record(
        _stats(accepts=4, cycles=3), "stop",
        keys=spec_stats.model_identity_keys(wrapper),
    )
    # o engine pode enxergar so o objeto interno e ainda achar a entrada
    got = spec_stats.get_speculation_stats({id(inner)})
    assert got is not None and got["totals"]["accepted_draft_tokens"] == 4
    spec_stats.reset()
