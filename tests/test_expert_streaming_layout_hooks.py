"""MoE layout resolution and model-family hook contracts.

Covers:
  * header-derived MoE geometry on the streaming estimate;
  * _resolve_moe_dims: config -> estimate -> fail loud (no guessing);
  * the model_hooks family registry (stream-eval targets, top-k
    applicability, weighted-sum kernel injection);
  * SlotBookkeeping semantics shared by _ExpertSlots and ExpertCache;
  * the admission estimator delegating supported types to the streaming
    estimate instead of the legacy regex mirror;
  * the legacy chunk scan's single host sync (route-boundary slices).
"""
import json

import numpy as np
import pytest
from streaming_fixtures import write_safetensors


def _moe_dir(tmp_path, *, n_experts=4, fused=False, model_type="qwen3_moe",
             moe=8, hidden=16, layers=1):
    """Fabricate a minimal supported-type checkpoint with real shapes."""
    rng = np.random.default_rng(3)
    tensors = {}
    projs = ("gate_up_proj", "down_proj") if fused else (
        "gate_proj", "up_proj", "down_proj",
    )
    for li in range(layers):
        for proj in projs:
            if proj == "gate_up_proj":
                shape = (n_experts, 2 * moe, hidden)
            elif proj == "down_proj":
                shape = (n_experts, hidden, moe)
            else:
                shape = (n_experts, moe, hidden)
            key = f"language_model.layers.{li}.mlp.switch_mlp.{proj}.weight"
            tensors[key] = (rng.standard_normal(shape).astype(np.float32), "F32")
    shard = tmp_path / "model.safetensors"
    write_safetensors(shard, tensors)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "num_experts": n_experts,
                "num_hidden_layers": layers,
                "num_experts_per_tok": 2,
            }
        )
    )
    return tensors


class TestHeaderDerivedDims:
    def test_split_layout(self, tmp_path):
        from omlx.patches.expert_streaming.residency import (
            expert_streaming_estimate,
        )

        _moe_dir(tmp_path, n_experts=4, moe=8, hidden=16)
        est = expert_streaming_estimate(tmp_path)
        assert est.supported
        assert est.moe_intermediate_size == 8
        assert est.hidden_size == 16  # header fallback (config is silent)
        assert est.expert_fused is False
        assert est.experts_per_layer == 4

    def test_fused_layout(self, tmp_path):
        from omlx.patches.expert_streaming.residency import (
            expert_streaming_estimate,
        )

        _moe_dir(tmp_path, n_experts=4, fused=True, moe=8, hidden=16)
        est = expert_streaming_estimate(tmp_path)
        assert est.supported
        assert est.expert_fused is True
        assert est.moe_intermediate_size == 8  # fused out dim halved
        assert est.hidden_size == 16

    def test_config_hidden_wins(self, tmp_path):
        from omlx.patches.expert_streaming.residency import (
            expert_streaming_estimate,
        )

        _moe_dir(tmp_path)
        cfg = json.loads((tmp_path / "config.json").read_text())
        cfg["hidden_size"] = 4096
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        est = expert_streaming_estimate(tmp_path)
        assert est.hidden_size == 4096  # config beats the header shape


class TestResolveMoeDims:
    def test_config_first(self):
        from omlx.patches.expert_streaming import _resolve_moe_dims

        assert _resolve_moe_dims([{"hidden_size": 32, "moe_intermediate_size": 8}]) == (
            32,
            8,
        )

    def test_estimate_fallback(self):
        from omlx.patches.expert_streaming import _resolve_moe_dims

        class _Est:
            hidden_size = 64
            moe_intermediate_size = 12

        # An estimate object supplies the header-derived dims when the
        # config is silent.
        assert _resolve_moe_dims([{}], _Est()) == (64, 12)

    def test_fail_loud(self):
        from omlx.patches.expert_streaming import _resolve_moe_dims

        with pytest.raises(ValueError, match="could not resolve"):
            _resolve_moe_dims([{}], None)


class TestModelHooks:
    def test_qwen_family(self):
        from omlx.patches.expert_streaming.model_hooks import hooks_for

        h = hooks_for("qwen4_exp")
        assert h.topk_supported
        assert h.apply_topk is not None
        assert len(h.stream_eval_targets) == 2
        # glm5_next inline decoder: no wrapper targets, kernel registered
        g = hooks_for("glm5_next")
        assert g.topk_supported
        assert g.stream_eval_targets == ()
        assert g.weighted_sum_kernel is not None

    def test_unlisted_family_gets_defaults(self):
        from omlx.patches.expert_streaming.model_hooks import (
            DEFAULT_PREFIX_TEMPLATES,
            hooks_for,
        )

        h = hooks_for("llama")  # no MoE — never registered
        assert h.stream_eval_targets == ()
        assert not h.topk_supported
        assert h.apply_topk is None
        assert h.weighted_sum_kernel is None
        assert h.moe_attr_chain == ("mlp", "ffn")
        assert h.prefix_templates is None  # → DEFAULT_PREFIX_TEMPLATES
        assert h.mtp_owner_chain == ("language_model", "model")
        assert h.verify_scope is None
        assert DEFAULT_PREFIX_TEMPLATES  # non-empty shared default

    def test_ffn_families_get_ffn_first_chain(self):
        from omlx.patches.expert_streaming.model_hooks import hooks_for

        for mt in ("deepseek_v4", "deepseek_v32", "glm_moe_dsa", "deepseek_v41"):
            h = hooks_for(mt)
            assert h.moe_attr_chain[0] == "ffn"
            assert h.prefix_templates[0].endswith("ffn.switch_mlp")

    def test_v41_verify_scope_resolves(self):
        from omlx.patches.expert_streaming.model_hooks import (
            resolve_verify_scope,
        )

        scope = resolve_verify_scope("deepseek_v41")
        assert callable(scope)  # context manager factory
        assert resolve_verify_scope("llama") is None


class _Box:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestFindMoeContainer:

    def _layer(self, layout="mlp", nested=False):
        sm = _Box(switch_mlp=object())
        if nested:
            return _Box(block=_Box(**{layout: sm}))
        return _Box(**{layout: sm})

    def test_direct_chain(self):
        from omlx.patches.expert_streaming.model_hooks import (
            find_moe_container,
        )

        layer = self._layer("ffn")
        assert find_moe_container(layer, ("ffn", "mlp")) is layer.ffn
        # mlp-first chain still finds it — order is preference only
        assert find_moe_container(layer, ("mlp", "ffn")) is layer.ffn

    def test_requires_switch_mlp(self):
        from omlx.patches.expert_streaming.model_hooks import (
            find_moe_container,
        )

        # dense mlp without switch_mlp + MoE ffn: the chain must skip the
        # dense container, not stop at it.
        layer = _Box(mlp=object(), ffn=_Box(switch_mlp=1))
        assert find_moe_container(layer, ("mlp", "ffn")) is layer.ffn

    def test_block_nesting(self):
        from omlx.patches.expert_streaming.model_hooks import (
            find_moe_container,
            find_moe_owner,
        )

        layer = self._layer("mlp", nested=True)
        moe = find_moe_container(layer, ("mlp", "ffn"))
        assert moe is layer.block.mlp
        assert find_moe_owner(layer, moe) is layer.block
        assert find_moe_owner(layer, self._layer("mlp").mlp) is not None

    def test_no_match_returns_none(self):
        from omlx.patches.expert_streaming.model_hooks import (
            find_moe_container,
        )

        assert find_moe_container(_Box(mlp=object()), ("mlp",)) is None
        assert find_moe_container(None, ("mlp", "ffn")) is None


class TestFindMtpStages:

    def test_finds_mtp_on_owner_child(self):
        from omlx.patches.expert_streaming.model_hooks import find_mtp_stages

        lm = _Box(mtp=[_Box(), _Box()])
        layers_owner = _Box(language_model=lm)
        model = _Box()
        stages = find_mtp_stages(
            (layers_owner, model), ("language_model", "model")
        )
        assert stages is lm.mtp

    def test_prefers_first_nonempty(self):
        from omlx.patches.expert_streaming.model_hooks import find_mtp_stages

        a = _Box(mtp=[_Box()])
        b = _Box(mtp=[_Box(), _Box()])
        assert find_mtp_stages((a, b), ("language_model",)) is a.mtp
        assert find_mtp_stages((_Box(), b), ("model",)) is b.mtp
        assert find_mtp_stages((_Box(), None), ("model",)) is None

    def test_weighted_sum_kernel_resolves(self):
        from omlx.patches.expert_streaming.model_hooks import (
            resolve_weighted_sum_kernel,
        )

        kern = resolve_weighted_sum_kernel("glm5_next")
        # The wrapper function exists even when the native ext is absent.
        assert callable(kern)
        assert resolve_weighted_sum_kernel("qwen3_moe") is None

    def test_topk_applicable_reads_registry(self):
        from omlx.patches.expert_streaming.adaptive_topk import is_topk_applicable

        assert is_topk_applicable("qwen4_exp")
        assert is_topk_applicable("glm5_next")
        assert not is_topk_applicable("qwen3_moe")


class TestSlotBookkeeping:
    def test_acquire_commit_lru_order(self):
        from omlx.patches.expert_streaming.slot_cache import SlotBookkeeping

        b = SlotBookkeeping(2)
        assert b.cap == 2 and b.rooms == 2
        # free is LIFO from the end (same order the caches always used)
        slot, victim, grow = b.acquire(())
        assert (slot, victim, grow) == (1, None, False)
        b.commit(5, slot)
        slot, victim, grow = b.acquire({5})
        assert (slot, victim, grow) == (0, None, False)
        b.commit(7, slot)
        # Full: evict the LRU-oldest expert not in needed
        slot, victim, grow = b.acquire({7})
        assert victim == 5 and slot == 1 and not grow
        assert b.evictions == 1
        b.commit(9, slot)
        assert list(b.slot_of) == [7, 9]

    def test_needs_grow_when_all_resident_needed(self):
        from omlx.patches.expert_streaming.slot_cache import SlotBookkeeping

        b = SlotBookkeeping(2)
        b.commit(1, b.acquire(())[0])
        b.commit(2, b.acquire({1})[0])
        slot, victim, grow = b.acquire({1, 2})
        assert (slot, victim, grow) == (2, None, True)
        b.grew_to(3)
        assert b.free == [2] and b.rooms == 3

    def test_rollback_restores_unstarted_victims(self):
        from omlx.patches.expert_streaming.slot_cache import SlotBookkeeping

        b = SlotBookkeeping(2)
        b.commit(1, b.acquire(())[0])
        b.commit(2, b.acquire({1})[0])
        # Batch of two misses: first evicts 1, second evicts 2.
        f1 = (3, *b.acquire({3, 4})[:2])  # row 1, victim 1
        f2 = (4, *b.acquire({3, 4})[:2])  # row 0, victim 2
        fetch_list = [(e, s, v) for e, s, v in (f1, f2)]
        # Commit entry 0; entry 1's commit starts but dies mid-write.
        b.commit(3, fetch_list[0][1])
        b.rollback(fetch_list, committed_through=0, pass3_started=True)
        # Entry 1 interrupted mid-commit: row freed, victim 2 stays evicted.
        assert b.free == [0]
        assert 3 in b.slot_of and 2 not in b.slot_of
        assert b.evictions == 2

    def test_rollback_unstarted_restores_victim(self):
        from omlx.patches.expert_streaming.slot_cache import SlotBookkeeping

        b = SlotBookkeeping(2)
        b.commit(1, b.acquire(())[0])
        b.commit(2, b.acquire({1})[0])
        f1 = (3, *b.acquire({3, 4})[:2])
        f2 = (4, *b.acquire({3, 4})[:2])
        fetch_list = [(e, s, v) for e, s, v in (f1, f2)]
        # Failure BEFORE pass 3: both entries unstarted, both victims back.
        b.rollback(fetch_list, committed_through=-1, pass3_started=False)
        assert b.slot_of == {1: 1, 2: 0}
        assert b.evictions == 0

    def test_trim_to_cap_and_reset(self):
        from omlx.patches.expert_streaming.slot_cache import SlotBookkeeping

        b = SlotBookkeeping(4)
        for e in range(4):
            b.commit(e, b.acquire(set(range(4)))[0])
        evicted = []
        b.cap = 2
        n = b.trim_to_cap({3}, on_evict=evicted.append)
        assert n == 2 and sorted(evicted) == [0, 1]
        assert list(b.slot_of) == [2, 3]
        # rows freed are the victims' rows (e0->3, e1->2 under LIFO pop)
        assert sorted(b.free) == [2, 3]
        out = b.reset()
        assert sorted(out) == [2, 3]
        assert b.free == [0, 1, 2, 3] and not b.slot_of

    def test_touch_and_release(self):
        from omlx.patches.expert_streaming.slot_cache import SlotBookkeeping

        b = SlotBookkeeping(2)
        b.commit(1, b.acquire(())[0])
        b.commit(2, b.acquire({1})[0])
        assert b.touch(1)  # 1 -> MRU end
        assert list(b.slot_of) == [2, 1]
        assert not b.touch(9)
        b.release(2)
        assert b.free == [0] and 2 not in b.slot_of

    def test_working_set_step(self):
        from omlx.patches.expert_streaming.slot_cache import working_set_step

        assert working_set_step(16, 8) == 2
        assert working_set_step(8, 8) == 1
        assert working_set_step(4, 8) == 1  # never below one token
        assert working_set_step(33, 8) == 4


class TestDecodeVisitStats:
    def test_shared_contract(self):
        from omlx.patches.deepseek_v41.streaming_backing import _V41CacheStats
        from omlx.patches.expert_streaming.slot_cache import DecodeVisitStats

        # The legacy offload state rides the base class directly — the
        # pass-subclass was dropped; both shapes must expose the contract.
        for cls in (_V41CacheStats, DecodeVisitStats):
            st = cls()
            assert isinstance(st, DecodeVisitStats)
            st.note_visit(3, missed=True)
            st.note_visit(3, missed=False)
            assert st.decode_layers == 2
            assert st.decode_layers_missed == 1
            assert st.decode_misses_by_layer == {3: 1}
            st.reset_visits()
            assert st.decode_layers == 0 and not st.decode_misses_by_layer


class TestAdmissionDelegation:
    def test_supported_type_uses_streaming_estimate(self, tmp_path):
        from omlx.patches.moe_expert_offload import (
            estimate_offload_admission_bytes,
        )

        # E=16 experts, fraction 0.25 -> capacity min(16, max(8, 4)) = 8
        tensors = _moe_dir(tmp_path, n_experts=16, moe=8, hidden=16)
        expert_bytes = sum(
            int(np.prod(a.shape)) * a.dtype.itemsize
            for a, _ in tensors.values()
        )
        full = 10_000_000
        got = estimate_offload_admission_bytes(tmp_path, full, 0.25)
        assert got == full - expert_bytes // 2

    def test_unsupported_type_falls_back_full(self, tmp_path):
        from omlx.patches.moe_expert_offload import (
            estimate_offload_admission_bytes,
        )

        _moe_dir(tmp_path, model_type="not_a_moe")
        assert estimate_offload_admission_bytes(tmp_path, 1234, 0.25) == 1234


class TestStackedKeyIndex:
    """_resolve_stacked_key resolves via a per-backing index, not a
    per-call weight-map scan — with identical results, including exotic
    spellings where mid is a substring of a longer component."""

    class _Backing:
        def __init__(self, wm):
            self._weight_map = wm

    def _wm(self):
        return {
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": "f0",
            "model.layers.0.mlp.switch_mlp.gate_proj.scales": "f0",
            "model.layers.1.mlp.switch_mlp.gate_proj.weight": "f0",
            "model.layers.1.mlp.switch_mlp.gate_proj.scales": "f0",
            # Canonical up_proj key for a DIFFERENT layer: the bucket
            # exists, but the needle for layer 1 misses on it — the
            # exotic key below resolves only via the per-(mid, needle)
            # legacy-scan fallback.
            "model.layers.0.mlp.switch_mlp.up_proj.weight": "f0",
            # Exotic: mid is a substring of a LONGER component — lands in
            # a different bucket than switch_mlp.up_proj.weight.
            "model.layers.1.mlp.switch_mlp.up_proj.weights_extra": "f0",
            "unrelated.dense.weight": "f0",
        }

    def test_indexed_hit(self):
        from omlx.patches.expert_streaming import _resolve_stacked_key

        backing = self._Backing(self._wm())
        # Candidate absent from wm — forces the indexed bucket path
        # (an exact hit returns before the index is consulted).
        got = _resolve_stacked_key(
            ["model.layers.0.mlp.experts.gate_proj.weight"],
            "gate_proj",
            "weight",
            backing,
            "layers.0.",
        )
        assert got == "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        # index memoized on the backing
        assert "switch_mlp.gate_proj.weight" in backing._stacked_key_index

    def test_exotic_substring_still_resolves(self):
        from omlx.patches.expert_streaming import _resolve_stacked_key

        backing = self._Backing(self._wm())
        # up_proj.weight has NO canonical bucket; the exotic
        # .weights_extra key contains mid as substring — found via the
        # memoized legacy scan.
        got = _resolve_stacked_key(
            ["nope"],
            "up_proj",
            "weight",
            backing,
            "layers.1.",
        )
        assert got == "model.layers.1.mlp.switch_mlp.up_proj.weights_extra"

    def test_missing_required_raises(self):
        from omlx.patches.expert_streaming import _resolve_stacked_key

        backing = self._Backing(self._wm())
        with pytest.raises(ValueError, match="no checkpoint key"):
            _resolve_stacked_key(
                ["nope"], "down_proj", "weight", backing, "layers.9."
            )


