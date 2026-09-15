# SPDX-License-Identifier: Apache-2.0
"""Generated-history sidecars use exact committed state and request ownership."""

import gc
import threading
import weakref
from collections import OrderedDict
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
# ruff: noqa: E402 -- MLX must be optional at test collection time.

from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_lm_mtp import generated_prefix as gp
from omlx.patches.mlx_lm_mtp import prompt_priming as pp
from tests.test_mtp_prompt_priming import (
    TestActivationHandoff as _Handoff,
)
from tests.test_mtp_prompt_priming import (
    _apply_patch,  # noqa: F401 -- activates the real model patch for these tests
    _make_cache,
    _make_tiny_model,
    _MemoryMtpPrefixCache,
    _reference_head_cache,
    _tokens,
)


class Batch(SimpleNamespace):
    Response = SimpleNamespace

    def extract_cache(self, index):
        return self.prompt_cache

    def filter(self, keep):
        self.uids = keep
        self.tokens = []


def activate(
    cache,
    *,
    uid=0,
    metadata=None,
    primed=True,
    depth=1,
    head_clone=False,
    model_type="qwen4_exp",
):
    mx.random.seed(11)
    model = _make_tiny_model()
    # The tiny Qwen3.5 fixture exercises the same fold/cache contract; only
    # admission is labeled Qwen4 here, not the architecture or model weights.
    model.model_type = model_type
    model._omlx_mtp_commit_align = cache.block_size
    # Pre-load tests can change the process-wide default; own this schedule.
    model._omlx_mtp_depth = depth
    model._omlx_mtp_head_clone = head_clone
    tokens = _tokens(cache.block_size - 1, seed=17)
    request = SimpleNamespace(prompt_token_ids=tokens.tolist(), **(metadata or {}))
    gp.register(model, uid, request, cache)
    backbone = _make_cache(model)
    if primed:
        logits = model(tokens[None, :], cache=backbone)
    else:
        with pp.suppress_capture():
            logits = model(tokens[None, :], cache=backbone)
    lp = logits[0, -1] - mx.logsumexp(logits[0, -1])
    seed = _Handoff()._gen_batch(model, backbone, tokens)
    batch = Batch(**vars(seed))
    batch.uids = [uid]
    batch._next_tokens = mx.argmax(lp, keepdims=True).astype(mx.uint32)
    batch._next_logprobs = [lp]
    batch._num_tokens = [0]
    batch.max_tokens = [16]
    batch.state_machines = [
        SimpleNamespace(match=lambda state, tok: (state, None, None))
    ]
    batch._matcher_states = [None]
    bg._post_init_mtp(batch)
    return batch, int(batch._next_tokens.item())


def native_cache():
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    blocks = {}
    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.block_size = 8
    cache.paged_cache = SimpleNamespace(
        model_name="generated-history-test",
        cached_block_hash_to_block=SimpleNamespace(get_block=blocks.get),
    )
    cache._prefix_index = {}
    cache._mtp_prefix_snapshots = OrderedDict()
    cache._mtp_prefix_snapshot_lock = threading.RLock()
    return cache, blocks


@pytest.mark.parametrize("wrapper", [None, "language_model", "_language_model"])
def test_native_qwen4_text_host_registers_and_cleans_up(wrapper):
    from tests.test_mlx_vlm_qwen4_exp_compat import (
        _make_bound_qwen4_language_model,
        _tiny_config,
    )

    # Use the language host's real constructor and owner binding, so a
    # mislabeled stand-in cannot hide changes in loader model types.
    host, owner = _make_bound_qwen4_language_model(_tiny_config())
    assert host.model_type == "qwen4_exp_text"
    assert host.get_mtp_module() is owner.mtp
    model = host if wrapper is None else SimpleNamespace(**{wrapper: host})
    cache = _MemoryMtpPrefixCache()
    request = SimpleNamespace(prompt_token_ids=[2, 3, 4])

    gp.register(model, 7, request, cache)

    plan = getattr(host, gp._PLANS, {}).get(7)
    assert plan is not None
    assert plan.tokens == (2, 3, 4)
    assert plan.cache() is cache
    gp.unregister(model, 7)
    assert not getattr(host, gp._PLANS)


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text"])
@pytest.mark.parametrize("terminal", [False, True])
def test_real_fold_restores_generated_boundary_and_detaches(terminal, model_type):
    cache, blocks = native_cache()
    batch, token = activate(cache, model_type=model_type)
    ledger = batch.tokens[0] + [token]
    state = batch._omlx_mtp_state
    if terminal:
        batch.max_tokens = [1]
    response = bg._emit_response(batch, token, mx.zeros((256,)))[0]
    assert response.token == token
    assert response.finish_reason == ("length" if terminal else None)
    assert cache.restore_mtp_prefix_snapshot(ledger, 8) is None
    tip = cache._mtp_prefix_chain_tip(ledger, 8)
    blocks[tip] = object()
    snapshot = cache.restore_mtp_prefix_snapshot(ledger, 8)
    assert snapshot is not None
    assert snapshot.mtp_cache[0].offset == 7
    ref = _reference_head_cache(batch.model, mx.array(ledger, dtype=mx.uint32))
    for expected, actual in zip(ref, snapshot.mtp_cache):
        for e, a in zip(expected.state, actual.state):
            assert mx.allclose(e[..., :7, :], a[..., :7, :], atol=2e-5).item()
    fresh = _make_cache(batch.model)
    _, hidden = batch.model(mx.array(ledger)[None, :], cache=fresh, return_hidden=True)
    assert mx.allclose(
        snapshot.pending_hidden, batch.model.model.norm(hidden[:, -1:]), atol=2e-5
    ).item()
    saved = [x + 0 for c in snapshot.mtp_cache for x in c.state]
    mx.eval(saved)
    # Mutating the live head cache cannot alter an already published snapshot.
    for c in state.mtp_cache:
        c.keys = mx.zeros_like(c.keys)
        c.values = mx.zeros_like(c.values)
    assert all(
        mx.array_equal(a, b).item()
        for a, b in zip(saved, [x for c in snapshot.mtp_cache for x in c.state])
    )
    assert pp.prepare_prefix_context(
        batch.model,
        request_id="continuation",
        prompt_tokens=ledger + [42],
        cached_tokens=8,
        prefix_cache=cache,
    )
    ctx = pp._find_ctx(batch.model)
    assert ctx.folded == 7
    assert ctx.expected_offset == 8
    cache._on_block_hash_dropped(tip)
    assert cache.restore_mtp_prefix_snapshot(ledger, 8) is None


@pytest.mark.parametrize(
    "metadata",
    [
        {"vlm_extra_keys_for_cache": ["image-hash"]},
        {"vlm_extra_key_token_start_for_cache": 0},
        {"vlm_extra_key_ranges_for_cache": []},
    ],
)
@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text"])
def test_media_requests_never_publish(metadata, model_type):
    cache = _MemoryMtpPrefixCache()
    batch, token = activate(cache, metadata=metadata, model_type=model_type)
    bg._emit_response(batch, token, mx.zeros((256,)))
    assert not cache.snapshots


def test_unprimed_state_never_publishes():
    cache = _MemoryMtpPrefixCache()
    batch, token = activate(cache, primed=False)
    bg._emit_response(batch, token, mx.zeros((256,)))
    assert not cache.snapshots


@pytest.mark.parametrize("failure", ["uid", "boundary", "short-cache", "emission"])
def test_failed_or_stale_emission_does_not_publish(failure):
    cache = _MemoryMtpPrefixCache()
    batch, token = activate(cache)
    if failure == "uid":
        batch.uids = [100]
    elif failure == "boundary":
        batch.tokens[0].append(8)
    elif failure == "short-cache":
        batch._omlx_mtp_state.mtp_cache[0].trim(4)
    else:

        def fail(*args):
            raise RuntimeError("synthetic matcher failure")

        batch.state_machines[0].match = fail
    if failure == "emission":
        with pytest.raises(RuntimeError, match="synthetic"):
            bg._emit_response(batch, token, mx.zeros((256,)))
    else:
        bg._emit_response(batch, token, mx.zeros((256,)))
    assert not cache.snapshots


def test_plans_are_uid_owned_and_cleanup_releases_them():
    from omlx.scheduler import _unregister_uid_row, _unregister_uid_rows_for_model

    cache = _MemoryMtpPrefixCache()
    batch, _ = activate(cache)
    model = batch.model
    tokens = list(range(7))
    plain = SimpleNamespace(prompt_token_ids=tokens)
    media = SimpleNamespace(
        prompt_token_ids=tokens, vlm_extra_keys_for_cache=["image-hash"]
    )
    gp.register(model, 10, plain, cache)
    gp.register(model, 11, media, cache)
    foreign = SimpleNamespace(model=model, tokens=[tokens], uids=[11])
    state = SimpleNamespace(uid=11, hist_offset=8)
    gp.record(foreign, state, mx.ones((1, 1, 64)))
    assert getattr(state, gp._STATE_PLAN) is None
    assert 10 in getattr(model, gp._PLANS)
    _unregister_uid_row(model, 10)
    assert not getattr(model, gp._PLANS)
    gp.register(model, 12, plain, cache)
    _unregister_uid_rows_for_model(model)
    assert not getattr(model, gp._PLANS)


def test_plan_and_state_do_not_retain_cache():
    cache = _MemoryMtpPrefixCache()
    ref = weakref.ref(cache)
    batch, token = activate(cache)
    gp.register(batch.model, 9, SimpleNamespace(prompt_token_ids=[1, 2]), cache)
    del cache
    gc.collect()
    assert ref() is None
    assert gp.candidate(batch, token) is None


def test_unsupported_model_and_bounded_pending_plans(monkeypatch):
    cache = _MemoryMtpPrefixCache()
    batch, _ = activate(cache)
    model = batch.model
    request = SimpleNamespace(prompt_token_ids=[1, 2])
    monkeypatch.setattr(gp, "_MAX_PLANS", 2)
    for uid in range(10, 13):
        gp.register(model, uid, request, cache)
    assert list(getattr(model, gp._PLANS)) == [11, 12]
    model.model_type = "unsupported"
    gp.register(model, 12, request, cache)
    assert list(getattr(model, gp._PLANS)) == [11]


@pytest.fixture(params=("cpu", "gpu"))
def parity_device(request):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu if request.param == "cpu" else mx.gpu)
    try:
        yield request.param
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("depth", (1, 3))
@pytest.mark.parametrize("head_clone", (False, True))
def test_repeated_verify_cycles_preserve_tokens_and_boundary_history(
    parity_device, depth, head_clone, monkeypatch
):
    outputs = []
    emit_response = bg._emit_response
    for enabled in (False, True):
        cache = _MemoryMtpPrefixCache()
        batch, _ = activate(cache, depth=depth, head_clone=head_clone)
        batch.max_tokens = [25]
        state = batch._omlx_mtp_state
        # Fix the schedule so the comparison isolates cache publication.
        state.controller = None
        if not enabled:
            setattr(state, gp._STATE_PLAN, None)

        def checked_emit(
            gen_batch,
            token_id,
            logprobs,
            stats=None,
            *,
            capture=enabled,
            prefix_cache=cache,
            mtp_state=state,
        ):
            boundary = len(gen_batch.tokens[0]) + 1
            ledger = gen_batch.tokens[0] + [token_id]
            expected = None
            if capture and boundary % prefix_cache.block_size == 0:
                expected = [
                    [row[..., : boundary - 1, :] + 0 for row in entry.state]
                    for entry in mtp_state.mtp_cache
                ]
                mx.eval(expected)
            response = emit_response(gen_batch, token_id, logprobs, stats)
            if expected is not None:
                snapshot = prefix_cache.restore_mtp_prefix_snapshot(ledger, boundary)
                assert snapshot is not None
                for expected_entry, actual_entry in zip(expected, snapshot.mtp_cache):
                    assert actual_entry.offset == boundary - 1
                    for original, copied in zip(expected_entry, actual_entry.state):
                        assert mx.array_equal(
                            original, copied[..., : boundary - 1, :]
                        ).item()
            return response

        monkeypatch.setattr(bg, "_emit_response", checked_emit)
        emitted = []
        while batch.uids:
            response = bg._mtp_next(batch, state)[0]
            emitted.append(response.token)
        outputs.append(emitted)
        if not enabled:
            assert not cache.snapshots
            continue
        assert {boundary for _, boundary in cache.snapshots} == {8, 16, 24, 32}
        # Verify the history against an independent full-prompt fold on
        # both devices, in addition to the exact live-cache checks above.
        for (ledger, boundary), snapshot in cache.snapshots.items():
            reference = _reference_head_cache(
                batch.model, mx.array(ledger, dtype=mx.uint32)
            )
            assert snapshot.mtp_cache[0].offset == boundary - 1
            for expected, actual in zip(reference, snapshot.mtp_cache):
                for e, a in zip(expected.state, actual.state):
                    assert mx.allclose(
                        e[..., : boundary - 1, :], a[..., : boundary - 1, :], atol=2e-5
                    ).item()
    assert outputs[0] == outputs[1]
