# SPDX-License-Identifier: Apache-2.0
"""Persistent per-row completed-block banks on BatchQSAKVCache.

The batched indexer pools every row anchored at its first real token. It used
to build a fresh singleton cache per row per step, which re-pooled (and copied)
the row's whole history every decode step. The banks keep those blocks across
steps; these tests pin the masks -- and therefore the layer output -- and the
cache state bit-identical to the fresh-cache path through every way a running
batch changes: decode, verify windows, ragged MTP rollback, filter, extend,
trim and state restore.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


def _language():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen4_exp.language as language

    return language


def _config(budget=8, ratio=2):
    from mlx_vlm.models.qwen4_exp import TextConfig

    return TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=8192,
        hc_count=2,
        hc_lowrank=8,
        head_dim=8,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        ple_layer_ids=[],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=budget,
        indexer_compress_ratio=ratio,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )


class _Pair:
    """The same batch driven twice: banks on (fast) and off (reference)."""

    def __init__(self, monkeypatch, language, attention, config, prefixes, seed):
        self.monkeypatch = monkeypatch
        self.language = language
        self.attention = attention
        self.config = config
        mx.random.seed(seed)
        self.fast = self._batch(prefixes)
        self.reference = self._batch(prefixes, reuse=True)

    def _batch(self, prefixes, reuse=False):
        if not reuse:
            self._inputs = [
                mx.random.normal((1, n, self.config.hidden_size)) for n in prefixes
            ]
        rows = []
        for prefix in self._inputs:
            row = self.language.QSAKVCache()
            mx.eval(self.attention(prefix, mask="causal", cache=row))
            rows.append(row)
        return self.language.BatchQSAKVCache.merge(rows)

    def step(self, length=1, target_verify=False, positions="none"):
        batch = self.fast.offset.shape[0]
        x = mx.random.normal((batch, length, self.config.hidden_size))
        outputs = []
        for enabled, cache in ((True, self.fast), (False, self.reference)):
            self.monkeypatch.setattr(self.language, "_BATCH_ROW_BANKS_ENABLED", enabled)
            mask = self.language._create_qwen3_5_attention_mask(x, cache)
            position_ids = None
            if positions == "mrope":
                position_ids = mx.broadcast_to(
                    cache.offset[None, :, None] + mx.arange(length)[None, None],
                    (3, batch, length),
                )
            out = self.attention(
                x,
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                target_verify=target_verify,
            )
            mx.eval(out)
            outputs.append(out)
        assert mx.array_equal(outputs[0], outputs[1]).item()
        self.assert_same_state()

    def both(self, fn):
        fn(self.fast)
        fn(self.reference)
        self.assert_same_state()

    def assert_same_state(self):
        fast, reference = self.fast, self.reference
        assert fast.index_offset == reference.index_offset
        assert fast._idx == reference._idx
        assert fast.left_padding.tolist() == reference.left_padding.tolist()
        assert fast.offset.tolist() == reference.offset.tolist()
        for a, b in zip(fast.state[1:], reference.state[1:]):
            assert mx.array_equal(a, b).item()
        width = fast._idx
        assert mx.array_equal(
            fast.keys[..., :width, :], reference.keys[..., :width, :]
        ).item()


def _setup(monkeypatch, prefixes=(12, 23, 5), seed=7, **config):
    language = _language()
    # The banks feed the dense masked path here; the batched gathered arm has
    # its own tests (test_qwen4_qsa_batch_gather.py).
    monkeypatch.setattr(language, "_GATHERED_BATCH_DISABLED", True, raising=False)
    cfg = _config(**config)
    attention = language.Qwen4ExpAttention(cfg)
    mx.eval(attention.parameters())
    return language, _Pair(monkeypatch, language, attention, cfg, prefixes, seed)


def _pooled_blocks(monkeypatch, language):
    """Count the blocks pooled through the shared pooling helper."""

    pooled = []
    original = language.pool_completed_index_keys

    def counting(*args, **kwargs):
        out = original(*args, **kwargs)
        pooled.append(int(out.shape[1]))
        return out

    monkeypatch.setattr(language, "pool_completed_index_keys", counting)
    return pooled


def _ragged_rollback(retained, window):
    """What mlx_vlm's vector rollback does for per-row accepted counts."""

    def apply(cache):
        keep = max(retained)
        if window - keep:
            cache.trim(window - keep)
        padding = [keep - value for value in retained]
        if any(padding):
            cache.prepare(right_padding=padding)
            cache.finalize()

    return apply


@pytest.mark.parametrize("positions", ["none", "mrope"])
def test_banks_match_fresh_row_caches_through_a_running_batch(monkeypatch, positions):
    language, pair = _setup(monkeypatch)
    for _ in range(3):
        pair.step(positions=positions)
    # A Lightning MTP verify window, then ragged acceptance.
    pair.step(length=3, target_verify=True, positions=positions)
    pair.both(_ragged_rollback([3, 1, 2], 3))
    for _ in range(2):
        pair.step(positions=positions)
    pair.step(length=3, target_verify=True, positions=positions)
    pair.both(_ragged_rollback([1, 2, 1], 3))
    pair.step(positions=positions)
    # Rows keep their banks when the batch is reordered.
    pair.both(lambda cache: cache.filter(mx.array([2, 0, 1])))
    pair.step(positions=positions)
    pair.both(lambda cache: cache.filter(mx.array([1, 2, 0])))
    pair.step(positions=positions)
    # The longest row (the smallest left padding) leaves: the rest shift left.
    pair.both(lambda cache: cache.filter(mx.array([0, 2])))
    for _ in range(2):
        pair.step(positions=positions)
    # A new request joins.
    joiner = _Pair(monkeypatch, language, pair.attention, pair.config, (17,), 3)
    pair.fast.extend(joiner.fast)
    pair.reference.extend(joiner.reference)
    pair.assert_same_state()
    for _ in range(3):
        pair.step(positions=positions)
    # A plain trim, then different tokens in the trimmed columns.
    pair.both(lambda cache: cache.trim(2))
    for _ in range(2):
        pair.step(positions=positions)
    assert pair.fast._pooled_bank is not None


def test_banks_pool_only_new_blocks(monkeypatch):
    language, pair = _setup(monkeypatch, prefixes=(40, 57, 31))
    pair.step()  # first step pools every row's history
    pooled = _pooled_blocks(monkeypatch, language)
    monkeypatch.setattr(language, "_BATCH_ROW_BANKS_ENABLED", True)
    batch = 3
    for _ in range(8):
        x = mx.random.normal((batch, 1, pair.config.hidden_size))
        mask = language._create_qwen3_5_attention_mask(x, pair.fast)
        mx.eval(pair.attention(x, mask=mask, cache=pair.fast))
    # Eight one-token steps complete at most four blocks per row (ratio 2).
    assert pooled and sum(pooled) <= batch * 4
    assert all(width <= 1 for width in pooled)


def test_rollback_repools_the_block_it_cut(monkeypatch):
    language, pair = _setup(monkeypatch, prefixes=(21, 30))
    pair.step()
    pair.step(length=3, target_verify=True)
    banks = list(pair.fast._pooled_bank.blocks)
    pair.both(_ragged_rollback([1, 3], 3))
    clamped = list(pair.fast._pooled_bank.blocks)
    assert clamped[0] < banks[0] and clamped[1] == banks[1]
    pair.step()


def test_reassigned_raw_bank_drops_banks(monkeypatch):
    language, pair = _setup(monkeypatch)
    pair.step()
    assert pair.fast._pooled_bank is not None
    pair.fast.state = pair.fast.state
    assert pair.fast._pooled_bank is None
    pair.reference.state = pair.reference.state
    pair.step()


def test_bank_bytes_are_counted(monkeypatch):
    language, pair = _setup(monkeypatch)
    before = pair.fast.nbytes
    pair.step()
    banks = pair.fast._pooled_bank.keys.nbytes
    assert banks and pair.fast.nbytes >= before + banks


def test_served_shape_banks_match(monkeypatch):
    """ratio 4 / top-k 512 as served: the native decode selection applies."""

    language, pair = _setup(
        monkeypatch, prefixes=(2100, 2600, 1900), seed=5, budget=2048, ratio=4
    )
    for _ in range(3):
        pair.step()
    pair.step(length=4, target_verify=True)
    pair.both(_ragged_rollback([2, 4, 1], 4))
    for _ in range(2):
        pair.step()


def test_right_padded_prompt_prefill_keeps_banks_exact(monkeypatch):
    """Batched prompt processing right-pads the shorter prompt: prepare()
    runs before the chunks are appended and finalize() rolls the padding into
    left padding after the last one. Blocks pooled over that padding must not
    survive the roll."""

    language, pair = _setup(monkeypatch, prefixes=(4, 4))
    lengths = (23, 9)
    width = max(lengths)
    prompts = [mx.random.normal((1, n, pair.config.hidden_size)) for n in lengths]
    padded = mx.concatenate(
        [mx.pad(p, [(0, 0), (0, width - p.shape[1]), (0, 0)]) for p in prompts]
    )
    caches = []
    for enabled in (True, False):
        monkeypatch.setattr(language, "_BATCH_ROW_BANKS_ENABLED", enabled)
        cache = language.BatchQSAKVCache([0, 0])
        cache.prepare(lengths=list(lengths), right_padding=[width - n for n in lengths])
        for start in range(0, width, 6):
            chunk = padded[:, start : start + 6]
            mask = language._create_qwen3_5_attention_mask(chunk, cache)
            mx.eval(pair.attention(chunk, mask=mask, cache=cache))
        cache.finalize()
        caches.append(cache)
    pair.fast, pair.reference = caches
    assert pair.fast._pooled_bank is not None
    pair.assert_same_state()
    for _ in range(4):
        pair.step()
    pair.step(length=3, target_verify=True)
    pair.both(_ragged_rollback([2, 1], 3))
    for _ in range(2):
        pair.step()
