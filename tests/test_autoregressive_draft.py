# SPDX-License-Identifier: Apache-2.0
"""Cache frontier checks for full accepts, partial accepts and first rejects."""

import unittest

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from omlx.speculative.autoregressive import GreedyQwenDraft


class _Model:
    def __init__(self, wrong_after=()):
        self.wrong_after = wrong_after

    def make_cache(self):
        return [ArraysCache(size=1), KVCache()]

    def __call__(self, inputs, cache, n_confirmed=0):
        recurrent, kv = cache
        self.before = recurrent[0]
        self.inputs = inputs.flatten()
        recurrent[0] = self._append(self.before, self.inputs)
        keys = inputs[None, :, :, None]
        kv.update_and_fetch(keys, keys)
        tokens = self.inputs.tolist()
        next_ids = [token + (2 if token in self.wrong_after else 1) for token in tokens]
        return mx.array(next_ids)[None, :, None] == mx.arange(32)[None, None, :]

    @staticmethod
    def _append(prefix, tokens):
        return tokens if prefix is None else mx.concatenate([prefix, tokens])

    def mtp_partial_rollback(self, cache, accepted, depth):
        recurrent, kv = cache
        recurrent[0] = self._append(self.before, self.inputs[: accepted + 1])
        return kv.trim(depth - accepted) == depth - accepted


class TestGreedyQwenDraft(unittest.TestCase):
    def test_both_caches_follow_committed_tokens_across_rounds(self):
        for wrong_after, first_accepted in [((), 3), ((3,), 1), ((2,), 0)]:
            with self.subTest(wrong_after=wrong_after):
                decode = GreedyQwenDraft(_Model(), _Model(wrong_after), [0, 1, 2])
                committed = [0, 1, 2]
                for depth in (3, 2, 0):
                    result = decode.step(depth)
                    if depth == 3:
                        self.assertEqual(result.accepted, first_accepted)
                    start = committed[-1] + 1
                    self.assertEqual(
                        result.tokens, list(range(start, start + result.accepted + 1))
                    )
                    committed.extend(result.tokens)
                    for recurrent, kv in (decode.target_cache, decode.draft_cache):
                        self.assertEqual(recurrent[0].tolist(), committed[:-1])
                        self.assertEqual(kv.offset, len(committed) - 1)
                        self.assertEqual(
                            kv.keys[0, 0, : kv.offset, 0].tolist(), committed[:-1]
                        )
