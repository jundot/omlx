"""Fast-reference regressions: exact quantization arithmetic and fewer eval barriers."""
import unittest
import numpy as np
from jangh2_bridge.reference_ops import (expert_forward, hadamard32, pack_codes)


def fixture(bits=2, *, rotation='hadamard32', tokens=7, routes=4):
    rng = np.random.default_rng(831 + bits)
    e, d, hidden = 8, 64, 32
    levels = np.linspace(-1.25, 1.25, 1 << bits).astype(np.float32)
    projs = {}
    for name, n, k in [('gate_proj', hidden, d), ('up_proj', hidden, d),
                       ('down_proj', d, hidden)]:
        q = rng.integers(0, 1 << bits, (e, n, k), dtype=np.uint8)
        scale = rng.uniform(.01, .04, (e, n)).astype(np.float16)
        projs[name] = (pack_codes(q, bits), scale, bits, levels, rotation)
    x = rng.normal(0, .2, (tokens, d)).astype(np.float32)
    idx = rng.integers(0, e, (tokens, routes), dtype=np.uint32)
    return x, idx, projs


class EvalCountingNumpy:
    float32 = np.float32
    uint32 = np.uint32
    int32 = np.int32
    def __init__(self):
        self.count = 0
    def __getattr__(self, name):
        return getattr(np, name)
    def eval(self, *args):
        self.count += 1


class MlxBuiltinStub(EvalCountingNumpy):
    def __init__(self):
        super().__init__()
        self.hadamard_calls = 0
    def hadamard_transform(self, array):
        self.hadamard_calls += 1
        h = np.ones((1, 1), np.float32)
        for _ in range(5):
            h = np.block([[h, h], [h, -h]])
        return array @ (h.T / np.sqrt(32))


class FastReferenceTests(unittest.TestCase):
    def test_matches_legacy_across_bits_and_routes(self):
        for bits in (2, 3, 4):
            for routes in (1, 4):
                x, idx, p = fixture(bits, tokens=5, routes=routes)
                expected = expert_forward(x, idx, p, np, chunk_size=1)
                actual = expert_forward(x, idx, p, np, chunk_size=8,
                                        evaluation='per_chunk')
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_different_rotations_not_reused(self):
        x, idx, p = fixture(2)
        p['up_proj'] = (*p['up_proj'][:4], 'none')
        np.testing.assert_allclose(
            expert_forward(x, idx, p, np, chunk_size=1),
            expert_forward(x, idx, p, np, chunk_size=8,
                           evaluation='per_chunk'),
            rtol=2e-5, atol=2e-5)

    def test_fewer_evaluation_boundaries(self):
        x, idx, p = fixture(2, tokens=1, routes=8)
        old = EvalCountingNumpy()
        fast = EvalCountingNumpy()
        legacy = expert_forward(x, idx, p, old, chunk_size=1)
        faster = expert_forward(x, idx, p, fast, chunk_size=8,
                                evaluation='per_chunk')
        np.testing.assert_allclose(legacy, faster, rtol=2e-5, atol=2e-5)
        self.assertEqual(old.count, 17)  # 1 range + 8 hidden + 8 output
        self.assertEqual(fast.count, 2)  # 1 range + 1 output

    def test_builtin_hadamard_transform(self):
        stub = MlxBuiltinStub()
        x = np.random.default_rng(9).normal(size=(3, 96)).astype(np.float32)
        expected = hadamard32(x, np)
        actual = hadamard32(x, stub, optimized=True)
        self.assertGreater(stub.hadamard_calls, 0)
        np.testing.assert_allclose(actual, expected, atol=2e-6)

    def test_invalid_options_still_fail(self):
        x, idx, p = fixture(2)
        with self.assertRaises(ValueError):
            expert_forward(x, idx, p, np, chunk_size=8,
                           evaluation='nonsense')
        with self.assertRaises(ValueError):
            expert_forward(x, idx, p, np, chunk_size=65,
                           evaluation='per_chunk')
        idx[0, 0] = 99
        with self.assertRaises(ValueError):
            expert_forward(x, idx, p, np, chunk_size=8,
                           evaluation='per_chunk')


if __name__ == '__main__':
    unittest.main()
