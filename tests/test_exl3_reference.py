"""Independent CPU tile oracle versus the packed Metal projection."""

import unittest

import mlx.core as mx
import numpy as np

from omlx.quantization.exl3 import Exl3Spec, Exl3SwitchLinear


def matrix(packed, window):
    # Pairwise 32-bit funnel matches published tail-biting trellis layout;
    # this deliberately differs from the per-output 16-bit GPU reader.
    e, kt, nt, hw = packed.shape
    words = packed.astype(np.uint32).reshape(e, kt, nt, hw // 2, 2)
    words = words[..., 0] | (words[..., 1] << 16)
    perm = []
    for thread in range(32):
        r = (thread % 4) * 2
        c = thread // 4
        perm.extend(
            [
                r * 16 + c,
                (r + 1) * 16 + c,
                (r + 8) * 16 + c,
                (r + 9) * 16 + c,
                r * 16 + c + 8,
                (r + 1) * 16 + c + 8,
                (r + 8) * 16 + c + 8,
                (r + 9) * 16 + c + 8,
            ]
        )
    code = []
    for t in range(128):
        a = ((t * 2 + 1) * hw) // 16
        b = ((t * 2 + 2) * hw) // 16
        i = (a + hw * 16 - 16) // 32
        j = (b + hw * 16 - 1) // 32
        shift = (j + 1) * 32 - (b + hw * 16)
        merged = (words[..., i % (hw // 2)].astype(np.uint64) << 32) | words[
            ..., j % (hw // 2)
        ].astype(np.uint64)
        funnel = (merged >> np.uint64(shift)) & np.uint64(0xFFFFFFFF)
        code.extend([(funnel >> np.uint64(b - a)) & 65535, funnel & 65535])
    cw = np.stack(code, axis=-1).astype(np.uint32) & ((1 << window) - 1)
    mixed = ((cw.astype(np.uint64) * 0xCBAC1FED) & 0xFFFFFFFF).astype(np.uint32)
    pair = np.uint32(0x3B603B60) ^ (mixed & np.uint32(0x8FFF8FFF))
    weights = (
        (pair & 65535).astype(np.uint16).view(np.float16).astype(np.float32)
        + (pair >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
    ).astype(np.float16)
    tiles = np.empty_like(weights)
    tiles[..., perm] = weights
    return (
        tiles.reshape(e, kt, nt, 16, 16)
        .transpose(0, 1, 3, 2, 4)
        .reshape(e, kt * 16, nt * 16)
    )


def h128(x):
    # Explicit Walsh matrix oracle, independent of the GPU butterfly graph.
    i = np.arange(128)
    sign = np.ones((128, 128), np.float32)
    for bit in range(7):
        sign *= 1 - 2 * ((i[:, None] >> bit) & (i[None, :] >> bit) & 1)
    return (x.reshape(-1, 128) @ sign * np.float32(128**-0.5)).reshape(x.shape)


def oracle(x, packed, suh, svh, ids, window):
    w = matrix(packed, window)
    z = h128(x.astype(np.float16).astype(np.float32) * suh[ids]).astype(np.float16)
    inner = np.einsum(
        "ri,rij->rj", z.astype(np.float32), w[ids].astype(np.float32)
    ).astype(np.float16)
    return (h128(inner.astype(np.float32)) * svh[ids]).astype(np.float16)


class Exl3Tests(unittest.TestCase):
    @unittest.skipUnless(mx.metal.is_available(), "requires Metal")
    def test_rates_windows_and_broadcast(self):
        rng = np.random.default_rng(42)
        for hw, win in [(32, 16), (42, 15), (48, 12), (64, 16)]:
            q = rng.integers(0, 65536, (2, 8, 8, hw), dtype=np.uint16)
            su = rng.choice([-1.0, 1.0], (2, 128)).astype(np.float16)
            sv = np.full((2, 128), 0.125, np.float16)
            x = rng.normal(size=(3, 128)).astype(np.float16)
            ids = np.array([0, 1, 0], np.uint32)
            layer = Exl3SwitchLinear(
                mx.array(q), mx.array(su), mx.array(sv), Exl3Spec(hw, win)
            )
            y = layer(mx.array(x[:, None, :]), mx.array(ids))
            mx.eval(y)
            np.testing.assert_allclose(
                np.asarray(y[:, 0, :]),
                oracle(x, q, su, sv, ids, win),
                rtol=0.02,
                atol=0.015,
            )

    @unittest.skipUnless(mx.metal.is_available(), "requires Metal")
    def test_routed_broadcast_and_invalid_ids(self):
        rng = np.random.default_rng(89)
        q = rng.integers(0, 65536, (2, 8, 8, 42), dtype=np.uint16)
        su = np.ones((2, 128), np.float16)
        sv = np.full((2, 128), 0.1, np.float16)
        x = rng.normal(size=(2, 3, 128)).astype(np.float16)
        ids = rng.integers(0, 2, (2, 3, 2), dtype=np.int32)
        layer = Exl3SwitchLinear(
            mx.array(q), mx.array(su), mx.array(sv), Exl3Spec(42, 15)
        )
        y = layer(mx.array(x[:, :, None, None, :]), mx.array(ids), sorted_indices=True)
        mx.eval(y)
        broadcast = np.broadcast_to(x[:, :, None, :], (2, 3, 2, 128)).reshape(-1, 128)
        ref = oracle(broadcast, q, su, sv, ids.reshape(-1), 15).reshape(2, 3, 2, 1, 128)
        np.testing.assert_allclose(np.asarray(y), ref, rtol=0.02, atol=0.015)
        bad = layer(mx.array(x[0, :2, None, :]), mx.array([-1, 2], dtype=mx.int32))
        mx.eval(bad)
        self.assertTrue(np.isnan(np.asarray(bad)).all())

    @unittest.skipUnless(mx.metal.is_available(), "requires Metal")
    def test_prefill_sorted_and_unsorted(self):
        rng = np.random.default_rng(101)
        for hw, win in [(32, 16), (42, 15), (48, 12), (64, 16)]:
            q = rng.integers(0, 65536, (2, 8, 8, hw), dtype=np.uint16)
            su = np.ones((2, 128), np.float16)
            sv = np.full((2, 128), 0.1, np.float16)
            x = rng.normal(size=(53, 128)).astype(np.float16)
            ids = rng.integers(0, 2, 53, dtype=np.uint32)
            layer = Exl3SwitchLinear(
                mx.array(q), mx.array(su), mx.array(sv), Exl3Spec(hw, win)
            )
            for sorted_indices in (False, True):
                if sorted_indices:
                    order = np.argsort(ids)
                    x, ids = x[order], ids[order]
                y = layer(
                    mx.array(x[:, None, :]),
                    mx.array(ids),
                    sorted_indices=sorted_indices,
                )
                mx.eval(y)
                np.testing.assert_allclose(
                    np.asarray(y[:, 0, :]),
                    oracle(x, q, su, sv, ids, win),
                    rtol=0.02,
                    atol=0.015,
                )

    def test_missing_bad_k_are_value_errors(self):
        for k in (None, False, "2.625", float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Exl3Spec.from_config(
                    {
                        "expert_quant": {
                            "format": "exl3",
                            "codebook": "mcg",
                            "k": k,
                            "out_scales": "svh",
                        }
                    }
                )

    def test_reject_unsupported_spec(self):
        for q in [
            {"format": "exl3", "codebook": "mul2", "k": 2.625, "out_scales": "svh"},
            {"format": "exl3", "codebook": "mcg", "k": 2.63, "out_scales": "svh"},
            {
                "format": "exl3",
                "codebook": "mcg",
                "k": 2.625,
                "out_scales": "svh",
                "window": 17,
            },
        ]:
            with self.assertRaises(ValueError):
                Exl3Spec.from_config({"expert_quant": q})


if __name__ == "__main__":
    unittest.main()
