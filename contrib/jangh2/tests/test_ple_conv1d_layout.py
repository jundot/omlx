"""Regression: actual Allosaurus PLE Conv1d has flattened single-channel kernels.

The checkpoint stores the four temporal taps as (out, 4); the live
mlx.nn.Conv1d module expects (out, 4, 1). Mapping the name alone isn't enough.
"""
import unittest
import numpy as np

from jangh2_bridge.loader import compare_parameters
from jangh2_bridge.metadata import reconcile_ple_module_parameters

PREFIX = 'language_model.model.layers.1.ple.'
SRC = PREFIX + 'conv1d_weight'
TARGET = PREFIX + 'conv1d.weight'


class PLEConv1dLayoutTests(unittest.TestCase):
    def test_real_checkpoint_flat_conv_kernel_reshapes_losslessly(self):
        checkpoint = np.arange(12 * 4, dtype=np.float16).reshape(12, 4)
        expected = {TARGET: np.empty((12, 4, 1), dtype=np.float16)}
        supplied = {SRC: checkpoint}
        out, changed = reconcile_ple_module_parameters(supplied, expected)
        self.assertEqual(changed, 1)
        self.assertEqual(out[TARGET].shape, (12, 4, 1))
        np.testing.assert_array_equal(out[TARGET][..., 0], checkpoint)
        self.assertEqual(out[TARGET].dtype, checkpoint.dtype)
        compare_parameters(expected, out)
        # Original payload must remain unchanged.
        self.assertIs(supplied[SRC], checkpoint)
        self.assertEqual(checkpoint.shape, (12, 4))

    def test_real_10240_by_4_shape_is_a_metadata_only_reshape(self):
        flat = np.arange(10240 * 4, dtype=np.float16).reshape(10240, 4)
        expected = {TARGET: np.empty((10240, 4, 1), dtype=np.float16)}
        result, remapped = reconcile_ple_module_parameters({SRC: flat}, expected)
        self.assertEqual(remapped, 1)
        self.assertEqual(result[TARGET].shape, (10240, 4, 1))
        self.assertTrue(np.shares_memory(result[TARGET], flat))
        np.testing.assert_array_equal(result[TARGET][..., 0], flat)
        compare_parameters(expected, result)

    def test_canonical_conv_name_also_reshapes_when_flat(self):
        flat = np.arange(20, dtype=np.float32).reshape(5, 4)
        expected = {TARGET: np.empty((5, 4, 1), dtype=np.float32)}
        result, changes = reconcile_ple_module_parameters({TARGET: flat}, expected)
        self.assertEqual(changes, 0)
        np.testing.assert_array_equal(result[TARGET][..., 0], flat)
        compare_parameters(expected, result)

    def test_matching_weight_never_reshaped(self):
        val = np.ones((5, 4, 1), dtype=np.float32)
        expected = {TARGET: np.empty((5, 4, 1), dtype=np.float32)}
        result, _ = reconcile_ple_module_parameters({SRC: val}, expected)
        self.assertIs(result[TARGET], val)

    def test_wrong_channel_or_kernel_refuses_conversion(self):
        expected = {TARGET: np.empty((5, 4, 2), dtype=np.float32)}
        source = np.ones((5, 4), dtype=np.float32)
        result, _ = reconcile_ple_module_parameters({SRC: source}, expected)
        self.assertIs(result[TARGET], source)
        with self.assertRaisesRegex(ValueError, 'Shape mismatches'):
            compare_parameters(expected, result)

    def test_wrong_width_refuses_conversion(self):
        expected = {TARGET: np.empty((5, 4, 1), dtype=np.float32)}
        source = np.ones((5, 5), dtype=np.float32)
        result, _ = reconcile_ple_module_parameters({SRC: source}, expected)
        self.assertIs(result[TARGET], source)
        with self.assertRaisesRegex(ValueError, 'Shape mismatches'):
            compare_parameters(expected, result)

    def test_other_tensors_untouched_and_still_strict(self):
        other = PREFIX + 'ple_embedding.layer_multipliers'
        src = PREFIX + 'layer_multipliers'
        expected = {TARGET: np.empty((5, 4, 1)), other: np.empty((3, 1))}
        a = np.ones((5, 4))
        b = np.ones((3,))
        result, changes = reconcile_ple_module_parameters({SRC: a, src: b}, expected)
        self.assertEqual(changes, 2)
        self.assertIs(result[other], b)
        with self.assertRaisesRegex(ValueError, 'Shape mismatches'):
            compare_parameters(expected, result)

    def test_name_set_without_shape_metadata_keeps_existing_behavior(self):
        value = np.ones((5, 4))
        result, changes = reconcile_ple_module_parameters({SRC: value}, {TARGET})
        self.assertEqual(changes, 1)
        self.assertIs(result[TARGET], value)


if __name__ == '__main__':
    unittest.main()
