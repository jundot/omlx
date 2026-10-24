"""Regression tests for Allosaurus global hyper-connection norm aliases.

The checkpoint stores the final hyper_connection_mixer under the language_model
root, whereas MLX-VLM's model tree places it under language_model.model.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
import numpy as np

from jangh2_bridge.metadata import canonical, canonical_map
from jangh2_bridge.loader import recenter_residual_norms


class GlobalHCNamespaceTests(unittest.TestCase):
    def test_global_hyperconnection_checkpoint_aliases(self):
        target = "language_model.model.hyper_connection_mixer.hc_norm.weight"
        for name in (
            "language_model.hyper_connection_mixer.hc_norm.weight",
            "model.language_model.hyper_connection_mixer.hc_norm.weight",
            "model.hyper_connection_mixer.hc_norm.weight",
            "hyper_connection_mixer.hc_norm.weight",
            target,
        ):
            with self.subTest(name=name):
                self.assertEqual(canonical(name), target)

    def test_entire_global_mixer_namespace_maps(self):
        for leaf in (
            "hc_norm.weight",
            "input_mix_weight_up.weight",
            "input_mix_weight_down.weight",
        ):
            self.assertEqual(
                canonical("language_model.hyper_connection_mixer." + leaf),
                "language_model.model.hyper_connection_mixer." + leaf,
            )

    def test_canonical_map_refuses_duplicate_aliases(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous tensor aliases"):
            canonical_map({
                "language_model.hyper_connection_mixer.hc_norm.weight": np.zeros((4,)),
                "language_model.model.hyper_connection_mixer.hc_norm.weight": np.ones((4,)),
            })

    def test_recenter_global_norm_after_mapping_preserves_others(self):
        class Residual: pass
        class Ordinary: pass
        model = SimpleNamespace(named_modules=lambda: [
            ("language_model.model.hyper_connection_mixer.hc_norm", Residual()),
            ("language_model.model.layers.0.input_layernorm", Ordinary()),
        ])
        root = "language_model.hyper_connection_mixer.hc_norm.weight"
        unrelated = "language_model.model.layers.0.input_layernorm.weight"
        raw = {root: np.array([0.5, 1.25], dtype=np.float32),
               unrelated: np.array([4.0], dtype=np.float32)}
        mapped = canonical_map(raw)
        count = recenter_residual_norms(model, mapped, Residual, np)
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(mapped[canonical(root)], [-0.5, 0.25])
        np.testing.assert_array_equal(mapped[unrelated], [4.0])

    def test_unrelated_names_unchanged(self):
        for key in ("vision_tower.blocks.0.weight", "mtp.hyper_connection_mixer.hc_norm.weight"):
            self.assertEqual(canonical(key), key)


if __name__ == "__main__":
    unittest.main()
