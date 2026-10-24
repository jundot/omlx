"""A stock MLX-VLM PLE sanitizer emits legacy names not matching module.parameters()."""
import unittest
import inspect
import numpy as np

from jangh2_bridge.metadata import reconcile_ple_module_parameters

PREFIX = 'language_model.model.layers.1.ple.'
PAIRS = (
    ('conv1d_weight', 'conv1d.weight'),
    ('layer_multipliers', 'ple_embedding.layer_multipliers'),
    ('ngram_heads_offsets', 'ple_embedding.ngram_heads_offsets'),
    ('ngram_heads_vocab_sizes', 'ple_embedding.ngram_heads_vocab_sizes'),
)

class PLERuntimeNamesTests(unittest.TestCase):
    def test_real_checkpoint_mismatches_are_reconciled_exactly(self):
        names = {PREFIX + target for source, target in PAIRS}
        names.add('language_model.model.layers.0.self_attn.q_proj.weight')
        supplied = {PREFIX + source: np.full((4,), i, dtype=np.float32)
                    for i, (source, target) in enumerate(PAIRS)}
        ordinary = np.full((4,), 99, dtype=np.float32)
        supplied['language_model.model.layers.0.self_attn.q_proj.weight'] = ordinary
        original_values = [supplied[PREFIX + name] for name, _ in PAIRS]
        result, changes = reconcile_ple_module_parameters(supplied, names)
        self.assertEqual(changes, 4)
        self.assertEqual(set(result), names)
        for (_, target), original in zip(PAIRS, original_values):
            self.assertIs(result[PREFIX + target], original)
        self.assertIs(result['language_model.model.layers.0.self_attn.q_proj.weight'], ordinary)
        # Do not mutate the dictionary passed by the caller.
        self.assertEqual(set(supplied), set(PREFIX + name for name, _ in PAIRS) | {'language_model.model.layers.0.self_attn.q_proj.weight'})

    def test_multiple_ple_layers(self):
        names = {f'language_model.model.layers.{i}.ple.conv1d.weight' for i in (1, 6)}
        supplied = {f'language_model.model.layers.{i}.ple.conv1d_weight': np.array([i]) for i in (1, 6)}
        result, changes = reconcile_ple_module_parameters(supplied, names)
        self.assertEqual(changes, 2)
        self.assertEqual(set(result), names)

    def test_collision_with_canonical_source_fails_closed(self):
        old = PREFIX + 'conv1d_weight'
        new = PREFIX + 'conv1d.weight'
        with self.assertRaisesRegex(ValueError, 'Ambiguous PLE parameter alias'):
            reconcile_ple_module_parameters({old: np.zeros(2), new: np.zeros(2)}, {new})

    def test_unknown_module_layout_fails_closed(self):
        old = PREFIX + 'layer_multipliers'
        with self.assertRaisesRegex(ValueError, 'Unrecognised PLE runtime parameter'):
            reconcile_ple_module_parameters({old: np.zeros(2)}, {'different.ple_embedding.layer_multipliers'})

    def test_unrelated_key_names_are_not_rewritten(self):
        names = {'vision_tower.ple.conv1d_weight', PREFIX + 'conv1d.weighty', 'other.layer_multipliers'}
        supplied = {name: np.ones(2) for name in names}
        result, changes = reconcile_ple_module_parameters(supplied, names)
        self.assertEqual(result, supplied)
        self.assertEqual(changes, 0)

    def test_full_loader_reconciles_before_strict_audit_and_load(self):
        from jangh2_bridge.loader import load_jangh2
        source = inspect.getsource(load_jangh2)
        self.assertLess(source.index('quantize_affine(model,config,weights)'),
                        source.index('reconcile_ple_module_parameters(weights, expected)'))
        self.assertLess(source.index('reconcile_ple_module_parameters(weights, expected)'),
                        source.index('compare_parameters(expected,weights)'))
        self.assertLess(source.index('compare_parameters(expected,weights)'),
                        source.index('model.load_weights(list(weights.items()),strict=True)'))

    def test_existing_canonical_names_unchanged(self):
        name = PREFIX + 'conv1d.weight'
        value = np.ones(3)
        result, changes = reconcile_ple_module_parameters({name: value}, {name})
        self.assertIs(result[name], value)
        self.assertEqual(changes, 0)

if __name__ == '__main__':
    unittest.main()
