import unittest
from jangh2_bridge.tuning import expert_count

class ExpertPopulationTests(unittest.TestCase):
    def test_real_model_default_uses_actual_full_population(self):
        self.assertEqual(expert_count({'text_config':{'num_experts':512}},None,real_model=True),512)
    def test_explicit_subset_is_supported_for_component_exploration(self):
        self.assertEqual(expert_count({'num_experts':512},32,real_model=True),32)
    def test_invalid_or_missing_population_is_not_guessed(self):
        for config,override in [({},None),({'num_experts':4},10),({'num_experts':0},None)]:
            with self.assertRaises(ValueError):expert_count(config,override,real_model=True)
    def test_synthetic_default_stays_small(self):
        self.assertEqual(expert_count({},None,real_model=False),32)
