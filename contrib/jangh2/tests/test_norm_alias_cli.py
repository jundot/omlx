"""Fast verification of the index-only preflight key scan."""
from __future__ import annotations

import unittest
from check_norm_aliases import matching_raw_keys


class NormAliasCLITests(unittest.TestCase):
    def test_aliases_from_raw_checkpoint_keys(self):
        self.assertEqual(matching_raw_keys([
            'language_model.hyper_connection_mixer.hc_norm.weight',
            'language_model.layers.1.ple.ngram_embedding.shards.0.weight',
        ]), ('language_model.hyper_connection_mixer.hc_norm.weight',))

    def test_missing_global_mixer_does_not_fabricate_key(self):
        self.assertEqual(matching_raw_keys(['language_model.layers.0.input_layernorm.weight']), ())

    def test_ambiguous_aliases_remain_visible(self):
        keys = ['language_model.hyper_connection_mixer.hc_norm.weight',
                'language_model.model.hyper_connection_mixer.hc_norm.weight']
        self.assertEqual(len(matching_raw_keys(keys)), 2)
