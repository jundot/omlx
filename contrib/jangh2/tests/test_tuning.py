import unittest
from unittest.mock import patch
import numpy as np
from jangh2_bridge.tuning import route_count, route_sample, choose_prefill, choose_decode
from jangh2_bridge.profiles import validate_profile, source_fingerprint

class TuningTests(unittest.TestCase):
    def test_routes_from_checkpoint_and_explicit_override(self):
        self.assertEqual(route_count({'text_config':{'num_experts_per_tok':10}},None),10)
        self.assertEqual(route_count({},7),7)
        with self.assertRaises(ValueError):route_count({},None)
    def test_samples_have_no_duplicate_experts_per_token(self):
        x=route_sample(3,20,32,10,seed=1)
        self.assertEqual(x.shape,(3,20,10))
        self.assertTrue(all(len(set(r))==10 for r in x.reshape(-1,10)))
        np.testing.assert_array_equal(x,route_sample(3,20,32,10,seed=1))
        with self.assertRaises(ValueError):route_sample(1,4,4,10,seed=0)
    def test_prefill_requires_win_at_every_tested_larger_size(self):
        results=[{'tokens':32,'backends':{'metal':{'median_ms':10},'grouped4':{'median_ms':7}}},
                 {'tokens':128,'backends':{'metal':{'median_ms':20},'grouped4':{'median_ms':16}}}]
        win=choose_prefill(results,min_gain=.05)
        self.assertEqual(win['prefill_tile'],4);self.assertEqual(win['prefill_min_tokens'],32)
        results[1]['backends']['grouped4']['median_ms']=30
        self.assertEqual(choose_prefill(results,min_gain=.05),{'prefill':'legacy'})
    def test_decode_never_infers_from_prefill_only_run(self):
        with self.assertRaises(ValueError):choose_decode([{'tokens':8,'backends':{}}],{})
    def test_profile_rejects_wrong_fingerprint_model_and_arbitrary_env(self):
        fp={'code':'x','versions':{'mlx':'1'}}
        base={'schema':1,'fingerprint':fp,'config_sha256':'m','environment':{'OMLX_JANGH_METAL_GATE_ROWS':'4'}}
        self.assertEqual(validate_profile(base,fp,'m'),base['environment'])
        for change in ({'fingerprint':{}},{'config_sha256':'n'},{'environment':{'PYTHONPATH':'bad'}}):
            with self.assertRaises(ValueError):validate_profile({**base,**change},fp,'m')
    def test_fingerprint_deterministic(self):
        self.assertEqual(source_fingerprint(),source_fingerprint())

    def test_prefill_compares_with_selected_decode_geometry_not_default(self):
        rows=[{'tokens':n,'backends':{'metal':{'median_ms':10},
              'metal-g2-d2':{'median_ms':5},'grouped4':{'median_ms':7}}} for n in (32,128)]
        self.assertEqual(choose_prefill(rows,baseline='metal-g2-d2'),{'prefill':'legacy'})
