import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from jangh2_bridge.profiles import launch_profile,require_qualification,atomic_json

class ProfileSafetyTests(unittest.TestCase):
    def test_explicit_environment_wins_and_defaults_restore_after_exception(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'profile.json';p.write_text(json.dumps({'schema':1,'fingerprint':{'code':'x'},
                'config_sha256':'m','environment':{'OMLX_JANGH_METAL_GATE_ROWS':'8','OMLX_JANGH_PLE_CACHE_MB':'64'}}))
            with patch.dict(os.environ,{'OMLX_JANGH_PROFILE':str(p),'OMLX_JANGH_METAL_GATE_ROWS':'2'},clear=True), \
                 patch('jangh2_bridge.profiles.runtime_fingerprint',return_value={'code':'x'}), \
                 patch('jangh2_bridge.profiles.config_digest',return_value='m'):
                with self.assertRaisesRegex(RuntimeError,'later failure'):
                    with launch_profile('/unused') as options:
                        self.assertEqual(options.gate_rows,2);self.assertEqual(options.ple_cache_mib,64)
                        raise RuntimeError('later failure')
                self.assertNotIn('OMLX_JANGH_PLE_CACHE_MB',os.environ)
                self.assertEqual(os.environ['OMLX_JANGH_METAL_GATE_ROWS'],'2')
    def test_experimental_paths_require_actual_matching_report(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'q.json'
            with patch('jangh2_bridge.profiles.runtime_fingerprint',return_value={'code':'x'}):
                for body in ({'status':'passed','actual_metal':False,'fingerprint':{'code':'x'}},
                             {'status':'passed','actual_metal':True,'fingerprint':{'code':'old'}},
                             {'status':'failed','actual_metal':True,'fingerprint':{'code':'x'}}):
                    atomic_json(p,body)
                    with self.assertRaises(RuntimeError):require_qualification(p)
                atomic_json(p,{'status':'passed','actual_metal':True,'fingerprint':{'code':'x'}})
                self.assertTrue(require_qualification(p)['actual_metal'])
    def test_legacy_path_never_requires_gpu_qualification(self):
        with patch.dict(os.environ,{},clear=True),patch('jangh2_bridge.profiles.require_qualification') as check:
            with launch_profile('/not-needed') as options:
                self.assertEqual(options.prefill,'legacy');self.assertEqual(options.ple_device_mib,0)
            check.assert_not_called()
    def test_no_shell_evaluation_of_profile(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'profile.json';p.write_text(json.dumps({'schema':1,'fingerprint':{},'config_sha256':'m',
                'environment':{'OMLX_JANGH_PLE_CACHE_MB':'$(echo unsafe)'}}))
            with patch.dict(os.environ,{'OMLX_JANGH_PROFILE':str(p)},clear=True), \
                 patch('jangh2_bridge.profiles.runtime_fingerprint',return_value={}), \
                 patch('jangh2_bridge.profiles.config_digest',return_value='m'):
                with self.assertRaises(ValueError):
                    with launch_profile('/unused'):pass
                self.assertNotIn('OMLX_JANGH_PLE_CACHE_MB',os.environ)
