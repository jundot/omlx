"""Protect JANGH2 cache hook installation and model marking from regressions."""
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parents[1]

class LaunchContractTests(unittest.TestCase):
    def test_model_marked_before_return_from_load(self):
        src=(ROOT/'jangh2_bridge/loader.py').read_text()
        self.assertIn('model._jangh2_bridge_model = True',src)
        self.assertLess(src.index('model._jangh2_bridge_model = True'),src.index("log.info('JANGH2: strict load complete"))

    def test_batch_hook_precedes_server_start(self):
        src=(ROOT/'serve.py').read_text()
        self.assertLess(src.index('uninstall_cache_compat = install_cache_compat()'),src.index('result = candidates[0].load()()'))
        self.assertIn('cleanup.callback(uninstall_cache_compat)',src)
        ast.parse(src)

if __name__=='__main__': unittest.main()
