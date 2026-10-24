import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

class LauncherLifecycleTests(unittest.TestCase):
    def run_launch(self, *, entries=1, entry_error=False, loader_error=False, chat_error=False):
        order=[]
        def cache_install():
            order.append('cache install');return lambda:order.append('cache undo')
        def loader_install(*args,**kwargs):
            order.append('loader install')
            if loader_error:raise RuntimeError('loader failed')
            return lambda:order.append('loader undo')
        def chat_install(*args,**kwargs):
            order.append('chat install')
            if chat_error:raise RuntimeError('chat failed')
            return lambda:order.append('chat undo')
        def entry():
            order.append('entry')
            if entry_error:raise RuntimeError('entry failed')
            return 0
        spec=importlib.util.spec_from_file_location('bridge_launcher_under_test',ROOT/'serve.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        candidates=[SimpleNamespace(name='omlx',load=lambda:entry) for _ in range(entries)]
        fake={'omlx':SimpleNamespace(__file__='/checkout/omlx/__init__.py'),
              'jangh2_bridge.loader':SimpleNamespace(install=loader_install),
              'jangh2_bridge.scheduler_compat':SimpleNamespace(install=cache_install),
              'jangh2_bridge.chat_compat':SimpleNamespace(install=chat_install)}
        with patch.dict(sys.modules,fake),patch.object(sys,'prefix','/venv'),patch.object(sys,'base_prefix','/base'),\
             patch.object(sys,'argv',['serve.py','--checkout','/checkout','--model','/models/test','serve']),\
             patch.object(module,'entry_points',return_value=candidates),patch.object(module,'version',return_value='test'):
            try:module.main()
            except RuntimeError:pass
        return order
    def test_entry_failure_rolls_back_all_hooks_in_reverse_order(self):
        self.assertEqual(self.run_launch(entry_error=True),
                         ['cache install','loader install','chat install','entry','chat undo','loader undo','cache undo'])
    def test_loader_failure_rolls_back_cache_hook(self):
        self.assertEqual(self.run_launch(loader_error=True),['cache install','loader install','cache undo'])
    def test_chat_failure_rolls_back_prior_hooks(self):
        self.assertEqual(self.run_launch(chat_error=True),['cache install','loader install','chat install','loader undo','cache undo'])
    def test_bad_entrypoint_never_installs_hooks(self):
        self.assertEqual(self.run_launch(entries=0),[])
    def test_success_also_tears_down(self):
        self.assertEqual(self.run_launch(),['cache install','loader install','chat install','entry','chat undo','loader undo','cache undo'])

if __name__=='__main__':unittest.main()
