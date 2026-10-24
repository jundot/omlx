from pathlib import Path
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]

class CLITests(unittest.TestCase):
    def test_installed_dispatch_does_not_load_mlx_for_help_and_restores_argv(self):
        from jangh2_bridge.cli import main
        old=['jangh2-bridge','profile','--help']
        with patch.object(sys,'argv',old),patch('jangh2_bridge.cli.import_module',return_value=SimpleNamespace(main=lambda:7)):
            self.assertEqual(main(),7);self.assertIs(sys.argv,old)
    def test_restore_script_syntax(self):
        import capture_environment
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'restore.sh';p.write_text(capture_environment.RESTORE)
            subprocess.run(['bash','-n',str(p)],check=True)
    def test_preparation_uses_same_python_stops_after_failure_and_records_it(self):
        import prepare
        with tempfile.TemporaryDirectory() as d:
            calls=[]
            def run(cmd,env):
                calls.append((cmd,env));return SimpleNamespace(returncode=0 if len(calls)==1 else 3)
            argv=['prepare','--model','/model','--output',d]
            with patch.object(sys,'argv',argv),patch.object(prepare.socket,'create_connection',side_effect=OSError), \
                 patch.object(prepare.subprocess,'run',side_effect=run):
                with self.assertRaises(SystemExit):prepare.main()
            self.assertEqual(len(calls),2);self.assertEqual(calls[0][0][0],sys.executable)
            report=json.loads((Path(d)/'workflow.json').read_text());self.assertEqual(report['status'],'failed')
            self.assertEqual(report['stages'][-1]['exit_code'],3)
    def test_prepare_refuses_running_server(self):
        import prepare
        from contextlib import nullcontext
        with patch.object(sys,'argv',['prepare','--model','/model']), \
             patch.object(prepare.socket,'create_connection',return_value=nullcontext()), \
             patch.object(prepare.subprocess,'run') as run:
            with self.assertRaises(SystemExit):prepare.main()
            run.assert_not_called()

    def test_namespaced_wheel_scripts_match_root_entrypoints(self):
        for p in ROOT.glob('*.py'):
            self.assertEqual(p.read_bytes(),(ROOT/'jangh2_bridge/_tools'/p.name).read_bytes(),p.name)
