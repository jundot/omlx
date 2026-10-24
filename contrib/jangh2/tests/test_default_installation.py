"""Installer rendering/atomic writes use no MLX, launchd or real HOME."""
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

def installer():
    path=ROOT/'scripts/install_default.py'
    spec=importlib.util.spec_from_file_location('default_installation_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class DefaultInstallationTests(unittest.TestCase):
    def options(self,home):
        return dict(home=home,checkout=home/'source with spaces',bridge=home/'bridge',
                    model_root=home/'models',target_model=home/'models/Org/JANGH2',
                    base_path=home/'state',port=8500,profile=home/'state/profile.json',
                    qualification=home/'state/qualification.json')

    def test_shell_syntax_and_exact_quoted_paths(self):
        mod=installer()
        with tempfile.TemporaryDirectory() as d:
            home=Path(d)/"home with ' apostrophe";home.mkdir()
            text=mod.render_wrapper(**self.options(home))
            p=home/'wrapper';p.write_text(text)
            result=subprocess.run(['bash','-n',str(p)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('serve.py',text)

    def test_serving_uses_exact_python_and_all_model_root(self):
        mod=installer()
        with tempfile.TemporaryDirectory() as d:
            home=Path(d);options=self.options(home)
            fake=options['checkout']/'.venv/bin/python';fake.parent.mkdir(parents=True)
            fake.write_text('#!'+sys.executable+'\nimport json,os,sys\nprint(json.dumps({"argv":sys.argv[1:],"cache":os.environ["OMLX_JANGH_PLE_DEVICE_MB"]}))\n')
            fake.chmod(0o755)
            (options['bridge']).mkdir();(options['bridge']/'serve.py').touch()
            options['target_model'].mkdir(parents=True);(options['target_model']/'config.json').write_text('{}')
            p=home/'wrapper';p.write_text(mod.render_wrapper(**options))
            env={k:v for k,v in os.environ.items() if not k.startswith('OMLX_')}
            out=subprocess.run(['bash',str(p),'serve','--log-level','debug'],env=env,capture_output=True,text=True)
            self.assertEqual(out.returncode,0,out.stderr)
            actual=json.loads(out.stdout);a=actual['argv']
            self.assertEqual(a[a.index('--model-dir')+1],str(options['model_root']))
            self.assertEqual(a[a.index('--model')+1],str(options['target_model']))
            self.assertEqual(a[-2:],['--log-level','debug'])
            self.assertEqual(actual['cache'],'0')

    def test_launcher_records_explicit_base_path_port_and_host(self):
        text=installer().render_wrapper(**self.options(Path('/fake/home')))
        self.assertIn('--base-path "$BASE_PATH"',text)
        self.assertIn('--port "$PORT"',text)
        self.assertIn('--host 127.0.0.1',text)

    def test_service_is_separate_from_homebrew_and_contains_no_secrets(self):
        mod=installer();home=Path('/fake/home')
        data=mod.launch_agent(home=home,wrapper=home/'.local/bin/omlx',base_path=home/'state')
        decoded=plistlib.loads(plistlib.dumps(data))
        self.assertEqual(decoded['Label'],'local.omlx.jangh2')
        self.assertEqual(decoded['ProgramArguments'],[str(home/'.local/bin/omlx'),'serve'])
        self.assertNotIn('API_KEY',str(decoded))
        self.assertTrue(decoded['RunAtLoad'])
        self.assertTrue(decoded['KeepAlive'])

    def test_replace_preserves_backup_and_does_not_follow_symlink(self):
        mod=installer()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);original=root/'brew-command';original.write_text('brew unchanged')
            dest=root/'omlx';dest.symlink_to(original)
            backup=mod.write_with_backup(dest,b'new launcher',0o755)
            self.assertEqual(original.read_text(),'brew unchanged')
            self.assertFalse(dest.is_symlink());self.assertTrue(backup.is_symlink())
            self.assertEqual(dest.read_text(),'new launcher')

    def test_installer_help_does_not_import_mlx(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/install_default.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--with-service',result.stdout)

    def test_no_root_or_source_changes_in_wrapper(self):
        text=installer().render_wrapper(**self.options(Path('/fake/home')))
        self.assertNotIn('sudo ',text)
        self.assertNotIn('pip install',text)
        self.assertNotIn('brew services',text)
        self.assertIn('launchctl bootout',text)
        self.assertIn('local.omlx.jangh2',text)

    def test_start_stop_do_not_fall_through_to_brew_cli(self):
        mod=installer()
        text=mod.render_wrapper(**self.options(Path('/fake/home')))
        # Verify that lifecycle dispatch is implemented by this launcher.
        for command in ('start)','stop)','restart)','status)','logs)'):
            self.assertIn(command,text)
