import json
from pathlib import Path
import tempfile
import unittest
from jangh2_bridge.repro import requirement,copy_source,verify_manifest

class ReproTests(unittest.TestCase):
    def test_versions_and_vcs_commits_are_pinned(self):
        self.assertEqual(requirement('mlx','0.32.4',None),'mlx==0.32.4')
        direct={'url':'https://github.com/a/b.git','vcs_info':{'vcs':'git','commit_id':'a'*40}}
        self.assertEqual(requirement('mlx-lm','dev',direct),'mlx-lm @ git+https://github.com/a/b.git@'+'a'*40)
    def test_credentials_and_unpinned_vcs_are_rejected(self):
        for url in ('https://secret@example.org/repo','https://example.org/repo?token=x','file:///private/tree'):
            with self.assertRaises(ValueError):requirement('pkg','1',{'url':url})
        with self.assertRaises(ValueError):requirement('pkg','1',{'url':'https://example.org/repo','vcs_info':{'vcs':'git'}})
    def test_modified_source_is_copied_and_secrets_env_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source';root.mkdir();dest=Path(tmp)/'snapshot'
            (root/'module.py').write_text('changed=True\n');(root/'.env').write_text('SECRET=do-not-copy')
            (root/'__pycache__').mkdir();(root/'__pycache__/cache.pyc').write_bytes(b'x')
            manifest=copy_source(root,dest)
            self.assertEqual((dest/'module.py').read_text(),'changed=True\n')
            self.assertFalse((dest/'.env').exists());self.assertEqual(len(manifest),1)
            verify_manifest(dest,manifest)
            (dest/'module.py').write_text('tampered')
            with self.assertRaises(ValueError):verify_manifest(dest,manifest)
    def test_symlinks_cannot_escape_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source';root.mkdir();(Path(tmp)/'secret').write_text('secret')
            (root/'module.py').symlink_to(Path(tmp)/'secret')
            with self.assertRaises(ValueError):copy_source(root,Path(tmp)/'copy')
