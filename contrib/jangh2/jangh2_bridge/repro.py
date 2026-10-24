"""Capture helpers: pinned requirements and hashed, secret-excluding source copies."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import re
import shutil
from urllib.parse import urlsplit

EXCLUDED={'.git','.venv','venv','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache',
          'node_modules','dist','build','.DS_Store','.env','secrets.json','credentials.json'}
EXTENSIONS={'.py','.pyi','.toml','.md','.txt','.json','.yaml','.yml','.js','.mjs','.html',
            '.css','.jinja','.jinja2','.j2','.sh','.svg','.png','.ico','.webp','.metal','.h','.cpp'}


def requirement(name,version,direct):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.!+_-]*',version):
        raise ValueError('Invalid package name/version in environment metadata')
    if not direct:return f'{name}=={version}'
    url=direct.get('url','');u=urlsplit(url)
    if u.scheme!='https' or not u.hostname or u.username or u.password or u.query or u.fragment:
        raise ValueError('Refusing unportable or credential-bearing dependency URL; capture a wheel instead')
    vcs=direct.get('vcs_info')
    if vcs:
        commit=vcs.get('commit_id','')
        if vcs.get('vcs')!='git' or not re.fullmatch('[0-9a-fA-F]{40,64}',commit):
            raise ValueError('VCS dependency lacks an immutable git commit')
        sub=direct.get('subdirectory')
        if sub and (not re.fullmatch(r'[A-Za-z0-9_./-]+',sub) or '..' in Path(sub).parts):
            raise ValueError('Unsupported VCS subdirectory')
        return f'{name} @ git+{url}@{commit}'+(f'#subdirectory={sub}' if sub else '')
    archive=direct.get('archive_info',{})
    sha=archive.get('hashes',{}).get('sha256')
    if sha is None and archive.get('hash','').startswith('sha256='):sha=archive['hash'][7:]
    if not sha or not re.fullmatch('[0-9a-fA-F]{64}',sha):
        raise ValueError('Direct archive dependency lacks a sha256 digest')
    return f'{name} @ {url}#sha256={sha}'


def copy_source(source,destination):
    source=Path(source).resolve();destination=Path(destination)
    if destination.resolve().is_relative_to(source):raise ValueError('Snapshot destination must be outside its source')
    destination.mkdir(parents=True,exist_ok=False);manifest={}
    for directory,dirs,files in os.walk(source,followlinks=False):
        dirs[:]=[d for d in dirs if d not in EXCLUDED and not d.endswith('.egg-info')]
        for d in dirs:
            if (Path(directory)/d).is_symlink():raise ValueError('Source contains a directory symlink; snapshot the real sources explicitly')
        for filename in files:
            p=Path(directory)/filename
            if filename in EXCLUDED or filename.startswith('.env') or (p.suffix.lower() not in EXTENSIONS and filename not in ('LICENSE','NOTICE','Makefile','requirements.lock')):
                continue
            if p.is_symlink():raise ValueError('Source contains a file symlink; snapshot the real sources explicitly')
            relative=p.relative_to(source);out=destination/relative;out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,out);manifest[relative.as_posix()]=hashlib.sha256(out.read_bytes()).hexdigest()
    return manifest


def verify_manifest(root,manifest):
    root=Path(root).resolve()
    for name,digest in manifest.items():
        p=root/name
        if not p.resolve().is_relative_to(root) or p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Snapshot integrity mismatch: {name}')
