#!/usr/bin/env python3
"""Capture exact installed version/VCS pins, modified sources and model metadata.

Read-only with respect to the active environment/checkpoint. Writes only a new
snapshot directory. --wheelhouse additionally builds/downloads pinned dependency
wheels; it does NOT install them. Without wheels this is a source/version lock,
not a promise that all upstream artifacts will remain available indefinitely.
"""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import distributions,distribution,version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from jangh2_bridge.repro import requirement,copy_source,verify_manifest
from jangh2_bridge.profiles import atomic_json

RESTORE=r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$ROOT/reproduced-venv}"
PYTHON="${PYTHON:-python3.11}"
if [[ -e "$DEST" ]]; then echo "Refusing existing destination: $DEST" >&2; exit 1; fi
"$PYTHON" "$ROOT/verify_snapshot.py"
"$PYTHON" -m venv "$DEST"
if [[ -d "$ROOT/wheels" ]]; then
  "$DEST/bin/python" -m pip install --no-index --find-links "$ROOT/wheels" -r "$ROOT/requirements-versions.lock"
else
  "$DEST/bin/python" -m pip install -r "$ROOT/requirements.lock"
fi
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OMLX="__OMLX_VERSION__"
"$DEST/bin/python" -m pip install --no-deps --no-build-isolation -e "$ROOT/sources/omlx"
if [[ -f "$ROOT/sources/bridge/pyproject.toml" ]]; then
  "$DEST/bin/python" -m pip install --no-deps --no-build-isolation -e "$ROOT/sources/bridge"
fi
"$DEST/bin/python" - "$ROOT" <<'PYRESTORE'
from pathlib import Path
from importlib.metadata import distribution
import shutil,sys
root=Path(sys.argv[1]);src=root/'sources/qwen4_exp'
if src.is_dir():
    dst=Path(distribution('mlx-vlm').locate_file('mlx_vlm/models/qwen4_exp'))
    if dst.is_symlink(): dst.unlink()
    elif dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src,dst)
PYRESTORE
"$DEST/bin/python" -m pip check
echo "Restored into $DEST. Original environment untouched. Rerun hardware qualification before serving."
'''
VERIFY='''from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
for name,digest in json.loads((root/'SHA256.json').read_text()).items():
    p=root/name
    if not p.resolve().is_relative_to(root) or not p.is_file() or p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
        raise SystemExit('Snapshot verification failed: '+name)
print('Snapshot hashes verified')
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout',default=str(Path.home()/'src/omlx-pr364'))
    p.add_argument('--bridge',default=str(Path(__file__).resolve().parent))
    p.add_argument('--model',required=True);p.add_argument('--out',required=True)
    p.add_argument('--wheelhouse',action='store_true');p.add_argument('--hash-weights',action='store_true')
    a=p.parse_args();out=Path(a.out).expanduser().resolve();checkout=Path(a.checkout).expanduser().resolve();bridge=Path(a.bridge).expanduser().resolve();model=Path(a.model).expanduser().resolve()
    if out.exists():p.error('Snapshot output already exists; use a new directory')
    for source in (checkout,bridge,model):
        if not source.is_dir() or out.is_relative_to(source):p.error('Sources must exist and output must be outside them')
    if not (checkout/'omlx').is_dir() or not (bridge/'jangh2_bridge').is_dir():p.error('Wrong checkout/bridge directories')
    out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.jangh2-snapshot-',dir=out.parent))
    try:
        lock=[];versions=[];editable=[]
        for dist in distributions():
            name=dist.metadata.get('Name');ver=dist.version
            if not name:continue
            text=dist.read_text('direct_url.json');direct=json.loads(text) if text else None
            if name.lower().replace('_','-') in ('omlx','omlx-jangh2-bridge'):continue
            if direct and direct.get('dir_info',{}).get('editable'):
                editable.append(name);continue
            lock.append(requirement(name,ver,direct));versions.append(f'{name}=={ver}')
        if editable:raise RuntimeError('Additional editable dependencies need their own source snapshot: '+', '.join(editable))
        (stage/'requirements.lock').write_text('\n'.join(sorted(set(lock),key=str.lower))+'\n')
        (stage/'requirements-versions.lock').write_text('\n'.join(sorted(set(versions),key=str.lower))+'\n')
        (stage/'sources').mkdir()
        # Preserve all relevant modified/untracked source files, not just git HEAD.
        for src,name in ((checkout,'omlx'),(bridge,'bridge')):
            manifest=copy_source(src,stage/'sources'/name);verify_manifest(stage/'sources'/name,manifest)
        # Earlier troubleshooting may have added a symlink/unmanaged Qwen4 implementation.
        qwen=Path(distribution('mlx-vlm').locate_file('mlx_vlm/models/qwen4_exp')).resolve()
        if qwen.is_dir():copy_source(qwen,stage/'sources/qwen4_exp')
        modelmeta=stage/'model-metadata';modelmeta.mkdir()
        for name in ('config.json','model.safetensors.index.json','generation_config.json','tokenizer_config.json','chat_template.jinja'):
            if (model/name).is_file():shutil.copy2(model/name,modelmeta/name)
        index=json.loads((model/'model.safetensors.index.json').read_text());shards={}
        for name in sorted(set(index['weight_map'].values())):
            shard=(model/name).resolve()
            if not shard.is_relative_to(model):raise ValueError('Shard escapes checkpoint directory')
            item={'bytes':shard.stat().st_size}
            if a.hash_weights:
                h=hashlib.sha256()
                with shard.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
                item['sha256']=h.hexdigest()
            shards[name]=item
        check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
        try:head=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.SubprocessError):head=None
        report={'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'omlx_git_head':head,
                'pip_check_ok':check.returncode==0,'pip_check_output':check.stdout.strip(),
                'model_directory':str(model),'model_shards':shards,'model_payloads_fully_hashed':a.hash_weights,
                'scope':'Exact source copies plus package/VCS pins; configured API keys, environment values and runtime settings are not collected. Source files may contain hard-coded secrets; review before sharing. Model weight payloads are not copied.',
                'wheelhouse':a.wheelhouse}
        if check.returncode:raise RuntimeError('pip check failed; refusing to label an inconsistent environment reproducible')
        if a.wheelhouse:
            wheels=stage/'wheels';wheels.mkdir()
            subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--wheel-dir',str(wheels),
                            '-r',str(stage/'requirements.lock')],check=True)
        (stage/'restore.sh').write_text(RESTORE.replace('__OMLX_VERSION__',version('omlx')));(stage/'restore.sh').chmod(0o755)
        (stage/'verify_snapshot.py').write_text(VERIFY)
        atomic_json(stage/'environment.json',report)
        manifest={p.relative_to(stage).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in stage.rglob('*') if p.is_file()}
        atomic_json(stage/'SHA256.json',manifest)
        verify_manifest(stage,manifest);os.replace(stage,out)
        print('Captured:',out)
        print('No active environment, model weights, API keys or settings modified.')
        print('Restore command:',str(out/'restore.sh'),'NEW_VENV_DIRECTORY')
    finally:
        if stage.exists():shutil.rmtree(stage)

if __name__=='__main__':main()
