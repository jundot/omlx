"""Fingerprint-bound JSON profiles; data only, never sourced as shell code."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
from importlib.metadata import version,PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

ENV_KEYS={
 'gate_rows':'OMLX_JANGH_METAL_GATE_ROWS','down_rows':'OMLX_JANGH_METAL_DOWN_ROWS',
 'sort_threshold':'OMLX_JANGH_METAL_SORT_THRESHOLD','prefill':'OMLX_JANGH_PREFILL',
 'prefill_tile':'OMLX_JANGH_PREFILL_TILE','prefill_min_tokens':'OMLX_JANGH_PREFILL_MIN_TOKENS',
 'ple_cache_mib':'OMLX_JANGH_PLE_CACHE_MB','ple_open_files':'OMLX_JANGH_PLE_OPEN_FILES',
 'ple_device_mib':'OMLX_JANGH_PLE_DEVICE_MB'}


def atomic_json(path,value):
    path=Path(path).expanduser();path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as out:
            json.dump(value,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def source_fingerprint():
    root=Path(__file__).parent;digest=hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        digest.update(p.relative_to(root).as_posix().encode()+b'\0'+p.read_bytes())
    return digest.hexdigest()


def runtime_fingerprint():
    versions={}
    for name in ('mlx','mlx-lm','mlx-vlm','omlx','numpy','transformers'):
        try:versions[name]=version(name)
        except PackageNotFoundError:versions[name]=None
    hardware=platform.machine()
    if platform.system()=='Darwin':
        try:hardware=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True,timeout=5).strip()
        except (OSError,subprocess.SubprocessError):pass
    return {'code':source_fingerprint(),'versions':versions,'system':platform.system(),
            'hardware':hardware,'python':platform.python_version()}


def config_digest(model):return hashlib.sha256((Path(model).expanduser()/'config.json').read_bytes()).hexdigest()


def validate_profile(profile,fingerprint,config_sha256):
    if profile.get('schema')!=1 or profile.get('fingerprint')!=fingerprint:
        raise ValueError('Performance profile is stale for this bridge/runtime/hardware; re-qualify and retune')
    if profile.get('config_sha256')!=config_sha256:raise ValueError('Performance profile belongs to a different model configuration')
    env=profile.get('environment')
    if not isinstance(env,dict) or set(env)-set(ENV_KEYS.values()) or not all(isinstance(v,str) for v in env.values()):
        raise ValueError('Performance profile contains unsupported environment keys/values')
    return env


def require_qualification(path=None):
    path=Path(path or os.environ.get('OMLX_JANGH_QUALIFICATION',
             str(Path.home()/'.omlx-pr364/jangh2-qualification.json'))).expanduser()
    try:report=json.loads(path.read_text())
    except (OSError,ValueError) as exc:raise RuntimeError(f'Run qualify.py before enabling grouped/device-cache paths; missing report: {path}') from exc
    if report.get('status')!='passed' or report.get('fingerprint')!=runtime_fingerprint() or not report.get('actual_metal'):
        raise RuntimeError('Qualification report is failed, simulated or stale; rerun qualify.py on this Mac')
    return report


@contextmanager
def launch_profile(model):
    """Optional profile defaults; explicit environment variables retain priority."""
    path=os.environ.get('OMLX_JANGH_PROFILE');changed=[]
    try:
        if path:
            env=validate_profile(json.loads(Path(path).expanduser().read_text()),runtime_fingerprint(),config_digest(model))
            for key,value in env.items():
                if key not in os.environ:os.environ[key]=value;changed.append(key)
        from .performance import PerformanceOptions
        options=PerformanceOptions.from_environment(fast_defaults=True)
        if options.prefill=='grouped' or options.ple_device_mib:
            require_qualification()
        yield options
    finally:
        for key in changed:os.environ.pop(key,None)
