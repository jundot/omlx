#!/usr/bin/env python3
"""Sequential on-Mac qualification/tuning workflow. Does not start or stop servers.
Writes a profile only after component tuning. Full-model comparison must pass
before using that profile. All subprocesses use the calling Python interpreter.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from jangh2_bridge.profiles import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True)
    p.add_argument('--output',default=str(Path.home()/'.omlx-pr364/validation'))
    p.add_argument('--iterations',type=int,default=7);p.add_argument('--server-port',type=int,default=8500)
    p.add_argument('--long-profile',action='store_true',help='Also profile fresh-KV 1024/4096-token prompts')
    a=p.parse_args()
    if not 2<=a.iterations<=50 or not 1<=a.server_port<=65535:p.error('Invalid iterations or port')
    try:
        with socket.create_connection(('127.0.0.1',a.server_port),timeout=.5):
            p.error('Stop the server before qualification/tuning; competing GPU work invalidates timings')
    except OSError:pass
    root=Path(__file__).resolve().parent;out=Path(a.output).expanduser();out.mkdir(parents=True,exist_ok=True)
    profile=out/'measured-profile.json';qualification=out/'qualification.json'
    env=os.environ.copy()
    # Validation is deliberately insulated from previous tuning overrides.
    for k in list(env):
        if k.startswith('OMLX_JANGH_METAL_') or k.startswith('OMLX_JANGH_PLE_') or k.startswith('OMLX_JANGH_PREFILL') or k=='OMLX_JANGH_PROFILE':env.pop(k)
    env['OMLX_JANGH_QUALIFICATION']=str(qualification)
    stages=[('qualification','qualify.py',['--report',str(qualification)]),
            ('tuning','benchmark_performance.py',['--model',a.model,'--tokens','1','32','128',
                 '--iterations',str(a.iterations),'--tune','--grouped','--write-profile',str(profile),
                 '--report',str(out/'expert-benchmark.json')]),
            ('whole-model-comparison','compare_model.py',['--model',a.model,'--tokens','64','--continuation','4',
                 '--report',str(out/'model-comparison.json')])]
    if a.long_profile:stages.append(('long-profile','profile_model.py',['--model',a.model,'--prompt-tokens','1024','4096','--report',str(out/'long-profile.json')]))
    report={'status':'failed','stages':[],'profile':str(profile)}
    try:
        for name,script,args in stages:
            active=env.copy()
            if name in ('whole-model-comparison','long-profile'):
                active['OMLX_JANGH_PROFILE']=str(profile)
                # Exercise device caching as well, but do not silently promote it:
                # component correctness is not evidence of a throughput benefit.
                active['OMLX_JANGH_PLE_CACHE_MB']='64';active['OMLX_JANGH_PLE_DEVICE_MB']='16'
            print('\n=== '+name+' ===',flush=True);start=time.perf_counter()
            proc=subprocess.run([sys.executable,str(root/script),*args],env=active)
            report['stages'].append({'name':name,'exit_code':proc.returncode,'seconds':time.perf_counter()-start})
            atomic_json(out/'workflow.json',report)
            if proc.returncode:raise SystemExit(f'{name} failed; do not enable this candidate profile')
        report['status']='passed';print('\nQualification and selected full-model checks passed. No server was started.',flush=True)
        print('Use OMLX_JANGH_QUALIFICATION='+str(qualification))
        print('Use OMLX_JANGH_PROFILE='+str(profile))
        print('Device PLE cache remains opt-in; compare workload timings before retaining it.')
    finally:atomic_json(out/'workflow.json',report)

if __name__=='__main__':main()
