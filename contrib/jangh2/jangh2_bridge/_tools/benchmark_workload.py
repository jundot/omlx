#!/usr/bin/env python3
"""Test the real HTTP server with distinct-prefix long prompts/concurrency/cancellation.

A successful run means these probes completed without detected marker mixing.
It is not a mathematical proof of cache isolation. Missing markers can also be
model instruction-following failures; outputs are hashed, never rewritten.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid
from jangh2_bridge.workloads import observed_request,make_prompt,aggregate_wave,validate_url
from jangh2_bridge.profiles import atomic_json


def rss_bytes(pid):
    if pid is None:return None
    try:
        return int(subprocess.check_output(['ps','-o','rss=','-p',str(pid)],text=True,timeout=5).strip())*1024
    except (OSError,ValueError,subprocess.SubprocessError):return None


def resolve_pid(url,explicit):
    if explicit:return explicit
    u=validate_url(url)
    if u.hostname not in ('localhost','127.0.0.1','::1') or not shutil.which('lsof'):return None
    try:
        values=subprocess.check_output(['lsof','-t','-nP',f'-iTCP:{u.port or (443 if u.scheme=="https" else 80)}','-sTCP:LISTEN'],text=True,timeout=5).split()
        pids=set(map(int,values));return next(iter(pids)) if len(pids)==1 else None
    except (OSError,ValueError,subprocess.SubprocessError):return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8500/v1')
    p.add_argument('--model',default='Allosaurus-v0.1-125B-A6B-JANGH2')
    p.add_argument('--label',required=True)
    p.add_argument('--tokenizer',help='Local checkpoint path for accurate input sizing; no network downloads')
    p.add_argument('--prompt-tokens',type=int,nargs='+',default=[256,4096])
    p.add_argument('--concurrency',type=int,nargs='+',default=[1,2,4,6])
    p.add_argument('--rounds',type=int,default=2);p.add_argument('--max-tokens',type=int,default=128)
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--idle-seconds',type=float,default=2)
    p.add_argument('--cancel',action='store_true',help='Disconnect one stream in each concurrent wave; partial tokens excluded from throughput')
    p.add_argument('--server-pid',type=int);p.add_argument('--report')
    a=p.parse_args();validate_url(a.url)
    if (not 1<=a.rounds<=20 or any(not 1<=n<=16 for n in a.concurrency)
            or any(not 0<=n<=32768 for n in a.prompt_tokens) or not 16<=a.max_tokens<=4096
            or not 1<=a.timeout<=3600 or not 0<=a.idle_seconds<=60 or (a.server_pid is not None and a.server_pid<1)):
        p.error('Invalid workload bounds')
    tokenizer=None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(Path(a.tokenizer).expanduser()),local_files_only=True,trust_remote_code=False)
    label=''.join(c if c.isalnum() or c in '-_' else '_' for c in a.label)
    path=a.report or str(Path.home()/'.omlx-pr364'/f'jangh2-workload-{label}.json')
    report={'schema':1,'status':'running','label':a.label,'model':a.model,'waves':[],
            'input_sizing':'local tokenizer' if tokenizer else 'approximate 4 characters/token',
            'note':'Unique request ID at start of each user prompt; fresh KV-prefix after the common role prefix, not cold model/OS/PLE caches. Cancellation measures client disconnect and recovery, not immediate server cancellation acknowledgement.'}
    key=os.environ.get('OMLX_API_KEY');pid=resolve_pid(a.url,a.server_pid);report['server_pid']=pid
    failed=False
    try:
        warm='RID_'+uuid.uuid4().hex[:12]
        print('Warmup (excluded)...',flush=True)
        observed_request(a.url,a.model,make_prompt(warm,64,tokenizer),warm,[warm],max_tokens=16,timeout=a.timeout,api_key=key)
        time.sleep(a.idle_seconds);report['warm_idle_rss_bytes']=rss_bytes(pid)
        for prompt_tokens in a.prompt_tokens:
            for concurrency in a.concurrency:
                for iteration in range(a.rounds):
                    markers=['RID_'+uuid.uuid4().hex[:12] for _ in range(concurrency)]
                    prompts=[make_prompt(m,prompt_tokens,tokenizer) for m in markers]
                    barrier=threading.Barrier(concurrency);samples=[];stop=threading.Event()
                    def monitor():
                        while not stop.wait(.2):
                            value=rss_bytes(pid)
                            if value is not None:samples.append(value)
                    monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
                    def run(i):
                        barrier.wait(timeout=10)
                        try:
                            row=observed_request(a.url,a.model,prompts[i],markers[i],markers,
                                max_tokens=a.max_tokens,timeout=a.timeout,api_key=key,
                                cancel_after=2 if a.cancel and concurrency>1 and i==0 else None)
                            row['request_marker']=markers[i]
                            return row
                        except Exception as exc:return {'completed':False,'cancelled':False,'error':str(exc),'request_marker':markers[i]}
                    started=time.perf_counter()
                    try:
                        with ThreadPoolExecutor(max_workers=concurrency) as pool:rows=list(pool.map(run,range(concurrency)))
                    finally:stop.set();monitor_thread.join(timeout=6)
                    elapsed=time.perf_counter()-started
                    failures=[r for r in rows if 'error' in r or (not r.get('cancelled') and
                        (not r.get('isolation',{}).get('marker_present') or r.get('isolation',{}).get('foreign_markers')))]
                    failed |= bool(failures)
                    row={'input_target_tokens':prompt_tokens,'concurrency':concurrency,'iteration':iteration,
                         'requests':rows,'summary':aggregate_wave(rows,elapsed),'probe_failures':len(failures),
                         'sampled_peak_rss_bytes':max(samples) if samples else None}
                    if a.cancel and concurrency>1:
                        marker='RID_'+uuid.uuid4().hex[:12]
                        recovery=observed_request(a.url,a.model,make_prompt(marker,64,tokenizer),marker,[marker],
                            max_tokens=32,timeout=a.timeout,api_key=key)
                        row['recovery']=recovery
                        failed |= not recovery['completed'] or not recovery['isolation']['marker_present']
                    time.sleep(a.idle_seconds);row['idle_rss_bytes']=rss_bytes(pid)
                    report['waves'].append(row);atomic_json(path,report)
                    print(json.dumps({k:row[k] for k in ('input_target_tokens','concurrency','iteration','summary','probe_failures','idle_rss_bytes')}),flush=True)
        report['status']='failed' if failed else 'passed'
        after=[w['idle_rss_bytes'] for w in report['waves'] if w['idle_rss_bytes'] is not None]
        baseline=report['warm_idle_rss_bytes']
        report['final_idle_rss_growth_bytes']=after[-1]-baseline if after and baseline is not None else None
        report['memory_note']='RSS includes retained allocator, prefix and PLE caches. Growth is reported, not automatically classified as a leak. No private server memory endpoint is assumed.'
    except BaseException as exc:report.update(status='failed',error=str(exc));raise
    finally:atomic_json(path,report);print('Workload report:',path,flush=True)
    if failed:raise SystemExit('One or more isolation/compliance/recovery probes failed; inspect report and server logs')

if __name__=='__main__':main()
