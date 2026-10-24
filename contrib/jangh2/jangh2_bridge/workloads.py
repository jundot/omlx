"""Bounded HTTP workload probes: distinct prefixes, output isolation and cancellation.

Generated prompts only; API keys and raw response bodies are never put in reports.
These are behavioural checks, not a proof of general scheduler correctness.
"""
from __future__ import annotations
import hashlib
import json
import statistics
import time
from urllib.parse import urlsplit
from urllib.request import Request,urlopen
from .api_benchmark import consume_events

class _Cancel(Exception):pass


def validate_url(url):
    p=urlsplit(url)
    if p.scheme not in ('http','https') or not p.hostname or p.username or p.password or p.query or p.fragment:
        raise ValueError('Use an http(s) API base URL without embedded credentials/query/fragment')
    return p


def assess_output(content,marker,all_markers):
    return {'marker_present':marker in content,
            'foreign_markers':[m for m in all_markers if m!=marker and m in content]}


def make_prompt(marker,target_tokens,tokenizer=None):
    if not 0<=target_tokens<=32768:raise ValueError('Input budget must be 0..32768 tokens')
    header=f'Request {marker}\n'
    instruction=(f'\nBegin your answer with exactly {marker} on its own line. '
                 'Then write a complete Python bounded LRU cache and four unit tests. '
                 'Do not repeat request identifiers other than your own.\n')
    if tokenizer is None:
        # Clearly reported as approximate unless a local tokenizer is supplied.
        filler=('context_value = 17  # read-only benchmark context\n'*max(1,target_tokens))[:max(0,target_tokens*4-len(header+instruction))]
    else:
        overhead=len(tokenizer.encode(header+instruction,add_special_tokens=False))
        encoded=tokenizer.encode('context_value = 17  # read-only benchmark context\n'*max(1,target_tokens),add_special_tokens=False)
        filler=tokenizer.decode(encoded[:max(0,target_tokens-overhead)])
    return header+filler+instruction


def observed_request(url,model,prompt,marker,all_markers,*,max_tokens=128,timeout=120,
                     api_key=None,cancel_after=None):
    validate_url(url)
    if cancel_after is not None and cancel_after<1:raise ValueError('cancel_after must be positive')
    payload={'model':model,'messages':[{'role':'user','content':prompt}],
             'max_tokens':max_tokens,'temperature':0,'stream':True,
             'chat_template_kwargs':{'enable_thinking':False},
             'stream_options':{'include_usage':True}}
    headers={'Content-Type':'application/json','Accept':'text/event-stream'}
    if api_key:headers['Authorization']='Bearer '+api_key
    request=Request(url.rstrip('/')+'/chat/completions',json.dumps(payload).encode(),headers,method='POST')
    content=[];events=0;chars=0;started=time.perf_counter();first=None
    def observer(value):
        nonlocal events,chars,first
        for choice in value.get('choices',[]):
            text=(choice.get('delta') or {}).get('content')
            if isinstance(text,str) and text:
                if first is None:first=time.perf_counter()-started
                events+=1;chars+=len(text)
                if chars>4*1024*1024:raise RuntimeError('Probe response exceeds the 4MiB content limit')
                content.append(text)
                if cancel_after is not None and events>=cancel_after:raise _Cancel()
    try:
        with urlopen(request,timeout=timeout) as response:
            row=consume_events(response,started=started,observer=observer,deadline=started+timeout)
        row['cancelled']=False
    except _Cancel:
        # Exiting urlopen's context closes this connection, not the server process.
        row={'completed':False,'cancelled':True,'total_seconds':time.perf_counter()-started,
             'completion_tokens':None,'first_content_seconds':first}
    text=''.join(content)
    row['isolation']=assess_output(text,marker,all_markers)
    row['content_sha256']=hashlib.sha256(text.encode()).hexdigest()
    row['content_characters']=len(text)
    return row


def aggregate_wave(rows,wall_seconds):
    completed=[r for r in rows if r.get('completed')]
    known=[r['completion_tokens'] for r in completed if type(r.get('completion_tokens')) is int]
    first=[r['first_content_seconds'] for r in completed if r.get('first_content_seconds') is not None]
    return {'wall_seconds':wall_seconds,'completed':len(completed),
            'cancelled':sum(bool(r.get('cancelled')) for r in rows),
            'aggregate_tokens_per_second':sum(known)/wall_seconds if wall_seconds>0 and len(known)==len(completed) and completed else None,
            'median_first_answer_seconds':statistics.median(first) if first else None,
            'truncated':sum(bool(r.get('truncated')) for r in rows)}
