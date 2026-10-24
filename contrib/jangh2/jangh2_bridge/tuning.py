"""Model-aware routing and conservative selection of measured tuning results."""
from __future__ import annotations
import numpy as np


def route_count(config,override):
    value=override if override is not None else config.get('text_config',config).get('num_experts_per_tok')
    if type(value) is not int or not 1<=value<=64:
        raise ValueError('Cannot determine routing top-k; supply --routes or num_experts_per_tok in config.json')
    return value


def route_sample(batch,tokens,experts,routes,*,seed):
    if not 1<=routes<=experts or min(batch,tokens,experts)<1:raise ValueError('Invalid route sample dimensions')
    rng=np.random.default_rng(seed)
    # A real top-k router cannot choose the same expert twice for one token.
    return np.argsort(rng.random((batch,tokens,experts)),axis=-1)[...,:routes].astype(np.uint32)


def choose_decode(results,tuning):
    row=next((r for r in results if r['tokens']==1),None)
    if row is None:raise ValueError('Decode tuning requires an actual one-token measurement')
    names=[n for n in tuning if n in row['backends'] and not n.startswith('grouped')]
    if not names:raise ValueError('No measured decode candidates')
    winner=min(names,key=lambda n:row['backends'][n]['median_ms'])
    return {k:v for k,v in tuning[winner].items() if k in ('gate_rows','down_rows','sort_threshold')}


def choose_prefill(results,*,min_gain=.05,baseline='metal'):
    rows=sorted((r for r in results if r['tokens']>=8),key=lambda r:r['tokens'])
    # At least two prompt sizes avoid selecting on one lucky microbenchmark.
    if len(rows)<2:return {'prefill':'legacy'}
    winners=[]
    for tile in (2,4):
        name=f'grouped{tile}'
        if not all(name in r['backends'] for r in rows):continue
        gains=[r['backends'][baseline]['median_ms']/r['backends'][name]['median_ms'] for r in rows]
        if all(g>=1+min_gain for g in gains):
            winners.append((float(np.exp(np.mean(np.log(gains)))),tile))
    if not winners:return {'prefill':'legacy'}
    _,tile=max(winners)
    return {'prefill':'grouped','prefill_tile':tile,'prefill_min_tokens':rows[0]['tokens']}


def expert_count(config,override,*,real_model):
    available=config.get('text_config',config).get('num_experts') if real_model else 2048
    if type(available) is not int or not 1<=available<=2048:
        raise ValueError('Missing/unsupported expert population in checkpoint configuration')
    count=override if override is not None else (available if real_model else 32)
    if type(count) is not int or not 1<=count<=available:
        raise ValueError('Requested expert sample exceeds checkpoint population')
    return count
