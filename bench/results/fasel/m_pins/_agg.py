#!/usr/bin/env python
import json
import statistics as st
from pathlib import Path

ROOT = Path('bench/results/fasel/m_pins')

def load(n):
    d = json.loads((ROOT / (n + '.json')).read_text())
    rs = d.get('read_stats') or {}
    dec = rs.get('decode') or {}
    stg = dec.get('stages_us') or {}
    pin = d.get('pin') or {}
    ec = d.get('effective_config') or {}
    def g(k):
        v = (stg.get(k) or {})
        return v.get('p50'), v.get('p95')
    e50, _e95 = g('component_e2e_us')
    p50, p95 = g('preadv_us')
    q50, q95 = g('queue_wait_us')
    s50, _s95 = g('scatter_us')
    l50, _l95 = g('plan_us')
    r50, _r95 = g('reader_resolve_us')
    t50, _t95 = g('future_tail_us')
    return {
        'tok': d.get('tok_s'),
        'ttft': d.get('ttft_s'),
        'tokens': d.get('tokens'),
        'pin_eff': pin.get('pin_sync_effective'),
        'pin_reg': pin.get('pin_regime_effective'),
        'pin_load_ms': pin.get('pin_load_time_ms'),
        'profile': ec.get('profile_enabled'),
        'knobs': ec.get('experiment_knobs'),
        'e2e': e50, 'preadv50': p50, 'preadv95': p95, 'queue95': q95,
        'queue50': q50, 'scatter50': s50, 'plan50': l50, 'resolve50': r50,
        'tail50': t50, 'req_peak': dec.get('requested_inflight_peak'),
        'dropped': rs.get('dropped_samples'),
        'dec_runs': dec.get('runs'),
        'pool': d.get('run_pool') or {},
    }

arms = ['m_a1', 'm_a2', 'm_a3', 'm_a4']
crms = ['m_c1', 'm_c2', 'm_c3']
hdr = '%-6s %6s %7s %8s %8s %8s %8s %7s %7s %7s %7s %9s %7s %6s'
print(hdr % ('arm', 'tok', 'ttft', 'e2e50', 'preadv50', 'preadv95', 'queue95', 'scat50', 'plan50', 'res50', 'tail50', 'req_peak', 'pin', 'drop'))
for n in arms + crms:
    r = load(n)
    print(hdr % (n, round(r['tok'] or 0, 3), round(r['ttft'] or 0, 2), r['e2e'],
                 r['preadv50'], r['preadv95'], r['queue95'], r['scatter50'],
                 r['plan50'], r['resolve50'], r['tail50'], r['req_peak'],
                 r['pin_eff'], r['dropped']))

def med(ns, key):
    v = [load(n)[key] for n in ns]
    return st.median([x for x in v if x is not None])

print()
for key in ('tok', 'ttft', 'e2e', 'preadv50', 'preadv95', 'queue95', 'tail50'):
    va, vc = med(arms, key), med(crms, key)
    if va and vc:
        print('%-10s A=%s C=%s delta %+.2f%%' % (key, va, vc, (vc - va) / va * 100))

ta = [load(n)['tokens'] for n in arms]
tc = [load(n)['tokens'] for n in crms]
print('tokens A==C:', all(t == ta[0] for t in tc), len(ta[0]))
print('pool delta m_c2:', json.dumps(load('m_c2')['pool'])[:260])
print('profile flag A:', load('m_a1')['profile'], 'knobs:', load('m_a1')['knobs'])
print('pin sync/regime effective m_c1:', load('m_c1')['pin_eff'], load('m_c1')['pin_reg'], 'load_ms', load('m_c1')['pin_load_ms'])