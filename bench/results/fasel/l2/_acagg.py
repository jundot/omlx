import json
import statistics as st
from pathlib import Path
ROOT = Path('bench/results/fasel/l2/arms')
def load(n):
    d = json.loads((ROOT / (n + '.json')).read_text())
    rd = d.get('read_stats') or {}
    return {
        'tok': d.get('tok_s'),
        'ttft': d.get('ttft_s'),
        'rr50': rd.get('lat_us_p50'),
        'rr95': rd.get('lat_us_p95'),
        'runs': rd.get('runs'),
        'bytes': rd.get('bytes'),
        'calls': rd.get('calls'),
        'peak_inflight': rd.get('peak_inflight'),
        'fb': d.get('ctx_fallback_to_legacy'),
    }
def med(key, nn):
    v = [load(n)[key] for n in nn]
    return st.median([x for x in v if x is not None])
a = ['p_a1', 'p_a2', 'p_a3']
c = ['p_c1', 'p_c2', 'p_c3']
for n in a + c:
    r = load(n)
    print('%-5s tok=%.3f ttft=%.1f rr50=%s rr95=%s fb=%s' % (
        n, r['tok'], r['ttft'], r['rr50'], r['rr95'], r['fb']))
for key in ('tok', 'ttft', 'rr50', 'rr95'):
    va, vc = med(key, a), med(key, c)
    if va and vc:
        print('%s: A=%s C=%s delta %+.2f%%' % (key, round(va, 2), round(vc, 2), (vc - va) / va * 100))
ta = [json.loads((ROOT / (n + '.json')).read_text())['tokens'] for n in a]
tc = [json.loads((ROOT / (n + '.json')).read_text())['tokens'] for n in c]
print('tokens A==C:', all(t == ta[0] for t in tc) and len(ta[0]) == 48)
