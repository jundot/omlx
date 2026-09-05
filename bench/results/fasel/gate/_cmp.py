import json
def tokens(p):
    d = json.load(open(p))
    return d.get('tokens') or d.get('bit_exact')
ref2k = tokens('bench/results/fasek/tokgate2/t1/qwen_0.0g_output.json')
fresh2k = tokens('bench/results/fasel/gate/fresh_2k_tokens.json')
print('2k fresh == K8 tokgate2 reference:', fresh2k == ref2k, len(fresh2k))
ref8k = tokens('bench/results/fasek/f4a/qwen_0.0g_output.json')
l4a8k = tokens('bench/results/fasel/l4a/run1_hf25.json')
t8 = l4a8k if isinstance(l4a8k, list) else None
print('8k HOBBIT L4A == F4A reference:', t8 == ref8k, len(t8 or []))
f5o1 = tokens('bench/results/fasek/f5ab/o1/qwen_0.0g_output.json')
print('8k L4A == f5ab o1:', t8 == f5o1, len(f5o1 or []))
if t8 and not (t8 == ref8k or t8 == f5o1):
    print('8k differs from both references - inspect')