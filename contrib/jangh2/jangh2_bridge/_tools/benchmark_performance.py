#!/usr/bin/env python3
"""Compare expert backends on Apple Metal; optionally use one real model block.

This is a component benchmark, NOT full-model tokens/second. Each measured call
builds and evaluates a fresh graph. Compilation is excluded by warmup. Candidates
are compared numerically before timing and measurement order alternates.
"""
from __future__ import annotations
import argparse
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import time


def benchmark(args):
    import numpy as np
    import mlx.core as mx
    from jangh2_bridge.perf_fixtures import synthetic,from_checkpoint,mlx_models,compare
    if not mx.metal.is_available():raise RuntimeError('This benchmark needs Apple Metal')
    mx.set_default_device(mx.gpu)
    from jangh2_bridge.tuning import route_count,route_sample,choose_decode,choose_prefill,expert_count
    config=json.loads((Path(args.model).expanduser()/'config.json').read_text()) if args.model else {'num_experts_per_tok':8}
    args.experts=expert_count(config,args.experts,real_model=bool(args.model))
    if args.model and args.grouped and args.write_profile:
        full=expert_count(config,None,real_model=True)
        if args.experts != full:
            raise ValueError('Grouped profile promotion requires the full checkpoint expert population; omit --experts')
    fixture=(from_checkpoint(args.model,layer=args.layer,experts=args.experts)
             if args.model else synthetic(experts=args.experts))
    args.routes=route_count(config,args.routes)
    if args.routes>fixture['experts']:
        raise ValueError('Use --experts at least as large as the model top-k / --routes')
    backends={'reference-fast':'reference-fast','metal':'metal'}
    tuning={'metal':{'gate_rows':4,'down_rows':4}}
    if args.tune:
        for gr,dr in ((2,2),(2,4),(2,8),(4,2),(4,8),(8,2),(8,4),(8,8)):
            name=f'metal-g{gr}-d{dr}';backends[name]='metal';tuning[name]={'gate_rows':gr,'down_rows':dr}
    if args.grouped:
        from jangh2_bridge.profiles import require_qualification
        require_qualification()
        for tile in (2,4):
            name=f'grouped{tile}';backends[name]='metal'
            tuning[name]={'prefill':'grouped','prefill_tile':tile,'prefill_min_tokens':2}
    if args.include_native:backends['native']='native'
    models=mlx_models(fixture,backends,tuning=tuning)
    metadata={k:v for k,v in fixture.items() if k not in ('weights','levels')}
    report={'kind':'one-expert-block-microbenchmark','fixture':metadata,'results':[],
            'versions':{n:version(n) for n in ('mlx','mlx-vlm','mlx-lm')},
            'warning':'Hot selected experts in one block do not predict whole-model tokens/second.'}
    rng=np.random.default_rng(16776);dtype={'bf16':mx.bfloat16,'f16':mx.float16,'f32':mx.float32}[args.dtype]
    for tokens in args.tokens:
        x=mx.array(rng.normal(0,.2,(1,tokens,fixture['d'])).astype(np.float32),dtype=dtype)
        ids=mx.array(route_sample(1,tokens,fixture['experts'],args.routes,seed=16776+tokens))
        mx.eval(x,ids)
        baseline=models['reference-fast'](x,ids);mx.eval(baseline)
        expected=np.array(baseline.astype(mx.float32))
        correctness={};samples={name:[] for name in models}
        for name,model in models.items():
            got=model(x,ids);mx.eval(got)
            correctness[name]=compare(np.array(got.astype(mx.float32)),expected,
                rtol=(.025 if name=='native' else 1e-4) if dtype==mx.float32 else .01,peak_rtol=.04)
            for _ in range(args.warmup):mx.eval(model(x,ids))
        names=list(models)
        for iteration in range(args.iterations):
            order=names if iteration%2==0 else list(reversed(names))
            for name in order:
                mx.synchronize()
                t0=time.perf_counter()
                out=models[name](x,ids)
                mx.eval(out);mx.synchronize()
                samples[name].append((time.perf_counter()-t0)*1000)
        base=statistics.median(samples['reference-fast'])
        rows={}
        for name,values in samples.items():
            median=statistics.median(values)
            rows[name]={'median_ms':median,'min_ms':min(values),'max_ms':max(values),
                        'speedup_vs_reference_fast':base/median,'samples_ms':values,'correctness':correctness[name]}
            print(f'Tokens={tokens:3d} routes={args.routes:2d} {name:17s} '
                  f'{median:9.3f} ms/block; relative speed={base/median:6.2f}x',flush=True)
        report['results'].append({'tokens':tokens,'routes':args.routes,'dtype':args.dtype,'backends':rows})
    if args.tune:
        params=choose_decode(report['results'],tuning)
        report['suggested_decode_tuning']=params
        legacy_name=next(n for n,values in tuning.items() if values==params and not n.startswith('grouped'))
        prefill=choose_prefill(report['results'],baseline=legacy_name) if args.grouped else {'prefill':'legacy'}
        report['suggested_prefill_tuning']=prefill
        print('Measured decode tuning:',params,flush=True)
        print('Measured prompt tuning:',prefill,flush=True)
        from jangh2_bridge.profiles import ENV_KEYS,atomic_json,runtime_fingerprint,config_digest
        environment={ENV_KEYS[k]:str(v) for k,v in {**params,**prefill}.items()}
        report['suggested_environment']=environment
        if args.write_profile:
            if not args.model:raise ValueError('--write-profile requires a real model checkpoint')
            atomic_json(args.write_profile,{'schema':1,'fingerprint':runtime_fingerprint(),
                'config_sha256':config_digest(args.model),'environment':environment,
                'evidence':'numerically checked expert-block timings; verify full API throughput separately'})
            print('Saved measured profile:',args.write_profile,flush=True)
        print('OMLX_JANGH_THINKING=off '+' '.join(f'{k}={v}' for k,v in environment.items())+
              ' OMLX_JANGH_BACKEND=metal omlx-dev',flush=True)
    print('Component measurements only. Confirm with a longer full-model/API generation.',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',help='Optional real checkpoint; reads only selected experts, not full model')
    p.add_argument('--layer',type=int,default=0);p.add_argument('--experts',type=int,default=None,help='Default: full checkpoint expert population; synthetic default 32')
    p.add_argument('--tokens',type=int,nargs='+',default=[1,8,32])
    p.add_argument('--routes',type=int,default=None,help='Default: checkpoint top-k; never assumed eight for a real model')
    p.add_argument('--dtype',choices=('bf16','f16','f32'),default='bf16')
    p.add_argument('--iterations',type=int,default=5);p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--tune',action='store_true');p.add_argument('--grouped',action='store_true')
    p.add_argument('--write-profile',help='Save measured tuning as fingerprint-bound JSON; requires --tune and --model')
    p.add_argument('--include-native',action='store_true')
    p.add_argument('--report',default=str(Path.home()/'.omlx-pr364/jangh2-benchmark.json'))
    args=p.parse_args()
    if ((args.experts is not None and not 1<=args.experts<=2048) or (args.routes is not None and not 1<=args.routes<=64) or not 1<=args.iterations<=100
            or not 1<=args.warmup<=20 or args.layer<0 or any(not 1<=t<=128 for t in args.tokens)):
        p.error('Invalid dimensions/runs; experts 1..2048, routes 1..64, tokens 1..128, iterations 1..100, warmup 1..20')
    if args.write_profile and (not args.tune or not args.model):p.error('--write-profile requires --tune --model')
    import mlx.core as mx
    from omlx.engine_core import get_mlx_executor
    from jangh2_bridge.scheduler_compat import install
    undo=install()
    try:
        generation=import_module('mlx_lm.generate')
        def on_worker():
            with mx.stream(generation.generation_stream):return benchmark(args)
        report=get_mlx_executor().submit(on_worker).result()
        path=Path(args.report).expanduser();path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n');print('Benchmark report:',path,flush=True)
    finally:undo()

if __name__=='__main__':main()
