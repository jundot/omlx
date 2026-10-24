#!/usr/bin/env python3
"""Qualify grouped prompt kernels and exact device PLE rows on the oMLX worker.
No full model is loaded. A fingerprint-bound report gates experimental serving.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from importlib import import_module
from pathlib import Path
import time
import traceback
from jangh2_bridge.profiles import atomic_json,runtime_fingerprint


def exercise():
    import numpy as np
    import mlx.core as mx
    from jangh2_bridge.perf_fixtures import synthetic,mlx_models,compare
    from jangh2_bridge.device_selftest import run as check_device_cache
    if not mx.metal.is_available():raise RuntimeError('Actual Apple Metal required')
    mx.set_default_device(mx.gpu)
    cases=[]
    scenarios=[
        (96,64,5,17,3,(2,3,4),('none','hadamard32','none'),mx.float32),
        (128,64,12,31,10,(2,2,2),('hadamard32',)*3,mx.float16),
        (128,64,12,65,10,(3,2,4),('hadamard32',)*3,mx.bfloat16),
        (2560,640,12,17,10,(2,2,2),('hadamard32',)*3,mx.bfloat16),
    ]
    rng=np.random.default_rng(1414)
    for d,h,e,t,k,bits,rot,dtype in scenarios:
        fixture=synthetic(d=d,hidden=h,experts=e,bits=bits,rotations=rot)
        models=mlx_models(fixture,{'reference':'reference-fast','grouped':'metal'})
        x=mx.array(rng.normal(0,.2,(1,t,d*2)).astype(np.float32),dtype=dtype)[...,::2]
        # Include duplicate routes in this defensive test; the live tuner uses no-replacement top-k.
        ids=mx.array(rng.integers(0,e,(1,t,k),dtype=np.uint32))
        expected=models['reference'](x,ids);mx.eval(expected)
        expected=np.asarray(expected.astype(mx.float32))
        for tile in (2,4):
            models['grouped']._performance=replace(models['grouped']._performance,
                prefill='grouped',prefill_min_tokens=2,prefill_tile=tile)
            out=models['grouped'](x,ids);mx.eval(out)
            metrics=compare(np.asarray(out.astype(mx.float32)),expected,
                            rtol=1e-4 if dtype==mx.float32 else .01,peak_rtol=.02)
            cases.append({'d':d,'h':h,'tokens':t,'routes':k,'tile':tile,'dtype':str(dtype),**metrics})
            print(f'PASS grouped: D={d}, H={h}, tokens={t}, top-k={k}, tile={tile}',flush=True)
        del models,fixture,x,ids,out,expected
        mx.synchronize();mx.clear_cache()
    # Invalid-ID and empty-expert tail coverage through the actual Metal path.
    fixture=synthetic(d=64,hidden=32,experts=4)
    model=mlx_models(fixture,{'m':'metal'})['m']
    model._performance=replace(model._performance,prefill='grouped',prefill_min_tokens=2)
    for bad in (-1,2**32):
        ids=mx.array([[[bad,0],[1,2],[3,bad]]],dtype=mx.int64)
        out=model(mx.ones((1,3,64)),ids);mx.eval(out);a=np.asarray(out)
        if not np.isnan(a[0,0,0]).all() or not np.isnan(a[0,2,1]).all() or not np.isfinite(a[0,1]).all():
            raise AssertionError('Grouped invalid-ID isolation failed')
    cache_cases=check_device_cache(mx)
    from jangh2_bridge.ple_selftest import run as check_quantized_ple
    quantized=check_quantized_ple(device_cache_bytes=16384)
    return {'grouped_cases':cases,'device_cache_cases':cache_cases,
            'quantized_ple_cache_cases':quantized,'invalid_routes_checked':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',default=str(Path.home()/'.omlx-pr364/jangh2-qualification.json'))
    p.add_argument('--skip-legacy',action='store_true',help='Still tests grouped kernels, but skips the already-qualified legacy kernel suite')
    args=p.parse_args();report={'schema':1,'status':'failed','actual_metal':False,
                               'fingerprint':runtime_fingerprint()};start=time.perf_counter();undo=None
    try:
        import mlx.core as mx
        from omlx.engine_core import get_mlx_executor
        from jangh2_bridge.scheduler_compat import install
        undo=install();generation=import_module('mlx_lm.generate')
        def worker():
            with mx.stream(generation.generation_stream):
                if not args.skip_legacy:
                    from check_performance import exercise as legacy
                    report['legacy']=legacy()
                result=exercise();mx.synchronize(generation.generation_stream);return result
        report['results']=get_mlx_executor().submit(worker).result()
        report.update(status='passed',actual_metal=True)
        print('PASS: grouped kernels and device PLE cache qualified on the oMLX worker; no full model loaded.',flush=True)
    except BaseException as exc:
        report['error']=str(exc);report['traceback']=traceback.format_exc();raise
    finally:
        if undo:undo()
        report['seconds']=time.perf_counter()-start;atomic_json(args.report,report)
        print('Qualification report:',args.report,flush=True)

if __name__=='__main__':main()
