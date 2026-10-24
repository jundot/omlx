#!/usr/bin/env python3
"""Load one real model and compare fixed-prefix logits with its reference experts.

Runs on the oMLX worker, fresh recurrent/KV caches for each pass. Includes cached
teacher-forced continuation, so sampling divergence cannot hide numerical errors.
This is a backend comparison, NOT independent validation of the model architecture.
Stop the API server first to avoid two resident model copies.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from importlib import import_module
import json
from pathlib import Path
import time
import traceback
from jangh2_bridge.profiles import atomic_json,runtime_fingerprint,config_digest,launch_profile

TEXTS=[
    'The quick brown fox jumps over the lazy dog. Explain why reliable tests matter. ',
    'def cached_lookup(key: str) -> int:\n    return values.get(key, 0)\n',
    'Calculate the sum of the first ten prime numbers and verify the result. ',
    'Bonjour. Expliquez les différences entre un cache et une base de données. ',
]


def check(args):
    import numpy as np
    import mlx.core as mx
    from mlx_vlm.utils import load
    from jangh2_bridge.model_checks import (temporary_expert_forward,reference_call,
        compare_logits,logits,ple_stats,reset_device_rows,ple_readers)
    model,processor=load(args.model)
    tokenizer=getattr(processor,'tokenizer',processor);language=model.language_model
    if not callable(getattr(language,'make_cache',None)):
        raise RuntimeError('Language model has no native make_cache()')
    results=[]
    try:
        for case,text in enumerate(TEXTS[:args.cases]):
            ids=tokenizer.encode(text*(args.tokens+args.continuation),add_special_tokens=False)
            ids=ids[:args.tokens+args.continuation]
            if len(ids)!=args.tokens+args.continuation:raise RuntimeError('Insufficient fixed input tokens')
            arr=mx.array([ids],dtype=mx.int32)
            def run():
                reset_device_rows(model);cache=language.make_cache();mx.synchronize();start=time.perf_counter()
                output=logits(language,arr[:,:args.tokens],cache,mx)[:,-min(8,args.tokens):,:]
                mx.eval(output);samples=[np.asarray(output.astype(mx.float32))]
                for index in range(args.tokens,len(ids)):
                    out=logits(language,arr[:,index:index+1],cache,mx)
                    mx.eval(out);samples.append(np.asarray(out.astype(mx.float32)))
                mx.synchronize();return np.concatenate(samples,axis=1),time.perf_counter()-start
            # Reference PLE path does not consult the candidate device cache.
            readers=ple_readers(model);budgets=[r._device_cache_bytes for r in readers]
            try:
                for r in readers:r._device_cache_bytes=0
                with temporary_expert_forward(model,reference_call):reference,reference_seconds=run()
            finally:
                for r,budget in zip(readers,budgets):r._device_cache_bytes=budget
            actual,seconds=run()
            metrics=compare_logits(actual,reference)
            row={'case':case,'prefill_tokens':args.tokens,'continuation_tokens':args.continuation,
                 'candidate_seconds':seconds,'reference_seconds':reference_seconds,**metrics}
            results.append(row);print('PASS whole-model fixed-token logits:',json.dumps(row),flush=True)
        return {'cases':results,'ple':ple_stats(model),'scope':'same architecture/weights, independent reference expert operations; not an independent model runtime'}
    finally:
        for reader in ple_readers(model):reader.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--tokens',type=int,default=64)
    p.add_argument('--continuation',type=int,default=4);p.add_argument('--cases',type=int,default=3)
    p.add_argument('--report',default=str(Path.home()/'.omlx-pr364/jangh2-model-comparison.json'))
    args=p.parse_args()
    if not 2<=args.tokens<=256 or not 1<=args.continuation<=32 or not 1<=args.cases<=4:
        p.error('tokens 2..256, continuation 1..32, cases 1..4')
    report={'schema':1,'status':'failed','fingerprint':runtime_fingerprint(),
            'config_sha256':config_digest(args.model),'speculative_decoding':False}
    try:
        import mlx.core as mx
        from omlx.engine_core import get_mlx_executor
        from jangh2_bridge.scheduler_compat import install as scheduler
        from jangh2_bridge.loader import install
        with ExitStack() as cleanup:
            cleanup.enter_context(launch_profile(args.model));cleanup.callback(scheduler())
            cleanup.callback(install(args.model,backend='metal'))
            generation=import_module('mlx_lm.generate')
            def worker():
                with mx.stream(generation.generation_stream):return check(args)
            report['results']=get_mlx_executor().submit(worker).result();report['status']='passed'
    except BaseException as exc:
        report.update(error=str(exc),traceback=traceback.format_exc());raise
    finally:atomic_json(args.report,report);print('Comparison report:',args.report,flush=True)

if __name__=='__main__':main()
