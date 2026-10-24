#!/usr/bin/env python3
"""Profile real-model load, fresh-KV prompt processing, greedy decode and PLE counters.
Stop the server first. This is standalone timing, not HTTP/batching performance.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
import time
import traceback
from jangh2_bridge.profiles import atomic_json,runtime_fingerprint,config_digest,launch_profile


def run(a):
    import mlx.core as mx
    from mlx_vlm.utils import load
    from jangh2_bridge.model_checks import logits,ple_stats,ple_readers
    start=time.perf_counter();model,processor=load(a.model);mx.synchronize();load_s=time.perf_counter()-start
    tok=getattr(processor,'tokenizer',processor);language=model.language_model;rows=[]
    try:
        for n in a.prompt_tokens:
            encoded=tok.encode('def calculate(x):\n    return x * x + 17\n'*n,add_special_tokens=False)[:n]
            ids=mx.array([encoded],dtype=mx.int32);mx.eval(ids);cache=language.make_cache()
            before=ple_stats(model);mx.synchronize();start=time.perf_counter()
            for begin in range(0,len(encoded),a.chunk):
                last=logits(language,ids[:,begin:begin+a.chunk],cache,mx)[:,-1:,:]
                mx.eval(last)
            mx.synchronize();prefill=time.perf_counter()-start;after_prefill=ple_stats(model)
            times=[]
            for _ in range(a.decode_tokens):
                start=time.perf_counter()
                next_id=mx.argmax(last[:,-1,:],axis=-1)[:,None]
                last=logits(language,next_id,cache,mx)[:,-1:,:];mx.eval(last);mx.synchronize()
                times.append(time.perf_counter()-start)
            row={'prompt_tokens':len(encoded),'prefill_chunk':a.chunk,'prefill_seconds':prefill,
                 'prefill_tokens_per_second':len(encoded)/prefill,'decode_steps':a.decode_tokens,
                 'decode_seconds':sum(times),'decode_steps_per_second':a.decode_tokens/sum(times),
                 'decode_step_seconds':times,'ple_before':before,'ple_after_prefill':after_prefill,
                 'ple_after_decode':ple_stats(model),'active_memory_bytes':mx.get_active_memory()}
            rows.append(row);print(f'Prompt={n}: prefill={row["prefill_tokens_per_second"]:.2f} tok/s; decode={row["decode_steps_per_second"]:.2f} steps/s',flush=True)
            del cache,last;mx.synchronize();mx.clear_cache()
        return {'load_seconds':load_s,'phases':rows,'scope':'Fresh native KV/recurrent caches for each phase; resident weights and CPU/PLE caches warm cumulatively. Fixed chunks may differ from server scheduling.'}
    finally:
        for reader in ple_readers(model):reader.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--prompt-tokens',type=int,nargs='+',default=[1024,4096])
    p.add_argument('--decode-tokens',type=int,default=32);p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--report',default=str(Path.home()/'.omlx-pr364/jangh2-profile.json'))
    a=p.parse_args()
    if any(not 8<=n<=32768 for n in a.prompt_tokens) or not 1<=a.decode_tokens<=256 or not 8<=a.chunk<=512:p.error('Invalid profile bounds')
    report={'schema':1,'status':'failed','fingerprint':runtime_fingerprint(),'config_sha256':config_digest(a.model)}
    try:
        import mlx.core as mx
        from omlx.engine_core import get_mlx_executor
        from jangh2_bridge.scheduler_compat import install as scheduler
        from jangh2_bridge.loader import install
        with ExitStack() as cleanup:
            cleanup.enter_context(launch_profile(a.model));cleanup.callback(scheduler());cleanup.callback(install(a.model,backend='metal'))
            stream=import_module('mlx_lm.generate').generation_stream
            def worker():
                with mx.stream(stream):return run(a)
            report['results']=get_mlx_executor().submit(worker).result();report['status']='passed'
    except BaseException as exc:report.update(error=str(exc),traceback=traceback.format_exc());raise
    finally:atomic_json(a.report,report);print('Profile report:',a.report,flush=True)

if __name__=='__main__':main()
