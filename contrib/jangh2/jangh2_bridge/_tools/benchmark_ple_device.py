#!/usr/bin/env python3
"""Compare PLE cache policies against real checkpoint rows, without full model loading.
Uses synthetic hot/mixed row-ID traces (NOT a measured language workload).
No profile is promoted automatically on the basis of this component benchmark.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
import statistics
import time
import traceback
from jangh2_bridge.profiles import atomic_json,runtime_fingerprint,require_qualification


def run(a):
    import numpy as np
    import mlx.core as mx
    from jangh2_bridge.checkpoint import inspect_checkpoint
    from jangh2_bridge.metadata import ple_layer_indices
    from jangh2_bridge.loader import ple_runtime_context
    from jangh2_bridge.ple import IndexedMMapEmbedding
    import mlx_vlm.utils as utils
    checkpoint=inspect_checkpoint(a.model);architecture=import_module('mlx_vlm.models.qwen4_exp')
    language=import_module('mlx_vlm.models.qwen4_exp.language')
    config=utils.load_config(checkpoint.root)
    config.setdefault('text_config',config.pop('llm_config',{}));config.setdefault('vision_config',{})
    cfg=architecture.ModelConfig.from_dict(config)
    cfg=utils.update_module_configs(cfg,architecture,config,['text','vision','perceiver','projector','audio'])
    text=cfg.text_config;layer=ple_layer_indices(text.ple_layer_ids,text.num_hidden_layers)[0]
    with ple_runtime_context(language,checkpoint) as state:
        wrapper=language.Qwen4ExpNGramEmbedding(text,text.ple_embed_dim,layer,0)
        exemplar=wrapper.ngram_embedding
        if not isinstance(exemplar,IndexedMMapEmbedding):
            raise RuntimeError('Device-cache tuner requires the stock indexed PLE reader')
        count,width=exemplar.row_count,exemplar.row_width
        exemplar.close()
    rng=np.random.default_rng(1433)
    hot=rng.integers(0,count,1024,dtype=np.int64)
    policies={'cpu16':(16,0),'cpu64':(64,0),'cpu64-device16':(64,16)}
    readers={};results=[];oracle=None
    try:
        for name,(cpu,device) in policies.items():
            readers[name]=IndexedMMapEmbedding(checkpoint,layer,count,width,int(text.split_ngram_parts),
                max_open_files=16,row_cache_bytes=cpu*1024**2,device_cache_bytes=device*1024**2)
        oracle=IndexedMMapEmbedding(checkpoint,layer,count,width,int(text.split_ngram_parts),max_open_files=16)
        for n in a.rows:
            for kind in ('hot','mixed'):
                traces=[]
                for _ in range(a.iterations+2):
                    ids=rng.choice(hot,n,replace=True)
                    if kind=='mixed':ids[::4]=rng.integers(0,count,len(ids[::4]))
                    traces.append(mx.array(ids,dtype=mx.int64))
                mx.eval(traces);samples={name:[] for name in readers}
                for index,ids in enumerate(traces):
                    expected=oracle(ids);mx.eval(expected)
                    # Use baseline value for parity only; order alternates for timing.
                    names=list(readers) if index%2==0 else list(reversed(readers))
                    for name in names:
                        mx.synchronize();start=time.perf_counter()
                        actual=readers[name](ids);mx.eval(actual);mx.synchronize()
                        elapsed=time.perf_counter()-start
                        if not bool(mx.array_equal(actual,expected).item()):raise AssertionError('PLE cache values differ: '+name)
                        if index>=2:samples[name].append(elapsed*1000)
                row={'rows':n,'trace':kind,'policies':{name:{'median_ms':statistics.median(times),'samples_ms':times}
                      for name,times in samples.items()}}
                results.append(row);print('PLE component:',row,flush=True)
        return {'cases':results,'cache_counters':{name:r.mapping_stats() for name,r in readers.items()},
                'limitations':'Synthetic row IDs, cumulatively warmed filesystem; independent uncached oracle reads precede all candidates. Not a whole-model tuning result. Cache policy is NOT promoted automatically.'}
    finally:
        for r in readers.values():r.close()
        if oracle is not None:oracle.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True)
    p.add_argument('--rows',type=int,nargs='+',default=[16,256,2048]);p.add_argument('--iterations',type=int,default=5)
    p.add_argument('--report',default=str(Path.home()/'.omlx-pr364/jangh2-ple-benchmark.json'))
    a=p.parse_args()
    if any(not 1<=n<=8192 for n in a.rows) or not 2<=a.iterations<=50:p.error('Invalid sample sizes')
    require_qualification();report={'status':'failed','fingerprint':runtime_fingerprint()}
    try:
        import mlx.core as mx
        from omlx.engine_core import get_mlx_executor
        from jangh2_bridge.scheduler_compat import install
        with ExitStack() as cleanup:
            cleanup.callback(install());stream=import_module('mlx_lm.generate').generation_stream
            def worker():
                with mx.stream(stream):return run(a)
            report['results']=get_mlx_executor().submit(worker).result();report['status']='passed'
    except BaseException as exc:report.update(error=str(exc),traceback=traceback.format_exc());raise
    finally:atomic_json(a.report,report);print('PLE report:',a.report,flush=True)

if __name__=='__main__':main()
