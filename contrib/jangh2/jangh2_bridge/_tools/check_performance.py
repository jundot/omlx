#!/usr/bin/env python3
"""Qualify the fused backend on the actual oMLX GPU worker (no full model).

Exercises Metal compilation, strict expert loading, FP32/F16/BF16, decode,
sorted prefill, mixed bit widths, rotations, tail dimensions, non-contiguous
inputs and invalid-ID memory guards. Writes only a local JSON report.
"""
from __future__ import annotations
import argparse
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import traceback


def check_reference_alignment(mx):
    """Catch FP32 changes small enough to escape ordinary component tolerances.

    A few ULPs at one expert can change a later hard top-k choice. Use the
    reference's regular GEMV dimensions here; small-K kernels can use a
    different reduction layout and are covered by the dense-oracle checks.
    """
    from dataclasses import replace
    import numpy as np
    from jangh2_bridge.perf_fixtures import synthetic, mlx_models
    rng = np.random.default_rng(1267)
    cases = []
    for d, h, bits, rotations in (
        (2560, 640, (2, 2, 2), ('hadamard32',) * 3),
        (128, 128, (2, 3, 4), ('none', 'hadamard32', 'none')),
    ):
        fixture = synthetic(d=d, hidden=h, experts=4, bits=bits, rotations=rotations)
        for tokens in (1, 9):
            x = mx.array(rng.normal(0, .2, (1, tokens, d)).astype(np.float32))
            ids = mx.array(rng.integers(0, 4, (1, tokens, 10), dtype=np.uint32))
            for rows in (2, 4, 8):
                models = mlx_models(fixture, {'reference': 'reference-fast', 'candidate': 'metal'},
                                    tuning={'candidate': {'gate_rows': rows, 'down_rows': rows}})
                reference = models['reference'](x, ids)
                candidate = models['candidate'](x, ids)
                mx.eval(reference, candidate)
                if not np.array_equal(np.asarray(candidate), np.asarray(reference)):
                    raise AssertionError(f'FP32 expert/reference rounding changed: D={d}, H={h}, '
                                         f'tokens={tokens}, rows={rows}')
                cases.append({'D': d, 'H': h, 'tokens': tokens, 'rows': rows,
                              'bits': bits, 'backend': 'metal', 'exact': True})
                if tokens > 1:
                    for tile in (2, 4):
                        models['candidate']._performance = replace(
                            models['candidate']._performance, prefill='grouped',
                            prefill_min_tokens=2, prefill_tile=tile)
                        candidate = models['candidate'](x, ids)
                        mx.eval(candidate)
                        if not np.array_equal(np.asarray(candidate), np.asarray(reference)):
                            raise AssertionError(f'Grouped FP32 expert/reference rounding changed: '
                                                 f'D={d}, H={h}, rows={rows}, tile={tile}')
                        cases.append({'D': d, 'H': h, 'tokens': tokens, 'rows': rows,
                                      'bits': bits, 'backend': 'grouped', 'tile': tile,
                                      'exact': True})
                del models, reference, candidate
    print(f'PASS FP32 reference alignment: {len(cases)} exact cases', flush=True)
    return cases


def exercise():
    import numpy as np
    import mlx.core as mx
    from jangh2_bridge.selftest import run
    from jangh2_bridge.perf_fixtures import synthetic,mlx_models,dense_oracle,compare
    if not mx.metal.is_available():raise RuntimeError('Apple Metal is required; CPU simulation is not qualification')
    mx.set_default_device(mx.gpu)
    original_cases=run('metal',full_dimensions=True)
    cases=[];rng=np.random.default_rng(1266)
    scenarios=[
        # D,H,bits,rotations,batch,tokens,routes,dtype,gate_rows,down_rows
        (2560,640,(2,2,2),('hadamard32',)*3,1,1,8,mx.bfloat16,4,4),
        (128,64,(2,2,2),('hadamard32',)*3,2,9,8,mx.float32,4,4),
        (96,64,(2,3,4),('none','hadamard32','none'),2,3,3,mx.float32,4,4),
        (96,64,(3,2,4),('hadamard32',)*3,2,13,3,mx.bfloat16,4,4),
        (128,64,(4,4,4),('none',)*3,1,1,8,mx.float16,4,4),
        (128,64,(2,2,2),('hadamard32',)*3,1,1,8,mx.float32,2,4),
        (128,64,(2,2,2),('hadamard32',)*3,1,1,8,mx.float32,8,4),
        (128,64,(2,2,2),('hadamard32',)*3,1,1,8,mx.float32,4,8),
    ]
    for d,h,bits,rots,b,t,k,dtype,gr,dr in scenarios:
        fixture=synthetic(d=d,hidden=h,experts=4,bits=bits,rotations=rots)
        models=mlx_models(fixture,{'candidate':'metal'},tuning={'candidate':{'gate_rows':gr,'down_rows':dr}})
        model=models['candidate']
        # A strided GPU view exercises ensure_row_contiguous, rather than only
        # testing already-contiguous NumPy input buffers.
        x=mx.array(rng.normal(0,.2,(b,t,d*2)).astype(np.float32),dtype=dtype)[...,::2]
        ids_np=rng.integers(0,4,(b,t,k),dtype=np.uint32)
        ids=mx.array(ids_np,dtype=mx.uint32)
        result=model(x,ids);mx.eval(result)
        actual=np.array(result.astype(mx.float32))
        expected=dense_oracle(np.array(x.astype(mx.float32)),ids_np,fixture)
        metrics=compare(actual,expected,rtol=1e-4 if dtype==mx.float32 else .01,peak_rtol=.02)
        cases.append({'D':d,'H':h,'bits':bits,'rotations':rots,'shape':(b,t,k),
                      'dtype':str(dtype),'gate_rows':gr,'down_rows':dr,**metrics})
        print(f'PASS fused Metal: D={d}, H={h}, bits={bits}, shape={(b,t,k)}, dtype={dtype}, '
              f'rows={gr}/{dr}, relative_L2={metrics["relative_l2"]:.4g}',flush=True)
        del model,models,x,ids,result,fixture
        mx.synchronize();mx.clear_cache()
    fixture=synthetic(d=64,hidden=32,experts=4)
    model=mlx_models(fixture,{'candidate':'metal'})['candidate']
    x=mx.ones((1,1,64),dtype=mx.float32)
    for dtype,bad in ((mx.int32,-1),(mx.uint32,999),(mx.uint64,2**32)):
        ids=mx.array([[[bad,1]]],dtype=dtype)
        result=model(x,ids);mx.eval(result)
        got=np.array(result)
        if not np.isnan(got[0,0,0]).all() or not np.isfinite(got[0,0,1]).all():
            raise AssertionError('Invalid route guard failed')
    empty=model(mx.zeros((1,0,64)),mx.zeros((1,0,8),dtype=mx.uint32));mx.eval(empty)
    if empty.shape!=(1,0,8,64):raise AssertionError('Empty-route shape failed')
    print('PASS: invalid-ID GPU guards and empty batches',flush=True)
    alignment = check_reference_alignment(mx)
    return {'legacy_strict_load_cases':original_cases,'extended_cases':cases,'invalid_id_cases':3,
            'fp32_reference_alignment': alignment,
            'empty_batch_passed':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',default=str(Path.home()/'.omlx-pr364/jangh2-kernels-qualification.json'))
    args=parser.parse_args();path=Path(args.report).expanduser()
    report={'schema':1,'kind':'actual-Metal-qualification','status':'failed',
            'platform':platform.platform(),'versions':{n:version(n) for n in ('mlx','mlx-vlm','mlx-lm','omlx')}}
    started=time.time();undo=None
    try:
        import mlx.core as mx
        from omlx.engine_core import get_mlx_executor
        from jangh2_bridge.scheduler_compat import install
        undo=install();generation=import_module('mlx_lm.generate')
        def on_worker():
            with mx.stream(generation.generation_stream):
                result=exercise()
                mx.synchronize(generation.generation_stream)
                return result
        report['results']=get_mlx_executor().submit(on_worker).result()
        report['status']='passed'
        print('PASS: fused Metal backend qualified on the oMLX executor; full model NOT loaded.',flush=True)
    except BaseException as exc:
        report['error']=str(exc);report['traceback']=traceback.format_exc()
        raise
    finally:
        if undo is not None:undo()
        report['seconds']=time.time()-started
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
        print('Qualification report:',path,flush=True)

if __name__=='__main__':main()
