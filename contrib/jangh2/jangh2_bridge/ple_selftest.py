"""Tiny synthetic PLE integration tests; real MLX execution is required."""
from pathlib import Path
import json
import struct
import tempfile
import numpy as np

from .checkpoint import Checkpoint
from .ple import IndexedMMapEmbedding


def run(*, device_cache_bytes=0):
    import mlx.core as mx
    if not mx.metal.is_available():
        raise RuntimeError('The PLE self-test requires Apple Metal; this is not a CPU substitute')
    cases=0
    for bits in (4,8):
        for dtype in ('F16','BF16','F32'):
            with tempfile.TemporaryDirectory(prefix='jangh2-ple-test-') as directory:
                root=Path(directory); weight_map={}; dense=[]
                quant={'bits':4,'group_size':32,'mode':'affine'}
                for shard,rows in enumerate((3,2)):
                    prefix=f'language_model.layers.1.ple.ngram_embedding.shards.{shard}'
                    codes=(np.arange(rows*64,dtype=np.uint32).reshape(rows,64)+shard*17)%(1<<bits)
                    packed=np.zeros((rows,64*bits//32),np.uint32)
                    for col in range(64):
                        packed[:,col*bits//32] |= codes[:,col] << ((col*bits)%32)
                    scale=np.tile(np.array([0.5,0.25],np.float32),(rows,1))
                    bias=np.tile(np.array([-1.,1.],np.float32),(rows,1))
                    def encode_aux(value):
                        if dtype=='BF16': return (value.view(np.uint32)>>16).astype('<u2')
                        return value.astype({'F16':'<f2','F32':'<f4'}[dtype])
                    tensors={'weight':('U32',packed),'scales':(dtype,encode_aux(scale)),
                             'biases':(dtype,encode_aux(bias))}
                    payload=b''; header={}; filename=f'part-{shard}.safetensors'
                    for name,(dt,array) in tensors.items():
                        key=prefix+'.'+name; data=array.tobytes()
                        header[key]={'dtype':dt,'shape':list(array.shape),
                                     'data_offsets':[len(payload),len(payload)+len(data)]}
                        payload+=data; weight_map[key]=filename
                    encoded=json.dumps(header).encode()
                    (root/filename).write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
                    quant[prefix]={'bits':bits,'group_size':32,'mode':'affine'}
                    dense.append(codes*np.repeat(scale,32,axis=1)+np.repeat(bias,32,axis=1))
                checkpoint=Checkpoint(root,{'quantization':quant},weight_map,{},'synthetic')
                reader=IndexedMMapEmbedding(checkpoint,1,5,64,2,device_cache_bytes=device_cache_bytes)
                try:
                    indices=np.array([[4,0,4],[2,3,2]],np.int64)
                    actual=reader(mx.array(indices,dtype=mx.int64))
                    mx.eval(actual)
                    expected=np.concatenate(dense).astype(np.float32)[indices]
                    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)),expected)
                    if reader.rows_read != 4:
                        raise AssertionError('PLE lookup did not deduplicate the selected rows')
                    if device_cache_bytes:
                        repeat=reader(mx.array(indices,dtype=mx.int64));mx.eval(repeat)
                        np.testing.assert_array_equal(np.array(repeat.astype(mx.float32)),expected)
                        if reader.mapping_stats().get('device_all_hit_calls',0)<1:
                            raise AssertionError('Exact device PLE cache did not hit on repeated rows')
                finally:
                    reader.close()
                print(f'PASS PLE MLX: Q{bits}, {dtype}, shuffled/duplicate/cross-shard rows',flush=True)
                cases+=1
    print(f'PLE MLX self-test passed: {cases} numerical cases.',flush=True)
    return cases
