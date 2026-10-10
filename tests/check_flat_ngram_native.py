"""Opt-in Metal check of the native disk-backed reader, using a tiny fixture."""
import importlib,json,struct,tempfile
from pathlib import Path
import mlx.core as mx
import mlx_vlm.models
import numpy as np
vendor=Path(__file__).resolve().parents[1]/'omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models'
mlx_vlm.models.__path__.insert(0,str(vendor))
language=importlib.import_module('mlx_vlm.models.qwen4_exp.language')
source=mx.random.normal((5,128)).astype(mx.float16)
q,scales,biases=mx.quantize(source,group_size=32,bits=4)
mx.eval(q,scales,biases)
header={'__metadata__': {'format':'mlx-serve-ngram','bits':'4','group_size':'32'}}
parts=[];offset=0
for name,a,dtype in [('weight',q,'U32'),('scales',scales,'F16'),('biases',biases,'F16')]:
 raw=np.asarray(a).tobytes();header[name]={'dtype':dtype,'shape':list(a.shape),'data_offsets':[offset,offset+len(raw)]};parts.append(raw);offset+=len(raw)
with tempfile.TemporaryDirectory() as d:
 p=Path(d);raw=json.dumps(header).encode()
 (p/'ngram_table.bin').write_bytes(struct.pack('<Q',len(raw))+raw+b''.join(parts))
 (p/'config.json').write_text(json.dumps({'ngram_table':{'file':'ngram_table.bin','bits':4,'group_size':32}}))
 layer=language.DiskBackedShardedEmbedding(p,'embedding',5,128,2)
 ids=mx.array([0,3,4,3],dtype=mx.int64)
 y=layer(ids);ref=mx.dequantize(q,scales,biases,group_size=32,bits=4)[ids];mx.eval(y,ref)
 np.testing.assert_array_equal(np.asarray(y.astype(mx.float32)),np.asarray(ref.astype(y.dtype).astype(mx.float32)))
 print('Native reader: selected rows across shards match quantized source; bytes',offset,flush=True)
 readers=list(layer._readers.values())
 layer.close()
 layer.close()
 assert readers and all(reader._mapping is None for reader in readers)
 print('Native reader cleanup passed',flush=True)
