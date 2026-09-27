"""Local opt-in real checkpoint test: copies one expert, never a full bank."""
import json,struct,time
from pathlib import Path
import numpy as np
import mlx.core as mx
from omlx.quantization.exl3 import Exl3Spec,Exl3SwitchLinear
from tests.test_exl3_reference import oracle
p=Path.home()/'.sushi/models/Qwen3.8-Flash-Next-Sushi-2.6bpw'
spec=Exl3Spec.from_config(json.load(open(p/'config.json')))
fpath=p/'model-exl3-L00-down.safetensors'
with fpath.open('rb') as f:size=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(size))
base=size+8;arrays=[]
for part in ['trellis','suh','svh']:
 key=next(k for k in h if k.endswith('.'+part));desc=h[key];shape=desc['shape'];dtype=np.uint16 if part=='trellis' else np.float16
 a=np.memmap(fpath,dtype=dtype,mode='r',offset=base+desc['data_offsets'][0],shape=tuple(shape));arrays.append(np.array(a[:1]));del a
q,su,sv=arrays
rng=np.random.default_rng(88);x=rng.normal(0,.1,(2,su.shape[1])).astype(np.float16);ids=np.zeros(2,np.uint32)
layer=Exl3SwitchLinear(mx.array(q),mx.array(su),mx.array(sv),spec)
y=layer(mx.array(x[:,None,:]),mx.array(ids));mx.eval(y)
ref=oracle(x,q,su,sv,ids,spec.window);actual=np.asarray(y[:,0,:]);diff=actual.astype(np.float32)-ref.astype(np.float32)
cos=np.sum(actual.astype(np.float32)*ref.astype(np.float32),axis=1)/(np.linalg.norm(actual.astype(np.float32),axis=1)*np.linalg.norm(ref.astype(np.float32),axis=1))
print('Real expert dimensions',su.shape[1],sv.shape[1],'cosine',cos,'max_abs',np.max(np.abs(diff)),flush=True)
np.testing.assert_allclose(actual,ref,rtol=.025,atol=.004)
start=time.perf_counter()
for _ in range(20):mx.eval(layer(mx.array(x[:1,None,:]),mx.array(ids[:1])))
print('Single expert ms',round((time.perf_counter()-start)*1000/20,3),flush=True)
