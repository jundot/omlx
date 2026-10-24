import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from test_bridge import config
from jangh2_bridge.performance import PerformanceOptions
from jangh2_bridge.perf_fixtures import synthetic,dense_oracle,compare,from_checkpoint
from jangh2_bridge.reference_ops import expert_forward
from jangh2_bridge.metadata import PROJECTIONS

class PerformanceToolTests(unittest.TestCase):
    def test_environment_defaults_leave_old_backend_unchanged(self):
        with patch.dict('os.environ',{},clear=True):
            old=PerformanceOptions.from_environment();new=PerformanceOptions.from_environment(fast_defaults=True)
        self.assertEqual((old.ple_cache_mib,old.ple_open_files),(0,8))
        self.assertEqual((new.ple_cache_mib,new.ple_open_files),(16,16))
        self.assertEqual(new.gate_rows,4)

    def test_overrides_are_bounded_and_no_boolean_or_float_strings(self):
        options={'OMLX_JANGH_METAL_GATE_ROWS':'8','OMLX_JANGH_PLE_CACHE_MB':'32','OMLX_JANGH_PLE_OPEN_FILES':'11'}
        with patch.dict('os.environ',options,clear=True):
            value=PerformanceOptions.from_environment()
        self.assertEqual((value.gate_rows,value.ple_cache_mib,value.ple_open_files),(8,32,11))
        for name,val in [('OMLX_JANGH_METAL_GATE_ROWS','3'),('OMLX_JANGH_PLE_CACHE_MB','257'),
                         ('OMLX_JANGH_PLE_OPEN_FILES','0'),('OMLX_JANGH_METAL_DOWN_ROWS','4.0')]:
            with self.subTest(name=name),patch.dict('os.environ',{name:val},clear=True):
                with self.assertRaises(ValueError):PerformanceOptions.from_environment()

    def test_independent_oracle_and_mixed_bits_agree_with_existing_reference(self):
        f=synthetic(d=96,hidden=64,experts=4,bits=(2,3,4),rotations=('none','hadamard32','hadamard32'))
        rng=np.random.default_rng(77);x=rng.normal(0,.2,(2,3,96)).astype(np.float32)
        ids=rng.integers(0,4,(2,3,3),dtype=np.uint32)
        projections={p:(*f['weights'][p],f['specs'][p]['bits'],np.array(f['levels'][f['specs'][p]['bits']],np.float32),f['specs'][p]['rotation']) for p in PROJECTIONS}
        old=expert_forward(x,ids,projections,np,chunk_size=8,evaluation='per_chunk')
        new=dense_oracle(x,ids,f)
        compare(new,old)

    def test_comparison_refuses_nonfinite_or_wrong_shape_or_corruption(self):
        good=np.array([1,2,3],np.float32)
        for bad in [np.array([1,2]),np.array([1,np.nan,3]),np.array([1,7,3])]:
            with self.subTest(bad=bad):
                with self.assertRaises(AssertionError):compare(bad,good)
        self.assertEqual(compare(good,good)['relative_l2'],0)

    def test_real_header_fixture_reads_only_requested_expert_rows(self):
        self._check_selected_experts(4,2)

    def test_real_header_fixture_supports_full_population_above_old_64_limit(self):
        self._check_selected_experts(80,80)

    def _check_selected_experts(self,total,selected_count):
        f=synthetic(d=64,hidden=32,experts=total)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c=config();c['text_config']={'hidden_size':64,'moe_intermediate_size':32,'num_experts':total}
            entries={};index={};payload=bytearray()
            for p in PROJECTIONS:
                for suffix,array in zip(('tq2_packed','tq2_scales'),f['weights'][p]):
                    key=f'language_model.layers.0.mlp.switch_mlp.{p}.{suffix}'
                    begin=len(payload);payload.extend(array.tobytes())
                    entries[key]={'dtype':'U32' if suffix=='tq2_packed' else 'F16','shape':list(array.shape),'data_offsets':[begin,len(payload)]}
                    index[key]='data.safetensors'
            header=json.dumps(entries).encode();(root/'data.safetensors').write_bytes(struct.pack('<Q',len(header))+header+payload)
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':index}))
            (root/'config.json').write_text(json.dumps(c))
            actual=from_checkpoint(root,experts=selected_count)
            selected=np.linspace(0,total-1,selected_count,dtype=np.int64)
            self.assertEqual(actual['selected_experts'],selected.tolist());self.assertEqual(actual['experts'],selected_count)
            for p in PROJECTIONS:
                for i in (0,1):np.testing.assert_array_equal(actual['weights'][p][i],f['weights'][p][i][selected])
            with self.assertRaises(ValueError):from_checkpoint(root,layer=77)
            with self.assertRaises(ValueError):from_checkpoint(root,experts=total+1)
