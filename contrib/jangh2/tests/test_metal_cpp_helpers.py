"""Compile the exact kernel bit reader / literal codebooks as host C++.

This tests C++ arithmetic syntax and packed decoding, NOT Metal compilation,
SIMD execution, barriers or GPU performance. Those need the on-Mac preflight.
"""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
from test_metal_backend import make_spec
from jangh2_bridge.metal_ops import build_gate_up
from jangh2_bridge.reference_ops import pack_codes

class CompiledHelperTests(unittest.TestCase):
    def test_exact_emitted_cpp_helper_matches_each_code_and_three_bit_spills(self):
        compiler=shutil.which('c++')
        if compiler is None:
            self.skipTest('No host C++ compiler; run check_performance.py for real GPU tests')
        for bits in (2,3,4):
            with self.subTest(bits=bits),tempfile.TemporaryDirectory() as td:
                header=build_gate_up(make_spec(bits=(bits,bits,bits)),8,False).header
                header=header.replace('const device uint*','const uint*')
                source='#include <cstdint>\nusing uint=uint32_t;\n'+header+f'''
extern "C" void decode(const uint* p, float* y, uint n) {{
  for (uint i=0;i<n;++i) y[i]=jangh_level_g(jangh_read<{bits}>(p,i));
}}
'''
                p=Path(td)/'test.cpp';p.write_text(source)
                lib=Path(td)/'test.so'
                result=subprocess.run([compiler,'-std=c++17','-shared','-fPIC','-O2',str(p),'-o',str(lib)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                fn=ctypes.CDLL(str(lib)).decode
                fn.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_float),ctypes.c_uint32]
                fn.restype=None
                rng=np.random.default_rng(31+bits)
                for width in (32,96,640,2560):
                    codes=rng.integers(0,1<<bits,width,dtype=np.uint32)
                    codes[:1<<bits]=np.arange(1<<bits)
                    packed=np.ascontiguousarray(pack_codes(codes,bits)); y=np.empty(width,np.float32)
                    fn(packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),width)
                    expected=np.asarray(make_spec(bits=(bits,bits,bits)).levels[0],np.float32)[codes]
                    np.testing.assert_array_equal(y,expected)

if __name__=='__main__': unittest.main()
