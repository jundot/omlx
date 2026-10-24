"""Compile emitted C++ with Metal builtin stand-ins; not a Metal/GPU test."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from jangh2_bridge.grouped_ops import build_grouped_gate,build_grouped_down
from test_metal_backend import make_spec

class GroupedSyntaxTests(unittest.TestCase):
    def test_emitted_grouped_cpp_is_well_formed_for_mixed_bits_and_rotations(self):
        cxx=shutil.which('c++')
        if not cxx:self.skipTest('Host C++ compiler unavailable')
        common='''#include <cmath>
#include <cstdint>
using uint=uint32_t; using ushort=uint16_t; using ulong=uint64_t; using OutType=float;
#define device
#define threadgroup
struct Position {uint x,y,z;};
Position thread_position_in_threadgroup,threadgroup_position_in_grid;
namespace mem_flags {enum Kind {mem_threadgroup};}
void threadgroup_barrier(mem_flags::Kind) {}
float simd_sum(float x) {return x;}
float simd_shuffle_xor(float x,ushort) {return x;}
float simd_shuffle_down(float x,ushort) {return x;}
template<class T> T as_type(uint x) {return T(x);}
using std::exp;using std::abs;using std::fma;
namespace metal { namespace precise { using std::exp; } }
'''
        with tempfile.TemporaryDirectory() as d:
            for block in (2,4):
                for rotations in (('hadamard32',)*3,('none','hadamard32','none')):
                    spec=make_spec(bits=(2,3,4),rotations=rotations,d=96,h=64,e=5)
                    for kernel in (build_grouped_gate(spec,10,block),build_grouped_down(spec,block)):
                        uints={'wg','wu','wd','starts','tile_ends','token_rows'}
                        args=[f'const {"uint" if name in uints else "float"}* {name}' for name in kernel.inputs]+['float* out']
                        src=common+kernel.header+'\nvoid test('+','.join(args)+') {\n'+kernel.source+'\n}\n'
                        path=Path(d)/'kernel.cpp';path.write_text(src)
                        result=subprocess.run([cxx,'-std=c++17','-fsyntax-only',str(path)],text=True,capture_output=True)
                        self.assertEqual(result.returncode,0,result.stderr)
