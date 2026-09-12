"""Host-only contract for the explicitly carried #3553 scale dependency."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

class ScaleDependencyTests(unittest.TestCase):
    def test_cached_bf16_q_and_k_scales_preserve_rounding_boundaries(self):
        path=Path(__file__).resolve().parents[1]/"omlx/patches/qwen35_gdn_prework.py"
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="_qwen4_scales")
        calls=[]
        def array(value,dtype):
            entry=SimpleNamespace(value=value,dtype=dtype);calls.append(entry);return entry
        ns={"mx":SimpleNamespace(array=array,bfloat16="bf16")}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),ns)
        module=SimpleNamespace(head_k_dim=128)
        q,k=ns["_qwen4_scales"](module)
        self.assertAlmostEqual(q.value,1/128)
        self.assertAlmostEqual(k.value,128**-0.5)
        self.assertEqual([x.dtype for x in calls],["bf16","bf16"])
        again=ns["_qwen4_scales"](module)
        self.assertIs(again[0],q);self.assertIs(again[1],k)
        self.assertEqual(len(calls),2)
