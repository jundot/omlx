"""CPU-only source and graph ABI checks for the dormant RMS candidate."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = (Path(__file__).parents[1] / "omlx/patches/qwen4_unified_gdn_verify.py").read_text()
TREE = ast.parse(SOURCE)
CONSTANTS = {node.targets[0].id: ast.literal_eval(node.value) for node in TREE.body
             if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.Tuple))}


class RMSCandidateContractTests(unittest.TestCase):
    def test_every_non_normalization_kernel_byte_preserved(self):
        source = CONSTANTS["_SOURCE"]
        begin = source.index("    // Preserve oMLX generic RMS/scaling operation order exactly:")
        end = source.index("    device float* state_dst =", begin)
        rest = source[:begin] + source[end:]
        self.assertEqual(hashlib.sha256(rest.encode()).hexdigest(), CONSTANTS["NONNORMALIZATION_SHA256"])
        self.assertIn("float pq = 0.0f, pk = 0.0f", source[begin:end])
        self.assertIn("pq / float(DK) + 1.0e-6f", source[begin:end])
        self.assertIn("const T qrms = T(1) * T(sq[d] * shr[0])", source[begin:end])
        self.assertIn("sq[d] = float(T(qscale * qrms))", source[begin:end])
        self.assertNotIn("metal::precise::rsqrt(qdenom)", source[begin:end])

    def _wrapper(self):
        fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "qwen4_unified_gdn_verify_rms")
        calls = []
        def dispatch(**kwargs):
            calls.append(kwargs)
            return list(range(5))
        namespace = dict(CONSTANTS, mx=SimpleNamespace(float32="float32"), _kernel=lambda: dispatch)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "candidate-wrapper", "exec"), namespace)
        return namespace[fn.name], calls

    def test_graph_abi_preserves_all_rollback_slots(self):
        fn, calls = self._wrapper()
        qkv = SimpleNamespace(shape=(1, 3, 10240), dtype="bfloat16")
        self.assertEqual(fn(qkv, *([None] * 9), 1e-6, threadgroup_y=8), (0, 1, 2, 3, 4))
        self.assertEqual(calls[0]["output_shapes"][3], (1, 2, 48, 128, 128))
        self.assertEqual(calls[0]["output_shapes"][4], (1, 2, 3, 10240))
        self.assertEqual(calls[0]["output_dtypes"], ["bfloat16", "bfloat16", "float32", "float32", "bfloat16"])

    def test_unqualified_width_and_geometry_refused_before_dispatch(self):
        fn, calls = self._wrapper()
        for width, ty in ((2, 8), (9, 8), (3, 7)):
            with self.assertRaises(ValueError):
                fn(SimpleNamespace(shape=(1, width, 10240), dtype="bfloat16"), *([None] * 10), threadgroup_y=ty)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
