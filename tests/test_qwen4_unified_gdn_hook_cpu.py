"""Run actual opt-in hook admission/transaction code with CPU host stubs."""
import ast
from importlib.metadata import distribution
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).parents[1] / "omlx/patches/qwen4_unified_gdn_verify.py"


def array(shape, dtype="bf16"):
    return SimpleNamespace(shape=shape, dtype=dtype)


class HookTests(unittest.TestCase):
    def setup_hook(self):
        names = {"_decline", "_admission", "reset_receipt", "receipt", "try_fused_rms"}
        body = [n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
        module = type("Qwen4ExpGatedDeltaNet", (), {})()
        for name, value in dict(training=False, num_k_heads=16, num_v_heads=48,
                                head_k_dim=128, head_v_dim=128, conv_kernel_size=4,
                                A_log=array((48,)), dt_bias=array((48,)),
                                norm=SimpleNamespace(activation="sigmoid", eps=1e-6, weight=array((128,))),
                                conv1d=SimpleNamespace(bias=None, weight=array((10240, 4, 1))),
                                in_proj_qkv="qkv", in_proj_z="z", in_proj_a="a", in_proj_b="b", out_proj="out").items():
            setattr(module, name, value)
        class ArraysCache(list):
            lengths = left_padding = None
            def advance(self, count):
                self.advanced = getattr(self, "advanced", 0) + count
        q35 = ModuleType("mlx_vlm.models.qwen3_5.language")
        q35._target_verify_linears = Mock(return_value=[array((1, 3, d)) for d in (10240, 6144, 48, 48)])
        q35._target_verify_linear = Mock(return_value="output")
        q35._qwen3_5_advance_left_padding_info = Mock()
        q35._qwen3_5_advance_lengths_info = Mock()
        generic = ModuleType("omlx.patches.qwen35_gdn_prework")
        generic._qwen4_scales = Mock(return_value=("qs", "ks"))
        generic.gdn_prework_fused = Mock(return_value=("lazy_q", "lazy_k", "lazy_v", "unused"))
        packages = {name: ModuleType(name) for name in ("mlx_vlm", "mlx_vlm.models", "mlx_vlm.models.qwen3_5", "omlx", "omlx.patches", "mlx_lm", "mlx_lm.models", "mlx_lm.models.cache", "mlx_vlm.models.qwen4_exp", "mlx_vlm.models.qwen4_exp.cache")}
        packages["mlx_lm.models.cache"].ArraysCache = ArraysCache
        packages["mlx_vlm.models.qwen4_exp.cache"].ArraysCache = ArraysCache
        packages["mlx_vlm.models.qwen3_5"].language = q35
        packages["omlx.patches"].qwen35_gdn_prework = generic
        packages[q35.__name__], packages[generic.__name__] = q35, generic
        kernel = Mock(return_value=("flat", "next_conv", "next_state", "snapshots", "conv_snapshots"))
        namespace = {"Cache": ArraysCache, "__package__": "omlx.patches", "_DISABLED": False,
                     "_RECEIPT": {"calls": 0, "declines": {}, "failures": 0},
                     "qwen4_unified_gdn_verify_rms": kernel,
                     "mx": SimpleNamespace(bfloat16="bf16", float32="fp32", gpu="gpu", default_device=lambda: "gpu",
                                           metal=SimpleNamespace(is_available=lambda: True), concatenate=lambda *a, **k: "conv_input")}
        exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
        return namespace, module, packages, q35, generic, kernel

    def test_admission_declines_width_padding_state_and_unknown_cache(self):
        ns, module, packages, _, _, kernel = self.setup_hook()
        good = ns["Cache"]([array((1, 3, 10240)), array((1, 48, 128, 128), "fp32")])
        cases = [(array((1, 8, 2560)), good, "input not"),
                 (array((1, 3, 2560)), ns["Cache"]([good[0], None]), "state/parameter")]
        padded = ns["Cache"](good); padded.left_padding = [1]
        cases.append((array((1, 3, 2560)), padded, "padded cache"))
        class UnknownCache(ns["Cache"]):
            pass
        cases.append((array((1, 3, 2560)), UnknownCache(good), "cache is outside"))
        with patch.dict(sys.modules, packages):
            for inputs, cache, reason in cases:
                self.assertTrue(ns["_admission"](module, inputs, None, cache, []).startswith(reason))
        self.assertFalse(kernel.called)

    def test_preserves_full_legacy_sink_and_commits_after_output(self):
        ns, module, packages, q35, generic, kernel = self.setup_hook()
        cache = ns["Cache"]([array((1, 3, 10240)), array((1, 48, 128, 128), "fp32")])
        original_state = cache[1]
        sink = []
        with patch.dict(sys.modules, packages):
            self.assertEqual(ns["try_fused_rms"](module, array((1, 3, 2560)), None, cache, sink), "output")
        self.assertEqual(cache, ["next_conv", "next_state"])
        self.assertEqual(cache.advanced, 3)
        q35._qwen3_5_advance_left_padding_info.assert_called_once_with(cache, 3)
        q35._qwen3_5_advance_lengths_info.assert_called_once_with(cache, 3)
        self.assertEqual(sink[0][:3], ("lazy_q", "lazy_k", "lazy_v"))
        self.assertIs(sink[0][7], original_state)
        self.assertEqual(sink[0][9:], ("conv_input", 4, "snapshots"))
        self.assertEqual(ns["receipt"]()["calls"], 1)
        self.assertEqual(q35._target_verify_linear.call_args.args, ("out", "flat", True))

    def test_cache_commit_error_fail_stops_instead_of_falling_back(self):
        ns, module, packages, _, _, kernel = self.setup_hook()
        class BrokenCache(ns["Cache"]):
            def __setitem__(self, index, value):
                if index == 1:
                    raise RuntimeError("injected commit failure")
                return super().__setitem__(index, value)
        packages["mlx_lm.models.cache"].ArraysCache = BrokenCache
        cache = BrokenCache([array((1, 3, 10240)), array((1, 48, 128, 128), "fp32")])
        with patch.dict(sys.modules, packages):
            with self.assertRaisesRegex(RuntimeError, "commit failure"):
                ns["try_fused_rms"](module, array((1, 3, 2560)), None, cache, [])
        self.assertEqual(kernel.call_count, 1)
        self.assertFalse(ns["receipt"]()["exception_fuse"])
        self.assertEqual(ns["receipt"]()["calls"], 0)

    def test_actual_arrays_cache_advance_methods_in_hook_are_noops_for_admitted_lane(self):
        root = SOURCE.parents[2]
        sources = [root / "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/cache.py"]
        sources.append(Path(distribution("mlx-lm").locate_file("mlx_lm/models/cache.py")))
        for source in sources:
            cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "ArraysCache")
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "advance")
            actual = {}
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), actual)
            ns, module, packages, _, _, _ = self.setup_hook()
            calls = []
            def advance(cache, count):
                calls.append(count)
                return actual["advance"](cache, count)
            ns["Cache"].advance = advance
            cache = ns["Cache"]([array((1, 3, 10240)), array((1, 48, 128, 128), "fp32")])
            cache._lengths = cache._left_padding = None
            cache._lengths_advance, cache._left_padding_advance = 7, 11
            with patch.dict(sys.modules, packages):
                self.assertEqual(ns["try_fused_rms"](module, array((1, 3, 2560)), None, cache, []), "output")
            self.assertEqual(calls, [3], str(source))
            self.assertIsNone(cache.lengths)
            self.assertIsNone(cache.left_padding)
            self.assertEqual((cache._lengths_advance, cache._left_padding_advance), (7, 11))

    def test_advance_exception_after_mutation_propagates_without_fallback(self):
        ns, module, packages, _, _, kernel = self.setup_hook()
        def advance(cache, count):
            cache.advanced = count
            raise RuntimeError("injected advance failure")
        ns["Cache"].advance = advance
        cache = ns["Cache"]([array((1, 3, 10240)), array((1, 48, 128, 128), "fp32")])
        with patch.dict(sys.modules, packages):
            with self.assertRaisesRegex(RuntimeError, "advance failure"):
                ns["try_fused_rms"](module, array((1, 3, 2560)), None, cache, [])
        self.assertEqual(cache, ["next_conv", "next_state"])
        self.assertEqual(cache.advanced, 3)
        self.assertFalse(ns["receipt"]()["exception_fuse"])
        self.assertEqual(ns["receipt"]()["calls"], 0)
        self.assertEqual(kernel.call_count, 1)

    def test_late_failure_restores_ownership_and_latches_only_candidate(self):
        ns, module, packages, q35, _, kernel = self.setup_hook()
        q35._target_verify_linear.side_effect = RuntimeError("injected output failure")
        cache = ns["Cache"]([array((1, 3, 10240)), array((1, 48, 128, 128), "fp32")])
        original = list(cache)
        sink = ["existing"]
        with patch.dict(sys.modules, packages), self.assertLogs(level="WARNING"):
            self.assertIsNone(ns["try_fused_rms"](module, array((1, 3, 2560)), None, cache, sink))
        self.assertEqual(cache, original)
        self.assertEqual(sink, ["existing"])
        self.assertTrue(ns["receipt"]()["exception_fuse"])
        self.assertIsNone(ns["try_fused_rms"](module, array((1, 3, 2560)), None, cache, sink))
        self.assertEqual(kernel.call_count, 1)
        self.assertEqual(ns["receipt"]()["calls"], 0)


if __name__ == "__main__":
    unittest.main()
