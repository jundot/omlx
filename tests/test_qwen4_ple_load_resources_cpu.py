"""CPU-only loader ownership tests retaining the failed model traceback."""
# ruff: noqa: SIM117

import ast
import contextlib
import importlib.util
import json
import logging
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_qwen4_runtime_ple_fork_cpu as runtime

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_FILE = ROOT / "omlx/patches/mlx_vlm_qwen4_exp_compat/ple_load_resources.py"
spec = importlib.util.spec_from_file_location("ple_load_scope_tested", RESOURCE_FILE)
resources = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resources)


def wrapper():
    tree = ast.parse((ROOT / "omlx/engine/vlm.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_force_qwen4_exp_sanitize_on_load"
    )
    # Supply the exact resource module without importing the MLX engine/package.
    imports = [
        n
        for n in node.body
        if isinstance(n, ast.ImportFrom) and n.module.endswith("ple_load_resources")
    ]
    assert len(imports) == 1 and imports[0].names[0].name == "ple_load_resources"
    node.body.remove(imports[0])
    ns = dict(
        contextlib=contextlib,
        Path=Path,
        ple_load_resources=resources.ple_load_resources,
        _read_config_model_type=lambda _: "qwen4_exp",
        _model_shard_matcher=lambda _: lambda _: False,
        logger=logging.getLogger(__name__),
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "live-vlm-wrapper", "exec"),
        ns,
    )
    return ns["_force_qwen4_exp_sanitize_on_load"]


class LoadResourceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = runtime.RuntimePLEForkTests()
        self.fixture.setUp()
        runtime.NS["register_ple_resource"] = resources.register_ple_resource
        self.real_mx = runtime.NS["mx"]
        runtime.NS["mx"] = types.SimpleNamespace(
            ones=lambda shape, dtype: np.ones(shape, dtype=dtype), bfloat16=np.float16
        )
        # Re-key the tiny real safetensors file for the actual embedding constructor.
        data = self.fixture.path.read_bytes()
        size = runtime.struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + size])
        header["table.shard_0.weight"] = header.pop("weight")
        encoded = json.dumps(header).encode()
        self.model_file = Path(self.fixture.temp.name) / "model.safetensors"
        self.model_file.write_bytes(
            runtime.struct.pack("<Q", len(encoded)) + encoded + data[8 + size :]
        )
        (Path(self.fixture.temp.name) / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"table.shard_0.weight": self.model_file.name}})
        )
        self.safe_open = lambda *a, **k: None
        self.modules = patch.dict(
            sys.modules,
            {"safetensors": types.SimpleNamespace(safe_open=self.safe_open)},
        )
        self.modules.start()

    def tearDown(self):
        self.modules.stop()
        runtime.NS["mx"] = self.real_mx
        self.fixture.tearDown()
        self.assertIsNone(resources._LOAD_RESOURCES.get())

    def embedding(self, shards=1):
        return runtime.NS["DiskBackedShardedEmbedding"](
            self.fixture.temp.name, "table", 16 * shards, 4, shards
        )

    def test_real_loader_wrapper_strict_failure_closes_retained_model(self):
        held = {}
        error = ValueError("strict missing weight")

        def loader():
            partial_model = self.embedding()
            held["model"] = partial_model
            held["reader"] = next(iter(partial_model._readers.values()))
            raise error

        caught = None
        try:
            with wrapper()(Path(self.fixture.temp.name)):
                loader()
        except ValueError as exc:
            caught = exc
        self.assertIs(caught, error)
        self.assertIsNotNone(error.__traceback__)
        self.assertIsNone(held["reader"]._file)
        self.assertIsNone(held["reader"]._mapping)
        self.assertTrue(held["model"]._prefetch_executor._shutdown)
        self.assertIs(sys.modules["safetensors"].safe_open, self.safe_open)
        self.assertIsNotNone(self.fixture.reader._file)  # Old model was never enlisted.

    def test_partial_embedding_constructor_failure_closes_opened_shard(self):
        error = None
        try:
            with wrapper()(Path(self.fixture.temp.name)):
                self.embedding(shards=2)  # Second shard is deliberately missing.
        except KeyError as exc:
            error = exc
        self.assertIsNotNone(error)
        tb = error.__traceback__
        while tb.tb_frame.f_code.co_name != "__init__":
            tb = tb.tb_next
        partial = tb.tb_frame.f_locals["self"]
        self.assertTrue(partial._prefetch_executor._shutdown)
        self.assertEqual(partial._readers, {})

    def test_success_transfers_ownership_and_nested_scopes_are_independent(self):
        with resources.ple_load_resources():
            successful = self.embedding()
        self.assertIsNotNone(next(iter(successful._readers.values()))._file)
        try:
            with resources.ple_load_resources():
                outer = self.embedding()
                with resources.ple_load_resources():
                    inner_success = self.embedding()
                try:
                    with resources.ple_load_resources():
                        inner_failure = self.embedding()
                        raise ValueError("inner")
                except ValueError:
                    pass
                self.assertEqual(inner_failure._readers, {})
                self.assertTrue(outer._readers)
                raise ValueError("outer")
        except ValueError:
            pass
        self.assertEqual(outer._readers, {})
        self.assertTrue(inner_success._readers)
        inner_success.close()
        successful.close()

    def test_concurrent_load_failure_does_not_close_other_scope(self):
        barrier = threading.Barrier(2)
        instances, errors = {}, []

        def load(name, fail):
            try:
                with resources.ple_load_resources():
                    instances[name] = self.embedding()
                    barrier.wait(timeout=2)
                    if fail:
                        raise ValueError(name)
            except ValueError:
                pass
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=load, args=("failure", True)),
            threading.Thread(target=load, args=("success", False)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(instances["failure"]._readers, {})
        self.assertTrue(instances["success"]._readers)
        instances["success"].close()

    def test_cleanup_orders_embedding_before_reader_and_preserves_error(self):
        events = []

        class Owner:
            def __init__(self, name, fails=False):
                self.name, self.fails = name, fails

            def close(self):
                events.append(self.name)
                if self.fails:
                    raise OSError("cleanup error")

        primary = RuntimeError("original strict error")
        with self.assertRaises(RuntimeError) as caught:
            with resources.ple_load_resources():
                resources.register_ple_resource(Owner("reader"))
                resources.register_ple_resource(Owner("embedding", True), priority=1)
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertEqual(events, ["embedding", "reader"])
        self.assertIn("PLE cleanup failed: OSError", primary.__notes__)


if __name__ == "__main__":
    unittest.main()
