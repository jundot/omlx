"""Qwen4 JANGH2 cache interop. Tests intentionally avoid GPU / MLX dependencies."""
import importlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

class FakeArray:
    def __init__(self, size=4):
        self.cache = [None] * size
        self.left_padding = None

class FakeKV:
    pass

class FakeBatchKV:
    def __init__(self, padding):
        self.padding = padding
        self.offset = [SimpleNamespace(item=lambda: 0) for _ in padding]

class FakeRotating:
    def __init__(self, max_size=1024, keep=0):
        self.max_size = max_size
        self.keep = keep

class FakeBatchRotating:
    def __init__(self, max_size, padding):
        self.max_size = max_size
        self.padding = padding

class FakeCacheList:
    def __init__(self, *caches):
        self.caches = caches

class FakeBatchQSA:
    def __init__(self, padding):
        self.padding = padding
        self._idx = 0
    def extend(self, other):
        pass

class FakeQSA:
    def to_batch(self, padding):
        return FakeBatchQSA(padding)

class FakeModel:
    def __init__(self, marked=True, cache=None):
        self._vlm_model = SimpleNamespace(_jangh2_bridge_model=marked)
        self._cache = cache if cache is not None else [FakeArray(), FakeQSA()]
    def make_cache(self):
        return self._cache

class CacheCompatTests(TestCase):
    def setUp(self):
        self.scheduler = ModuleType('omlx.scheduler')
        self.scheduler.Scheduler = type('Scheduler', (), {'shutdown': lambda self: None})
        self.scheduler._make_cache = self.unpatched_make_cache
        self.vlm = ModuleType('omlx.models.vlm')
        self.vlm._wrap_caches = lambda caches: [('PROXIED', c) for c in caches]
        fake = {
            'omlx': ModuleType('omlx'),
            'omlx.scheduler': self.scheduler,
            'omlx.engine_core': ModuleType('omlx.engine_core'),
            'omlx.models': ModuleType('omlx.models'),
            'omlx.models.vlm': self.vlm,
            'mlx': ModuleType('mlx'),
            'mlx.core': ModuleType('mlx.core'),
            'mlx_vlm': ModuleType('mlx_vlm'),
            'mlx_vlm.models': ModuleType('mlx_vlm.models'),
            'mlx_vlm.models.cache': ModuleType('mlx_vlm.models.cache'),
            'mlx_vlm.models.qwen4_exp': ModuleType('mlx_vlm.models.qwen4_exp'),
            'mlx_vlm.models.qwen4_exp.language': ModuleType('mlx_vlm.models.qwen4_exp.language'),
            'mlx_lm': ModuleType('mlx_lm'),
            'mlx_lm.generate': ModuleType('mlx_lm.generate'),
        }
        fake['mlx.core'].array = lambda seq: ('MX_ARRAY',tuple(seq))
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mlx-global-test')
        fake['omlx.engine_core'].get_mlx_executor = lambda: self.executor
        fake['mlx.core'].default_device = lambda: 'gpu'
        fake['mlx.core'].new_stream = lambda device: ('STREAM', threading.get_ident(), device)
        self.main_thread_stream = ('STREAM', threading.get_ident(), 'gpu')
        self.scheduler.generation_stream = self.main_thread_stream
        fake['mlx_vlm.models.cache'].ArraysCache = FakeArray
        fake['mlx_vlm.models.cache'].KVCache = FakeKV
        fake['mlx_vlm.models.cache'].BatchKVCache = FakeBatchKV
        fake['mlx_vlm.models.cache'].RotatingKVCache = FakeRotating
        fake['mlx_vlm.models.cache'].BatchRotatingKVCache = FakeBatchRotating
        fake['mlx_vlm.models.cache'].CacheList = FakeCacheList
        fake['mlx_vlm.models.qwen4_exp.language'].QSAKVCache = FakeQSA
        fake['mlx_vlm.models.qwen4_exp.language'].BatchQSAKVCache = FakeBatchQSA
        fake['mlx_lm.generate']._make_cache = self.unpatched_make_cache
        fake['mlx_lm.generate'].generation_stream = self.main_thread_stream
        # mlx_lm/__init__.py exposes `generate` as a callable; this can
        # shadow the module when imported with `import mlx_lm.generate as x`.
        fake['mlx_lm'].generate = lambda *a, **k: None
        self.patcher = patch.dict(sys.modules,fake)
        self.patcher.start()
        sys.modules.pop('jangh2_bridge.scheduler_compat',None)
        self.compat = importlib.import_module('jangh2_bridge.scheduler_compat')
    def tearDown(self):
        self.compat_module = sys.modules.pop('jangh2_bridge.scheduler_compat',None)
        self.patcher.stop()
        self.executor.shutdown(wait=True)

    @staticmethod
    def unpatched_make_cache(model, padding, max_kv_size):
        for c in model.make_cache():
            raise ValueError(f'{type(c)} does not yet support batching')

    def test_upstream_batcher_rejects_vlm_cache(self):
        with self.assertRaisesRegex(ValueError,'does not yet support batching'):
            self.unpatched_make_cache(FakeModel(),[0],None)

    def test_jangh2_arrays_and_qsa_are_converted(self):
        uninstall = self.compat.install()
        try:
            values=self.scheduler._make_cache(FakeModel(),[3,0],None)
            self.assertEqual(len(values),2)
            self.assertIsInstance(values[0],FakeArray)
            self.assertEqual(values[0].left_padding,('MX_ARRAY',(3,0)))
            self.assertIsInstance(values[1],FakeBatchQSA)
            self.assertEqual(values[1].padding,[3,0])
            self.assertEqual(self.vlm._wrap_caches(values),values)
            # mlx-lm's copy also uses the model-scoped conversion
            self.assertEqual(len(sys.modules['mlx_lm.generate']._make_cache(FakeModel(),[0],None)),2)
        finally:
            uninstall()

    def test_unmarked_qsa_cache_fails_closed(self):
        # A JANGH2 checkpoint loaded without its marker must not silently
        # convert QSA state through the generic VLM path.
        uninstall=self.compat.install()
        try:
            with self.assertRaisesRegex(ValueError,'does not yet support batching'):
                self.scheduler._make_cache(FakeModel(marked=False),[0],None)
        finally: uninstall()

    def test_unmarked_vlm_model_converts_native_caches(self):
        # Qwen3.5-family hybrids: mlx-lm cannot batch mlx_vlm-native layers,
        # so the hook converts them with MLX-VLM's own batch classes.
        uninstall=self.compat.install()
        try:
            values=self.scheduler._make_cache(
                FakeModel(marked=False,cache=[FakeArray(),FakeKV()]),[3,0],None)
            self.assertEqual(len(values),2)
            self.assertIsInstance(values[0],FakeArray)
            self.assertEqual(values[0].left_padding,('MX_ARRAY',(3,0)))
            self.assertIsInstance(values[1],FakeBatchKV)
            self.assertEqual(values[1].padding,[3,0])
        finally: uninstall()

    def test_stock_supported_model_passes_through_untouched(self):
        self.scheduler._make_cache=lambda model,padding,max_kv_size: [
            ('STOCK',c) for c in model.make_cache()]
        uninstall=self.compat.install()
        try:
            values=self.scheduler._make_cache(FakeModel(marked=False),[0],None)
            self.assertEqual([v[0] for v in values],['STOCK','STOCK'])
        finally: uninstall()

    def test_unmarked_rotating_cache_converts(self):
        uninstall=self.compat.install()
        try:
            values=self.scheduler._make_cache(
                FakeModel(marked=False,cache=[FakeRotating(max_size=512)]),[2],None)
            self.assertIsInstance(values[0],FakeBatchRotating)
            self.assertEqual(values[0].max_size,512)
            self.assertEqual(values[0].padding,[2])
        finally: uninstall()

    def test_unmarked_rotating_with_keep_fails_closed(self):
        uninstall=self.compat.install()
        try:
            with self.assertRaisesRegex(ValueError,'keep tokens'):
                self.scheduler._make_cache(
                    FakeModel(marked=False,cache=[FakeRotating(keep=4)]),[0],None)
        finally: uninstall()

    def test_unmarked_cachelist_recurses(self):
        uninstall=self.compat.install()
        try:
            values=self.scheduler._make_cache(
                FakeModel(marked=False,cache=[FakeCacheList(FakeKV(),FakeArray())]),[5],None)
            self.assertIsInstance(values[0],FakeCacheList)
            self.assertIsInstance(values[0].caches[0],FakeBatchKV)
            self.assertEqual(values[0].caches[1].left_padding,('MX_ARRAY',(5,)))
        finally: uninstall()

    def test_unmarked_model_without_make_cache_reraises_original(self):
        def boom(model,padding,max_kv_size):
            raise ValueError('does not yet support batching: boom-original')
        self.scheduler._make_cache=boom
        uninstall=self.compat.install()
        try:
            model=SimpleNamespace(_vlm_model=SimpleNamespace(_jangh2_bridge_model=False))
            with self.assertRaisesRegex(ValueError,'boom-original'):
                self.scheduler._make_cache(model,[0],None)
        finally: uninstall()

    def test_unmarked_non_list_make_cache_reraises_original(self):
        def boom(model,padding,max_kv_size):
            raise ValueError('does not yet support batching: boom-original')
        self.scheduler._make_cache=boom
        uninstall=self.compat.install()
        try:
            model=FakeModel(marked=False)
            model.make_cache=lambda: object()
            with self.assertRaisesRegex(ValueError,'boom-original'):
                self.scheduler._make_cache(model,[0],None)
        finally: uninstall()

    def test_unmarked_unrelated_value_error_propagates(self):
        def boom(model,padding,max_kv_size):
            raise ValueError('boom-unrelated-config-error')
        self.scheduler._make_cache=boom
        uninstall=self.compat.install()
        try:
            with self.assertRaisesRegex(ValueError,'boom-unrelated-config-error'):
                self.scheduler._make_cache(FakeModel(marked=False,cache=[FakeKV()]),[0],None)
        finally: uninstall()

    def test_unmarked_unsupported_cache_fails_closed(self):
        uninstall=self.compat.install()
        try:
            with self.assertRaisesRegex(ValueError,'does not yet support batching'):
                self.scheduler._make_cache(FakeModel(marked=False,cache=[object()]),[0],None)
        finally: uninstall()

    def test_hybrid_vlm_wrap_preserves_native_batched_offsets_and_list(self):
        uninstall=self.compat.install()
        try:
            values=self.scheduler._make_cache(
                FakeModel(marked=False,cache=[FakeArray(),FakeKV()]),[0,0],None)
            wrapped=self.vlm._wrap_caches(values)
            # Qwen3.5-family forwards index the native offset array, then
            # replace cache entries after extracting/merging singleton rows.
            # Both operations must be visible to the scheduler's cache list.
            self.assertEqual(wrapped[1].offset[0].item(),0)
            self.assertIs(wrapped,values)
            self.assertIs(wrapped[0],values[0])
            replacement=FakeBatchKV([0,0])
            wrapped[1]=replacement
            self.assertIs(values[1],replacement)
        finally: uninstall()

    def test_hybrid_vlm_rotating_cache_remains_native(self):
        uninstall=self.compat.install()
        try:
            values=self.scheduler._make_cache(
                FakeModel(marked=False,cache=[FakeArray(),FakeRotating()]),[0],None)
            self.assertIs(self.vlm._wrap_caches(values),values)
            self.assertIsInstance(values[1],FakeBatchRotating)
        finally: uninstall()

    def test_kv_only_vlm_batch_stays_fully_proxied(self):
        uninstall=self.compat.install()
        try:
            values=self.scheduler._make_cache(
                FakeModel(marked=False,cache=[FakeKV(),FakeKV()]),[0,0],None)
            self.assertEqual(self.vlm._wrap_caches(values),[('PROXIED',v) for v in values])
        finally: uninstall()

    def test_unknown_cache_fails_closed(self):
        uninstall=self.compat.install()
        try:
            with self.assertRaisesRegex(TypeError,'unsupported Qwen4 JANGH2 cache'):
                self.scheduler._make_cache(FakeModel(cache=[object()]),[0],None)
        finally: uninstall()

    def test_mixed_batch_cache_type_refused(self):
        uninstall=self.compat.install()
        try:
            with self.assertRaisesRegex(TypeError,'unsupported Qwen4 JANGH2 cache'):
                self.scheduler._make_cache(FakeModel(cache=[FakeBatchQSA([0])]),[0],None)
        finally: uninstall()

    def test_generic_vlm_wrapping_is_unchanged(self):
        uninstall=self.compat.install()
        try:
            other=[object()]
            self.assertEqual(self.vlm._wrap_caches(other),[('PROXIED',other[0])])
        finally: uninstall()

    def test_generation_stream_belongs_to_global_mlx_worker(self):
        """MLX 0.31.2+ rejects a stream created on a different thread."""
        generate = sys.modules['mlx_lm.generate']
        worker_id = self.executor.submit(threading.get_ident).result()
        self.assertNotEqual(worker_id, threading.get_ident())
        self.assertEqual(generate.generation_stream[1], threading.get_ident())
        undo = self.compat.install()
        try:
            self.assertEqual(generate.generation_stream[1], worker_id)
            self.assertIs(self.scheduler.generation_stream, generate.generation_stream)
        finally:
            undo()

    def test_stream_restored_on_uninstall(self):
        generation = sys.modules['mlx_lm.generate']
        old = generation.generation_stream
        undo = self.compat.install()
        self.assertIsNot(generation.generation_stream, old)
        undo()
        self.assertIs(generation.generation_stream, old)
        self.assertIs(self.scheduler.generation_stream, old)

    def test_uninstall_restores_all_globals(self):
        old_make=self.scheduler._make_cache
        old_wrap=self.vlm._wrap_caches
        old_generate=sys.modules['mlx_lm.generate']._make_cache
        uninstall=self.compat.install()
        uninstall()
        self.assertIs(self.scheduler._make_cache,old_make)
        self.assertIs(self.vlm._wrap_caches,old_wrap)
        self.assertIs(sys.modules['mlx_lm.generate']._make_cache,old_generate)

    def test_repeated_install_refused(self):
        uninstall=self.compat.install()
        try:
            with self.assertRaisesRegex(RuntimeError,'already installed'):
                self.compat.install()
        finally: uninstall()

if __name__ == '__main__':
    import unittest
    unittest.main()
