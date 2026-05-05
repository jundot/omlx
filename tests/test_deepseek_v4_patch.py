# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek V4 monkey-patch (PR 1192 port)."""

import importlib
import sys

import pytest


@pytest.fixture(scope="module")
def applied_patch():
    """Apply the patch once for the whole module. The patch itself is
    idempotent so repeated calls are safe."""
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    return True


class TestPatchOrchestration:
    """Top-level apply / idempotency / module registration checks."""

    def test_apply_returns_true_first_time(self):
        from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch, is_applied

        # The patch may have been applied by a previous test run in the
        # same process; force-reset is_applied to validate the flow.
        # The module-level _APPLIED guard means we cannot un-apply, so
        # this test is informational about the *current* state.
        if is_applied():
            assert apply_deepseek_v4_patch() is False
        else:
            assert apply_deepseek_v4_patch() is True
            assert is_applied() is True

    def test_apply_is_idempotent(self, applied_patch):
        from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

        # After fixture has applied the patch, a second call must return False.
        assert apply_deepseek_v4_patch() is False

    def test_hyper_connection_registered(self, applied_patch):
        assert "mlx_lm.models.hyper_connection" in sys.modules

    def test_deepseek_v4_registered(self, applied_patch):
        assert "mlx_lm.models.deepseek_v4" in sys.modules

    def test_deepseek_v4_module_package(self, applied_patch):
        mod = sys.modules["mlx_lm.models.deepseek_v4"]
        # __package__ must be mlx_lm.models so relative imports inside
        # the loaded file resolve through the real mlx_lm package.
        assert mod.__package__ == "mlx_lm.models"


class TestCacheInjection:
    """PoolingCache / BatchPoolingCache injected into mlx_lm.models.cache."""

    def test_pooling_cache_attribute(self, applied_patch):
        import mlx_lm.models.cache as cache_mod

        assert hasattr(cache_mod, "PoolingCache")
        assert hasattr(cache_mod, "BatchPoolingCache")

    def test_pooling_cache_module_attribute(self, applied_patch):
        from mlx_lm.models.cache import BatchPoolingCache, PoolingCache

        # The injected classes claim to live in mlx_lm.models.cache so
        # any introspection (e.g. type(c).__module__) sees the right name.
        assert PoolingCache.__module__ == "mlx_lm.models.cache"
        assert BatchPoolingCache.__module__ == "mlx_lm.models.cache"

    def test_pooling_cache_instantiation(self, applied_patch):
        from mlx_lm.models.cache import PoolingCache

        cache = PoolingCache(ratio=4)
        assert cache.ratio == 4
        assert cache.empty()
        assert cache.size() == 0
        assert cache.offset == 0


class TestUtilsPatch:
    """mlx_lm.utils.load_model + _load_safetensors + SAFETENSORS_DTYPE_FALLBACKS."""

    def test_load_model_replaced(self, applied_patch):
        import mlx_lm.utils as utils_mod

        # The replaced function carries our docstring marker via its
        # bound name; just check it's not the upstream one by virtue of
        # the new attributes around it.
        assert hasattr(utils_mod, "_load_safetensors")
        assert hasattr(utils_mod, "SAFETENSORS_DTYPE_FALLBACKS")

    def test_dtype_fallback_map(self, applied_patch):
        import mlx_lm.utils as utils_mod

        assert utils_mod.SAFETENSORS_DTYPE_FALLBACKS == {"F8_E8M0": "U8"}

    def test_load_safetensors_passthrough_for_normal_dtype(
        self, applied_patch, tmp_path
    ):
        """A safetensors file with a standard dtype must round-trip
        through _load_safetensors unchanged (no header rewrite)."""
        import mlx.core as mx
        from mlx_lm.utils import _load_safetensors

        path = tmp_path / "model.safetensors"
        data = {"x": mx.zeros((4, 4), dtype=mx.float32)}
        mx.save_safetensors(str(path), data)
        loaded = _load_safetensors(str(path))
        assert "x" in loaded
        assert loaded["x"].shape == (4, 4)


class TestGeneratePatch:
    """mlx_lm.generate._make_cache replaced."""

    def test_make_cache_replaced(self, applied_patch):
        gen_mod = importlib.import_module("mlx_lm.generate")

        assert hasattr(gen_mod, "_make_cache")
        # Source must include PoolingCache → BatchPoolingCache branch.
        # We can't easily compare functions, so just verify the new
        # behavior: passing a model with a PoolingCache in make_cache
        # produces a BatchPoolingCache.
        from mlx_lm.models.cache import BatchPoolingCache, PoolingCache

        class FakeModel:
            def __init__(self):
                self.layers = [None]

            def make_cache(self):
                return [PoolingCache(ratio=4)]

        result = gen_mod._make_cache(FakeModel(), [0], None)
        assert len(result) == 1
        assert isinstance(result[0], BatchPoolingCache)


class TestTokenizerPatch:
    """mlx_lm.tokenizer_utils.AutoTokenizer wrapped with deepseek_v4 fallback."""

    def test_autotokenizer_wrapped(self, applied_patch):
        import mlx_lm.tokenizer_utils as tu

        # Wrapped class still exposes from_pretrained.
        assert hasattr(tu.AutoTokenizer, "from_pretrained")
        # Class name preserved for any introspection.
        assert tu.AutoTokenizer.__name__ == "AutoTokenizer"

    def test_passthrough_on_success(self, applied_patch):
        """When upstream AutoTokenizer.from_pretrained succeeds, the wrapper
        must return its result unmodified — no fallback path taken."""
        from unittest.mock import patch as mock_patch

        from omlx.patches.deepseek_v4 import tokenizer_patch

        sentinel = object()

        class _FakeUpstream:
            calls = []

            @staticmethod
            def from_pretrained(model_path, *args, **kwargs):
                _FakeUpstream.calls.append((model_path, args, kwargs))
                return sentinel

        with mock_patch("transformers.AutoTokenizer", _FakeUpstream):
            wrapper = tokenizer_patch._build_wrapper()
            result = wrapper.from_pretrained("/fake/path", trust_remote_code=True)

        assert result is sentinel
        assert len(_FakeUpstream.calls) == 1
        # Fallback never injected its own config kwarg.
        assert "config" not in _FakeUpstream.calls[0][2]

    def test_fallback_on_max_position_embeddings_error(self, applied_patch):
        """The exact AttributeError that transformers raises when it cannot
        recognize deepseek_v4 must trigger a retry with PreTrainedConfig()."""
        from unittest.mock import patch as mock_patch

        from omlx.patches.deepseek_v4 import tokenizer_patch

        class _FakeUpstream:
            calls = []

            @staticmethod
            def from_pretrained(model_path, *args, **kwargs):
                _FakeUpstream.calls.append((model_path, args, kwargs))
                if "config" in kwargs:
                    return "FALLBACK_OK"
                raise AttributeError(
                    "'PreTrainedConfig' object has no attribute "
                    "'max_position_embeddings'"
                )

        with mock_patch("transformers.AutoTokenizer", _FakeUpstream):
            wrapper = tokenizer_patch._build_wrapper()
            result = wrapper.from_pretrained("/fake/path")

        assert result == "FALLBACK_OK"
        assert len(_FakeUpstream.calls) == 2
        # Second call must inject config=PreTrainedConfig().
        assert "config" in _FakeUpstream.calls[1][2]

    def test_fallback_on_deepseek_v4_value_error(self, applied_patch):
        """ValueError mentioning deepseek_v4 also triggers fallback."""
        from unittest.mock import patch as mock_patch

        from omlx.patches.deepseek_v4 import tokenizer_patch

        class _FakeUpstream:
            calls = []

            @staticmethod
            def from_pretrained(model_path, *args, **kwargs):
                _FakeUpstream.calls.append((model_path, args, kwargs))
                if "config" in kwargs:
                    return "FALLBACK_OK"
                raise ValueError("Unrecognized configuration class for deepseek_v4")

        with mock_patch("transformers.AutoTokenizer", _FakeUpstream):
            wrapper = tokenizer_patch._build_wrapper()
            result = wrapper.from_pretrained("/fake/path")

        assert result == "FALLBACK_OK"
        assert len(_FakeUpstream.calls) == 2

    def test_unrelated_error_reraises(self, applied_patch):
        """Errors outside the deepseek_v4 / max_position_embeddings signature
        must NOT be swallowed."""
        import pytest as _pytest
        from unittest.mock import patch as mock_patch

        from omlx.patches.deepseek_v4 import tokenizer_patch

        class _FakeUpstream:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise ValueError("totally unrelated error")

        with mock_patch("transformers.AutoTokenizer", _FakeUpstream):
            wrapper = tokenizer_patch._build_wrapper()
            with _pytest.raises(ValueError, match="totally unrelated"):
                wrapper.from_pretrained("/fake/path")

    def test_explicit_config_skips_fallback(self, applied_patch):
        """If the caller already passed config=, we must not override it
        even when the inner call raises a matching error."""
        import pytest as _pytest
        from unittest.mock import patch as mock_patch

        from omlx.patches.deepseek_v4 import tokenizer_patch

        class _FakeUpstream:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                # Caller-provided config is in kwargs; we still raise the
                # max_position_embeddings error to verify the wrapper does
                # not silently retry.
                raise AttributeError(
                    "'PreTrainedConfig' object has no attribute "
                    "'max_position_embeddings'"
                )

        with mock_patch("transformers.AutoTokenizer", _FakeUpstream):
            wrapper = tokenizer_patch._build_wrapper()
            with _pytest.raises(AttributeError, match="max_position_embeddings"):
                wrapper.from_pretrained("/fake/path", config="caller_supplied")

    def test_class_attribute_forwarding(self, applied_patch):
        """Class-level attribute access (e.g. AutoTokenizer.register) must
        forward to the upstream class so mlx-lm's NewlineTokenizer
        registration still works."""
        import mlx_lm.tokenizer_utils as tu
        from transformers import AutoTokenizer as upstream_at

        # register is an upstream classmethod — wrapped class must expose it.
        assert tu.AutoTokenizer.register is upstream_at.register


class TestModelClassResolution:
    """mlx_lm.utils._get_classes resolves deepseek_v4 to our injected classes."""

    def test_get_classes_returns_injected_module(self, applied_patch):
        from mlx_lm.utils import _get_classes

        model_class, args_class = _get_classes({"model_type": "deepseek_v4"})
        assert model_class.__module__ == "mlx_lm.models.deepseek_v4"
        assert args_class.__module__ == "mlx_lm.models.deepseek_v4"
        assert model_class.__name__ == "Model"
        assert args_class.__name__ == "ModelArgs"


class TestCacheHandlerRegistration:
    """omlx CacheTypeRegistry resolves the new cache types to their handlers."""

    def test_pooling_cache_resolves_to_handler(self, applied_patch):
        from omlx.cache.type_registry import CacheTypeRegistry

        handler = CacheTypeRegistry.get_handler_by_class_name("PoolingCache")
        assert type(handler).__name__ == "PoolingCacheHandler"

    def test_batch_pooling_cache_resolves_to_handler(self, applied_patch):
        from omlx.cache.type_registry import CacheTypeRegistry

        handler = CacheTypeRegistry.get_handler_by_class_name("BatchPoolingCache")
        assert type(handler).__name__ == "BatchPoolingCacheHandler"

    def test_pooling_cache_not_block_sliceable(self, applied_patch):
        from omlx.cache.type_registry import CacheTypeRegistry

        handler = CacheTypeRegistry.get_handler_by_class_name("PoolingCache")
        assert handler.supports_block_slicing is False

    def test_batch_pooling_cache_not_block_sliceable(self, applied_patch):
        from omlx.cache.type_registry import CacheTypeRegistry

        handler = CacheTypeRegistry.get_handler_by_class_name("BatchPoolingCache")
        assert handler.supports_block_slicing is False

    def test_detect_cache_type_pooling(self, applied_patch):
        from mlx_lm.models.cache import PoolingCache

        from omlx.cache.type_handlers import CacheType
        from omlx.cache.type_registry import CacheTypeRegistry

        cache = PoolingCache(ratio=4)
        assert CacheTypeRegistry.detect_cache_type(cache) == CacheType.POOLING_CACHE


class TestPoolingCacheStateRoundTrip:
    """Handler extract_state → reconstruct_cache must preserve the pool tensor."""

    def test_round_trip_with_pooled_tensor(self, applied_patch):
        import mlx.core as mx
        from mlx_lm.models.cache import PoolingCache

        from omlx.cache.type_registry import CacheTypeRegistry

        # Build a PoolingCache with a known pool.
        ratio = 4
        cache = PoolingCache(ratio=ratio)
        # Simulate update_and_fetch having stuffed the pool with 8
        # compressed tokens of dim 32.
        pooled = mx.arange(1 * 8 * 32, dtype=mx.float32).reshape(1, 8, 32)
        cache.pooled = pooled

        handler = CacheTypeRegistry.get_handler_by_class_name("PoolingCache")
        state = handler.extract_state(cache)
        assert state["pooled"] is not None
        assert state["pooled"].shape == (1, 8, 32)

        restored = handler.reconstruct_cache(state, meta_state=ratio)
        assert restored is not None
        assert restored.ratio == ratio
        assert restored.pooled.shape == (1, 8, 32)
        # Verify content matches.
        diff = mx.max(mx.abs(restored.pooled - pooled)).item()
        assert diff == 0.0

    def test_round_trip_empty_cache(self, applied_patch):
        from mlx_lm.models.cache import PoolingCache

        from omlx.cache.type_registry import CacheTypeRegistry

        cache = PoolingCache(ratio=8)
        handler = CacheTypeRegistry.get_handler_by_class_name("PoolingCache")
        state = handler.extract_state(cache)
        assert state["pooled"] is None
        assert state["buf_kv"] is None

        restored = handler.reconstruct_cache(state, meta_state=8)
        assert restored is not None
        assert restored.empty()
        assert restored.ratio == 8

    def test_seq_len_from_state(self, applied_patch):
        import mlx.core as mx
        from mlx_lm.models.cache import PoolingCache

        from omlx.cache.type_registry import CacheTypeRegistry

        cache = PoolingCache(ratio=4)
        cache.pooled = mx.zeros((1, 12, 16), dtype=mx.float32)
        handler = CacheTypeRegistry.get_handler_by_class_name("PoolingCache")
        state = handler.extract_state(cache)
        assert handler.get_seq_len(state) == 12


class TestPreLoadDispatch:
    """maybe_apply_pre_load_patches gates correctly on config.json model_type."""

    def test_no_dispatch_for_other_model_type(self, tmp_path):
        # Create a fake model dir with a non-deepseek config.
        config_path = tmp_path / "config.json"
        config_path.write_text('{"model_type": "llama"}')

        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        # Should be a no-op (no exception). We can't easily assert that
        # apply_deepseek_v4_patch was NOT called because earlier tests
        # may have applied it already. Just verify no crash.
        maybe_apply_pre_load_patches(str(tmp_path))

    def test_no_dispatch_for_missing_config(self, tmp_path):
        # No config.json present.
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        maybe_apply_pre_load_patches(str(tmp_path))

    def test_dispatch_for_deepseek_v4(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text('{"model_type": "deepseek_v4"}')

        from omlx.patches.deepseek_v4 import is_applied
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        maybe_apply_pre_load_patches(str(tmp_path))
        # Patch must be applied after this dispatch (or already applied).
        assert is_applied() is True
