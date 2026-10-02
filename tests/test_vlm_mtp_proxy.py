# SPDX-License-Identifier: Apache-2.0
"""Tests for VLM MTP proxy classes.

Validates that _VLMAdapterMTPProxy and _MTPResetBindingProxy correctly
control attribute visibility so that:
- _mtp_rounds / _mtp_rounds_batch see a target that does NOT expose
  ``language_model`` (forcing ``lm = model`` and routing verify through
  the adapter);
- drafter.reset() can temporarily expose ``language_model`` for bind().
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.speculative.vlm_mtp import (
    _MTPResetBindingProxy,
    _VLMAdapterMTPProxy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeLanguageModel:
    """Mimics the patched LanguageModel with rollback_speculative_cache."""

    def __init__(self):
        self.rollback_called = False
        self.model = object()

    def __call__(self, *args, **kwargs):
        from mlx_vlm.models.base import LanguageModelOutput

        return LanguageModelOutput(
            logits=mx.zeros((1, 1, 4)),
            hidden_states=[mx.zeros((1, 1, 8))],
            gdn_states=[],
            shared_kv_states={},
        )

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        self.rollback_called = True
        return accepted

    def speculative_logits_from_hidden(self, hidden):
        return hidden


class FakeVLMAdapter:
    """Mimics VLMModelAdapter with _language_model and patched methods."""

    def __init__(self, expose_rollback: bool = True, uses_mrope: bool = False):
        self._language_model = FakeLanguageModel()
        self._uses_mrope = uses_mrope
        self.forward_called = False
        if expose_rollback:
            # Mimic _patch_vlm_model_adapter() which delegates to _language_model.
            self.rollback_speculative_cache = (
                self._language_model.rollback_speculative_cache
            )

    def __call__(self, *args, **kwargs):
        self.forward_called = True
        return self._language_model(*args, **kwargs)

    def set_batch_rope_deltas(self, deltas):
        pass


class FakeDrafter(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_target = None
        self.reset_called = False

    def reset(self, target_model, *args, **kwargs):
        self.reset_target = target_model
        self.reset_called = True
        # bind() accesses target_model.language_model.model.embed_tokens
        _ = target_model.language_model
        return []


# ---------------------------------------------------------------------------
# _VLMAdapterMTPProxy tests
# ---------------------------------------------------------------------------


class TestVLMAdapterMTPProxy:
    def test_language_model_not_exposed_by_default(self):
        adapter = FakeVLMAdapter()
        proxy = _VLMAdapterMTPProxy(adapter, adapter._language_model)

        assert not hasattr(proxy, "language_model")
        with pytest.raises(AttributeError):
            _ = proxy.language_model

    def test_language_model_exposed_when_flag_set(self):
        adapter = FakeVLMAdapter()
        proxy = _VLMAdapterMTPProxy(adapter, adapter._language_model)
        proxy._expose_language_model = True

        assert hasattr(proxy, "language_model")
        assert proxy.language_model is adapter._language_model

    def test_call_delegates_to_adapter(self):
        adapter = FakeVLMAdapter()
        proxy = _VLMAdapterMTPProxy(adapter, adapter._language_model)

        proxy(mx.array([1]), cache=[])
        assert adapter.forward_called

    def test_non_language_model_attrs_delegate_to_adapter(self):
        adapter = FakeVLMAdapter()
        proxy = _VLMAdapterMTPProxy(adapter, adapter._language_model)

        assert proxy._language_model is adapter._language_model

    def test_rollback_speculative_cache_delegates(self):
        adapter = FakeVLMAdapter()
        proxy = _VLMAdapterMTPProxy(adapter, adapter._language_model)

        proxy.rollback_speculative_cache([], [], 0, 4)
        assert adapter._language_model.rollback_called

    def test_rollback_falls_back_to_language_model_when_adapter_lacks_passthrough(self):
        adapter = FakeVLMAdapter(expose_rollback=False)
        proxy = _VLMAdapterMTPProxy(adapter, adapter._language_model)

        proxy.rollback_speculative_cache([], [], 0, 4)
        assert adapter._language_model.rollback_called

    def test_mrope_proxy_hides_fast_path_attrs_but_keeps_rollback(self):
        adapter = FakeVLMAdapter(expose_rollback=False, uses_mrope=True)
        proxy = _VLMAdapterMTPProxy(adapter, adapter._language_model)

        assert hasattr(proxy, "rollback_speculative_cache")
        assert not hasattr(proxy, "model")
        assert not hasattr(proxy, "speculative_logits_from_hidden")

    def test_mtp_rounds_sees_no_language_model(self):
        """Simulates the hasattr check in _mtp_rounds / _mtp_rounds_batch."""
        adapter = FakeVLMAdapter()
        proxy = _VLMAdapterMTPProxy(adapter, adapter._language_model)

        # _mtp_rounds line 547: lm = model.language_model if hasattr(...) else model
        lm = proxy.language_model if hasattr(proxy, "language_model") else proxy
        assert lm is proxy


# ---------------------------------------------------------------------------
# _MTPResetBindingProxy tests
# ---------------------------------------------------------------------------


class TestMTPResetBindingProxy:
    def test_reset_temporarily_exposes_language_model(self):
        adapter = FakeVLMAdapter()
        proxy = _VLMAdapterMTPProxy(adapter, adapter._language_model)
        drafter = FakeDrafter()
        reset_proxy = _MTPResetBindingProxy(drafter, proxy)

        # Before reset: language_model not exposed
        assert not hasattr(proxy, "language_model")

        reset_proxy.reset(proxy)

        # Drafter's bind() saw language_model during reset
        assert drafter.reset_called
        assert drafter.reset_target is proxy

        # After reset: language_model hidden again
        assert not hasattr(proxy, "language_model")

    def test_reset_non_proxy_target_passes_through(self):
        adapter = FakeVLMAdapter()
        proxy = _VLMAdapterMTPProxy(adapter, adapter._language_model)
        drafter = FakeDrafter()
        reset_proxy = _MTPResetBindingProxy(drafter, proxy)

        other_target = SimpleNamespace(language_model=object())
        reset_proxy.reset(other_target)

        assert drafter.reset_called
        assert drafter.reset_target is other_target

    def test_other_attrs_delegate_to_drafter(self):
        adapter = FakeVLMAdapter()
        proxy = _VLMAdapterMTPProxy(adapter, adapter._language_model)
        drafter = FakeDrafter()
        reset_proxy = _MTPResetBindingProxy(drafter, proxy)

        assert reset_proxy.reset_called is False  # getattr on drafter

    def test_reset_exception_restores_hidden_state(self):
        """language_model must be hidden again even if reset() raises."""
        adapter = FakeVLMAdapter()
        proxy = _VLMAdapterMTPProxy(adapter, adapter._language_model)

        class FailingDrafter:
            def reset(self, target_model, *args, **kwargs):
                _ = target_model.language_model  # needs it exposed
                raise RuntimeError("bind failed")

        reset_proxy = _MTPResetBindingProxy(FailingDrafter(), proxy)

        with pytest.raises(RuntimeError, match="bind failed"):
            reset_proxy.reset(proxy)

        # Must be hidden even after exception
        assert not hasattr(proxy, "language_model")
