# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.dflash_lifecycle (issue #1388)."""

from __future__ import annotations

import pytest


@pytest.fixture
def _clear_backup_state():
    """Reset the backup table before / after each test."""
    from omlx.patches import dflash_lifecycle as life
    life._DFLASH_BACKUP.clear()
    yield
    life._DFLASH_BACKUP.clear()


def _make_fake_dflash_module():
    """Build an object that quacks like dflash's target_qwen_gdn module
    for the purpose of testing the wrap helper independent of dflash-mlx.
    """
    from types import SimpleNamespace

    captures: list = []

    def fake_installer(linear_attn):
        cls = type(linear_attn)
        if getattr(cls, "_dflash_speculative_call_installed", False):
            return
        # Mimic dflash: overwrite cls.__call__ and set its idempotency flag.
        def fake_speculative_call(self, inputs, mask=None, cache=None):
            return inputs
        cls.__call__ = fake_speculative_call
        cls._dflash_speculative_call_installed = True
        captures.append(linear_attn)

    def fake_gqa_installer(attn):
        cls = type(attn)
        if getattr(cls, "_dflash_full_attention_gqa_installed", False):
            return
        # Mimic dflash 0.1.7's full-attention GQA hook: overwrite __call__
        # and set its idempotency flag. The real hook's first line does
        # int(cache.offset), which is what crashes on batched offsets.
        def fake_attention_call(self, x, mask=None, cache=None):
            return x
        cls.__call__ = fake_attention_call
        cls._dflash_full_attention_gqa_installed = True
        captures.append(attn)

    mod = SimpleNamespace(
        _install_speculative_linear_cache_hook=fake_installer,
        _install_full_attention_gqa_hook=fake_gqa_installer,
        _captures=captures,
    )
    return mod


class TestWrapInstaller:
    def test_wrap_records_pre_dflash_call(self, _clear_backup_state):
        """Wrapped installer must snapshot cls.__call__ before dflash overwrites."""
        from omlx.patches.dflash_lifecycle import _wrap_installer, _DFLASH_BACKUP

        mod = _make_fake_dflash_module()

        class FakeLinearAttn:
            def __call__(self, x, mask=None, cache=None):
                return "stock-result"

        installed = _wrap_installer(
            mod,
            "_install_speculative_linear_cache_hook",
            "_dflash_speculative_call_installed",
        )
        assert installed is True

        instance = FakeLinearAttn()
        original_call = FakeLinearAttn.__call__
        mod._install_speculative_linear_cache_hook(instance)

        # cls.__call__ is now the dflash-fake one (rejects n_confirmed-style kwargs).
        assert FakeLinearAttn.__call__ is not original_call
        # Backup table must hold a reference to the original stock __call__.
        assert FakeLinearAttn in _DFLASH_BACKUP
        assert _DFLASH_BACKUP[FakeLinearAttn]["call"] is original_call

    def test_wrap_is_idempotent(self, _clear_backup_state):
        from omlx.patches.dflash_lifecycle import _wrap_installer

        mod = _make_fake_dflash_module()
        installed_once = _wrap_installer(
            mod,
            "_install_speculative_linear_cache_hook",
            "_dflash_speculative_call_installed",
        )
        first_wrapped = mod._install_speculative_linear_cache_hook
        installed_twice = _wrap_installer(
            mod,
            "_install_speculative_linear_cache_hook",
            "_dflash_speculative_call_installed",
        )
        assert installed_once and installed_twice
        # Second call must NOT re-wrap (would double-record on subsequent install).
        assert mod._install_speculative_linear_cache_hook is first_wrapped


class TestRestore:
    def test_restore_reverts_call_and_clears_flag(self, _clear_backup_state):
        """After restore: cls.__call__ back to original, dflash flag gone."""
        from omlx.patches.dflash_lifecycle import (
            _wrap_installer,
            restore_dflash_class_patches,
        )

        mod = _make_fake_dflash_module()

        class FakeLinearAttn:
            def __call__(self, x, mask=None, cache=None):
                return "stock-result"

        _wrap_installer(
            mod,
            "_install_speculative_linear_cache_hook",
            "_dflash_speculative_call_installed",
        )
        original_call = FakeLinearAttn.__call__
        instance = FakeLinearAttn()
        mod._install_speculative_linear_cache_hook(instance)
        assert FakeLinearAttn.__call__ is not original_call
        assert FakeLinearAttn._dflash_speculative_call_installed is True

        restore_dflash_class_patches()

        assert FakeLinearAttn.__call__ is original_call
        assert "_dflash_speculative_call_installed" not in FakeLinearAttn.__dict__

    def test_restore_empty_table_is_noop(self, _clear_backup_state):
        """Restore with no backup recorded must not raise."""
        from omlx.patches.dflash_lifecycle import restore_dflash_class_patches
        restore_dflash_class_patches()  # no-op


class TestRoundTrip:
    def test_dflash_mtp_dflash_round_trip(self, _clear_backup_state):
        """Sequence: stock → dflash install → restore → simulate mtp patch
        replacing __call__ → dflash install again. Each transition must
        leave the class in the expected state with the right idempotency
        flag on / off.
        """
        from omlx.patches.dflash_lifecycle import (
            _wrap_installer,
            restore_dflash_class_patches,
        )

        mod = _make_fake_dflash_module()

        class FakeLinearAttn:
            def __call__(self, x, mask=None, cache=None):
                return "stock"

        stock_call = FakeLinearAttn.__call__
        _wrap_installer(
            mod,
            "_install_speculative_linear_cache_hook",
            "_dflash_speculative_call_installed",
        )

        # Round 1: dflash arms.
        mod._install_speculative_linear_cache_hook(FakeLinearAttn())
        first_dflash_call = FakeLinearAttn.__call__
        assert first_dflash_call is not stock_call
        assert FakeLinearAttn._dflash_speculative_call_installed is True

        # dflash engine stops → restore.
        restore_dflash_class_patches()
        assert FakeLinearAttn.__call__ is stock_call
        assert "_dflash_speculative_call_installed" not in FakeLinearAttn.__dict__

        # Simulate a Native MTP patch replacing __call__.
        def mtp_call(self, x, mask=None, cache=None, n_confirmed=0):
            return ("mtp", n_confirmed)
        FakeLinearAttn.__call__ = mtp_call

        # Round 2: dflash arms again. The wrap should capture mtp_call as
        # the pre-dflash backup so a later restore drops back to mtp_call.
        mod._install_speculative_linear_cache_hook(FakeLinearAttn())
        assert FakeLinearAttn._dflash_speculative_call_installed is True
        restore_dflash_class_patches()
        assert FakeLinearAttn.__call__ is mtp_call


class TestQwenGqaHook:
    """The Qwen full-attention GQA hook (dflash 0.1.7) must round-trip too.

    Regression for issue #1510: dflash renamed the Qwen full-attention
    installer to ``_install_full_attention_gqa_hook``; the lifecycle wrap
    must track it so a DFlash -> MTP transition restores the attention
    class instead of leaving dflash's offset-unsafe hook on it.
    """

    def test_gqa_hook_round_trips(self, _clear_backup_state):
        from omlx.patches.dflash_lifecycle import (
            _DFLASH_BACKUP,
            _wrap_installer,
            restore_dflash_class_patches,
        )

        mod = _make_fake_dflash_module()

        class FakeAttention:
            def __call__(self, x, mask=None, cache=None):
                return "stock-attn"

        installed = _wrap_installer(
            mod,
            "_install_full_attention_gqa_hook",
            "_dflash_full_attention_gqa_installed",
        )
        assert installed is True

        stock_call = FakeAttention.__call__
        mod._install_full_attention_gqa_hook(FakeAttention())
        # dflash hook is now on the class.
        assert FakeAttention.__call__ is not stock_call
        assert FakeAttention._dflash_full_attention_gqa_installed is True
        assert FakeAttention in _DFLASH_BACKUP

        # DFlash engine stops -> restore must revert the class and drop flag.
        restore_dflash_class_patches()
        assert FakeAttention.__call__ is stock_call
        assert "_dflash_full_attention_gqa_installed" not in FakeAttention.__dict__


class TestBatchCacheGuard:
    """Regression for issue #2252.

    While a DFlash engine is armed, its class hooks are visible to every
    engine sharing the patched Python class. A concurrent BatchedEngine
    decode hands the hook a BatchKVCache whose ``offset`` is a per-row
    ``mx.array``, which the raw dflash hook converts with ``int()`` and
    crashes. The lifecycle wrap must guard the installed hook so such
    caches fall through to the pre-dflash ``__call__``.
    """

    @staticmethod
    def _arm(mod, attn_cls):
        from omlx.patches.dflash_lifecycle import _wrap_installer

        _wrap_installer(
            mod,
            "_install_full_attention_gqa_hook",
            "_dflash_full_attention_gqa_installed",
        )
        mod._install_full_attention_gqa_hook(attn_cls())

    def test_batch_cache_falls_through_to_pre_dflash_call(
        self, _clear_backup_state
    ):
        import mlx.core as mx

        mod = _make_fake_dflash_module()

        class FakeAttention:
            def __call__(self, x, mask=None, cache=None):
                return "stock-attn"

        self._arm(mod, FakeAttention)
        assert getattr(
            FakeAttention.__call__, "_omlx_dflash_batch_guard", False
        )

        class FakeBatchCache:
            offset = mx.array([3, 7])

        # Multi-row batch cache must bypass the dflash hook entirely.
        result = FakeAttention()("x", cache=FakeBatchCache())
        assert result == "stock-attn"

    def test_scalar_offset_cache_still_routes_to_dflash_hook(
        self, _clear_backup_state
    ):
        mod = _make_fake_dflash_module()

        class FakeAttention:
            def __call__(self, x, mask=None, cache=None):
                return "stock-attn"

        self._arm(mod, FakeAttention)

        class FakeKVCache:
            offset = 42

        # dflash's own caches carry int offsets; the hook keeps running.
        assert FakeAttention()("x", cache=FakeKVCache()) == "x"
        # No cache at all also stays on the dflash hook.
        assert FakeAttention()("x") == "x"

    def test_guard_is_idempotent_across_layer_installs(
        self, _clear_backup_state
    ):
        mod = _make_fake_dflash_module()

        class FakeAttention:
            def __call__(self, x, mask=None, cache=None):
                return "stock-attn"

        self._arm(mod, FakeAttention)
        guarded = FakeAttention.__call__
        # Second layer of the same class re-runs the installer; the guard
        # must not wrap itself again.
        mod._install_full_attention_gqa_hook(FakeAttention())
        assert FakeAttention.__call__ is guarded

    def test_restore_drops_guard_and_hook(self, _clear_backup_state):
        from omlx.patches.dflash_lifecycle import restore_dflash_class_patches

        mod = _make_fake_dflash_module()

        class FakeAttention:
            def __call__(self, x, mask=None, cache=None):
                return "stock-attn"

        stock_call = FakeAttention.__call__
        self._arm(mod, FakeAttention)
        assert FakeAttention.__call__ is not stock_call

        restore_dflash_class_patches()
        assert FakeAttention.__call__ is stock_call
        assert "_dflash_full_attention_gqa_installed" not in FakeAttention.__dict__


class TestRealDflashIntegration:
    """Integration tests against the real dflash-mlx module if installed."""

    def test_install_wrap_against_real_dflash(self, _clear_backup_state):
        from omlx.patches.dflash_lifecycle import install_dflash_lifecycle_wrap
        try:
            from dflash_mlx.engine import target_qwen_gdn
        except ImportError:
            pytest.skip("dflash-mlx not installed in this environment")

        # Must report at least one wrap installed.
        assert install_dflash_lifecycle_wrap() is True
        # Idempotent.
        assert install_dflash_lifecycle_wrap() is True
        # The Qwen full-attention GQA hook (dflash 0.1.7) must be wrapped so
        # its class patch is restorable on DFlash teardown (issue #1510).
        if hasattr(target_qwen_gdn, "_install_full_attention_gqa_hook"):
            assert (
                getattr(
                    target_qwen_gdn,
                    "_omlx_wrapped__install_full_attention_gqa_hook",
                    False,
                )
                is True
            )
