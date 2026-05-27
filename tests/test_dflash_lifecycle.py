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
        # Mimic dflash: overwrite cls.__call__ and set its idempotency flag.
        def fake_speculative_call(self, inputs, mask=None, cache=None):
            return inputs
        cls.__call__ = fake_speculative_call
        cls._dflash_speculative_call_installed = True
        captures.append(linear_attn)

    mod = SimpleNamespace(
        _install_speculative_linear_cache_hook=fake_installer,
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


class TestRealDflashIntegration:
    """Integration tests against the real dflash-mlx module if installed."""

    def test_install_wrap_against_real_dflash(self, _clear_backup_state):
        from omlx.patches.dflash_lifecycle import install_dflash_lifecycle_wrap
        try:
            from dflash_mlx.engine import target_qwen_gdn  # noqa: F401
        except ImportError:
            pytest.skip("dflash-mlx not installed in this environment")

        # Must report at least one wrap installed.
        assert install_dflash_lifecycle_wrap() is True
        # Idempotent.
        assert install_dflash_lifecycle_wrap() is True
