# SPDX-License-Identifier: Apache-2.0
"""Tests for the deprecated dataclass constructor-keyword alias shim."""

import dataclasses
from dataclasses import dataclass

import pytest

from omlx.config import PagedSSDCacheConfig
from omlx.scheduler import SchedulerConfig
from omlx.settings import CacheSettings
from omlx.utils.legacy_kwargs import deprecated_init_kwargs


def test_decorator_maps_legacy_kwarg_to_canonical():
    @deprecated_init_kwargs(legacy_name="canonical_name")
    @dataclass
    class Cfg:
        canonical_name: str = "fp32"

    assert Cfg(legacy_name="int8").canonical_name == "int8"
    # canonical still works
    assert Cfg(canonical_name="bf16").canonical_name == "bf16"
    # explicit legacy overrides canonical (replace re-injects canonical)
    assert Cfg(canonical_name="bf16", legacy_name="int8").canonical_name == ("int8")
    # legacy=None means "not provided"
    assert Cfg(legacy_name=None).canonical_name == "fp32"
    # default untouched
    assert Cfg().canonical_name == "fp32"


def test_decorator_supports_dataclasses_replace():
    @deprecated_init_kwargs(legacy_name="canonical_name")
    @dataclass
    class Cfg:
        canonical_name: str = "fp32"

    cfg = Cfg(canonical_name="rht_int16")
    replaced = dataclasses.replace(cfg, legacy_name="fp32")
    assert replaced.canonical_name == "fp32"


def test_decorator_rejects_unknown_kwargs():
    @deprecated_init_kwargs(legacy_name="canonical_name")
    @dataclass
    class Cfg:
        canonical_name: str = "fp32"

    with pytest.raises(TypeError):
        Cfg(totally_unknown="x")


@pytest.mark.parametrize("cls", [PagedSSDCacheConfig, CacheSettings, SchedulerConfig])
class TestGdnDtypeAliasOnConfigClasses:
    """The gdn_sidecar_state_dtype -> gdn_snapshot_state_dtype rename kept
    the old name working on every surface it worked on before the rename."""

    def test_default_is_fp32(self, cls):
        assert cls().gdn_snapshot_state_dtype == "fp32"
        assert cls().gdn_sidecar_state_dtype == "fp32"

    def test_canonical_kwarg(self, cls):
        cfg = cls(gdn_snapshot_state_dtype="rht_int16")
        assert cfg.gdn_snapshot_state_dtype == "rht_int16"
        assert cfg.gdn_sidecar_state_dtype == "rht_int16"

    def test_legacy_constructor_kwarg(self, cls):
        cfg = cls(gdn_sidecar_state_dtype="rht_int16")
        assert cfg.gdn_snapshot_state_dtype == "rht_int16"
        assert cfg.gdn_sidecar_state_dtype == "rht_int16"

    def test_attribute_setter_alias(self, cls):
        cfg = cls()
        cfg.gdn_sidecar_state_dtype = "bf16"
        assert cfg.gdn_snapshot_state_dtype == "bf16"

    def test_replace_with_legacy_kwarg(self, cls):
        cfg = cls(gdn_snapshot_state_dtype="rht_int16")
        replaced = dataclasses.replace(cfg, gdn_sidecar_state_dtype="fp32")
        assert replaced.gdn_snapshot_state_dtype == "fp32"
