# SPDX-License-Identifier: Apache-2.0
"""Tests for the cross-engine decode-activity registry."""

from omlx.decode_activity import DecodeActivityRegistry, get_decode_activity


class TestDecodeActivityRegistry:
    def test_others_decoding_excludes_self(self):
        reg = DecodeActivityRegistry()
        reg.publish("engine-a", 1)
        assert not reg.others_decoding("engine-a")
        assert reg.others_decoding("engine-b")

    def test_zero_count_removes_entry(self):
        reg = DecodeActivityRegistry()
        reg.publish("engine-a", 2)
        assert reg.others_decoding("engine-b")
        reg.publish("engine-a", 0)
        assert not reg.others_decoding("engine-b")

    def test_ttl_expires_stale_entries(self):
        reg = DecodeActivityRegistry()
        reg.publish("engine-a", 1)
        # A wedged engine that stopped publishing must not throttle others.
        reg._active["engine-a"] = (
            reg._active["engine-a"][0] - 10.0,
            1,
        )
        assert not reg.others_decoding("engine-b", ttl_s=2.5)

    def test_remove_and_clear(self):
        reg = DecodeActivityRegistry()
        reg.publish("engine-a", 1)
        reg.remove("engine-a")
        assert not reg.others_decoding("engine-b")
        reg.publish("engine-a", 1)
        reg.clear()
        assert not reg.others_decoding("engine-b")

    def test_singleton(self):
        assert get_decode_activity() is get_decode_activity()
