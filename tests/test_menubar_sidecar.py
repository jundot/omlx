# SPDX-License-Identifier: Apache-2.0
"""Tests for the menubar sidecar's pure logic (no AppKit required)."""

from __future__ import annotations

from omlx.menubar_sidecar import (
    compute_live_state,
    fmt_duration,
    fmt_tokens,
    should_autostart,
)


class TestFormatting:
    def test_fmt_tokens(self):
        assert fmt_tokens(999) == "999"
        assert fmt_tokens(1000) == "1.0k"
        assert fmt_tokens(1500) == "1.5k"
        assert fmt_tokens(2_500_000) == "2.5M"
        assert fmt_tokens(None) == "?"

    def test_fmt_duration(self):
        assert fmt_duration(5) == "5s"
        assert fmt_duration(60) == "1m"
        assert fmt_duration(125) == "2m5s"
        assert fmt_duration(None) == ""


def _activity(models=None, waiting=0):
    return {
        "active_models": {"models": models or [], "total_waiting_requests": waiting}
    }


def _prefill(processed=450, total=1000, speed=800.0, eta=0.7):
    return {"processed": processed, "total": total, "speed": speed, "eta": eta}


def _gen(tps=42.1, toks=210, elapsed=5.0):
    return {
        "tokens_per_second": tps,
        "generated_tokens": toks,
        "elapsed_seconds": elapsed,
    }


class TestComputeLiveState:
    def test_offline(self):
        title, kind, _ = compute_live_state(None, online=False)
        assert (title, kind) == ("offline", "offline")
        # online flag False beats a populated snapshot
        title, kind, _ = compute_live_state(_activity(), online=False)
        assert kind == "offline"

    def test_idle(self):
        title, kind, detail = compute_live_state(_activity(), online=True)
        assert (title, kind) == ("idle", "idle")
        assert detail == ["no models"]

    def test_idle_lists_loaded_models(self):
        act = _activity([{"id": "M1", "is_loading": False}])
        title, kind, detail = compute_live_state(act, online=True)
        assert kind == "idle"
        assert detail == ["loaded: M1"]

    def test_waiting(self):
        title, kind, _ = compute_live_state(_activity(waiting=3), online=True)
        assert (title, kind) == ("WAIT 3", "waiting")

    def test_prefill_takes_priority_over_decode(self):
        act = _activity(
            [{"id": "M1", "prefilling": [_prefill()], "generating": [_gen()]}]
        )
        title, kind, detail = compute_live_state(act, online=True)
        assert kind == "prefill"
        assert title == "PP 45%"
        assert detail[0] == "M1"
        assert "PP 45%" in detail[1] and "t/s" in detail[1]

    def test_prefill_percentage_zero_total(self):
        act = _activity([{"id": "M", "prefilling": [_prefill(total=0)]}])
        title, kind, _ = compute_live_state(act, online=True)
        assert (title, kind) == ("PP 0%", "prefill")

    def test_prefill_extra_suffix(self):
        act = _activity(
            [
                {"id": "M1", "prefilling": [_prefill()]},
                {"id": "M2", "prefilling": [_prefill(), _prefill()]},
            ]
        )
        title, kind, _ = compute_live_state(act, online=True)
        assert title == "PP 45% +2"

    def test_decode(self):
        act = _activity(
            [{"id": "M1", "generating": [_gen(tps=52.1, toks=285, elapsed=7)]}]
        )
        title, kind, detail = compute_live_state(act, online=True)
        assert (title, kind) == ("GEN 52.1 t/s", "decode")
        assert detail[1] == "GEN 52.1 t/s · 285 tok · 7s"

    def test_decode_extra_suffix_counts_other_models(self):
        act = _activity(
            [
                {"id": "M1", "generating": [_gen()]},
                {"id": "M2", "generating": [_gen(), _gen()]},
            ]
        )
        title, kind, _ = compute_live_state(act, online=True)
        assert title == "GEN 42.1 t/s +2"


class TestShouldAutostart:
    KW = dict(platform="darwin", pyobjc=True)

    def test_default_enabled_on_macos(self):
        assert should_autostart(env={}, **self.KW) == (True, None)

    def test_non_macos_disabled(self):
        assert should_autostart(env={}, platform="linux", pyobjc=True) == (
            False,
            None,
        )

    def test_supervised_by_native_app_disabled(self):
        env = {"OMLX_SUPERVISED": "menubar"}
        assert should_autostart(env=env, **self.KW) == (False, None)
        # even an explicit --menubar does not double up with the app icon
        assert should_autostart(cli_flag=True, env=env, **self.KW) == (False, None)

    def test_cli_no_overrides_settings(self):
        assert should_autostart(
            cli_flag=False, settings_enabled=True, env={}, **self.KW
        ) == (
            False,
            None,
        )

    def test_settings_off_default_cli(self):
        assert should_autostart(
            cli_flag=None, settings_enabled=False, env={}, **self.KW
        ) == (
            False,
            None,
        )

    def test_cli_on_overrides_settings_off(self):
        assert should_autostart(
            cli_flag=True, settings_enabled=False, env={}, **self.KW
        ) == (
            True,
            None,
        )

    def test_missing_pyobjc_disabled_with_hint(self):
        enabled, hint = should_autostart(
            cli_flag=None, env={}, platform="darwin", pyobjc=False
        )
        assert enabled is False
        assert hint and "menubar" in hint.lower()

    def test_missing_pyobjc_hint_only_when_wanted(self):
        # explicit opt-out must not nag about pyobjc
        assert should_autostart(
            cli_flag=False, env={}, platform="darwin", pyobjc=False
        ) == (
            False,
            None,
        )


def test_module_importable_headless():
    """Importing the sidecar must not pull in AppKit (CI / ssh context)."""
    import subprocess
    import sys

    code = (
        "import sys, omlx.menubar_sidecar as m;"
        "assert 'AppKit' not in sys.modules;"
        "assert callable(m.pyobjc_available)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30)
    assert r.returncode == 0, r.stderr.decode()
