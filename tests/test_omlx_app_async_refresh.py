"""Fail-first tests for async menubar stats refresh behavior.

These tests import the native app module with PyObjC/AppKit mocked out so we
can tighten the polling contract without requiring a real macOS GUI runtime.
"""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture
def app_module():
    """Import omlx_app.app with GUI/runtime dependencies stubbed out."""
    packaging_dir = str(Path(__file__).parent.parent / "packaging")
    module_names = [
        "omlx",
        "objc",
        "requests",
        "AppKit",
        "Foundation",
        "omlx._version",
        "omlx_app",
        "omlx_app.app",
        "omlx_app.config",
        "omlx_app.server_manager",
    ]
    saved_modules = {name: sys.modules.get(name) for name in module_names}

    fake_objc = types.ModuleType("objc")
    fake_objc.super = super
    fake_objc.IBAction = lambda func: func

    fake_requests = types.ModuleType("requests")

    class FakeRequestException(Exception):
        pass

    fake_requests.Session = MagicMock()
    fake_requests.RequestException = FakeRequestException
    fake_requests.ConnectionError = FakeRequestException

    fake_appkit = types.ModuleType("AppKit")
    for name in [
        "NSAlert",
        "NSAlertFirstButtonReturn",
        "NSAlertSecondButtonReturn",
        "NSAlertThirdButtonReturn",
        "NSApp",
        "NSAppearanceNameDarkAqua",
        "NSApplication",
        "NSApplicationActivationPolicyAccessory",
        "NSApplicationActivationPolicyRegular",
        "NSAttributedString",
        "NSBundle",
        "NSColor",
        "NSFloatingWindowLevel",
        "NSFont",
        "NSFontAttributeName",
        "NSForegroundColorAttributeName",
        "NSImage",
        "NSLinkAttributeName",
        "NSMenu",
        "NSMenuItem",
        "NSMutableParagraphStyle",
        "NSParagraphStyleAttributeName",
        "NSRightTabStopType",
        "NSStatusBar",
        "NSTextField",
        "NSTextTab",
        "NSTextAlignmentCenter",
        "NSView",
        "NSWorkspace",
    ]:
        setattr(fake_appkit, name, MagicMock())
    fake_appkit.NSVariableStatusItemLength = -1

    fake_foundation = types.ModuleType("Foundation")
    fake_foundation.NSObject = type("NSObject", (), {})
    for name in ["NSData", "NSMutableAttributedString", "NSRunLoop", "NSTimer", "NSURL"]:
        setattr(fake_foundation, name, MagicMock())
    fake_foundation.NSDefaultRunLoopMode = "NSDefaultRunLoopMode"

    fake_omlx = types.ModuleType("omlx")
    # Mark as package so `import omlx._version` resolves against our stub.
    fake_omlx.__path__ = []  # type: ignore[attr-defined]

    fake_version = types.ModuleType("omlx._version")
    fake_version.__version__ = "0.0.0-test"
    fake_omlx._version = fake_version

    fake_config = types.ModuleType("omlx_app.config")

    class FakeServerConfig:
        @classmethod
        def load(cls):
            return types.SimpleNamespace()

    fake_config.ServerConfig = FakeServerConfig

    fake_server_manager = types.ModuleType("omlx_app.server_manager")

    class FakeServerStatus:
        RUNNING = "running"
        STARTING = "starting"
        STOPPING = "stopping"
        STOPPED = "stopped"
        ERROR = "error"
        UNRESPONSIVE = "unresponsive"

    class FakePortConflict:
        pass

    class FakeServerManager:
        def __init__(self, config):
            self.config = config
            self.status = FakeServerStatus.STOPPED

    fake_server_manager.PortConflict = FakePortConflict
    fake_server_manager.ServerManager = FakeServerManager
    fake_server_manager.ServerStatus = FakeServerStatus

    sys.path.insert(0, packaging_dir)
    try:
        for name in module_names:
            sys.modules.pop(name, None)

        sys.modules["objc"] = fake_objc
        sys.modules["requests"] = fake_requests
        sys.modules["AppKit"] = fake_appkit
        sys.modules["Foundation"] = fake_foundation
        sys.modules["omlx"] = fake_omlx
        sys.modules["omlx._version"] = fake_version
        sys.modules["omlx_app.config"] = fake_config
        sys.modules["omlx_app.server_manager"] = fake_server_manager

        importlib.invalidate_caches()
        yield importlib.import_module("omlx_app.app")
    finally:
        if packaging_dir in sys.path:
            sys.path.remove(packaging_dir)
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _make_delegate(
    app_module,
    *,
    status=None,
    last_health_status=None,
    last_stats_fetch=0.0,
    last_started_at=0.0,
    in_flight=False,
    refresh_token=0,
):
    if status is None:
        status = app_module.ServerStatus.RUNNING
    delegate = types.SimpleNamespace(
        config=types.SimpleNamespace(get_server_api_key=lambda: "test-key", port=11434),
        server_manager=types.SimpleNamespace(status=status, stop=Mock()),
        _last_health_status=last_health_status,
        _last_stats_fetch=last_stats_fetch,
        _last_stats_refresh_started_at=last_started_at,
        _stats_refresh_in_flight=in_flight,
        _stats_refresh_token=refresh_token,
        _start_stats_refresh=Mock(),
        _fetch_stats=Mock(),
        _build_menu=Mock(),
        _update_status_display=Mock(),
        _update_menubar_icon=Mock(),
        _cached_stats=None,
        _cached_alltime_stats=None,
        _admin_session=None,
        performSelectorOnMainThread_withObject_waitUntilDone_=Mock(),
    )
    delegate._invalidate_stats_refresh_generation = (
        app_module.OMLXAppDelegate._invalidate_stats_refresh_generation.__get__(
            delegate
        )
    )
    delegate._close_session = app_module.OMLXAppDelegate._close_session.__get__(delegate)
    delegate._close_admin_session = app_module.OMLXAppDelegate._close_admin_session.__get__(
        delegate
    )
    return delegate


def _make_response(status_code, payload=None):
    """Create a tiny fake HTTP response object for stats polling tests."""
    if payload is None:
        payload = {}
    return types.SimpleNamespace(status_code=status_code, json=lambda: payload)


class TestAsyncStatsRefreshContract:
    def test_launch_registers_status_callback(self, app_module):
        """The app should subscribe to server status transitions, not just sample them on timer ticks."""
        server_manager = Mock(
            status=app_module.ServerStatus.STOPPED,
            set_status_callback=Mock(),
        )
        delegate = types.SimpleNamespace(
            _icon_outline=None,
            _icon_filled=None,
            status_item=Mock(),
            menu=None,
            health_timer=None,
            welcome_controller=None,
            config=types.SimpleNamespace(
                is_first_run=False, start_server_on_launch=False
            ),
            server_manager=server_manager,
            _load_menubar_icon=Mock(return_value=None),
            _create_status_item=Mock(),
            _update_menubar_icon=Mock(),
            _build_menu=Mock(),
            _check_for_updates=Mock(),
        )
        delegate._on_server_status_changed = (
            app_module.OMLXAppDelegate._on_server_status_changed.__get__(delegate)
        )

        app_module.OMLXAppDelegate._doFinishLaunching(delegate)

        server_manager.set_status_callback.assert_called_once()

    def test_health_check_starts_background_refresh_instead_of_fetching_inline(
        self, app_module
    ):
        """The UI timer should queue stats refresh work instead of blocking on HTTP inline."""
        delegate = _make_delegate(
            app_module,
            last_stats_fetch=0.0,
            last_started_at=0.0,
            in_flight=False,
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(app_module.time, "time", lambda: 10.0)
            app_module.OMLXAppDelegate.healthCheck_(delegate, None)

        delegate._start_stats_refresh.assert_called_once_with()
        delegate._fetch_stats.assert_not_called()
        delegate._build_menu.assert_not_called()
        assert delegate._last_stats_fetch == 0.0

    def test_health_check_does_not_queue_overlapping_refreshes(self, app_module):
        """A second timer tick should not start another refresh while one is already in flight."""
        delegate = _make_delegate(
            app_module,
            last_stats_fetch=0.0,
            last_started_at=0.0,
            in_flight=True,
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(app_module.time, "time", lambda: 10.0)
            app_module.OMLXAppDelegate.healthCheck_(delegate, None)

        delegate._start_stats_refresh.assert_not_called()
        delegate._fetch_stats.assert_not_called()
        delegate._build_menu.assert_not_called()
        assert delegate._last_stats_fetch == 0.0

    def test_health_check_sets_in_flight_guard_before_queueing(self, app_module):
        """Queueing work should flip the single-flight guard immediately."""
        delegate = _make_delegate(
            app_module,
            last_stats_fetch=0.0,
            last_started_at=0.0,
            in_flight=False,
        )

        def _assert_guard_set():
            assert delegate._stats_refresh_in_flight is True

        delegate._start_stats_refresh.side_effect = _assert_guard_set

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(app_module.time, "time", lambda: 10.0)
            app_module.OMLXAppDelegate.healthCheck_(delegate, None)

        delegate._start_stats_refresh.assert_called_once_with()
        delegate._fetch_stats.assert_not_called()

    def test_health_check_does_not_queue_when_interval_not_elapsed(self, app_module):
        """No refresh should be queued while still inside the polling cooldown window."""
        delegate = _make_delegate(
            app_module,
            last_stats_fetch=8.0,
            last_started_at=8.0,
            in_flight=False,
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(app_module.time, "time", lambda: 10.0)
            app_module.OMLXAppDelegate.healthCheck_(delegate, None)

        delegate._start_stats_refresh.assert_not_called()
        delegate._fetch_stats.assert_not_called()
        delegate._build_menu.assert_not_called()
        assert delegate._last_stats_fetch == 8.0

    def test_health_check_uses_queue_start_time_for_cadence(self, app_module):
        """The next 5s tick should still queue a refresh after prior completion lag."""
        delegate = _make_delegate(
            app_module,
            last_stats_fetch=5.1,
            last_started_at=5.0,
            in_flight=False,
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(app_module.time, "time", lambda: 10.0)
            app_module.OMLXAppDelegate.healthCheck_(delegate, None)

        delegate._start_stats_refresh.assert_called_once_with()
        assert delegate._stats_refresh_in_flight is True
        assert delegate._last_stats_refresh_started_at == 10.0

    def test_stats_refresh_completion_clears_in_flight_and_repaints(self, app_module):
        """Completion path should release the single-flight guard and apply UI-visible updates."""
        delegate = _make_delegate(app_module, last_stats_fetch=0.0, in_flight=True)

        app_module.OMLXAppDelegate.statsRefreshFinishedOnMain_(
            delegate,
            {
                "token": 0,
                "finished_at": 42.0,
                "stats": None,
                "alltime_stats": None,
                "session": None,
            },
        )

        assert delegate._stats_refresh_in_flight is False
        assert delegate._last_stats_fetch == 42.0
        delegate._build_menu.assert_called_once_with()
        delegate._update_menubar_icon.assert_called_once_with()

    def test_unexpected_worker_failure_clears_in_flight_without_timestamp_bump(
        self, app_module
    ):
        """Unexpected worker exceptions should release single-flight without consuming cooldown."""
        delegate = _make_delegate(app_module, last_stats_fetch=17.0, in_flight=True)

        app_module.OMLXAppDelegate.statsRefreshFailedOnMain_(delegate, "worker crashed")

        assert delegate._stats_refresh_in_flight is False
        assert delegate._last_stats_fetch == 17.0
        delegate._update_menubar_icon.assert_called_once_with()

    def test_request_exception_uses_finished_path_and_consumes_cooldown(
        self, app_module
    ):
        """Request failures still consume the polling interval, matching upstream behavior."""
        delegate = _make_delegate(app_module, last_stats_fetch=17.0, in_flight=True)

        failing_session = Mock()
        failing_session.get.side_effect = app_module.requests.RequestException(
            "network down"
        )

        class InlineThread:
            def __init__(self, *, target, name, daemon):
                self._target = target

            def start(self):
                self._target()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                app_module.requests, "Session", Mock(return_value=failing_session)
            )
            mp.setattr(app_module.threading, "Thread", InlineThread)
            mp.setattr(app_module.time, "time", lambda: 42.0)
            app_module.OMLXAppDelegate._start_stats_refresh(delegate)

        selector, payload, wait = (
            delegate.performSelectorOnMainThread_withObject_waitUntilDone_.call_args[0]
        )
        assert selector == "statsRefreshFinishedOnMain:"
        assert payload["stats"] is None
        assert payload["alltime_stats"] is None
        assert payload["session"] is None
        assert wait is False

        app_module.OMLXAppDelegate.statsRefreshFinishedOnMain_(delegate, payload)

        assert delegate._stats_refresh_in_flight is False
        assert delegate._last_stats_fetch == 42.0
        assert delegate._cached_stats is None
        assert delegate._cached_alltime_stats is None
        assert delegate._admin_session is None

    def test_stale_success_completion_does_not_clear_current_in_flight(
        self, app_module
    ):
        """A stale token callback must not drop the guard for a newer active refresh."""
        delegate = _make_delegate(
            app_module,
            in_flight=True,
            refresh_token=2,
            last_stats_fetch=30.0,
        )
        session = Mock()

        app_module.OMLXAppDelegate.statsRefreshFinishedOnMain_(
            delegate,
            {
                "token": 1,
                "finished_at": 31.0,
                "stats": {"stale": True},
                "alltime_stats": {"stale": True},
                "session": session,
            },
        )

        session.close.assert_called_once_with()
        assert delegate._stats_refresh_in_flight is True
        assert delegate._last_stats_fetch == 30.0
        assert delegate._cached_stats is None
        assert delegate._cached_alltime_stats is None
        delegate._build_menu.assert_not_called()

    def test_stale_failure_completion_does_not_clear_current_in_flight(
        self, app_module
    ):
        """A stale token failure must not drop the guard for a newer active refresh."""
        delegate = _make_delegate(
            app_module,
            in_flight=True,
            refresh_token=2,
            last_stats_fetch=30.0,
        )

        app_module.OMLXAppDelegate.statsRefreshFailedOnMain_(
            delegate,
            {"token": 1, "error": "stale failure"},
        )

        assert delegate._stats_refresh_in_flight is True
        assert delegate._last_stats_fetch == 30.0
        delegate._update_menubar_icon.assert_not_called()

    def test_invalidated_generations_do_not_share_worker_sessions(self, app_module):
        """Old and new generation workers should not reuse the same Session object."""
        delegate = _make_delegate(app_module, in_flight=True, refresh_token=0)

        session_one = Mock()
        session_one.get.side_effect = [
            _make_response(200, {"session": "one"}),
            _make_response(200, {"alltime": "one"}),
        ]
        session_two = Mock()
        session_two.get.side_effect = [
            _make_response(200, {"session": "two"}),
            _make_response(200, {"alltime": "two"}),
        ]
        scheduled_targets = []

        class DeferredThread:
            def __init__(self, *, target, name, daemon):
                self._target = target

            def start(self):
                scheduled_targets.append(self._target)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                app_module.requests,
                "Session",
                Mock(side_effect=[session_one, session_two]),
            )
            mp.setattr(app_module.threading, "Thread", DeferredThread)
            mp.setattr(app_module.time, "time", lambda: 42.0)
            app_module.OMLXAppDelegate._start_stats_refresh(delegate)
            app_module.OMLXAppDelegate._invalidate_stats_refresh_generation(delegate)
            app_module.OMLXAppDelegate._start_stats_refresh(delegate)
            scheduled_targets[0]()
            scheduled_targets[1]()

        assert delegate._admin_session is None
        assert len(scheduled_targets) == 2

        payloads = [
            call.args[1]
            for call in (
                delegate.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
            )
        ]
        assert payloads[0]["token"] == 0
        assert payloads[1]["token"] == 1
        assert payloads[0]["session"] is session_one
        assert payloads[1]["session"] is session_two
        assert payloads[0]["session"] is not payloads[1]["session"]
        assert delegate._admin_session is None

        app_module.OMLXAppDelegate.statsRefreshFinishedOnMain_(delegate, payloads[0])
        payloads[0]["session"].close.assert_called_once_with()
        assert delegate._admin_session is None

        app_module.OMLXAppDelegate.statsRefreshFinishedOnMain_(delegate, payloads[1])
        assert delegate._admin_session is session_two

    def test_stale_completion_after_stop_is_discarded(self, app_module):
        """Results from pre-stop refreshes should not repopulate caches or bump cooldown."""
        server_manager = types.SimpleNamespace(
            status=app_module.ServerStatus.RUNNING,
            stop=Mock(
                side_effect=lambda: setattr(
                    server_manager, "status", app_module.ServerStatus.STOPPED
                )
            ),
        )
        delegate = types.SimpleNamespace(
            server_manager=server_manager,
            _last_stats_fetch=11.0,
            _last_stats_refresh_started_at=11.0,
            _stats_refresh_in_flight=True,
            _stats_refresh_token=3,
            _cached_stats={"old": 1},
            _cached_alltime_stats={"old_total": 2},
            _admin_session=object(),
            _update_status_display=Mock(),
            _build_menu=Mock(),
            _update_menubar_icon=Mock(),
        )
        delegate._invalidate_stats_refresh_generation = (
            app_module.OMLXAppDelegate._invalidate_stats_refresh_generation.__get__(
                delegate
            )
        )
        delegate._close_session = app_module.OMLXAppDelegate._close_session.__get__(
            delegate
        )
        delegate._close_admin_session = (
            app_module.OMLXAppDelegate._close_admin_session.__get__(delegate)
        )

        app_module.OMLXAppDelegate.stopServer_(delegate, None)
        stale_payload = {
            "token": 3,
            "finished_at": 12.0,
            "stats": {"should_not": "apply"},
            "alltime_stats": {"should_not": "apply"},
            "session": object(),
        }
        app_module.OMLXAppDelegate.statsRefreshFinishedOnMain_(delegate, stale_payload)

        assert delegate._stats_refresh_token == 4
        assert delegate._stats_refresh_in_flight is False
        assert delegate._last_stats_fetch == 0
        assert delegate._last_stats_refresh_started_at == 0
        assert delegate._cached_stats is None
        assert delegate._cached_alltime_stats is None
        assert delegate._admin_session is None
        delegate._build_menu.assert_not_called()

    def test_non_running_completion_closes_unapplied_session(self, app_module):
        """A current-generation completion should close its session if the server is no longer running."""
        delegate = _make_delegate(
            app_module,
            status=app_module.ServerStatus.STOPPED,
            in_flight=True,
            refresh_token=5,
            last_stats_fetch=30.0,
        )
        session = Mock()

        app_module.OMLXAppDelegate.statsRefreshFinishedOnMain_(
            delegate,
            {
                "token": 5,
                "finished_at": 31.0,
                "stats": {"stale": True},
                "alltime_stats": {"stale": True},
                "session": session,
            },
        )

        session.close.assert_called_once_with()
        assert delegate._stats_refresh_in_flight is False
        assert delegate._last_stats_fetch == 30.0
        assert delegate._admin_session is None
        delegate._build_menu.assert_not_called()

    def test_non_running_transition_invalidates_refresh_generation_token(
        self, app_module
    ):
        """RUNNING -> non-running transitions should invalidate prior refresh callbacks."""
        delegate = _make_delegate(
            app_module,
            status=app_module.ServerStatus.ERROR,
            last_health_status=app_module.ServerStatus.RUNNING,
            in_flight=True,
            refresh_token=7,
            last_stats_fetch=20.0,
            last_started_at=20.0,
        )
        delegate._cached_stats = {"old": 1}
        delegate._cached_alltime_stats = {"old_total": 2}

        app_module.OMLXAppDelegate.healthCheck_(delegate, None)

        assert delegate._stats_refresh_token == 8
        assert delegate._stats_refresh_in_flight is False
        assert delegate._cached_stats is None
        assert delegate._cached_alltime_stats is None

    def test_invalidate_generation_clears_admin_session_without_closing(self, app_module):
        """Generation invalidation should drop the session reference but not
        close it eagerly — the in-flight worker still holds a reference and
        the stale-token rejection in the completion handler closes it."""
        delegate = _make_delegate(app_module, refresh_token=4)
        session = Mock()
        delegate._admin_session = session

        app_module.OMLXAppDelegate._invalidate_stats_refresh_generation(delegate)

        session.close.assert_not_called()
        assert delegate._stats_refresh_token == 5
        assert delegate._admin_session is None

    def test_health_check_watchdog_recovers_stuck_refresh(self, app_module):
        """Stuck in-flight refresh state should be invalidated and immediately recovered."""
        delegate = _make_delegate(
            app_module,
            status=app_module.ServerStatus.RUNNING,
            in_flight=True,
            refresh_token=8,
            last_stats_fetch=20.0,
            last_started_at=10.0,
        )
        delegate._cached_stats = {"old": 1}
        delegate._cached_alltime_stats = {"old_total": 2}
        delegate._admin_session = Mock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(app_module.time, "time", lambda: 30.0)
            app_module.OMLXAppDelegate.healthCheck_(delegate, None)

        assert delegate._stats_refresh_token == 9
        assert delegate._cached_stats is None
        assert delegate._cached_alltime_stats is None
        assert delegate._admin_session is None
        delegate._start_stats_refresh.assert_called_once_with()
        assert delegate._stats_refresh_in_flight is True
        assert delegate._last_stats_refresh_started_at == 30.0

    def test_health_check_stopping_transition_invalidates_old_refresh_generation(
        self, app_module
    ):
        """A RUNNING -> STOPPING timer sample should invalidate stale in-flight refreshes."""
        delegate = _make_delegate(
            app_module,
            status=app_module.ServerStatus.STOPPING,
            last_health_status=app_module.ServerStatus.RUNNING,
            in_flight=True,
            refresh_token=9,
            last_stats_fetch=25.0,
            last_started_at=25.0,
        )
        delegate._cached_stats = {"old": 1}
        delegate._cached_alltime_stats = {"old_total": 2}
        delegate._admin_session = object()

        app_module.OMLXAppDelegate.healthCheck_(delegate, None)

        assert delegate._stats_refresh_token == 10
        assert delegate._stats_refresh_in_flight is False
        assert delegate._cached_stats is None
        assert delegate._cached_alltime_stats is None
        assert delegate._admin_session is None

        delegate.server_manager.status = app_module.ServerStatus.RUNNING
        app_module.OMLXAppDelegate.healthCheck_(delegate, None)

        stale_payload = {
            "token": 9,
            "finished_at": 26.0,
            "stats": {"stale": True},
            "alltime_stats": {"stale": True},
            "session": object(),
        }
        app_module.OMLXAppDelegate.statsRefreshFinishedOnMain_(delegate, stale_payload)

        assert delegate._stats_refresh_token == 10
        assert delegate._cached_stats is None
        assert delegate._cached_alltime_stats is None
        assert delegate._admin_session is None

    def test_status_flap_between_timer_ticks_invalidates_old_refresh_generation(
        self, app_module
    ):
        """A quick RUNNING->STARTING->RUNNING flap should still discard stale old-generation completions."""
        delegate = _make_delegate(
            app_module,
            status=app_module.ServerStatus.STARTING,
            last_health_status=app_module.ServerStatus.RUNNING,
            in_flight=True,
            refresh_token=9,
            last_stats_fetch=25.0,
            last_started_at=25.0,
        )
        delegate._cached_stats = {"before": 1}
        delegate._cached_alltime_stats = {"before_total": 2}
        delegate._admin_session = object()

        app_module.OMLXAppDelegate.serverStatusChangedOnMain_(
            delegate, app_module.ServerStatus.STARTING
        )

        assert delegate._stats_refresh_token == 10
        assert delegate._stats_refresh_in_flight is False
        assert delegate._cached_stats is None
        assert delegate._cached_alltime_stats is None

        delegate.server_manager.status = app_module.ServerStatus.RUNNING
        app_module.OMLXAppDelegate.serverStatusChangedOnMain_(
            delegate, app_module.ServerStatus.RUNNING
        )

        stale_payload = {
            "token": 9,
            "finished_at": 26.0,
            "stats": {"stale": True},
            "alltime_stats": {"stale": True},
            "session": object(),
        }
        app_module.OMLXAppDelegate.statsRefreshFinishedOnMain_(delegate, stale_payload)

        assert delegate._stats_refresh_token == 10
        assert delegate._cached_stats is None
        assert delegate._cached_alltime_stats is None
        assert delegate._admin_session is None

    def test_health_check_stopping_then_delayed_callback_still_invalidates_generation(
        self, app_module
    ):
        """RUNNING -> STOPPING sampled by timer must invalidate even if callback arrives late."""
        delegate = _make_delegate(
            app_module,
            status=app_module.ServerStatus.STOPPING,
            last_health_status=app_module.ServerStatus.RUNNING,
            in_flight=True,
            refresh_token=12,
            last_stats_fetch=30.0,
            last_started_at=30.0,
        )
        delegate._cached_stats = {"old": 1}
        delegate._cached_alltime_stats = {"old_total": 2}
        delegate._admin_session = object()

        app_module.OMLXAppDelegate.healthCheck_(delegate, None)

        assert delegate._stats_refresh_token == 13
        assert delegate._stats_refresh_in_flight is False
        assert delegate._cached_stats is None
        assert delegate._cached_alltime_stats is None
        assert delegate._admin_session is None
        assert delegate._last_health_status == app_module.ServerStatus.STOPPING

        app_module.OMLXAppDelegate.serverStatusChangedOnMain_(
            delegate, app_module.ServerStatus.STOPPING
        )
        assert delegate._stats_refresh_token == 13

        delegate.server_manager.status = app_module.ServerStatus.RUNNING
        app_module.OMLXAppDelegate.serverStatusChangedOnMain_(
            delegate, app_module.ServerStatus.RUNNING
        )

        stale_payload = {
            "token": 12,
            "finished_at": 31.0,
            "stats": {"stale": True},
            "alltime_stats": {"stale": True},
            "session": object(),
        }
        app_module.OMLXAppDelegate.statsRefreshFinishedOnMain_(delegate, stale_payload)

        assert delegate._stats_refresh_token == 13
        assert delegate._cached_stats is None
        assert delegate._cached_alltime_stats is None
        assert delegate._admin_session is None

    @pytest.mark.parametrize("pass_existing_session", [False, True], ids=["new", "reused"])
    @pytest.mark.parametrize(
        "login_status,stats_retry_status",
        [
            (403, None),  # login rejected — early return before stats retry
            (200, 500),  # login OK, but stats retry fails
        ],
        ids=["login_failure", "stats_retry_failure"],
    )
    def test_fetch_stats_snapshot_closes_session_on_failure(
        self, app_module, pass_existing_session, login_status, stats_retry_status
    ):
        """Failure paths should close sessions whose ownership is dropped by the snapshot fetch."""
        session = Mock()
        responses = [_make_response(401)]  # initial stats -> 401 triggers login
        responses.append(_make_response(login_status))
        if stats_retry_status is not None:
            responses.append(_make_response(stats_retry_status))
        session.get.side_effect = lambda *a, **kw: responses.pop(0) if responses else _make_response(500)
        session.post.side_effect = lambda *a, **kw: responses.pop(0) if responses else _make_response(500)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(app_module.requests, "Session", Mock(return_value=session))
            stats, alltime, returned_session = app_module.OMLXAppDelegate._fetch_stats_snapshot(
                "test-key", "http://127.0.0.1:11434",
                session if pass_existing_session else None,
            )

        assert stats is None
        assert alltime is None
        assert returned_session is None
        session.close.assert_called_once()

    @pytest.mark.parametrize("pass_existing_session", [False, True], ids=["new", "reused"])
    def test_fetch_stats_snapshot_closes_session_on_request_exception(
        self, app_module, pass_existing_session
    ):
        """Request exceptions should also close sessions whose ownership is abandoned."""
        session = Mock()
        session.get.side_effect = app_module.requests.RequestException("boom")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(app_module.requests, "Session", Mock(return_value=session))
            stats, alltime, returned_session = app_module.OMLXAppDelegate._fetch_stats_snapshot(
                "test-key", "http://127.0.0.1:11434",
                session if pass_existing_session else None,
            )

        assert stats is None
        assert alltime is None
        assert returned_session is None
        session.close.assert_called_once()

    def test_is_newer_version_uses_packaging_pep440_semantics(self, app_module):
        """Version checks should follow packaging's PEP 440 parser semantics."""
        delegate = types.SimpleNamespace()
        assert app_module.OMLXAppDelegate._is_newer_version(delegate, "1.10.0", "1.9.9")
        assert app_module.OMLXAppDelegate._is_newer_version(
            delegate, "v1.10.0", "1.9.9"
        )
        assert app_module.OMLXAppDelegate._is_newer_version(
            delegate, "1!0.2.20", "0.2.20"
        )
        assert app_module.OMLXAppDelegate._is_newer_version(
            delegate, "0.2.20", "0.2.20rc1"
        )
        assert app_module.OMLXAppDelegate._is_newer_version(
            delegate, "0.2.3.post4", "0.2.3"
        )
        assert not app_module.OMLXAppDelegate._is_newer_version(
            delegate, "1.10.0rc1", "1.9.9"
        )
        assert not app_module.OMLXAppDelegate._is_newer_version(
            delegate, "1.10.0.dev1", "1.9.9"
        )
        assert not app_module.OMLXAppDelegate._is_newer_version(
            delegate, "1.9.0", "1.10.0"
        )
        assert not app_module.OMLXAppDelegate._is_newer_version(
            delegate, "not-a-version", "1.10.0"
        )
