"""
oMLX Native Menubar Application using PyObjC.

A native macOS menubar app for managing the oMLX LLM inference server.
"""

import logging
import os
import platform
import plistlib
import subprocess
import time
import webbrowser
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Optional

import objc
import requests

from omlx._version import __version__
from AppKit import (
    NSAlert,
    NSAlertFirstButtonReturn,
    NSAlertSecondButtonReturn,
    NSAlertThirdButtonReturn,
    NSApp,
    NSAppearanceNameDarkAqua,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSApplicationActivationPolicyRegular,
    NSAttributedString,
    NSBundle,
    NSColor,
    NSFloatingWindowLevel,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSImage,
    NSLinkAttributeName,
    NSMenu,
    NSMenuItem,
    NSMutableParagraphStyle,
    NSParagraphStyleAttributeName,
    NSRightTabStopType,
    NSStatusBar,
    NSTextField,
    NSTextTab,
    NSTextAlignmentCenter,
    NSVariableStatusItemLength,
    NSView,
    NSWorkspace,
)
from Foundation import (
    NSData,
    NSMutableAttributedString,
    NSObject,
    NSRunLoop,
    NSRunLoopCommonModes,
    NSTimer,
    NSURL,
)

from .config import ServerConfig, resolve_local_server_base_url
from .server_manager import PortConflict, ServerManager, ServerStatus

logger = logging.getLogger(__name__)


def _find_matching_dmg(assets: list[dict]) -> str | None:
    """Select the DMG asset matching the current macOS version.

    DMG filenames follow the pattern: oMLX-0.2.10-macos15-sequoia_260210.dmg
    Matches 'macosNN' from filename against the running OS major version.
    Falls back to the single DMG if only one is available.
    """
    mac_ver = platform.mac_ver()[0]  # e.g., "15.3.1" or "26.0"
    os_major = mac_ver.split(".")[0]  # e.g., "15" or "26"
    os_tag = f"macos{os_major}"  # e.g., "macos15" or "macos26"

    dmg_assets = [a for a in assets if a.get("name", "").endswith(".dmg")]

    # Exact OS match
    for asset in dmg_assets:
        name = asset["name"]
        if f"-{os_tag}-" in name or f"-{os_tag}_" in name:
            return asset["browser_download_url"]

    # Fallback: single DMG release (no platform tag or only one DMG)
    if len(dmg_assets) == 1:
        return dmg_assets[0]["browser_download_url"]

    return None


class OMLXAppDelegate(NSObject):
    """Main application delegate for oMLX menubar app."""

    def init(self):
        self = objc.super(OMLXAppDelegate, self).init()
        if self is None:
            return None

        self.config = ServerConfig.load()
        self.server_manager = ServerManager(self.config)
        self.status_item = None
        self.menu = None
        self.health_timer = None
        self.welcome_controller = None
        self.preferences_controller = None
        self._cached_stats: Optional[dict] = None
        self._cached_alltime_stats: Optional[dict] = None
        self._last_stats_fetch: float = 0
        self._admin_session: Optional[requests.Session] = None
        self._icon_outline: Optional[NSImage] = None
        self._icon_filled: Optional[NSImage] = None
        self._update_info: Optional[dict] = None
        self._last_update_check: float = 0
        self._updater = None  # AppUpdater instance during download
        self._update_progress_text = ""  # Current download progress text
        self._menu_is_open = False  # True while the status-bar menu is visible
        # Menubar visibility tracking — Tahoe ControlCenter can hide the item
        # silently, and isVisible() returns True even when hidden (see issue #725)
        self._visibility_check_timer = None
        self._warned_hidden = False
        # One-shot auto recovery: if the initial NSStatusItem registers but
        # isn't rendered (known Tahoe race), we try removing and recreating
        # it exactly once before giving up and alerting the user.
        self._recreate_attempted = False
        self._policy_switch_timer = None
        # Weak references to dynamic menu items for in-place updates
        self._status_header_item = None
        self._stop_item = None
        self._restart_item = None
        self._start_item = None
        self._admin_panel_item = None
        self._chat_item = None

        return self

    def applicationDidFinishLaunching_(self, notification):
        """Called when app finishes launching."""
        try:
            self._doFinishLaunching()
        except Exception as e:
            logger.error(f"Launch failed: {e}", exc_info=True)
            self._show_fatal_error_and_quit(str(e))

    def applicationShouldTerminateAfterLastWindowClosed_(self, app):
        """Prevent termination when the last window closes."""
        return False

    def applicationShouldHandleReopen_hasVisibleWindows_(self, app, flag):
        """Respond when user clicks the app icon while already running."""
        if self.server_manager.is_running():
            self.openDashboard_(None)
        return True

    def _show_fatal_error_and_quit(self, message: str):
        """Show a fatal error dialog and terminate the application."""
        alert = NSAlert.alloc().init()
        alert.setMessageText_("oMLX Failed to Launch")
        alert.setInformativeText_(message)
        alert.addButtonWithTitle_("Quit")
        alert.runModal()
        NSApp.terminate_(None)

    def _doFinishLaunching(self):
        """Actual launch logic (separated for proper exception handling)."""
        # Pre-load menubar icons (template images auto-adjust to menubar background)
        self._icon_outline = self._load_menubar_icon("menubar-outline.svg")
        self._icon_filled = self._load_menubar_icon("menubar-filled.svg")

        # Create status bar item (with accessibility metadata so menu-bar
        # managers like Bartender / Ice can enumerate it correctly).
        self._create_status_item()

        # Build menu
        self._build_menu()

        # Start health check timer
        self.health_timer = (
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                5.0, self, "healthCheck:", None, True
            )
        )
        NSRunLoop.currentRunLoop().addTimer_forMode_(
            self.health_timer, NSRunLoopCommonModes
        )

        # Switch from Regular to Accessory policy now that the status bar
        # item exists. This hides the Dock icon while keeping the menubar item.
        # We start as Regular (in main()) so macOS grants full GUI access,
        # then switch here — required on macOS Tahoe where Accessory apps
        # launched via LaunchServices remain "NotVisible" otherwise.
        # IMPORTANT: Info.plist must NOT contain LSUIElement=true. Combining
        # LSUIElement with this runtime policy switch causes ControlCenter
        # to block the NSStatusItem on Sonoma+. See issue #725.
        #
        # Deferred by one runloop tick so the status-item registration with
        # WindowServer settles before the activation policy changes. Doing
        # both in the same tick seems to interleave on Tahoe and sometimes
        # results in the item being registered but never composited.
        self._policy_switch_timer = (
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.0, self, "switchToAccessoryPolicy:", None, False
            )
        )

        logger.info("oMLX menubar app launched successfully")

        # Clean up leftover staged update from previous attempt
        from .updater import AppUpdater

        AppUpdater.cleanup_staged_app()

        # Check for updates (non-blocking, cached for 24h)
        self._check_for_updates()

        # First run: show welcome screen
        if self.config.is_first_run:
            from .welcome import WelcomeWindowController

            self.welcome_controller = (
                WelcomeWindowController.alloc().initWithConfig_serverManager_(
                    self.config, self.server_manager
                )
            )
            self.welcome_controller.showWindow()
        elif self.config.start_server_on_launch:
            result = self.server_manager.start()
            if isinstance(result, PortConflict):
                self._handle_port_conflict(result)
            else:
                self._update_status_display()

        # Delayed check: warn user if ControlCenter blocked the status item.
        # 3s delay gives ControlCenter time to settle its visibility decision.
        # Retain the timer reference to prevent early dealloc under PyObjC.
        self._visibility_check_timer = (
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                3.0, self, "checkStatusItemVisibility:", None, False
            )
        )

    def _create_status_item(self):
        """Create the NSStatusItem and set accessibility metadata.

        Pulled out so we can invoke it again from `_recreate_status_item()`
        when Tahoe registers the item but never composites it (issue #725).
        Each field lands in the AX tree menu-bar managers read:

        - autosaveName persists the item's placement across launches and
          gives ControlCenter a stable identity to key against.
        - accessibility identifier / title / label fill the fields AX
          tools show for third-party items. Without them a PyObjC-created
          button lands as an anonymous entry that some managers skip.

        Note: dev4 also tried setAccessibilityElement_ / setAccessibilityRole_
        and NSAccessibilityPostNotification with NSAccessibilityCreatedNotification
        in an attempt to make Bartender discover us. They made no observable
        difference on real-device testing, so they're not here anymore.
        Bartender's filter runs above the AX metadata we can reach.
        """
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength
        )
        self.status_item.setAutosaveName_("com.omlx.app-statusItem")

        button = self.status_item.button()
        if button is not None:
            if hasattr(button, "setAccessibilityIdentifier_"):
                button.setAccessibilityIdentifier_(
                    "com.omlx.app.statusItemButton"
                )
            if hasattr(button, "setAccessibilityTitle_"):
                button.setAccessibilityTitle_("oMLX")
            if hasattr(button, "setAccessibilityLabel_"):
                button.setAccessibilityLabel_("oMLX")
            button.setToolTip_("oMLX")

        self._update_menubar_icon()

    def switchToAccessoryPolicy_(self, timer):
        """Switch activation policy on the next runloop tick (see _doFinishLaunching)."""
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        NSApp.activateIgnoringOtherApps_(True)

    def _recreate_status_item(self) -> None:
        """Last-ditch recovery: remove and recreate the NSStatusItem once.

        Some Tahoe launches end with button/window registered but the
        WindowServer never composites the icon. Community reports (Maccy
        #1224, Stats #2734) find that removing the item and re-adding it
        sometimes re-attaches it to a visible slot. Gated by
        `_recreate_attempted` so this runs at most once per session.
        """
        if self._recreate_attempted:
            return
        self._recreate_attempted = True
        logger.warning(
            "recreating NSStatusItem as a one-shot recovery attempt"
        )
        old = self.status_item
        try:
            NSStatusBar.systemStatusBar().removeStatusItem_(old)
        except Exception as e:
            logger.warning("removeStatusItem failed: %s", e)
        self._create_status_item()
        # Re-attach menu (setMenu_ is idempotent but the new item has no menu yet)
        if self.menu is not None:
            self.status_item.setMenu_(self.menu)

    def _is_status_item_hidden(self) -> bool:
        """Detect whether the menubar icon is actually rendered.

        There's no single reliable signal on Tahoe, so we combine:

        - NSStatusItem.isVisible(): app-side setVisible: flag only. Stays
          True when Menu Bar settings hide the item, so it alone can't
          catch Tahoe's toggle-off.
        - button.window().isVisible: NSWindow's "attached to screen" flag;
          tends to flip False when the status item is blocked.
        - button.window().occlusionState & NSWindowOcclusionStateVisible:
          finer-grained compositing flag that's cleared when macOS parks
          the window off-screen or in the blocked list.

        Treat the item as hidden if ANY of those strong signals say hidden.
        Emit a single WARNING with the raw signals when we return True, so
        a hidden icon always leaves a breadcrumb in menubar.log without
        spamming the log on every check while the icon is fine.
        """
        NS_WINDOW_OCCLUSION_STATE_VISIBLE = 1 << 1  # NSWindowOcclusionStateVisible

        button = self.status_item.button() if self.status_item else None
        window = button.window() if button else None
        api_visible = bool(self.status_item and self.status_item.isVisible())
        window_visible = bool(window and window.isVisible())
        occlusion = int(window.occlusionState()) if window else 0
        occlusion_visible = bool(occlusion & NS_WINDOW_OCCLUSION_STATE_VISIBLE)

        hidden = (
            not button
            or not window
            or not api_visible
            or not window_visible
            or not occlusion_visible
        )
        if hidden:
            frame = window.frame() if window else None
            frame_str = (
                f"({frame.origin.x:.1f},{frame.origin.y:.1f},"
                f"{frame.size.width:.1f}x{frame.size.height:.1f})"
                if frame
                else None
            )
            logger.warning(
                "status item hidden: pid=%d api_visible=%s window_visible=%s "
                "occlusion=0x%x button=%s window=%s frame=%s recreated=%s",
                os.getpid(),
                api_visible,
                window_visible,
                occlusion,
                bool(button),
                bool(window),
                frame_str,
                self._recreate_attempted,
            )
        return hidden

    def checkStatusItemVisibility_(self, timer):
        """One-shot post-launch check for menubar icon visibility.

        If the icon is hidden on the first probe, try the recovery path
        (recreate the NSStatusItem) exactly once before alerting the user.
        This covers the Tahoe case where the initial registration races
        with WindowServer and the item never composites; users on menu-
        bar managers (Bartender, Ice) also benefit since the recreated
        item has full accessibility metadata attached. The check runs
        only once at launch; mid-session probes would false-trigger when
        macOS auto-hides the menu bar (fullscreen video, slideshow, etc.).
        """
        if not self._is_status_item_hidden():
            return

        if not self._recreate_attempted:
            self._recreate_status_item()
            # Give macOS ~1s to finish registering the new item before we
            # re-probe. If it's still hidden, show the alert.
            self._visibility_check_timer = (
                NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                    1.0, self, "checkStatusItemVisibilityAfterRecreate:", None, False
                )
            )
            return

        # Recovery already used or disabled; escalate to the user.
        self._show_menubar_hidden_alert()

    def checkStatusItemVisibilityAfterRecreate_(self, timer):
        """Second probe after the one-shot recreate. Alerts if still hidden."""
        if self._is_status_item_hidden():
            self._show_menubar_hidden_alert()

    def _show_menubar_hidden_alert(self):
        """Inform the user about the hidden menubar icon and offer recovery.

        Tahoe (26.x) adds a dedicated Menu Bar settings pane with per-app
        toggles, so the alert deep-links there. It also exposes a StatusKit
        approval mechanism (`trackedApplications` in group.com.apple.
        controlcenter.plist) where `isAllowed: false` silently blocks the
        icon. The Auto-Fix button flips oMLX's flag to true and restarts
        ControlCenter, but needs Full Disk Access to touch the Group
        Container plist.

        Earlier versions of macOS have no System Settings UI for third-
        party status items, so on Sequoia and older we drop the Settings
        and Auto-Fix buttons entirely to avoid pointing users at dead ends.
        """
        if self._warned_hidden:
            return
        self._warned_hidden = True

        try:
            mac_major = int(platform.mac_ver()[0].split(".")[0])
        except (ValueError, IndexError):
            mac_major = 0
        is_tahoe_or_newer = mac_major >= 26

        # Accessory apps don't steal focus, so the alert would otherwise land
        # behind every other window. Activate first and raise the alert window
        # to floating level so it surfaces above the browser/editor the user
        # is likely looking at.
        NSApp.activateIgnoringOtherApps_(True)

        alert = NSAlert.alloc().init()
        alert.setMessageText_("oMLX Menubar Icon Hidden")

        settings_label = "Open Menu Bar Settings"
        settings_url = (
            "x-apple.systempreferences:com.apple.ControlCenter-Settings."
            "extension?MenuBar"
        )

        if is_tahoe_or_newer and self._is_bartender_running():
            # Bartender's menubar filter hides oMLX regardless of StatusKit
            # state, so swap in a Bartender-specific dialog with no
            # Auto-Fix or System Settings buttons (neither would help here).
            self._show_bartender_conflict_alert()
            return

        if is_tahoe_or_newer:
            alert.setInformativeText_(
                "The oMLX menubar icon isn't showing up.\n\n"
                "On macOS Tahoe this is usually caused by the StatusKit "
                "approval flag being false in the system preferences. "
                "\"Auto-Fix\" will flip that flag and restart ControlCenter "
                "(needs Full Disk Access), or you can toggle the app manually "
                "in System Settings > Menu Bar."
            )
            alert.addButtonWithTitle_("Auto-Fix")       # 1000
            alert.addButtonWithTitle_(settings_label)    # 1001
            alert.addButtonWithTitle_("View Log")        # 1002
            alert.addButtonWithTitle_("Dismiss")         # 1003
        else:
            alert.setInformativeText_(
                "The oMLX menubar icon isn't showing up.\n\n"
                "macOS before Tahoe doesn't offer a System Settings toggle "
                "for third-party menubar apps. Try quitting and relaunching "
                "oMLX, and check menubar manager tools like Bartender or "
                "Ice if you use them.\n\n"
                "Click \"View Log\" to see what the app detected."
            )
            alert.addButtonWithTitle_("View Log")        # 1000
            alert.addButtonWithTitle_("Dismiss")         # 1001

        alert_window = alert.window()
        if alert_window is not None:
            alert_window.setLevel_(NSFloatingWindowLevel)

        response = alert.runModal()
        log_path = (
            Path.home()
            / "Library"
            / "Application Support"
            / "oMLX"
            / "logs"
            / "menubar.log"
        )

        if is_tahoe_or_newer:
            if response == NSAlertFirstButtonReturn:
                self._run_autofix_flow()
            elif response == NSAlertSecondButtonReturn:
                NSWorkspace.sharedWorkspace().openURL_(
                    NSURL.URLWithString_(settings_url)
                )
            elif response == NSAlertThirdButtonReturn:
                NSWorkspace.sharedWorkspace().openURL_(
                    NSURL.fileURLWithPath_(str(log_path))
                )
        else:
            if response == NSAlertFirstButtonReturn:
                NSWorkspace.sharedWorkspace().openURL_(
                    NSURL.fileURLWithPath_(str(log_path))
                )

    # --- StatusKit auto-fix ---

    _STATUSKIT_PLIST_PATH = os.path.expanduser(
        "~/Library/Group Containers/group.com.apple.controlcenter"
        "/Library/Preferences/group.com.apple.controlcenter.plist"
    )

    def _run_autofix_flow(self) -> None:
        """Orchestrate the StatusKit Auto-Fix: check FDA, write plist, report.

        Flow: verify Full Disk Access -> patch the tracked applications
        plist -> kill ControlCenter -> show result. If FDA is missing,
        deep-link the user to System Settings and bail early so they can
        grant permission and retry from a fresh launch.
        """
        logger.info("Auto-Fix triggered by user.")

        if not self._has_full_disk_access():
            logger.info("Auto-Fix blocked: Full Disk Access not granted.")
            self._show_fda_request_alert()
            return

        success, message = self._fix_statuskit_permission()
        self._show_autofix_result_alert(success, message)

    def _has_full_disk_access(self) -> bool:
        """Probe read access on the StatusKit plist to infer FDA grant.

        If the plist doesn't exist, assume FDA is grantable (the write
        step would fail loudly anyway). If opening fails with
        PermissionError, TCC is blocking us and the user needs to add
        oMLX to Full Disk Access in Privacy & Security.
        """
        if not os.path.exists(self._STATUSKIT_PLIST_PATH):
            return True
        try:
            with open(self._STATUSKIT_PLIST_PATH, "rb") as f:
                f.read(1)
            return True
        except PermissionError:
            return False
        except OSError as e:
            logger.warning("FDA probe unexpected OSError: %s", e)
            return False

    def _open_full_disk_access_settings(self) -> None:
        """Deep-link to System Settings > Privacy & Security > Full Disk Access."""
        url = NSURL.URLWithString_(
            "x-apple.systempreferences:com.apple.settings.PrivacySecurity."
            "extension?Privacy_AllFiles"
        )
        NSWorkspace.sharedWorkspace().openURL_(url)

    def _is_bartender_running(self) -> bool:
        """Check whether Bartender (any version) is currently running.

        Used to swap the hidden-icon dialog for a Bartender-specific one
        when Bartender is active. Bartender's menubar filter excludes our
        status item regardless of what we do on the AX side, so the
        generic StatusKit Auto-Fix guidance is misleading in that case.
        Prefix-matches `com.surteesstudios.Bartender` to cover version 4,
        version 5 (`...Bartender 5`), and any future variants.
        """
        try:
            running = NSWorkspace.sharedWorkspace().runningApplications()
            for app in running:
                bid = app.bundleIdentifier()
                if bid and bid.startswith("com.surteesstudios.Bartender"):
                    return True
        except Exception as e:
            logger.debug("Bartender detection failed: %s", e)
        return False

    def _show_bartender_conflict_alert(self) -> None:
        """Tell the user Bartender is hiding oMLX and suggest alternatives.

        Called from `_show_menubar_hidden_alert` when Bartender is
        detected. No Auto-Fix or Open-Settings buttons because neither
        resolves a Bartender filter; the only real remedies are on
        Bartender's side (quit it, or switch to Ice).
        """
        NSApp.activateIgnoringOtherApps_(True)

        alert = NSAlert.alloc().init()
        alert.setMessageText_("oMLX Menubar Icon Hidden (Bartender Detected)")
        alert.setInformativeText_(
            "Bartender is currently running, and it appears to hide the "
            "oMLX menubar icon regardless of oMLX's own settings. "
            "Bartender's menubar filter excludes some apps (including "
            "Docker for Mac and other PyObjC-based menubar apps) for "
            "reasons that aren't configurable from oMLX's side.\n\n"
            "What to try:\n"
            "  • Check Bartender's item list. If oMLX is missing there, "
            "Bartender can't see us and no toggle will bring it back.\n"
            "  • Disable or quit Bartender while using oMLX.\n"
            "  • Consider Ice (https://icemenubar.app), an open-source "
            "alternative that tends to play better with PyObjC menubar "
            "apps."
        )
        alert.addButtonWithTitle_("View Log")  # 1000
        alert.addButtonWithTitle_("Dismiss")   # 1001

        alert_window = alert.window()
        if alert_window is not None:
            alert_window.setLevel_(NSFloatingWindowLevel)

        if alert.runModal() == NSAlertFirstButtonReturn:
            log_path = (
                Path.home()
                / "Library"
                / "Application Support"
                / "oMLX"
                / "logs"
                / "menubar.log"
            )
            NSWorkspace.sharedWorkspace().openURL_(
                NSURL.fileURLWithPath_(str(log_path))
            )

    def _show_fda_request_alert(self) -> None:
        """Explain why FDA is needed and offer to open the right settings pane."""
        NSApp.activateIgnoringOtherApps_(True)
        alert = NSAlert.alloc().init()
        alert.setMessageText_("Full Disk Access Required")
        alert.setInformativeText_(
            "Auto-Fix needs Full Disk Access so oMLX can edit the StatusKit "
            "approval file in your Group Containers folder. macOS blocks "
            "that path by default.\n\n"
            "1. Click \"Open Privacy Settings\" below.\n"
            "2. Find oMLX in the Full Disk Access list (drag it in from "
            "/Applications if it isn't listed).\n"
            "3. Toggle oMLX on.\n"
            "4. Quit oMLX and relaunch, then click Auto-Fix again."
        )
        alert.addButtonWithTitle_("Open Privacy Settings")
        alert.addButtonWithTitle_("Cancel")
        alert_window = alert.window()
        if alert_window is not None:
            alert_window.setLevel_(NSFloatingWindowLevel)
        if alert.runModal() == NSAlertFirstButtonReturn:
            self._open_full_disk_access_settings()

    def _show_autofix_result_alert(self, success: bool, message: str) -> None:
        """Surface the outcome of _fix_statuskit_permission() back to the user."""
        NSApp.activateIgnoringOtherApps_(True)
        alert = NSAlert.alloc().init()
        alert.setMessageText_(
            "Auto-Fix Succeeded" if success else "Auto-Fix Failed"
        )
        alert.setInformativeText_(message)
        alert.addButtonWithTitle_("OK")
        alert_window = alert.window()
        if alert_window is not None:
            alert_window.setLevel_(NSFloatingWindowLevel)
        alert.runModal()

    def _backup_statuskit_plist(self) -> Optional[Path]:
        """Snapshot the StatusKit plist before mutating it.

        Returns the backup path on success, None if the backup couldn't be
        written. A missing original (first-ever tracked app) isn't an
        error: we just have nothing to back up and return None.
        """
        src = Path(self._STATUSKIT_PLIST_PATH)
        if not src.exists():
            return None
        backup_dir = (
            Path.home() / "Library" / "Application Support" / "oMLX" / "backups"
        )
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = backup_dir / f"statuskit-{timestamp}.plist"
        try:
            backup.write_bytes(src.read_bytes())
            logger.info("StatusKit plist backed up to %s", backup)
            return backup
        except OSError as e:
            logger.warning("StatusKit plist backup failed: %s", e)
            return None

    def _fix_statuskit_permission(self) -> tuple[bool, str]:
        """Flip com.omlx.app's StatusKit isAllowed flag to True.

        Steps: back up the plist, load the outer binary plist, decode the
        nested `trackedApplications` list (either raw list or a nested
        binary plist blob depending on the Tahoe build), locate or append
        a com.omlx.app entry, rewrite atomically (preserving the original
        `bytes` vs `list` format), verify by re-reading, then `killall
        ControlCenter` so the daemon picks up the change. Returns
        (success, message) for display in the result dialog.
        """
        plist_path = Path(self._STATUSKIT_PLIST_PATH)

        if not plist_path.exists():
            return (
                False,
                "The StatusKit preferences file does not exist on this Mac. "
                "Your macOS version may not use the approval flow yet; "
                "the issue is likely not auto-fixable.",
            )

        backup = self._backup_statuskit_plist()

        try:
            with open(plist_path, "rb") as f:
                data = plistlib.load(f)
        except PermissionError:
            return (
                False,
                "Permission denied reading the StatusKit preferences. "
                "Grant oMLX Full Disk Access in Privacy & Security.",
            )
        except Exception as e:
            return False, f"Failed to read the StatusKit preferences: {e}"

        raw = data.get("trackedApplications")
        # Record the original container type so we can re-encode identically.
        nested_as_bytes = isinstance(raw, (bytes, bytearray))

        if raw is None:
            inner: list = []
        elif nested_as_bytes:
            try:
                inner = plistlib.loads(bytes(raw))
            except Exception as e:
                return False, f"Failed to decode trackedApplications: {e}"
            if not isinstance(inner, list):
                return (
                    False,
                    f"trackedApplications decoded to {type(inner).__name__}, "
                    "expected list. Aborting to avoid corrupting the file.",
                )
        elif isinstance(raw, list):
            inner = raw
        else:
            return (
                False,
                f"Unexpected trackedApplications type: {type(raw).__name__}.",
            )

        target = "com.omlx.app"
        changed = False
        found_already_allowed = False
        original_states: list[object] = []
        for entry in inner:
            if not isinstance(entry, dict):
                continue
            bid = entry.get("location", {}).get("bundle", {}).get("_0")
            if bid != target:
                continue
            original_states.append(entry.get("isAllowed", "<missing>"))
            if entry.get("isAllowed") is True:
                found_already_allowed = True
                continue
            entry["isAllowed"] = True
            changed = True

        if changed or found_already_allowed:
            logger.info(
                "%s found in StatusKit with prior isAllowed states %r",
                target,
                original_states,
            )

        appended_new = False
        if not changed and not found_already_allowed:
            new_entry = {
                "location": {"bundle": {"_0": target}},
                "menuItemLocations": [{"bundle": {"_0": target}}],
                "isAllowed": True,
            }
            inner.append(new_entry)
            changed = True
            appended_new = True
            logger.info(
                "%s not in StatusKit list; appended new entry (isAllowed=True).",
                target,
            )

        if not changed:
            return (
                True,
                "oMLX is already approved in StatusKit. If the icon still "
                "doesn't appear, the root cause is something else. Share "
                "the latest menubar.log with the maintainer.",
            )

        # Re-encode preserving the original container format.
        if nested_as_bytes or raw is None:
            data["trackedApplications"] = plistlib.dumps(
                inner, fmt=plistlib.FMT_BINARY
            )
        else:
            data["trackedApplications"] = inner

        tmp_path = plist_path.with_suffix(plist_path.suffix + ".omlx-tmp")

        def _cleanup_tmp() -> None:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        replaced = False
        try:
            with open(tmp_path, "wb") as f:
                plistlib.dump(data, f, fmt=plistlib.FMT_BINARY)
            os.replace(tmp_path, plist_path)
            replaced = True
            logger.info("StatusKit plist rewritten at %s", plist_path)
        except PermissionError:
            _cleanup_tmp()
            return (
                False,
                "Permission denied writing the StatusKit preferences. "
                "Full Disk Access may have been revoked mid-operation.",
            )
        except Exception as e:
            _cleanup_tmp()
            if backup is not None:
                try:
                    plist_path.write_bytes(backup.read_bytes())
                    logger.warning(
                        "Write failed (%s); restored plist from %s", e, backup
                    )
                except OSError as restore_err:
                    logger.error(
                        "Write failed and restore also failed: %s", restore_err
                    )
            return False, f"Failed to write the StatusKit preferences: {e}"

        # Validate the file we just wrote by re-reading it. If it doesn't
        # parse, restore from backup so we don't leave the user with a
        # ControlCenter that can't read its own prefs.
        if replaced:
            try:
                with open(plist_path, "rb") as f:
                    plistlib.load(f)
            except Exception as e:
                logger.error("Post-write validation failed: %s", e)
                if backup is not None:
                    try:
                        plist_path.write_bytes(backup.read_bytes())
                        logger.warning("Restored plist from %s", backup)
                        return (
                            False,
                            "Wrote a plist macOS rejected and rolled back. "
                            "No change applied.",
                        )
                    except OSError as restore_err:
                        logger.error(
                            "Restore from backup failed: %s", restore_err
                        )
                return (
                    False,
                    "Plist post-write validation failed and the backup "
                    "could not be restored. Check "
                    "~/Library/Application Support/oMLX/backups for a "
                    "manual restore.",
                )

        try:
            subprocess.run(
                ["killall", "ControlCenter"], timeout=5, check=False
            )
        except subprocess.SubprocessError as e:
            logger.warning("killall ControlCenter failed: %s", e)
            return (
                True,
                "StatusKit flag was updated but I couldn't restart "
                "ControlCenter. Run `killall ControlCenter` manually.",
            )

        detail = (
            "appended a new com.omlx.app entry" if appended_new
            else "flipped the existing com.omlx.app entry to isAllowed=True"
        )
        return (
            True,
            f"Auto-Fix {detail} in StatusKit and restarted ControlCenter. "
            "The menubar icon should appear within a few seconds. "
            "If it still doesn't, quit and relaunch oMLX.",
        )

    # --- Icon management ---

    def _get_resources_dir(self) -> Path:
        """Get the Resources directory (bundle or development fallback)."""
        # App bundle: __file__ is Resources/omlx_app/app.py → parent.parent = Resources/
        bundle_resources = Path(__file__).parent.parent
        if (bundle_resources / "navbar-logo-dark.svg").exists():
            return bundle_resources
        # NSBundle fallback
        bundle = NSBundle.mainBundle()
        if bundle and bundle.resourcePath():
            res = Path(bundle.resourcePath())
            if (res / "navbar-logo-dark.svg").exists():
                return res
        # Development fallback: omlx/admin/static/
        dev_path = (
            Path(__file__).parent.parent.parent / "omlx" / "admin" / "static"
        )
        if dev_path.exists():
            return dev_path
        return Path(__file__).parent

    def _load_menubar_icon(self, svg_name: str) -> Optional[NSImage]:
        """Load an SVG file as a template image for the menubar.

        Template images automatically adapt to menubar background:
        - Light menubar background → dark rendering
        - Dark menubar background → light rendering
        This works even when dark mode uses a light wallpaper!
        """
        resources = self._get_resources_dir()
        svg_path = resources / svg_name
        if not svg_path.exists():
            logger.warning(f"Icon not found: {svg_path}")
            return None

        try:
            svg_data = NSData.dataWithContentsOfFile_(str(svg_path))
            if svg_data is None:
                return None
            image = NSImage.alloc().initWithData_(svg_data)
            if image:
                image.setSize_((18, 18))
                image.setTemplate_(True)  # macOS auto color adjustment
                return image
        except Exception as e:
            logger.error(f"Failed to load icon {svg_name}: {e}")
        return None

    def _is_dark_mode(self) -> bool:
        """Check if the system is in dark mode."""
        try:
            appearance = NSApp.effectiveAppearance()
            if appearance:
                best = appearance.bestMatchFromAppearancesWithNames_(
                    [NSAppearanceNameDarkAqua]
                )
                return best == NSAppearanceNameDarkAqua
        except Exception:
            pass
        return False

    def _update_menubar_icon(self):
        """Update menubar icon based on server state.

        Template images automatically adapt to menubar background color,
        so we only need to switch between outline (OFF) and filled (ON).
        """
        if self.status_item is None:
            return

        is_running = self.server_manager.status in (
            ServerStatus.RUNNING,
            ServerStatus.STARTING,
        )

        # Simple: only server state matters (theme handled by template image)
        icon = self._icon_filled if is_running else self._icon_outline

        if icon:
            button = self.status_item.button()
            if button:
                button.setImage_(icon)
            self.status_item.setTitle_("")
        else:
            # Fallback to text if icons not available
            self.status_item.setTitle_("oMLX")

    # --- Update checking ---

    def _check_for_updates(self):
        """Check GitHub Releases for new version (cached for 24 hours)."""
        now = time.time()
        if now - self._last_update_check < 86400:  # 24 hours
            return  # Use cached result

        try:
            # GitHub Releases API
            resp = requests.get(
                "https://api.github.com/repos/jundot/omlx/releases/latest",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                latest = data["tag_name"].lstrip("v")
                current = __version__

                if self._is_newer_version(latest, current):
                    # Find DMG asset matching current macOS version
                    dmg_url = _find_matching_dmg(data.get("assets", []))

                    self._update_info = {
                        "version": latest,
                        "url": data["html_url"],
                        "dmg_url": dmg_url,
                        "notes": data.get("body", ""),
                    }
                    logger.info(f"Update available: {latest}")
                    self._build_menu()
                else:
                    self._update_info = None
            else:
                self._update_info = None

            self._last_update_check = now
        except Exception as e:
            logger.debug(f"Update check failed: {e}")
            self._update_info = None

    def _is_newer_version(self, latest: str, current: str) -> bool:
        """PEP 440 version comparison. Ignores pre-release versions."""
        try:
            from packaging.version import Version

            latest_ver = Version(latest)
            return latest_ver > Version(current) and not latest_ver.is_prerelease
        except Exception:
            return False

    def openUpdate_(self, sender):
        """Show confirmation dialog and start auto-update."""
        if not self._update_info:
            return

        # If no DMG URL available, fall back to browser
        if not self._update_info.get("dmg_url"):
            self._open_update_browser()
            return

        from AppKit import NSAlert, NSAlertFirstButtonReturn

        alert = NSAlert.alloc().init()
        alert.setMessageText_(
            f"Update to oMLX {self._update_info['version']}?"
        )

        notes = self._update_info.get("notes", "")
        if len(notes) > 500:
            notes = notes[:500] + "..."
        alert.setInformativeText_(
            f"{notes}\n\n"
            "The update will be downloaded and installed automatically. "
            "The app will restart when ready."
        )
        alert.addButtonWithTitle_("Update")
        alert.addButtonWithTitle_("Cancel")

        if alert.runModal() != NSAlertFirstButtonReturn:
            return

        self._start_auto_update()

    def _open_update_browser(self):
        """Fallback: open GitHub releases page in browser."""
        url = (
            self._update_info.get("url")
            if self._update_info
            else "https://github.com/jundot/omlx/releases"
        )
        webbrowser.open(url)

    def _start_auto_update(self):
        """Begin the background download + staging process."""
        from .updater import AppUpdater

        # Check write permissions first
        app_path = AppUpdater.get_app_bundle_path()
        if not AppUpdater.is_writable(app_path):
            from AppKit import NSAlert, NSAlertFirstButtonReturn

            alert = NSAlert.alloc().init()
            alert.setMessageText_("Cannot Auto-Update")
            alert.setInformativeText_(
                f"oMLX does not have write permission to {app_path.parent}.\n\n"
                "Please download the update manually from GitHub."
            )
            alert.addButtonWithTitle_("Open GitHub")
            alert.addButtonWithTitle_("Cancel")
            if alert.runModal() == NSAlertFirstButtonReturn:
                self._open_update_browser()
            return

        self._updater = AppUpdater(
            dmg_url=self._update_info["dmg_url"],
            version=self._update_info["version"],
            on_progress=self._on_update_progress,
            on_error=self._on_update_error,
            on_ready=self._on_update_ready,
        )
        self._updater.start()
        self._build_menu()

    def _on_update_progress(self, message: str):
        """Called from background thread with progress updates."""
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "updateProgressOnMain:", message, False
        )

    def updateProgressOnMain_(self, message):
        """Main thread: rebuild menu to show download progress."""
        self._update_progress_text = message
        if not self._menu_is_open:
            self._build_menu()

    def _on_update_error(self, message: str):
        """Called from background thread on failure."""
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "updateErrorOnMain:", message, False
        )

    def updateErrorOnMain_(self, message):
        """Main thread: show error and offer browser fallback."""
        self._updater = None
        self._update_progress_text = ""
        if not self._menu_is_open:
            self._build_menu()

        from AppKit import NSAlert, NSAlertFirstButtonReturn

        alert = NSAlert.alloc().init()
        alert.setMessageText_("Update Failed")
        alert.setInformativeText_(
            f"{message}\n\n"
            "Would you like to download the update manually?"
        )
        alert.addButtonWithTitle_("Open GitHub")
        alert.addButtonWithTitle_("Cancel")
        if alert.runModal() == NSAlertFirstButtonReturn:
            self._open_update_browser()

    def _on_update_ready(self):
        """Called from background thread when staged app is ready."""
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "updateReadyOnMain:", None, False
        )

    def updateReadyOnMain_(self, _):
        """Main thread: download complete, auto-install and relaunch."""
        self._updater = None
        self._update_progress_text = "Installing update..."
        if not self._menu_is_open:
            self._build_menu()
        self._perform_update_and_relaunch()

    def _perform_update_and_relaunch(self):
        """Stop server, spawn swap script, terminate app."""
        from .updater import AppUpdater

        # Stop server gracefully
        if self.server_manager.is_running():
            self.server_manager.stop()

        # Stop health timer
        if self.health_timer:
            self.health_timer.invalidate()

        # Spawn detached swap script and terminate
        if AppUpdater.perform_swap_and_relaunch():
            NSApp.terminate_(None)
        else:
            from AppKit import NSAlert

            alert = NSAlert.alloc().init()
            alert.setMessageText_("Update Failed")
            alert.setInformativeText_(
                "Could not find the staged update. Please try again."
            )
            alert.addButtonWithTitle_("OK")
            alert.runModal()

    # --- Menu building ---

    def _create_menu_icon(self, sf_symbol: str) -> Optional[NSImage]:
        """Create a menu item icon from SF Symbol (macOS 11+).

        Returns a template image that automatically adapts to menu theme.
        """
        try:
            # macOS 11+ SF Symbols support
            if hasattr(NSImage, 'imageWithSystemSymbolName_accessibilityDescription_'):
                icon = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                    sf_symbol, None
                )
                if icon:
                    icon.setSize_((16, 16))
                    return icon

            # Fallback: try imageNamed (won't work for SF Symbols, but for custom icons)
            icon = NSImage.imageNamed_(sf_symbol)
            if icon:
                icon.setSize_((16, 16))
                return icon
        except Exception as e:
            logger.debug(f"Failed to load SF Symbol {sf_symbol}: {e}")
        return None

    def _menu_font(self) -> Optional[NSFont]:
        """Return the default menu font for measurement and rendering."""
        try:
            return NSFont.menuFontOfSize_(0.0)
        except Exception:
            return None

    def _measure_menu_text_width(self, text: str, font: Optional[NSFont]) -> float:
        """Measure menu text width in points, with a safe fallback."""
        try:
            attrs = {}
            if font is not None:
                attrs[NSFontAttributeName] = font
            attributed = NSAttributedString.alloc().initWithString_attributes_(
                text, attrs
            )
            return float(attributed.size().width)
        except Exception:
            return float(max(1, len(text)) * 7)

    def _compute_stats_tab_stop(self, entries: list[tuple[str, str]]) -> float:
        """Compute right-tab position for aligned stats rows."""
        if not entries:
            return 240.0

        font = self._menu_font()
        max_label_width = max(
            self._measure_menu_text_width(label, font) for label, _ in entries
        )
        max_value_width = max(
            self._measure_menu_text_width(value, font) for _, value in entries
        )

        gap = 16.0
        return max(200.0, max_label_width + gap + max_value_width)

    def _format_compact_count(self, value) -> tuple[str, str]:
        """Format large counts with compact units and return raw full value."""
        if value is None or isinstance(value, bool):
            return "--", "--"

        try:
            if isinstance(value, int):
                n = Decimal(value)
            else:
                s = str(value).strip().replace(",", "")
                if not s:
                    return "--", "--"
                n = Decimal(s)
        except (InvalidOperation, ValueError, TypeError):
            return "--", "--"

        is_integer = n == n.to_integral_value()
        raw_value = f"{int(n):,}" if is_integer else f"{n:,.2f}"

        abs_n = abs(n)
        units: list[tuple[str, Decimal]] = [
            ("E", Decimal("1000000000000000000")),  # 10^18
            ("P", Decimal("1000000000000000")),  # 10^15
            ("T", Decimal("1000000000000")),  # 10^12
            ("B", Decimal("1000000000")),  # 10^9
            ("M", Decimal("1000000")),  # 10^6
            ("K", Decimal("1000")),  # 10^3
        ]
        for suffix, factor in units:
            if abs_n >= factor:
                compact = (n / factor).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                return f"{compact}{suffix}", raw_value

        if is_integer:
            return str(int(n)), raw_value
        return str(n.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)), raw_value

    def _make_aligned_stats_item(
        self, label: str, value: str, tab_stop: float, tooltip: Optional[str] = None
    ) -> NSMenuItem:
        """Create one stats row with left-aligned label and right-aligned value."""
        plain_text = f"{label}: {value}"
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            plain_text, "noOp:", ""
        )
        item.setTarget_(self)
        if tooltip and tooltip != "--":
            try:
                item.setToolTip_(tooltip)
            except Exception:
                pass

        try:
            paragraph = NSMutableParagraphStyle.alloc().init()
            tab = NSTextTab.alloc().initWithType_location_(
                NSRightTabStopType, tab_stop
            )
            paragraph.setTabStops_([tab])

            attrs = {NSParagraphStyleAttributeName: paragraph}
            font = self._menu_font()
            if font is not None:
                attrs[NSFontAttributeName] = font

            attributed = NSAttributedString.alloc().initWithString_attributes_(
                f"{label}\t{value}", attrs
            )
            item.setAttributedTitle_(attributed)
        except Exception as e:
            logger.debug(f"Failed to align stats row '{plain_text}': {e}")

        return item

    def _make_centered_stats_header(self, title: str, row_width: float) -> NSMenuItem:
        """Create a centered, disabled header item for stats sections."""
        text = f"── {title} ──"
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(text, None, "")
        item.setEnabled_(False)
        header_width = max(220.0, float(row_width))

        view = NSView.alloc().initWithFrame_(((0.0, 0.0), (header_width, 20.0)))
        label = NSTextField.alloc().initWithFrame_(((0.0, 1.0), (header_width, 18.0)))
        label.setStringValue_(text)
        label.setEditable_(False)
        label.setBordered_(False)
        label.setDrawsBackground_(False)
        label.setSelectable_(False)
        label.setAlignment_(NSTextAlignmentCenter)
        label.setTextColor_(NSColor.secondaryLabelColor())
        font = self._menu_font()
        if font is not None:
            label.setFont_(font)
        view.addSubview_(label)
        item.setView_(view)
        return item

    def _get_status_display(self):
        """Return (text, color) for the current server status header."""
        status = self.server_manager.status
        if status == ServerStatus.RUNNING:
            return "● oMLX Server is running", NSColor.systemGreenColor()
        elif status == ServerStatus.STARTING:
            return "● oMLX Server is starting...", NSColor.systemOrangeColor()
        elif status == ServerStatus.STOPPING:
            return "● oMLX Server is stopping...", NSColor.systemOrangeColor()
        elif status == ServerStatus.UNRESPONSIVE:
            return "● oMLX Server is not responding", NSColor.systemOrangeColor()
        elif status == ServerStatus.ERROR:
            err = self.server_manager.error_message or "Unknown error"
            return f"● {err}", NSColor.systemRedColor()
        else:
            return "● oMLX Server is stopped", NSColor.secondaryLabelColor()

    def _build_menu(self):
        """Build the status bar menu (Docker Desktop style with icons)."""
        self.menu = NSMenu.alloc().init()
        self.menu.setAutoenablesItems_(False)
        status = self.server_manager.status
        is_running = status == ServerStatus.RUNNING

        # --- Status Header (colored dot + text) ---
        status_text, status_color = self._get_status_display()

        attributed_status = NSAttributedString.alloc().initWithString_attributes_(
            status_text, {NSForegroundColorAttributeName: status_color}
        )
        status_header = NSMenuItem.alloc().init()
        status_header.setAttributedTitle_(attributed_status)
        status_header.setEnabled_(False)
        self._status_header_item = status_header
        self.menu.addItem_(status_header)

        # --- Update Available (if newer version found) ---
        if self._update_info:
            self.menu.addItem_(NSMenuItem.separatorItem())

            if self._updater is not None:
                progress = self._update_progress_text or "Downloading..."
                update_text = f"⬇️ {progress}"
                update_action = None
            else:
                update_text = (
                    f"🔔 Update Available ({self._update_info['version']})"
                )
                update_action = "openUpdate:"

            attributed_update = (
                NSAttributedString.alloc().initWithString_attributes_(
                    update_text,
                    {NSForegroundColorAttributeName: NSColor.systemGreenColor()},
                )
            )
            update_item = NSMenuItem.alloc().init()
            update_item.setAttributedTitle_(attributed_update)
            if update_action:
                update_item.setTarget_(self)
                update_item.setAction_(update_action)
            else:
                update_item.setEnabled_(False)
            self.menu.addItem_(update_item)

        self.menu.addItem_(NSMenuItem.separatorItem())

        # --- Start/Stop/Force Restart Server ---
        # All three items are always present; setHidden_ controls visibility so
        # _refresh_menu_in_place() can toggle them without replacing the NSMenu.

        # Force Restart — visible when UNRESPONSIVE / ERROR (most important, shown first)
        restart_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Force Restart", "forceRestart:", ""
        )
        restart_item.setTarget_(self)
        restart_icon = self._create_menu_icon("arrow.clockwise.circle")
        if restart_icon:
            restart_item.setImage_(restart_icon)
        restart_item.setHidden_(
            status not in (ServerStatus.UNRESPONSIVE, ServerStatus.ERROR)
        )
        self.menu.addItem_(restart_item)
        self._restart_item = restart_item

        # Stop Server — visible when RUNNING / STARTING / STOPPING / UNRESPONSIVE
        stop_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Stop Server", "stopServer:", ""
        )
        stop_item.setTarget_(self)
        stop_icon = self._create_menu_icon("stop.circle")
        if stop_icon:
            stop_item.setImage_(stop_icon)
        stop_item.setHidden_(
            status not in (
                ServerStatus.RUNNING,
                ServerStatus.STARTING,
                ServerStatus.STOPPING,
                ServerStatus.UNRESPONSIVE,
            )
        )
        self.menu.addItem_(stop_item)
        self._stop_item = stop_item

        # Start Server — visible when STOPPED
        start_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Start Server", "startServer:", ""
        )
        start_item.setTarget_(self)
        start_icon = self._create_menu_icon("play.circle")
        if start_icon:
            start_item.setImage_(start_icon)
        start_item.setHidden_(status != ServerStatus.STOPPED)
        self.menu.addItem_(start_item)
        self._start_item = start_item

        self.menu.addItem_(NSMenuItem.separatorItem())

        # --- Serving Stats submenu ---
        stats_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Serving Stats", None, ""
        )
        stats_icon = self._create_menu_icon("chart.bar")
        if stats_icon:
            stats_item.setImage_(stats_icon)

        stats_submenu = NSMenu.alloc().init()

        if is_running and self._cached_stats:
            s = self._cached_stats
            a = self._cached_alltime_stats or {}

            session_total_display, session_total_raw = self._format_compact_count(
                s.get("total_prompt_tokens", 0)
            )
            session_cached_display, session_cached_raw = self._format_compact_count(
                s.get("total_cached_tokens", 0)
            )
            alltime_total_display, alltime_total_raw = self._format_compact_count(
                a.get("total_prompt_tokens", 0)
            )
            alltime_cached_display, alltime_cached_raw = self._format_compact_count(
                a.get("total_cached_tokens", 0)
            )
            alltime_requests_display, alltime_requests_raw = self._format_compact_count(
                a.get("total_requests", 0)
            )

            session_entries = [
                (
                    "Total Tokens Processed",
                    session_total_display,
                    session_total_raw,
                ),
                ("Cached Tokens", session_cached_display, session_cached_raw),
                ("Cache Efficiency", f"{s.get('cache_efficiency', 0):.1f}%", None),
                ("Avg PP Speed", f"{s.get('avg_prefill_tps', 0):.1f} tok/s", None),
                ("Avg TG Speed", f"{s.get('avg_generation_tps', 0):.1f} tok/s", None),
            ]
            alltime_entries = [
                (
                    "Total Tokens Processed",
                    alltime_total_display,
                    alltime_total_raw,
                ),
                ("Cached Tokens", alltime_cached_display, alltime_cached_raw),
                ("Cache Efficiency", f"{a.get('cache_efficiency', 0):.1f}%", None),
                ("Total Requests", alltime_requests_display, alltime_requests_raw),
            ]

            # One shared tab stop keeps the right value edge aligned across both sections.
            shared_tab_stop = self._compute_stats_tab_stop(
                [(label, value) for label, value, _ in (session_entries + alltime_entries)]
            )
            header_row_width = shared_tab_stop + 28.0

            # Session stats
            session_header = self._make_centered_stats_header(
                "Session", header_row_width
            )
            stats_submenu.addItem_(session_header)
            for label, value, tooltip in session_entries:
                stats_submenu.addItem_(
                    self._make_aligned_stats_item(
                        label, value, shared_tab_stop, tooltip=tooltip
                    )
                )

            # All-time stats
            stats_submenu.addItem_(NSMenuItem.separatorItem())
            alltime_header = self._make_centered_stats_header(
                "All-Time", header_row_width
            )
            stats_submenu.addItem_(alltime_header)
            for label, value, tooltip in alltime_entries:
                stats_submenu.addItem_(
                    self._make_aligned_stats_item(
                        label, value, shared_tab_stop, tooltip=tooltip
                    )
                )
        else:
            off_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Server is off" if not is_running else "Loading stats...",
                None,
                "",
            )
            off_item.setEnabled_(False)
            stats_submenu.addItem_(off_item)

        stats_item.setSubmenu_(stats_submenu)
        self.menu.addItem_(stats_item)

        self.menu.addItem_(NSMenuItem.separatorItem())

        # --- Admin Panel ---
        dash_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Admin Panel", "openDashboard:", ""
        )
        dash_item.setTarget_(self)

        dash_icon = self._create_menu_icon("globe")
        if dash_icon:
            if not is_running:
                dash_icon.setTemplate_(True)  # Template + disabled = gray
            dash_item.setImage_(dash_icon)
        dash_item.setEnabled_(is_running)
        self._admin_panel_item = dash_item

        self.menu.addItem_(dash_item)

        # --- Chat with oMLX ---
        chat_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Chat with oMLX", "openChat:", ""
        )
        chat_item.setTarget_(self)

        chat_icon = self._create_menu_icon("message")
        if chat_icon:
            if not is_running:
                chat_icon.setTemplate_(True)  # Template + disabled = gray
            chat_item.setImage_(chat_icon)
        chat_item.setEnabled_(is_running)
        self._chat_item = chat_item

        self.menu.addItem_(chat_item)

        self.menu.addItem_(NSMenuItem.separatorItem())

        # --- Settings ---
        prefs_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Settings…", "openPreferences:", ","
        )
        prefs_item.setTarget_(self)
        prefs_icon = self._create_menu_icon("gearshape")
        if prefs_icon:
            prefs_item.setImage_(prefs_icon)
        self.menu.addItem_(prefs_item)

        # --- About ---
        about_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "About oMLX", "showAbout:", ""
        )
        about_item.setTarget_(self)
        about_icon = self._create_menu_icon("info.circle")
        if about_icon:
            about_item.setImage_(about_icon)
        self.menu.addItem_(about_item)

        self.menu.addItem_(NSMenuItem.separatorItem())

        # --- Quit ---
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit oMLX", "quitApp:", "q"
        )
        quit_item.setTarget_(self)
        quit_icon = self._create_menu_icon("power")
        if quit_icon:
            quit_item.setImage_(quit_icon)
        self.menu.addItem_(quit_item)

        self.status_item.setMenu_(self.menu)
        self.menu.setDelegate_(self)

    def _update_status_display(self):
        """Update the menubar icon and rebuild menu."""
        self._update_menubar_icon()
        self._build_menu()

    def _refresh_menu_in_place(self):
        """Update key menu items in-place without replacing the NSMenu object.

        Safe to call while the menu is open (used by healthCheck_ and
        menuWillOpen_ to avoid replacing a live NSMenu).
        """
        if self._status_header_item is None:
            return  # Menu not yet built

        status = self.server_manager.status
        is_running = status == ServerStatus.RUNNING

        # Update status header color and text
        text, color = self._get_status_display()

        self._status_header_item.setAttributedTitle_(
            NSAttributedString.alloc().initWithString_attributes_(
                text, {NSForegroundColorAttributeName: color}
            )
        )

        # Toggle server-control item visibility
        if self._stop_item:
            self._stop_item.setHidden_(
                status not in (
                    ServerStatus.RUNNING,
                    ServerStatus.STARTING,
                    ServerStatus.STOPPING,
                    ServerStatus.UNRESPONSIVE,
                )
            )
        if self._restart_item:
            self._restart_item.setHidden_(
                status not in (ServerStatus.UNRESPONSIVE, ServerStatus.ERROR)
            )
        if self._start_item:
            self._start_item.setHidden_(status != ServerStatus.STOPPED)

        # Toggle Admin Panel / Chat enabled state and keep icon template in sync
        if self._admin_panel_item:
            self._admin_panel_item.setEnabled_(is_running)
            icon = self._admin_panel_item.image()
            if icon:
                icon.setTemplate_(True)
        if self._chat_item:
            self._chat_item.setEnabled_(is_running)
            icon = self._chat_item.image()
            if icon:
                icon.setTemplate_(True)

    # --- NSMenuDelegate ---

    def menuWillOpen_(self, menu):
        """Refresh menu content right before it appears to the user."""
        self._menu_is_open = True
        self._refresh_menu_in_place()
        self._update_menubar_icon()

    def menuDidClose_(self, menu):
        """Track that the menu is no longer visible."""
        self._menu_is_open = False

    # --- Stats fetching ---

    def _fetch_stats(self):
        """Fetch serving stats from the admin API.

        Reuses a persistent session to avoid re-login on every poll cycle.
        Only calls /admin/api/login when the session cookie is missing or
        expired (server returns 401).
        """
        try:
            api_key = self.config.get_server_api_key()
            base_url = resolve_local_server_base_url(
                self.config.get_server_bind_host(), self.config.port
            )

            if not api_key:
                self._cached_stats = None
                self._cached_alltime_stats = None
                return

            if self._admin_session is None:
                self._admin_session = requests.Session()
                self._admin_session.trust_env = False

            session = self._admin_session

            # Try fetching stats directly (session cookie may still be valid)
            stats_resp = session.get(
                f"{base_url}/admin/api/stats",
                timeout=2,
            )

            # Session expired or missing — login and retry
            if stats_resp.status_code == 401:
                login_resp = session.post(
                    f"{base_url}/admin/api/login",
                    json={"api_key": api_key},
                    timeout=2,
                )
                if login_resp.status_code != 200:
                    self._cached_stats = None
                    self._cached_alltime_stats = None
                    self._admin_session = None
                    return

                stats_resp = session.get(
                    f"{base_url}/admin/api/stats",
                    timeout=2,
                )

            if stats_resp.status_code == 200:
                self._cached_stats = stats_resp.json()
            else:
                self._cached_stats = None
                self._cached_alltime_stats = None
                return

            alltime_resp = session.get(
                f"{base_url}/admin/api/stats",
                params={"scope": "alltime"},
                timeout=2,
            )
            if alltime_resp.status_code == 200:
                self._cached_alltime_stats = alltime_resp.json()
            else:
                self._cached_alltime_stats = None

        except requests.RequestException:
            self._cached_stats = None
            self._cached_alltime_stats = None
            self._admin_session = None

    # --- Timer callback ---

    def healthCheck_(self, timer):
        """Periodic icon/menu update and stats refresh.

        Crash detection and auto-restart are handled by
        ServerManager._health_check_loop in a background thread.
        This timer only refreshes the UI.
        """
        prev_status = self.server_manager.status

        if self.server_manager.status == ServerStatus.RUNNING:
            # Refresh stats periodically — skip blocking HTTP when menu is open
            now = time.time()
            if now - self._last_stats_fetch >= 5:
                if self._menu_is_open:
                    # Menu is tracking on main thread; avoid sync HTTP (up to 6s).
                    # In-place refresh only; fetch will run after menu closes.
                    self._refresh_menu_in_place()
                else:
                    self._fetch_stats()
                    self._last_stats_fetch = now
                    self._build_menu()

        elif self.server_manager.status in (
            ServerStatus.ERROR,
            ServerStatus.UNRESPONSIVE,
        ):
            self._cached_stats = None
            self._cached_alltime_stats = None

        # Update icon/menu if status changed
        if self.server_manager.status != prev_status:
            if self._menu_is_open:
                self._refresh_menu_in_place()
            else:
                self._update_status_display()

        # Always refresh icon in case theme changed
        self._update_menubar_icon()

    # --- Menu actions ---

    def _handle_port_conflict(self, conflict: PortConflict) -> None:
        """Show a dialog for port conflicts and handle user choice."""
        from AppKit import NSAlert, NSAlertFirstButtonReturn, NSAlertSecondButtonReturn

        alert = NSAlert.alloc().init()

        if conflict.is_omlx:
            alert.setMessageText_("oMLX Server Already Running")
            pid_info = f" (PID {conflict.pid})" if conflict.pid else ""
            alert.setInformativeText_(
                f"An oMLX server is already running on port "
                f"{self.server_manager.config.port}{pid_info}.\n\n"
                f"You can adopt it (monitor without restarting) "
                f"or kill it and start a new one."
            )
            alert.addButtonWithTitle_("Adopt")
            alert.addButtonWithTitle_("Kill & Restart")
            alert.addButtonWithTitle_("Cancel")

            response = alert.runModal()
            if response == NSAlertFirstButtonReturn:
                if not self.server_manager.adopt():
                    self.server_manager._update_status(
                        ServerStatus.ERROR, "Failed to adopt — server may have stopped"
                    )
            elif response == NSAlertSecondButtonReturn:
                if conflict.pid:
                    self.server_manager._kill_external_server(conflict.pid)
                    import time
                    time.sleep(0.5)
                result = self.server_manager.start()
                if isinstance(result, PortConflict):
                    self.server_manager._update_status(
                        ServerStatus.ERROR, "Port still in use after kill"
                    )
            # Cancel: do nothing
        else:
            alert.setMessageText_(f"Port {self.server_manager.config.port} In Use")
            pid_info = f" (PID {conflict.pid})" if conflict.pid else ""
            alert.setInformativeText_(
                f"Port {self.server_manager.config.port} is in use by another "
                f"application{pid_info}.\n\n"
                f"Change the port in Settings."
            )
            alert.addButtonWithTitle_("Open Settings")
            alert.addButtonWithTitle_("Cancel")

            response = alert.runModal()
            if response == NSAlertFirstButtonReturn:
                self.openPreferences_(None)

        self._update_status_display()

    @objc.IBAction
    def startServer_(self, sender):
        """Start the server."""
        result = self.server_manager.start()
        if isinstance(result, PortConflict):
            self._handle_port_conflict(result)
            return
        self._update_status_display()

    @objc.IBAction
    def stopServer_(self, sender):
        """Stop the server."""
        self.server_manager.stop()
        self._cached_stats = None
        self._cached_alltime_stats = None
        self._admin_session = None
        self._update_status_display()

    @objc.IBAction
    def forceRestart_(self, sender):
        """Force restart the server (kill + start fresh)."""
        self._admin_session = None
        result = self.server_manager.force_restart()
        if isinstance(result, PortConflict):
            self._handle_port_conflict(result)
            return
        self._update_status_display()

    @objc.IBAction
    def noOp_(self, sender):
        """No-op action for display-only menu items."""
        pass

    def _open_with_auto_login(self, redirect_path: str):
        """Open a browser with auto-login to the admin panel.

        Args:
            redirect_path: The admin path to redirect to (e.g., "/admin/dashboard").
        """
        if self.server_manager.status != ServerStatus.RUNNING:
            return

        base_url = resolve_local_server_base_url(
            self.config.get_server_bind_host(), self.config.port
        )
        api_key = self.config.get_server_api_key()

        if api_key:
            from urllib.parse import quote

            webbrowser.open(
                f"{base_url}/admin/auto-login"
                f"?key={quote(api_key, safe='')}&redirect={quote(redirect_path, safe='/')}"
            )
        else:
            webbrowser.open(f"{base_url}{redirect_path}")

    @objc.IBAction
    def openDashboard_(self, sender):
        """Open admin dashboard in the default browser."""
        self._open_with_auto_login("/admin/dashboard")

    @objc.IBAction
    def openChat_(self, sender):
        """Open chat page in the default browser."""
        self._open_with_auto_login("/admin/chat")

    @objc.IBAction
    def openPreferences_(self, sender):
        """Open the Settings window."""
        from .preferences import PreferencesWindowController

        self.preferences_controller = (
            PreferencesWindowController.alloc().initWithConfig_serverManager_onSave_(
                self.config, self.server_manager, self._on_prefs_saved
            )
        )
        self.preferences_controller.show_welcome = self._show_welcome
        self.preferences_controller.showWindow()

    def _show_welcome(self):
        """Show the welcome window (called from preferences)."""
        from .welcome import WelcomeWindowController

        self.welcome_controller = (
            WelcomeWindowController.alloc().initWithConfig_serverManager_(
                self.config, self.server_manager
            )
        )
        self.welcome_controller.showWindow()

    def _on_prefs_saved(self):
        """Callback after settings are saved."""
        self.server_manager.update_config(self.config)
        self._build_menu()

    @objc.IBAction
    def showAbout_(self, sender):
        """Show the standard macOS About panel with a clickable GitHub link.

        Using orderFrontStandardAboutPanelWithOptions_ gives the centered
        Aqua layout that matches every other Mac app and sidesteps NSAlert's
        left-aligned icon-plus-text rendering. The GitHub URL is embedded as
        a real NSLinkAttributeName in the Credits NSAttributedString, so
        AppKit renders it as a clickable hyperlink.
        """
        try:
            from omlx._build_info import build_number
        except ImportError:
            build_number = None

        github_url = "https://github.com/jundot/omlx"
        # Put the build number at the top of Credits with a newline so it
        # renders on its own line. The standard About panel keeps
        # ApplicationVersion on a single line and doesn't respect \n
        # inside it, but Credits accepts NSAttributedString with real line
        # breaks.
        if build_number:
            credits_text = (
                f"({build_number})\n\n"
                "LLM inference, optimized for your Mac\n\n"
                "Built with MLX, mlx-lm, and mlx-vlm\n"
                "Special Thanks to 1212.H.\n\n"
                f"{github_url}"
            )
        else:
            credits_text = (
                "LLM inference, optimized for your Mac\n\n"
                "Built with MLX, mlx-lm, and mlx-vlm\n"
                "Special Thanks to 1212.H.\n\n"
                f"{github_url}"
            )
        credits = NSMutableAttributedString.alloc().initWithString_(credits_text)

        # Center the whole credits block to match the panel's header alignment.
        paragraph = NSMutableParagraphStyle.alloc().init()
        paragraph.setAlignment_(NSTextAlignmentCenter)
        credits.addAttribute_value_range_(
            NSParagraphStyleAttributeName,
            paragraph,
            (0, credits.length()),
        )

        # Embed the URL as a link attribute so clicking opens the browser.
        loc = credits_text.find(github_url)
        if loc >= 0:
            credits.addAttribute_value_range_(
                NSLinkAttributeName,
                NSURL.URLWithString_(github_url),
                (loc, len(github_url)),
            )

        options = {
            "ApplicationName": "oMLX",
            "ApplicationVersion": __version__,
            "Credits": credits,
        }

        NSApp.activateIgnoringOtherApps_(True)
        NSApplication.sharedApplication().orderFrontStandardAboutPanelWithOptions_(
            options
        )

    @objc.IBAction
    def quitApp_(self, sender):
        """Quit the application."""
        if self.health_timer:
            self.health_timer.invalidate()

        if self.server_manager.is_running():
            self.server_manager.stop()

        NSApp.terminate_(None)


def main():
    """Run the menubar application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from PyObjCTools import AppHelper

    app = NSApplication.sharedApplication()
    # Set Regular policy first so macOS grants full GUI access on launch,
    # then switch to Accessory in applicationDidFinishLaunching_ after
    # the status bar item is created. This ensures the menubar icon is
    # visible on macOS Tahoe where Accessory apps launched via
    # LaunchServices may remain in "NotVisible" state.
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    delegate = OMLXAppDelegate.alloc().init()
    app.setDelegate_(delegate)
    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
