"""
oMLX Native Menubar Application using PyObjC.

A native macOS menubar app for managing the oMLX LLM inference server.
"""

import logging
import platform
import time
import webbrowser
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

from .config import ServerConfig
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

        # Create status bar item
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength
        )
        # Stable identity for ControlCenter so it persists visibility prefs
        # across app relaunches and distinguishes from previously blocked items.
        self.status_item.setAutosaveName_("com.omlx.app-statusItem")
        self._update_menubar_icon()

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
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        NSApp.activateIgnoringOtherApps_(True)

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

    def _is_status_item_hidden(self) -> bool:
        """Detect whether the menubar icon is actually rendered.

        There's no single reliable signal on macOS Tahoe, so probe several:

        - NSStatusItem.isVisible(): app-side setVisible: flag only. Stays True
          when ControlCenter/Menu Bar settings hide the item, so it alone
          can't catch Tahoe's toggle-off.
        - button.window().isVisible: NSWindow's own "hooked to the screen"
          flag. On a hidden status item this tends to flip False even when
          the app hasn't touched anything.
        - button.window().occlusionState: finer-grained visibility bitmask.
          The NSWindowOcclusionStateVisible bit (1<<1) is what we look for.
        - frame: mostly diagnostic. The size/position is typically preserved
          even when hidden (autosaveName persists Preferred Position), so
          it's weak for detection but useful in logs.

        Treat the item as hidden if ANY of the strong signals say hidden.
        Always log the raw probe so `omlx diagnose menubar` can surface it.
        """
        NS_WINDOW_OCCLUSION_STATE_VISIBLE = 1 << 1  # NSWindowOcclusionStateVisible

        button = self.status_item.button() if self.status_item else None
        window = button.window() if button else None
        frame = window.frame() if window else None
        api_visible = bool(self.status_item and self.status_item.isVisible())
        window_visible = bool(window and window.isVisible())
        occlusion = int(window.occlusionState()) if window else 0
        occlusion_visible = bool(occlusion & NS_WINDOW_OCCLUSION_STATE_VISIBLE)

        frame_str = (
            f"({frame.origin.x:.1f},{frame.origin.y:.1f},"
            f"{frame.size.width:.1f}x{frame.size.height:.1f})"
            if frame
            else None
        )
        logger.info(
            "menubar visibility probe: isVisible=%s window.isVisible=%s "
            "occlusion=0x%x(visible=%s) button=%s window=%s frame=%s",
            api_visible,
            window_visible,
            occlusion,
            occlusion_visible,
            bool(button),
            bool(window),
            frame_str,
        )

        if not button or not window:
            return True
        if not api_visible:
            return True
        # If the NSWindow is not visible or not marked occlusion-visible, the
        # icon isn't reaching the menubar even if frame numbers look normal.
        if not window_visible:
            return True
        if not occlusion_visible:
            return True
        return False

    def checkStatusItemVisibility_(self, timer):
        """One-shot post-launch check for menubar icon visibility."""
        if self._is_status_item_hidden():
            logger.warning(
                "NSStatusItem appears hidden after launch — likely blocked by "
                "ControlCenter or disabled in System Settings > Menu Bar."
            )
            self._show_menubar_hidden_alert()

    def _show_menubar_hidden_alert(self):
        """Inform the user about the hidden menubar icon and offer recovery.

        Tahoe (26.x) adds a dedicated Menu Bar settings pane with per-app
        toggles, so the alert deep-links there. Earlier versions of macOS
        have no System Settings UI for third-party status items — the only
        recovery is restarting oMLX (or checking Bartender/Ice style tools
        if the user has them) — so on Sequoia and older we drop the
        Settings button entirely to avoid pointing users at a dead end.
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

        if is_tahoe_or_newer:
            alert.setInformativeText_(
                "The oMLX menubar icon isn't showing up.\n\n"
                "macOS may be hiding it, or oMLX has been toggled off in "
                "System Settings > Menu Bar.\n\n"
                f"Click \"{settings_label}\" to check, or \"View Log\" to "
                "see what the app detected."
            )
            alert.addButtonWithTitle_(settings_label)  # 1000
            alert.addButtonWithTitle_("View Log")      # 1001
            alert.addButtonWithTitle_("Dismiss")       # 1002
        else:
            alert.setInformativeText_(
                "The oMLX menubar icon isn't showing up.\n\n"
                "macOS before Tahoe doesn't offer a System Settings toggle "
                "for third-party menubar apps. Try quitting and relaunching "
                "oMLX, and check menubar manager tools like Bartender or "
                "Ice if you use them.\n\n"
                "Click \"View Log\" to see what the app detected."
            )
            alert.addButtonWithTitle_("View Log")      # 1000
            alert.addButtonWithTitle_("Dismiss")       # 1001

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
                NSWorkspace.sharedWorkspace().openURL_(
                    NSURL.URLWithString_(settings_url)
                )
            elif response == NSAlertSecondButtonReturn:
                NSWorkspace.sharedWorkspace().openURL_(
                    NSURL.fileURLWithPath_(str(log_path))
                )
        else:
            if response == NSAlertFirstButtonReturn:
                NSWorkspace.sharedWorkspace().openURL_(
                    NSURL.fileURLWithPath_(str(log_path))
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
            base_url = f"http://127.0.0.1:{self.config.port}"

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

        # Catch runtime changes: user toggles oMLX off in System Settings
        # after the 3s one-shot has already fired. Warn once per session.
        if not self._warned_hidden and self._is_status_item_hidden():
            logger.warning(
                "NSStatusItem turned hidden at runtime — user likely toggled "
                "oMLX off in System Settings > Menu Bar."
            )
            self._show_menubar_hidden_alert()

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

        base_url = f"http://127.0.0.1:{self.config.port}"
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
        if build_number:
            options["Version"] = str(build_number)

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
