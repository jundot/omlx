"""
oMLX Native Menubar Application using PyObjC.

A native macOS menubar app for managing the oMLX LLM inference server.
"""

import logging
import time
import webbrowser
from pathlib import Path
from typing import Optional

import objc
import requests

from omlx._version import __version__
from AppKit import (
    NSApp,
    NSAppearanceNameDarkAqua,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSApplicationActivationPolicyRegular,
    NSAttributedString,
    NSBundle,
    NSColor,
    NSForegroundColorAttributeName,
    NSImage,
    NSMenu,
    NSMenuItem,
    NSStatusBar,
    NSVariableStatusItemLength,
)
from Foundation import NSData, NSObject, NSRunLoop, NSDefaultRunLoopMode, NSTimer

from .config import ServerConfig
from .server_manager import PortConflict, ServerManager, ServerStatus

logger = logging.getLogger(__name__)


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
        self._last_stats_fetch: float = 0
        self._icon_outline: Optional[NSImage] = None
        self._icon_filled: Optional[NSImage] = None
        self._update_info: Optional[dict] = None
        self._last_update_check: float = 0

        return self

    def applicationDidFinishLaunching_(self, notification):
        """Called when app finishes launching."""
        try:
            self._doFinishLaunching()
        except Exception as e:
            logger.error(f"Launch failed: {e}", exc_info=True)
            self._show_fatal_error_and_quit(str(e))

    def _show_fatal_error_and_quit(self, message: str):
        """Show a fatal error dialog and terminate the application."""
        from AppKit import NSAlert

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
            self.health_timer, NSDefaultRunLoopMode
        )

        # Switch from Regular to Accessory policy now that the status bar
        # item exists. This hides the Dock icon while keeping the menubar item.
        # We start as Regular (in main()) so macOS grants full GUI access,
        # then switch here — required on macOS Tahoe where Accessory apps
        # launched via LaunchServices remain "NotVisible" otherwise.
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        NSApp.activateIgnoringOtherApps_(True)

        logger.info("oMLX menubar app launched successfully")

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
                    self._update_info = {
                        "version": latest,
                        "url": data["html_url"],
                        "notes": data.get("body", ""),
                    }
                    logger.info(f"Update available: {latest}")
                else:
                    self._update_info = None
            else:
                self._update_info = None

            self._last_update_check = now
        except Exception as e:
            logger.debug(f"Update check failed: {e}")
            self._update_info = None

    def _is_newer_version(self, latest: str, current: str) -> bool:
        """Simple semantic version comparison."""
        try:
            latest_parts = [int(x) for x in latest.split(".")[:3]]
            current_parts = [int(x) for x in current.split(".")[:3]]
            return latest_parts > current_parts
        except Exception:
            return False

    def openUpdate_(self, sender):
        """Open GitHub releases page when update menu item is clicked."""
        url = (
            self._update_info.get("url")
            if self._update_info
            else "https://github.com/jundot/omlx/releases"
        )
        webbrowser.open(url)

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

    def _build_menu(self):
        """Build the status bar menu (Docker Desktop style with icons)."""
        self.menu = NSMenu.alloc().init()
        self.menu.setAutoenablesItems_(False)
        status = self.server_manager.status
        is_running = status == ServerStatus.RUNNING

        # --- Status Header (colored dot + text) ---
        if status == ServerStatus.RUNNING:
            status_text = "● oMLX Server is running"
            status_color = NSColor.systemGreenColor()
        elif status == ServerStatus.STARTING:
            status_text = "● oMLX Server is starting..."
            status_color = NSColor.systemOrangeColor()
        elif status == ServerStatus.ERROR:
            status_text = "● oMLX Server error"
            status_color = NSColor.systemRedColor()
        else:
            status_text = "● oMLX Server is stopped"
            status_color = NSColor.secondaryLabelColor()

        attributed_status = NSAttributedString.alloc().initWithString_attributes_(
            status_text, {NSForegroundColorAttributeName: status_color}
        )
        status_header = NSMenuItem.alloc().init()
        status_header.setAttributedTitle_(attributed_status)
        status_header.setEnabled_(False)
        self.menu.addItem_(status_header)

        # --- Update Available (if newer version found) ---
        if self._update_info:
            self.menu.addItem_(NSMenuItem.separatorItem())
            update_text = f"🔔 Update Available ({self._update_info['version']})"
            attributed_update = NSAttributedString.alloc().initWithString_attributes_(
                update_text, {NSForegroundColorAttributeName: NSColor.systemGreenColor()}
            )
            update_item = NSMenuItem.alloc().init()
            update_item.setAttributedTitle_(attributed_update)
            update_item.setTarget_(self)
            update_item.setAction_("openUpdate:")
            self.menu.addItem_(update_item)

        self.menu.addItem_(NSMenuItem.separatorItem())

        # --- Start/Stop Server ---
        if status in (ServerStatus.RUNNING, ServerStatus.STARTING):
            stop_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Stop Server", "stopServer:", ""
            )
            stop_item.setTarget_(self)
            stop_icon = self._create_menu_icon("stop.circle")
            if stop_icon:
                stop_item.setImage_(stop_icon)
            self.menu.addItem_(stop_item)
        else:
            start_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Start Server", "startServer:", ""
            )
            start_item.setTarget_(self)
            start_icon = self._create_menu_icon("play.circle")
            if start_icon:
                start_item.setImage_(start_icon)
            self.menu.addItem_(start_item)

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
            entries = [
                ("Total Tokens Processed", f"{s.get('total_prompt_tokens', 0):,}"),
                ("Cached Tokens", f"{s.get('total_cached_tokens', 0):,}"),
                ("Cache Efficiency", f"{s.get('cache_efficiency', 0):.1f}%"),
                ("Avg PP Speed", f"{s.get('avg_prefill_tps', 0):.1f} tok/s"),
                ("Avg TG Speed", f"{s.get('avg_generation_tps', 0):.1f} tok/s"),
            ]
            for label, value in entries:
                text = f"{label}: {value}"
                mi = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    text, "noOp:", ""
                )
                mi.setTarget_(self)
                stats_submenu.addItem_(mi)
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

        self.menu.addItem_(chat_item)

        self.menu.addItem_(NSMenuItem.separatorItem())

        # --- Preferences ---
        prefs_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Preferences...", "openPreferences:", ","
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

    def _update_status_display(self):
        """Update the menubar icon and rebuild menu."""
        self._update_menubar_icon()
        self._build_menu()

    # --- Stats fetching ---

    def _fetch_stats(self):
        """Fetch serving stats from the admin API."""
        try:
            api_key = self.config.get_server_api_key()
            base_url = f"http://127.0.0.1:{self.config.port}"

            if not api_key:
                self._cached_stats = None
                return

            session = requests.Session()
            login_resp = session.post(
                f"{base_url}/admin/api/login",
                json={"api_key": api_key},
                timeout=2,
            )
            if login_resp.status_code != 200:
                self._cached_stats = None
                return

            stats_resp = session.get(
                f"{base_url}/admin/api/stats",
                timeout=2,
            )
            if stats_resp.status_code == 200:
                self._cached_stats = stats_resp.json()
            else:
                self._cached_stats = None

        except requests.RequestException:
            self._cached_stats = None

    # --- Timer callback ---

    def healthCheck_(self, timer):
        """Periodic health check, stats refresh, and icon update."""
        prev_status = self.server_manager.status

        if self.server_manager.status == ServerStatus.RUNNING:
            if not self.server_manager.check_health():
                if not self.server_manager.is_running():
                    self.server_manager._update_status(
                        ServerStatus.ERROR, "Server process terminated"
                    )
                    self._cached_stats = None
                    self._update_status_display()
            else:
                # Refresh stats
                now = time.time()
                if now - self._last_stats_fetch >= 5:
                    self._fetch_stats()
                    self._last_stats_fetch = now
                    self._build_menu()

        elif self.server_manager.status == ServerStatus.STARTING:
            if self.server_manager.check_health():
                self.server_manager._update_status(ServerStatus.RUNNING)
                self._update_status_display()

        # Update icon if status changed
        if self.server_manager.status != prev_status:
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
                f"Change the port in Preferences."
            )
            alert.addButtonWithTitle_("Open Preferences")
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
        """Open the Preferences window."""
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
        """Callback after preferences are saved."""
        self.server_manager.update_config(self.config)
        self._build_menu()

    @objc.IBAction
    def showAbout_(self, sender):
        """Show About dialog."""
        import webbrowser

        from AppKit import NSAlert, NSAlertFirstButtonReturn

        alert = NSAlert.alloc().init()
        alert.setMessageText_("About oMLX")
        alert.setInformativeText_(
            "oMLX - LLM inference,\noptimized for your Mac\n\n"
            "Built with MLX and mlx-lm\n"
            f"Version: {__version__}"
        )
        alert.addButtonWithTitle_("OK")
        alert.addButtonWithTitle_("GitHub")

        if alert.runModal() != NSAlertFirstButtonReturn:
            webbrowser.open("https://github.com/jundot")

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
