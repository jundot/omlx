"""Welcome screen for first-run experience - modern macOS native design."""

import logging
import webbrowser
from pathlib import Path
from typing import Optional

import objc
from AppKit import (
    NSApp,
    NSBackingStoreBuffered,
    NSBezelStyleRounded,
    NSBox,
    NSBoxCustom,
    NSBoxSeparator,
    NSButton,
    NSCenterTextAlignment,
    NSColor,
    NSFont,
    NSImage,
    NSImageScaleProportionallyUpOrDown,
    NSImageView,
    NSMakeRect,
    NSOpenPanel,
    NSSecureTextField,
    NSTextField,
    NSView,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectMaterialSidebar,
    NSVisualEffectView,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSData, NSObject

from .server_manager import PortConflict

logger = logging.getLogger(__name__)

WINDOW_WIDTH = 540
WINDOW_HEIGHT = 676


class WelcomeWindowController(NSObject):
    """Controller for the 3-step welcome wizard - modern macOS design."""

    def initWithConfig_serverManager_(self, config, server_manager):
        self = objc.super(WelcomeWindowController, self).init()
        if self is None:
            return None
        self.config = config
        self.server_manager = server_manager
        self.window = None
        self.base_path_label = None
        self.model_dir_label = None
        self.port_field = None
        self.api_key_secure = None
        self.api_key_plain = None
        self._api_key_visible = False
        self._eye_btn = None
        self.status_label = None
        self.start_button = None
        self.status_check_timer = None
        return self

    def _get_logo_path(self) -> Optional[Path]:
        """Find the logo SVG file."""
        svg_name = "navbar-logo-dark.svg"
        bundle_res = Path(__file__).parent.parent
        logo = bundle_res / svg_name
        if logo.exists():
            return logo
        dev = (
            Path(__file__).parent.parent.parent / "omlx" / "admin" / "static" / svg_name
        )
        if dev.exists():
            return dev
        return None

    def _create_card(self) -> NSBox:
        """Create a modern card-style box."""
        box = NSBox.alloc().init()
        box.setBoxType_(NSBoxCustom)
        box.setTransparent_(False)
        box.setFillColor_(NSColor.controlBackgroundColor())
        box.setCornerRadius_(10)
        box.setBorderWidth_(0)
        return box

    def _create_separator(self) -> NSBox:
        """Create a 1px separator line."""
        sep = NSBox.alloc().init()
        sep.setBoxType_(NSBoxSeparator)
        return sep

    def showWindow(self):
        """Create and show the welcome window."""
        # Sync from server settings.json (Web admin changes)
        self.config.sync_from_server_settings()

        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        frame = NSMakeRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Welcome to oMLX")
        self.window.center()
        self.window.setReleasedWhenClosed_(False)

        # Visual effect background layer
        effect_view = NSVisualEffectView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        effect_view.setMaterial_(NSVisualEffectMaterialSidebar)
        effect_view.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        self.window.setContentView_(effect_view)

        # Content container on top of blur
        container = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        effect_view.addSubview_(container)

        y = WINDOW_HEIGHT - 20

        # === Logo ===
        logo_path = self._get_logo_path()
        if logo_path:
            try:
                svg_data = NSData.dataWithContentsOfFile_(str(logo_path))
                if svg_data:
                    logo_image = NSImage.alloc().initWithData_(svg_data)
                    if logo_image:
                        y -= 64
                        logo_view = NSImageView.alloc().initWithFrame_(
                            NSMakeRect((WINDOW_WIDTH - 56) // 2, y, 56, 56)
                        )
                        logo_view.setImage_(logo_image)
                        logo_view.setImageScaling_(NSImageScaleProportionallyUpOrDown)
                        container.addSubview_(logo_view)
            except Exception:
                pass

        # === Title ===
        y -= 34
        title = NSTextField.labelWithString_("Welcome to oMLX")
        title.setFont_(NSFont.systemFontOfSize_weight_(24, 0.5))
        title.setAlignment_(NSCenterTextAlignment)
        title.setFrame_(NSMakeRect(40, y, WINDOW_WIDTH - 80, 32))
        container.addSubview_(title)

        # Subtitle
        y -= 22
        subtitle = NSTextField.labelWithString_(
            "LLM inference, optimized for your Mac"
        )
        subtitle.setFont_(NSFont.systemFontOfSize_(13))
        subtitle.setTextColor_(NSColor.secondaryLabelColor())
        subtitle.setAlignment_(NSCenterTextAlignment)
        subtitle.setFrame_(NSMakeRect(40, y, WINDOW_WIDTH - 80, 18))
        container.addSubview_(subtitle)

        # === Step 1: Configure Card ===
        y -= 32
        step1_card = self._create_card()
        step1_card.setFrame_(NSMakeRect(24, y - 226, WINDOW_WIDTH - 48, 226))
        container.addSubview_(step1_card)

        step1_content = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WINDOW_WIDTH - 48, 226)
        )
        step1_card.setContentView_(step1_content)

        cy = 226 - 16

        # Header
        cy -= 20
        step1_header = NSTextField.labelWithString_("Step 1  Initial Configuration")
        step1_header.setFont_(NSFont.systemFontOfSize_weight_(14, 0.65))
        step1_header.setTextColor_(NSColor.labelColor())
        step1_header.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 20))
        step1_content.addSubview_(step1_header)

        # Base Directory
        cy -= 30
        bp_label = NSTextField.labelWithString_("Base Directory")
        bp_label.setFont_(NSFont.systemFontOfSize_(12))
        bp_label.setFrame_(NSMakeRect(16, cy, 110, 18))
        step1_content.addSubview_(bp_label)

        self.base_path_label = NSTextField.labelWithString_(self.config.base_path)
        self.base_path_label.setFont_(NSFont.monospacedSystemFontOfSize_weight_(10, 0))
        self.base_path_label.setTextColor_(NSColor.secondaryLabelColor())
        self.base_path_label.setLineBreakMode_(5)
        self.base_path_label.setFrame_(NSMakeRect(130, cy, WINDOW_WIDTH - 278, 18))
        step1_content.addSubview_(self.base_path_label)

        browse_base_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(WINDOW_WIDTH - 160, cy - 2, 80, 22)
        )
        browse_base_btn.setTitle_("Browse...")
        browse_base_btn.setBezelStyle_(NSBezelStyleRounded)
        browse_base_btn.setFont_(NSFont.systemFontOfSize_(11))
        browse_base_btn.setTarget_(self)
        browse_base_btn.setAction_(objc.selector(self.browseBaseDir_, signature=b"v@:@"))
        step1_content.addSubview_(browse_base_btn)

        # Separator
        cy -= 16
        sep1 = self._create_separator()
        sep1.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 1))
        step1_content.addSubview_(sep1)

        # Model Directory
        cy -= 30
        md_label = NSTextField.labelWithString_("Model Directory")
        md_label.setFont_(NSFont.systemFontOfSize_(12))
        md_label.setFrame_(NSMakeRect(16, cy, 110, 18))
        step1_content.addSubview_(md_label)

        default_model = self.config.get_effective_model_dir()
        model_display = self.config.model_dir if self.config.model_dir else default_model
        self.model_dir_label = NSTextField.labelWithString_(model_display)
        self.model_dir_label.setFont_(NSFont.monospacedSystemFontOfSize_weight_(10, 0))
        self.model_dir_label.setTextColor_(NSColor.secondaryLabelColor())
        self.model_dir_label.setLineBreakMode_(5)
        self.model_dir_label.setFrame_(NSMakeRect(130, cy, WINDOW_WIDTH - 278, 18))
        step1_content.addSubview_(self.model_dir_label)

        browse_model_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(WINDOW_WIDTH - 160, cy - 2, 80, 22)
        )
        browse_model_btn.setTitle_("Browse...")
        browse_model_btn.setBezelStyle_(NSBezelStyleRounded)
        browse_model_btn.setFont_(NSFont.systemFontOfSize_(11))
        browse_model_btn.setTarget_(self)
        browse_model_btn.setAction_(
            objc.selector(self.browseModelDir_, signature=b"v@:@")
        )
        step1_content.addSubview_(browse_model_btn)

        # Optional hint
        cy -= 16
        md_hint = NSTextField.labelWithString_(
            "Specify if you have pre-downloaded models elsewhere"
        )
        md_hint.setFont_(NSFont.systemFontOfSize_(10))
        md_hint.setTextColor_(NSColor.tertiaryLabelColor())
        md_hint.setFrame_(NSMakeRect(130, cy, WINDOW_WIDTH - 240, 12))
        step1_content.addSubview_(md_hint)

        # Separator
        cy -= 12
        sep2 = self._create_separator()
        sep2.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 1))
        step1_content.addSubview_(sep2)

        # Port
        cy -= 30
        port_label = NSTextField.labelWithString_("Port")
        port_label.setFont_(NSFont.systemFontOfSize_(12))
        port_label.setFrame_(NSMakeRect(16, cy, 110, 18))
        step1_content.addSubview_(port_label)

        self.port_field = NSTextField.alloc().initWithFrame_(
            NSMakeRect(130, cy - 2, 100, 22)
        )
        self.port_field.setStringValue_(str(self.config.port))
        self.port_field.setFont_(NSFont.monospacedSystemFontOfSize_weight_(12, 0))
        step1_content.addSubview_(self.port_field)

        # Separator
        cy -= 16
        sep_api = self._create_separator()
        sep_api.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 1))
        step1_content.addSubview_(sep_api)

        # API Key row
        cy -= 30
        api_label = NSTextField.labelWithString_("API Key")
        api_label.setFont_(NSFont.systemFontOfSize_(12))
        api_label.setFrame_(NSMakeRect(16, cy, 110, 18))
        step1_content.addSubview_(api_label)

        current_key = self.config.get_server_api_key() or ""

        self.api_key_secure = NSSecureTextField.alloc().initWithFrame_(
            NSMakeRect(130, cy - 2, 200, 22)
        )
        self.api_key_secure.setStringValue_(current_key)
        self.api_key_secure.setFont_(NSFont.monospacedSystemFontOfSize_weight_(12, 0))
        self.api_key_secure.setPlaceholderString_("Minimum 4 characters")
        step1_content.addSubview_(self.api_key_secure)

        self.api_key_plain = NSTextField.alloc().initWithFrame_(
            NSMakeRect(130, cy - 2, 200, 22)
        )
        self.api_key_plain.setStringValue_(current_key)
        self.api_key_plain.setFont_(NSFont.monospacedSystemFontOfSize_weight_(12, 0))
        self.api_key_plain.setPlaceholderString_("Minimum 4 characters")
        self.api_key_plain.setHidden_(True)
        step1_content.addSubview_(self.api_key_plain)

        self._eye_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(334, cy - 2, 28, 22)
        )
        self._eye_btn.setBordered_(False)
        eye_icon = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "eye", None
        )
        if eye_icon:
            eye_icon.setSize_((16, 16))
            self._eye_btn.setImage_(eye_icon)
        self._eye_btn.setTarget_(self)
        self._eye_btn.setAction_(
            objc.selector(self.toggleApiKeyVisibility_, signature=b"v@:@")
        )
        step1_content.addSubview_(self._eye_btn)

        y -= 242

        # === Step 2: Start Server Card ===
        step2_card = self._create_card()
        step2_card.setFrame_(NSMakeRect(24, y - 68, WINDOW_WIDTH - 48, 68))
        container.addSubview_(step2_card)

        step2_content = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WINDOW_WIDTH - 48, 68)
        )
        step2_card.setContentView_(step2_content)

        cy = 68 - 16

        # Header
        cy -= 20
        step2_header = NSTextField.labelWithString_("Step 2  Start Server")
        step2_header.setFont_(NSFont.systemFontOfSize_weight_(14, 0.65))
        step2_header.setTextColor_(NSColor.labelColor())
        step2_header.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 20))
        step2_content.addSubview_(step2_header)

        # Start button
        cy -= 32
        self.start_button = NSButton.alloc().initWithFrame_(
            NSMakeRect((WINDOW_WIDTH - 208) // 2, cy, 160, 28)
        )
        self.start_button.setTitle_("Start Server")
        self.start_button.setBezelStyle_(NSBezelStyleRounded)
        self.start_button.setFont_(NSFont.systemFontOfSize_(12))
        self.start_button.setTarget_(self)
        self.start_button.setAction_(objc.selector(self.startServer_, signature=b"v@:@"))
        step2_content.addSubview_(self.start_button)

        y -= 84

        # Status label (between cards)
        self.status_label = NSTextField.labelWithString_("")
        self.status_label.setFont_(NSFont.systemFontOfSize_(11))
        self.status_label.setTextColor_(NSColor.secondaryLabelColor())
        self.status_label.setAlignment_(NSCenterTextAlignment)
        self.status_label.setFrame_(NSMakeRect(40, y, WINDOW_WIDTH - 80, 16))
        container.addSubview_(self.status_label)

        # === Step 3: Open Settings Card ===
        step3_card = self._create_card()
        step3_card.setFrame_(NSMakeRect(24, y - 96, WINDOW_WIDTH - 48, 96))
        container.addSubview_(step3_card)

        step3_content = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WINDOW_WIDTH - 48, 96)
        )
        step3_card.setContentView_(step3_content)

        cy = 96 - 16

        # Header
        cy -= 20
        step3_header = NSTextField.labelWithString_("Step 3  Open Settings")
        step3_header.setFont_(NSFont.systemFontOfSize_weight_(14, 0.65))
        step3_header.setTextColor_(NSColor.labelColor())
        step3_header.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 20))
        step3_content.addSubview_(step3_header)

        # Description
        cy -= 18
        desc = NSTextField.labelWithString_(
            "Configure models and advanced settings in the admin panel."
        )
        desc.setFont_(NSFont.systemFontOfSize_(11))
        desc.setTextColor_(NSColor.secondaryLabelColor())
        desc.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 16))
        step3_content.addSubview_(desc)

        # Dashboard button
        cy -= 32
        dash_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(((WINDOW_WIDTH - 48) - 220) // 2, cy, 220, 28)
        )
        dash_btn.setTitle_("Open Admin Panel & Close")
        dash_btn.setBezelStyle_(NSBezelStyleRounded)
        dash_btn.setFont_(NSFont.systemFontOfSize_(12))
        dash_btn.setTarget_(self)
        dash_btn.setAction_(objc.selector(self.openDashboard_, signature=b"v@:@"))
        step3_content.addSubview_(dash_btn)

        # === Bottom Close button ===
        close_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect((WINDOW_WIDTH - 100) // 2, 16, 100, 28)
        )
        close_btn.setTitle_("Close")
        close_btn.setBezelStyle_(NSBezelStyleRounded)
        close_btn.setTarget_(self)
        close_btn.setAction_(objc.selector(self.closeWindow_, signature=b"v@:@"))
        container.addSubview_(close_btn)

        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    @objc.IBAction
    def browseBaseDir_(self, sender):
        """Open folder picker for base directory selection."""
        panel = NSOpenPanel.openPanel()
        panel.setCanChooseDirectories_(True)
        panel.setCanChooseFiles_(False)
        panel.setCanCreateDirectories_(True)
        panel.setAllowsMultipleSelection_(False)
        panel.setPrompt_("Select")
        panel.setMessage_(
            "Choose a parent folder. An .omlx directory will be created inside it."
        )

        if panel.runModal() == 1:
            selected = str(panel.URL().path())
            new_base = str(Path(selected) / ".omlx")
            self.base_path_label.setStringValue_(new_base)

    @objc.IBAction
    def browseModelDir_(self, sender):
        """Open folder picker for model directory selection."""
        panel = NSOpenPanel.openPanel()
        panel.setCanChooseDirectories_(True)
        panel.setCanChooseFiles_(False)
        panel.setCanCreateDirectories_(True)
        panel.setAllowsMultipleSelection_(False)
        panel.setPrompt_("Select")
        panel.setMessage_("Choose the directory containing your model files.")

        if panel.runModal() == 1:
            selected = str(panel.URL().path())
            self.model_dir_label.setStringValue_(selected)

    @objc.IBAction
    def startServer_(self, sender):
        """Save config and start the server."""
        base_path = str(self.base_path_label.stringValue()).strip()
        model_dir = str(self.model_dir_label.stringValue()).strip()
        port_str = str(self.port_field.stringValue()).strip()

        if not base_path:
            self.status_label.setStringValue_("Please select a base directory.")
            self.status_label.setTextColor_(NSColor.systemRedColor())
            return

        try:
            port = int(port_str)
            if not (1024 <= port <= 65535):
                raise ValueError
        except ValueError:
            self.status_label.setStringValue_("Port must be 1024-65535.")
            self.status_label.setTextColor_(NSColor.systemRedColor())
            return

        # Get API key from whichever field is visible
        if self._api_key_visible:
            api_key = str(self.api_key_plain.stringValue()).strip()
        else:
            api_key = str(self.api_key_secure.stringValue()).strip()

        if not api_key:
            self.status_label.setStringValue_(
                "API key is required to start the server."
            )
            self.status_label.setTextColor_(NSColor.systemRedColor())
            return

        if len(api_key) < 4 or " " in api_key:
            self.status_label.setStringValue_(
                "API key must be at least 4 characters with no spaces."
            )
            self.status_label.setTextColor_(NSColor.systemRedColor())
            return

        self.config.base_path = base_path
        self.config.port = port
        default_md = str(Path(base_path).expanduser() / "models")
        if model_dir and model_dir != default_md:
            self.config.model_dir = model_dir
        else:
            self.config.model_dir = ""
        self.config.save()

        # Save API key to server's settings.json
        if api_key:
            try:
                self.config.set_server_api_key(api_key)
            except Exception as e:
                logger.warning(f"Failed to save API key: {e}")

        try:
            Path(base_path).expanduser().mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.status_label.setStringValue_(f"Cannot create directory: {e}")
            self.status_label.setTextColor_(NSColor.systemRedColor())
            return

        self.server_manager.update_config(self.config)
        result = self.server_manager.start()

        if isinstance(result, PortConflict):
            self._handle_port_conflict(result)
            return

        self.start_button.setEnabled_(False)
        self.start_button.setTitle_("Starting...")

        self.status_label.setStringValue_("Server is starting...")
        self.status_label.setTextColor_(NSColor.systemGreenColor())

        self._start_status_timer()

    def _start_status_timer(self):
        """Start the timer that polls server startup status."""
        from Foundation import NSDefaultRunLoopMode, NSRunLoop, NSTimer

        self.status_check_timer = (
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1.0, self, "checkServerStatus:", None, True
            )
        )
        NSRunLoop.currentRunLoop().addTimer_forMode_(
            self.status_check_timer, NSDefaultRunLoopMode
        )

    def _handle_port_conflict(self, conflict: PortConflict):
        """Handle a port conflict inline in the wizard UI."""
        from AppKit import NSAlert, NSAlertFirstButtonReturn, NSAlertSecondButtonReturn

        if conflict.is_omlx:
            alert = NSAlert.alloc().init()
            alert.setMessageText_("oMLX Server Already Running")
            pid_info = f" (PID {conflict.pid})" if conflict.pid else ""
            alert.setInformativeText_(
                f"An oMLX server is already running on port "
                f"{self.config.port}{pid_info}.\n\n"
                f"You can adopt it (monitor without restarting) "
                f"or kill it and start a new one."
            )
            alert.addButtonWithTitle_("Adopt")
            alert.addButtonWithTitle_("Kill & Restart")
            alert.addButtonWithTitle_("Cancel")

            response = alert.runModal()
            if response == NSAlertFirstButtonReturn:
                if self.server_manager.adopt():
                    self.start_button.setEnabled_(False)
                    self.start_button.setTitle_("Server Running")
                    self.status_label.setStringValue_(
                        "Adopted existing server."
                    )
                    self.status_label.setTextColor_(NSColor.systemGreenColor())
                else:
                    self.status_label.setStringValue_(
                        "Failed to adopt — server may have stopped."
                    )
                    self.status_label.setTextColor_(NSColor.systemRedColor())
            elif response == NSAlertSecondButtonReturn:
                if conflict.pid:
                    self.server_manager._kill_external_server(conflict.pid)
                    import time
                    time.sleep(0.5)
                result = self.server_manager.start()
                if isinstance(result, PortConflict):
                    self.status_label.setStringValue_(
                        "Port still in use. Try again."
                    )
                    self.status_label.setTextColor_(NSColor.systemRedColor())
                else:
                    self.start_button.setEnabled_(False)
                    self.start_button.setTitle_("Starting...")
                    self.status_label.setStringValue_("Server is starting...")
                    self.status_label.setTextColor_(NSColor.systemGreenColor())
                    self._start_status_timer()
            # Cancel: do nothing
        else:
            pid_info = f" (PID {conflict.pid})" if conflict.pid else ""
            self.status_label.setStringValue_(
                f"Port {self.config.port} is in use by another "
                f"application{pid_info}. Change the port above."
            )
            self.status_label.setTextColor_(NSColor.systemRedColor())

    def checkServerStatus_(self, timer):
        """Check if server has started or failed."""
        from .server_manager import ServerStatus

        status = self.server_manager.status
        if status == ServerStatus.RUNNING:
            self.status_label.setStringValue_("Server started successfully!")
            self.status_label.setTextColor_(NSColor.systemGreenColor())
            self.start_button.setTitle_("Server Running")
            if self.status_check_timer:
                self.status_check_timer.invalidate()
                self.status_check_timer = None
        elif status == ServerStatus.ERROR:
            error_msg = self.server_manager.error_message or "Unknown error"
            self.status_label.setStringValue_(f"Error: {error_msg}")
            self.status_label.setTextColor_(NSColor.systemRedColor())
            self.start_button.setEnabled_(True)
            self.start_button.setTitle_("Start Server")
            if self.status_check_timer:
                self.status_check_timer.invalidate()
                self.status_check_timer = None

    @objc.IBAction
    def toggleApiKeyVisibility_(self, sender):
        """Toggle API key between masked and plain text."""
        if self._api_key_visible:
            # Switch to masked
            value = str(self.api_key_plain.stringValue())
            self.api_key_secure.setStringValue_(value)
            self.api_key_secure.setHidden_(False)
            self.api_key_plain.setHidden_(True)
            icon = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                "eye", None
            )
            self._api_key_visible = False
        else:
            # Switch to plain text
            value = str(self.api_key_secure.stringValue())
            self.api_key_plain.setStringValue_(value)
            self.api_key_secure.setHidden_(True)
            self.api_key_plain.setHidden_(False)
            icon = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                "eye.slash", None
            )
            self._api_key_visible = True
        if icon:
            icon.setSize_((16, 16))
            sender.setImage_(icon)

    @objc.IBAction
    def openDashboard_(self, sender):
        """Open the admin dashboard and close welcome window."""
        port = self.config.port
        webbrowser.open(f"http://127.0.0.1:{port}/admin")
        self.window.close()

    @objc.IBAction
    def closeWindow_(self, sender):
        """Save config and close the welcome window."""
        if self.status_check_timer:
            self.status_check_timer.invalidate()
            self.status_check_timer = None

        base_path = str(self.base_path_label.stringValue()).strip()
        model_dir = str(self.model_dir_label.stringValue()).strip()
        port_str = str(self.port_field.stringValue()).strip()

        if base_path:
            self.config.base_path = base_path
        try:
            port = int(port_str)
            if 1024 <= port <= 65535:
                self.config.port = port
        except ValueError:
            pass

        default_md = str(Path(base_path).expanduser() / "models") if base_path else ""
        if model_dir and model_dir != default_md:
            self.config.model_dir = model_dir
        else:
            self.config.model_dir = ""

        # Save API key to server's settings.json
        if self._api_key_visible:
            api_key = str(self.api_key_plain.stringValue()).strip()
        else:
            api_key = str(self.api_key_secure.stringValue()).strip()
        if api_key and len(api_key) >= 4 and " " not in api_key:
            try:
                self.config.set_server_api_key(api_key)
            except Exception as e:
                logger.warning(f"Failed to save API key: {e}")

        self.config.save()
        self.window.close()
