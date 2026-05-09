"""Settings window for oMLX app settings - modern macOS native design."""

import logging
import plistlib
import shutil
from pathlib import Path
from typing import Callable, Optional

import objc
from AppKit import (
    NSAlert,
    NSAlertFirstButtonReturn,
    NSAlertStyleCritical,
    NSAlertStyleWarning,
    NSApp,
    NSBackingStoreBuffered,
    NSBezelStyleRounded,
    NSBox,
    NSBoxCustom,
    NSBoxSeparator,
    NSButton,
    NSButtonTypeSwitch,
    NSColor,
    NSControlStateValueOff,
    NSControlStateValueOn,
    NSFont,
    NSImage,
    NSMakeRect,
    NSOpenPanel,
    NSTextField,
    NSView,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectMaterialSidebar,
    NSVisualEffectView,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSObject

from .widgets import PastableSecureTextField

logger = logging.getLogger(__name__)

LAUNCH_AGENT_LABEL = "com.omlx.app"
LAUNCH_AGENT_DIR = Path.home() / "Library" / "LaunchAgents"
LAUNCH_AGENT_PLIST = LAUNCH_AGENT_DIR / f"{LAUNCH_AGENT_LABEL}.plist"

WINDOW_WIDTH = 520
WINDOW_HEIGHT = 606


class PreferencesWindowController(NSObject):
    """Controller for the Settings window - modern macOS design."""

    def initWithConfig_serverManager_onSave_(
        self, config, server_manager, on_save_callback
    ):
        self = objc.super(PreferencesWindowController, self).init()
        if self is None:
            return None
        self.config = config
        self.server_manager = server_manager
        self.on_save: Optional[Callable] = on_save_callback
        self.show_welcome: Optional[Callable] = None
        self.window = None
        self.launch_at_login_checkbox = None
        self.auto_start_checkbox = None
        self.base_path_label = None
        self.model_dir_label = None
        self.port_field = None
        self.api_key_secure = None
        self.api_key_plain = None
        self._api_key_visible = False
        self._eye_btn = None
        self._original_base_path = config.base_path
        return self

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
        """Create and show the settings window."""
        # Sync from server settings.json (Web admin changes)
        self.config.sync_from_server_settings()

        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        frame = NSMakeRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, NSBackingStoreBuffered, False
        )
        self.window.setTitle_("oMLX Settings")
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

        # === Title ===
        y -= 30
        title = NSTextField.labelWithString_("Settings")
        title.setFont_(NSFont.systemFontOfSize_weight_(22, 0.5))
        title.setFrame_(NSMakeRect(24, y, WINDOW_WIDTH - 48, 30))
        container.addSubview_(title)

        # === Server Settings Card ===
        y -= 46
        server_card = self._create_card()
        server_card.setFrame_(NSMakeRect(24, y - 218, WINDOW_WIDTH - 48, 218))
        container.addSubview_(server_card)

        server_content = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WINDOW_WIDTH - 48, 218)
        )
        server_card.setContentView_(server_content)

        cy = 218 - 16

        # Header
        cy -= 20
        server_header = NSTextField.labelWithString_("Server Settings")
        server_header.setFont_(NSFont.systemFontOfSize_weight_(14, 0.65))
        server_header.setTextColor_(NSColor.labelColor())
        server_header.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 20))
        server_content.addSubview_(server_header)

        # Base Directory row
        cy -= 30
        bp_label = NSTextField.labelWithString_("Base Directory")
        bp_label.setFont_(NSFont.systemFontOfSize_(12))
        bp_label.setFrame_(NSMakeRect(16, cy, 120, 18))
        server_content.addSubview_(bp_label)

        self.base_path_label = NSTextField.labelWithString_(self.config.base_path)
        self.base_path_label.setFont_(NSFont.monospacedSystemFontOfSize_weight_(10, 0))
        self.base_path_label.setTextColor_(NSColor.secondaryLabelColor())
        self.base_path_label.setLineBreakMode_(5)
        self.base_path_label.setFrame_(NSMakeRect(140, cy, WINDOW_WIDTH - 298, 18))
        server_content.addSubview_(self.base_path_label)

        browse_base_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(WINDOW_WIDTH - 160, cy - 2, 80, 22)
        )
        browse_base_btn.setTitle_("Browse...")
        browse_base_btn.setBezelStyle_(NSBezelStyleRounded)
        browse_base_btn.setFont_(NSFont.systemFontOfSize_(11))
        browse_base_btn.setTarget_(self)
        browse_base_btn.setAction_(objc.selector(self.browseBaseDir_, signature=b"v@:@"))
        server_content.addSubview_(browse_base_btn)

        # Separator
        cy -= 16
        sep1 = self._create_separator()
        sep1.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 1))
        server_content.addSubview_(sep1)

        # Model Directory row
        cy -= 30
        md_label = NSTextField.labelWithString_("Model Directory")
        md_label.setFont_(NSFont.systemFontOfSize_(12))
        md_label.setFrame_(NSMakeRect(16, cy, 120, 18))
        server_content.addSubview_(md_label)

        model_display = (
            self.config.model_dir
            if self.config.model_dir
            else self.config.get_effective_model_dir()
        )
        self.model_dir_label = NSTextField.labelWithString_(model_display)
        self.model_dir_label.setFont_(NSFont.monospacedSystemFontOfSize_weight_(10, 0))
        self.model_dir_label.setTextColor_(NSColor.secondaryLabelColor())
        self.model_dir_label.setLineBreakMode_(5)
        self.model_dir_label.setFrame_(NSMakeRect(140, cy, WINDOW_WIDTH - 298, 18))
        server_content.addSubview_(self.model_dir_label)

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
        server_content.addSubview_(browse_model_btn)

        # Separator
        cy -= 16
        sep2 = self._create_separator()
        sep2.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 1))
        server_content.addSubview_(sep2)

        # Port row
        cy -= 30
        port_label = NSTextField.labelWithString_("Port")
        port_label.setFont_(NSFont.systemFontOfSize_(12))
        port_label.setFrame_(NSMakeRect(16, cy, 120, 18))
        server_content.addSubview_(port_label)

        self.port_field = NSTextField.alloc().initWithFrame_(
            NSMakeRect(140, cy - 2, 100, 22)
        )
        self.port_field.setStringValue_(str(self.config.port))
        self.port_field.setFont_(NSFont.monospacedSystemFontOfSize_weight_(12, 0))
        server_content.addSubview_(self.port_field)

        # Separator
        cy -= 16
        sep_api = self._create_separator()
        sep_api.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 1))
        server_content.addSubview_(sep_api)

        # API Key row
        cy -= 30
        api_label = NSTextField.labelWithString_("API Key")
        api_label.setFont_(NSFont.systemFontOfSize_(12))
        api_label.setFrame_(NSMakeRect(16, cy, 120, 18))
        server_content.addSubview_(api_label)

        current_key = self.config.get_server_api_key() or ""

        self.api_key_secure = PastableSecureTextField.alloc().initWithFrame_(
            NSMakeRect(140, cy - 2, 200, 22)
        )
        self.api_key_secure.setStringValue_(current_key)
        self.api_key_secure.setFont_(NSFont.monospacedSystemFontOfSize_weight_(12, 0))
        self.api_key_secure.setPlaceholderString_("Minimum 4 characters")
        server_content.addSubview_(self.api_key_secure)

        self.api_key_plain = NSTextField.alloc().initWithFrame_(
            NSMakeRect(140, cy - 2, 200, 22)
        )
        self.api_key_plain.setStringValue_(current_key)
        self.api_key_plain.setFont_(NSFont.monospacedSystemFontOfSize_weight_(12, 0))
        self.api_key_plain.setPlaceholderString_("Minimum 4 characters")
        self.api_key_plain.setHidden_(True)
        server_content.addSubview_(self.api_key_plain)

        self._eye_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(344, cy - 2, 28, 22)
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
        server_content.addSubview_(self._eye_btn)

        y -= 234

        # === Behavior Card ===
        behavior_card = self._create_card()
        behavior_card.setFrame_(NSMakeRect(24, y - 104, WINDOW_WIDTH - 48, 104))
        container.addSubview_(behavior_card)

        behavior_content = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WINDOW_WIDTH - 48, 104)
        )
        behavior_card.setContentView_(behavior_content)

        cy = 104 - 16

        # Header
        cy -= 20
        behavior_header = NSTextField.labelWithString_("Behavior")
        behavior_header.setFont_(NSFont.systemFontOfSize_weight_(14, 0.65))
        behavior_header.setTextColor_(NSColor.labelColor())
        behavior_header.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 20))
        behavior_content.addSubview_(behavior_header)

        # Launch at login
        cy -= 26
        self.launch_at_login_checkbox = NSButton.alloc().initWithFrame_(
            NSMakeRect(16, cy, WINDOW_WIDTH - 96, 20)
        )
        self.launch_at_login_checkbox.setButtonType_(NSButtonTypeSwitch)
        self.launch_at_login_checkbox.setTitle_("Launch oMLX at login")
        self.launch_at_login_checkbox.setFont_(NSFont.systemFontOfSize_(12))
        self.launch_at_login_checkbox.setState_(
            NSControlStateValueOn
            if self.config.launch_at_login
            else NSControlStateValueOff
        )
        behavior_content.addSubview_(self.launch_at_login_checkbox)

        # Separator
        cy -= 12
        sep3 = self._create_separator()
        sep3.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 1))
        behavior_content.addSubview_(sep3)

        # Auto-start
        cy -= 26
        self.auto_start_checkbox = NSButton.alloc().initWithFrame_(
            NSMakeRect(16, cy, WINDOW_WIDTH - 96, 20)
        )
        self.auto_start_checkbox.setButtonType_(NSButtonTypeSwitch)
        self.auto_start_checkbox.setTitle_("Start server automatically on launch")
        self.auto_start_checkbox.setFont_(NSFont.systemFontOfSize_(12))
        self.auto_start_checkbox.setState_(
            NSControlStateValueOn
            if self.config.start_server_on_launch
            else NSControlStateValueOff
        )
        behavior_content.addSubview_(self.auto_start_checkbox)

        y -= 120

        # === Actions Card ===
        actions_card = self._create_card()
        actions_card.setFrame_(NSMakeRect(24, y - 88, WINDOW_WIDTH - 48, 88))
        container.addSubview_(actions_card)

        actions_content = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WINDOW_WIDTH - 48, 88)
        )
        actions_card.setContentView_(actions_content)

        cy = 88 - 16

        # Header
        cy -= 20
        actions_header = NSTextField.labelWithString_("Actions")
        actions_header.setFont_(NSFont.systemFontOfSize_weight_(14, 0.65))
        actions_header.setTextColor_(NSColor.labelColor())
        actions_header.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 20))
        actions_content.addSubview_(actions_header)

        # Buttons
        cy -= 28
        welcome_btn = NSButton.alloc().initWithFrame_(NSMakeRect(16, cy, 170, 24))
        welcome_btn.setTitle_("Show Welcome Screen")
        welcome_btn.setBezelStyle_(NSBezelStyleRounded)
        welcome_btn.setFont_(NSFont.systemFontOfSize_(12))
        welcome_btn.setTarget_(self)
        welcome_btn.setAction_(objc.selector(self.showWelcome_, signature=b"v@:@"))
        actions_content.addSubview_(welcome_btn)

        reset_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(WINDOW_WIDTH - 234, cy, 170, 24)
        )
        reset_btn.setTitle_("Reset All Settings...")
        reset_btn.setBezelStyle_(NSBezelStyleRounded)
        reset_btn.setFont_(NSFont.systemFontOfSize_(12))
        reset_btn.setTarget_(self)
        reset_btn.setAction_(objc.selector(self.resetSettings_, signature=b"v@:@"))
        actions_content.addSubview_(reset_btn)

        # Description
        cy -= 20
        reset_desc = NSTextField.labelWithString_(
            "Removes all data including downloaded models, cache, and settings."
        )
        reset_desc.setFont_(NSFont.systemFontOfSize_(10))
        reset_desc.setTextColor_(NSColor.tertiaryLabelColor())
        reset_desc.setFrame_(NSMakeRect(16, cy, WINDOW_WIDTH - 96, 14))
        actions_content.addSubview_(reset_desc)

        # === Bottom buttons ===
        save_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(WINDOW_WIDTH - 200, 16, 80, 28)
        )
        save_btn.setTitle_("Save")
        save_btn.setBezelStyle_(NSBezelStyleRounded)
        save_btn.setTarget_(self)
        save_btn.setAction_(objc.selector(self.savePrefs_, signature=b"v@:@"))
        container.addSubview_(save_btn)

        close_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(WINDOW_WIDTH - 110, 16, 80, 28)
        )
        close_btn.setTitle_("Close")
        close_btn.setBezelStyle_(NSBezelStyleRounded)
        close_btn.setTarget_(self)
        close_btn.setAction_(objc.selector(self.closePrefs_, signature=b"v@:@"))
        container.addSubview_(close_btn)

        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    # --- Actions ---

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

            if new_base != self._original_base_path:
                old_path = Path(self._original_base_path).expanduser()
                if old_path.exists():
                    alert = NSAlert.alloc().init()
                    alert.setMessageText_("Change Base Directory?")
                    alert.setInformativeText_(
                        f"Changing from:\n{self._original_base_path}\n\n"
                        f"To:\n{new_base}\n\n"
                        "The old directory and its data (models, cache, settings) "
                        "will NOT be moved automatically."
                    )
                    alert.setAlertStyle_(NSAlertStyleWarning)
                    alert.addButtonWithTitle_("Change")
                    alert.addButtonWithTitle_("Cancel")
                    if alert.runModal() != NSAlertFirstButtonReturn:
                        return

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
    def savePrefs_(self, sender):
        """Save settings and apply changes."""
        new_base_path = str(self.base_path_label.stringValue())
        new_model_dir = str(self.model_dir_label.stringValue())
        port_str = str(self.port_field.stringValue()).strip()

        # Snapshot the effective model dir the form was populated with. We only
        # push to the server when this changes, otherwise an unrelated save
        # (API key, port, launch-at-login) would collapse a multi-dir setup
        # made through the web admin to whatever single dir the form shows.
        original_effective_model_dir = self.config.get_effective_model_dir()

        try:
            port = int(port_str)
            if not (1024 <= port <= 65535):
                raise ValueError
        except ValueError:
            alert = NSAlert.alloc().init()
            alert.setMessageText_("Invalid Port")
            alert.setInformativeText_("Port must be between 1024 and 65535.")
            alert.runModal()
            return

        # Get API key from whichever field is visible
        if self._api_key_visible:
            api_key = str(self.api_key_plain.stringValue()).strip()
        else:
            api_key = str(self.api_key_secure.stringValue()).strip()

        if api_key and (len(api_key) < 4 or " " in api_key):
            alert = NSAlert.alloc().init()
            alert.setMessageText_("Invalid API Key")
            alert.setInformativeText_(
                "API key must be at least 4 characters with no spaces."
            )
            alert.runModal()
            return

        self.config.base_path = new_base_path
        self.config.port = port
        default_md = str(Path(new_base_path).expanduser() / "models")
        if new_model_dir and new_model_dir != default_md:
            self.config.model_dir = new_model_dir
        else:
            self.config.model_dir = ""
        self.config.launch_at_login = bool(self.launch_at_login_checkbox.state())
        self.config.start_server_on_launch = bool(self.auto_start_checkbox.state())
        self.config.save()

        from .server_manager import ServerStatus

        # Save API key
        if api_key:
            if self.server_manager.status == ServerStatus.RUNNING:
                # Update running server via admin API (also saves to settings.json)
                if not self.config.update_server_api_key_runtime(api_key):
                    # Fallback: write to settings.json directly
                    self.config.set_server_api_key(api_key)
            else:
                self.config.set_server_api_key(api_key)

        # Keep the server settings in sync with the DMG app's single model
        # directory preference. The server reads settings.json on startup and
        # keeps an in-memory copy while running. Only sync when the value
        # actually changed so unrelated saves don't overwrite multi-dir setups.
        new_effective_model_dir = self.config.get_effective_model_dir()
        if new_effective_model_dir != original_effective_model_dir:
            if self.server_manager.status == ServerStatus.RUNNING:
                if not self.config.update_model_dir_runtime(new_effective_model_dir):
                    self.config.sync_model_dir_to_server_settings(overwrite=True)
            else:
                self.config.sync_model_dir_to_server_settings(overwrite=True)

        self._apply_launch_at_login(self.config.launch_at_login)

        if self.on_save:
            self.on_save()

        self.window.close()

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
    def closePrefs_(self, sender):
        """Close the settings window without saving."""
        self.window.close()

    @objc.IBAction
    def showWelcome_(self, sender):
        """Show the welcome screen."""
        self.window.close()
        if self.show_welcome:
            self.show_welcome()

    @objc.IBAction
    def resetSettings_(self, sender):
        """Reset all settings with confirmation dialog."""
        base_path = str(Path(self.config.base_path).expanduser())

        alert = NSAlert.alloc().init()
        alert.setMessageText_("Reset All Settings?")
        alert.setInformativeText_(
            f"This will delete cache, logs, and server settings in:\n"
            f"{base_path}\n\n"
            "Downloaded models will be preserved.\n\n"
            "This action cannot be undone."
        )
        alert.setAlertStyle_(NSAlertStyleCritical)
        alert.addButtonWithTitle_("Cancel")
        alert.addButtonWithTitle_("Reset Everything")

        response = alert.runModal()
        if response != NSAlertFirstButtonReturn:
            if self.server_manager.is_running():
                self.server_manager.stop()

            base = Path(base_path)
            if base.exists():
                model_dir = Path(
                    self.config.get_effective_model_dir()
                ).resolve()
                for item in base.iterdir():
                    if item.resolve() == model_dir:
                        continue
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    except OSError as e:
                        logger.error(f"Failed to delete {item}: {e}")
                logger.info(
                    f"Reset base directory: {base} (preserved {model_dir})"
                )

            from .config import ServerConfig

            defaults = ServerConfig()
            self.config.base_path = defaults.base_path
            self.config.port = defaults.port
            self.config.model_dir = ""
            self.config.launch_at_login = False
            self.config.start_server_on_launch = False
            self.config.save()

            self.base_path_label.setStringValue_(defaults.base_path)
            self.model_dir_label.setStringValue_(defaults.get_effective_model_dir())
            self.port_field.setStringValue_(str(defaults.port))
            self.api_key_secure.setStringValue_("")
            self.api_key_plain.setStringValue_("")
            self._api_key_visible = False
            self.api_key_secure.setHidden_(False)
            self.api_key_plain.setHidden_(True)
            eye_icon = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                "eye", None
            )
            if eye_icon:
                eye_icon.setSize_((16, 16))
                self._eye_btn.setImage_(eye_icon)
            self.launch_at_login_checkbox.setState_(NSControlStateValueOff)
            self.auto_start_checkbox.setState_(NSControlStateValueOff)
            self._original_base_path = defaults.base_path

            if self.on_save:
                self.on_save()

    def _get_app_path(self) -> Optional[str]:
        """Find the oMLX.app bundle path."""
        app_py = Path(__file__)
        for parent in app_py.parents:
            if parent.suffix == ".app" and (parent / "Contents" / "MacOS").exists():
                return str(parent)
        standard = Path("/Applications/oMLX.app")
        if standard.exists():
            return str(standard)
        return None

    def _apply_launch_at_login(self, enabled: bool):
        """Register/unregister launch at login via LaunchAgent plist."""
        try:
            if enabled:
                app_path = self._get_app_path()
                if not app_path:
                    logger.warning("Cannot find oMLX.app bundle for launch agent")
                    return

                LAUNCH_AGENT_DIR.mkdir(parents=True, exist_ok=True)
                plist_data = {
                    "Label": LAUNCH_AGENT_LABEL,
                    "ProgramArguments": ["open", "-a", app_path],
                    "RunAtLoad": True,
                }
                with open(LAUNCH_AGENT_PLIST, "wb") as f:
                    plistlib.dump(plist_data, f)
                logger.info(f"Created launch agent: {LAUNCH_AGENT_PLIST}")
            else:
                if LAUNCH_AGENT_PLIST.exists():
                    LAUNCH_AGENT_PLIST.unlink()
                    logger.info(f"Removed launch agent: {LAUNCH_AGENT_PLIST}")
        except Exception as e:
            logger.warning(f"Failed to update launch at login: {e}")
