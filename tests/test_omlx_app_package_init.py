# SPDX-License-Identifier: Apache-2.0
"""Import-safety tests for packaging/omlx_app package init."""

import importlib
import sys
import types
from pathlib import Path


def test_omlx_app_init_does_not_import_omlx_runtime(monkeypatch):
    """Importing omlx_app package should not import heavy omlx runtime modules."""
    packaging_dir = Path(__file__).parent.parent / "packaging"
    monkeypatch.syspath_prepend(str(packaging_dir))

    blocked_prefixes = ("omlx",)

    orig_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith(blocked_prefixes):
            raise AssertionError(f"blocked import: {name}")
        return orig_import(name, globals, locals, fromlist, level)

    # Ensure a fresh import path for the package under test.
    sys.modules.pop("omlx_app", None)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    # This import should not try to pull in `omlx`.
    module = importlib.import_module("omlx_app")
    assert hasattr(module, "__version__")


def test_omlx_app_app_module_does_not_import_omlx_runtime(monkeypatch):
    """Importing omlx_app.app should avoid importing `omlx` package runtime."""
    packaging_dir = Path(__file__).parent.parent / "packaging"
    monkeypatch.syspath_prepend(str(packaging_dir))

    blocked = ("omlx", "omlx.")
    orig_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == blocked[0] or name.startswith(blocked[1]):
            raise AssertionError(f"blocked import: {name}")
        return orig_import(name, globals, locals, fromlist, level)

    # Stub GUI/runtime dependencies so this import remains unit-test friendly.
    fake_objc = types.ModuleType("objc")
    fake_objc.super = super
    fake_objc.IBAction = lambda func: func

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
        setattr(fake_appkit, name, type(name, (), {}))
    fake_appkit.NSVariableStatusItemLength = -1

    fake_foundation = types.ModuleType("Foundation")
    fake_foundation.NSObject = type("NSObject", (), {})
    fake_foundation.NSData = type("NSData", (), {})
    fake_foundation.NSMutableAttributedString = type(
        "NSMutableAttributedString", (), {}
    )
    fake_foundation.NSRunLoop = type("NSRunLoop", (), {})
    fake_foundation.NSDefaultRunLoopMode = "NSDefaultRunLoopMode"
    fake_foundation.NSTimer = type("NSTimer", (), {})
    fake_foundation.NSURL = type("NSURL", (), {})

    fake_config = types.ModuleType("omlx_app.config")
    fake_config.ServerConfig = object()
    fake_server_manager = types.ModuleType("omlx_app.server_manager")
    fake_server_manager.PortConflict = object()
    fake_server_manager.ServerManager = object()
    fake_server_manager.ServerStatus = object()

    saved = {}
    for mod_name, stub in {
        "objc": fake_objc,
        "AppKit": fake_appkit,
        "Foundation": fake_foundation,
        "omlx_app.config": fake_config,
        "omlx_app.server_manager": fake_server_manager,
    }.items():
        saved[mod_name] = sys.modules.get(mod_name)
        sys.modules[mod_name] = stub

    sys.modules.pop("omlx_app", None)
    sys.modules.pop("omlx_app.app", None)
    monkeypatch.setattr("builtins.__import__", guarded_import)
    try:
        module = importlib.import_module("omlx_app.app")
        assert hasattr(module, "__version__")
    finally:
        for mod_name, prior in saved.items():
            if prior is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = prior


def test_omlx_app_init_loads_version_from_resources_layout(tmp_path, monkeypatch):
    """Bundle-like Resources layout should resolve omlx/_version.py correctly."""
    packaging_dir = Path(__file__).parent.parent / "packaging"
    monkeypatch.syspath_prepend(str(packaging_dir))

    sys.modules.pop("omlx_app", None)
    module = importlib.import_module("omlx_app")

    bundle_init = (
        tmp_path / "oMLX.app" / "Contents" / "Resources" / "omlx_app" / "__init__.py"
    )
    bundle_init.parent.mkdir(parents=True, exist_ok=True)
    bundle_init.write_text("# bundle stub\n", encoding="utf-8")

    version_file = tmp_path / "oMLX.app" / "Contents" / "Resources" / "omlx" / "_version.py"
    version_file.parent.mkdir(parents=True, exist_ok=True)
    version_file.write_text('__version__ = "9.9.9"\n', encoding="utf-8")

    monkeypatch.setattr(module, "__file__", str(bundle_init))

    assert module._load_version() == "9.9.9"


def test_omlx_app_init_logs_warning_when_version_fallback_used(
    tmp_path, monkeypatch, caplog
):
    """Missing version files should emit a warning before falling back to 0.0.0."""
    packaging_dir = Path(__file__).parent.parent / "packaging"
    monkeypatch.syspath_prepend(str(packaging_dir))

    sys.modules.pop("omlx_app", None)
    module = importlib.import_module("omlx_app")

    missing_init = tmp_path / "omlx_app" / "__init__.py"
    missing_init.parent.mkdir(parents=True, exist_ok=True)
    missing_init.write_text("# missing-version stub\n", encoding="utf-8")

    monkeypatch.setattr(module, "__file__", str(missing_init))

    with caplog.at_level("WARNING"):
        assert module._load_version() == "0.0.0"

    assert "Falling back to default oMLX version 0.0.0" in caplog.text
