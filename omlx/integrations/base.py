"""Base class for external tool integrations."""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Integration:
    """Base integration definition."""

    name: str  # "codex", "opencode", "openclaw", "pi"
    display_name: str  # "Codex", "OpenCode", "OpenClaw", "Pi"
    type: str  # "env_var" or "config_file"
    install_check: str  # binary name to check with `which`
    install_hint: str  # installation instructions

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        """Generate the command string for clipboard/display."""
        raise NotImplementedError

    def configure(self, port: int, api_key: str, model: str, host: str = "127.0.0.1") -> None:
        """Configure the tool (write config files, etc.)."""
        pass

    def launch(self, port: int, api_key: str, model: str, host: str = "127.0.0.1", **kwargs) -> None:
        """Configure and launch the tool."""
        raise NotImplementedError

    def is_installed(self) -> bool:
        """Check if the tool binary is available."""
        return shutil.which(self.install_check) is not None

    def select_model(
        self, models_info: list[dict], tool_name: str | None = None
    ) -> str:
        """Select a model interactively.

        Shows a textual TUI when running in a TTY; falls back to numbered
        terminal selection when textual is unavailable or stdout is not a TTY.

        Returns the selected model id (empty string when models_info is empty).
        """
        if not models_info:
            return ""

        if len(models_info) == 1:
            return models_info[0]["id"]

        name = tool_name or "Tool"

        if sys.stdout.isatty():
            try:
                return _select_model_tui(models_info, name)
            except ImportError:
                pass

        # Fallback: numbered terminal selection
        print("Available models:")
        for i, m in enumerate(models_info, 1):
            ctx = m.get("max_context_window")
            ctx_str = f"  [{ctx:,} ctx]" if ctx else ""
            print(f"  {i}. {m['id']}{ctx_str}")
        while True:
            try:
                choice = input("Select model number: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(models_info):
                    return models_info[idx]["id"]
                print(f"Please enter 1-{len(models_info)}")
            except (ValueError, EOFError):
                print(f"Please enter 1-{len(models_info)}")

    def _write_json_config(
        self,
        config_path: Path,
        updater: callable,
    ) -> None:
        """Read, update, and write a JSON config file with backup.

        Args:
            config_path: Path to the config file.
            updater: Function that takes existing config dict and modifies it in-place.
        """
        existing: dict = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: could not parse {config_path}: {e}")
                print("Creating new config file.")
                existing = {}

            # Create timestamped backup
            timestamp = int(time.time())
            backup = config_path.with_suffix(f".{timestamp}.bak")
            try:
                shutil.copy2(config_path, backup)
                print(f"Backup: {backup}")
            except OSError as e:
                print(f"Warning: could not create backup: {e}")

        updater(existing)

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Config written: {config_path}")


def _select_model_tui(models_info: list[dict], tool_name: str) -> str:
    """Show a compact inline textual TUI for interactive model selection.

    Loaded models appear first with a filled bullet; unloaded (available on disk)
    appear after with an empty bullet and a warning on selection.

    Raises ImportError if textual is not installed.
    Returns the selected model id, or exits with 130 on cancel.
    """
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.widgets import Label, ListItem, ListView

    # Sort: loaded first, then unloaded. Default to False so a missing
    # "loaded" key (e.g. status fetch failed) renders as ○ rather than ●.
    loaded = [m for m in models_info if m.get("loaded", False)]
    unloaded = [m for m in models_info if not m.get("loaded", False)]
    ordered = loaded + unloaded

    result: list[str] = []

    class ModelSelectorApp(App):
        ENABLE_COMMAND_PALETTE = False
        CSS = """
        Screen {
            background: transparent;
            height: auto;
            max-height: 20;
        }
        #title {
            padding: 0 2;
            text-style: bold;
            height: 1;
        }
        ListView {
            border: none;
            background: transparent;
            height: auto;
            max-height: 15;
        }
        ListItem {
            padding: 0 1;
            height: 1;
            background: transparent;
        }
        ListItem:focus-within {
            background: $accent 20%;
        }
        #hint {
            padding: 0 2;
            height: 1;
        }
        """

        BINDINGS = [
            Binding("q", "quit_cancel", "Cancel", show=False),
            Binding("escape", "quit_cancel", "Cancel", show=False),
        ]

        def compose(self) -> ComposeResult:
            yield Label(f" oMLX › Launch {tool_name}", id="title")
            items = []
            for m in ordered:
                is_loaded = m.get("loaded", False)
                bullet = "●" if is_loaded else "○"
                ctx = m.get("max_context_window")
                ctx_str = f"  {ctx // 1000}k" if ctx else ""
                label = f" {bullet} {m['id']}{ctx_str}"
                items.append(ListItem(Label(label)))
            yield ListView(*items)
            yield Label(" ↑↓ navigate   Enter launch   q cancel", id="hint")

        def on_list_view_selected(self, event: ListView.Selected) -> None:
            idx = event.list_view.index
            if idx is not None and 0 <= idx < len(ordered):
                result.append(ordered[idx]["id"])
            self.exit()

        def action_quit_cancel(self) -> None:
            self.exit()

    app = ModelSelectorApp()
    app.run(inline=True)

    if not result:
        print("No model selected.")
        # 130 is the conventional shell exit code for SIGINT/cancel.
        sys.exit(130)

    return result[0]
