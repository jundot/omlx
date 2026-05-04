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

        Shows a curses arrow-key picker when running in a TTY; falls back to
        numbered terminal selection when curses is unavailable (e.g. native
        Windows Python) or stdout is not a TTY.

        Returns the selected model id (empty string when models_info is empty).
        """
        if not models_info:
            return ""

        if len(models_info) == 1:
            return models_info[0]["id"]

        name = tool_name or "Tool"

        if sys.stdout.isatty():
            try:
                return _select_model_curses(models_info, name)
            except ImportError:
                # Stdlib curses missing (e.g. native windows python).
                pass
            except Exception:
                # Curses init/runtime failure (dumb terminal, no terminfo
                # entry, broken pipe, etc.). Fall through to numbered.
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


def _select_model_curses(models_info: list[dict], tool_name: str) -> str:
    """Show a fullscreen curses arrow-key picker for model selection.

    Loaded models appear first with a filled bullet; unloaded (available on
    disk) appear after with an empty bullet. Curses uses terminfo so this
    works reliably across SSH/PuTTY/tmux/screen, unlike inline ANSI TUIs.

    Raises ImportError if stdlib curses is not available.
    Returns the selected model id, or exits with 130 on cancel.
    """
    import curses
    import locale

    # Required so curses renders unicode bullets (●○) correctly.
    locale.setlocale(locale.LC_ALL, "")

    # Sort: loaded first, then unloaded. Default to False so a missing
    # "loaded" key (e.g. status fetch failed) renders as ○ rather than ●.
    loaded = [m for m in models_info if m.get("loaded", False)]
    unloaded = [m for m in models_info if not m.get("loaded", False)]
    ordered = loaded + unloaded

    selected: list[str] = []

    def _picker(stdscr) -> None:
        curses.curs_set(0)
        stdscr.keypad(True)
        idx = 0
        while True:
            stdscr.erase()
            try:
                stdscr.addstr(0, 1, f"oMLX > Launch {tool_name}", curses.A_BOLD)
                for i, m in enumerate(ordered):
                    bullet = "●" if m.get("loaded", False) else "○"
                    ctx = m.get("max_context_window")
                    ctx_str = f"  {ctx // 1000}k" if ctx else ""
                    line = f"  {bullet}  {m['id']}{ctx_str}"
                    attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                    stdscr.addstr(i + 2, 1, line, attr)
                stdscr.addstr(
                    len(ordered) + 3,
                    1,
                    "↑↓ navigate   Enter launch   q cancel",
                    curses.A_DIM,
                )
            except curses.error:
                # Window too small to render the full picker; keep going so
                # the user can resize and the next loop redraws cleanly.
                pass
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                idx = (idx - 1) % len(ordered)
            elif key in (curses.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(ordered)
            elif key in (curses.KEY_ENTER, 10, 13):
                selected.append(ordered[idx]["id"])
                return
            elif key in (ord("q"), 27):  # q or ESC
                return

    curses.wrapper(_picker)

    if not selected:
        print("No model selected.")
        # 130 is the conventional shell exit code for SIGINT/cancel.
        sys.exit(130)

    return selected[0]
