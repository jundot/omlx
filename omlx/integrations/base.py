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
        scroll = 0
        hint = "↑↓ navigate   PgUp/PgDn page   Enter launch   q cancel"
        while True:
            max_y, max_x = stdscr.getmaxyx()
            # Layout: row 0 title, row 1 blank, rows [items_top, items_bottom)
            # for items, last row pinned for the hint.
            items_top = 2
            items_bottom = max(items_top + 1, max_y - 1)
            visible_count = items_bottom - items_top

            # Keep the cursor visible inside the viewport.
            if idx < scroll:
                scroll = idx
            elif idx >= scroll + visible_count:
                scroll = idx - visible_count + 1
            max_scroll = max(0, len(ordered) - visible_count)
            scroll = max(0, min(scroll, max_scroll))
            visible_end = min(len(ordered), scroll + visible_count)

            stdscr.erase()
            try:
                stdscr.addstr(
                    0, 1, f"oMLX > Launch {tool_name}"[: max_x - 2], curses.A_BOLD
                )
                for row_offset, i in enumerate(range(scroll, visible_end)):
                    m = ordered[i]
                    bullet = "●" if m.get("loaded", False) else "○"
                    ctx = m.get("max_context_window")
                    ctx_str = f"  {ctx // 1000}k" if ctx else ""
                    line = f"  {bullet}  {m['id']}{ctx_str}"
                    # Leave 2 cols on the right for scroll indicators.
                    line = line[: max(0, max_x - 4)]
                    attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                    stdscr.addstr(items_top + row_offset, 1, line, attr)
                # Scroll indicators on the right edge.
                if scroll > 0:
                    stdscr.addstr(items_top, max_x - 2, "▲", curses.A_DIM)
                if visible_end < len(ordered):
                    stdscr.addstr(items_bottom - 1, max_x - 2, "▼", curses.A_DIM)
                stdscr.addstr(max_y - 1, 1, hint[: max_x - 2], curses.A_DIM)
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
            elif key == curses.KEY_PPAGE:
                idx = max(0, idx - max(1, visible_count - 1))
            elif key == curses.KEY_NPAGE:
                idx = min(len(ordered) - 1, idx + max(1, visible_count - 1))
            elif key == curses.KEY_HOME:
                idx = 0
            elif key == curses.KEY_END:
                idx = len(ordered) - 1
            elif key in (curses.KEY_ENTER, 10, 13):
                selected.append(ordered[idx]["id"])
                return
            elif key in (ord("q"), 27):  # q or ESC
                return
            elif key == curses.KEY_RESIZE:
                # Re-query getmaxyx on the next loop iteration.
                continue

    curses.wrapper(_picker)

    if not selected:
        print("No model selected.")
        # 130 is the conventional shell exit code for SIGINT/cancel.
        sys.exit(130)

    return selected[0]
