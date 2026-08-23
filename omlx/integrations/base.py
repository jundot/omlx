"""Base class for external tool integrations."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IntegrationContext:
    """Resolved launch/configuration inputs for an external tool integration."""

    host: str
    port: int
    api_key: str = ""
    model: str = ""
    opus_model: str | None = None
    sonnet_model: str | None = None
    haiku_model: str | None = None
    context_window: int | None = None
    max_tokens: int | None = None
    model_type: str | None = None
    reasoning: bool | None = None
    tools_profile: str = "coding"
    extra_args: tuple[str, ...] = ()
    cross_session: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def openai_base_url(self) -> str:
        return f"{self.base_url}/v1"

    @property
    def auth_token(self) -> str:
        return self.api_key or "omlx"

    @property
    def supports_images(self) -> bool:
        return self.model_type == "vlm"


@dataclass
class Integration:
    """Base integration definition."""

    name: str
    display_name: str
    type: str
    install_check: str
    install_hint: str

    def get_command(self, ctx: IntegrationContext) -> str:
        """Generate the command string for clipboard/display."""
        raise NotImplementedError

    def configure(self, ctx: IntegrationContext) -> None:
        """Configure the tool (write config files, etc.)."""
        pass

    def launch(self, ctx: IntegrationContext) -> None:
        """Configure and launch the tool."""
        raise NotImplementedError

    def is_installed(self) -> bool:
        """Check if the tool binary is available."""
        return shutil.which(self.install_check) is not None

    def model_disabled_reason(self, model_info: dict) -> str | None:
        """Return why a model cannot be selected for this integration."""
        return None

    def _scrubbed_env(self) -> dict[str, str]:
        """Return an os.environ copy with bundled-Python vars removed."""
        env = os.environ.copy()
        for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE"):
            env.pop(key, None)
        return env

    def select_model(
        self, models_info: list[dict], tool_name: str | None = None
    ) -> str:
        """Select a model interactively."""
        if not models_info:
            return ""

        annotated_models = []
        for model_info in models_info:
            model_info = dict(model_info)
            reason = self.model_disabled_reason(model_info)
            if reason:
                model_info["disabled_reason"] = reason
            annotated_models.append(model_info)
        models_info = annotated_models

        if len(models_info) == 1:
            reason = models_info[0].get("disabled_reason")
            if reason:
                print(f"Cannot select {models_info[0]['id']}: {reason}")
                sys.exit(1)
            return models_info[0]["id"]

        name = tool_name or "Tool"

        if sys.stdout.isatty():
            try:
                return _select_model_curses(models_info, name)
            except ImportError:
                pass
            except Exception:
                pass

        print("Available models:")
        for i, m in enumerate(models_info, 1):
            ctx = m.get("max_context_window")
            ctx_str = f"  [{ctx:,} ctx]" if ctx else ""
            reason = m.get("disabled_reason")
            unavailable = f"  [unavailable: {reason}]" if reason else ""
            print(f"  {i}. {m['id']}{ctx_str}{unavailable}")
        while True:
            try:
                choice = input("Select model number: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(models_info):
                    reason = models_info[idx].get("disabled_reason")
                    if reason:
                        print(f"Cannot select {models_info[idx]['id']}: {reason}")
                        continue
                    return models_info[idx]["id"]
                print(f"Please enter 1-{len(models_info)}")
            except (ValueError, EOFError):
                print(f"Please enter 1-{len(models_info)}")

    def _write_json_config(
        self,
        config_path: Path,
        updater: callable,
        *,
        backup: bool = True,
    ) -> None:
        """Read, update, and write a JSON config file.

        Args:
            config_path: Path to the config file.
            updater: Function that takes existing config dict and modifies it in-place.
            backup: Whether to create a timestamped backup of an existing file.
        """
        existing: dict = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: could not parse {config_path}: {e}")
                print("Creating new config file.")
                existing = {}

            if backup:
                timestamp = int(time.time())
                backup_path = config_path.with_suffix(f".{timestamp}.bak")
                try:
                    shutil.copy2(config_path, backup_path)
                    print(f"Backup: {backup_path}")
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
    """Show a fullscreen curses arrow-key picker for model selection."""
    import curses
    import locale

    locale.setlocale(locale.LC_ALL, "")
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
        warning = ""
        while True:
            max_y, max_x = stdscr.getmaxyx()
            items_top = 2
            items_bottom = max(items_top + 1, max_y - 1)
            visible_count = items_bottom - items_top

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
                    unavailable = "  unavailable" if m.get("disabled_reason") else ""
                    line = f"  {bullet}  {m['id']}{ctx_str}{unavailable}"
                    line = line[: max(0, max_x - 4)]
                    attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                    if m.get("disabled_reason"):
                        attr |= curses.A_DIM
                    stdscr.addstr(items_top + row_offset, 1, line, attr)
                if scroll > 0:
                    stdscr.addstr(items_top, max_x - 2, "▲", curses.A_DIM)
                if visible_end < len(ordered):
                    stdscr.addstr(items_bottom - 1, max_x - 2, "▼", curses.A_DIM)
                footer = warning or hint
                stdscr.addstr(max_y - 1, 1, footer[: max_x - 2], curses.A_DIM)
            except curses.error:
                pass
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                warning = ""
                idx = (idx - 1) % len(ordered)
            elif key in (curses.KEY_DOWN, ord("j")):
                warning = ""
                idx = (idx + 1) % len(ordered)
            elif key == curses.KEY_PPAGE:
                warning = ""
                idx = max(0, idx - max(1, visible_count - 1))
            elif key == curses.KEY_NPAGE:
                warning = ""
                idx = min(len(ordered) - 1, idx + max(1, visible_count - 1))
            elif key == curses.KEY_HOME:
                warning = ""
                idx = 0
            elif key == curses.KEY_END:
                warning = ""
                idx = len(ordered) - 1
            elif key in (curses.KEY_ENTER, 10, 13):
                reason = ordered[idx].get("disabled_reason")
                if reason:
                    warning = f"Cannot select {ordered[idx]['id']}: {reason}"
                    continue
                selected.append(ordered[idx]["id"])
                return
            elif key in (ord("q"), 27):
                return
            elif key == curses.KEY_RESIZE:
                continue

    curses.wrapper(_picker)

    if not selected:
        print("No model selected.")
        sys.exit(130)

    return selected[0]
