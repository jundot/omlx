# SPDX-License-Identifier: Apache-2.0
"""Expand a small seed palette into the 22 editable theme tokens.

This is the optional high-level layer for an external agent: pick a
background, text and accent (and optionally a surface and status colors) per
mode, and the server derives the rest, nudging text, links and the focus ring
until they pass. The result is checked like any other token set; the agent
can still send raw tokens instead.
"""

from __future__ import annotations

from .theme_contract import (
    CONTRAST_FLOOR,
    CUSTOM_THEME_TOKENS,
    MODES,
    THEME_HEX,
    TOKEN_GROUPS,
    best_foreground,
    ensure_foreground,
    mix,
    stock_colors,
)

REQUIRED_SEEDS = ("background", "text", "accent")
SEEDS = (*REQUIRED_SEEDS, "surface", *TOKEN_GROUPS["status"])


def check_seeds(seeds: dict) -> tuple[dict[str, dict[str, str]], list[dict]]:
    """Clean both modes' seeds; return them with every missing or bad seed."""
    clean: dict[str, dict[str, str]] = {mode: {} for mode in MODES}
    errors: list[dict] = []

    def error(code: str, mode: str, key: str, message: str) -> None:
        errors.append({"code": code, "mode": mode, "token": key, "message": message})

    for mode in MODES:
        values = seeds.get(mode) or {}
        for key in REQUIRED_SEEDS:
            if key not in values:
                error("missing_seed", mode, key, f"{mode}: missing seed {key!r}")
        for key, value in values.items():
            color = value.strip().lower() if isinstance(value, str) else value
            if key not in SEEDS:
                error("unknown_seed", mode, key, f"{mode}: unknown seed {key!r}")
            elif not isinstance(color, str) or not THEME_HEX.fullmatch(color):
                error("invalid_hex", mode, key, f"{key}: expected #rrggbb")
            else:
                clean[mode][key] = color
    return clean, errors


def _readable(
    color: str, backgrounds: tuple[str, ...], floor: float = CONTRAST_FLOOR
) -> str:
    # An unreachable floor keeps the color; check_theme then refuses it.
    return ensure_foreground(color, backgrounds, floor) or color


def build_mode(seed: dict[str, str], mode: str) -> dict[str, str]:
    """Derive one mode's full token map from its seed colors."""
    stock = stock_colors()[mode]
    background, text, accent = seed["background"], seed["text"], seed["accent"]
    surface = seed.get("surface") or mix(background, text, 0.04)
    tertiary = mix(surface, text, 0.06)
    surfaces = (background, surface, tertiary)
    text = _readable(text, surfaces)
    status = {name: seed.get(name) or stock[name] for name in TOKEN_GROUPS["status"]}
    tokens = {
        "bg-primary": background,
        "bg-secondary": surface,
        "bg-tertiary": tertiary,
        "text-primary": text,
        "text-secondary": _readable(mix(text, background, 0.25), surfaces),
        "text-tertiary": _readable(mix(text, background, 0.40), surfaces),
        "text-muted": mix(text, background, 0.50),
        # Borders are mostly background with a little text mixed in.
        "border-faint": mix(background, text, 0.15),
        "border-normal": mix(background, text, 0.30),
        "btn-primary": accent,
        "btn-primary-hover": mix(accent, text, 0.12),
        "btn-primary-text": best_foreground(accent),
        "switch-on": accent,
        "focus-ring-color": _readable(accent, (background,), 3.0),
        **status,
        "timeline-accent": accent,
        "link-color": _readable(accent, surfaces),
        "code-bg": mix(surface, text, 0.08),
        "text-danger": status["danger"],
    }
    return {token: tokens[token] for token in CUSTOM_THEME_TOKENS}


def build_recipe(seeds: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Build both modes from clean seeds (see check_seeds)."""
    return {mode: build_mode(seeds[mode], mode) for mode in MODES}
