# SPDX-License-Identifier: Apache-2.0
"""The web UI's editable color tokens and the contrast floor they must keep.

The settings editor and the server-side validation read the same token list
and contrast rules. Keep TOKEN_GROUPS and CONTRAST_RULES in sync with
static/js/custom_theme.js; apps/omlx-web/tests/custom_theme_ui.test.cjs
checks both.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources

MODES = ("light", "dark")
THEME_HEX = re.compile(r"#[0-9a-f]{6}")
# WCAG AA for body text. Every rule below is a hard floor on every save path.
CONTRAST_FLOOR = 4.5

TOKEN_GROUPS = {
    "surfaces": ("bg-primary", "bg-secondary", "bg-tertiary"),
    "text": ("text-primary", "text-secondary", "text-tertiary", "text-muted"),
    "borders": ("border-faint", "border-normal"),
    "controls": (
        "btn-primary",
        "btn-primary-hover",
        "btn-primary-text",
        "switch-on",
        "focus-ring-color",
    ),
    "status": ("success", "warning", "danger", "info"),
    "accents": ("timeline-accent", "link-color", "code-bg", "text-danger"),
}
CUSTOM_THEME_TOKENS = tuple(
    token for tokens in TOKEN_GROUPS.values() for token in tokens
)

SURFACES = TOKEN_GROUPS["surfaces"]
# (foreground, backgrounds it must stay readable on)
CONTRAST_RULES = (
    ("text-primary", SURFACES),
    ("text-secondary", SURFACES),
    ("link-color", SURFACES),
    ("btn-primary-text", ("btn-primary",)),
)


def _channels(color: str) -> tuple[int, ...]:
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))


def contrast_ratio(first: str, second: str) -> float:
    """WCAG 2 contrast ratio of two opaque #rrggbb colors."""

    def luminance(color: str) -> float:
        linear = (
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
            for value in (channel / 255 for channel in _channels(color))
        )
        return sum(w * c for w, c in zip((0.2126, 0.7152, 0.0722), linear))

    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@lru_cache(maxsize=1)
def stock_colors() -> dict[str, dict[str, str]]:
    """Resolve every editable token from theme.css.

    Raises instead of returning a partial map: without the stock colors the
    contrast floor cannot be checked, and a save must not skip it.
    """
    css = (resources.files("omlx_web") / "static/css/theme.css").read_text(
        encoding="utf-8"
    )
    palette = {
        name: "#" + "".join(f"{int(step):02x}" for step in steps.split())
        for name, steps in re.findall(r"--palette-([a-z0-9-]+):\s*(\d+ \d+ \d+);", css)
    }
    resolved: dict[str, dict[str, str]] = {}
    for mode, selector in zip(MODES, (":root", '[data-theme="dark"]')):
        block = re.search(rf"{re.escape(selector)}\s*\{{(.*?)\}}", css, re.DOTALL)
        if block is None:
            raise ValueError(f"theme.css has no {selector} block")
        resolved[mode] = {}
        for token in CUSTOM_THEME_TOKENS:
            match = re.search(
                rf"--{re.escape(token)}:\s*rgb\(var\(--palette-([a-z0-9-]+)\)\)",
                block.group(1),
            )
            if match is None:
                raise ValueError(f"theme.css does not define --{token} in {mode}")
            resolved[mode][token] = palette[match.group(1)]
    return resolved


def _error(code: str, mode: str, token: str, message: str, **extra) -> dict:
    return {"code": code, "mode": mode, "token": token, "message": message, **extra}


def check_theme(theme: dict | None) -> tuple[dict[str, dict[str, str]], list[dict]]:
    """Clean a {"light": {...}, "dark": {...}} override map.

    Returns the lowercased overrides and every error: unknown tokens, values
    that are not opaque #rrggbb, and contrast-rule pairs below the floor once
    the overrides are merged over the stock colors.
    """
    stock = stock_colors()
    clean: dict[str, dict[str, str]] = {mode: {} for mode in MODES}
    errors: list[dict] = []
    for mode in MODES:
        for token, value in ((theme or {}).get(mode) or {}).items():
            color = value.strip().lower() if isinstance(value, str) else value
            if token not in CUSTOM_THEME_TOKENS:
                message = f"unknown theme token {token!r}"
                errors.append(_error("unknown_token", mode, token, message))
            elif not isinstance(color, str) or not THEME_HEX.fullmatch(color):
                errors.append(
                    _error(
                        "invalid_hex",
                        mode,
                        token,
                        f"{token}: expected #rrggbb, got {value!r}",
                        value=value,
                    )
                )
            else:
                clean[mode][token] = color
        values = {**stock[mode], **clean[mode]}
        for foreground, backgrounds in CONTRAST_RULES:
            for background in backgrounds:
                ratio = contrast_ratio(values[foreground], values[background])
                if ratio < CONTRAST_FLOOR:
                    errors.append(
                        _error(
                            "contrast_below_floor",
                            mode,
                            foreground,
                            f"{mode}: {foreground} on {background} is "
                            f"{ratio:.2f}:1, below {CONTRAST_FLOOR}:1",
                            surface=background,
                            ratio=round(ratio, 2),
                            floor=CONTRAST_FLOOR,
                        )
                    )
    return clean, errors
