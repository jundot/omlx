# SPDX-License-Identifier: Apache-2.0
"""Browser admin pages for oMLX.

The pages read and change server state only through the HTTP admin API.
Request-time values come from the WebUIHost that the server sets, so this
module does not import omlx.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates


@dataclass(frozen=True)
class WebUIHost:
    """Server callbacks used to render the admin pages."""

    version: str
    require_admin: Callable[[Request], Awaitable[bool]]
    is_admin: Callable[[Request], bool]
    ui_language: Callable[[], str]
    main_api_key: Callable[[], str | None]


router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
static_dir = Path(__file__).parent / "static"
_host: WebUIHost | None = None


def _static_version(path: str) -> str:
    """Append file mtime as query string for cache busting."""
    file_path = static_dir / path
    if file_path.is_file():
        mtime = int(file_path.stat().st_mtime)
        return f"/admin/static/{path}?v={mtime}"
    return f"/admin/static/{path}"


templates.env.globals["static"] = _static_version

# i18n defaults (English). _sync_locale() switches them per render.
_i18n_dir = Path(__file__).parent / "i18n"
_en_locale: dict = {}
try:
    _en_locale = json.loads((_i18n_dir / "en.json").read_text(encoding="utf-8"))
except Exception:
    pass
templates.env.globals["t"] = lambda key: _en_locale.get(key, key)
templates.env.globals["locale_json"] = json.dumps(_en_locale, ensure_ascii=False)
templates.env.globals["current_lang"] = "en"


def _load_locale(language: str) -> dict:
    """Load locale dict and fill missing keys from English."""
    fallback = dict(_en_locale)
    path = _i18n_dir / f"{language}.json"
    if language == "en":
        return fallback
    try:
        locale = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads((_i18n_dir / "en.json").read_text(encoding="utf-8"))
        except Exception:
            return {}
    fallback.update(locale)
    return fallback


def _make_t(locale: dict):
    """Return a Jinja2-compatible t() function for the given locale dict."""

    def t(key: str) -> str:
        return locale.get(key, key)

    return t


def _sync_locale() -> None:
    """Load the host UI language into the template globals if it changed."""
    lang = _host.ui_language()
    if lang == templates.env.globals["current_lang"]:
        return
    locale = _load_locale(lang)
    templates.env.globals["t"] = _make_t(locale)
    templates.env.globals["locale_json"] = json.dumps(locale, ensure_ascii=False)
    templates.env.globals["current_lang"] = lang


def set_host(host: WebUIHost) -> None:
    """Connect the pages to the running server."""
    global _host
    _host = host
    templates.env.globals["version"] = host.version


async def require_admin(request: Request) -> bool:
    """Apply the host admin dependency to a page."""
    return await _host.require_admin(request)


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    """
    Render the admin login page or setup page.

    If no API key is configured, the page will show the initial setup form.
    Otherwise, it shows the standard login form.

    Returns:
        HTML login/setup page.
    """
    if _host.is_admin(request):
        return RedirectResponse(url="/admin/dashboard", status_code=302)

    _sync_locale()
    return templates.TemplateResponse(
        request,
        "login.html",
        {"api_key_configured": bool(_host.main_api_key())},
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, is_admin: bool = Depends(require_admin)):
    """
    Render the admin dashboard page.

    Requires admin authentication via session cookie.

    Returns:
        HTML dashboard page with server status and model list.
    """
    _sync_locale()
    return templates.TemplateResponse(request, "dashboard.html", {})


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, is_admin: bool = Depends(require_admin)):
    """
    Render the chat page for interacting with models.

    Requires admin authentication via session cookie.
    The API key is injected into the template context so that
    the chat page can auto-set it in localStorage, bypassing
    the manual API key entry modal.

    Returns:
        HTML chat page.
    """
    _sync_locale()
    api_key = _host.main_api_key()
    return templates.TemplateResponse(request, "chat.html", {"api_key": api_key or ""})


@router.get("/static/{path:path}")
async def admin_static(path: str):
    """Serve static files for admin panel (CSS, JS, fonts, logos, etc.)."""
    file_path = static_dir / path
    if not file_path.is_file() or not file_path.resolve().is_relative_to(
        static_dir.resolve()
    ):
        raise HTTPException(status_code=404, detail="File not found")
    media_types = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".ico": "image/x-icon",
        ".css": "text/css",
        ".js": "application/javascript",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".ttf": "font/ttf",
    }
    media_type = media_types.get(file_path.suffix, "application/octet-stream")
    return FileResponse(file_path, media_type=media_type)
