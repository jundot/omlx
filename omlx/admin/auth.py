# SPDX-License-Identifier: Apache-2.0
"""Authentication utilities for the oMLX admin panel.

This module provides session-based authentication using signed tokens
and API key verification for admin panel access.
"""

import os
import secrets
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

# Session configuration
SESSION_COOKIE_NAME = "omlx_admin_session"
SESSION_MAX_AGE = 86400  # 24 hours in seconds

# Secret key for signing session tokens
# Use environment variable if set, otherwise generate a random key
# Note: Random key means sessions won't persist across server restarts
SECRET_KEY = os.environ.get("OMLX_SECRET_KEY") or secrets.token_hex(32)

# Initialize the serializer for creating and verifying session tokens
_serializer = URLSafeTimedSerializer(SECRET_KEY)


def create_session_token() -> str:
    """Create a signed session token for admin authentication.

    Returns:
        A URL-safe signed token string containing admin session data.

    Example:
        >>> token = create_session_token()
        >>> verify_session_token(token)
        True
    """
    payload = {"admin": True}
    return _serializer.dumps(payload)


def verify_session_token(token: str, max_age: int = SESSION_MAX_AGE) -> bool:
    """Verify and decode a session token.

    Args:
        token: The signed session token to verify.
        max_age: Maximum age of the token in seconds. Defaults to 24 hours.

    Returns:
        True if the token is valid and not expired, False otherwise.

    Example:
        >>> token = create_session_token()
        >>> verify_session_token(token)
        True
        >>> verify_session_token("invalid_token")
        False
    """
    try:
        data = _serializer.loads(token, max_age=max_age)
        return data.get("admin", False) is True
    except (BadSignature, SignatureExpired):
        return False


def verify_api_key(api_key: str, server_api_key: str) -> bool:
    """Verify an API key using constant-time comparison.

    This function uses secrets.compare_digest to prevent timing attacks
    when comparing the provided API key with the server's API key.

    Args:
        api_key: The API key provided by the client.
        server_api_key: The server's configured API key.

    Returns:
        True if the API keys match, False otherwise.

    Example:
        >>> verify_api_key("secret123", "secret123")
        True
        >>> verify_api_key("wrong", "secret123")
        False
    """
    if not api_key or not server_api_key:
        return False
    return secrets.compare_digest(api_key, server_api_key)


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """Validate API key format requirements.

    Rules:
    - Minimum 4 characters
    - No whitespace characters (space, tab, newline, etc.)
    - Printable characters only (no control characters)

    Args:
        api_key: The API key string to validate.

    Returns:
        Tuple of (is_valid, error_message). Error message is empty if valid.
    """
    if len(api_key) < 4:
        return False, "API key must be at least 4 characters"
    if any(c.isspace() for c in api_key):
        return False, "API key must not contain whitespace"
    if not api_key.isprintable():
        return False, "API key must contain only printable characters"
    return True, ""


def verify_session(request: Request) -> bool:
    """Verify if the request has a valid admin session.

    Checks for a valid session cookie in the request.

    Args:
        request: The FastAPI request object.

    Returns:
        True if the session is valid, False otherwise.
    """
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return False
    return verify_session_token(token)


async def require_admin(request: Request) -> bool:
    """FastAPI dependency to require admin authentication.

    This dependency can be used in route definitions to protect
    admin-only endpoints. It checks for a valid session cookie.

    Args:
        request: The FastAPI request object (injected by FastAPI).

    Returns:
        True if authentication is successful.

    Raises:
        HTTPException: 401 Unauthorized if not authenticated.

    Example:
        >>> from fastapi import Depends
        >>> @app.get("/admin/settings")
        ... async def get_settings(is_admin: bool = Depends(require_admin)):
        ...     return {"settings": "..."}
    """
    if not verify_session(request):
        # Browser requests (Accept: text/html) get redirected to login page
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            raise _RedirectToLogin()
        raise HTTPException(
            status_code=401,
            detail="Admin authentication required",
            headers={"WWW-Authenticate": "Cookie"},
        )
    return True


class _RedirectToLogin(Exception):
    """Raised to trigger a redirect to the admin login page."""
    pass
