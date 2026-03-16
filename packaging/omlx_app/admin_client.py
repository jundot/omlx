"""Admin API client for oMLX menubar app."""

from typing import Optional

import requests


class AdminStatsClient:
    """Manages a persistent admin session for fetching server stats.

    Reuses the session cookie across calls so /admin/api/login is only
    called once (or after server restart / session expiry).
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._session: Optional[requests.Session] = None

    def _login(self) -> bool:
        """Create a new session and authenticate. Returns True on success."""
        session = requests.Session()
        try:
            resp = session.post(
                f"{self._base_url}/admin/api/login",
                json={"api_key": self._api_key},
                timeout=2,
            )
        except requests.RequestException:
            return False
        if resp.status_code != 200:
            return False
        self._session = session
        return True

    def fetch_stats(self, scope: str = "session", model: str = "") -> Optional[dict]:
        """Fetch stats, logging in if needed. Returns None on failure."""
        if self._session is None and not self._login():
            return None

        params: dict = {}
        if scope != "session":
            params["scope"] = scope
        if model:
            params["model"] = model

        try:
            resp = self._session.get(
                f"{self._base_url}/admin/api/stats",
                params=params,
                timeout=2,
            )
        except requests.RequestException:
            self._session = None
            return None

        if resp.status_code == 401:
            self._session = None
            if not self._login():
                return None
            try:
                resp = self._session.get(
                    f"{self._base_url}/admin/api/stats",
                    params=params,
                    timeout=2,
                )
            except requests.RequestException:
                self._session = None
                return None

        if resp.status_code == 200:
            return resp.json()
        return None

    def invalidate(self) -> None:
        """Discard the current session (e.g. server restarted)."""
        self._session = None
