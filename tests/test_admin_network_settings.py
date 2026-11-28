"""Network settings applied from the admin panel reach model downloads.

huggingface_hub shares one HTTP client that reads the proxy and CA bundle
environment when it is created. Applying network settings must make the
next download build a new client, without closing the one a download may
still be streaming through.
"""

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock

import certifi
import pytest
from huggingface_hub.utils import _http as hf_http
from huggingface_hub.utils import get_session

import omlx.admin.routes as admin_routes
import omlx.server  # noqa: F401  (registers the admin getters)
from omlx.admin.routes import GlobalSettingsRequest

_ENV_KEYS = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
)


def _make_global_settings():
    gs = MagicMock()
    gs.server.host = "127.0.0.1"
    gs.server.port = 8000
    gs.server.log_level = "info"
    gs.server.server_aliases = []
    gs.auth.api_key = None
    gs.auth.skip_api_key_verification = False
    gs.validate.return_value = []
    gs.save.return_value = None
    return gs


@contextmanager
def _patched_global_settings(gs):
    original = admin_routes._get_global_settings
    admin_routes._get_global_settings = lambda: gs
    try:
        yield
    finally:
        admin_routes._get_global_settings = original


@pytest.fixture
def clean_hf_session(monkeypatch):
    """Isolate the shared huggingface_hub client and the proxy env."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    saved = hf_http._GLOBAL_CLIENT
    hf_http._GLOBAL_CLIENT = None
    created = []
    yield created
    for client in created:
        client.close()
    hf_http._GLOBAL_CLIENT = saved


def _apply(request):
    with _patched_global_settings(_make_global_settings()):
        return asyncio.run(
            admin_routes.update_global_settings(request=request, is_admin=True)
        )


@pytest.mark.parametrize(
    "request_fields",
    [
        {"network_https_proxy": "http://127.0.0.1:3128"},
        {"network_http_proxy": "http://127.0.0.1:3128"},
        {"network_no_proxy": "localhost,127.0.0.1"},
        {"network_ca_bundle": certifi.where()},
    ],
)
def test_network_change_gives_downloads_a_new_client(clean_hf_session, request_fields):
    before = get_session()
    clean_hf_session.append(before)

    result = _apply(GlobalSettingsRequest(**request_fields))

    after = get_session()
    clean_hf_session.append(after)
    assert result["success"] is True
    assert "network" in result["runtime_applied"]
    assert after is not before


def test_network_change_leaves_the_old_client_open(clean_hf_session):
    before = get_session()
    clean_hf_session.append(before)

    _apply(GlobalSettingsRequest(network_https_proxy="http://127.0.0.1:3128"))

    assert before.is_closed is False


def test_other_settings_keep_the_shared_client(clean_hf_session):
    before = get_session()
    clean_hf_session.append(before)

    _apply(GlobalSettingsRequest())

    assert get_session() is before
