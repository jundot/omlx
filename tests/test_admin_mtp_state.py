"""Admin models list passes the pool's load-time MTP resolution through (#3342).

``mtp_compatible`` / ``mtp_compatibility_reason`` are a disk verdict computed
at list time; ``mtp_requested`` / ``mtp_active`` / ``mtp_inactive_reason`` are
what the loaded engine is actually doing. Both ride the same row.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from omlx.admin import routes as admin_routes

MTP_KEYS = ("mtp_requested", "mtp_active", "mtp_inactive_reason")


def _list_models_with(status_entry: dict) -> dict:
    pool = MagicMock()
    pool.get_status.return_value = {"models": [status_entry]}
    pool._fallback_admission_ceiling.return_value = 0
    pool._scheduler_config = MagicMock(paged_ssd_cache_dir=None)
    manager = MagicMock()
    manager.get_all_settings.return_value = {}
    state = MagicMock(default_model=None)

    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch.object(admin_routes, "_get_settings_manager", return_value=manager),
        patch.object(admin_routes, "_get_server_state", return_value=state),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
        patch.object(
            admin_routes, "_dflash_compat_for_model", return_value=(False, "")
        ),
        patch.object(
            admin_routes,
            "_mtp_compat_for_model",
            return_value=(
                False,
                "Qwen4-Exp Lightning MTP requires embedded mtp.* tensors",
            ),
        ),
        patch.object(
            admin_routes, "_paroquant_compat_for_model", return_value=(False, "")
        ),
    ):
        return asyncio.run(admin_routes.list_models(is_admin=True))


def _status_entry(tmp_path, **mtp_fields) -> dict:
    model_path = tmp_path / "qwen4"
    model_path.mkdir(exist_ok=True)
    entry = {
        "id": "qwen4",
        "model_path": str(model_path),
        "config_model_type": "llama",
        "estimated_size": 1000,
        "loaded": True,
    }
    entry.update(mtp_fields)
    return entry


@pytest.mark.parametrize(
    "resolution",
    [
        (True, False, "no_embedded_mtp_tensors"),
        (True, True, None),
        (False, False, None),
        (True, None, None),
        (None, None, None),
    ],
)
def test_list_models_passes_pool_resolution_through(tmp_path, resolution):
    requested, active, reason = resolution
    result = _list_models_with(
        _status_entry(
            tmp_path,
            mtp_requested=requested,
            mtp_active=active,
            mtp_inactive_reason=reason,
        )
    )

    model = result["models"][0]
    assert (
        model["mtp_requested"],
        model["mtp_active"],
        model["mtp_inactive_reason"],
    ) == resolution
    # The disk verdict is unchanged and still rides alongside.
    assert model["mtp_compatible"] is False
    assert "embedded mtp.*" in model["mtp_compatibility_reason"]


def test_list_models_omits_nothing_when_pool_predates_the_fields(tmp_path):
    # A status entry without the keys (older pool, stubbed pool) still yields
    # the keys as None rather than KeyError -- "not resolved".
    result = _list_models_with(_status_entry(tmp_path))
    model = result["models"][0]
    assert all(model[key] is None for key in MTP_KEYS)


def test_markitdown_builtin_row_carries_the_keys_as_none(tmp_path):
    result = _list_models_with(_status_entry(tmp_path))
    builtin = [
        m for m in result["models"] if m["id"] == admin_routes.MARKITDOWN_MODEL_ID
    ]
    if not builtin:
        pytest.skip("markitdown builtin row not present in this build")
    assert all(builtin[0][key] is None for key in MTP_KEYS)


def test_models_status_docstring_states_the_null_contract():
    from omlx.server import list_models_status

    doc = list_models_status.__doc__ or ""
    assert "mtp_active" in doc
    assert "null" in doc
    assert "not resolved" in doc
