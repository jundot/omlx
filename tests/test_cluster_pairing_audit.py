# SPDX-License-Identifier: Apache-2.0
"""Pairing failures expose diagnostic metadata without secrets."""

import json
from urllib.error import HTTPError, URLError

import pytest

from omlx.cluster.pairing import PairingRequestError
from tests.test_cluster_pairing import _loopback_pair


@pytest.mark.parametrize(
    "error, expected",
    [
        (
            URLError(ConnectionRefusedError("private-host secret-value")),
            {
                "error_type": "URLError",
                "reason_type": "ConnectionRefusedError",
                "http_status": None,
            },
        ),
        (
            HTTPError(
                "http://private-host/secret-value", 503, "private detail", {}, None
            ),
            {"error_type": "HTTPError", "reason_type": "str", "http_status": 503},
        ),
        (
            TimeoutError("private-host secret-value"),
            {"error_type": "TimeoutError", "reason_type": None, "http_status": None},
        ),
    ],
)
def test_failed_join_audits_metadata_without_private_error_text(
    tmp_path, error, expected
):
    _, joiner, *_ = _loopback_pair(tmp_path)
    events = []
    joiner._record_audit = lambda event, **kwargs: events.append((event, kwargs))

    def fail(*args):
        raise error

    joiner._http_post = fail
    with pytest.raises(PairingRequestError):
        joiner.ui_session.begin("coordinator:8000")
    failures = [details for event, details in events if event == "join_request_failed"]
    assert len(failures) == 1
    assert failures[0]["detail"] == expected
    serialized = json.dumps(failures)
    assert "private-host" not in serialized and "secret-value" not in serialized
    token = joiner.ui_session.attempt.get("cancel_token")
    if token:
        assert token not in serialized
    assert joiner.ui_session.snapshot()["state"] == "error"
