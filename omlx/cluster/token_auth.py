# SPDX-License-Identifier: Apache-2.0
"""Authentication helpers for the out-of-band cluster pairing flow."""

from __future__ import annotations

import hashlib
import hmac

MIN_PAIRING_SECRET_LENGTH = 16
MAX_PAIRING_SECRET_LENGTH = 256


def validate_pairing_secret(value: str) -> str:
    """Return a usable shared secret or reject weak/oversized input."""

    if not isinstance(value, str):
        raise ValueError("pairing secret must be text")
    if value != value.strip():
        raise ValueError("pairing secret cannot start or end with whitespace")
    if not MIN_PAIRING_SECRET_LENGTH <= len(value) <= MAX_PAIRING_SECRET_LENGTH:
        raise ValueError(
            "pairing secret must be between "
            f"{MIN_PAIRING_SECRET_LENGTH} and {MAX_PAIRING_SECRET_LENGTH} characters"
        )
    return value


def sign_pairing_payload(payload: str, *, shared_secret: str) -> str:
    """Authenticate a serialized pairing payload with the out-of-band secret."""

    secret = validate_pairing_secret(shared_secret)
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_pairing_signature(
    payload: str,
    signature: str,
    *,
    shared_secret: str,
) -> bool:
    """Verify a pairing MAC without leaking comparison timing."""

    if not isinstance(signature, str):
        return False
    try:
        expected = sign_pairing_payload(payload, shared_secret=shared_secret)
    except ValueError:
        return False
    return hmac.compare_digest(signature, expected)
