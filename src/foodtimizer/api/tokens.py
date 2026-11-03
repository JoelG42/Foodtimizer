"""Bearer tokens for the phone API.

Signed with HMAC-SHA256 so the API can check them without a session store.
Sign in with Apple (later) should verify Apple's identity token, then call
:func:`issue_token` with that account's ``sub``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


class InvalidToken(Exception):
    """The bearer token is missing, altered, or expired."""


def issue_token(
    secret: str,
    sub: str,
    email: str | None = None,
    *,
    ttl_seconds: int,
    now: float | None = None,
) -> str:
    """Return a token that :func:`read_token` will accept until it expires."""
    issued_at = time.time() if now is None else now
    payload: dict[str, Any] = {"sub": sub, "exp": int(issued_at) + int(ttl_seconds)}
    if email:
        payload["email"] = email
    body = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return body.decode("ascii") + "." + _b64encode(signature).decode("ascii")


def read_token(secret: str, token: str, *, now: float | None = None) -> dict[str, Any]:
    """Return the payload, or raise :class:`InvalidToken`."""
    try:
        body_text, signature_text = token.split(".", 1)
    except ValueError as exc:
        raise InvalidToken("Malformed token") from exc
    body = body_text.encode("ascii")
    try:
        expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
        given = _b64decode(signature_text.encode("ascii"))
    except Exception as exc:  # noqa: BLE001 - any decode failure is an invalid token
        raise InvalidToken("Malformed token") from exc
    if not hmac.compare_digest(expected, given):
        raise InvalidToken("Bad signature")
    try:
        payload = json.loads(_b64decode(body))
        expires = int(payload["exp"])
        sub = payload["sub"]
    except Exception as exc:  # noqa: BLE001
        raise InvalidToken("Malformed token") from exc
    if not isinstance(sub, str) or not sub:
        raise InvalidToken("Malformed token")
    current = time.time() if now is None else now
    if expires < current:
        raise InvalidToken("Expired token")
    return payload


def _b64encode(raw: bytes) -> bytes:
    return base64.urlsafe_b64encode(raw).rstrip(b"=")


def _b64decode(raw: bytes) -> bytes:
    pad = b"=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + pad)
