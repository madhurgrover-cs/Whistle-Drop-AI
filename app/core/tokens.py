"""Moderator access tokens: creation and verification.

Design
------
**Algorithm.** HS256 — HMAC-SHA256. The same service signs and verifies, so
there is no public key to distribute and a symmetric algorithm is the simpler,
smaller choice. The configured algorithm is restricted to the HMAC family in
``Settings``, and :func:`decode_access_token` passes exactly one algorithm to
the verifier. That is what closes the classic algorithm-confusion attacks: a
token whose header says ``"alg": "none"``, or one re-signed with RS256 using an
attacker-supplied public key, is rejected before its signature is even
considered.

**Expiry.** 30 minutes by default. There is no refresh token in this system:
when a token expires the moderator logs in again. That is a deliberate
simplification — refresh tokens are long-lived credentials that need storage,
rotation and revocation to be safe, and a moderation console does not need
them. :class:`ExpiredTokenError` is raised separately from
:class:`InvalidTokenError` so a client can tell "log in again" from "something
is wrong with this token"; that distinction concerns only the holder's own
token and reveals nothing about any account.

**Claims.** Kept to the minimum that identifies the bearer:

==========  ==================================================================
``sub``     The moderator's UUID, as a string. The only identity in the token.
``exp``     Expiry. Enforced by the library, with no leeway.
``iat``     Issued-at, so a token's age is auditable.
``jti``     A random id, so a future revocation list can name a single token
            without changing the token format.
``type``    Always ``"access"``. Checked explicitly, so that if a refresh or
            password-reset token is ever added, one kind can never be replayed
            where another is expected.
==========  ==================================================================

There is deliberately **no** username, no password material, no report data and
no case code. A JWT is signed, not encrypted: anyone holding it can read every
claim. Nothing goes in that would matter if read.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import jwt

__all__ = [
    "ACCESS_TOKEN_TYPE",
    "ExpiredTokenError",
    "InvalidTokenError",
    "TokenClaims",
    "TokenError",
    "create_access_token",
    "decode_access_token",
]

#: Value of the ``type`` claim on a moderator access token. Not a secret —
#: S105 matches the name, not the meaning.
ACCESS_TOKEN_TYPE: Final = "access"  # noqa: S105

#: Claims that must be present, enforced by the library rather than by us.
_REQUIRED_CLAIMS: Final = ["sub", "exp", "iat", "type"]


class TokenError(Exception):
    """Base class for every token failure.

    Carries no detail from the underlying library. Callers map this to one
    generic authentication failure; the specific reason is logged server-side
    and never returned.
    """


class InvalidTokenError(TokenError):
    """The token is malformed, unsigned, wrongly signed, or missing a claim."""


class ExpiredTokenError(TokenError):
    """The token was valid but its ``exp`` has passed."""


@dataclass(frozen=True, slots=True)
class TokenClaims:
    """The verified contents of an access token."""

    subject: uuid.UUID
    issued_at: datetime
    expires_at: datetime
    token_id: str


def create_access_token(
    *,
    subject: uuid.UUID,
    secret_key: str,
    algorithm: str = "HS256",
    expires_minutes: int = 30,
    now: datetime | None = None,
) -> str:
    """Mint a signed access token for ``subject``.

    ``now`` is injectable purely so tests can produce an already-expired token
    without sleeping; nothing in the application passes it.
    """
    issued_at = now or datetime.now(UTC)
    expires_at = issued_at + timedelta(minutes=expires_minutes)

    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": issued_at,
        "exp": expires_at,
        "jti": uuid.uuid4().hex,
        "type": ACCESS_TOKEN_TYPE,
    }

    return jwt.encode(payload, secret_key, algorithm=algorithm)


def decode_access_token(
    token: str,
    *,
    secret_key: str,
    algorithm: str = "HS256",
) -> TokenClaims:
    """Verify ``token`` and return its claims.

    Raises :class:`ExpiredTokenError` if it has expired and
    :class:`InvalidTokenError` for every other failure — bad signature, wrong
    algorithm, missing claim, wrong token type, unparseable subject.
    """
    if not token:
        raise InvalidTokenError("No token supplied.")

    try:
        payload = jwt.decode(
            token,
            secret_key,
            # A list of exactly one algorithm. PyJWT refuses any token whose
            # header names something else, which is what makes "alg": "none"
            # and HS/RS confusion non-starters.
            algorithms=[algorithm],
            options={
                "require": _REQUIRED_CLAIMS,
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
            },
            # No clock skew allowance: one service issues and verifies these,
            # so there are no two clocks to reconcile.
            leeway=0,
        )
    except jwt.ExpiredSignatureError as exc:
        raise ExpiredTokenError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        # Covers bad signature, wrong algorithm, malformed structure and any
        # missing required claim. The library's message stays here.
        raise InvalidTokenError("Token could not be verified.") from exc

    if payload.get("type") != ACCESS_TOKEN_TYPE:
        raise InvalidTokenError("Token is not an access token.")

    subject = payload.get("sub")
    if not subject or not isinstance(subject, str):
        raise InvalidTokenError("Token has no usable subject.")

    try:
        subject_id = uuid.UUID(subject)
    except (ValueError, AttributeError, TypeError) as exc:
        raise InvalidTokenError("Token subject is not a moderator id.") from exc

    return TokenClaims(
        subject=subject_id,
        issued_at=datetime.fromtimestamp(payload["iat"], tz=UTC),
        expires_at=datetime.fromtimestamp(payload["exp"], tz=UTC),
        token_id=str(payload.get("jti", "")),
    )
