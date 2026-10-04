"""Unit tests for access-token creation and verification.

These forge tokens deliberately — signed with the wrong key, with
``"alg": "none"``, with claims removed — to prove each is refused. Every one is
a real attack against a JWT implementation, and none of them can be provoked
through the API, so they are tested against the token layer directly.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest

from app.core.tokens import (
    ACCESS_TOKEN_TYPE,
    ExpiredTokenError,
    InvalidTokenError,
    TokenError,
    create_access_token,
    decode_access_token,
)

KEY = "a-test-signing-key-that-is-at-least-32-characters"  # noqa: S105
OTHER_KEY = "a-completely-different-key-of-sufficient-length"  # noqa: S105


def forge(claims: dict[str, Any], *, key: str = KEY, algorithm: str = "HS256") -> str:
    """Hand-build a token with exactly the claims given."""
    return jwt.encode(claims, key, algorithm=algorithm)


def valid_claims(**overrides: Any) -> dict[str, Any]:
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "sub": str(uuid.uuid4()),
        "iat": now,
        "exp": now + timedelta(minutes=30),
        "jti": uuid.uuid4().hex,
        "type": ACCESS_TOKEN_TYPE,
    }
    claims.update(overrides)
    return claims


def payload_of(token: str) -> dict[str, Any]:
    """Read a token's claims without verifying it — as any holder can."""
    return jwt.decode(token, options={"verify_signature": False})


# ---------------------------------------------------------------------------
# 14. Creation and round trip
# ---------------------------------------------------------------------------


def test_a_token_round_trips() -> None:
    moderator_id = uuid.uuid4()

    claims = decode_access_token(
        create_access_token(subject=moderator_id, secret_key=KEY), secret_key=KEY
    )

    assert claims.subject == moderator_id
    assert claims.expires_at > claims.issued_at
    assert claims.token_id


def test_expiry_matches_the_configured_lifetime() -> None:
    claims = decode_access_token(
        create_access_token(subject=uuid.uuid4(), secret_key=KEY, expires_minutes=45),
        secret_key=KEY,
    )

    lifetime = claims.expires_at - claims.issued_at
    assert lifetime == timedelta(minutes=45)


def test_every_token_has_a_distinct_id() -> None:
    """``jti`` is what a future revocation list would name."""
    ids = {
        decode_access_token(
            create_access_token(subject=uuid.uuid4(), secret_key=KEY), secret_key=KEY
        ).token_id
        for _ in range(20)
    }

    assert len(ids) == 20


def test_the_configured_hmac_algorithms_all_work() -> None:
    for algorithm in ("HS256", "HS384", "HS512"):
        token = create_access_token(subject=uuid.uuid4(), secret_key=KEY, algorithm=algorithm)

        assert decode_access_token(token, secret_key=KEY, algorithm=algorithm)


# ---------------------------------------------------------------------------
# 26-27. What is, and is not, in the payload
# ---------------------------------------------------------------------------


def test_the_payload_holds_only_the_documented_claims() -> None:
    token = create_access_token(subject=uuid.uuid4(), secret_key=KEY)

    assert set(payload_of(token)) == {"sub", "iat", "exp", "jti", "type"}


def test_the_payload_holds_no_password_material() -> None:
    """A JWT is signed, not encrypted: every claim is readable by its holder."""
    token = create_access_token(subject=uuid.uuid4(), secret_key=KEY)
    body = str(payload_of(token)).lower()

    for forbidden in ("password", "password_hash", "hash", "secret", "bcrypt", "$2b$"):
        assert forbidden not in body


def test_the_payload_holds_no_report_information() -> None:
    token = create_access_token(subject=uuid.uuid4(), secret_key=KEY)
    body = str(payload_of(token)).lower()

    for forbidden in ("case_code", "case", "report", "description", "evidence", "category"):
        assert forbidden not in body


def test_the_payload_does_not_carry_the_signing_key() -> None:
    """The key signs the token; it is never itself a claim."""
    token = create_access_token(subject=uuid.uuid4(), secret_key=KEY)

    assert KEY not in str(payload_of(token))
    assert KEY not in token


def test_the_subject_is_the_only_identity_in_the_token() -> None:
    """No username: ``sub`` is sufficient, and a name is one more thing to leak."""
    payload = payload_of(create_access_token(subject=uuid.uuid4(), secret_key=KEY))

    assert "username" not in payload
    assert "name" not in payload
    assert "email" not in payload
    assert uuid.UUID(payload["sub"])


# ---------------------------------------------------------------------------
# 15-20. Rejection
# ---------------------------------------------------------------------------


def test_an_expired_token_is_rejected() -> None:
    long_ago = datetime.now(UTC) - timedelta(hours=2)
    token = create_access_token(subject=uuid.uuid4(), secret_key=KEY, now=long_ago)

    with pytest.raises(ExpiredTokenError):
        decode_access_token(token, secret_key=KEY)


def test_a_token_expiring_one_second_ago_is_rejected() -> None:
    """No leeway: one service issues and verifies, so there is no skew."""
    now = datetime.now(UTC)
    token = forge(valid_claims(iat=now - timedelta(minutes=2), exp=now - timedelta(seconds=1)))

    with pytest.raises(ExpiredTokenError):
        decode_access_token(token, secret_key=KEY)


def test_a_token_signed_with_another_key_is_rejected() -> None:
    token = create_access_token(subject=uuid.uuid4(), secret_key=OTHER_KEY)

    with pytest.raises(InvalidTokenError):
        decode_access_token(token, secret_key=KEY)


def test_a_tampered_payload_is_rejected() -> None:
    """Swapping in another moderator's id without the key breaks the signature.

    The attack this models: a moderator who holds a valid token edits `sub` to
    name a different account and re-attaches the original signature.
    """
    import base64
    import json

    token = create_access_token(subject=uuid.uuid4(), secret_key=KEY)
    header, original_payload, signature = token.split(".")

    claims = json.loads(base64.urlsafe_b64decode(original_payload + "=="))
    claims["sub"] = str(uuid.uuid4())
    forged_payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()

    assert forged_payload != original_payload

    with pytest.raises(TokenError):
        decode_access_token(f"{header}.{forged_payload}.{signature}", secret_key=KEY)


@pytest.mark.parametrize(
    "token",
    ["", "   ", "not-a-token", "a.b", "a.b.c.d", "eyJhbGciOiJIUzI1NiJ9", "null", "{}"],
    ids=["empty", "spaces", "prose", "two-parts", "four-parts", "header-only", "null", "braces"],
)
def test_a_malformed_token_is_rejected(token: str) -> None:
    with pytest.raises(InvalidTokenError):
        decode_access_token(token, secret_key=KEY)


def test_an_unsigned_token_is_rejected() -> None:
    """The ``alg: none`` attack. A single-algorithm allowlist closes it."""
    token = jwt.encode(valid_claims(), key=None, algorithm="none")  # type: ignore[arg-type]

    with pytest.raises(InvalidTokenError):
        decode_access_token(token, secret_key=KEY)


def test_a_token_signed_with_a_different_algorithm_is_rejected() -> None:
    """Algorithm confusion: the header does not get to choose the verifier."""
    token = create_access_token(subject=uuid.uuid4(), secret_key=KEY, algorithm="HS512")

    with pytest.raises(InvalidTokenError):
        decode_access_token(token, secret_key=KEY, algorithm="HS256")


@pytest.mark.parametrize("missing", ["sub", "exp", "iat", "type"])
def test_a_token_missing_a_required_claim_is_rejected(missing: str) -> None:
    claims = valid_claims()
    del claims[missing]

    with pytest.raises(InvalidTokenError):
        decode_access_token(forge(claims), secret_key=KEY)


@pytest.mark.parametrize(
    "subject",
    [None, "", "not-a-uuid", "admin", "1", "../../etc/passwd", 12345, {"id": "x"}],
    ids=["null", "empty", "prose", "name", "number-string", "traversal", "number", "object"],
)
def test_a_token_with_an_unusable_subject_is_rejected(subject: object) -> None:
    with pytest.raises(InvalidTokenError):
        decode_access_token(forge(valid_claims(sub=subject)), secret_key=KEY)


@pytest.mark.parametrize(
    "token_type",
    ["refresh", "reset", "id", "", "ACCESS", "access "],
    ids=["refresh", "reset", "id", "empty", "wrong-case", "trailing-space"],
)
def test_a_token_of_the_wrong_type_is_rejected(token_type: str) -> None:
    """So a refresh or reset token can never be replayed as an access token."""
    with pytest.raises(InvalidTokenError):
        decode_access_token(forge(valid_claims(type=token_type)), secret_key=KEY)


# ---------------------------------------------------------------------------
# Error hygiene
# ---------------------------------------------------------------------------


def test_token_errors_carry_no_library_detail() -> None:
    """Whatever PyJWT says stays in the log, not in the exception we raise."""
    wrongly_signed = create_access_token(subject=uuid.uuid4(), secret_key=OTHER_KEY)

    with pytest.raises(TokenError) as caught:
        decode_access_token(wrongly_signed, secret_key=KEY)

    message = str(caught.value).lower()
    assert "signature" not in message
    assert KEY.lower() not in message
    assert OTHER_KEY.lower() not in message


def test_no_error_message_contains_the_signing_key() -> None:
    for token in ("garbage", forge(valid_claims(), key=OTHER_KEY)):
        with pytest.raises(TokenError) as caught:
            decode_access_token(token, secret_key=KEY)

        assert KEY not in str(caught.value)
