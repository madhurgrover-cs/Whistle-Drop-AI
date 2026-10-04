"""Integration tests for ``get_current_moderator``.

Phase 3 adds no protected endpoint — the moderation routes are Phase 4 work —
so the dependency is exercised against a probe route mounted on a throwaway
app built inside this module. Nothing here touches the production router, and
the probe never appears in the real OpenAPI document.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import CurrentModerator
from app.core.config import Settings, get_settings
from app.core.tokens import create_access_token
from app.db.session import get_db
from app.main import create_app
from app.models import Moderator
from tests.conftest import TEST_JWT_SECRET_KEY

pytestmark = pytest.mark.integration

PROBE_URL = "/probe"
OTHER_KEY = "a-totally-different-signing-key-of-good-length"  # noqa: S105


@pytest.fixture
def protected_client(test_settings: Settings, db_session: Session) -> Iterator[TestClient]:
    """A client for an app with one route behind ``get_current_moderator``.

    Built here rather than in the application so that Phase 3 ships no
    protected endpoint of its own. Phase 4's routes will declare exactly this
    dependency, so what is verified here is what they will inherit.
    """
    app: FastAPI = create_app(test_settings)
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_db] = lambda: db_session

    @app.get(PROBE_URL)
    def probe(moderator: CurrentModerator) -> dict[str, str]:
        return {"username": moderator.username, "id": str(moderator.id)}

    with TestClient(app) as client:
        yield client


def token_for(moderator: Moderator, **kwargs) -> str:
    return create_access_token(
        subject=moderator.id,
        secret_key=kwargs.pop("secret_key", TEST_JWT_SECRET_KEY),
        **kwargs,
    )


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# 23. The happy path
# ---------------------------------------------------------------------------


def test_a_valid_token_resolves_to_the_right_moderator(
    protected_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    account, _ = moderator

    response = protected_client.get(PROBE_URL, headers=bearer(token_for(account)))

    assert response.status_code == 200, response.text
    assert response.json() == {"username": account.username, "id": str(account.id)}


def test_a_token_obtained_from_the_login_endpoint_works(
    protected_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    """End to end: log in, then use the token the endpoint actually returned."""
    account, password = moderator

    login = protected_client.post(
        "/api/v1/auth/login", json={"username": account.username, "password": password}
    )
    assert login.status_code == 200

    response = protected_client.get(PROBE_URL, headers=bearer(login.json()["access_token"]))

    assert response.status_code == 200
    assert response.json()["username"] == account.username


def test_the_bearer_scheme_is_case_insensitive(
    protected_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    """RFC 9110 makes the auth scheme case-insensitive."""
    account, _ = moderator
    token = token_for(account)

    for scheme in ("Bearer", "bearer", "BEARER", "BeArEr"):
        response = protected_client.get(PROBE_URL, headers={"Authorization": f"{scheme} {token}"})

        assert response.status_code == 200, scheme


# ---------------------------------------------------------------------------
# 21-22. The Authorization header
# ---------------------------------------------------------------------------


def test_a_missing_authorization_header_is_rejected(protected_client: TestClient) -> None:
    response = protected_client.get(PROBE_URL)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTHENTICATION_FAILED"
    assert response.headers["www-authenticate"] == 'Bearer realm="whistledrop"'


@pytest.mark.parametrize(
    "header",
    [
        "",
        "   ",
        "token-with-no-scheme",
        "Basic dXNlcjpwYXNz",
        "Bearer",
        "Bearer ",
        "Bearer  ",
        "Digest abc",
        "bearer_token_value",
        "Bearer one two three",
        "Token abc.def.ghi",
        "JWT abc.def.ghi",
    ],
    ids=[
        "empty",
        "spaces",
        "no-scheme",
        "basic",
        "scheme-only",
        "scheme-and-space",
        "scheme-and-spaces",
        "digest",
        "underscored",
        "extra-parts",
        "token-scheme",
        "jwt-scheme",
    ],
)
def test_a_malformed_authorization_header_is_rejected(
    protected_client: TestClient, header: str
) -> None:
    response = protected_client.get(PROBE_URL, headers={"Authorization": header})

    assert response.status_code == 401, header
    assert response.json()["error"]["code"] == "AUTHENTICATION_FAILED"


def test_a_raw_token_without_the_bearer_scheme_is_rejected(
    protected_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    """Only the documented format is accepted."""
    account, _ = moderator

    response = protected_client.get(PROBE_URL, headers={"Authorization": token_for(account)})

    assert response.status_code == 401


def test_a_token_in_a_query_parameter_is_not_accepted(
    protected_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    """A credential in a URL would land in every access log on the path."""
    account, _ = moderator

    response = protected_client.get(f"{PROBE_URL}?token={token_for(account)}")

    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 15-20, 24-25. Token and account rejection
# ---------------------------------------------------------------------------


def test_an_expired_token_is_rejected_and_says_so(
    protected_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    """The one distinguishable failure: it concerns the holder's own token.

    A client uses this to know that logging in again will help, and it reveals
    nothing about any account.
    """
    account, _ = moderator
    stale = token_for(account, now=datetime.now(UTC) - timedelta(hours=2))

    response = protected_client.get(PROBE_URL, headers=bearer(stale))

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "TOKEN_EXPIRED"
    assert "log in again" in response.json()["error"]["message"].lower()


def test_a_token_signed_with_another_key_is_rejected(
    protected_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    account, _ = moderator

    response = protected_client.get(
        PROBE_URL, headers=bearer(token_for(account, secret_key=OTHER_KEY))
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTHENTICATION_FAILED"


def test_a_token_signed_with_the_wrong_algorithm_is_rejected(
    protected_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    account, _ = moderator

    response = protected_client.get(
        PROBE_URL, headers=bearer(token_for(account, algorithm="HS512"))
    )

    assert response.status_code == 401


def test_an_unsigned_token_is_rejected(
    protected_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    """``alg: none`` through the full stack."""
    import jwt

    account, _ = moderator
    now = datetime.now(UTC)
    forged = jwt.encode(
        {
            "sub": str(account.id),
            "iat": now,
            "exp": now + timedelta(minutes=30),
            "type": "access",
        },
        key=None,  # type: ignore[arg-type]
        algorithm="none",
    )

    response = protected_client.get(PROBE_URL, headers=bearer(forged))

    assert response.status_code == 401


@pytest.mark.parametrize(
    "token",
    ["not-a-token", "a.b.c", "....", "eyJhbGciOiJIUzI1NiJ9", "null", "%00", "' OR 1=1 --"],
    ids=["prose", "three-dots", "dots", "header-only", "null", "null-byte", "sql"],
)
def test_an_invalid_token_is_rejected(protected_client: TestClient, token: str) -> None:
    response = protected_client.get(PROBE_URL, headers=bearer(token))

    assert response.status_code == 401


def test_a_token_of_the_wrong_type_is_rejected(
    protected_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    """A refresh token must not be replayable where an access token is expected."""
    import jwt

    account, _ = moderator
    now = datetime.now(UTC)
    refresh = jwt.encode(
        {
            "sub": str(account.id),
            "iat": now,
            "exp": now + timedelta(minutes=30),
            "type": "refresh",
        },
        TEST_JWT_SECRET_KEY,
        algorithm="HS256",
    )

    response = protected_client.get(PROBE_URL, headers=bearer(refresh))

    assert response.status_code == 401


def test_a_token_with_no_subject_is_rejected(protected_client: TestClient) -> None:
    import jwt

    now = datetime.now(UTC)
    token = jwt.encode(
        {"iat": now, "exp": now + timedelta(minutes=30), "type": "access"},
        TEST_JWT_SECRET_KEY,
        algorithm="HS256",
    )

    response = protected_client.get(PROBE_URL, headers=bearer(token))

    assert response.status_code == 401


def test_a_token_naming_a_nonexistent_moderator_is_rejected(
    protected_client: TestClient,
) -> None:
    """Correctly signed, but the account it names has never existed."""
    token = create_access_token(subject=uuid.uuid4(), secret_key=TEST_JWT_SECRET_KEY)

    response = protected_client.get(PROBE_URL, headers=bearer(token))

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTHENTICATION_FAILED"


def test_a_token_naming_a_deactivated_moderator_is_rejected(
    protected_client: TestClient, inactive_moderator: tuple[Moderator, str]
) -> None:
    account, _ = inactive_moderator

    response = protected_client.get(PROBE_URL, headers=bearer(token_for(account)))

    assert response.status_code == 401


def test_deactivation_takes_effect_on_the_very_next_request(
    protected_client: TestClient, db_session: Session, moderator: tuple[Moderator, str]
) -> None:
    """Why the account is re-read on every request instead of trusted from claims.

    A moderator disabled a moment ago still holds a cryptographically valid
    token. Only the per-request lookup stops it working.
    """
    account, _ = moderator
    token = token_for(account)

    assert protected_client.get(PROBE_URL, headers=bearer(token)).status_code == 200

    account.is_active = False
    db_session.commit()

    assert protected_client.get(PROBE_URL, headers=bearer(token)).status_code == 401


# ---------------------------------------------------------------------------
# Error hygiene
# ---------------------------------------------------------------------------


def test_no_rejection_leaks_why_it_was_rejected(
    protected_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    """Every failure but expiry is byte-identical, so nothing can be inferred."""
    account, _ = moderator

    bodies = {
        protected_client.get(PROBE_URL, headers=headers).text
        for headers in (
            {},
            {"Authorization": "Bearer garbage"},
            {"Authorization": "Basic dXNlcjpwYXNz"},
            bearer(token_for(account, secret_key=OTHER_KEY)),
            bearer(create_access_token(subject=uuid.uuid4(), secret_key=TEST_JWT_SECRET_KEY)),
        )
    }

    assert len(bodies) == 1, f"rejection responses differ: {bodies}"


def test_rejections_expose_no_internal_detail(protected_client: TestClient) -> None:
    response = protected_client.get(PROBE_URL, headers=bearer("garbage.token.here"))
    body = response.text.lower()

    for leak in (
        "jwt",
        "signature",
        "pyjwt",
        "traceback",
        "sqlalchemy",
        "psycopg",
        "postgresql",
        TEST_JWT_SECRET_KEY.lower(),
    ):
        assert leak not in body, f"response leaked {leak!r}"


def test_the_signing_key_never_appears_in_any_response(
    protected_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    account, password = moderator

    responses = [
        protected_client.get(PROBE_URL),
        protected_client.get(PROBE_URL, headers=bearer(token_for(account))),
        protected_client.post(
            "/api/v1/auth/login", json={"username": account.username, "password": password}
        ),
        protected_client.get("/openapi.json"),
    ]

    for response in responses:
        assert TEST_JWT_SECRET_KEY not in response.text


def test_the_probe_route_is_not_part_of_the_real_application() -> None:
    """The probe exists only in this module and never in the application."""
    from app.main import create_app as build

    settings = Settings(  # type: ignore[call-arg]
        app_env="test",
        database_url="postgresql+psycopg://x:y@localhost:5433/whistledrop_test",
        case_code_pepper="test-pepper-not-a-real-secret-0123456789abcdef",
        jwt_secret_key=TEST_JWT_SECRET_KEY,
        _env_file=None,
    )

    paths = set(build(settings).openapi()["paths"])

    assert PROBE_URL not in paths
    assert paths == {
        "/api/v1/health",
        "/api/v1/reports",
        "/api/v1/cases/lookup",
        "/api/v1/auth/login",
        "/api/v1/moderation/reports",
        "/api/v1/moderation/reports/{report_id}",
        "/api/v1/moderation/reports/{report_id}/status",
    }
