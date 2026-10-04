"""Integration tests for ``POST /api/v1/auth/login``.

Real HTTP, real service, real PostgreSQL. Every moderator here is created with
a randomly generated password through the same service path the seeding CLI
uses, so nothing in this file is a working credential.
"""

import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.tokens import decode_access_token
from app.models import Moderator
from tests.conftest import TEST_JWT_SECRET_KEY, make_moderator_password

pytestmark = pytest.mark.integration

LOGIN_URL = "/api/v1/auth/login"
REPORTS_URL = "/api/v1/reports"


def login(client: TestClient, username: str, password: str):
    return client.post(LOGIN_URL, json={"username": username, "password": password})


# ---------------------------------------------------------------------------
# 7-9. Successful login
# ---------------------------------------------------------------------------


def test_valid_credentials_return_a_token(
    api_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    account, password = moderator

    response = login(api_client, account.username, password)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 30 * 60


def test_the_returned_token_names_the_moderator(
    api_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    account, password = moderator

    token = login(api_client, account.username, password).json()["access_token"]
    claims = decode_access_token(token, secret_key=TEST_JWT_SECRET_KEY)

    assert claims.subject == account.id


def test_the_response_contains_only_the_token_fields(
    api_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    """No account internals ride along with the token."""
    account, password = moderator

    response = login(api_client, account.username, password)
    body = response.json()

    assert set(body) == {"access_token", "token_type", "expires_in"}
    assert str(account.id) not in response.text
    assert account.password_hash not in response.text
    assert password not in response.text
    for forbidden in ("password", "is_active", "created_at", "username"):
        assert forbidden not in body


def test_the_token_response_is_not_cacheable(
    api_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    account, password = moderator

    response = login(api_client, account.username, password)

    assert response.headers["cache-control"] == "no-store"


def test_the_username_is_case_and_whitespace_insensitive(
    api_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    account, password = moderator

    for spelling in (account.username.upper(), f"  {account.username}  ", account.username.title()):
        assert login(api_client, spelling, password).status_code == 200, spelling


def test_logging_in_twice_yields_two_distinct_tokens(
    api_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    account, password = moderator

    first = login(api_client, account.username, password).json()["access_token"]
    second = login(api_client, account.username, password).json()["access_token"]

    assert first != second  # distinct jti


# ---------------------------------------------------------------------------
# 10-13. Failure
# ---------------------------------------------------------------------------


def test_an_incorrect_password_is_rejected(
    api_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    account, _ = moderator

    response = login(api_client, account.username, make_moderator_password())

    assert response.status_code == 401
    assert response.json() == {
        "error": {"code": "AUTHENTICATION_FAILED", "message": "Authentication failed."}
    }


def test_an_unknown_username_is_rejected(api_client: TestClient) -> None:
    response = login(api_client, "nobody-by-that-name", make_moderator_password())

    assert response.status_code == 401


def test_an_inactive_moderator_cannot_log_in(
    api_client: TestClient, inactive_moderator: tuple[Moderator, str]
) -> None:
    """Even with the correct password."""
    account, password = inactive_moderator

    response = login(api_client, account.username, password)

    assert response.status_code == 401
    assert "access_token" not in response.json()


def test_unknown_username_and_wrong_password_are_indistinguishable(
    api_client: TestClient,
    moderator: tuple[Moderator, str],
    inactive_moderator: tuple[Moderator, str],
) -> None:
    """The account-enumeration guarantee.

    A real account with a wrong password, a username that does not exist, and a
    deactivated account with the *correct* password must be impossible to tell
    apart from outside.
    """
    account, _ = moderator
    disabled, disabled_password = inactive_moderator

    responses = [
        login(api_client, account.username, make_moderator_password()),
        login(api_client, "no-such-moderator-at-all", make_moderator_password()),
        login(api_client, disabled.username, disabled_password),
    ]

    statuses = {r.status_code for r in responses}
    bodies = {r.text for r in responses}
    headers = {r.headers.get("www-authenticate") for r in responses}

    assert statuses == {401}
    assert len(bodies) == 1, f"responses differ: {bodies}"
    assert len(headers) == 1


def test_failure_responses_carry_a_bearer_challenge(api_client: TestClient) -> None:
    """RFC 9110 requires a WWW-Authenticate header on a 401."""
    response = login(api_client, "nobody", make_moderator_password())

    assert response.headers["www-authenticate"] == 'Bearer realm="whistledrop"'
    # The realm and nothing else: no error_description naming the reason.
    assert "error_description" not in response.headers["www-authenticate"]


def test_an_unknown_username_costs_about_as_much_time_as_a_real_one(
    api_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    """Timing must not become an account-enumeration oracle.

    Returning early on an unknown username would skip bcrypt entirely and
    answer far faster. The service verifies against a throwaway hash instead.
    Bounds are loose: this asserts the same order of magnitude, not a duration.
    """
    account, _ = moderator
    wrong = make_moderator_password()

    def time_login(username: str, repeats: int = 8) -> float:
        start = time.perf_counter()
        for _ in range(repeats):
            login(api_client, username, wrong)
        return time.perf_counter() - start

    time_login(account.username, repeats=2)  # warm caches and connections

    real_account = time_login(account.username)
    unknown_account = time_login("definitely-no-such-moderator")

    ratio = unknown_account / real_account
    assert 0.2 < ratio < 5.0, f"real={real_account:.3f}s unknown={unknown_account:.3f}s"


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"username": "someone"},
        {"password": "a-password-value"},
        {"username": "", "password": "a-password-value"},
        {"username": "someone", "password": ""},
        {"username": "someone", "password": "p", "extra": "field"},
        {"username": "x" * 200, "password": "a-password-value"},
    ],
    ids=[
        "empty",
        "no-password",
        "no-username",
        "blank-username",
        "blank-password",
        "unknown-field",
        "oversized-username",
    ],
)
def test_a_malformed_login_body_is_rejected(api_client: TestClient, body: dict) -> None:
    response = api_client.post(LOGIN_URL, json=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_validation_errors_never_echo_the_password(api_client: TestClient) -> None:
    """A rejected body must not reflect the credential into the response."""
    attempted = "my-actual-password-oops"

    response = api_client.post(LOGIN_URL, json={"password": attempted, "unknown": 1})

    assert response.status_code == 422
    assert attempted not in response.text


def test_get_is_not_allowed_on_login(api_client: TestClient) -> None:
    """There is no GET form that would put credentials in a URL."""
    assert api_client.get(LOGIN_URL).status_code == 405


def test_there_is_no_registration_endpoint(api_client: TestClient) -> None:
    """Moderator accounts are never created over HTTP."""
    for path in (
        "/api/v1/auth/register",
        "/api/v1/auth/signup",
        "/api/v1/moderators",
        "/api/v1/auth/moderators",
    ):
        response = api_client.post(path, json={"username": "x", "password": "y"})

        assert response.status_code in (404, 405), f"{path} exists"


# ---------------------------------------------------------------------------
# 4. Storage
# ---------------------------------------------------------------------------


def test_the_plaintext_password_is_never_stored(
    api_client: TestClient, db_session: Session, moderator: tuple[Moderator, str]
) -> None:
    """Scan every text column of every table for the password."""
    account, password = moderator

    login(api_client, account.username, password)

    stored = db_session.scalars(select(Moderator).where(Moderator.id == account.id)).one()
    assert stored.password_hash != password
    assert password not in stored.password_hash
    assert stored.password_hash.startswith("$2b$")

    from sqlalchemy import text

    columns = db_session.execute(
        text(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND data_type IN "
            "('text','character varying','character')"
        )
    ).all()

    # The table and column names come from information_schema, not from input,
    # and the password itself is a bound parameter. S608 sees the f-string and
    # cannot tell the difference, so it is silenced deliberately here.
    hits = [
        f"{table}.{column}"
        for table, column in columns
        if db_session.execute(
            text(f'SELECT count(*) FROM "{table}" WHERE "{column}" LIKE :needle'),  # noqa: S608
            {"needle": f"%{password}%"},
        ).scalar()
    ]

    assert hits == [], f"plaintext password found in {hits}"


def test_the_password_does_not_appear_in_the_openapi_document(
    client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    _, password = moderator

    assert password not in client.get("/openapi.json").text


# ---------------------------------------------------------------------------
# 28-29. Public API regression
# ---------------------------------------------------------------------------


def test_report_submission_still_needs_no_authentication(api_client: TestClient) -> None:
    """The core requirement of the whole project, re-checked after adding auth."""
    assert "authorization" not in {k.lower() for k in api_client.headers}

    response = api_client.post(
        REPORTS_URL,
        json={
            "category": "SECURITY",
            "description": "Anonymous submission must keep working after auth exists.",
        },
    )

    assert response.status_code == 201
    assert response.json()["case_code"]


def test_case_lookup_still_needs_no_authentication(api_client: TestClient) -> None:
    case_code = api_client.post(
        REPORTS_URL,
        json={"category": "OTHER", "description": "A report used to check anonymous tracking."},
    ).json()["case_code"]

    response = api_client.post("/api/v1/cases/lookup", json={"case_code": case_code})

    assert response.status_code == 200
    assert response.json()["status"] == "SUBMITTED"


def test_a_bogus_authorization_header_does_not_break_public_endpoints(
    api_client: TestClient,
) -> None:
    """A reporter's browser extension adding a stale header must not lock them out."""
    response = api_client.post(
        REPORTS_URL,
        json={"category": "OTHER", "description": "Submitted with a junk Authorization header."},
        headers={"Authorization": "Bearer not-a-real-token-at-all"},
    )

    assert response.status_code == 201


def test_the_public_endpoints_declare_no_security_requirement(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    for path in (REPORTS_URL, "/api/v1/cases/lookup", LOGIN_URL):
        assert "security" not in schema["paths"][path]["post"], path
