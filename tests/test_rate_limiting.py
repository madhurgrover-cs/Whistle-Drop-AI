"""Rate limiting.

The suite at large runs with limiting off — hundreds of tests hitting three
endpoints from one address would throttle each other. These build their own
application with limiting on and deliberately tiny limits, and reset the shared
counters around every test so order cannot matter.
"""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from app.core import rate_limit
from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.main import create_app
from app.models import Moderator, Report

pytestmark = pytest.mark.integration

REPORTS_URL = "/api/v1/reports"
LOOKUP_URL = "/api/v1/cases/lookup"
LOGIN_URL = "/api/v1/auth/login"

VALID_REPORT = {
    "category": "SECURITY",
    "description": "A report body long enough to pass validation on its own.",
}


@pytest.fixture(autouse=True)
def clear_counters() -> Iterator[None]:
    """Start and finish each test with empty counters."""
    rate_limit.reset()
    yield
    rate_limit.reset()


@pytest.fixture
def limited_settings(test_settings: Settings) -> Settings:
    """Test settings with limiting on and limits small enough to reach."""
    return test_settings.model_copy(
        update={
            "rate_limit_enabled": True,
            "rate_limit_reports": "3/minute",
            "rate_limit_case_lookup": "4/minute",
            "rate_limit_login": "2/minute",
        }
    )


@pytest.fixture
def limited_client(limited_settings: Settings, db_session: Session) -> Iterator[TestClient]:
    app: FastAPI = create_app(limited_settings)
    app.dependency_overrides[get_settings] = lambda: limited_settings
    app.dependency_overrides[get_db] = lambda: db_session

    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# 1-4. The three public endpoints are limited
# ---------------------------------------------------------------------------


def test_report_submission_is_rate_limited(limited_client: TestClient) -> None:
    allowed = [limited_client.post(REPORTS_URL, json=VALID_REPORT) for _ in range(3)]
    blocked = limited_client.post(REPORTS_URL, json=VALID_REPORT)

    assert [r.status_code for r in allowed] == [201, 201, 201]
    assert blocked.status_code == 429


def test_case_lookup_is_rate_limited(limited_client: TestClient) -> None:
    code = limited_client.post(REPORTS_URL, json=VALID_REPORT).json()["case_code"]

    results = [
        limited_client.post(LOOKUP_URL, json={"case_code": code}).status_code for _ in range(6)
    ]

    assert results[:4] == [200, 200, 200, 200]
    assert results[4:] == [429, 429]


def test_login_is_rate_limited(
    limited_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    """The tightest limit in the service: this is the credential-stuffing surface."""
    account, password = moderator
    body = {"username": account.username, "password": password}

    results = [limited_client.post(LOGIN_URL, json=body).status_code for _ in range(4)]

    assert results == [200, 200, 429, 429]


def test_failed_logins_also_count(
    limited_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    """Otherwise the limit would not constrain guessing at all."""
    account, _ = moderator

    for _ in range(2):
        limited_client.post(
            LOGIN_URL, json={"username": account.username, "password": "wrong-password-here"}
        )

    blocked = limited_client.post(
        LOGIN_URL, json={"username": account.username, "password": "wrong-password-here"}
    )

    assert blocked.status_code == 429


def test_the_limit_response_uses_the_standard_envelope(limited_client: TestClient) -> None:
    for _ in range(3):
        limited_client.post(REPORTS_URL, json=VALID_REPORT)

    response = limited_client.post(REPORTS_URL, json=VALID_REPORT)

    assert response.status_code == 429
    assert response.json() == {
        "error": {
            "code": "RATE_LIMIT_EXCEEDED",
            "message": "Too many requests. Please wait a moment and try again.",
        }
    }


# ---------------------------------------------------------------------------
# 5. Retry-After
# ---------------------------------------------------------------------------


def test_the_limit_response_carries_retry_after(limited_client: TestClient) -> None:
    for _ in range(3):
        limited_client.post(REPORTS_URL, json=VALID_REPORT)

    response = limited_client.post(REPORTS_URL, json=VALID_REPORT)

    assert "retry-after" in response.headers
    seconds = int(response.headers["retry-after"])
    assert 1 <= seconds <= 61


def test_the_limit_response_leaks_nothing_about_the_limit_or_the_caller(
    limited_client: TestClient,
) -> None:
    """No remaining count, no window, no address, no fingerprint."""
    for _ in range(3):
        limited_client.post(REPORTS_URL, json=VALID_REPORT)

    response = limited_client.post(REPORTS_URL, json=VALID_REPORT)
    body = response.text.lower()

    for leak in ("testclient", "127.0.0.1", "remaining", "window", "limits", "memorystorage"):
        assert leak not in body
    assert "x-ratelimit-remaining" not in {k.lower() for k in response.headers}


# ---------------------------------------------------------------------------
# Scope and isolation
# ---------------------------------------------------------------------------


def test_the_limits_are_per_endpoint(
    limited_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    """Exhausting one endpoint must not lock a reporter out of another."""
    account, password = moderator
    code = limited_client.post(REPORTS_URL, json=VALID_REPORT).json()["case_code"]

    for _ in range(4):
        limited_client.post(REPORTS_URL, json=VALID_REPORT)
    assert limited_client.post(REPORTS_URL, json=VALID_REPORT).status_code == 429

    # Tracking a case and logging in are untouched.
    assert limited_client.post(LOOKUP_URL, json={"case_code": code}).status_code == 200
    assert (
        limited_client.post(
            LOGIN_URL, json={"username": account.username, "password": password}
        ).status_code
        == 200
    )


def test_different_callers_have_separate_allowances(limited_client: TestClient) -> None:
    """Two clients must not share one budget."""
    for _ in range(3):
        limited_client.post(REPORTS_URL, json=VALID_REPORT)
    assert limited_client.post(REPORTS_URL, json=VALID_REPORT).status_code == 429

    # TestClient reports 'testclient' as the address; overriding it is what a
    # second client looks like to the limiter.
    other = limited_client.post(
        REPORTS_URL, json=VALID_REPORT, headers={"X-Forwarded-For": "198.51.100.7"}
    )

    # Proxy headers are NOT trusted by default, so this is still the same
    # caller and still blocked. That is the point of the default.
    assert other.status_code == 429


def test_proxy_headers_are_ignored_unless_trusted(
    test_settings: Settings, db_session: Session
) -> None:
    """X-Forwarded-For is spoofable; believing it would hand out free identities."""
    settings = test_settings.model_copy(
        update={
            "rate_limit_enabled": True,
            "rate_limit_reports": "2/minute",
            "trust_proxy_headers": False,
        }
    )
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = lambda: db_session

    with TestClient(app) as client:
        for index in range(2):
            assert (
                client.post(
                    REPORTS_URL,
                    json=VALID_REPORT,
                    headers={"X-Forwarded-For": f"203.0.113.{index}"},
                ).status_code
                == 201
            )

        # A third, claiming to be yet another address, is still refused.
        blocked = client.post(
            REPORTS_URL, json=VALID_REPORT, headers={"X-Forwarded-For": "203.0.113.99"}
        )

    assert blocked.status_code == 429


def test_limiting_can_be_turned_off(api_client: TestClient) -> None:
    """The default test application has it off, and is not throttled."""
    results = [api_client.post(REPORTS_URL, json=VALID_REPORT).status_code for _ in range(12)]

    assert set(results) == {201}


def test_moderation_endpoints_are_not_rate_limited(
    limited_client: TestClient, auth_headers: dict
) -> None:
    """Authenticated staff work is not the abuse surface.

    A moderator paging through a queue would trip a public-facing limit
    immediately, and they are already identified and accountable.
    """
    results = [
        limited_client.get("/api/v1/moderation/reports", headers=auth_headers).status_code
        for _ in range(15)
    ]

    assert set(results) == {200}


# ---------------------------------------------------------------------------
# 6. Privacy of the limiter itself
# ---------------------------------------------------------------------------


def test_rate_limiting_adds_no_column_to_the_schema(db_session: Session) -> None:
    """No ip_address anywhere. The limiter touches the database not at all."""
    inspector = inspect(db_session.get_bind())

    report_columns = {column["name"] for column in inspector.get_columns("reports")}
    assert report_columns == {
        "id",
        "case_code_hash",
        "category",
        "description",
        "evidence_url",
        "status",
        "created_at",
        "updated_at",
    }

    for table in inspector.get_table_names():
        columns = {column["name"].lower() for column in inspector.get_columns(table)}
        assert not columns & {"ip", "ip_address", "client_ip", "remote_addr", "fingerprint"}


def test_no_report_row_records_the_caller(limited_client: TestClient, db_session: Session) -> None:
    limited_client.post(REPORTS_URL, json=VALID_REPORT, headers={"X-Forwarded-For": "198.51.100.9"})

    report = db_session.scalars(select(Report)).one()
    stored = " ".join(
        str(value) for value in (report.description, report.evidence_url, report.case_code_hash)
    )

    assert "198.51.100.9" not in stored
    assert "testclient" not in stored


def test_the_fingerprint_is_not_the_address(test_settings: Settings) -> None:
    """The counter key is a salted digest, so the limiter's memory holds no IPs."""
    from starlette.datastructures import Headers

    class FakeClient:
        host = "198.51.100.23"

    class FakeApp:
        dependency_overrides: dict = {}

    class FakeRequest:
        app = FakeApp()
        client = FakeClient()
        headers = Headers({})

    FakeApp.dependency_overrides[get_settings] = lambda: test_settings

    fingerprint = rate_limit.client_fingerprint(FakeRequest())  # type: ignore[arg-type]

    assert "198.51.100.23" not in fingerprint
    assert len(fingerprint) == 32
    assert all(character in "0123456789abcdef" for character in fingerprint)

    # Stable within a process, so the counter works at all.
    assert fingerprint == rate_limit.client_fingerprint(FakeRequest())  # type: ignore[arg-type]


def test_the_salt_makes_the_digest_unguessable() -> None:
    """Without a salt, a digest of an IPv4 address is reversed by brute force."""
    import hashlib

    address = "198.51.100.23"
    unsalted = hashlib.sha256(address.encode()).hexdigest()[:32]

    class FakeClient:
        host = address

    class FakeApp:
        dependency_overrides: dict = {}

    class FakeRequest:
        from starlette.datastructures import Headers as _H

        app = FakeApp()
        client = FakeClient()
        headers = _H({})

    salted = rate_limit.client_fingerprint(FakeRequest())  # type: ignore[arg-type]

    assert salted != unsalted


def test_a_malformed_limit_expression_does_not_take_the_endpoint_down(
    test_settings: Settings, db_session: Session
) -> None:
    """A configuration typo must not stop people filing reports.

    The failure is loud in the log and invisible to the reporter: an
    unparseable limit means no limit, never an outage on the one endpoint the
    whole service exists to offer.

    Captured with a handler attached here rather than through ``caplog``,
    because the application installs its own root handler during
    ``create_app`` and the two interact unpredictably.
    """
    import logging

    settings = test_settings.model_copy(
        update={"rate_limit_enabled": True, "rate_limit_reports": "not-a-limit"}
    )
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = lambda: db_session

    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Capture(level=logging.ERROR)
    limiter_logger = logging.getLogger("app.core.rate_limit")
    limiter_logger.addHandler(handler)
    try:
        with TestClient(app) as client:
            response = client.post(REPORTS_URL, json=VALID_REPORT)
    finally:
        limiter_logger.removeHandler(handler)

    # The reporter is served.
    assert response.status_code == 201
    # The operator is told.
    messages = [record.getMessage() for record in records]
    assert any("unparseable rate-limit" in message for message in messages), messages
