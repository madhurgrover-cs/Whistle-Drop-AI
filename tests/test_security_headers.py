"""Security headers, CORS, and the request-body ceiling."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.main import create_app
from app.schemas.reports import DESCRIPTION_MAX_LENGTH, EVIDENCE_URL_MAX_LENGTH

pytestmark = pytest.mark.integration

REPORTS_URL = "/api/v1/reports"
HEALTH_URL = "/api/v1/health"
QUEUE_URL = "/api/v1/moderation/reports"

VALID_REPORT = {
    "category": "SECURITY",
    "description": "A report body long enough to pass validation on its own.",
}

EXPECTED_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-resource-policy": "same-origin",
}


def client_with(settings: Settings, db_session: Session) -> TestClient:
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = lambda: db_session
    return TestClient(app)


# ---------------------------------------------------------------------------
# 10-11. Headers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("header,value", sorted(EXPECTED_HEADERS.items()))
def test_the_health_endpoint_carries_the_headers(
    api_client: TestClient, header: str, value: str
) -> None:
    response = api_client.get(HEALTH_URL)

    assert response.headers.get(header) == value


def test_a_public_endpoint_carries_the_headers(api_client: TestClient) -> None:
    response = api_client.post(REPORTS_URL, json=VALID_REPORT)

    for header, value in EXPECTED_HEADERS.items():
        assert response.headers.get(header) == value, header


def test_an_authenticated_endpoint_carries_the_headers(
    api_client: TestClient, auth_headers: dict
) -> None:
    response = api_client.get(QUEUE_URL, headers=auth_headers)

    for header, value in EXPECTED_HEADERS.items():
        assert response.headers.get(header) == value, header


def test_error_responses_carry_the_headers_too(api_client: TestClient) -> None:
    """A 401 or 422 is still a response a browser may act on."""
    for response in (
        api_client.get(QUEUE_URL),  # 401
        api_client.post(REPORTS_URL, json={}),  # 422
        api_client.get("/api/v1/nope"),  # 404
    ):
        for header, value in EXPECTED_HEADERS.items():
            assert response.headers.get(header) == value, (response.status_code, header)


def test_the_api_content_security_policy_denies_everything(api_client: TestClient) -> None:
    """A JSON response is never a document; nothing should load from it."""
    policy = api_client.post(REPORTS_URL, json=VALID_REPORT).headers["content-security-policy"]

    assert "default-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert "base-uri 'none'" in policy
    assert "form-action 'none'" in policy


def test_the_docs_page_gets_a_policy_that_lets_swagger_work(client: TestClient) -> None:
    """A strict API policy would blank the docs page; it gets its own."""
    response = client.get("/docs")

    assert response.status_code == 200
    policy = response.headers["content-security-policy"]
    assert "cdn.jsdelivr.net" in policy
    # Still no framing, even on the one page that renders HTML.
    assert "frame-ancestors 'none'" in policy


def test_permissions_policy_disables_device_access(api_client: TestClient) -> None:
    policy = api_client.get(HEALTH_URL).headers["permissions-policy"]

    for feature in ("camera=()", "geolocation=()", "microphone=()"):
        assert feature in policy


def test_the_application_sets_a_server_header_that_names_no_stack(
    api_client: TestClient,
) -> None:
    """What the application controls — and what it does not.

    This asserts the header *this application* sends. It is deliberately not a
    claim about production: uvicorn appends its own ``server: uvicorn`` at the
    protocol layer, below ASGI, where no middleware can reach it. Run it with
    ``--no-server-header`` and ours is the only one left. Verified on a real
    server; see ``docs/security-checklist.md``.
    """
    server = api_client.get(HEALTH_URL).headers.get("server", "")

    assert server == "whistledrop"
    assert "uvicorn" not in server.lower()
    assert "starlette" not in server.lower()
    assert "python" not in server.lower()


def test_the_deployment_guidance_requires_suppressing_the_server_header() -> None:
    """The limitation above is documented, not quietly left."""
    from tests.conftest import PROJECT_ROOT

    checklist = (PROJECT_ROOT / "docs" / "security-checklist.md").read_text(encoding="utf-8")

    assert "--no-server-header" in checklist


def test_no_framework_version_is_disclosed_in_headers(api_client: TestClient) -> None:
    joined = " ".join(f"{k}: {v}" for k, v in api_client.get(HEALTH_URL).headers.items()).lower()

    for leak in ("x-powered-by", "uvicorn", "starlette", "fastapi/", "python/"):
        assert leak not in joined


# ---------------------------------------------------------------------------
# HSTS: only where it means something
# ---------------------------------------------------------------------------


def test_hsts_is_absent_by_default(api_client: TestClient) -> None:
    """Over plain HTTP browsers ignore HSTS entirely.

    Emitting it in development would be decoration that invites the false
    belief that local traffic is protected.
    """
    assert "strict-transport-security" not in api_client.get(HEALTH_URL).headers


def test_hsts_is_sent_when_configured_for_a_tls_deployment(
    test_settings: Settings, db_session: Session
) -> None:
    settings = test_settings.model_copy(update={"hsts_max_age_seconds": 31_536_000})

    with client_with(settings, db_session) as client:
        header = client.get(HEALTH_URL).headers.get("strict-transport-security")

    assert header == "max-age=31536000; includeSubDomains"


# ---------------------------------------------------------------------------
# 12-14. CORS
# ---------------------------------------------------------------------------


@pytest.fixture
def cors_client(test_settings: Settings, db_session: Session) -> Iterator[TestClient]:
    settings = test_settings.model_copy(
        update={"cors_allowed_origins": ["https://console.example.org"]}
    )
    with client_with(settings, db_session) as client:
        yield client


def test_no_cors_headers_at_all_by_default(api_client: TestClient) -> None:
    """This API has no browser front end, so the safest policy is none."""
    response = api_client.get(HEALTH_URL, headers={"Origin": "https://anywhere.example.com"})

    assert response.status_code == 200  # non-browser clients are unaffected
    assert "access-control-allow-origin" not in response.headers


def test_an_allowed_origin_is_granted(cors_client: TestClient) -> None:
    response = cors_client.get(HEALTH_URL, headers={"Origin": "https://console.example.org"})

    assert response.headers["access-control-allow-origin"] == "https://console.example.org"


def test_a_disallowed_origin_is_not_granted(cors_client: TestClient) -> None:
    """The browser is what enforces this; the header simply is not sent."""
    response = cors_client.get(HEALTH_URL, headers={"Origin": "https://evil.example.net"})

    assert "access-control-allow-origin" not in response.headers


def test_preflight_succeeds_for_an_allowed_origin(cors_client: TestClient) -> None:
    response = cors_client.options(
        REPORTS_URL,
        headers={
            "Origin": "https://console.example.org",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://console.example.org"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_preflight_is_refused_for_a_disallowed_origin(cors_client: TestClient) -> None:
    response = cors_client.options(
        REPORTS_URL,
        headers={
            "Origin": "https://evil.example.net",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert "access-control-allow-origin" not in response.headers


def test_credentials_are_not_allowed_by_default(cors_client: TestClient) -> None:
    """The API authenticates with a bearer header, not cookies.

    Nothing needs the browser to attach credentials, so it is not invited to.
    """
    response = cors_client.get(HEALTH_URL, headers={"Origin": "https://console.example.org"})

    assert response.headers.get("access-control-allow-credentials") != "true"


def test_wildcard_origins_with_credentials_is_refused_at_startup() -> None:
    """The combination that would let any site make authenticated calls."""
    from pydantic import ValidationError

    from tests.conftest import TEST_CASE_CODE_PEPPER, TEST_JWT_SECRET_KEY

    with pytest.raises(ValidationError, match="must not contain"):
        Settings(  # type: ignore[call-arg]
            app_env="production",
            database_url="postgresql+psycopg://x:y@localhost:5432/whistledrop",
            case_code_pepper=TEST_CASE_CODE_PEPPER,
            jwt_secret_key=TEST_JWT_SECRET_KEY,
            cors_allowed_origins=["*"],
            cors_allow_credentials=True,
            _env_file=None,
        )


def test_origins_can_be_configured_as_a_comma_separated_string() -> None:
    """Which is the only shape an environment variable can take."""
    from tests.conftest import TEST_CASE_CODE_PEPPER, TEST_JWT_SECRET_KEY

    settings = Settings(  # type: ignore[call-arg]
        app_env="test",
        database_url="postgresql+psycopg://x:y@localhost:5433/whistledrop_test",
        case_code_pepper=TEST_CASE_CODE_PEPPER,
        jwt_secret_key=TEST_JWT_SECRET_KEY,
        cors_allowed_origins="https://a.example.org, https://b.example.org",
        _env_file=None,
    )

    assert settings.cors_allowed_origins == ["https://a.example.org", "https://b.example.org"]


# ---------------------------------------------------------------------------
# 7-9. Request size
# ---------------------------------------------------------------------------


def test_a_normal_report_succeeds(api_client: TestClient) -> None:
    assert api_client.post(REPORTS_URL, json=VALID_REPORT).status_code == 201


def test_a_description_at_the_maximum_still_succeeds(api_client: TestClient) -> None:
    """The body ceiling must not shadow the application's own limit."""
    response = api_client.post(
        REPORTS_URL, json={"category": "OTHER", "description": "x" * DESCRIPTION_MAX_LENGTH}
    )

    assert response.status_code == 201


def test_an_oversized_description_is_rejected_by_validation(api_client: TestClient) -> None:
    response = api_client.post(
        REPORTS_URL, json={"category": "OTHER", "description": "x" * (DESCRIPTION_MAX_LENGTH + 1)}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_an_oversized_evidence_url_is_rejected(api_client: TestClient) -> None:
    response = api_client.post(
        REPORTS_URL,
        json={
            **VALID_REPORT,
            "evidence_url": "https://example.com/" + "a" * EVIDENCE_URL_MAX_LENGTH,
        },
    )

    assert response.status_code == 422


def test_an_oversized_body_is_rejected_before_it_is_parsed(api_client: TestClient) -> None:
    """Larger than the configured ceiling: refused at the HTTP layer."""
    oversized = "x" * (200 * 1024)

    response = api_client.post(REPORTS_URL, json={"category": "OTHER", "description": oversized})

    assert response.status_code == 413
    assert response.json() == {
        "error": {"code": "REQUEST_TOO_LARGE", "message": "The request body is too large."}
    }


def test_an_oversized_body_is_rejected_on_every_endpoint(
    api_client: TestClient, auth_headers: dict
) -> None:
    oversized = {"junk": "x" * (200 * 1024)}

    # client.request(...) rather than client.get(...), because httpx refuses a
    # json= body on GET and the ceiling must apply to every method.
    for method, url, headers in (
        ("POST", REPORTS_URL, {}),
        ("POST", "/api/v1/cases/lookup", {}),
        ("POST", "/api/v1/auth/login", {}),
        ("GET", QUEUE_URL, auth_headers),
        ("PATCH", f"{QUEUE_URL}/00000000-0000-4000-8000-000000000000/status", auth_headers),
    ):
        response = api_client.request(method, url, json=oversized, headers=headers)

        assert response.status_code == 413, f"{method} {url}"


def test_an_oversized_body_without_a_content_length_is_still_rejected(
    api_client: TestClient,
) -> None:
    """A chunked upload declares no length, so the bytes must be counted.

    Without this the body would be assembled in memory before any
    application-level check could look at it.
    """

    def chunks():
        # Well past the 64 KiB ceiling, streamed in pieces.
        for _ in range(40):
            yield b"x" * 4096

    response = api_client.post(
        REPORTS_URL,
        content=chunks(),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413


def test_a_body_just_under_the_ceiling_is_accepted(api_client: TestClient) -> None:
    """The limit must not be so tight that a long, legitimate report fails."""
    # 20,000 characters is the largest description the schema allows; it must
    # comfortably fit inside the 64 KiB HTTP ceiling.
    response = api_client.post(
        REPORTS_URL,
        json={
            "category": "OTHER",
            "description": "x" * DESCRIPTION_MAX_LENGTH,
            "evidence_url": "https://example.com/" + "a" * 1_900,
        },
    )

    assert response.status_code == 201


def test_a_malformed_content_type_does_not_crash(api_client: TestClient) -> None:
    for content_type in ("text/plain", "application/xml", "", "not/a-type"):
        response = api_client.post(
            REPORTS_URL, content=b'{"category":"OTHER"}', headers={"Content-Type": content_type}
        )

        assert response.status_code in (415, 422), content_type
        assert "Traceback" not in response.text
