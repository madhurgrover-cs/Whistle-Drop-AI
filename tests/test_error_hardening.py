"""Error responses, secret configuration, and database safety.

Three reviews that share one concern: nothing internal may cross the boundary —
not into a response, not into a tracked file, not into a SQL string.
"""

import ast
import subprocess
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.main import create_app
from app.models import Moderator, Report
from tests.conftest import (
    PROJECT_ROOT,
    TEST_CASE_CODE_PEPPER,
    TEST_JWT_SECRET_KEY,
    make_moderator_password,
)

pytestmark = pytest.mark.integration

REPORTS_URL = "/api/v1/reports"
LOOKUP_URL = "/api/v1/cases/lookup"
LOGIN_URL = "/api/v1/auth/login"
QUEUE_URL = "/api/v1/moderation/reports"

APP_DIR = PROJECT_ROOT / "app"

# Strings that must never appear in a response body, whatever went wrong.
LEAK_MARKERS = (
    "traceback",
    "sqlalchemy",
    "psycopg",
    "postgresql://",
    "postgresql+psycopg",
    "/users/",
    "c:\\users",
    "site-packages",
    '.py", line',
    "select ",
    "insert into",
    "duplicate key",
    "constraint",
    "$2b$",
    TEST_CASE_CODE_PEPPER.lower(),
    TEST_JWT_SECRET_KEY.lower(),
)


def assert_no_leak(response) -> None:
    body = response.text.lower()
    for marker in LEAK_MARKERS:
        assert marker not in body, f"{marker!r} leaked in {response.status_code}: {response.text}"


# ---------------------------------------------------------------------------
# 21-22. Unexpected failures
# ---------------------------------------------------------------------------


def test_an_unexpected_error_returns_an_opaque_500(
    serving_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash in a service must not become a traceback on the wire."""

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "internal detail: /srv/whistledrop/app/services/reports.py line 42, pepper=super-secret"
        )

    monkeypatch.setattr("app.services.reports.ReportService.submit", explode)

    response = serving_client.post(
        REPORTS_URL,
        json={"category": "OTHER", "description": "A body that triggers the failure below."},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "INTERNAL_ERROR", "message": "The request could not be completed."}
    }
    assert "pepper" not in response.text
    assert "services/reports.py" not in response.text
    assert_no_leak(response)


def test_a_database_error_does_not_expose_sql(
    serving_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy.exc import OperationalError

    def explode(*args: object, **kwargs: object) -> None:
        raise OperationalError(
            "SELECT reports.case_code_hash FROM reports WHERE id = %(id)s",
            {"id": "secret"},
            Exception("connection to server at 10.0.0.5 port 5432 failed"),
        )

    monkeypatch.setattr("app.services.cases.CaseService.lookup", explode)

    response = serving_client.post(LOOKUP_URL, json={"case_code": "WD-4K7PQ-92MRT-XJ3HN-B8VZ6"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "10.0.0.5" not in response.text
    assert_no_leak(response)


def test_no_endpoint_leaks_internals_on_any_kind_of_bad_input(
    api_client: TestClient, auth_headers: dict
) -> None:
    """A sweep across every route and several shapes of nonsense."""
    attempts = [
        api_client.post(REPORTS_URL, json={}),
        api_client.post(REPORTS_URL, json={"category": "NOPE", "description": "x"}),
        api_client.post(
            REPORTS_URL, content=b"{bad json", headers={"Content-Type": "application/json"}
        ),
        api_client.post(LOOKUP_URL, json={"case_code": "' OR 1=1 --"}),
        api_client.post(LOOKUP_URL, json={}),
        api_client.post(LOGIN_URL, json={"username": "x", "password": "y"}),
        api_client.get(QUEUE_URL),
        api_client.get(f"{QUEUE_URL}/not-a-uuid", headers=auth_headers),
        api_client.get(f"{QUEUE_URL}/{uuid.uuid4()}", headers=auth_headers),
        api_client.patch(
            f"{QUEUE_URL}/{uuid.uuid4()}/status", headers=auth_headers, json={"status": "NOPE"}
        ),
        api_client.get("/api/v1/does-not-exist"),
        api_client.delete(REPORTS_URL),
    ]

    for response in attempts:
        assert_no_leak(response)
        assert response.status_code >= 400
        # Every failure uses the one envelope.
        assert set(response.json()) == {"error"}, response.text
        assert set(response.json()["error"]) <= {"code", "message", "details"}


def test_debug_mode_never_reaches_the_framework(test_settings: Settings) -> None:
    """FastAPI's debug mode would serve an interactive traceback page.

    The settings' own ``debug`` flag is deliberately not wired to it, so a
    misconfigured deployment cannot turn source-code disclosure on.
    """
    assert test_settings.debug is True  # the setting is on in tests

    app = create_app(test_settings)

    assert app.debug is False


def test_validation_errors_never_echo_the_submitted_value(api_client: TestClient) -> None:
    """Pydantic's raw errors carry the rejected input; ours must not."""
    secrets_attempted = {
        "case_code": "WD-SECRET-VALUE-HERE-XXXXX",
        "password": "my-actual-password-oops",
        "description": "a-report-body-that-was-rejected",
    }

    responses = [
        api_client.post(LOOKUP_URL, json={"case_code": secrets_attempted["case_code"], "x": 1}),
        api_client.post(LOGIN_URL, json={"password": secrets_attempted["password"]}),
        api_client.post(REPORTS_URL, json={"description": secrets_attempted["description"]}),
    ]

    for response, value in zip(responses, secrets_attempted.values(), strict=True):
        assert response.status_code == 422
        assert value not in response.text


# ---------------------------------------------------------------------------
# 24. Deliberate behaviour that must survive hardening
# ---------------------------------------------------------------------------


def test_authentication_failures_are_still_indistinguishable(
    api_client: TestClient,
    moderator: tuple[Moderator, str],
    inactive_moderator: tuple[Moderator, str],
) -> None:
    account, _ = moderator
    disabled, disabled_password = inactive_moderator

    bodies = {
        api_client.post(
            LOGIN_URL, json={"username": account.username, "password": make_moderator_password()}
        ).text,
        api_client.post(
            LOGIN_URL, json={"username": "no-such-account", "password": make_moderator_password()}
        ).text,
        api_client.post(
            LOGIN_URL, json={"username": disabled.username, "password": disabled_password}
        ).text,
    }

    assert len(bodies) == 1, f"hardening made authentication failures distinguishable: {bodies}"


def test_case_lookup_failures_are_still_indistinguishable(api_client: TestClient) -> None:
    from app.core.case_codes import generate_case_code

    malformed = api_client.post(LOOKUP_URL, json={"case_code": "garbage"})
    unknown = api_client.post(LOOKUP_URL, json={"case_code": generate_case_code()})

    assert malformed.status_code == unknown.status_code == 404
    assert malformed.json() == unknown.json()


# ---------------------------------------------------------------------------
# 25-27. Secrets
# ---------------------------------------------------------------------------


def _settings_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "app_env": "test",
        "database_url": "postgresql+psycopg://x:y@localhost:5433/whistledrop_test",
        "case_code_pepper": TEST_CASE_CODE_PEPPER,
        "jwt_secret_key": TEST_JWT_SECRET_KEY,
        "_env_file": None,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize("secret", ["case_code_pepper", "jwt_secret_key"])
def test_a_required_secret_has_no_default(secret: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError

    monkeypatch.delenv(secret.upper(), raising=False)
    kwargs = _settings_kwargs()
    kwargs.pop(secret)

    with pytest.raises(ValidationError, match=secret):
        Settings(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "secret,placeholder",
    [
        ("case_code_pepper", Settings.PLACEHOLDER_PEPPER),
        ("jwt_secret_key", Settings.PLACEHOLDER_JWT_SECRET),
    ],
)
def test_a_shipped_placeholder_cannot_be_used(secret: str, placeholder: str) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="placeholder"):
        Settings(**_settings_kwargs(**{secret: placeholder}))  # type: ignore[arg-type]


@pytest.mark.parametrize("secret", ["case_code_pepper", "jwt_secret_key"])
def test_a_secret_is_masked_in_every_representation(secret: str) -> None:
    settings = Settings(**_settings_kwargs())  # type: ignore[arg-type]
    value = getattr(settings, secret).get_secret_value()

    assert value not in repr(settings)
    assert value not in str(settings)
    assert value not in str(settings.model_dump())
    assert value not in settings.model_dump_json()


def test_the_two_secrets_are_not_interchangeable(test_settings: Settings) -> None:
    """Reusing one key for both would make a leak of either a leak of both."""
    assert (
        test_settings.case_code_pepper.get_secret_value()
        != test_settings.jwt_secret_key.get_secret_value()
    )


def test_the_env_file_is_not_tracked() -> None:
    tracked = subprocess.run(  # noqa: S603
        ["git", "ls-files"],  # noqa: S607
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    assert ".env" not in tracked
    assert not [path for path in tracked if path.endswith("/.env")]
    assert ".env.example" in tracked  # the template is tracked, and holds no key


def test_no_real_secret_appears_in_any_tracked_or_untracked_file() -> None:
    """Whatever the local .env holds must exist nowhere else in the tree."""
    from dotenv import dotenv_values

    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        pytest.skip("No local .env to cross-check.")

    secrets = [
        value
        for key, value in dotenv_values(env_file).items()
        if value
        and len(value) >= 16
        and any(marker in key for marker in ("SECRET", "PEPPER", "PASSWORD"))
    ]
    assert secrets, "expected the local .env to hold at least one secret"

    files = subprocess.run(  # noqa: S603
        ["git", "ls-files", "-co", "--exclude-standard"],  # noqa: S607
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    offenders: list[str] = []
    for relative in files:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        offenders.extend(relative for secret in secrets if secret in content)

    assert offenders == [], f"real secrets found in {sorted(set(offenders))}"


def test_gitignore_still_covers_the_env_file() -> None:
    ignored = subprocess.run(  # noqa: S603
        ["git", "check-ignore", "-v", ".env"],  # noqa: S607
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert ignored.returncode == 0
    assert ".gitignore" in ignored.stdout


# ---------------------------------------------------------------------------
# 28-30. Database safety
# ---------------------------------------------------------------------------


def test_no_application_code_builds_sql_from_a_string() -> None:
    """Every query goes through SQLAlchemy's expression language.

    The only ``text()`` calls under ``app/`` are static server defaults in the
    model definitions. Checked against the parsed source: a ``text()`` whose
    argument is anything but a plain literal — an f-string, a concatenation, a
    ``.format()`` — would be a place user input could reach SQL.
    """
    offenders: list[str] = []

    for path in APP_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name not in {"text", "execute", "exec_driver_sql"}:
                continue
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    continue  # a literal is fine
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} {name}() "
                    f"with a {type(argument).__name__} argument"
                )

    assert offenders == [], f"dynamic SQL construction: {offenders}"


def test_user_input_reaches_the_database_only_as_a_bound_parameter(
    api_client: TestClient, db_session: Session
) -> None:
    """A classic injection payload is stored as text and changes nothing."""
    payload = "'; DROP TABLE reports; --"

    response = api_client.post(
        REPORTS_URL,
        json={"category": "OTHER", "description": f"A report containing {payload} verbatim."},
    )

    assert response.status_code == 201
    assert inspect(db_session.get_bind()).has_table("reports")
    stored = db_session.scalars(select(Report)).one()
    assert payload in stored.description  # stored, not executed


def test_a_sql_payload_in_a_case_code_is_never_executed(
    api_client: TestClient, db_session: Session
) -> None:
    for payload in ("' OR '1'='1", "'; DROP TABLE reports; --", "%' --"):
        response = api_client.post(LOOKUP_URL, json={"case_code": payload})

        assert response.status_code == 404  # refused before any query runs
        assert inspect(db_session.get_bind()).has_table("reports")


def test_a_sql_payload_in_a_login_is_never_executed(
    api_client: TestClient, db_session: Session
) -> None:
    response = api_client.post(
        LOGIN_URL, json={"username": "admin' --", "password": make_moderator_password()}
    )

    assert response.status_code == 401
    assert inspect(db_session.get_bind()).has_table("moderators")


def test_case_lookup_is_still_an_indexed_digest_lookup(db_session: Session) -> None:
    """The unique index is what makes tracking a single-row read.

    Losing it would turn every lookup into a full scan — and a scan is what a
    plaintext search would have required, which is the design this avoids.
    """
    indexes = {
        index["name"]: index for index in inspect(db_session.get_bind()).get_indexes("reports")
    }

    assert "ix_reports_case_code_hash" in indexes
    assert indexes["ix_reports_case_code_hash"]["unique"] is True
    assert indexes["ix_reports_case_code_hash"]["column_names"] == ["case_code_hash"]


def test_no_query_searches_reports_by_plaintext_case_code() -> None:
    """There is no column to search, and no code that tries."""
    columns = {"case_code", "code", "plaintext_case_code"}
    sources = "\n".join(path.read_text(encoding="utf-8") for path in APP_DIR.rglob("*.py"))

    assert "Report.case_code ==" not in sources
    for column in columns:
        assert f'Report.{column}"' not in sources

    from app.models import Report as ReportModel

    assert not columns & set(ReportModel.__table__.columns.keys())


def test_the_moderation_filters_are_bound_not_interpolated(
    api_client: TestClient, auth_headers: dict
) -> None:
    """Filter values come from an enum, so nothing arbitrary reaches SQL."""
    for value in ("SUBMITTED'; DROP TABLE reports; --", "%", "_", "' OR 1=1 --"):
        response = api_client.get(QUEUE_URL, headers=auth_headers, params={"status": value})

        assert response.status_code == 422  # rejected by the enum, before SQL


def test_a_report_id_must_be_a_uuid_before_it_reaches_the_database(
    api_client: TestClient, auth_headers: dict
) -> None:
    response = api_client.get(f"{QUEUE_URL}/1 OR 1=1", headers=auth_headers)

    assert response.status_code == 422
    assert_no_leak(response)


# ---------------------------------------------------------------------------
# Authentication review
# ---------------------------------------------------------------------------


def test_moderator_identity_still_comes_only_from_the_token(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    report_id = report_id_for(submit_report()["case_code"])

    response = api_client.patch(
        f"{QUEUE_URL}/{report_id}/status",
        headers=auth_headers,
        json={"status": "UNDER_REVIEW", "moderator_id": str(uuid.uuid4())},
    )

    assert response.status_code == 422
    fields = {detail["field"] for detail in response.json()["error"]["details"]}
    assert "body.moderator_id" in fields


def test_the_token_algorithm_is_still_restricted(test_settings: Settings) -> None:
    from pydantic import ValidationError

    assert test_settings.jwt_algorithm in {"HS256", "HS384", "HS512"}

    for rejected in ("none", "RS256", "ES256"):
        with pytest.raises(ValidationError):
            Settings(**_settings_kwargs(jwt_algorithm=rejected))  # type: ignore[arg-type]


def test_hardening_did_not_open_the_moderation_routes(api_client: TestClient) -> None:
    for method, url in (
        ("get", QUEUE_URL),
        ("get", f"{QUEUE_URL}/{uuid.uuid4()}"),
    ):
        assert getattr(api_client, method)(url).status_code == 401


def test_hardening_did_not_close_the_public_routes(api_client: TestClient) -> None:
    submitted = api_client.post(
        REPORTS_URL,
        json={"category": "SECURITY", "description": "Anonymous reporting after hardening."},
    )
    assert submitted.status_code == 201

    lookup = api_client.post(LOOKUP_URL, json={"case_code": submitted.json()["case_code"]})
    assert lookup.status_code == 200

    assert api_client.get("/api/v1/health").status_code == 200


def test_production_settings_refuse_a_weak_configuration() -> None:
    """A deployment cannot start with test-grade password hashing."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="at least 12 in production"):
        Settings(**_settings_kwargs(app_env="production", password_hash_rounds=4))  # type: ignore[arg-type]


def test_a_500_still_carries_the_security_headers(
    serving_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headers middleware is outermost precisely so this holds."""

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("app.services.reports.ReportService.submit", explode)

    response = serving_client.post(
        REPORTS_URL, json={"category": "OTHER", "description": "A body long enough to pass."}
    )

    assert response.status_code == 500
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_the_health_endpoint_discloses_nothing_sensitive(api_client: TestClient) -> None:
    body = api_client.get("/api/v1/health").json()

    assert set(body) == {"status", "service", "version", "environment"}
    assert "database" not in str(body).lower()
    assert "postgres" not in str(body).lower()


def test_settings_are_never_exposed_through_an_endpoint(
    api_client: TestClient, auth_headers: dict
) -> None:
    for url in ("/api/v1/health", "/openapi.json", QUEUE_URL):
        response = api_client.get(url, headers=auth_headers)

        assert TEST_CASE_CODE_PEPPER not in response.text
        assert TEST_JWT_SECRET_KEY not in response.text
        assert "DATABASE_URL" not in response.text
        assert "whistledrop_test" not in response.text
