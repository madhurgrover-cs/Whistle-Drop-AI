"""What the application is allowed to write to a log.

Logs are a lower-trust store than the database: shipped to aggregators, read by
more people, retained longer, backed up more casually. These tests drive real
traffic with a handler attached to the root logger and then assert that the
captured output contains none of the values that must never be written.

They capture at DEBUG on the root logger, so they see everything the
application and its libraries emit during the request — not only what the
configured level would normally let through.
"""

import logging
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import CaseUpdate, Moderator, Report
from tests.conftest import TEST_CASE_CODE_PEPPER, TEST_JWT_SECRET_KEY

pytestmark = pytest.mark.integration

REPORTS_URL = "/api/v1/reports"
LOOKUP_URL = "/api/v1/cases/lookup"
LOGIN_URL = "/api/v1/auth/login"
QUEUE_URL = "/api/v1/moderation/reports"

# Deliberately distinctive so a substring search cannot match by accident.
SECRET_DESCRIPTION = "UNIQUEMARKER-payroll-spreadsheet-with-every-salary-was-left-on-a-public-share"
SECRET_NOTE = "UNIQUEMARKER-internal-the-reporter-is-probably-in-the-finance-team"
SECRET_EVIDENCE_URL = "https://example.com/UNIQUEMARKER-share?token=UNIQUEMARKER-secret-token"


class LogCollector:
    """Captures every record reaching the root logger while active."""

    def __init__(self) -> None:
        self.records: list[logging.LogRecord] = []
        self._handler: logging.Handler | None = None
        self._previous_level: int | None = None

    def __enter__(self) -> "LogCollector":
        collector = self

        class Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                collector.records.append(record)

        root = logging.getLogger()
        self._previous_level = root.level
        root.setLevel(logging.DEBUG)
        self._handler = Handler(level=logging.DEBUG)
        root.addHandler(self._handler)
        return self

    def __exit__(self, *exc_info: object) -> None:
        root = logging.getLogger()
        if self._handler is not None:
            root.removeHandler(self._handler)
        if self._previous_level is not None:
            root.setLevel(self._previous_level)

    @property
    def text(self) -> str:
        """Everything captured, formatted the way a log file would hold it."""
        parts: list[str] = []
        for record in self.records:
            parts.append(record.name)
            parts.append(str(record.levelname))
            try:
                parts.append(record.getMessage())
            except Exception:  # pragma: no cover - a broken format string
                parts.append(str(record.msg))
            if record.exc_info:
                parts.append(logging.Formatter().formatException(record.exc_info))
        return "\n".join(parts)


@pytest.fixture
def logs() -> Iterator[LogCollector]:
    with LogCollector() as collector:
        yield collector


# ---------------------------------------------------------------------------
# 15-16, 20. Report contents and case codes
# ---------------------------------------------------------------------------


def test_a_successful_submission_logs_neither_the_code_nor_the_report(
    api_client: TestClient, db_session: Session, logs: LogCollector
) -> None:
    response = api_client.post(
        REPORTS_URL,
        json={
            "category": "CORRUPTION",
            "description": SECRET_DESCRIPTION,
            "evidence_url": SECRET_EVIDENCE_URL,
        },
    )
    case_code = response.json()["case_code"]
    stored = db_session.scalars(select(Report)).one()

    captured = logs.text
    assert case_code not in captured
    assert case_code.replace("-", "") not in captured
    assert SECRET_DESCRIPTION not in captured
    assert SECRET_EVIDENCE_URL not in captured
    assert stored.case_code_hash not in captured
    assert "UNIQUEMARKER" not in captured


def test_a_case_lookup_logs_neither_the_submitted_code_nor_the_digest(
    api_client: TestClient, db_session: Session, logs: LogCollector
) -> None:
    code = api_client.post(
        REPORTS_URL, json={"category": "OTHER", "description": SECRET_DESCRIPTION}
    ).json()["case_code"]
    stored = db_session.scalars(select(Report)).one()

    api_client.post(LOOKUP_URL, json={"case_code": code})

    captured = logs.text
    assert code not in captured
    assert stored.case_code_hash not in captured


def test_a_failed_lookup_does_not_log_the_attempted_code(
    api_client: TestClient, logs: LogCollector
) -> None:
    """A mistyped code is often a real one with a typo."""
    attempted = "WD-MYTYPO-BADCODE-9999X"

    api_client.post(LOOKUP_URL, json={"case_code": attempted})

    assert attempted not in logs.text
    assert "MYTYPO" not in logs.text


def test_a_database_failure_does_not_log_the_report_body(
    api_client: TestClient, db_session: Session, logs: LogCollector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finding this phase exists to catch.

    A DBAPI error normally embeds the bound parameters — the report body and
    the case-code hash among them — and ``logger.exception`` would write the
    lot to disk. The engine is created with ``hide_parameters=True`` to stop
    that. Here the second insert collides on the unique case-code index, which
    is exactly such an error.
    """
    from app.core.case_codes import generate_case_code

    fixed = generate_case_code()
    monkeypatch.setattr("app.services.reports.generate_case_code", lambda: fixed)

    api_client.post(REPORTS_URL, json={"category": "OTHER", "description": SECRET_DESCRIPTION})
    # Every retry now collides, exhausting the attempts and raising.
    api_client.post(REPORTS_URL, json={"category": "OTHER", "description": SECRET_DESCRIPTION})

    captured = logs.text
    assert SECRET_DESCRIPTION not in captured
    assert "UNIQUEMARKER" not in captured


def test_moderation_traffic_does_not_log_report_contents(
    api_client: TestClient, auth_headers: dict, logs: LogCollector, report_id_for, submit_report
) -> None:
    submitted = submit_report(description=SECRET_DESCRIPTION)
    report_id = report_id_for(submitted["case_code"])

    api_client.get(QUEUE_URL, headers=auth_headers)
    api_client.get(f"{QUEUE_URL}/{report_id}", headers=auth_headers)
    api_client.patch(
        f"{QUEUE_URL}/{report_id}/status",
        headers=auth_headers,
        json={"status": "UNDER_REVIEW", "note": SECRET_NOTE},
    )

    captured = logs.text
    assert SECRET_DESCRIPTION not in captured
    assert SECRET_NOTE not in captured
    assert submitted["case_code"] not in captured


# ---------------------------------------------------------------------------
# 17-19. Credentials and secrets
# ---------------------------------------------------------------------------


def test_a_successful_login_logs_neither_password_nor_token(
    api_client: TestClient, moderator: tuple[Moderator, str], logs: LogCollector
) -> None:
    account, password = moderator

    response = api_client.post(LOGIN_URL, json={"username": account.username, "password": password})
    token = response.json()["access_token"]

    captured = logs.text
    assert password not in captured
    assert token not in captured
    assert account.password_hash not in captured
    assert "$2b$" not in captured


def test_a_failed_login_logs_neither_the_username_nor_the_attempt(
    api_client: TestClient, moderator: tuple[Moderator, str], logs: LogCollector
) -> None:
    """A failed login is often a real moderator's typo."""
    account, _ = moderator
    attempted = "UNIQUEMARKER-wrong-password-attempt"

    api_client.post(LOGIN_URL, json={"username": account.username, "password": attempted})

    captured = logs.text
    assert attempted not in captured
    assert account.username not in captured


def test_a_rejected_token_is_not_written_to_the_log(
    api_client: TestClient, logs: LogCollector
) -> None:
    forged = "eyJhbGciOiJIUzI1NiJ9.UNIQUEMARKERPAYLOAD.UNIQUEMARKERSIGNATURE"

    api_client.get(QUEUE_URL, headers={"Authorization": f"Bearer {forged}"})

    captured = logs.text
    assert forged not in captured
    assert "UNIQUEMARKERPAYLOAD" not in captured


def test_no_configured_secret_ever_reaches_a_log(
    api_client: TestClient, moderator: tuple[Moderator, str], logs: LogCollector
) -> None:
    """Drive a bit of everything, then look for the two keys."""
    account, password = moderator

    code = api_client.post(
        REPORTS_URL, json={"category": "OTHER", "description": SECRET_DESCRIPTION}
    ).json()["case_code"]
    api_client.post(LOOKUP_URL, json={"case_code": code})
    api_client.post(LOGIN_URL, json={"username": account.username, "password": password})
    api_client.get(QUEUE_URL)

    captured = logs.text
    assert TEST_CASE_CODE_PEPPER not in captured
    assert TEST_JWT_SECRET_KEY not in captured
    assert "CASE_CODE_PEPPER" not in captured
    assert "JWT_SECRET_KEY" not in captured


def test_the_rate_limiter_logs_no_address_and_no_fingerprint(
    test_settings: Settings, db_session: Session, logs: LogCollector
) -> None:
    from app.core import rate_limit
    from app.core.config import get_settings
    from app.db.session import get_db
    from app.main import create_app

    rate_limit.reset()
    settings = test_settings.model_copy(
        update={"rate_limit_enabled": True, "rate_limit_reports": "1/minute"}
    )
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = lambda: db_session

    try:
        with TestClient(app) as client:
            body = {"category": "OTHER", "description": SECRET_DESCRIPTION}
            client.post(REPORTS_URL, json=body)
            blocked = client.post(REPORTS_URL, json=body)
            assert blocked.status_code == 429

            fingerprint_seen = False
            captured = logs.text
            for record in logs.records:
                if "Rate limit reached" in record.getMessage():
                    fingerprint_seen = True

            assert fingerprint_seen, "the throttle was not logged at all"
            # The event is recorded; the caller is not.
            assert "testclient" not in captured
            assert "127.0.0.1" not in captured
    finally:
        rate_limit.reset()


# ---------------------------------------------------------------------------
# The policy, asserted against the source
# ---------------------------------------------------------------------------


def test_no_log_call_interpolates_a_sensitive_variable() -> None:
    """Every log call site logs an event, not a value.

    Checked against the parsed source: a ``logger.*`` call whose arguments
    mention a sensitive name is a defect, however carefully it was written.
    """
    import ast
    from pathlib import Path

    from tests.conftest import PROJECT_ROOT

    forbidden = {
        "password",
        "password_hash",
        "case_code",
        "case_code_hash",
        "description",
        "token",
        "access_token",
        "secret",
        "pepper",
        "evidence_url",
        "note",
    }
    offenders: list[str] = []

    for path in Path(PROJECT_ROOT, "app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if not (isinstance(func.value, ast.Name) and func.value.id == "logger"):
                continue

            for argument in node.args[1:] + [kw.value for kw in node.keywords]:
                for inner in ast.walk(argument):
                    name = None
                    if isinstance(inner, ast.Name):
                        name = inner.id
                    elif isinstance(inner, ast.Attribute):
                        name = inner.attr
                    if name and name.lower() in forbidden:
                        offenders.append(
                            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} logs {name!r}"
                        )

    assert offenders == [], f"sensitive values reach a log call: {offenders}"


def test_sql_echo_is_never_enabled(test_settings: Settings) -> None:
    """Echoing statements would print report text at INFO level."""
    from app.db.session import create_db_engine

    engine = create_db_engine(test_settings)
    try:
        assert engine.echo is False
    finally:
        engine.dispose()


def test_the_engine_hides_bound_parameters(test_settings: Settings) -> None:
    """The setting that keeps report bodies out of DBAPI error messages."""
    from app.db.session import create_db_engine

    engine = create_db_engine(test_settings)
    try:
        assert engine.hide_parameters is True
    finally:
        engine.dispose()


def test_running_migrations_does_not_disable_application_logging() -> None:
    """Alembic's fileConfig would otherwise silence every ``app.*`` logger.

    ``logging.config.fileConfig`` defaults to ``disable_existing_loggers=True``,
    which sets ``.disabled`` on every logger not named in ``alembic.ini``. A
    migrate-then-serve entrypoint would afterwards emit no application logs at
    all, without saying so. ``alembic/env.py`` passes False; this is the guard
    that keeps it that way.
    """
    from pathlib import Path

    from tests.conftest import PROJECT_ROOT

    # The db_engine fixture already ran migrations in this process.
    for name in ("app", "app.services.auth", "app.core.rate_limit"):
        assert logging.getLogger(name).disabled is False, name

    env_source = Path(PROJECT_ROOT, "alembic", "env.py").read_text(encoding="utf-8")
    assert "disable_existing_loggers=False" in env_source


def test_no_module_prints_to_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """``print`` bypasses the logging configuration entirely.

    The seeding CLI is a terminal program and prints deliberately; nothing
    under ``app/`` may.
    """
    import ast
    from pathlib import Path

    from tests.conftest import PROJECT_ROOT

    offenders: list[str] = []
    for path in Path(PROJECT_ROOT, "app").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")

    assert offenders == [], f"print() in application code: {offenders}"


def test_report_ids_in_paths_are_the_only_identifier_logged(
    api_client: TestClient, auth_headers: dict, logs: LogCollector
) -> None:
    """A 404 logs the path, which carries a report id — and nothing else.

    Report ids are internal identifiers that say nothing about who filed a
    report. Documented rather than hidden.
    """
    missing = uuid.uuid4()

    api_client.get(f"{QUEUE_URL}/{missing}", headers=auth_headers)

    captured = logs.text
    assert "REPORT_NOT_FOUND" in captured  # the event is recorded
    assert str(missing) in captured  # the path, and therefore the id


def test_case_codes_never_travel_in_a_url(client: TestClient) -> None:
    """Why the access log cannot contain one.

    Case codes are sent in request bodies, never in paths or query strings —
    the reason ``/cases/lookup`` is a POST. Uvicorn's access log records the
    request line only, so no access log line has ever held a case code.
    """
    schema = client.get("/openapi.json").json()

    for path, operations in schema["paths"].items():
        assert "case_code" not in path
        for operation in operations.values():
            for parameter in operation.get("parameters", []):
                assert parameter["name"] != "case_code"
                assert "case" not in parameter["name"].lower()


def test_case_updates_and_reports_hold_no_ip_address(db_session: Session) -> None:
    """Rate limiting must not have introduced a stored identifier."""
    from sqlalchemy import inspect

    inspector = inspect(db_session.get_bind())
    for table in inspector.get_table_names():
        columns = {column["name"].lower() for column in inspector.get_columns(table)}
        assert not columns & {
            "ip",
            "ip_address",
            "client_ip",
            "remote_addr",
            "user_agent",
            "fingerprint",
        }

    # And no row content carries one either.
    assert db_session.scalars(select(CaseUpdate)).all() is not None
