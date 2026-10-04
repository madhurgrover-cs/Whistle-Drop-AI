"""Shared pytest fixtures.

Two rules govern this file.

*The developer's database is never touched.* Integration tests run against the
disposable container on ``TEST_DATABASE_URL`` (port 5433), and
:func:`_assert_is_a_test_database` refuses to proceed if that DSN looks
anything like the development one.

*Schema comes from migrations, not from ``create_all``.* The tables under test
are built by running Alembic, so a passing suite is also evidence that the
migrations themselves work.
"""

import os
import secrets
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import create_db_engine
from app.main import create_app
from app.models import Moderator
from app.services.auth import AuthService

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# A DSN that is not obviously disposable is treated as production and refused.
_REQUIRED_TEST_DB_MARKER = "_test"

# Fixed, obviously-fake HMAC key for the test session. Fixed rather than random
# because several tests assert that hashing is deterministic, and the value is
# visibly not a secret so it cannot be mistaken for one that matters.
TEST_CASE_CODE_PEPPER = "test-pepper-not-a-real-secret-0123456789abcdef"

# Likewise for token signing. Distinct from the pepper so a test that confused
# the two keys would fail rather than silently pass.
TEST_JWT_SECRET_KEY = "test-jwt-signing-key-not-a-real-secret-fedcba9876543210"  # noqa: S105

# bcrypt at its minimum work factor. Hashing is ~1 ms instead of ~250 ms, which
# keeps a suite with dozens of logins fast. Settings refuses this in production.
TEST_PASSWORD_HASH_ROUNDS = 4


def _resolve_test_database_url() -> str | None:
    """Find the test DSN, preferring an explicit environment variable."""
    from dotenv import dotenv_values

    explicit = os.getenv("TEST_DATABASE_URL")
    if explicit:
        return explicit

    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        return dotenv_values(env_file).get("TEST_DATABASE_URL")

    return None


def _assert_is_a_test_database(url: str) -> None:
    """Refuse to run against anything that is not plainly a test database."""
    database_name = url.rsplit("/", 1)[-1].split("?", 1)[0]
    if not database_name.endswith(_REQUIRED_TEST_DB_MARKER):
        raise RuntimeError(
            f"Refusing to run integration tests against database {database_name!r}: "
            f"its name must end in {_REQUIRED_TEST_DB_MARKER!r}."
        )

    configured = os.getenv("DATABASE_URL")
    if configured and configured == url:
        raise RuntimeError(
            "TEST_DATABASE_URL and DATABASE_URL are the same database. "
            "Tests must never run against the development database."
        )


# ---------------------------------------------------------------------------
# Application fixtures (no database required)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Settings used by the whole test session."""
    return Settings(
        app_env="test",
        debug=True,
        enable_docs=True,
        database_url=_resolve_test_database_url()
        or "postgresql+psycopg://test:test@localhost:5433/whistledrop_test",
        case_code_pepper=TEST_CASE_CODE_PEPPER,
        jwt_secret_key=TEST_JWT_SECRET_KEY,
        jwt_algorithm="HS256",
        jwt_access_token_expire_minutes=30,
        password_hash_rounds=TEST_PASSWORD_HASH_ROUNDS,
        # Off for the suite at large: hundreds of tests hitting the same three
        # endpoints from one address would throttle each other and make
        # failures depend on test order. tests/test_rate_limiting.py builds its
        # own application with limiting on and tiny limits.
        rate_limit_enabled=False,
        # The model is loaded lazily on first use instead. Warming would make
        # every one of the many apps this suite builds pay the sklearn import.
        ml_warm_start=False,
    )


@pytest.fixture(scope="session")
def app(test_settings: Settings) -> FastAPI:
    """A FastAPI app wired with test settings."""
    application = create_app(test_settings)
    application.dependency_overrides[get_settings] = lambda: test_settings
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """HTTP client bound to the test application."""
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Database fixtures (integration tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """The disposable test DSN, or skip the whole integration suite."""
    url = _resolve_test_database_url()
    if not url:
        pytest.skip(
            "TEST_DATABASE_URL is not set. Start the test container with "
            "`docker compose up -d db_test` and set it in .env."
        )
    _assert_is_a_test_database(url)
    return url


@pytest.fixture(scope="session")
def db_engine(test_database_url: str) -> Iterator[Engine]:
    """A session-wide engine bound to a freshly migrated test database.

    The database is taken down to base and back up to head, so every run starts
    from a schema built by the migrations and not by leftovers from last time.
    """
    engine = create_db_engine(
        Settings(  # type: ignore[call-arg]
            app_env="test",
            database_url=test_database_url,
            case_code_pepper=TEST_CASE_CODE_PEPPER,
            jwt_secret_key=TEST_JWT_SECRET_KEY,
            password_hash_rounds=TEST_PASSWORD_HASH_ROUNDS,
        )
    )

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except OperationalError as exc:  # pragma: no cover - environment problem
        engine.dispose()
        pytest.skip(f"Test PostgreSQL is not reachable: {exc}")

    alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
    alembic_config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    # env.py reads this; it is what keeps migrations pointed at the test
    # container rather than at DATABASE_URL.
    alembic_config.set_main_option("sqlalchemy.url", test_database_url)
    os.environ["ALEMBIC_DATABASE_URL"] = test_database_url

    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")

    yield engine

    engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Iterator[Session]:
    """A session whose work is rolled back when the test ends.

    Each test runs inside an outer transaction on a dedicated connection. The
    test can commit as much as it likes — those commits land in a SAVEPOINT —
    and the outer transaction is rolled back afterwards, so tests share a schema
    but never share data.
    """
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


# ---------------------------------------------------------------------------
# API fixtures (integration tests that go through HTTP)
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client(app: FastAPI, db_session: Session) -> Iterator[TestClient]:
    """A client whose request handlers share the test's own session.

    Overriding ``get_db`` with the rolled-back session is what lets a test make
    a real HTTP call and then assert against the rows it produced. Because the
    handler and the assertions run on one session inside one transaction,
    nothing survives the test.
    """
    from app.db.session import get_db

    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def submitted_report(api_client: TestClient) -> dict[str, object]:
    """A report filed through the real endpoint.

    Returns the submission response, so a test that needs an existing case has
    the one plaintext case code that will ever exist for it.
    """
    response = api_client.post(
        "/api/v1/reports",
        json={
            "category": "CORRUPTION",
            "description": "Procurement contracts are being awarded without any tender process.",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Moderator authentication fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_service(db_session: Session, test_settings: Settings) -> AuthService:
    """An :class:`AuthService` on the test session."""
    return AuthService(
        db_session,
        jwt_secret_key=test_settings.jwt_secret_key.get_secret_value(),
        jwt_algorithm=test_settings.jwt_algorithm,
        access_token_expire_minutes=test_settings.jwt_access_token_expire_minutes,
        password_hash_rounds=test_settings.password_hash_rounds,
    )


def make_moderator_password() -> str:
    """A fresh random password for a test account.

    Generated rather than hardcoded so that no literal in this repository is
    ever a working credential, even against the disposable test database.
    """
    return f"test-{secrets.token_urlsafe(16)}"


def make_moderator_username() -> str:
    """A username unique to this call.

    Generated per call rather than held in a shared fixture: a test that asks
    for both an active and an inactive moderator must get two distinct
    accounts, and one credentials fixture would hand both the same username
    and collide on the unique index.
    """
    return f"mod-{uuid.uuid4().hex[:10]}"


@pytest.fixture
def moderator_credentials() -> tuple[str, str]:
    """A unique username and a generated password, not yet persisted."""
    return make_moderator_username(), make_moderator_password()


@pytest.fixture
def moderator(auth_service: AuthService, db_session: Session) -> tuple[Moderator, str]:
    """An active moderator, plus the plaintext password used to create it."""
    password = make_moderator_password()
    account = auth_service.create_moderator(username=make_moderator_username(), password=password)
    db_session.commit()
    return account, password


@pytest.fixture
def inactive_moderator(auth_service: AuthService, db_session: Session) -> tuple[Moderator, str]:
    """A deactivated moderator, plus its plaintext password."""
    password = make_moderator_password()
    account = auth_service.create_moderator(
        username=make_moderator_username(), password=password, is_active=False
    )
    db_session.commit()
    return account, password


# ---------------------------------------------------------------------------
# Moderation fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def moderator_token(moderator: tuple[Moderator, str], test_settings: Settings) -> str:
    """A valid access token for the active moderator fixture."""
    from app.core.tokens import create_access_token

    account, _ = moderator
    return create_access_token(
        subject=account.id,
        secret_key=test_settings.jwt_secret_key.get_secret_value(),
        algorithm=test_settings.jwt_algorithm,
        expires_minutes=test_settings.jwt_access_token_expire_minutes,
    )


@pytest.fixture
def auth_headers(moderator_token: str) -> dict[str, str]:
    """The ``Authorization`` header an authenticated moderator would send."""
    return {"Authorization": f"Bearer {moderator_token}"}


@pytest.fixture
def serving_client(app: FastAPI, db_session: Session) -> Iterator[TestClient]:
    """A client that returns a 500 instead of re-raising the exception.

    ``TestClient`` re-raises server exceptions by default, which is useful for
    debugging but hides the response a real deployment would send. The
    error-hardening tests need to inspect that response.
    """
    from app.db.session import get_db

    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def submit_report(api_client: TestClient):
    """Factory: file a report through the public endpoint and return its response.

    Reports are created the way a reporter creates them rather than by
    inserting rows, so the moderation tests operate on exactly the shape the
    system really produces — including the initial timeline entry.
    """

    def _submit(
        category: str = "SECURITY",
        description: str = "A report used as fixture data for the moderation tests.",
        evidence_url: str | None = None,
    ) -> dict:
        body: dict[str, object] = {"category": category, "description": description}
        if evidence_url is not None:
            body["evidence_url"] = evidence_url

        response = api_client.post("/api/v1/reports", json=body)
        assert response.status_code == 201, response.text
        return response.json()

    return _submit


@pytest.fixture
def report_id_for(db_session: Session, test_settings: Settings):
    """Factory: resolve a case code to the report id behind it."""
    from sqlalchemy import select

    from app.core.case_codes import canonicalise_case_code, hash_case_code
    from app.models import Report

    def _resolve(case_code: str):
        canonical = canonicalise_case_code(case_code)
        assert canonical is not None
        digest = hash_case_code(canonical, test_settings.case_code_pepper.get_secret_value())
        return db_session.scalars(select(Report).where(Report.case_code_hash == digest)).one().id

    return _resolve
