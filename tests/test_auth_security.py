"""Security properties of the authentication layer.

Separate from the behavioural tests because these assert on the *shape* of the
system rather than on what a request returns: that no secret is hardcoded, that
configuration comes from the environment, that nothing credential-bearing is
tracked by Git, and that no public endpoint acquired an authentication
requirement when authentication was added.
"""

import ast
import re
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import Moderator
from tests.conftest import (
    PROJECT_ROOT,
    TEST_CASE_CODE_PEPPER,
    TEST_JWT_SECRET_KEY,
    make_moderator_password,
)

pytestmark = pytest.mark.integration

APP_DIR = PROJECT_ROOT / "app"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _settings_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "app_env": "test",
        "database_url": "postgresql+psycopg://x:y@localhost:5433/whistledrop_test",
        "case_code_pepper": TEST_CASE_CODE_PEPPER,
        "jwt_secret_key": TEST_JWT_SECRET_KEY,
        "_env_file": None,  # never read the developer's .env
    }
    values.update(overrides)
    return values


# ---------------------------------------------------------------------------
# Secrets come from configuration, never from source
# ---------------------------------------------------------------------------


def test_the_jwt_secret_is_required_with_no_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fallback key would let anyone holding the source mint a valid token."""
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    kwargs = _settings_kwargs()
    kwargs.pop("jwt_secret_key")

    with pytest.raises(ValidationError, match="jwt_secret_key"):
        Settings(**kwargs)  # type: ignore[arg-type]


def test_the_example_placeholder_jwt_secret_is_refused() -> None:
    with pytest.raises(ValidationError, match="placeholder"):
        Settings(**_settings_kwargs(jwt_secret_key=Settings.PLACEHOLDER_JWT_SECRET))  # type: ignore[arg-type]


def test_a_short_jwt_secret_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(**_settings_kwargs(jwt_secret_key="too-short"))  # type: ignore[arg-type]


def test_the_example_file_ships_the_placeholder_the_validator_rejects() -> None:
    """`.env.example` and the guard must not drift apart."""
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    assert f"JWT_SECRET_KEY={Settings.PLACEHOLDER_JWT_SECRET}" in example


def test_the_jwt_secret_is_masked_everywhere_it_could_be_logged() -> None:
    settings = Settings(**_settings_kwargs())  # type: ignore[arg-type]

    assert TEST_JWT_SECRET_KEY not in repr(settings)
    assert TEST_JWT_SECRET_KEY not in str(settings)
    assert TEST_JWT_SECRET_KEY not in str(settings.model_dump())
    assert TEST_JWT_SECRET_KEY not in settings.model_dump_json()


def test_only_hmac_algorithms_are_configurable() -> None:
    """A misconfiguration must not be able to select `none` or an asymmetric alg.

    An asymmetric algorithm would let an attacker who can supply a public key
    sign their own tokens; `none` removes signing entirely.
    """
    for algorithm in ("HS256", "HS384", "HS512"):
        assert Settings(**_settings_kwargs(jwt_algorithm=algorithm)).jwt_algorithm  # type: ignore[arg-type]

    for rejected in ("none", "None", "RS256", "ES256", "HS128", ""):
        with pytest.raises(ValidationError):
            Settings(**_settings_kwargs(jwt_algorithm=rejected))  # type: ignore[arg-type]


def test_production_refuses_a_test_grade_work_factor() -> None:
    """A low bcrypt cost is a test convenience and must not reach a deployment."""
    with pytest.raises(ValidationError, match="at least 12 in production"):
        Settings(**_settings_kwargs(app_env="production", password_hash_rounds=4))  # type: ignore[arg-type]

    assert Settings(**_settings_kwargs(app_env="production", password_hash_rounds=12))  # type: ignore[arg-type]


def test_the_token_lifetime_is_bounded() -> None:
    """A token that never effectively expires is a permanent credential."""
    for rejected in (0, -1, 1441, 100_000):
        with pytest.raises(ValidationError):
            Settings(**_settings_kwargs(jwt_access_token_expire_minutes=rejected))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# No secret is written into the source tree
# ---------------------------------------------------------------------------


def _python_sources(*directories: Path) -> list[Path]:
    return [
        path
        for directory in directories
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def test_no_source_file_assigns_a_literal_secret() -> None:
    """Every key must arrive from configuration, not from an assignment.

    Parsed rather than grepped, so that prose in a docstring about secrets does
    not register, and an actual ``JWT_SECRET_KEY = "..."`` does.
    """
    secret_names = re.compile(
        r"(secret|password|pepper|token_key|signing_key|api_key)", re.IGNORECASE
    )
    allowed = {"PLACEHOLDER_JWT_SECRET", "PLACEHOLDER_PEPPER"}
    offenders: list[str] = []

    for path in _python_sources(APP_DIR, SCRIPTS_DIR):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            value = node.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            if len(value.value) < 8:
                continue  # an enum member or a short constant, not a key

            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if name and secret_names.search(name) and name not in allowed:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} {name}")

    assert offenders == [], f"literal secrets in source: {offenders}"


def test_no_source_file_contains_the_real_development_secrets() -> None:
    """Whatever is in the local .env must not also be committed in the tree."""
    from dotenv import dotenv_values

    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        pytest.skip("No local .env to cross-check.")

    real_secrets = [
        value
        for key, value in dotenv_values(env_file).items()
        if value
        and len(value) >= 16
        and any(marker in key for marker in ("SECRET", "PEPPER", "PASSWORD"))
    ]
    assert real_secrets, "expected the local .env to hold at least one secret"

    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _python_sources(APP_DIR, SCRIPTS_DIR, PROJECT_ROOT / "tests")
    )
    sources += (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    for secret in real_secrets:
        assert secret not in sources


def test_no_real_secret_is_tracked_by_git() -> None:
    """The one file holding real keys must never be committed."""
    tracked = subprocess.run(  # noqa: S603
        ["git", "ls-files"],  # noqa: S607
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    assert ".env" not in tracked
    assert not [path for path in tracked if path.endswith("/.env")]

    from dotenv import dotenv_values

    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return

    secrets_in_env = [
        value
        for key, value in dotenv_values(env_file).items()
        if value
        and len(value) >= 16
        and any(marker in key for marker in ("SECRET", "PEPPER", "PASSWORD"))
    ]

    for path in tracked:
        full = PROJECT_ROOT / path
        if not full.is_file():
            continue
        try:
            content = full.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for secret in secrets_in_env:
            assert secret not in content, f"{path} contains a real secret"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_every_stored_moderator_holds_a_bcrypt_hash(
    db_session: Session, moderator: tuple[Moderator, str], inactive_moderator: tuple[Moderator, str]
) -> None:
    accounts = db_session.scalars(select(Moderator)).all()

    assert accounts
    for account in accounts:
        assert account.password_hash.startswith("$2b$")
        assert len(account.password_hash) == 60


def test_the_moderators_table_has_no_plaintext_password_column(db_session: Session) -> None:
    from sqlalchemy import inspect

    columns = {c["name"] for c in inspect(db_session.get_bind()).get_columns("moderators")}

    assert "password" not in columns
    assert "password_hash" in columns
    assert columns == {"id", "username", "password_hash", "is_active", "created_at"}


def test_no_schema_change_was_needed_for_authentication(db_session: Session) -> None:
    """Phase 1's moderators table was already sufficient.

    Authentication added no column and no migration — worth pinning, because a
    schema change here would be the moment an email address or a reset token
    quietly appeared.
    """
    from sqlalchemy import inspect

    inspector = inspect(db_session.get_bind())
    assert set(inspector.get_table_names()) == {
        "alembic_version",
        "reports",
        "report_triage",
        "case_updates",
        "moderators",
    }


# ---------------------------------------------------------------------------
# The public API stayed public
# ---------------------------------------------------------------------------


def test_no_public_endpoint_declares_the_bearer_dependency() -> None:
    """The reporting routers must not have acquired an auth dependency."""
    for name in ("reports.py", "cases.py", "health.py"):
        source = (APP_DIR / "api" / "v1" / "routers" / name).read_text(encoding="utf-8")

        assert "CurrentModerator" not in source
        assert "get_current_moderator" not in source


def test_anonymous_reporting_works_end_to_end_without_a_token(
    api_client: TestClient,
) -> None:
    """Submit and track, with no Authorization header anywhere."""
    assert "authorization" not in {k.lower() for k in api_client.headers}

    submission = api_client.post(
        "/api/v1/reports",
        json={
            "category": "HARASSMENT",
            "description": "Anonymous reporting must survive the addition of moderator auth.",
        },
    )
    assert submission.status_code == 201

    lookup = api_client.post(
        "/api/v1/cases/lookup", json={"case_code": submission.json()["case_code"]}
    )
    assert lookup.status_code == 200
    assert lookup.json()["status"] == "SUBMITTED"


def test_the_health_endpoint_is_still_public(client: TestClient) -> None:
    assert client.get("/api/v1/health").status_code == 200


def test_moderator_credentials_never_appear_in_a_public_response(
    api_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    """A reporter-facing response must not leak anything about staff."""
    account, password = moderator

    submission = api_client.post(
        "/api/v1/reports",
        json={"category": "OTHER", "description": "Checking for staff detail leakage."},
    )
    lookup = api_client.post(
        "/api/v1/cases/lookup", json={"case_code": submission.json()["case_code"]}
    )

    for response in (submission, lookup):
        assert account.username not in response.text
        assert account.password_hash not in response.text
        assert password not in response.text
        assert str(account.id) not in response.text


def test_the_login_endpoint_is_documented(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/auth/login"]["post"]

    assert operation["summary"]
    assert "200" in operation["responses"]
    assert "401" in operation["responses"]
    assert "422" in operation["responses"]
    assert "LoginRequest" in schema["components"]["schemas"]
    assert "TokenResponse" in schema["components"]["schemas"]


def test_the_documented_surface_is_exactly_what_is_built(client: TestClient) -> None:
    """The contract lists these routes and no others.

    Pinned as an exact set so that an endpoint cannot appear without a
    deliberate change here. Later phases add to this list as they add routes;
    nothing from a future phase may be advertised before it exists.
    """
    paths = set(client.get("/openapi.json").json()["paths"])

    assert paths == {
        "/api/v1/health",
        "/api/v1/reports",
        "/api/v1/cases/lookup",
        "/api/v1/auth/login",
        "/api/v1/moderation/reports",
        "/api/v1/moderation/reports/{report_id}",
        "/api/v1/moderation/reports/{report_id}/status",
    }


def test_the_documentation_does_not_promise_public_signup(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    prose = schema["paths"]["/api/v1/auth/login"]["post"]["description"].lower()

    assert "no sign-up" in prose or "no signup" in prose


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/auth/register",
        "/api/v1/auth/signup",
        "/api/v1/auth/moderators",
        "/api/v1/moderators",
        "/api/v1/auth/reset-password",
    ],
)
def test_no_account_management_endpoint_exists(api_client: TestClient, path: str) -> None:
    response = api_client.post(path, json={"username": "x", "password": make_moderator_password()})

    assert response.status_code in (404, 405), f"{path} responded {response.status_code}"
