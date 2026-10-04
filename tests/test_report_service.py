"""Service-level tests: collision retry, atomicity and configuration.

These reach past HTTP to the service itself, because the behaviour under test —
what happens when a case code collides, or when the second write of a
transaction fails — cannot be provoked through the API without controlling the
random generator.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import CaseCodeGenerationError
from app.models import CaseUpdate, Report, ReportCategory
from app.services.reports import MAX_CASE_CODE_ATTEMPTS, ReportService
from tests.conftest import TEST_CASE_CODE_PEPPER, TEST_JWT_SECRET_KEY

pytestmark = pytest.mark.integration

REPORTS_URL = "/api/v1/reports"
DESCRIPTION = "A supplier is invoicing for hours that were never worked on this project."


@pytest.fixture
def service(db_session: Session) -> ReportService:
    return ReportService(db_session, case_code_pepper=TEST_CASE_CODE_PEPPER)


def _fixed_generator(*codes: str) -> Iterator[str]:
    """Yield the given codes in order, then keep yielding the last one."""
    yield from codes
    while True:
        yield codes[-1]


# ---------------------------------------------------------------------------
# 27. Collision handling
# ---------------------------------------------------------------------------


def test_a_collision_is_retried_and_succeeds(
    service: ReportService, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A duplicate code is regenerated rather than surfaced to the reporter."""
    first = service.submit(category=ReportCategory.OTHER, description=DESCRIPTION)

    # Hand out the already-used code once, then a fresh one.
    codes = _fixed_generator(first.case_code, "WD-11111-22222-33333-44444")
    monkeypatch.setattr("app.services.reports.generate_case_code", lambda: next(codes))

    second = service.submit(category=ReportCategory.OTHER, description=DESCRIPTION)

    assert second.case_code == "WD-11111-22222-33333-44444"
    assert second.case_code != first.case_code
    assert db_session.scalar(select(func.count()).select_from(Report)) == 2


def test_retries_are_bounded_and_then_fail_cleanly(
    service: ReportService, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generator stuck on one value must terminate, not spin forever."""
    first = service.submit(category=ReportCategory.OTHER, description=DESCRIPTION)

    calls = {"n": 0}

    def always_the_same() -> str:
        calls["n"] += 1
        return first.case_code

    monkeypatch.setattr("app.services.reports.generate_case_code", always_the_same)

    with pytest.raises(CaseCodeGenerationError):
        service.submit(category=ReportCategory.OTHER, description=DESCRIPTION)

    assert calls["n"] == MAX_CASE_CODE_ATTEMPTS
    # The failed attempts left nothing behind.
    assert db_session.scalar(select(func.count()).select_from(Report)) == 1


def test_exhausted_retries_surface_as_a_clean_503(
    api_client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the API: a service-unavailable envelope, with nothing internal."""
    existing = api_client.post(
        REPORTS_URL, json={"category": "OTHER", "description": DESCRIPTION}
    ).json()

    monkeypatch.setattr("app.services.reports.generate_case_code", lambda: existing["case_code"])

    response = api_client.post(REPORTS_URL, json={"category": "OTHER", "description": DESCRIPTION})

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "CASE_CODE_GENERATION_FAILED",
            "message": "The report could not be filed. Please try again.",
        }
    }
    # No constraint name, no SQL, no traceback, no case code.
    assert "case_code_hash" not in response.text
    assert "psycopg" not in response.text
    assert existing["case_code"] not in response.text


def test_a_non_collision_integrity_error_is_not_retried(
    service: ReportService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrying a real defect with a new random code would only hide it."""
    calls = {"n": 0}
    real_generator = __import__(
        "app.core.case_codes", fromlist=["generate_case_code"]
    ).generate_case_code

    def counting_generator() -> str:
        calls["n"] += 1
        return real_generator()

    monkeypatch.setattr("app.services.reports.generate_case_code", counting_generator)

    # A null description violates a NOT NULL constraint, not the case-code index.
    with pytest.raises(IntegrityError):
        service.submit(category=ReportCategory.OTHER, description=None)  # type: ignore[arg-type]

    assert calls["n"] == 1, "a non-collision error must not be retried"


def test_collision_detection_distinguishes_the_constraint() -> None:
    """The retry decision is made on the constraint name, not on the error type."""

    class FakeDiagnostic:
        def __init__(self, name: str) -> None:
            self.constraint_name = name

    class FakeOrig(Exception):
        def __init__(self, name: str) -> None:
            self.diag = FakeDiagnostic(name)

    def integrity_error(constraint: str) -> IntegrityError:
        return IntegrityError("stmt", {}, FakeOrig(constraint))

    assert ReportService._is_case_code_collision(integrity_error("ix_reports_case_code_hash"))
    assert not ReportService._is_case_code_collision(integrity_error("ix_moderators_username"))
    assert not ReportService._is_case_code_collision(
        integrity_error("fk_case_updates_report_id_reports")
    )


# ---------------------------------------------------------------------------
# 28. Atomicity
# ---------------------------------------------------------------------------


def test_report_and_initial_update_commit_together(
    service: ReportService, db_session: Session
) -> None:
    result = service.submit(category=ReportCategory.TECHNICAL, description=DESCRIPTION)

    updates = db_session.scalars(
        select(CaseUpdate).where(CaseUpdate.report_id == result.report.id)
    ).all()

    assert len(updates) == 1


def test_a_failed_timeline_write_rolls_back_the_report(
    service: ReportService, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stated guarantee, tested by breaking the second write.

    If the timeline entry cannot be written, the report must not exist either —
    a report whose own creation is missing from its audit trail is a gap from
    the very first row.
    """
    before = db_session.scalar(select(func.count()).select_from(Report))

    def explode(**kwargs: object) -> None:
        raise RuntimeError("timeline write failed")

    monkeypatch.setattr(
        "app.repositories.reports.CaseUpdateRepository.create", staticmethod(explode)
    )

    with pytest.raises(RuntimeError, match="timeline write failed"):
        service.submit(category=ReportCategory.OTHER, description=DESCRIPTION)

    db_session.rollback()

    assert db_session.scalar(select(func.count()).select_from(Report)) == before
    assert db_session.scalar(select(func.count()).select_from(CaseUpdate)) == 0


# ---------------------------------------------------------------------------
# 12. The pepper is required configuration
# ---------------------------------------------------------------------------


def _settings_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "app_env": "test",
        "database_url": "postgresql+psycopg://x:y@localhost:5433/whistledrop_test",
        "case_code_pepper": TEST_CASE_CODE_PEPPER,
        "jwt_secret_key": TEST_JWT_SECRET_KEY,
        "_env_file": None,  # never read the developer's .env in this test
    }
    values.update(overrides)
    return values


def test_settings_load_with_a_valid_pepper() -> None:
    settings = Settings(**_settings_kwargs())  # type: ignore[arg-type]

    assert settings.case_code_pepper.get_secret_value() == TEST_CASE_CODE_PEPPER


def test_a_missing_pepper_stops_the_application(monkeypatch: pytest.MonkeyPatch) -> None:
    """No default, no fallback: a deployment without a key must not start."""
    monkeypatch.delenv("CASE_CODE_PEPPER", raising=False)
    kwargs = _settings_kwargs()
    kwargs.pop("case_code_pepper")

    with pytest.raises(ValidationError, match="case_code_pepper"):
        Settings(**kwargs)  # type: ignore[arg-type]


def test_the_example_placeholder_is_refused() -> None:
    """The value shipped in .env.example must never reach a real deployment."""
    with pytest.raises(ValidationError, match="placeholder"):
        Settings(**_settings_kwargs(case_code_pepper=Settings.PLACEHOLDER_PEPPER))  # type: ignore[arg-type]


def test_a_short_pepper_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(**_settings_kwargs(case_code_pepper="too-short"))  # type: ignore[arg-type]


def test_the_pepper_is_masked_everywhere_it_could_be_logged() -> None:
    """SecretStr, so an accidental log line or traceback cannot spill the key."""
    settings = Settings(**_settings_kwargs())  # type: ignore[arg-type]

    assert TEST_CASE_CODE_PEPPER not in repr(settings)
    assert TEST_CASE_CODE_PEPPER not in str(settings)
    assert TEST_CASE_CODE_PEPPER not in str(settings.model_dump())
    assert TEST_CASE_CODE_PEPPER not in settings.model_dump_json()


def test_the_pepper_is_not_exposed_through_the_api(client: TestClient) -> None:
    """Nothing the API serves — health, OpenAPI, docs — mentions the key."""
    for url in ("/api/v1/health", "/openapi.json", "/docs"):
        response = client.get(url)

        assert TEST_CASE_CODE_PEPPER not in response.text
        assert "CASE_CODE_PEPPER" not in response.text


def test_the_example_file_ships_the_placeholder_the_validator_rejects() -> None:
    """.env.example and the validator must not drift apart.

    If someone edits the example's placeholder without updating the constant,
    the guard silently stops guarding and a deployment could boot on a
    published value.
    """
    from tests.conftest import PROJECT_ROOT

    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    assert f"CASE_CODE_PEPPER={Settings.PLACEHOLDER_PEPPER}" in example


def test_the_real_env_file_is_not_tracked_by_git() -> None:
    """The one file holding a real key must never be committed."""
    import subprocess

    from tests.conftest import PROJECT_ROOT

    tracked = subprocess.run(  # noqa: S603
        ["git", "ls-files"],  # noqa: S607
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    assert ".env" not in tracked
    assert not [path for path in tracked if path.endswith("/.env")]
