"""Integration tests for the data layer.

These run against the real PostgreSQL test container, never SQLite. Behaviour
being asserted here — native enums, JSONB, ``ON DELETE RESTRICT``, check
constraints — either does not exist in SQLite or behaves differently there, so
testing against a stand-in would prove nothing about production.

The schema under test is built by Alembic (see the ``db_engine`` fixture), so
these tests also serve as verification that the migrations are correct.
"""

from decimal import Decimal

import pytest
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    CaseUpdate,
    Moderator,
    Report,
    ReportCategory,
    ReportStatus,
    ReportTriage,
    TriagePriority,
    TriageStatus,
)
from tests.factories import (
    make_case_update,
    make_moderator,
    make_report,
    make_triage,
    unique_case_code_hash,
    unique_username,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 1-2. Connection and schema
# ---------------------------------------------------------------------------


def test_database_connection_works(db_engine: Engine) -> None:
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT 1")).scalar_one() == 1
        version = connection.execute(text("SHOW server_version")).scalar_one()
    assert version.startswith("16."), f"expected PostgreSQL 16, got {version}"


def test_alembic_created_every_table(db_engine: Engine) -> None:
    tables = set(inspect(db_engine).get_table_names())

    assert {"reports", "report_triage", "case_updates", "moderators"} <= tables


def test_database_is_at_head_revision(db_engine: Engine) -> None:
    """The stamped revision must be the newest migration on disk."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    from tests.conftest import PROJECT_ROOT

    script = ScriptDirectory.from_config(Config(str(PROJECT_ROOT / "alembic.ini")))
    expected = script.get_current_head()

    with db_engine.connect() as connection:
        result = connection.execute(text("SELECT version_num FROM alembic_version"))
        stamped = list(result.scalars())

    assert stamped == [expected]


def test_there_is_no_reporters_table(db_engine: Engine) -> None:
    """Anonymity is a schema property, not a convention.

    There is nowhere to record who filed a report, so there is nothing to leak
    and nothing to hand over.
    """
    tables = set(inspect(db_engine).get_table_names())

    assert "reporters" not in tables
    report_columns = {column["name"] for column in inspect(db_engine).get_columns("reports")}
    assert not {"reporter_id", "email", "name", "phone", "ip_address"} & report_columns


def test_every_enum_type_exists(db_engine: Engine) -> None:
    with db_engine.connect() as connection:
        enum_names = set(
            connection.execute(text("SELECT typname FROM pg_type WHERE typtype = 'e'")).scalars()
        )

    assert {
        "report_category",
        "report_status",
        "triage_priority",
        "triage_status",
    } <= enum_names


# ---------------------------------------------------------------------------
# 3. Reports
# ---------------------------------------------------------------------------


def test_report_can_be_inserted(db_session: Session) -> None:
    report = make_report()
    db_session.add(report)
    db_session.commit()

    stored = db_session.get(Report, report.id)
    assert stored is not None
    assert stored.id is not None
    assert stored.category is ReportCategory.SECURITY
    # Defaulted by the database, not by the caller.
    assert stored.status is ReportStatus.SUBMITTED
    assert stored.created_at is not None
    assert stored.updated_at is not None
    assert stored.evidence_url is None


@pytest.mark.parametrize("category", list(ReportCategory))
def test_every_official_category_is_accepted(db_session: Session, category: ReportCategory) -> None:
    report = make_report(category=category)
    db_session.add(report)
    db_session.commit()

    assert db_session.get(Report, report.id).category is category


def test_report_description_cannot_be_null(db_session: Session) -> None:
    db_session.add(make_report(description=None))

    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_invalid_category_is_rejected_by_the_enum(db_session: Session) -> None:
    """The enum is enforced by PostgreSQL, so raw SQL cannot slip past it."""
    with pytest.raises(DBAPIError):
        db_session.execute(
            text(
                "INSERT INTO reports (case_code_hash, category, description) "
                "VALUES (:h, 'NOT_A_CATEGORY', 'x')"
            ),
            {"h": unique_case_code_hash()},
        )
    db_session.rollback()


# ---------------------------------------------------------------------------
# 4. Triage
# ---------------------------------------------------------------------------


def test_triage_can_be_associated_with_a_report(db_session: Session) -> None:
    report = make_report()
    triage = make_triage(
        report,
        suggested_category=ReportCategory.CORRUPTION,
        category_confidence=Decimal("0.873"),
        suggested_priority=TriagePriority.HIGH,
        keywords=["procurement", "kickback"],
        model_version="tfidf-logreg-v1",
        status=TriageStatus.COMPLETED,
    )
    db_session.add_all([report, triage])
    db_session.commit()

    stored = db_session.get(Report, report.id)
    assert stored.triage is not None
    assert stored.triage.suggested_category is ReportCategory.CORRUPTION
    assert stored.triage.category_confidence == Decimal("0.873")
    assert stored.triage.keywords == ["procurement", "kickback"]
    assert stored.triage.report_id == report.id
    # Navigable in both directions.
    assert stored.triage.report.id == report.id


def test_triage_defaults_to_pending_with_no_prediction(db_session: Session) -> None:
    """A report is fully valid before inference has ever run."""
    report = make_report()
    db_session.add_all([report, make_triage(report)])
    db_session.commit()

    triage = db_session.get(Report, report.id).triage
    assert triage.status is TriageStatus.PENDING
    assert triage.suggested_category is None
    assert triage.category_confidence is None
    assert triage.keywords == []


def test_triage_never_changes_the_official_record(db_session: Session) -> None:
    """The AI's opinion may differ from the human decision, and must not win."""
    report = make_report(category=ReportCategory.OTHER)
    db_session.add_all(
        [
            report,
            make_triage(
                report,
                suggested_category=ReportCategory.SECURITY,
                suggested_priority=TriagePriority.CRITICAL,
                category_confidence=Decimal("0.990"),
                status=TriageStatus.COMPLETED,
            ),
        ]
    )
    db_session.commit()

    stored = db_session.get(Report, report.id)
    assert stored.category is ReportCategory.OTHER
    assert stored.status is ReportStatus.SUBMITTED
    assert stored.triage.suggested_category is ReportCategory.SECURITY


def test_a_report_cannot_have_two_triage_rows(db_session: Session) -> None:
    report = make_report()
    db_session.add_all([report, make_triage(report)])
    db_session.commit()

    db_session.add(ReportTriage(report_id=report.id))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


# ---------------------------------------------------------------------------
# 5. Case updates
# ---------------------------------------------------------------------------


def test_case_update_can_reference_a_report(db_session: Session) -> None:
    report = make_report()
    update = make_case_update(report, note="Received.", visible_to_reporter=True)
    db_session.add_all([report, update])
    db_session.commit()

    stored = db_session.get(Report, report.id)
    assert len(stored.updates) == 1
    assert stored.updates[0].to_status is ReportStatus.SUBMITTED
    assert stored.updates[0].from_status is None
    assert stored.updates[0].visible_to_reporter is True
    assert stored.updates[0].moderator_id is None
    assert stored.updates[0].report.id == report.id


def test_case_update_defaults_to_hidden_from_the_reporter(db_session: Session) -> None:
    """Disclosure must be an explicit choice, never the fallback."""
    report = make_report()
    db_session.add_all([report, make_case_update(report)])
    db_session.commit()

    assert db_session.get(Report, report.id).updates[0].visible_to_reporter is False


def test_case_updates_form_an_ordered_timeline(db_session: Session) -> None:
    report = make_report()
    db_session.add(report)
    db_session.flush()
    db_session.add_all(
        [
            make_case_update(report, to_status=ReportStatus.SUBMITTED),
            make_case_update(
                report,
                from_status=ReportStatus.SUBMITTED,
                to_status=ReportStatus.UNDER_REVIEW,
            ),
            make_case_update(
                report,
                from_status=ReportStatus.UNDER_REVIEW,
                to_status=ReportStatus.RESOLVED,
            ),
        ]
    )
    db_session.commit()
    db_session.expire_all()

    timeline = db_session.get(Report, report.id).updates
    assert [entry.to_status for entry in timeline] == [
        ReportStatus.SUBMITTED,
        ReportStatus.UNDER_REVIEW,
        ReportStatus.RESOLVED,
    ]


# ---------------------------------------------------------------------------
# 6-7. Moderators
# ---------------------------------------------------------------------------


def test_moderator_can_be_inserted(db_session: Session) -> None:
    moderator = make_moderator()
    db_session.add(moderator)
    db_session.commit()

    stored = db_session.get(Moderator, moderator.id)
    assert stored is not None
    assert stored.is_active is True
    assert stored.created_at is not None


def test_moderator_username_must_be_unique(db_session: Session) -> None:
    username = unique_username()
    db_session.add(make_moderator(username=username))
    db_session.commit()

    db_session.add(make_moderator(username=username))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_moderator_password_hash_is_required(db_session: Session) -> None:
    db_session.add(make_moderator(password_hash=None))

    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


# ---------------------------------------------------------------------------
# 8-9. Constraints
# ---------------------------------------------------------------------------


def test_case_code_hash_must_be_unique(db_session: Session) -> None:
    """Two reports sharing a case code would let one reporter read another's case."""
    case_code_hash = unique_case_code_hash()
    db_session.add(make_report(case_code_hash=case_code_hash))
    db_session.commit()

    db_session.add(make_report(case_code_hash=case_code_hash))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_case_code_hash_is_indexed(db_engine: Engine) -> None:
    """Lookup by case code is the reporter's only route to their report."""
    indexes = inspect(db_engine).get_indexes("reports")
    by_name = {index["name"]: index for index in indexes}

    assert "ix_reports_case_code_hash" in by_name
    assert by_name["ix_reports_case_code_hash"]["unique"] is True
    assert by_name["ix_reports_case_code_hash"]["column_names"] == ["case_code_hash"]


@pytest.mark.parametrize("confidence", [Decimal("-0.001"), Decimal("1.001"), Decimal("5")])
def test_invalid_confidence_is_rejected(db_session: Session, confidence: Decimal) -> None:
    report = make_report()
    db_session.add_all([report, make_triage(report, category_confidence=confidence)])

    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


@pytest.mark.parametrize("confidence", [Decimal("0"), Decimal("0.5"), Decimal("1")])
def test_confidence_bounds_are_inclusive(db_session: Session, confidence: Decimal) -> None:
    report = make_report()
    db_session.add_all([report, make_triage(report, category_confidence=confidence)])
    db_session.commit()

    assert db_session.get(Report, report.id).triage.category_confidence == confidence


def test_keywords_must_be_a_json_array(db_session: Session) -> None:
    report = make_report()
    db_session.add_all([report, make_triage(report, keywords={"not": "a list"})])

    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


# ---------------------------------------------------------------------------
# 10. Foreign keys and delete behaviour
# ---------------------------------------------------------------------------


def test_case_update_requires_an_existing_report(db_session: Session) -> None:
    import uuid

    db_session.add(CaseUpdate(report_id=uuid.uuid4(), to_status=ReportStatus.SUBMITTED))

    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_triage_requires_an_existing_report(db_session: Session) -> None:
    import uuid

    db_session.add(ReportTriage(report_id=uuid.uuid4()))

    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_case_update_can_be_attributed_to_a_moderator(db_session: Session) -> None:
    report = make_report()
    moderator = make_moderator()
    db_session.add_all([report, moderator])
    db_session.flush()
    db_session.add(
        make_case_update(
            report,
            from_status=ReportStatus.SUBMITTED,
            to_status=ReportStatus.UNDER_REVIEW,
            moderator=moderator,
        )
    )
    db_session.commit()

    entry = db_session.get(Report, report.id).updates[0]
    assert entry.moderator is not None
    assert entry.moderator.id == moderator.id
    assert moderator.case_updates[0].id == entry.id


def test_deleting_a_report_with_history_is_refused(db_session: Session) -> None:
    """The audit trail is not collateral damage of a DELETE."""
    report = make_report()
    db_session.add_all([report, make_case_update(report)])
    db_session.commit()

    db_session.delete(report)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_deleting_a_report_removes_its_triage(db_session: Session) -> None:
    """Triage is derived data owned by its report, so it goes with it."""
    report = make_report()
    db_session.add_all([report, make_triage(report)])
    db_session.commit()
    report_id = report.id

    db_session.delete(report)
    db_session.commit()

    remaining = db_session.execute(
        select(ReportTriage).where(ReportTriage.report_id == report_id)
    ).scalars()
    assert list(remaining) == []


def test_deleting_a_moderator_preserves_their_audit_entries(db_session: Session) -> None:
    """History survives the account; the actor simply becomes anonymous."""
    report = make_report()
    moderator = make_moderator()
    db_session.add_all([report, moderator])
    db_session.flush()
    update = make_case_update(report, moderator=moderator)
    db_session.add(update)
    db_session.commit()
    update_id = update.id

    db_session.delete(moderator)
    db_session.commit()
    db_session.expire_all()

    surviving = db_session.get(CaseUpdate, update_id)
    assert surviving is not None
    assert surviving.moderator_id is None
