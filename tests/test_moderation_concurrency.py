"""Atomicity and concurrency around status transitions.

The concurrency tests here cannot use the usual ``db_session`` fixture: that
fixture keeps everything inside one rolled-back transaction on one connection,
and two moderators racing each other is by definition two connections
committing for real. So these open their own sessions against the test engine,
commit genuinely, and clean up after themselves in a ``finally``.
"""

import threading
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from app.core.errors import InvalidStatusTransitionError, ReportNotFoundError
from app.models import CaseUpdate, Moderator, Report, ReportCategory, ReportStatus
from app.services.moderation import ModerationService
from tests.conftest import (
    TEST_PASSWORD_HASH_ROUNDS,
    make_moderator_password,
    make_moderator_username,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 37-38. Atomicity, on the rolled-back session
# ---------------------------------------------------------------------------


def test_the_status_change_and_its_audit_entry_commit_together(
    db_session: Session, moderator: tuple[Moderator, str]
) -> None:
    account, _ = moderator
    service = ModerationService(db_session)
    report = Report(
        case_code_hash=uuid.uuid4().hex * 2,
        category=ReportCategory.SECURITY,
        description="A report used to check that a transition is written atomically.",
    )
    db_session.add(report)
    db_session.commit()

    service.change_status(
        report.id, moderator=account, new_status=ReportStatus.UNDER_REVIEW, note="Picked up."
    )

    db_session.expire_all()
    stored = db_session.scalars(select(Report).where(Report.id == report.id)).one()
    entries = db_session.scalars(select(CaseUpdate).where(CaseUpdate.report_id == report.id)).all()

    assert stored.status is ReportStatus.UNDER_REVIEW
    assert len(entries) == 1
    assert entries[0].to_status is ReportStatus.UNDER_REVIEW


def test_a_failed_audit_write_leaves_the_status_unchanged(
    db_session: Session, moderator: tuple[Moderator, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guarantee, tested by breaking the second write.

    A report whose status moved without a matching audit entry would be a hole
    in the trail exactly where accountability matters, so neither may land
    alone.
    """
    account, _ = moderator
    service = ModerationService(db_session)
    report = Report(
        case_code_hash=uuid.uuid4().hex * 2,
        category=ReportCategory.SECURITY,
        description="A report used to check rollback when the audit write fails.",
    )
    db_session.add(report)
    db_session.commit()

    def explode(**kwargs: object) -> None:
        raise RuntimeError("audit write failed")

    monkeypatch.setattr(
        "app.repositories.reports.CaseUpdateRepository.create", staticmethod(explode)
    )

    with pytest.raises(RuntimeError, match="audit write failed"):
        service.change_status(report.id, moderator=account, new_status=ReportStatus.UNDER_REVIEW)

    db_session.expire_all()
    stored = db_session.scalars(select(Report).where(Report.id == report.id)).one()
    entries = db_session.scalars(select(CaseUpdate).where(CaseUpdate.report_id == report.id)).all()

    assert stored.status is ReportStatus.SUBMITTED, "the status moved without an audit entry"
    assert entries == []


def test_a_rejected_transition_changes_nothing(
    db_session: Session, moderator: tuple[Moderator, str]
) -> None:
    account, _ = moderator
    service = ModerationService(db_session)
    report = Report(
        case_code_hash=uuid.uuid4().hex * 2,
        category=ReportCategory.OTHER,
        description="A report used to check that a refused transition is inert.",
    )
    db_session.add(report)
    db_session.commit()

    with pytest.raises(InvalidStatusTransitionError):
        service.change_status(report.id, moderator=account, new_status=ReportStatus.RESOLVED)

    db_session.expire_all()
    assert (
        db_session.scalars(select(Report).where(Report.id == report.id)).one().status
        is ReportStatus.SUBMITTED
    )
    entries = db_session.scalars(select(CaseUpdate).where(CaseUpdate.report_id == report.id))
    assert entries.all() == []


def test_a_missing_report_raises_before_anything_is_written(
    db_session: Session, moderator: tuple[Moderator, str]
) -> None:
    account, _ = moderator
    service = ModerationService(db_session)

    with pytest.raises(ReportNotFoundError):
        service.change_status(uuid.uuid4(), moderator=account, new_status=ReportStatus.UNDER_REVIEW)


# ---------------------------------------------------------------------------
# 39. Two moderators at once, on real connections
# ---------------------------------------------------------------------------


@pytest.fixture
def committed_fixture(db_engine: Engine) -> Iterator[tuple[uuid.UUID, uuid.UUID]]:
    """A genuinely committed report and moderator, removed afterwards.

    Committed for real because the race below needs two connections to see the
    same row; the usual per-test rollback would hide it inside one transaction.
    Cleanup deletes the audit entries first — ``case_updates.report_id`` is
    ``ON DELETE RESTRICT``, which is the Phase 1 decision that a report with
    history cannot be quietly removed.
    """
    from app.services.auth import AuthService

    with Session(db_engine) as setup:
        moderator = AuthService(
            setup,
            jwt_secret_key="unused-for-account-creation-but-required",  # noqa: S106
            password_hash_rounds=TEST_PASSWORD_HASH_ROUNDS,
        ).create_moderator(username=make_moderator_username(), password=make_moderator_password())
        report = Report(
            case_code_hash=uuid.uuid4().hex * 2,
            category=ReportCategory.SECURITY,
            description="A report used to exercise two moderators acting at once.",
        )
        setup.add(report)
        setup.commit()
        report_id, moderator_id = report.id, moderator.id

    try:
        yield report_id, moderator_id
    finally:
        with Session(db_engine) as cleanup:
            cleanup.execute(delete(CaseUpdate).where(CaseUpdate.report_id == report_id))
            cleanup.execute(delete(Report).where(Report.id == report_id))
            cleanup.execute(delete(Moderator).where(Moderator.id == moderator_id))
            cleanup.commit()


def test_concurrent_identical_transitions_cannot_both_succeed(
    db_engine: Engine, committed_fixture: tuple[uuid.UUID, uuid.UUID]
) -> None:
    """Four moderators all claim the same report at the same instant.

    Without the row lock every thread would read ``SUBMITTED``, every one would
    judge its move legal, and the database would end up with four audit entries
    all claiming to have started from ``SUBMITTED`` — a history that never
    happened. With ``SELECT ... FOR UPDATE`` the threads serialise: the first
    commits, the rest re-read the status it wrote and are refused.
    """
    report_id, moderator_id = committed_fixture
    workers = 4
    barrier = threading.Barrier(workers)
    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt() -> None:
        with Session(db_engine) as session:
            moderator = session.get(Moderator, moderator_id)
            service = ModerationService(session)
            barrier.wait(timeout=10)  # start together
            try:
                service.change_status(
                    report_id, moderator=moderator, new_status=ReportStatus.UNDER_REVIEW
                )
                result = "succeeded"
            except InvalidStatusTransitionError:
                result = "rejected"
            except Exception as exc:  # pragma: no cover - surfaced in the assertion
                result = f"error:{type(exc).__name__}"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=attempt) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert outcomes.count("succeeded") == 1, f"expected exactly one winner, got {outcomes}"
    assert outcomes.count("rejected") == workers - 1, outcomes

    with Session(db_engine) as check:
        report = check.get(Report, report_id)
        entries = check.scalars(select(CaseUpdate).where(CaseUpdate.report_id == report_id)).all()

        assert report.status is ReportStatus.UNDER_REVIEW
        assert len(entries) == 1, "more than one transition was recorded"
        assert entries[0].from_status is ReportStatus.SUBMITTED


def test_concurrent_conflicting_closures_cannot_both_succeed(
    db_engine: Engine, committed_fixture: tuple[uuid.UUID, uuid.UUID]
) -> None:
    """One moderator resolves while another dismisses, simultaneously.

    Both moves are legal from ``UNDER_REVIEW``, so this is the case where
    contradictory state is most plausible: the report cannot be both resolved
    and dismissed. Exactly one must win, and the trail must show one closure.
    """
    report_id, moderator_id = committed_fixture

    with Session(db_engine) as setup:
        moderator = setup.get(Moderator, moderator_id)
        ModerationService(setup).change_status(
            report_id, moderator=moderator, new_status=ReportStatus.UNDER_REVIEW
        )

    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, str]] = []
    lock = threading.Lock()

    def attempt(target: ReportStatus) -> None:
        with Session(db_engine) as session:
            moderator = session.get(Moderator, moderator_id)
            service = ModerationService(session)
            barrier.wait(timeout=10)
            try:
                service.change_status(report_id, moderator=moderator, new_status=target)
                result = "succeeded"
            except InvalidStatusTransitionError:
                result = "rejected"
            except Exception as exc:  # pragma: no cover
                result = f"error:{type(exc).__name__}"
        with lock:
            outcomes.append((target.value, result))

    threads = [
        threading.Thread(target=attempt, args=(ReportStatus.RESOLVED,)),
        threading.Thread(target=attempt, args=(ReportStatus.DISMISSED,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    results = [result for _, result in outcomes]
    assert results.count("succeeded") == 1, f"expected one winner, got {outcomes}"
    assert results.count("rejected") == 1, outcomes

    with Session(db_engine) as check:
        report = check.get(Report, report_id)
        closures = check.scalars(
            select(CaseUpdate).where(
                CaseUpdate.report_id == report_id,
                CaseUpdate.from_status == ReportStatus.UNDER_REVIEW,
            )
        ).all()

        assert report.status in (ReportStatus.RESOLVED, ReportStatus.DISMISSED)
        assert len(closures) == 1, "the report was closed twice"
        # The stored status and the audit entry agree.
        assert closures[0].to_status is report.status


def test_the_transition_is_judged_against_the_locked_row(
    db_engine: Engine, committed_fixture: tuple[uuid.UUID, uuid.UUID]
) -> None:
    """A stale view of the status must not be what the decision rests on.

    One session reads the report as ``SUBMITTED``, another advances it, and
    only then does the first attempt its move. Because the service re-reads
    under the lock rather than trusting anything read earlier, the first is
    refused.
    """
    report_id, moderator_id = committed_fixture

    with Session(db_engine) as stale_reader, Session(db_engine) as actor:
        # A stale read: still SUBMITTED as far as this session knows.
        assert stale_reader.get(Report, report_id).status is ReportStatus.SUBMITTED

        ModerationService(actor).change_status(
            report_id,
            moderator=actor.get(Moderator, moderator_id),
            new_status=ReportStatus.UNDER_REVIEW,
        )

        # The stale session now tries the move it believed was available.
        with pytest.raises(InvalidStatusTransitionError):
            ModerationService(stale_reader).change_status(
                report_id,
                moderator=stale_reader.get(Moderator, moderator_id),
                new_status=ReportStatus.UNDER_REVIEW,
            )

    with Session(db_engine) as check:
        entries = check.scalars(select(CaseUpdate).where(CaseUpdate.report_id == report_id)).all()
        assert len(entries) == 1
