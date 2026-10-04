"""Data access for reports and their timeline entries.

Repositories know about the database and nothing else: no HTTP status codes, no
Pydantic schemas, no request objects. They take and return ORM objects, and they
never commit — transaction boundaries belong to the service that called them, so
that a single unit of work can span several repository calls.
"""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, selectinload

from app.models import CaseUpdate, Report, ReportCategory, ReportStatus


class ReportRepository:
    """Reads and writes ``reports``."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        case_code_hash: str,
        category: ReportCategory,
        description: str,
        evidence_url: str | None,
    ) -> Report:
        """Stage a new report and flush it so its generated columns are populated.

        Flushing rather than committing is what lets the caller add the initial
        timeline entry in the same transaction: the report has an id afterwards,
        but nothing is durable until the service commits.
        """
        report = Report(
            case_code_hash=case_code_hash,
            category=category,
            description=description,
            evidence_url=evidence_url,
        )
        self._session.add(report)
        self._session.flush()
        return report

    def get_by_case_code_hash(self, case_code_hash: str) -> Report | None:
        """Find a report by the digest of its case code.

        A single equality lookup on the unique index ``ix_reports_case_code_hash``.
        That index is the reason case codes are hashed deterministically rather
        than salted per row.
        """
        return self._session.scalar(select(Report).where(Report.case_code_hash == case_code_hash))

    def get_by_id(self, report_id: UUID) -> Report | None:
        """Fetch a report by primary key, for the moderator detail view."""
        return self._session.get(Report, report_id)

    def get_for_update(self, report_id: UUID) -> Report | None:
        """Fetch a report and hold a row lock on it until the transaction ends.

        ``SELECT ... FOR UPDATE``. This is what makes a status transition safe
        when two moderators act at once: the second request blocks here until
        the first commits, then reads the status the first actually wrote and
        re-checks the transition against it. Without the lock both would read
        ``SUBMITTED``, both would consider their move legal, and the later write
        would silently overwrite the earlier one while leaving both audit
        entries claiming to have started from ``SUBMITTED``.

        The lock is on one row and is held for the few milliseconds of a single
        transaction, so it costs nothing at this scale.
        """
        return self._session.scalar(select(Report).where(Report.id == report_id).with_for_update())

    @staticmethod
    def _apply_filters(
        statement: Select,
        *,
        status: ReportStatus | None,
        category: ReportCategory | None,
    ) -> Select:
        """Add the moderation queue's filters to a statement.

        Shared by the page query and the count query so the two can never
        disagree about what is being counted. Both filters are optional and
        compose: neither, either, or both.
        """
        if status is not None:
            statement = statement.where(Report.status == status)
        if category is not None:
            statement = statement.where(Report.category == category)
        return statement

    def list_for_moderation(
        self,
        *,
        status: ReportStatus | None = None,
        category: ReportCategory | None = None,
        limit: int,
        offset: int,
    ) -> Sequence[Report]:
        """One page of the moderation queue, newest first.

        Filtering, ordering and slicing all happen in SQL — never by loading
        every report and narrowing it in Python, which would read the whole
        table into memory to show twenty rows.

        The ordering is ``created_at DESC, id DESC``. The second key is not
        decoration: ``reports.created_at`` defaults to ``now()``, which in
        PostgreSQL is *transaction* start time, so reports filed in one
        transaction share a timestamp exactly and would otherwise come back in
        an arbitrary order that could differ between two requests for the same
        page.
        """
        statement = self._apply_filters(select(Report), status=status, category=category)

        return self._session.scalars(
            statement.order_by(Report.created_at.desc(), Report.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()

    def count_for_moderation(
        self,
        *,
        status: ReportStatus | None = None,
        category: ReportCategory | None = None,
    ) -> int:
        """How many reports match the filters, for pagination metadata."""
        statement = self._apply_filters(
            select(func.count()).select_from(Report), status=status, category=category
        )
        return self._session.scalar(statement) or 0


class CaseUpdateRepository:
    """Reads and writes ``case_updates``."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        report_id: UUID,
        to_status: ReportStatus,
        from_status: ReportStatus | None = None,
        note: str | None = None,
        visible_to_reporter: bool = False,
        moderator_id: UUID | None = None,
    ) -> CaseUpdate:
        """Append one entry to a report's timeline.

        ``visible_to_reporter`` defaults to false here exactly as it does in the
        schema: publishing an update to the reporter is always an explicit act.
        """
        update = CaseUpdate(
            report_id=report_id,
            from_status=from_status,
            to_status=to_status,
            note=note,
            visible_to_reporter=visible_to_reporter,
            moderator_id=moderator_id,
        )
        self._session.add(update)
        return update

    def list_visible_for_report(self, report_id: UUID) -> Sequence[CaseUpdate]:
        """Return the reporter-visible timeline, oldest first.

        The ``visible_to_reporter`` filter is applied in the SQL rather than in
        Python. A hidden update is therefore never loaded into the process that
        renders the reporter's response, so no later refactor can accidentally
        serialise one.
        """
        return self._session.scalars(
            select(CaseUpdate)
            .where(
                CaseUpdate.report_id == report_id,
                CaseUpdate.visible_to_reporter.is_(True),
            )
            .order_by(CaseUpdate.created_at, CaseUpdate.id)
        ).all()

    def list_all_for_report(self, report_id: UUID) -> Sequence[CaseUpdate]:
        """The complete timeline, internal entries included, oldest first.

        The moderator-facing counterpart of
        :meth:`list_visible_for_report`. Deliberately a separate method rather
        than a ``visible_only=True`` flag on one: a boolean argument is easy to
        get backwards at a call site, and getting it backwards here means
        publishing internal notes to an anonymous reporter. Two methods make
        the call site say which audience it is serving.

        The moderator who wrote each entry is eager-loaded, so rendering a long
        timeline does not issue one query per row.
        """
        return self._session.scalars(
            select(CaseUpdate)
            .where(CaseUpdate.report_id == report_id)
            .options(selectinload(CaseUpdate.moderator))
            .order_by(CaseUpdate.created_at, CaseUpdate.id)
        ).all()
