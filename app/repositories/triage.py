"""Data access for ``report_triage``.

Only what triage needs. As everywhere in this layer: no HTTP, no inference, no
transaction control — the service that calls this owns the commit.
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ReportTriage


class ReportTriageRepository:
    """Reads and writes the one triage row a report may have."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_for_report(self, report_id: UUID) -> ReportTriage | None:
        """The report's triage row, if one exists."""
        return self._session.scalar(select(ReportTriage).where(ReportTriage.report_id == report_id))

    def get_for_update(self, report_id: UUID) -> ReportTriage | None:
        """As :meth:`get_for_report`, holding a row lock until the transaction ends.

        Used by the upsert so that two inference attempts for one report — a
        retry racing the original, say — serialise instead of both deciding
        the row does not exist yet and both inserting.
        """
        return self._session.scalar(
            select(ReportTriage).where(ReportTriage.report_id == report_id).with_for_update()
        )

    def upsert(self, report_id: UUID, **values: object) -> ReportTriage:
        """Create the report's triage row, or update it in place.

        ``report_triage.report_id`` carries a unique index, so a report can
        only ever have one triage row; this is what keeps a re-run from
        violating it. Updating in place also means a ``FAILED`` row becomes a
        ``COMPLETED`` one on a successful retry rather than accumulating
        history — the audit trail for a report is ``case_updates``, and triage
        is a current-state record, not a log.

        Takes an already-validated suggestion as keyword values; it does not
        inspect or coerce them.
        """
        existing = self.get_for_update(report_id)

        if existing is None:
            triage = ReportTriage(report_id=report_id, **values)
            self._session.add(triage)
            self._session.flush()
            return triage

        for field, value in values.items():
            setattr(existing, field, value)
        self._session.flush()
        return existing
