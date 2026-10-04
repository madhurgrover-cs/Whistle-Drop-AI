"""Case tracking: turning a case code into a reporter-safe view of a report."""

import logging

from sqlalchemy.orm import Session

from app.core.case_codes import canonicalise_case_code, hash_case_code
from app.core.errors import CaseNotFoundError
from app.repositories.reports import CaseUpdateRepository, ReportRepository
from app.schemas.cases import CaseLookupResponse, CaseTimelineEntry

logger = logging.getLogger(__name__)


class CaseService:
    """Looks up a report on behalf of the person who filed it."""

    def __init__(self, session: Session, *, case_code_pepper: str) -> None:
        self._session = session
        self._pepper = case_code_pepper
        self._reports = ReportRepository(session)
        self._updates = CaseUpdateRepository(session)

    def lookup(self, submitted_case_code: str) -> CaseLookupResponse:
        """Resolve a case code to the reporter's view of their report.

        Raises :class:`~app.core.errors.CaseNotFoundError` for a code that is
        malformed *and* for one that is well-formed but unknown. The two are
        deliberately indistinguishable — same status, same body, same error
        code — so that the endpoint cannot be used as an oracle that confirms
        when a guess has the right shape. Structurally invalid input is rejected
        before any query runs, so garbage costs a database round trip.
        """
        canonical = canonicalise_case_code(submitted_case_code)
        if canonical is None:
            # Logged without the submitted value: it may be a real code with a
            # typo, and a log line is exactly where it must not end up.
            logger.info("Case lookup rejected: malformed case code.")
            raise CaseNotFoundError

        report = self._reports.get_by_case_code_hash(hash_case_code(canonical, self._pepper))
        if report is None:
            logger.info("Case lookup found no matching report.")
            raise CaseNotFoundError

        timeline = self._updates.list_visible_for_report(report.id)

        # Every field is named explicitly. Nothing is spread from the ORM row,
        # so a column added to `reports` later cannot surface here by itself.
        return CaseLookupResponse(
            status=report.status,
            category=report.category,
            submitted_at=report.created_at,
            last_updated_at=report.updated_at,
            timeline=[
                CaseTimelineEntry(
                    status=entry.to_status,
                    note=entry.note,
                    occurred_at=entry.created_at,
                )
                for entry in timeline
            ],
        )
