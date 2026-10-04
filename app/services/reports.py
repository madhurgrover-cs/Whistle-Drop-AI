"""Report submission: case-code minting, hashing, persistence and retry.

This is where the phase's security-critical work happens, and it is deliberately
in a service rather than a router so that it can be exercised without an HTTP
client and reused later by anything else that needs to file a report.
"""

import logging

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.case_codes import generate_case_code, hash_case_code
from app.core.errors import CaseCodeGenerationError
from app.models import Report, ReportCategory, ReportStatus
from app.repositories.reports import CaseUpdateRepository, ReportRepository

logger = logging.getLogger(__name__)

#: Bounded retries when a freshly minted case code collides with a stored one.
#: Bounded, never a ``while True``: if the random source has failed, retrying
#: forever would pin a worker thread instead of surfacing the fault.
MAX_CASE_CODE_ATTEMPTS = 5

#: The unique index whose violation means "that code is taken". Any other
#: integrity error is a real bug and is re-raised untouched.
_CASE_CODE_INDEX = "ix_reports_case_code_hash"

#: Note attached to the automatic first timeline entry. Worded to describe
#: exactly what happened — a submission was received — and nothing more. It
#: must not imply that anyone has read the report yet.
INITIAL_UPDATE_NOTE = "Report received. It is queued for review by a moderator."


class SubmissionResult:
    """What the caller needs after a successful submission.

    Carries the plaintext case code alongside the persisted report because this
    is the only moment the plaintext exists. It is never stored, so it cannot be
    read back off ``report``.
    """

    __slots__ = ("case_code", "report")

    def __init__(self, case_code: str, report: Report) -> None:
        self.case_code = case_code
        self.report = report


class ReportService:
    """Files anonymous reports.

    Constructed per request with a session and the pepper. Taking the pepper as
    a constructor argument rather than reaching for global settings keeps the
    service directly testable and keeps the secret's path through the process
    explicit.
    """

    def __init__(self, session: Session, *, case_code_pepper: str) -> None:
        self._session = session
        self._pepper = case_code_pepper
        self._reports = ReportRepository(session)
        self._updates = CaseUpdateRepository(session)

    def submit(
        self,
        *,
        category: ReportCategory,
        description: str,
        evidence_url: str | None = None,
    ) -> SubmissionResult:
        """Create a report and its first timeline entry, atomically.

        The report row and its initial ``case_updates`` row are written in one
        transaction and committed together. A report that exists without the
        timeline entry that explains its own creation would be a gap in the
        audit trail from the very first row, so the two either both land or
        neither does.

        On a case-code collision the whole transaction is rolled back and the
        attempt is made again with a fresh code, up to
        :data:`MAX_CASE_CODE_ATTEMPTS` times.
        """
        for attempt in range(1, MAX_CASE_CODE_ATTEMPTS + 1):
            case_code = generate_case_code()
            case_code_hash = hash_case_code(case_code, self._pepper)

            try:
                report = self._reports.create(
                    case_code_hash=case_code_hash,
                    category=category,
                    description=description,
                    evidence_url=evidence_url,
                )

                self._updates.create(
                    report_id=report.id,
                    from_status=None,
                    to_status=ReportStatus.SUBMITTED,
                    note=INITIAL_UPDATE_NOTE,
                    # Published: the reporter should see that their submission
                    # landed the moment they look it up.
                    visible_to_reporter=True,
                    # No moderator has touched this. Claiming otherwise would
                    # misrepresent the case's history.
                    moderator_id=None,
                )

                self._session.commit()

            except IntegrityError as exc:
                self._session.rollback()

                if not self._is_case_code_collision(exc):
                    # A different constraint failed. That is a defect, not a
                    # collision, and must not be retried into oblivion.
                    raise

                # Never log the code itself, only that a collision occurred.
                logger.warning(
                    "Case-code collision on attempt %d of %d; regenerating.",
                    attempt,
                    MAX_CASE_CODE_ATTEMPTS,
                )
                continue

            return SubmissionResult(case_code=case_code, report=report)

        logger.error(
            "Exhausted %d case-code attempts. Check the entropy source and the pepper.",
            MAX_CASE_CODE_ATTEMPTS,
        )
        raise CaseCodeGenerationError

    @staticmethod
    def _is_case_code_collision(exc: IntegrityError) -> bool:
        """Whether ``exc`` is specifically a duplicate ``case_code_hash``.

        Matched on the constraint name reported by PostgreSQL. Anything else —
        a null violation, a foreign key, a check constraint — means the caller
        handed us bad data or a model is wrong, and retrying with a new random
        code would only hide it.
        """
        constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
        if constraint:
            return constraint == _CASE_CODE_INDEX
        # Fall back to the driver's message if the diagnostic is unavailable.
        return _CASE_CODE_INDEX in str(exc.orig)
