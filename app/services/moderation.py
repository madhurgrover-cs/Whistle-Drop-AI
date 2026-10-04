"""The moderation workflow: the queue, report detail, and status transitions.

This is where the rules about *what a moderator may do* live. The router above
it decides nothing; the repository below it enforces nothing. In particular the
legal-transition map is here and only here, because a rule that also exists in
a client is a rule that can be bypassed by not using that client.
"""

import logging
from collections.abc import Mapping, Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.errors import InvalidStatusTransitionError, ReportNotFoundError
from app.models import CaseUpdate, Moderator, Report, ReportCategory, ReportStatus
from app.repositories.reports import CaseUpdateRepository, ReportRepository
from app.schemas.moderation import (
    MAX_PAGE_SIZE,
    ModerationQueueItem,
    ModerationQueueResponse,
    ModerationReportDetail,
    ModeratorCaseUpdate,
    ModeratorTriageView,
    PageMeta,
    QueueFilters,
    StatusUpdateResponse,
    build_preview,
)

logger = logging.getLogger(__name__)

#: The complete set of legal status moves.
#:
#: A report is received (``SUBMITTED``), picked up by a moderator
#: (``UNDER_REVIEW``), and then closed one way or the other. ``RESOLVED`` and
#: ``DISMISSED`` are terminal: they map to the empty set, so nothing reopens a
#: closed case.
#:
#: Written as data rather than as a chain of ``if`` statements so the whole
#: policy can be read at once — and so a test can assert that every pair of
#: statuses *not* listed here is refused, rather than enumerating the
#: forbidden moves by hand and missing one.
ALLOWED_TRANSITIONS: Mapping[ReportStatus, frozenset[ReportStatus]] = {
    ReportStatus.SUBMITTED: frozenset({ReportStatus.UNDER_REVIEW}),
    ReportStatus.UNDER_REVIEW: frozenset({ReportStatus.RESOLVED, ReportStatus.DISMISSED}),
    ReportStatus.RESOLVED: frozenset(),
    ReportStatus.DISMISSED: frozenset(),
}


def is_transition_allowed(current: ReportStatus, requested: ReportStatus) -> bool:
    """Whether ``current -> requested`` is a legal move.

    A same-status move is refused along with everything else, because no status
    lists itself as a target. That is deliberate: re-applying the current
    status is not a decision, and allowing it would append an audit entry
    recording a change that did not happen. A moderator who wants to add a note
    without changing anything needs a comment feature, not a fake transition.
    """
    return requested in ALLOWED_TRANSITIONS.get(current, frozenset())


def _describe_options(current: ReportStatus) -> str:
    allowed = sorted(status.value for status in ALLOWED_TRANSITIONS.get(current, frozenset()))
    if not allowed:
        return f"A report that is {current.value} is closed and cannot change status again."
    return f"From {current.value} the only allowed next status is: {', '.join(allowed)}."


class ModerationService:
    """Serves the moderator queue and applies status decisions."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._reports = ReportRepository(session)
        self._updates = CaseUpdateRepository(session)

    # --- Queue -------------------------------------------------------------

    def list_queue(
        self,
        *,
        status: ReportStatus | None = None,
        category: ReportCategory | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> ModerationQueueResponse:
        """One page of the queue, filtered and ordered in the database.

        Two queries: one for the page, one for the total. The alternative —
        fetching every match to count it — is the thing pagination exists to
        avoid.
        """
        page = max(page, 1)
        page_size = min(max(page_size, 1), MAX_PAGE_SIZE)

        total_items = self._reports.count_for_moderation(status=status, category=category)
        reports = self._reports.list_for_moderation(
            status=status,
            category=category,
            limit=page_size,
            offset=(page - 1) * page_size,
        )

        total_pages = (total_items + page_size - 1) // page_size

        return ModerationQueueResponse(
            items=[self._to_queue_item(report) for report in reports],
            page=PageMeta(
                page=page,
                page_size=page_size,
                total_items=total_items,
                total_pages=total_pages,
                has_next=page < total_pages,
                has_previous=page > 1 and total_items > 0,
            ),
            filters=QueueFilters(status=status, category=category),
        )

    # --- Detail ------------------------------------------------------------

    def get_report_detail(self, report_id: UUID) -> ModerationReportDetail:
        """The full moderator view of one report.

        Raises :class:`~app.core.errors.ReportNotFoundError` if there is no
        such report.
        """
        report = self._reports.get_by_id(report_id)
        if report is None:
            logger.info("Moderation detail requested for a report that does not exist.")
            raise ReportNotFoundError

        timeline = self._updates.list_all_for_report(report.id)

        return ModerationReportDetail(
            id=report.id,
            category=report.category,
            status=report.status,
            description=report.description,
            evidence_url=report.evidence_url,
            created_at=report.created_at,
            updated_at=report.updated_at,
            triage=self._to_triage_view(report),
            timeline=[self._to_case_update(entry) for entry in timeline],
        )

    # --- Status transitions ------------------------------------------------

    def change_status(
        self,
        report_id: UUID,
        *,
        moderator: Moderator,
        new_status: ReportStatus,
        note: str | None = None,
        visible_to_reporter: bool = False,
    ) -> StatusUpdateResponse:
        """Move a report to ``new_status`` and record it in the audit trail.

        **Concurrency.** The report row is locked with ``SELECT ... FOR UPDATE``
        before its status is read, and the transition is validated against the
        value read under that lock — not against anything the caller supplied
        or saw earlier. Two moderators acting at the same instant therefore
        serialise: the second blocks until the first commits, re-reads the
        status the first wrote, and is refused if its move is no longer legal
        from there. Both attempting ``SUBMITTED -> UNDER_REVIEW`` means exactly
        one succeeds and one gets a 409, rather than two audit entries both
        claiming to have started from ``SUBMITTED``.

        This is a single-row lock held for the length of one short transaction.
        Nothing distributed is needed: one PostgreSQL row is the authority on
        one report's status.

        **Atomicity.** The status change and its ``case_updates`` row are
        written in the same transaction and committed together. A report whose
        status moved without a corresponding audit entry would be a hole in the
        trail precisely where accountability matters most, so if either write
        fails the transaction is rolled back and neither lands.
        """
        report = self._reports.get_for_update(report_id)
        if report is None:
            logger.info("Status change requested for a report that does not exist.")
            raise ReportNotFoundError

        current_status = report.status

        if not is_transition_allowed(current_status, new_status):
            logger.info(
                "Rejected status transition %s -> %s.", current_status.value, new_status.value
            )
            raise InvalidStatusTransitionError(
                f"A report cannot move from {current_status.value} to {new_status.value}. "
                f"{_describe_options(current_status)}"
            )

        try:
            report.status = new_status

            update = self._updates.create(
                report_id=report.id,
                from_status=current_status,
                to_status=new_status,
                note=note,
                visible_to_reporter=visible_to_reporter,
                # The acting moderator comes from the verified bearer token and
                # from nowhere else. A moderator_id in the request body is
                # rejected by the schema before reaching this point, so one
                # moderator cannot sign a decision with another's name.
                moderator_id=moderator.id,
            )

            self._session.flush()
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        logger.info(
            "Report moved %s -> %s by an authenticated moderator.",
            current_status.value,
            new_status.value,
        )

        return StatusUpdateResponse(
            report_id=report.id,
            previous_status=current_status,
            status=new_status,
            updated_at=report.updated_at,
            update=self._to_case_update(update, moderator=moderator),
        )

    # --- Mapping -----------------------------------------------------------
    #
    # Written out field by field rather than with from_attributes. Verbose on
    # purpose: it is the boundary that decides what leaves the system, and a
    # new column should have to be added here consciously to appear in a
    # response.

    @staticmethod
    def _to_queue_item(report: Report) -> ModerationQueueItem:
        return ModerationQueueItem(
            id=report.id,
            category=report.category,
            status=report.status,
            description_preview=build_preview(report.description),
            evidence_url=report.evidence_url,
            created_at=report.created_at,
            updated_at=report.updated_at,
        )

    @staticmethod
    def _to_case_update(
        entry: CaseUpdate, *, moderator: Moderator | None = None
    ) -> ModeratorCaseUpdate:
        """Render one timeline entry.

        ``moderator`` is passed explicitly by :meth:`change_status`, which
        already holds the authenticated account; otherwise the eagerly loaded
        relationship is used. Neither path emits anything from the moderator
        row but the username.
        """
        actor = moderator if moderator is not None else entry.moderator

        return ModeratorCaseUpdate(
            id=entry.id,
            from_status=entry.from_status,
            to_status=entry.to_status,
            note=entry.note,
            visible_to_reporter=entry.visible_to_reporter,
            moderator_id=entry.moderator_id,
            moderator_username=actor.username if actor is not None else None,
            created_at=entry.created_at,
        )

    @staticmethod
    def _to_triage_view(report: Report) -> ModeratorTriageView | None:
        """Report triage exactly as stored, or ``None`` when there is no row.

        Nothing is fabricated here. Until a later phase populates this table,
        the honest answer is ``null``, and a row in state ``PENDING`` is shown
        as pending rather than dressed up as a prediction.
        """
        triage = report.triage
        if triage is None:
            return None

        return ModeratorTriageView(
            status=triage.status,
            suggested_category=triage.suggested_category,
            category_confidence=triage.category_confidence,
            suggested_priority=triage.suggested_priority,
            keywords=list(triage.keywords or []),
            model_version=triage.model_version,
        )


def forbidden_transitions() -> Sequence[tuple[ReportStatus, ReportStatus]]:
    """Every status pair the map refuses. Used by the tests.

    Derived from :data:`ALLOWED_TRANSITIONS` rather than written out by hand,
    so a future change to the workflow cannot leave a stale list of forbidden
    moves silently passing.
    """
    return [
        (current, requested)
        for current in ReportStatus
        for requested in ReportStatus
        if not is_transition_allowed(current, requested)
    ]
