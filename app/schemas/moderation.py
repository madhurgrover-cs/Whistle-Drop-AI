"""Request and response schemas for the moderation API.

Kept entirely separate from the reporter-facing schemas in ``cases.py``, and
deliberately not sharing a base class with them. The two audiences see
different things — a moderator sees internal notes, the report body and who
acted; an anonymous reporter sees none of that — and a shared parent is exactly
how a field added for one audience ends up served to the other.

Nothing here is built from an ORM object by ``from_attributes``. Every field is
listed and assigned explicitly, so a column added to ``reports`` or
``case_updates`` later cannot join a response on its own.
"""

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import ReportCategory, ReportStatus, TriagePriority, TriageStatus

# --- Limits ----------------------------------------------------------------
#
# DESCRIPTION_PREVIEW_LENGTH = 240
#   The queue is a scanning view: a moderator reads it to choose what to open.
#   Roughly forty words is enough to recognise a report and decide, and it
#   keeps a hundred-row page from carrying a hundred full accounts — both
#   wasteful and a wider spread of sensitive text than the view needs.
#
# DEFAULT_PAGE_SIZE = 20 / MAX_PAGE_SIZE = 100
#   Twenty fills a screen without scrolling past what a person will actually
#   read. The ceiling exists so that no request can ask for the whole table;
#   an integration pulling everything must page for it, which bounds both the
#   query cost and the size of any single response that could be intercepted.
#
# NOTE_MAX_LENGTH = 2000
#   A moderator note explains a decision; it is not the case file.
DESCRIPTION_PREVIEW_LENGTH = 240
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
NOTE_MAX_LENGTH = 2_000


def build_preview(description: str, limit: int = DESCRIPTION_PREVIEW_LENGTH) -> str:
    """Shorten a report body for the queue, marking that it was shortened."""
    if len(description) <= limit:
        return description
    return description[:limit].rstrip() + "…"


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


class ModeratorCaseUpdate(BaseModel):
    """One timeline entry as a moderator sees it.

    Two things appear here that the reporter's view has no equivalent of:
    ``visible_to_reporter``, so a moderator can tell at a glance which of their
    notes the reporter can read, and the acting moderator, so the trail is
    attributable.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "b2c3d4e5-6f7a-4b8c-9d0e-1f2a3b4c5d6e",
                    "from_status": "SUBMITTED",
                    "to_status": "UNDER_REVIEW",
                    "note": "Confirmed the affected service. Escalating to the platform team.",
                    "visible_to_reporter": False,
                    "moderator_id": "7c9d1b2a-3e4f-456a-8b9c-1d2e3f4a5b6c",
                    "moderator_username": "alex",
                    "created_at": "2026-09-25T09:02:14.907000Z",
                }
            ]
        }
    )

    id: UUID = Field(description="Id of this timeline entry.")
    from_status: ReportStatus | None = Field(
        default=None,
        description="Status before this entry. Null for the original submission.",
    )
    to_status: ReportStatus = Field(description="Status after this entry.")
    note: str | None = Field(default=None, description="The note attached to this entry.")
    visible_to_reporter: bool = Field(
        description="Whether the reporter can see this entry through case lookup.",
    )
    moderator_id: UUID | None = Field(
        default=None,
        description="Which moderator made this change. Null for system-generated entries.",
    )
    moderator_username: str | None = Field(
        default=None,
        description="Username of that moderator, for display. Null for system entries.",
    )
    created_at: datetime = Field(description="When this entry was recorded.")


# ---------------------------------------------------------------------------
# Triage (read-only; populated in a later phase)
# ---------------------------------------------------------------------------


class ModeratorTriageView(BaseModel):
    """AI triage for a report, reported exactly as stored.

    This phase writes nothing here and invents nothing. If no triage row exists
    the detail response carries ``null``; if one exists in state ``PENDING`` or
    ``FAILED``, that is what is shown, with the prediction fields null. A
    machine's guess is never dressed up as a decision.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "PENDING",
                    "suggested_category": None,
                    "category_confidence": None,
                    "suggested_priority": None,
                    "keywords": [],
                    "model_version": None,
                }
            ]
        }
    )

    status: TriageStatus = Field(description="State of the triage job, not of the report.")
    suggested_category: ReportCategory | None = Field(
        default=None,
        description="The model's suggestion. Advisory: it never changes the official category.",
    )
    category_confidence: Decimal | None = Field(
        default=None, description="Model confidence, 0.000-1.000."
    )
    suggested_priority: TriagePriority | None = Field(
        default=None, description="The model's urgency suggestion. Advisory."
    )
    keywords: list[Any] = Field(default_factory=list, description="Terms behind the prediction.")
    model_version: str | None = Field(
        default=None, description="Which model produced this, for reproducibility."
    )


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


class ModerationQueueItem(BaseModel):
    """One row of the moderator queue.

    Carries a shortened description rather than the whole body: enough to
    triage by, without shipping every full account in the system to render a
    list. The full text is one request away, in the detail view.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "a1b2c3d4-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
                    "category": "SECURITY",
                    "status": "SUBMITTED",
                    "description_preview": (
                        "Production database credentials are being shared in a public team chat…"
                    ),
                    "evidence_url": "https://example.com/shared/screenshot",
                    "created_at": "2026-09-24T16:10:31.482000Z",
                    "updated_at": "2026-09-24T16:10:31.482000Z",
                }
            ]
        }
    )

    id: UUID = Field(description="Report id. Use it with the detail and status endpoints.")
    category: ReportCategory = Field(description="Official category.")
    status: ReportStatus = Field(description="Current status.")
    description_preview: str = Field(
        description=(
            f"First {DESCRIPTION_PREVIEW_LENGTH} characters of the report, "
            "with an ellipsis if it was shortened."
        )
    )
    evidence_url: str | None = Field(
        default=None,
        description="Link the reporter supplied. Never fetched by this service — open with care.",
    )
    created_at: datetime = Field(description="When the report was filed.")
    updated_at: datetime = Field(description="When the report was last changed.")


class PageMeta(BaseModel):
    """Where this page sits in the filtered result set."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "page": 1,
                    "page_size": 20,
                    "total_items": 47,
                    "total_pages": 3,
                    "has_next": True,
                    "has_previous": False,
                }
            ]
        }
    )

    page: int = Field(description="This page number, 1-based.")
    page_size: int = Field(description="Rows requested per page.")
    total_items: int = Field(description="Total rows matching the filters, across all pages.")
    total_pages: int = Field(description="How many pages those rows fill.")
    has_next: bool = Field(description="Whether a later page exists.")
    has_previous: bool = Field(description="Whether an earlier page exists.")


class ModerationQueueResponse(BaseModel):
    """A page of the moderation queue, plus the filters that produced it."""

    items: list[ModerationQueueItem] = Field(
        default_factory=list, description="Reports on this page, newest first."
    )
    page: PageMeta = Field(description="Pagination metadata.")
    filters: "QueueFilters" = Field(description="The filters this page was built with.")


class QueueFilters(BaseModel):
    """Echo of the filters applied, so a client can render its own state."""

    status: ReportStatus | None = Field(default=None, description="Status filter, if any.")
    category: ReportCategory | None = Field(default=None, description="Category filter, if any.")


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


class ModerationReportDetail(BaseModel):
    """Everything a moderator may see about one report.

    Absent, and absent on purpose: ``case_code_hash``. A moderator has no use
    for it, and it is the one stored value that could be tested against a
    guessed case code offline.

    There is no reporter identity to omit — the schema has nowhere to record
    one.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "a1b2c3d4-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
                    "category": "SECURITY",
                    "status": "UNDER_REVIEW",
                    "description": "Production database credentials are in a public chat channel.",
                    "evidence_url": "https://example.com/shared/screenshot",
                    "created_at": "2026-09-24T16:10:31.482000Z",
                    "updated_at": "2026-09-25T09:02:14.907000Z",
                    "triage": None,
                    "timeline": [
                        {
                            "id": "b2c3d4e5-6f7a-4b8c-9d0e-1f2a3b4c5d6e",
                            "from_status": None,
                            "to_status": "SUBMITTED",
                            "note": "Report received. It is queued for review by a moderator.",
                            "visible_to_reporter": True,
                            "moderator_id": None,
                            "moderator_username": None,
                            "created_at": "2026-09-24T16:10:31.482000Z",
                        }
                    ],
                }
            ]
        }
    )

    id: UUID = Field(description="Report id.")
    category: ReportCategory = Field(description="Official category.")
    status: ReportStatus = Field(description="Current status.")
    description: str = Field(description="The full report, exactly as written.")
    evidence_url: str | None = Field(
        default=None,
        description="Link the reporter supplied. Never fetched by this service — open with care.",
    )
    created_at: datetime = Field(description="When the report was filed.")
    updated_at: datetime = Field(description="When the report was last changed.")
    triage: ModeratorTriageView | None = Field(
        default=None,
        description="AI triage if a row exists, otherwise null. Advisory only, never a decision.",
    )
    timeline: list[ModeratorCaseUpdate] = Field(
        default_factory=list,
        description="The complete history, oldest first, internal entries included.",
    )


# ---------------------------------------------------------------------------
# Status update
# ---------------------------------------------------------------------------


class StatusUpdateRequest(BaseModel):
    """A moderator's decision about a report.

    There is deliberately no ``moderator_id`` field, and unknown keys are
    rejected rather than ignored — so a request that tries to attribute the
    change to someone else fails loudly instead of being quietly stripped. The
    acting moderator comes from the bearer token, never from the body.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "status": "UNDER_REVIEW",
                    "note": "Confirmed the affected service. Escalating to the platform team.",
                    "visible_to_reporter": False,
                }
            ]
        },
    )

    status: ReportStatus = Field(description="The status to move the report to.")
    note: Annotated[
        str | None,
        Field(
            default=None,
            max_length=NOTE_MAX_LENGTH,
            description=(
                "Why. Whether the reporter ever reads it is decided by `visible_to_reporter` below."
            ),
        ),
    ] = None
    visible_to_reporter: bool = Field(
        default=False,
        description=(
            "Whether this entry — including the note — is shown to the reporter through "
            "case lookup. **Defaults to false**: publishing to the reporter is an "
            "explicit choice, so a note written as an internal aside can never reach "
            "them by omission."
        ),
    )


class StatusUpdateResponse(BaseModel):
    """Confirmation of a completed transition."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "report_id": "a1b2c3d4-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
                    "previous_status": "SUBMITTED",
                    "status": "UNDER_REVIEW",
                    "updated_at": "2026-09-25T09:02:14.907000Z",
                    "update": {
                        "id": "b2c3d4e5-6f7a-4b8c-9d0e-1f2a3b4c5d6e",
                        "from_status": "SUBMITTED",
                        "to_status": "UNDER_REVIEW",
                        "note": "Confirmed the affected service.",
                        "visible_to_reporter": False,
                        "moderator_id": "7c9d1b2a-3e4f-456a-8b9c-1d2e3f4a5b6c",
                        "moderator_username": "alex",
                        "created_at": "2026-09-25T09:02:14.907000Z",
                    },
                }
            ]
        }
    )

    report_id: UUID = Field(description="The report that changed.")
    previous_status: ReportStatus = Field(
        description="The status it actually held in the database when the change was applied."
    )
    status: ReportStatus = Field(description="The status it holds now.")
    updated_at: datetime = Field(description="When the change was applied.")
    update: ModeratorCaseUpdate = Field(
        description="The audit entry this change appended to the timeline."
    )


ModerationQueueResponse.model_rebuild()
