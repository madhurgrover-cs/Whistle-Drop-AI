"""The moderation API.

Every route here declares ``CurrentModerator``, so every one requires a valid
bearer token for an active moderator. That dependency is the whole
authorisation story for this phase: there are no roles yet, and a moderator is
a moderator.

The handlers stay thin — parse, delegate, return. The transition rules,
locking, transaction boundaries and audit writes are all in
``app.services.moderation``, where they can be exercised without HTTP and
cannot be bypassed by a client that talks to the API directly.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Path, Query, Response, status

from app.api.deps import CurrentModerator, ModerationServiceDep
from app.core.errors import ErrorResponse
from app.models.enums import ReportCategory, ReportStatus
from app.schemas.moderation import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    ModerationQueueResponse,
    ModerationReportDetail,
    StatusUpdateRequest,
    StatusUpdateResponse,
)

router = APIRouter(prefix="/moderation", tags=["Moderation"])

_UNAUTHORISED = {
    "model": ErrorResponse,
    "description": (
        "No token, a malformed `Authorization` header, an invalid or expired token, "
        "or a deactivated account."
    ),
    "content": {
        "application/json": {
            "example": {
                "error": {"code": "AUTHENTICATION_FAILED", "message": "Authentication failed."}
            }
        }
    },
}

_NOT_FOUND = {
    "model": ErrorResponse,
    "description": "No report exists with that id.",
    "content": {
        "application/json": {
            "example": {
                "error": {"code": "REPORT_NOT_FOUND", "message": "No report exists with that id."}
            }
        }
    },
}

ReportId = Annotated[UUID, Path(description="The report's id, as returned by the queue.")]


@router.get(
    "/reports",
    response_model=ModerationQueueResponse,
    summary="List the moderation queue",
    description=f"""
Returns reports for review, newest first.

### Filtering

`status` and `category` are independent and compose: pass neither, either, or
both. Filtering happens in the database, not after loading rows.

Only these two are accepted as filters. There is deliberately no free-text
search parameter — report bodies are sensitive, and a query string is written
into access logs, proxy telemetry and browser history.

### Ordering

`created_at DESC`, then `id DESC` as a tiebreaker. The second key matters:
reports filed inside one transaction share a `created_at` exactly, and without
it the same page could come back in a different order twice.

### Pagination

`page` is 1-based; `page_size` defaults to {DEFAULT_PAGE_SIZE} and is capped at
{MAX_PAGE_SIZE}. The cap is not negotiable by the client — a larger value is
clamped, not honoured — so no single request can pull the whole table.

### Not included

The queue carries a shortened description, never the full body, and never
`case_code_hash`. Open a report for the rest.
""",
    responses={status.HTTP_401_UNAUTHORIZED: _UNAUTHORISED},
)
def list_queue(
    moderator: CurrentModerator,
    service: ModerationServiceDep,
    response: Response,
    status_filter: Annotated[
        ReportStatus | None,
        Query(alias="status", description="Only reports currently in this status."),
    ] = None,
    category: Annotated[
        ReportCategory | None,
        Query(description="Only reports filed under this category."),
    ] = None,
    page: Annotated[int, Query(ge=1, description="Page number, 1-based.")] = 1,
    page_size: Annotated[
        int,
        Query(
            ge=1,
            le=MAX_PAGE_SIZE,
            description=f"Rows per page. Default {DEFAULT_PAGE_SIZE}, maximum {MAX_PAGE_SIZE}.",
        ),
    ] = DEFAULT_PAGE_SIZE,
) -> ModerationQueueResponse:
    # Report contents must not sit in a shared cache or the browser's
    # back-forward cache.
    response.headers["Cache-Control"] = "no-store"

    return service.list_queue(
        status=status_filter,
        category=category,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/reports/{report_id}",
    response_model=ModerationReportDetail,
    summary="Read one report in full",
    description="""
The complete moderator view: the report as written, and its entire history.

### The timeline differs from the reporter's

A moderator sees **every** entry, including those marked internal, each tagged
with `visible_to_reporter` so it is obvious which ones the reporter can read.
The anonymous lookup endpoint filters the internal ones out in SQL and never
loads them at all.

### AI triage — advisory only

`triage` is `null` when no triage row exists, and otherwise reports exactly
what is stored, including state `PENDING` or `FAILED` with no prediction.
Nothing here is invented.

**`suggested_category` is a suggestion, never the report's category.** The
official category is the top-level `category` field, set by the reporter and
changeable only by a moderator. Nothing the model produces can alter it, the
report's status, or its timeline.

**`category_confidence` is a model probability, not a certainty** and not an
estimate of correctness. The model has not been calibration-tested.

**`keywords` are model-associated terms** — the words that most moved this
model's decision. They are not causal explanations and not reasons.

**`suggested_priority` comes from a deterministic heuristic, not a trained
model.** There are no priority labels to train one on.

> ⚠ **`whistledrop-category-v1` was trained on synthetic data.** No labelled
> corpus of real reports exists yet, so the training set was generated from
> templates. Its evaluation scores do not establish real-world accuracy and
> must not be read as though they did. Treat every suggestion as a prompt to
> look, never as a finding. Your judgement is authoritative.
> See `ml/reports/model-card.md`.

### Not included

`case_code_hash` is never returned. A moderator has no use for it, and it is
the one stored value that could be tested offline against a guessed case code.
There is no reporter identity to withhold: the schema has nowhere to record one.
""",
    responses={
        status.HTTP_401_UNAUTHORIZED: _UNAUTHORISED,
        status.HTTP_404_NOT_FOUND: _NOT_FOUND,
    },
)
def get_report(
    report_id: ReportId,
    moderator: CurrentModerator,
    service: ModerationServiceDep,
    response: Response,
) -> ModerationReportDetail:
    response.headers["Cache-Control"] = "no-store"
    return service.get_report_detail(report_id)


@router.patch(
    "/reports/{report_id}/status",
    response_model=StatusUpdateResponse,
    summary="Change a report's status",
    description="""
Moves a report through the workflow and appends an entry to its audit trail.

### The permitted workflow

```
SUBMITTED ──> UNDER_REVIEW ──> RESOLVED
                          └──> DISMISSED
```

`RESOLVED` and `DISMISSED` are final: a closed report does not reopen. Anything
else — skipping `UNDER_REVIEW`, reopening a closed case, or re-applying the
status a report already has — is refused with `409
INVALID_STATUS_TRANSITION`. The rule is enforced here, in the service, not in
any client.

### Who gets recorded

The acting moderator is taken from your bearer token. `moderator_id` is not a
field of this request and sending one is rejected, so a decision cannot be
signed with someone else's name.

### Notes and who reads them

`visible_to_reporter` defaults to **false**. A note written as an internal
aside therefore stays internal unless you deliberately publish it; nothing
reaches the reporter by omission.

### Atomicity and concurrency

The status change and its audit entry are committed together — never one
without the other. The report row is locked for the transaction and the
transition is checked against the status read under that lock, so two
moderators acting at once cannot both succeed: the second sees the first's
result and is refused if the move is no longer legal.
""",
    responses={
        status.HTTP_401_UNAUTHORIZED: _UNAUTHORISED,
        status.HTTP_404_NOT_FOUND: _NOT_FOUND,
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "The requested move is not legal from the report's current status.",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "INVALID_STATUS_TRANSITION",
                            "message": (
                                "A report cannot move from SUBMITTED to RESOLVED. "
                                "From SUBMITTED the only allowed next status is: UNDER_REVIEW."
                            ),
                        }
                    }
                }
            },
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "The body or the report id failed validation.",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "The request body failed validation.",
                            "details": [
                                {
                                    "field": "body.moderator_id",
                                    "message": "Extra inputs are not permitted",
                                }
                            ],
                        }
                    }
                }
            },
        },
    },
)
def change_report_status(
    report_id: ReportId,
    payload: StatusUpdateRequest,
    moderator: CurrentModerator,
    service: ModerationServiceDep,
    response: Response,
) -> StatusUpdateResponse:
    response.headers["Cache-Control"] = "no-store"

    return service.change_status(
        report_id,
        moderator=moderator,
        new_status=payload.status,
        note=payload.note,
        visible_to_reporter=payload.visible_to_reporter,
    )
