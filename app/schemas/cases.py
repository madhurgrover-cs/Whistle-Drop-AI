"""Request and response schemas for case tracking.

The response here is the tightest boundary in the application. A case code is a
bearer credential: whoever holds it sees this payload, and there is no second
factor behind it. So the schema lists only what a reporter actually needs to
follow their own report, and every internal identifier, every moderator
attribution and every hidden note is absent by construction rather than by
filtering.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.case_codes import MAX_SUBMITTED_LENGTH
from app.models.enums import ReportCategory, ReportStatus


class CaseLookupRequest(BaseModel):
    """A case code, submitted in a request body rather than a URL.

    Sending it in the body is the whole reason this endpoint is a POST — see
    the router docstring for the full argument.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"case_code": "WD-4K7PQ-92MRT-XJ3HN-B8VZ6"}]},
    )

    case_code: str = Field(
        min_length=1,
        max_length=MAX_SUBMITTED_LENGTH,
        description=(
            "The code you were given when you filed the report. Case, spacing and "
            "hyphens do not matter."
        ),
    )


class CaseTimelineEntry(BaseModel):
    """One reporter-visible event in a case's history.

    Built by hand from a ``CaseUpdate`` rather than read off it: the ORM row
    also carries ``moderator_id``, ``from_status``, its own primary key and the
    ``visible_to_reporter`` flag, none of which belong in a reporter's view.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "SUBMITTED",
                    "note": "Report received. It is queued for review by a moderator.",
                    "occurred_at": "2026-09-24T16:10:31.482000Z",
                }
            ]
        }
    )

    status: ReportStatus = Field(description="The status the report moved to at this point.")
    note: str | None = Field(
        default=None,
        description="Explanation attached to this update, when one was published.",
    )
    occurred_at: datetime = Field(description="When this entry was recorded.")


class CaseLookupResponse(BaseModel):
    """Everything the holder of a case code is shown.

    Absent on purpose:

    * the report's id and ``case_code_hash`` — internal identifiers with no use
      to a reporter, and two more things that could tie a person to a record;
    * the description — the reporter wrote it and already has it; echoing it
      back would mean a stolen code yields the full text of the report as well
      as its progress;
    * any moderator name or id, and any update whose ``visible_to_reporter``
      flag is false;
    * anything from the AI triage table. Those are internal suggestions, and
      showing a machine's guessed priority to a reporter would misrepresent it
      as a decision.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "UNDER_REVIEW",
                    "category": "SECURITY",
                    "submitted_at": "2026-09-24T16:10:31.482000Z",
                    "last_updated_at": "2026-09-25T09:02:14.907000Z",
                    "timeline": [
                        {
                            "status": "SUBMITTED",
                            "note": "Report received. It is queued for review by a moderator.",
                            "occurred_at": "2026-09-24T16:10:31.482000Z",
                        },
                        {
                            "status": "UNDER_REVIEW",
                            "note": "A moderator has begun reviewing this report.",
                            "occurred_at": "2026-09-25T09:02:14.907000Z",
                        },
                    ],
                }
            ]
        }
    )

    status: ReportStatus = Field(description="Where the report stands right now.")
    category: ReportCategory = Field(
        description="The category the report is filed under. A moderator may have changed it.",
    )
    submitted_at: datetime = Field(description="When the report was filed.")
    last_updated_at: datetime = Field(description="When the report was last changed.")
    timeline: list[CaseTimelineEntry] = Field(
        default_factory=list,
        description=("Reporter-visible history, oldest first. Internal notes are never included."),
    )
