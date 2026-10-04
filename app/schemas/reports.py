"""Request and response schemas for anonymous report submission.

The request schema is the system's outermost boundary and is deliberately
*closed*: ``extra="forbid"`` means a client that sends ``email`` or ``name``
gets a 422 rather than having the field quietly ignored. That turns the
project's central privacy promise into something the API actively enforces
instead of merely omitting.

Response schemas are declared explicitly rather than derived from the ORM
models, so no column can become part of the public contract by being added to a
table later.
"""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from app.models.enums import ReportCategory, ReportStatus

# --- Field limits ----------------------------------------------------------
#
# Deliberately modest, and documented here so the reasoning survives.
#
# DESCRIPTION_MIN_LENGTH = 10
#   Enough to reject "empty", whitespace, "test" and a stray keypress, while
#   still admitting a terse but real report ("Payroll data is public"). The
#   requirement is only that blank input is refused; this is a small judgement
#   call above that floor and is trivial to lower.
#
# DESCRIPTION_MAX_LENGTH = 20000
#   About 3,500 words — comfortably more than a long, detailed account, while
#   bounding request size, database rows and the eventual feature extraction.
#   A reporter with more to say has an evidence URL.
#
# EVIDENCE_URL_MAX_LENGTH = 2048
#   The de facto ceiling browsers and proxies honour.
DESCRIPTION_MIN_LENGTH = 10
DESCRIPTION_MAX_LENGTH = 20_000
EVIDENCE_URL_MAX_LENGTH = 2_048

_SAVE_THIS_CODE = (
    "Save this case code somewhere safe. It is the only way to check your "
    "report, and it cannot be recovered or reissued if you lose it."
)


class ReportSubmissionRequest(BaseModel):
    """An anonymous report.

    There is no field here for a name, an email address, a phone number or an
    account, and none may be added: unknown keys are rejected outright.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "category": "SECURITY",
                    "description": (
                        "Production database credentials are being shared in a public "
                        "team chat channel, and the same password is reused for the "
                        "backup server."
                    ),
                    "evidence_url": "https://example.com/shared/screenshot",
                }
            ]
        },
    )

    category: ReportCategory = Field(
        description=(
            "Official category. One of SECURITY, HARASSMENT, CORRUPTION, TECHNICAL, OTHER."
        ),
    )

    description: Annotated[
        str,
        Field(
            min_length=DESCRIPTION_MIN_LENGTH,
            max_length=DESCRIPTION_MAX_LENGTH,
            description=(
                f"What happened, in your own words. Between {DESCRIPTION_MIN_LENGTH} and "
                f"{DESCRIPTION_MAX_LENGTH:,} characters. Anything identifying that you "
                "put in this text is stored as written, so leave out details you do "
                "not want recorded."
            ),
        ),
    ]

    evidence_url: Annotated[
        HttpUrl | None,
        Field(
            default=None,
            description=(
                "Optional http(s) link to supporting material held elsewhere. The link "
                "is stored as a string and is never opened, fetched or inspected by "
                "this service."
            ),
        ),
    ] = None

    @field_validator("description", mode="before")
    @classmethod
    def _normalise_description(cls, value: object) -> object:
        """Trim the edges and nothing else.

        Only two changes are made, both safe:

        * surrounding whitespace is stripped, so a body that is nothing but
          spaces or newlines fails the minimum-length check rather than being
          stored as blank;
        * ``\\r\\n`` becomes ``\\n``, so the same text submitted from Windows and
          from Linux is stored identically.

        Internal spacing, blank lines, indentation, capitalisation and
        punctuation are left exactly as written. A report is evidence; silently
        reflowing it would destroy meaning the reporter may have intended.
        """
        if not isinstance(value, str):
            return value

        if "\x00" in value:
            # PostgreSQL text cannot hold a NUL byte. Rejecting beats stripping:
            # a body containing one is not something a person typed.
            raise ValueError("Description must not contain null bytes.")

        return value.replace("\r\n", "\n").strip()

    @field_validator("evidence_url")
    @classmethod
    def _bound_url_length(cls, value: HttpUrl | None) -> HttpUrl | None:
        if value is not None and len(str(value)) > EVIDENCE_URL_MAX_LENGTH:
            raise ValueError(f"Evidence URL must be at most {EVIDENCE_URL_MAX_LENGTH} characters.")
        return value


class ReportSubmissionResponse(BaseModel):
    """The one and only time the plaintext case code is ever returned.

    The code is not stored anywhere in plaintext, so it cannot be looked up,
    re-sent or recovered — not by the reporter, not by a moderator, and not by
    anyone with full access to the database. This response is the single
    opportunity to save it.

    Note what is absent: no report id, no ``case_code_hash``, no internal UUID.
    The reporter needs none of them, and each would be one more identifier able
    to link them to the report.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "case_code": "WD-4K7PQ-92MRT-XJ3HN-B8VZ6",
                    "status": "SUBMITTED",
                    "submitted_at": "2026-09-24T16:10:31.482000Z",
                    "message": _SAVE_THIS_CODE,
                }
            ]
        }
    )

    case_code: str = Field(
        description="Your case code. Shown once, here, and never again.",
    )
    status: ReportStatus = Field(
        description="Status of the new report. Always SUBMITTED.",
    )
    submitted_at: datetime = Field(
        description=(
            "When the report was filed. Returned so you can tell your own reports "
            "apart if you file more than one; it is the same value the tracking "
            "endpoint shows."
        ),
    )
    message: str = Field(
        default=_SAVE_THIS_CODE,
        description="Plain-language reminder that the code cannot be recovered.",
    )
