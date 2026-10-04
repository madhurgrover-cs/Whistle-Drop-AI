"""Enumerated domain values.

Every enum here is materialised as a native PostgreSQL ``ENUM`` type so that
invalid values are rejected by the database itself, not merely by application
code. Member *values* are the strings stored in the database and are identical
to the member names, which keeps ``psql`` output, API payloads and Python code
reading the same way.

Adding a member later requires a migration (``ALTER TYPE ... ADD VALUE``);
that friction is deliberate for a records system whose history must stay
interpretable.
"""

from enum import StrEnum

from sqlalchemy import Enum as SAEnum


class ReportCategory(StrEnum):
    """Official category of a report.

    These are the five categories defined by the task specification. They are
    the *authoritative* classification: only a human moderator sets this value.
    The AI triage layer records its own opinion separately — see
    :class:`app.models.triage.ReportTriage`.
    """

    SECURITY = "SECURITY"
    HARASSMENT = "HARASSMENT"
    CORRUPTION = "CORRUPTION"
    TECHNICAL = "TECHNICAL"
    OTHER = "OTHER"


class ReportStatus(StrEnum):
    """Lifecycle state of a report.

    Legal transitions between these states are *not* modelled here; that is
    workflow logic and belongs to the moderation service layer.
    """

    SUBMITTED = "SUBMITTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class TriagePriority(StrEnum):
    """Urgency suggested by AI triage. Advisory only."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TriageStatus(StrEnum):
    """State of the AI triage job for a report.

    Distinct from :class:`ReportStatus`: this describes whether the *inference
    job* ran, never the standing of the report itself. A report is fully valid
    and fully actionable while its triage is ``PENDING`` or ``FAILED``.
    """

    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# SQL type instances
#
# Each PostgreSQL ENUM type is instantiated exactly once and shared by every
# column that uses it. That matters: ``report_status`` backs three columns
# across two tables, and a separate instance per column would make SQLAlchemy
# try to ``CREATE TYPE`` it more than once.
#
# The instances are deliberately *not* bound to ``Base.metadata``. Binding them
# makes Alembic render ``metadata=MetaData()`` into generated migrations, which
# is both invalid and misleading; migrations create the types explicitly
# instead.
# ---------------------------------------------------------------------------


def _pg_enum(python_enum: type[StrEnum], name: str) -> SAEnum:
    """Build a native PostgreSQL ENUM type from a Python enum."""
    return SAEnum(
        python_enum,
        name=name,
        native_enum=True,
        # Store member values, not member names. They are identical today; being
        # explicit means renaming a member in Python cannot silently change what
        # is written to disk.
        values_callable=lambda enum_cls: [member.value for member in enum_cls],
    )


report_category_enum = _pg_enum(ReportCategory, "report_category")
report_status_enum = _pg_enum(ReportStatus, "report_status")
triage_priority_enum = _pg_enum(TriagePriority, "triage_priority")
triage_status_enum = _pg_enum(TriageStatus, "triage_status")
