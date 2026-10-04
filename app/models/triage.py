"""AI triage: the model's advisory opinion about a report.

Kept in its own table rather than as extra columns on ``reports`` for one
reason above all: the official record and the machine's guess must never be
confusable. A moderator reading ``reports.category`` is reading a human
decision. Anything in this table is a suggestion, and deleting every row here
would leave the official record completely intact.
"""

from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, Numeric, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import (
    ReportCategory,
    TriagePriority,
    TriageStatus,
    report_category_enum,
    triage_priority_enum,
    triage_status_enum,
)

if TYPE_CHECKING:  # pragma: no cover
    from app.models.report import Report


class ReportTriage(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One row per report holding the triage model's output.

    The prediction columns are nullable because a row exists from the moment a
    report is created, in state ``PENDING``, before any inference has run — and
    keeps existing in state ``FAILED`` if inference never succeeds.
    """

    __tablename__ = "report_triage"

    report_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        # Advisory data is owned by its report: purging the report purges this.
        ForeignKey("reports.id", ondelete="CASCADE"),
        nullable=False,
        # Unique rather than a plain index: this is what makes the relationship
        # one-to-one at the database level, not merely by ORM convention.
        unique=True,
        index=True,
    )

    suggested_category: Mapped[ReportCategory | None] = mapped_column(
        report_category_enum,
        nullable=True,
        comment="Model's guess. Advisory: it never overwrites reports.category.",
    )

    category_confidence: Mapped[Decimal | None] = mapped_column(
        # NUMERIC(4, 3) is exact, unlike float: a value that round-trips as
        # 1.0000000000000002 could slip past a 0..1 check written against a
        # floating-point column. Three decimals is ample for a probability.
        Numeric(4, 3),
        nullable=True,
        comment="Model confidence in suggested_category, 0.000-1.000.",
    )

    suggested_priority: Mapped[TriagePriority | None] = mapped_column(
        triage_priority_enum,
        nullable=True,
        comment="Model's urgency suggestion. Advisory.",
    )

    keywords: Mapped[list[Any]] = mapped_column(
        # JSONB rather than a PostgreSQL array: the extracted terms are likely
        # to grow richer than bare strings (term plus weight, say), and JSONB
        # absorbs that without a migration while still being queryable.
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        comment="Salient terms extracted from the description.",
    )

    model_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="Identifier of the model that produced this row, for reproducibility.",
    )

    status: Mapped[TriageStatus] = mapped_column(
        triage_status_enum,
        nullable=False,
        server_default=TriageStatus.PENDING.value,
        comment="State of the inference job. Unrelated to the report's own status.",
    )

    # --- Relationships -----------------------------------------------------

    report: Mapped["Report"] = relationship(back_populates="triage")

    __table_args__ = (
        # A confidence outside 0..1 is not a business rule, it is a corrupt
        # probability, so the database rejects it outright. NULL passes, which
        # is what PENDING and FAILED rows need.
        CheckConstraint(
            "category_confidence IS NULL"
            " OR (category_confidence >= 0 AND category_confidence <= 1)",
            name="category_confidence_range",
        ),
        # Guards against a scalar or object being written where the application
        # will later iterate a list.
        CheckConstraint("jsonb_typeof(keywords) = 'array'", name="keywords_is_array"),
        {"comment": "AI-generated triage suggestions. Advisory only; never authoritative."},
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ReportTriage report_id={self.report_id!s} status={self.status}>"
