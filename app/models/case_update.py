"""The case timeline: an append-only audit trail for a report."""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Text, text
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, UUIDPrimaryKeyMixin
from app.models.enums import ReportStatus, report_status_enum

if TYPE_CHECKING:  # pragma: no cover
    from app.models.moderator import Moderator
    from app.models.report import Report


class CaseUpdate(Base, UUIDPrimaryKeyMixin):
    """One entry in a report's history.

    Append-only by design: rows are written and never edited, which is why this
    model carries ``created_at`` but no ``updated_at``. An audit trail that can
    be rewritten is not an audit trail. Nothing in the schema *prevents* an
    UPDATE — that is enforced by the service layer and by database grants — but
    the shape of the table states the intent.

    Writing entries, and deciding which status transitions are legal, is
    moderation-workflow logic and belongs to a later phase.
    """

    __tablename__ = "case_updates"

    # Not the shared CreatedAtMixin. That defaults to now(), which in
    # PostgreSQL is *transaction* start time and is therefore identical for
    # every row written in one transaction — two entries appended together
    # would be indistinguishable in order, and a timeline that cannot be
    # ordered is not a timeline. clock_timestamp() reads the wall clock at each
    # statement, so entries always sort the way they were written.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("clock_timestamp()"),
        sort_order=100,
    )

    report_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        # RESTRICT, not CASCADE: a report with history cannot be quietly
        # deleted out from under its audit trail.
        ForeignKey("reports.id", ondelete="RESTRICT"),
        nullable=False,
    )

    from_status: Mapped[ReportStatus | None] = mapped_column(
        report_status_enum,
        nullable=True,
        comment="Status before this entry. Null for the initial submission entry.",
    )

    to_status: Mapped[ReportStatus] = mapped_column(
        report_status_enum,
        nullable=False,
        comment="Status after this entry. Always present.",
    )

    note: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Optional moderator note explaining the change.",
    )

    visible_to_reporter: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
        # Defaults to hidden: an internal note must never reach the reporter by
        # accident. Disclosure is an explicit choice, not the fallback.
        comment="Whether this entry is exposed through case-code lookup.",
    )

    moderator_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        # SET NULL: removing a moderator account must not destroy the history
        # they created. The entry survives with an anonymous actor.
        ForeignKey("moderators.id", ondelete="SET NULL"),
        nullable=True,
        comment="Actor, when there was one. Null for system-generated entries.",
    )

    # --- Relationships -----------------------------------------------------

    report: Mapped["Report"] = relationship(back_populates="updates")
    moderator: Mapped["Moderator | None"] = relationship(back_populates="case_updates")

    __table_args__ = (
        # Case-code lookup reads one report's timeline in chronological order.
        Index("ix_case_updates_report_id_created_at", "report_id", "created_at"),
        Index("ix_case_updates_moderator_id", "moderator_id"),
        {"comment": "Append-only audit trail of report status changes."},
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<CaseUpdate report_id={self.report_id!s} to_status={self.to_status}>"
