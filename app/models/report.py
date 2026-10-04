"""The report: the central record of the system."""

from typing import TYPE_CHECKING

from sqlalchemy import Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import (
    ReportCategory,
    ReportStatus,
    report_category_enum,
    report_status_enum,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    from app.models.case_update import CaseUpdate
    from app.models.triage import ReportTriage


class Report(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """An anonymously submitted report.

    There is deliberately no ``reporter_id`` and no reporters table. Identity is
    never collected, so there is nothing to link to and nothing that could be
    compelled out of the database later.

    The only handle a reporter keeps is the case code, and even that is stored
    as a keyed hash: possession of the database does not let anyone recover a
    case code or brute-force one cheaply. Generating and hashing case codes is
    Phase 2 work; this model only reserves the column and its uniqueness.
    """

    __tablename__ = "reports"

    case_code_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        # unique + index together produce a single UNIQUE INDEX rather than a
        # unique constraint plus a redundant second index.
        unique=True,
        index=True,
        comment="Keyed hash of the reporter's case code. The plaintext is never stored.",
    )

    category: Mapped[ReportCategory] = mapped_column(
        report_category_enum,
        nullable=False,
        comment="Official category. Set by a human; never by AI triage.",
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Free-text body of the report. The substance of the record, so never null.",
    )

    evidence_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Optional link to supporting material held elsewhere.",
    )

    status: Mapped[ReportStatus] = mapped_column(
        report_status_enum,
        nullable=False,
        server_default=ReportStatus.SUBMITTED.value,
        comment="Official lifecycle state. Set by a human; never by AI triage.",
    )

    # --- Relationships -----------------------------------------------------

    triage: Mapped["ReportTriage | None"] = relationship(
        back_populates="report",
        # Triage is derived, advisory data with no meaning apart from its
        # report, so it is owned outright: purging a report purges its triage.
        cascade="all, delete-orphan",
        passive_deletes=True,
        single_parent=True,
        uselist=False,
        lazy="selectin",
    )

    updates: Mapped[list["CaseUpdate"]] = relationship(
        back_populates="report",
        order_by="CaseUpdate.created_at",
        # No delete cascade. The timeline is the audit trail; the foreign key is
        # ON DELETE RESTRICT, so the database refuses to delete a report that
        # still has history. Erasing a report must be a deliberate, explicit act.
        passive_deletes="all",
        lazy="selectin",
    )

    __table_args__ = (
        # The moderation queue (Phase 4) lists reports filtered by status and
        # ordered by arrival; one composite index serves both halves of that.
        Index("ix_reports_status_created_at", "status", "created_at"),
        {"comment": "Anonymously submitted reports. Contains no reporter identity."},
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Report id={self.id!s} status={self.status} category={self.category}>"
