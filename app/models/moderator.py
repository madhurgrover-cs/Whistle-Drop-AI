"""Moderator accounts.

This phase defines storage only. Password hashing, login, tokens and
authorisation are Phase 3; nothing here reads or verifies ``password_hash``.
"""

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:  # pragma: no cover
    from app.models.case_update import CaseUpdate


class Moderator(Base, UUIDPrimaryKeyMixin, CreatedAtMixin):
    """A staff account able to review reports.

    Moderators are the only identified parties in the system. Reporters are
    not, and there is no table for them.
    """

    __tablename__ = "moderators"

    username: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        # unique + index produce a single UNIQUE INDEX, which is also the
        # lookup path for login in Phase 3.
        unique=True,
        index=True,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Hash only. A plaintext password is never stored or logged.",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
        comment="Deactivated accounts are retained so their audit history stays attributed.",
    )

    # --- Relationships -----------------------------------------------------

    case_updates: Mapped[list["CaseUpdate"]] = relationship(
        back_populates="moderator",
        # No cascade: deleting a moderator must not delete audit entries. The
        # foreign key is ON DELETE SET NULL and the database applies it.
        passive_deletes=True,
    )

    __table_args__ = ({"comment": "Staff accounts. The only identified parties in the system."},)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Moderator id={self.id!s} username={self.username!r}>"
