"""Data access for moderator accounts.

Only the operations authentication needs. No password hashing, no token
minting, no HTTP: this layer reads and writes rows and nothing else. Creation
lives here rather than in a service because the seeding CLI is the only caller
and it needs nothing else from a service layer.
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Moderator


class ModeratorRepository:
    """Reads and writes ``moderators``."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_username(self, username: str) -> Moderator | None:
        """Find an account by its normalised username.

        Returns inactive accounts too. Whether a deactivated moderator may log
        in is an authorisation decision, and it belongs to the service — a
        repository that silently hid rows would make that rule invisible.
        """
        return self._session.scalar(select(Moderator).where(Moderator.username == username))

    def get_by_id(self, moderator_id: UUID) -> Moderator | None:
        """Find an account by primary key. Used to resolve a token's subject."""
        return self._session.get(Moderator, moderator_id)

    def exists_with_username(self, username: str) -> bool:
        """Whether a username is already taken.

        A convenience for the seeding CLI so it can refuse early with a clear
        message. It is not the real guarantee — the unique index is, and the
        CLI still handles the integrity error it can lose a race to.
        """
        return (
            self._session.scalar(
                select(Moderator.id).where(Moderator.username == username).limit(1)
            )
            is not None
        )

    def create(self, *, username: str, password_hash: str, is_active: bool = True) -> Moderator:
        """Stage a new moderator and flush so its generated id is available.

        Takes an already-hashed password. A repository that accepted a
        plaintext one would be a place where plaintext could reach the
        database by mistake.
        """
        moderator = Moderator(
            username=username,
            password_hash=password_hash,
            is_active=is_active,
        )
        self._session.add(moderator)
        self._session.flush()
        return moderator
