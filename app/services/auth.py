"""Moderator authentication: credential checking and token issuing.

Everything that decides *whether* someone is a moderator lives here. The router
above it only moves data; the repository below it only reads rows. That split
is what lets the rules below be tested without an HTTP client, and reused
unchanged by the Phase 4 endpoints.
"""

import logging
import uuid

from sqlalchemy.orm import Session

from app.core import passwords, tokens
from app.core.errors import AuthenticationError, ExpiredCredentialsError
from app.models import Moderator
from app.repositories.moderators import ModeratorRepository

logger = logging.getLogger(__name__)


def normalise_username(username: str) -> str:
    """Fold a submitted username to the form stored in the database.

    Usernames are case-insensitive and surrounding whitespace is ignored.
    Applied identically at creation and at login — if the two ever diverged,
    an account could be created that nobody could log in to. ``moderators``
    has a unique index on the stored form, so folding here is also what stops
    ``Alex`` and ``alex`` becoming two accounts.
    """
    return username.strip().lower()


class AuthService:
    """Verifies moderator credentials and issues access tokens."""

    def __init__(
        self,
        session: Session,
        *,
        jwt_secret_key: str,
        jwt_algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        password_hash_rounds: int = 12,
    ) -> None:
        self._moderators = ModeratorRepository(session)
        self._secret_key = jwt_secret_key
        self._algorithm = jwt_algorithm
        self._expire_minutes = access_token_expire_minutes
        self._hash_rounds = password_hash_rounds

    # --- Login -------------------------------------------------------------

    def authenticate(self, username: str, password: str) -> Moderator:
        """Return the moderator these credentials belong to.

        Raises :class:`~app.core.errors.AuthenticationError` — the same error,
        with the same message, for every failure:

        * no account with that username;
        * the account exists but the password is wrong;
        * the password is right but the account is deactivated.

        Two things keep those indistinguishable. The obvious one is the shared
        error. The subtler one is timing: returning early on an unknown
        username would skip the bcrypt verification and answer in microseconds
        instead of a quarter-second, and that difference alone is a reliable
        account-enumeration oracle over enough samples. So an unknown username
        still pays for one hash comparison, against a throwaway digest.
        """
        moderator = self._moderators.get_by_username(normalise_username(username))

        if moderator is None:
            passwords.dummy_verify(self._hash_rounds)
            # Never logs the attempted username: a failed login is often a
            # typo by a real moderator, and the log is a lower-trust place
            # than the database.
            logger.info("Failed login: no matching account.")
            raise AuthenticationError

        if not passwords.verify_password(password, moderator.password_hash):
            logger.info("Failed login: incorrect password.")
            raise AuthenticationError

        if not moderator.is_active:
            # Checked after the password so that a deactivated account costs
            # the same as an active one with a wrong password.
            logger.info("Failed login: account is deactivated.")
            raise AuthenticationError

        logger.info("Moderator authenticated.")
        return moderator

    def issue_access_token(self, moderator: Moderator) -> str:
        """Mint a signed access token naming ``moderator`` as its subject."""
        return tokens.create_access_token(
            subject=moderator.id,
            secret_key=self._secret_key,
            algorithm=self._algorithm,
            expires_minutes=self._expire_minutes,
        )

    @property
    def access_token_expires_in_seconds(self) -> int:
        """Token lifetime, for the login response."""
        return self._expire_minutes * 60

    # --- Token resolution --------------------------------------------------

    def resolve_moderator(self, token: str) -> Moderator:
        """Turn a bearer token back into the moderator it names.

        Raises :class:`~app.core.errors.ExpiredCredentialsError` if the token
        has simply expired, and :class:`~app.core.errors.AuthenticationError`
        for everything else.

        The account is re-read on every request rather than trusted from the
        token's claims. That is what makes deactivation take effect
        immediately: a moderator disabled a minute ago still holds a
        cryptographically valid token, and this is the check that stops it
        working. It costs one indexed primary-key lookup.
        """
        try:
            claims = tokens.decode_access_token(
                token, secret_key=self._secret_key, algorithm=self._algorithm
            )
        except tokens.ExpiredTokenError as exc:
            logger.info("Rejected an expired access token.")
            raise ExpiredCredentialsError from exc
        except tokens.TokenError as exc:
            # The library's reason stays in the log; the caller gets nothing.
            logger.info("Rejected an invalid access token: %s", exc)
            raise AuthenticationError from exc

        moderator = self._moderators.get_by_id(claims.subject)

        if moderator is None:
            # A correctly signed token for an account that no longer exists.
            logger.warning("Access token names a moderator that does not exist.")
            raise AuthenticationError

        if not moderator.is_active:
            logger.info("Access token names a deactivated moderator.")
            raise AuthenticationError

        return moderator

    # --- Account creation (seeding CLI only) -------------------------------

    def create_moderator(
        self, *, username: str, password: str, is_active: bool = True
    ) -> Moderator:
        """Create a moderator account from a plaintext password.

        Called by ``scripts/create_moderator.py`` and by test fixtures. There
        is deliberately **no HTTP route that reaches this**: moderator accounts
        are provisioned by an administrator with database access, never through
        a public endpoint.

        The password is validated and hashed here and is never held anywhere
        else — the repository below takes only the digest.
        """
        normalised = normalise_username(username)
        password_hash = passwords.hash_password(password, rounds=self._hash_rounds)

        return self._moderators.create(
            username=normalised,
            password_hash=password_hash,
            is_active=is_active,
        )

    def username_exists(self, username: str) -> bool:
        """Whether an account already uses this username."""
        return self._moderators.exists_with_username(normalise_username(username))

    def get_by_id(self, moderator_id: uuid.UUID) -> Moderator | None:
        """Fetch a moderator by primary key."""
        return self._moderators.get_by_id(moderator_id)
