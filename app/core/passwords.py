"""Moderator password hashing, verification and policy.

Why bcrypt directly, and not Passlib
------------------------------------
The usual recipe is ``passlib[bcrypt]``. It does not work here. Passlib 1.7.4
(the last release, from 2020) reads ``bcrypt.__about__.__version__`` during
backend detection, and modern ``bcrypt`` has removed that attribute — so the
combination raises on first use. Passlib has had no release since, so this is
not something a version bump fixes.

The alternative chosen is the smallest one available: the ``bcrypt`` package on
its own. It is maintained by the Python Cryptographic Authority, has no
transitive dependencies, and exposes exactly the two operations needed —
``hashpw`` and ``checkpw``. Nothing here implements any hashing itself;
``checkpw`` does its own constant-time comparison internally.

Argon2 (``argon2-cffi``) would also have been defensible and is the stronger
algorithm on paper. bcrypt was chosen because it is one dependency rather than
four, is universally understood, and is entirely adequate for a small set of
staff accounts.

The 72-byte trap
----------------
bcrypt hashes at most 72 bytes of input and — in the version pinned here —
**silently ignores the rest** rather than raising. Left alone that is a real
vulnerability: two distinct long passwords sharing a 72-byte prefix would both
authenticate. :data:`MAX_PASSWORD_BYTES` closes it by rejecting such passwords
outright at the policy boundary, so nothing longer ever reaches ``hashpw``.
"""

import logging
import unicodedata
from functools import lru_cache

import bcrypt

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_PASSWORD_BYTES",
    "MIN_PASSWORD_LENGTH",
    "PasswordPolicyError",
    "dummy_verify",
    "hash_password",
    "validate_password",
    "verify_password",
]

# --- Policy ----------------------------------------------------------------
#
# Deliberately length-based and nothing else, following NIST SP 800-63B: forced
# composition rules ("one capital, one symbol") measurably push people towards
# predictable substitutions such as "Password1!" without adding real entropy.
#
# MIN_PASSWORD_LENGTH = 12
#   These are staff accounts created by an administrator at a terminal, not
#   consumer signups, so a 12-character floor costs nothing in usability and
#   rules out the entire short-password space.
#
# MAX_PASSWORD_BYTES = 72
#   bcrypt's hard algorithmic limit, measured in UTF-8 *bytes* rather than
#   characters because a single emoji or accented letter can take four. See the
#   module docstring: the cap is a correctness requirement, not a preference.
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_BYTES = 72


class PasswordPolicyError(ValueError):
    """A password was refused before it was ever hashed.

    Deliberately a plain ``ValueError`` subclass rather than an
    :class:`~app.core.errors.AppError`: passwords are set through the seeding
    CLI, not through an HTTP endpoint, so there is no status code to carry.
    """


def validate_password(password: str) -> None:
    """Raise :class:`PasswordPolicyError` if ``password`` is unacceptable.

    Never logs, echoes or includes the password in the exception message —
    every message below describes the rule, never the value that broke it.
    """
    if not password:
        raise PasswordPolicyError("Password must not be empty.")

    if not password.strip():
        raise PasswordPolicyError("Password must not be only whitespace.")

    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")

    encoded = _encode(password)
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise PasswordPolicyError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes when UTF-8 encoded "
            f"(it is {len(encoded)}). Note that accented and non-Latin characters "
            "take more than one byte each."
        )


def hash_password(password: str, *, rounds: int = 12) -> str:
    """Validate and hash a password, returning the encoded bcrypt digest.

    The returned string carries its own algorithm, work factor and salt, so a
    future change of ``rounds`` does not invalidate existing hashes —
    :func:`verify_password` reads the cost from the stored value.
    """
    validate_password(password)

    digest = bcrypt.hashpw(_encode(password), bcrypt.gensalt(rounds=rounds))
    return digest.decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """Whether ``password`` matches ``password_hash``.

    Returns ``False`` rather than raising for a malformed or empty stored hash:
    a corrupt row must fail closed as an authentication failure, not surface a
    library exception to a caller that would then have to decide what it meant.
    """
    if not password or not password_hash:
        return False

    try:
        return bcrypt.checkpw(_encode(password), password_hash.encode("ascii"))
    except (ValueError, TypeError):
        # Malformed stored hash. Logged without either the hash or the
        # attempted password, so the operator learns a row is broken and
        # nothing sensitive reaches the log.
        logger.warning("Refusing a login against a malformed stored password hash.")
        return False


@lru_cache(maxsize=8)
def _dummy_hash(rounds: int) -> bytes:
    """A throwaway hash at a given work factor, computed once per process."""
    return bcrypt.hashpw(b"a-password-no-account-uses", bcrypt.gensalt(rounds=rounds))


def dummy_verify(rounds: int = 12) -> None:
    """Burn one verification's worth of time and discard the result.

    Called when no moderator matches the submitted username, so that a request
    for an unknown account costs about as much as one for a real account. The
    hash is cached per work factor so this performs exactly one ``checkpw`` —
    the same single operation a genuine verification performs.

    See ``AuthService.authenticate`` for why the timing has to match.
    """
    bcrypt.checkpw(b"wrong-password", _dummy_hash(rounds))


def _encode(password: str) -> bytes:
    """Normalise to NFC and encode as UTF-8.

    Without normalisation the same accented character typed on two keyboards —
    composed versus decomposed — produces different bytes and therefore a
    different hash, and a moderator could find a correct password rejected on a
    different machine.
    """
    return unicodedata.normalize("NFC", password).encode("utf-8")
