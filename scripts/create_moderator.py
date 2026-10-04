"""Create a moderator account.

    python -m scripts.create_moderator
    python -m scripts.create_moderator --username alex
    python -m scripts.create_moderator --username alex --inactive

This is the **only** way a moderator account comes into existence. There is no
registration endpoint and there will not be one: anyone who can run this
already has the database credentials, which is the privilege level that
provisioning a reviewer should require.

How the password is handled
---------------------------
There is deliberately no ``--password`` option. A password passed as an
argument is visible in ``ps``, in the shell's history file, and in any process
accounting the machine keeps — and it would then sit in that history long after
the account was forgotten. Instead:

* interactively, it is read with :func:`getpass.getpass`, which does not echo,
  and asked for twice so a typo cannot silently create an account nobody can
  log in to;
* non-interactively (CI, a provisioning script, a container entrypoint), pass
  ``--password-stdin`` and pipe it in — the same convention ``docker login``
  uses, which keeps it out of the argument list.

The password is never printed, never logged, and never written anywhere but the
bcrypt digest that goes into the database.
"""

import argparse
import getpass
import re
import sys
from collections.abc import Sequence

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.passwords import MIN_PASSWORD_LENGTH, PasswordPolicyError
from app.db.session import create_db_engine
from app.services.auth import AuthService, normalise_username

# Conservative: lower-case letters, digits and a few separators, starting with
# an alphanumeric. Narrow on purpose — a username is displayed in a moderation
# console and may end up in a log line, so it should hold no surprises.
USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
USERNAME_RULE = (
    "3-64 characters, lower case, starting with a letter or digit; "
    "letters, digits, dot, underscore and hyphen only"
)

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_DUPLICATE = 3
EXIT_CANCELLED = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.create_moderator",
        description="Create a WhistleDrop moderator account.",
        epilog=(
            "The password is never taken as a command-line argument, so it cannot "
            "reach your shell history or the process list."
        ),
    )
    parser.add_argument(
        "--username",
        help=f"Account username ({USERNAME_RULE}). Prompted for if omitted.",
    )
    parser.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read the password from standard input instead of prompting. For automation.",
    )
    parser.add_argument(
        "--inactive",
        action="store_true",
        help=(
            "Create the account deactivated. The default is active: an account is "
            "created in order to be used, and deactivation is a later decision."
        ),
    )
    return parser


def validate_username(username: str) -> str:
    """Normalise and check a username, or raise :class:`ValueError`."""
    normalised = normalise_username(username)
    if not USERNAME_PATTERN.match(normalised):
        raise ValueError(f"Username must be {USERNAME_RULE}.")
    return normalised


def prompt_for_username() -> str:
    while True:
        try:
            raw = input("Username: ")
        except EOFError:
            raise KeyboardInterrupt from None
        try:
            return validate_username(raw)
        except ValueError as exc:
            print(f"  {exc}", file=sys.stderr)


def prompt_for_password() -> str:
    """Read a password twice without echoing, and confirm the two match."""
    while True:
        try:
            first = getpass.getpass("Password: ")
            second = getpass.getpass("Confirm password: ")
        except EOFError:
            raise KeyboardInterrupt from None

        if first != second:
            # Says only that they differ. Showing where would show the password.
            print("  Passwords do not match. Try again.", file=sys.stderr)
            continue

        return first


def read_password_from_stdin() -> str:
    """Take the first line of stdin as the password, minus its newline."""
    password = sys.stdin.readline()
    return password.rstrip("\n").rstrip("\r")


def create(
    session: Session,
    settings: Settings,
    *,
    username: str,
    password: str,
    is_active: bool,
) -> str:
    """Create the account and commit. Returns the stored username."""
    service = AuthService(
        session,
        jwt_secret_key=settings.jwt_secret_key.get_secret_value(),
        jwt_algorithm=settings.jwt_algorithm,
        access_token_expire_minutes=settings.jwt_access_token_expire_minutes,
        password_hash_rounds=settings.password_hash_rounds,
    )

    # Checked up front so the common case gets a clear message rather than a
    # constraint violation. The unique index remains the real authority, and
    # the IntegrityError below is what catches a lost race.
    if service.username_exists(username):
        raise _DuplicateUsername(username)

    try:
        moderator = service.create_moderator(
            username=username, password=password, is_active=is_active
        )
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise _DuplicateUsername(username) from exc

    return moderator.username


class _DuplicateUsername(Exception):
    def __init__(self, username: str) -> None:
        self.username = username
        super().__init__(username)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.username is not None:
            username = validate_username(args.username)
        else:
            username = prompt_for_username()

        password = read_password_from_stdin() if args.password_stdin else prompt_for_password()
    except KeyboardInterrupt:
        print("\nCancelled. No account was created.", file=sys.stderr)
        return EXIT_CANCELLED
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    settings = get_settings()
    engine = create_db_engine(settings)

    try:
        with Session(engine) as session:
            stored = create(
                session,
                settings,
                username=username,
                password=password,
                is_active=not args.inactive,
            )
    except PasswordPolicyError as exc:
        # The policy message describes the rule, never the password.
        print(f"Error: {exc}", file=sys.stderr)
        print(f"  (minimum {MIN_PASSWORD_LENGTH} characters)", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except _DuplicateUsername as exc:
        print(f"Error: a moderator named {exc.username!r} already exists.", file=sys.stderr)
        return EXIT_DUPLICATE
    finally:
        engine.dispose()
        # Drop the plaintext from this frame as soon as it is no longer needed.
        # Python strings are immutable so this is not a secure wipe, only a way
        # to keep it out of any traceback raised further up.
        del password

    state = "inactive" if args.inactive else "active"
    print(f"Created moderator {stored!r} ({state}).")
    print("The password was not stored or displayed; only its bcrypt hash was saved.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
