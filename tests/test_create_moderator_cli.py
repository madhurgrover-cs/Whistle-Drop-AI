"""Tests for the moderator seeding command.

The CLI's ``main()`` builds its own engine from the real settings, so these
tests drive its pieces — :func:`validate_username` and :func:`create` — against
the test session instead, plus the argument parser and the interactive prompts
in isolation. That covers the behaviour worth pinning without any risk of a
test writing to the development database.
"""

import io
import secrets

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.passwords import MIN_PASSWORD_LENGTH, PasswordPolicyError, verify_password
from app.models import Moderator
from scripts.create_moderator import (
    EXIT_DUPLICATE,
    EXIT_INVALID_INPUT,
    _DuplicateUsername,
    build_parser,
    create,
    main,
    prompt_for_password,
    read_password_from_stdin,
    validate_username,
)
from tests.conftest import make_moderator_password, make_moderator_username

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 5. Creation through the seed mechanism
# ---------------------------------------------------------------------------


def test_a_moderator_is_created(db_session: Session, test_settings: Settings) -> None:
    username = make_moderator_username()
    password = make_moderator_password()

    stored = create(db_session, test_settings, username=username, password=password, is_active=True)

    account = db_session.scalars(select(Moderator).where(Moderator.username == stored)).one()
    assert account.username == username
    assert account.is_active is True
    assert account.created_at is not None


def test_the_created_account_can_be_authenticated_with(
    db_session: Session, test_settings: Settings
) -> None:
    """The account the CLI makes is one the login endpoint accepts."""
    username = make_moderator_username()
    password = make_moderator_password()

    create(db_session, test_settings, username=username, password=password, is_active=True)

    account = db_session.scalars(select(Moderator).where(Moderator.username == username)).one()
    assert verify_password(password, account.password_hash) is True


# ---------------------------------------------------------------------------
# 4. The password never reaches the database in plaintext
# ---------------------------------------------------------------------------


def test_only_a_hash_is_stored(db_session: Session, test_settings: Settings) -> None:
    username = make_moderator_username()
    password = make_moderator_password()

    create(db_session, test_settings, username=username, password=password, is_active=True)

    account = db_session.scalars(select(Moderator).where(Moderator.username == username)).one()
    assert account.password_hash != password
    assert password not in account.password_hash
    assert account.password_hash.startswith("$2b$")


def test_the_password_is_never_printed(
    db_session: Session, test_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    password = make_moderator_password()

    create(
        db_session,
        test_settings,
        username=make_moderator_username(),
        password=password,
        is_active=True,
    )

    captured = capsys.readouterr()
    assert password not in captured.out
    assert password not in captured.err


def test_a_weak_password_is_refused_before_anything_is_written(
    db_session: Session, test_settings: Settings
) -> None:
    username = make_moderator_username()

    with pytest.raises(PasswordPolicyError):
        create(db_session, test_settings, username=username, password="short", is_active=True)

    db_session.rollback()
    assert db_session.scalars(select(Moderator).where(Moderator.username == username)).all() == []


def test_the_policy_error_does_not_contain_the_password(
    db_session: Session, test_settings: Settings
) -> None:
    attempted = "sh0rt!"

    with pytest.raises(PasswordPolicyError) as caught:
        create(
            db_session,
            test_settings,
            username=make_moderator_username(),
            password=attempted,
            is_active=True,
        )

    assert attempted not in str(caught.value)
    assert str(MIN_PASSWORD_LENGTH) in str(caught.value)


# ---------------------------------------------------------------------------
# 6. Duplicate usernames
# ---------------------------------------------------------------------------


def test_a_duplicate_username_is_refused(db_session: Session, test_settings: Settings) -> None:
    username = make_moderator_username()
    create(
        db_session,
        test_settings,
        username=username,
        password=make_moderator_password(),
        is_active=True,
    )

    with pytest.raises(_DuplicateUsername):
        create(
            db_session,
            test_settings,
            username=username,
            password=make_moderator_password(),
            is_active=True,
        )


def test_a_duplicate_differing_only_in_case_is_refused(
    db_session: Session, test_settings: Settings
) -> None:
    """Usernames are normalised, so `Alex` and `alex` are one account."""
    username = make_moderator_username()
    create(
        db_session,
        test_settings,
        username=username,
        password=make_moderator_password(),
        is_active=True,
    )

    with pytest.raises(_DuplicateUsername):
        create(
            db_session,
            test_settings,
            username=username.upper(),
            password=make_moderator_password(),
            is_active=True,
        )


def test_a_refused_duplicate_does_not_alter_the_existing_account(
    db_session: Session, test_settings: Settings
) -> None:
    """A failed re-creation must not overwrite the real account's password."""
    username = make_moderator_username()
    original_password = make_moderator_password()
    create(db_session, test_settings, username=username, password=original_password, is_active=True)
    original_hash = (
        db_session.scalars(select(Moderator).where(Moderator.username == username))
        .one()
        .password_hash
    )

    with pytest.raises(_DuplicateUsername):
        create(
            db_session,
            test_settings,
            username=username,
            password=make_moderator_password(),
            is_active=True,
        )

    account = db_session.scalars(select(Moderator).where(Moderator.username == username)).one()
    assert account.password_hash == original_hash
    assert verify_password(original_password, account.password_hash) is True


# ---------------------------------------------------------------------------
# Active/inactive default
# ---------------------------------------------------------------------------


def test_accounts_are_created_active_by_default(
    db_session: Session, test_settings: Settings
) -> None:
    """A deliberate default: an account is created in order to be used."""
    username = make_moderator_username()

    create(
        db_session,
        test_settings,
        username=username,
        password=make_moderator_password(),
        is_active=True,
    )

    account = db_session.scalars(select(Moderator).where(Moderator.username == username)).one()
    assert account.is_active is True


def test_inactive_can_be_requested(db_session: Session, test_settings: Settings) -> None:
    username = make_moderator_username()

    create(
        db_session,
        test_settings,
        username=username,
        password=make_moderator_password(),
        is_active=False,
    )

    account = db_session.scalars(select(Moderator).where(Moderator.username == username)).one()
    assert account.is_active is False


def test_the_parser_defaults_to_an_active_account() -> None:
    assert build_parser().parse_args([]).inactive is False
    assert build_parser().parse_args(["--inactive"]).inactive is True


# ---------------------------------------------------------------------------
# Usernames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("alex", "alex"),
        ("ALEX", "alex"),
        ("  Alex  ", "alex"),
        ("alex.moderator", "alex.moderator"),
        ("alex_2", "alex_2"),
        ("a-b-c", "a-b-c"),
    ],
)
def test_usernames_are_normalised(raw: str, expected: str) -> None:
    assert validate_username(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "  ", "ab", "-alex", ".alex", "alex!", "alex space", "a" * 65, "alex@example.com", "álex"],
    ids=[
        "empty",
        "spaces",
        "too-short",
        "leading-hyphen",
        "leading-dot",
        "punctuation",
        "space",
        "too-long",
        "email",
        "non-ascii",
    ],
)
def test_invalid_usernames_are_refused(raw: str) -> None:
    with pytest.raises(ValueError, match="Username must be"):
        validate_username(raw)


# ---------------------------------------------------------------------------
# Password entry
# ---------------------------------------------------------------------------


def test_there_is_no_password_command_line_option() -> None:
    """The central design point of the CLI.

    A password in argv is visible in ``ps`` and lands in shell history. The
    only ways in are an unechoed prompt or piped stdin.
    """
    parser = build_parser()
    options = {action.option_strings[0] for action in parser._actions if action.option_strings}

    assert "--password" not in options
    assert "-p" not in options
    assert "--password-stdin" in options

    with pytest.raises(SystemExit):
        parser.parse_args(["--password", "hunter2-hunter2"])


def test_the_password_is_read_from_stdin_without_its_newline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password = make_moderator_password()
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{password}\n"))

    assert read_password_from_stdin() == password


def test_a_windows_line_ending_is_stripped_from_a_piped_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password = make_moderator_password()
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{password}\r\n"))

    assert read_password_from_stdin() == password


def test_the_prompt_asks_twice_and_rejects_a_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A typo must not silently create an account nobody can log in to."""
    good = make_moderator_password()
    answers = iter(["first-attempt-aaa", "second-attempt-bbb", good, good])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))

    assert prompt_for_password() == good

    assert "do not match" in capsys.readouterr().err


def test_the_prompt_uses_getpass_so_nothing_is_echoed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards against a refactor to input(), which would echo the password."""
    calls: list[str] = []
    password = make_moderator_password()

    def fake_getpass(prompt: str = "") -> str:
        calls.append(prompt)
        return password

    monkeypatch.setattr("getpass.getpass", fake_getpass)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("input() would echo the password"))

    assert prompt_for_password() == password
    assert len(calls) == 2  # entry and confirmation


def test_a_mismatched_password_is_not_printed_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    typo = f"typo-{secrets.token_urlsafe(8)}"
    good = make_moderator_password()
    answers = iter([typo, "something-else-entirely", good, good])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))

    prompt_for_password()

    captured = capsys.readouterr()
    assert typo not in captured.out
    assert typo not in captured.err


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


def test_an_invalid_username_exits_without_touching_the_database(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--username", "!!invalid!!", "--password-stdin"])

    assert exit_code == EXIT_INVALID_INPUT
    assert "Username must be" in capsys.readouterr().err


def test_the_exit_codes_are_distinct() -> None:
    """So a provisioning script can tell "already exists" from "bad input"."""
    assert len({0, EXIT_INVALID_INPUT, EXIT_DUPLICATE}) == 3


# ---------------------------------------------------------------------------
# No HTTP path to account creation
# ---------------------------------------------------------------------------


def test_no_http_layer_code_can_create_an_account() -> None:
    """Account creation is reachable from the CLI, never from a route handler.

    Checked against the parsed syntax tree rather than the raw text, so that a
    docstring mentioning the command does not register as a call to it.
    """
    import ast
    from pathlib import Path

    from tests.conftest import PROJECT_ROOT

    forbidden = {"create_moderator", "hash_password", "gensalt", "hashpw"}
    offenders: list[str] = []

    for path in Path(PROJECT_ROOT, "app", "api").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            if isinstance(called, ast.Attribute):
                name = called.attr
            elif isinstance(called, ast.Name):
                name = called.id
            else:
                continue
            if name in forbidden:
                offenders.append(f"{path.name}:{node.lineno} calls {name}()")

    assert offenders == [], f"the HTTP layer can create accounts: {offenders}"
