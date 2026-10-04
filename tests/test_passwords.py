"""Unit tests for password hashing, verification and policy.

No database and no HTTP: these exercise ``app/core/passwords.py`` directly.
Every password used here is generated at runtime, so this file contains no
string that is a working credential anywhere.
"""

import secrets

import pytest

from app.core.passwords import (
    MAX_PASSWORD_BYTES,
    MIN_PASSWORD_LENGTH,
    PasswordPolicyError,
    dummy_verify,
    hash_password,
    validate_password,
    verify_password,
)

# The lowest work factor bcrypt allows. Tests assert behaviour, not cost.
ROUNDS = 4


def a_valid_password() -> str:
    return f"test-{secrets.token_urlsafe(12)}"


# ---------------------------------------------------------------------------
# 1-3. Hashing and verification
# ---------------------------------------------------------------------------


def test_a_valid_password_hashes() -> None:
    digest = hash_password(a_valid_password(), rounds=ROUNDS)

    assert digest.startswith("$2b$")
    assert len(digest) == 60  # fits moderators.password_hash comfortably


def test_verification_succeeds_for_the_correct_password() -> None:
    password = a_valid_password()

    assert verify_password(password, hash_password(password, rounds=ROUNDS)) is True


def test_verification_fails_for_an_incorrect_password() -> None:
    digest = hash_password(a_valid_password(), rounds=ROUNDS)

    assert verify_password(a_valid_password(), digest) is False


def test_verification_is_case_sensitive() -> None:
    password = "Correct-Horse-Battery"
    digest = hash_password(password, rounds=ROUNDS)

    assert verify_password(password.lower(), digest) is False


def test_the_same_password_hashes_differently_every_time() -> None:
    """Per-hash salting: two moderators with one password get unlike rows."""
    password = a_valid_password()

    first = hash_password(password, rounds=ROUNDS)
    second = hash_password(password, rounds=ROUNDS)

    assert first != second
    assert verify_password(password, first)
    assert verify_password(password, second)


def test_the_hash_does_not_contain_the_password() -> None:
    password = a_valid_password()

    assert password not in hash_password(password, rounds=ROUNDS)


def test_a_hash_carries_its_own_work_factor() -> None:
    """So raising the cost later does not invalidate existing hashes."""
    cheap = hash_password(a_valid_password(), rounds=4)
    dearer = hash_password(a_valid_password(), rounds=6)

    assert cheap.startswith("$2b$04$")
    assert dearer.startswith("$2b$06$")


def test_a_hash_made_at_one_cost_still_verifies_at_another() -> None:
    password = a_valid_password()
    digest = hash_password(password, rounds=6)

    # verify_password reads the cost from the stored value, not from config.
    assert verify_password(password, digest) is True


# ---------------------------------------------------------------------------
# Failing closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stored",
    ["", "not-a-hash", "$2b$12$tooshort", "plaintext-password-here"],
    ids=["empty", "garbage", "truncated", "plaintext"],
)
def test_a_malformed_stored_hash_fails_closed(stored: str) -> None:
    """A corrupt row must deny access, not raise a library exception."""
    assert verify_password(a_valid_password(), stored) is False


def test_an_empty_password_never_verifies() -> None:
    assert verify_password("", hash_password(a_valid_password(), rounds=ROUNDS)) is False


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "password",
    ["", " ", "   ", "\t\n", "short", "elevenchar"],
    ids=["empty", "one-space", "spaces", "whitespace", "too-short", "one-under-minimum"],
)
def test_unacceptable_passwords_are_rejected(password: str) -> None:
    with pytest.raises(PasswordPolicyError):
        validate_password(password)


def test_a_password_at_the_minimum_length_is_accepted() -> None:
    validate_password("x" * MIN_PASSWORD_LENGTH)


def test_no_composition_rules_are_imposed() -> None:
    """Length is the policy. Forced character classes are not.

    NIST SP 800-63B advises against them: they push people towards predictable
    substitutions without adding real entropy.
    """
    validate_password("all lower case words no digits")
    validate_password("aaaaaaaaaaaaaaaaa")
    validate_password("123456789012345678")


def test_a_password_over_bcrypts_limit_is_rejected_not_truncated() -> None:
    """The important one.

    bcrypt hashes at most 72 bytes and the pinned version silently discards the
    rest. Were that allowed through, two different long passwords sharing a
    72-byte prefix would both authenticate. The policy refuses them instead.
    """
    base = "x" * MAX_PASSWORD_BYTES

    with pytest.raises(PasswordPolicyError, match="72 bytes"):
        validate_password(base + "a")

    with pytest.raises(PasswordPolicyError):
        hash_password(base + "-completely-different-tail", rounds=ROUNDS)


def test_the_truncation_hazard_is_real_and_is_what_the_limit_prevents() -> None:
    """Demonstrates the underlying bcrypt behaviour the policy guards against."""
    import bcrypt

    shared_prefix = b"y" * MAX_PASSWORD_BYTES
    digest = bcrypt.hashpw(shared_prefix + b"-first-tail", bcrypt.gensalt(rounds=ROUNDS))

    # A different password, accepted by raw bcrypt against the same hash.
    assert bcrypt.checkpw(shared_prefix + b"-SECOND-tail", digest) is True

    # Neither could ever reach bcrypt through this application.
    with pytest.raises(PasswordPolicyError):
        validate_password(shared_prefix.decode() + "-first-tail")


def test_the_byte_limit_is_measured_in_bytes_not_characters() -> None:
    """A 40-character accented password is 80 bytes and must be refused."""
    accented = "é" * 40

    assert len(accented) < MAX_PASSWORD_BYTES
    assert len(accented.encode("utf-8")) > MAX_PASSWORD_BYTES

    with pytest.raises(PasswordPolicyError, match="bytes"):
        validate_password(accented)


def test_policy_messages_never_echo_the_password() -> None:
    """An error message is a thing that gets logged and screenshotted."""
    secret = "s3cr3t-do-not-repeat-this-value"

    with pytest.raises(PasswordPolicyError) as caught:
        validate_password(secret[:4])
    assert secret[:4] not in str(caught.value)

    with pytest.raises(PasswordPolicyError) as caught:
        validate_password(secret * 10)
    assert secret not in str(caught.value)


def test_hashing_enforces_the_policy() -> None:
    """The policy cannot be bypassed by calling hash_password directly."""
    with pytest.raises(PasswordPolicyError):
        hash_password("short", rounds=ROUNDS)


# ---------------------------------------------------------------------------
# Unicode normalisation
# ---------------------------------------------------------------------------


def test_equivalent_unicode_spellings_verify_against_each_other() -> None:
    """The same accented password typed on two keyboards must still work.

    U+00E9 and "e" + U+0301 render identically. Without NFC normalisation they
    are different bytes, and a moderator would find a correct password rejected
    on a different machine.
    """
    composed = "passe-café-secure"
    decomposed = "passe-café-secure"

    assert composed != decomposed
    assert verify_password(decomposed, hash_password(composed, rounds=ROUNDS)) is True


# ---------------------------------------------------------------------------
# Timing equalisation
# ---------------------------------------------------------------------------


def test_dummy_verify_runs_without_raising() -> None:
    dummy_verify(ROUNDS)


def test_dummy_verify_costs_about_as_much_as_a_real_verification() -> None:
    """The point of it: an unknown username must not answer faster.

    Bounds are deliberately loose — this asserts the same order of magnitude,
    not a precise duration, so it cannot flake on a busy machine.
    """
    import time

    password = a_valid_password()
    digest = hash_password(password, rounds=ROUNDS)

    dummy_verify(ROUNDS)  # warm the cached dummy hash, as a running app would be

    def time_it(fn, repeats: int = 20) -> float:
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        return time.perf_counter() - start

    real = time_it(lambda: verify_password("wrong-password-entirely", digest))
    dummy = time_it(lambda: dummy_verify(ROUNDS))

    assert 0.2 < (dummy / real) < 5.0, f"dummy={dummy:.4f}s real={real:.4f}s"
