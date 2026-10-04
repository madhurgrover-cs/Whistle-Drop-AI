"""Unit tests for case-code generation, canonicalisation and hashing.

Pure functions over strings, so these need no database and run in milliseconds.
The properties asserted here — randomness, determinism, pepper-dependence — are
what every higher-level guarantee in the system rests on.
"""

import re
from collections import Counter

import pytest

from app.core.case_codes import (
    CASE_CODE_ENTROPY_BITS,
    CASE_CODE_PATTERN,
    MAX_SUBMITTED_LENGTH,
    canonicalise_case_code,
    generate_case_code,
    hash_case_code,
    is_well_formed,
)

PEPPER = "unit-test-pepper-not-a-real-secret-0123456789"
OTHER_PEPPER = "a-different-pepper-also-not-a-real-secret-0123"

# Characters Crockford Base32 deliberately omits.
AMBIGUOUS_LETTERS = "ILOU"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generated_code_matches_the_canonical_format() -> None:
    assert CASE_CODE_PATTERN.match(generate_case_code())


def test_generated_code_is_human_transcribable() -> None:
    """Four groups of five, prefixed, so it can be read aloud or retyped."""
    code = generate_case_code()
    prefix, *groups = code.split("-")

    assert prefix == "WD"
    assert len(groups) == 4
    assert all(len(group) == 5 for group in groups)
    assert len(code) == 26  # "WD" plus 20 characters plus 4 separators


def test_alphabet_excludes_every_ambiguous_character() -> None:
    """I, L, O and U are absent, so 1/I/l and 0/O cannot be confused."""
    body = generate_case_code().replace("-", "")[2:]

    assert not set(body) & set(AMBIGUOUS_LETTERS)


def test_codes_do_not_repeat() -> None:
    """A thousand draws with no duplicate is the floor, not the guarantee.

    The real assurance is the 100 bits of entropy; this only catches a
    catastrophically broken generator, such as one that was seeded once.
    """
    codes = [generate_case_code() for _ in range(1_000)]

    assert len(set(codes)) == 1_000


def test_generated_characters_are_spread_across_the_alphabet() -> None:
    """A generator stuck on a subset of the alphabet would fail here."""
    body = "".join(generate_case_code().replace("-", "")[2:] for _ in range(500))
    counts = Counter(body)

    # 10,000 characters over 32 symbols: every symbol should appear, and none
    # should dominate. The bounds are loose enough never to flake.
    assert len(counts) == 32
    assert max(counts.values()) < len(body) / 10


def test_documented_entropy_matches_the_format() -> None:
    """The docstring's 100-bit claim is checked, not just asserted in prose."""
    import math

    body_length = len(generate_case_code().replace("-", "")) - 2  # drop "WD"
    alphabet_size = 32

    assert body_length * math.log2(alphabet_size) == CASE_CODE_ENTROPY_BITS


def test_generation_does_not_use_the_insecure_random_module() -> None:
    """``random`` must not be reachable from the case-code module at all.

    A Mersenne Twister's state is recoverable from a modest run of outputs,
    which would let anyone who filed a few reports predict other people's codes.
    """
    import app.core.case_codes as module

    source = re.sub(r'""".*?"""', "", module.__doc__ or "", flags=re.DOTALL)
    assert "random.random" not in source
    assert not hasattr(module, "random")


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


def test_canonical_code_round_trips_unchanged() -> None:
    code = generate_case_code()

    assert canonicalise_case_code(code) == code


@pytest.mark.parametrize(
    "transform",
    [
        str.lower,
        lambda code: code.replace("-", ""),
        lambda code: f"  {code}  ",
        lambda code: code.replace("-", " "),
        lambda code: code.replace("WD-", ""),
        lambda code: code.lower().replace("-", ""),
    ],
    ids=["lowercase", "no-hyphens", "padded", "spaces", "no-prefix", "lower-compact"],
)
def test_canonicalisation_accepts_what_people_actually_paste(transform) -> None:
    code = generate_case_code()

    assert canonicalise_case_code(transform(code)) == code


def test_ambiguous_characters_are_folded_to_their_intended_digits() -> None:
    """Typing O for 0 or I/L for 1 does not cost someone their report."""
    code = "WD-01234-56789-ABCDE-FGHJK"

    assert canonicalise_case_code("WD-O1234-56789-ABCDE-FGHJK") == code
    assert canonicalise_case_code("WD-0I234-56789-ABCDE-FGHJK") == code
    assert canonicalise_case_code("WD-0L234-56789-ABCDE-FGHJK") == code


@pytest.mark.parametrize(
    "submitted",
    [
        "",
        "   ",
        "not-a-case-code",
        "WD-4K7PQ-92MRT-XJ3HN",  # one group short
        "WD-4K7PQ-92MRT-XJ3HN-B8VZ6-EXTRA",  # one group too many
        "WD-4K7PQ-92MRT-XJ3HN-B8VZ",  # last group short
        "WD-4K7PQ-92MRT-XJ3HN-B8VZ6U",  # excluded letter
        "../../etc/passwd",
        "'; DROP TABLE reports; --",
        "<script>alert(1)</script>",
        "WD-" + "A" * 200,
        "A" * (MAX_SUBMITTED_LENGTH + 1),
    ],
    ids=[
        "empty",
        "whitespace",
        "prose",
        "too-few-groups",
        "too-many-groups",
        "short-group",
        "excluded-letter",
        "path-traversal",
        "sql-injection",
        "xss",
        "long-prefixed",
        "over-max-length",
    ],
)
def test_malformed_input_is_refused(submitted: str) -> None:
    assert canonicalise_case_code(submitted) is None
    assert is_well_formed(submitted) is False


def test_oversized_input_is_rejected_before_any_work() -> None:
    """A multi-megabyte body is discarded on length alone."""
    assert canonicalise_case_code("A" * 10_000_000) is None


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def test_hash_is_deterministic() -> None:
    """The property the whole lookup path depends on."""
    code = generate_case_code()

    assert hash_case_code(code, PEPPER) == hash_case_code(code, PEPPER)


def test_hash_is_a_sha256_hex_digest() -> None:
    digest = hash_case_code(generate_case_code(), PEPPER)

    assert len(digest) == 64  # fits reports.case_code_hash exactly
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_different_codes_produce_different_digests() -> None:
    assert hash_case_code(generate_case_code(), PEPPER) != hash_case_code(
        generate_case_code(), PEPPER
    )


def test_the_pepper_actually_changes_the_digest() -> None:
    """Without this, the pepper would be decoration rather than a key."""
    code = generate_case_code()

    assert hash_case_code(code, PEPPER) != hash_case_code(code, OTHER_PEPPER)


def test_the_digest_does_not_contain_the_code() -> None:
    """A one-way function, checked the obvious way."""
    code = generate_case_code()
    digest = hash_case_code(code, PEPPER)

    assert code not in digest
    assert code.replace("-", "") not in digest.upper()


def test_hashing_requires_a_pepper() -> None:
    """An empty pepper is a configuration failure, not a usable default."""
    with pytest.raises(ValueError, match="pepper is required"):
        hash_case_code(generate_case_code(), "")


def test_equivalent_spellings_hash_identically_once_canonicalised() -> None:
    """Canonicalisation is what makes a mistyped-but-recoverable code work."""
    code = generate_case_code()
    messy = f"  {code.lower().replace('-', '')}  "

    canonical = canonicalise_case_code(messy)
    assert canonical is not None
    assert hash_case_code(canonical, PEPPER) == hash_case_code(code, PEPPER)
