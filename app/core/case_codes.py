"""Case-code generation, canonicalisation and hashing.

A case code is the reporter's only handle on their report, and the only
credential the system issues to them. It is therefore both a *secret* and
something a stressed person has to write down correctly, and the design below
is the compromise between those two demands.

Nothing in this module touches the database, HTTP or logging. It is pure
functions over strings, which is what makes it cheap to test exhaustively.

Format
------
Canonical form::

    WD-4K7PQ-92MRT-XJ3HN-B8VZ6
    ^^ ^^^^^ ^^^^^ ^^^^^ ^^^^^
    |  20 characters of randomness in four groups of five

* **Alphabet** — Crockford Base32: the digits plus the uppercase letters,
  minus ``I``, ``L``, ``O`` and ``U``. Dropping ``I``/``L``/``O`` removes the
  three pairs people confuse when copying by hand (1/I/l, 0/O); dropping ``U``
  is Crockford's trick for making accidental profanity far less likely.
* **Length** — 20 random characters. Each carries log2(32) = 5 bits, so a code
  holds **100 bits of entropy**. For scale: at a sustained one million guesses
  per second, an attacker needs on the order of 10^15 years to expect a single
  hit. Guessing is not a viable attack, which is what lets us skip a slow hash
  (see :func:`hash_case_code`).
* **Grouping** — four groups of five separated by hyphens, with a ``WD-``
  prefix so the string is recognisable as a WhistleDrop code when someone finds
  it saved in a notes app. Neither the hyphens nor the prefix add entropy; both
  exist purely so the code survives being read aloud or retyped.

Canonicalisation
----------------
Lookup accepts what people actually paste: any case, any spacing, hyphens
present or absent, prefix present or absent. :func:`canonicalise_case_code`
folds all of that to one form before hashing, so a transcription slip does not
cost someone their report. It also applies Crockford's ambiguity mapping
(``I``/``L`` to ``1``, ``O`` to ``0``), which is unambiguous *because* those
letters are not in the alphabet.
"""

import hmac
import re
import secrets
from hashlib import sha256

__all__ = [
    "CASE_CODE_ENTROPY_BITS",
    "CASE_CODE_PATTERN",
    "canonicalise_case_code",
    "generate_case_code",
    "hash_case_code",
    "is_well_formed",
]

# Crockford Base32. Deliberately excludes I, L, O and U.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

_PREFIX = "WD"
_GROUP_SIZE = 5
_GROUP_COUNT = 4
_RANDOM_LENGTH = _GROUP_SIZE * _GROUP_COUNT  # 20 characters

#: Entropy of a single generated code, in bits.
CASE_CODE_ENTROPY_BITS = 100  # 20 characters x log2(32) bits

#: The canonical presentation form, e.g. ``WD-4K7PQ-92MRT-XJ3HN-B8VZ6``.
CASE_CODE_PATTERN = re.compile(
    rf"^{_PREFIX}-(?:[{_ALPHABET}]{{{_GROUP_SIZE}}}-){{{_GROUP_COUNT - 1}}}"
    rf"[{_ALPHABET}]{{{_GROUP_SIZE}}}$"
)

# Characters a reader may reasonably substitute for one in the alphabet.
_AMBIGUITY_MAP = str.maketrans({"I": "1", "L": "1", "O": "0"})

# Stripped before matching: hyphens, whitespace of every kind, underscores.
_SEPARATORS = re.compile(r"[\s\-_]+")

# Generous upper bound on what the lookup endpoint will even look at, so a
# multi-megabyte string is discarded before any work happens.
MAX_SUBMITTED_LENGTH = 128


def generate_case_code() -> str:
    """Return a new random case code in canonical form.

    Uses :func:`secrets.choice`, which draws from the operating system's
    cryptographically secure source. ``random`` is unsuitable here: its Mersenne
    Twister state is recoverable from a modest number of outputs, which would
    let anyone who submitted a few reports predict other people's codes.
    """
    body = "".join(secrets.choice(_ALPHABET) for _ in range(_RANDOM_LENGTH))
    groups = [body[i : i + _GROUP_SIZE] for i in range(0, _RANDOM_LENGTH, _GROUP_SIZE)]
    return "-".join([_PREFIX, *groups])


def canonicalise_case_code(submitted: str) -> str | None:
    """Fold a user-supplied string into canonical form, or ``None`` if it cannot be.

    Accepts lower case, stray whitespace, missing or extra hyphens, and a
    missing ``WD`` prefix. Returns ``None`` for anything that is not a
    structurally valid code — the caller decides what to do about that, and in
    this application it deliberately looks identical to "no such case".
    """
    if not submitted or len(submitted) > MAX_SUBMITTED_LENGTH:
        return None

    compact = _SEPARATORS.sub("", submitted).upper().translate(_AMBIGUITY_MAP)

    if compact.startswith(_PREFIX):
        compact = compact[len(_PREFIX) :]

    if len(compact) != _RANDOM_LENGTH:
        return None
    if any(character not in _ALPHABET for character in compact):
        return None

    groups = [compact[i : i + _GROUP_SIZE] for i in range(0, _RANDOM_LENGTH, _GROUP_SIZE)]
    return "-".join([_PREFIX, *groups])


def is_well_formed(submitted: str) -> bool:
    """Whether ``submitted`` can be read as a case code at all."""
    return canonicalise_case_code(submitted) is not None


def hash_case_code(case_code: str, pepper: str) -> str:
    """Return the HMAC-SHA256 hex digest stored in ``reports.case_code_hash``.

    The caller must pass a canonical code; :func:`canonicalise_case_code` is not
    applied here so that hashing stays a single, obvious operation.

    **Why HMAC and not bcrypt/argon2.** A password hash is deliberately slow
    because passwords are low-entropy and guessable. A case code is neither: it
    is 100 uniformly random bits, so there is nothing to guess and no dictionary
    to run. What we need instead is *determinism* — the same code and pepper
    must always produce the same digest — because that is what allows the single
    indexed equality lookup that case tracking depends on. A salted password
    hash cannot do that; it would force a full table scan and a slow verify
    against every row.

    The pepper supplies what the case code cannot: it is server-side only, so an
    attacker holding a database dump has digests they cannot connect to any code
    without also stealing the application's environment.
    """
    if not pepper:
        raise ValueError("A pepper is required to hash a case code.")

    return hmac.new(
        key=pepper.encode("utf-8"),
        msg=case_code.encode("utf-8"),
        digestmod=sha256,
    ).hexdigest()
