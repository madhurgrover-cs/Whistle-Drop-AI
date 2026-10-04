"""Heuristic priority suggestion.

This is **not a trained model**, and nothing in the codebase may describe it as
one. It is a deterministic scoring rule over a published list of signals.

Why a heuristic
---------------
A priority classifier needs priority labels, and there are none. No dataset in
this project carries a severity or urgency target, and urgency is not derivable
from the category — a password in a public channel and a stale dashboard are
both ``TECHNICAL``-adjacent and worlds apart in how fast someone should look.

The options were: invent priority labels (dishonest, and the model would learn
whatever rule invented them), train on a proxy such as how quickly a report was
resolved (no such data, and it would encode past staffing rather than severity),
or write the rule down. Writing it down is the only one that is truthful, and it
has the advantage of being fully explainable: :func:`suggest_priority` returns
the exact signals that fired.

When labelled priority data exists, this becomes the baseline to beat.

Properties
----------
* **Deterministic** — same text, same category, same answer, always. No
  randomness, no clock, no model.
* **Explainable** — every suggestion comes with the signals that produced it.
* **Advisory** — it suggests a queue order. It cannot change a report's
  category, its status, or anything else a moderator decides.
* **Conservative at the top** — ``CRITICAL`` requires more than one independent
  signal, because a level that fires easily stops meaning anything.
"""

import re
from dataclasses import dataclass, field
from typing import Final

from ml.src.config import PRIORITY_HEURISTIC_VERSION, PRIORITY_LEVELS
from ml.src.text import normalise_text

# --- Signals ---------------------------------------------------------------
#
# Each entry is (name, weight, pattern). Weights are small integers, chosen
# relative to one another rather than calibrated against anything — there is no
# data to calibrate against, and pretending otherwise would be the same mistake
# as training on invented labels.
#
# Patterns match on normalised (lower-cased, whitespace-collapsed) text and use
# word boundaries, so "safety" does not match inside another word.

SignalSpec = tuple[str, int, str]

SIGNALS: Final[tuple[SignalSpec, ...]] = (
    # Someone may be in danger. The strongest single signal in the list.
    (
        "risk_to_a_person",
        4,
        r"\b(threat(en(ed|ing)?)?|assault|violence|unsafe|"
        r"injur(y|ed)|harm|suicid\w*|abuse|retaliat\w*)\b",
    ),
    # Live exposure of credentials or personal data.
    (
        "credential_or_data_exposure",
        3,
        r"\b(password|passwords|credential\w*|api key\w*|"
        r"private key|secret\w*|token\w*|personal data|"
        r"customer data|payroll|medical record\w*)\b",
    ),
    (
        "publicly_accessible",
        3,
        r"\b(public(ly)?|anyone can|everyone can|world.readable|"
        r"unrestricted|open to all|no password)\b",
    ),
    # Still happening, so the cost grows while it waits in the queue.
    (
        "ongoing_or_recurring",
        2,
        r"\b(ongoing|still happening|every day|daily|repeatedly|"
        r"continues|keeps happening|each week|every night)\b",
    ),
    (
        "urgent_language",
        2,
        r"\b(urgent(ly)?|immediate(ly)?|emergency|as soon as possible|"
        r"right now|asap)\b",
    ),
    # Scale: more people affected, or money at stake.
    (
        "many_people_affected",
        2,
        r"\b(everyone|all staff|whole (team|company|department)|"
        r"hundreds|thousands|all customers|company.wide)\b",
    ),
    (
        "financial_scale",
        2,
        r"\b(million\w*|thousands of (pounds|dollars|euros)|"
        r"[£$€]\s?\d[\d,]{3,})\b",
    ),
    # Legal or regulatory exposure.
    (
        "legal_or_regulatory",
        2,
        r"\b(illegal|unlawful|fraud\w*|bribe\w*|regulator\w*|"
        r"gdpr|breach of (law|contract)|criminal)\b",
    ),
    # Weak signals that nudge rather than decide.
    ("evidence_supplied", 1, r"\b(screenshot\w*|evidence|recording|email trail|logs?)\b"),
    (
        "previously_raised",
        1,
        r"\b(raised (it|this) (with|before)|reported (this )?before|"
        r"nothing (was done|changed)|ignored)\b",
    ),
)

_COMPILED: Final[tuple[tuple[str, int, re.Pattern[str]], ...]] = tuple(
    (name, weight, re.compile(pattern)) for name, weight, pattern in SIGNALS
)

#: A small base score per official category, reflecting how quickly the class
#: typically needs eyes on it. Never decisive on its own: the gap between the
#: highest and lowest base is smaller than a single strong signal.
CATEGORY_BASE: Final[dict[str, int]] = {
    "SECURITY": 2,
    "HARASSMENT": 2,
    "CORRUPTION": 1,
    "TECHNICAL": 1,
    "OTHER": 0,
}

#: Score thresholds. Read as: anything with no signals at all is LOW; one
#: moderate signal reaches MEDIUM; a strong signal or two moderate ones reach
#: HIGH; CRITICAL additionally requires two *distinct* signals, so that one
#: emotive word cannot escalate a report on its own.
THRESHOLD_MEDIUM: Final[int] = 3
THRESHOLD_HIGH: Final[int] = 6
THRESHOLD_CRITICAL: Final[int] = 9
MIN_SIGNALS_FOR_CRITICAL: Final[int] = 2


@dataclass(frozen=True)
class PrioritySuggestion:
    """A suggested priority and the reasoning behind it."""

    priority: str
    score: int
    signals: list[str] = field(default_factory=list)
    heuristic_version: str = PRIORITY_HEURISTIC_VERSION

    def explain(self) -> str:
        if not self.signals:
            return f"{self.priority} (score {self.score}; no severity signals matched)"
        return f"{self.priority} (score {self.score}; signals: {', '.join(self.signals)})"


def suggest_priority(description: str, category: str | None = None) -> PrioritySuggestion:
    """Suggest a priority for ``description``.

    ``category`` contributes a small base score when supplied. Pass the
    *official* category when a moderator has set one; passing the model's
    suggestion instead is acceptable but means one advisory signal is feeding
    another, which is worth knowing when reading the result.

    Returns the level, the score, and the names of every signal that fired.
    """
    text = normalise_text(description)

    matched: list[str] = []
    score = 0
    for name, weight, pattern in _COMPILED:
        if pattern.search(text):
            matched.append(name)
            score += weight

    if category:
        score += CATEGORY_BASE.get(category.upper(), 0)

    return PrioritySuggestion(
        priority=_level_for(score, len(matched)),
        score=score,
        signals=matched,
    )


def _level_for(score: int, signal_count: int) -> str:
    """Map a score to a level.

    ``CRITICAL`` needs two distinct signals as well as the score, so that a
    single strong match — "threatened", say, in a sentence about a threatened
    deadline — cannot push a report to the top of the queue alone.
    """
    if score >= THRESHOLD_CRITICAL and signal_count >= MIN_SIGNALS_FOR_CRITICAL:
        return "CRITICAL"
    if score >= THRESHOLD_HIGH:
        return "HIGH"
    if score >= THRESHOLD_MEDIUM:
        return "MEDIUM"
    return "LOW"


def describe() -> dict[str, object]:
    """The heuristic's configuration, for the artifact metadata.

    Records the rule, not any text it has ever been applied to.
    """
    return {
        "type": "deterministic_rule_heuristic",
        "is_trained_model": False,
        "version": PRIORITY_HEURISTIC_VERSION,
        "levels": list(PRIORITY_LEVELS),
        "signals": [{"name": name, "weight": weight} for name, weight, _ in SIGNALS],
        "category_base_scores": dict(CATEGORY_BASE),
        "thresholds": {
            "MEDIUM": THRESHOLD_MEDIUM,
            "HIGH": THRESHOLD_HIGH,
            "CRITICAL": THRESHOLD_CRITICAL,
            "min_signals_for_critical": MIN_SIGNALS_FOR_CRITICAL,
        },
    }
