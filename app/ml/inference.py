"""Runtime triage inference.

The whole ML surface the rest of the application sees is
:meth:`TriageModel.predict`. Everything below it — joblib, the fitted
estimators, numpy — stays inside ``app/ml/``.

What this does and does not decide
----------------------------------
It produces a **suggestion**: a category the model finds most likely, the
probability it assigned, a heuristic priority, and the terms that moved the
decision. None of it is authoritative. The official category is the one the
reporter chose and a moderator may change; nothing here can touch it, and the
database keeps the two in separate tables so that it structurally cannot.

Reused rather than reimplemented
--------------------------------
Keyword extraction and the priority heuristic come from ``ml/src`` — the same
code Phase 6 evaluated and documented. Writing a second implementation here
would mean a moderator could be shown keywords produced by logic that was never
evaluated, and the two would drift. Only the *inference-side* helpers are
imported; the training entrypoint is not, and a test asserts it never is.

Trusting the model's output no further than necessary
-----------------------------------------------------
Every field is validated before it leaves this module. A model is a file on
disk: swap it, and it can return anything. ``report_triage`` has database
constraints for exactly this reason, but a constraint violation surfacing as a
failed insert is a worse outcome than catching it here, so the checks are made
before the value is ever offered to the database.
"""

import logging
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from app.ml.artifacts import ArtifactLoader, ArtifactUnavailableError
from app.models.enums import ReportCategory, TriagePriority
from ml.src.keywords import extract_keywords
from ml.src.priority import suggest_priority
from ml.src.text import normalise_text

logger = logging.getLogger(__name__)

#: Keywords are advisory decoration on an advisory suggestion. Bounded so a
#: strange model cannot write an unbounded blob into JSONB.
MAX_KEYWORDS = 10
MAX_KEYWORD_LENGTH = 64

#: ``report_triage.category_confidence`` is ``NUMERIC(4, 3)``. Quantising to
#: three decimals here means the value stored is exactly the value checked,
#: and a float artefact such as 1.0000000000000002 cannot trip the database's
#: ``0 <= x <= 1`` constraint.
CONFIDENCE_PLACES = Decimal("0.001")


class InferenceError(RuntimeError):
    """Inference could not produce a usable suggestion.

    Always carries an operator-facing reason with no report text in it. The
    caller records a ``FAILED`` triage row; the reporter sees nothing at all.
    """


@dataclass(frozen=True, slots=True)
class TriageSuggestion:
    """A validated, ready-to-store triage suggestion."""

    suggested_category: ReportCategory
    confidence: Decimal
    priority: TriagePriority
    keywords: list[str] = field(default_factory=list)
    model_version: str = ""
    priority_signals: list[str] = field(default_factory=list)


class TriageModel:
    """Turns report text into a triage suggestion.

    Holds the loader, not the artifact: nothing is read from disk until the
    first call, so constructing one is free and an unusable artifact cannot
    affect startup.
    """

    def __init__(self, *, artifact_root: Path, model_version: str) -> None:
        self._loader = ArtifactLoader(root=artifact_root, version=model_version)

    @property
    def model_version(self) -> str:
        return self._loader.version

    def is_available(self) -> bool:
        """Whether the artifact can be loaded. Loads it if it has not been."""
        try:
            self._loader.get()
        except ArtifactUnavailableError:
            return False
        return True

    def predict(self, description: str) -> TriageSuggestion:
        """Suggest a category, confidence, priority and keywords for ``description``.

        Raises :class:`InferenceError` for any failure — a missing artifact, a
        model that will not predict, or output that fails validation. It never
        raises anything else, so a caller has exactly one thing to handle.
        """
        text = normalise_text(description)
        if not text:
            raise InferenceError("description is empty after normalisation")

        try:
            artifact = self._loader.get()
        except ArtifactUnavailableError as exc:
            raise InferenceError(str(exc)) from exc

        try:
            matrix = artifact.vectorizer.transform([text])
            probabilities = artifact.classifier.predict_proba(matrix)[0]
            classes = list(artifact.classifier.classes_)
        except Exception as exc:
            # Broad on purpose: a mis-shaped artifact can fail in many ways and
            # all of them mean the same thing to the caller. The type is
            # logged; the text that caused it never is.
            logger.warning("Triage inference failed: %s", type(exc).__name__)
            raise InferenceError(f"inference failed ({type(exc).__name__})") from exc

        index = _winning_index(classes, probabilities)
        category = _validated_category(classes, index)
        confidence = _validated_confidence(probabilities, index)

        # Keywords are best-effort: a failure here loses decoration, not the
        # suggestion, so it must not fail the whole triage.
        keywords = self._safe_keywords(text, artifact, category.value)

        suggestion = suggest_priority(description, category.value)
        priority = _validated_priority(suggestion.priority)

        return TriageSuggestion(
            suggested_category=category,
            confidence=confidence,
            priority=priority,
            keywords=keywords,
            model_version=artifact.model_version,
            priority_signals=list(suggestion.signals),
        )

    @staticmethod
    def _safe_keywords(text: str, artifact: Any, predicted: str) -> list[str]:
        try:
            raw = extract_keywords(
                text,
                vectorizer=artifact.vectorizer,
                classifier=artifact.classifier,
                predicted_class=predicted,
                top_k=MAX_KEYWORDS,
            )
        except Exception as exc:
            logger.warning("Keyword extraction failed: %s", type(exc).__name__)
            return []
        return sanitise_keywords(raw)


# ---------------------------------------------------------------------------
# Validation of whatever the model returned
# ---------------------------------------------------------------------------


def _winning_index(classes: list[Any], probabilities: Any) -> int:
    """The position of the class the model scored highest."""
    if len(classes) != len(probabilities):
        raise InferenceError("model returned a probability vector of the wrong length")
    if not classes:
        raise InferenceError("model returned no classes")

    return max(range(len(classes)), key=lambda index: probabilities[index])


def _validated_category(classes: list[Any], index: int) -> ReportCategory:
    """The selected class, if it is an official category.

    A model whose classes are not the official five is not this project's
    model, and its output must not reach the database.
    """
    label = str(classes[index])

    try:
        return ReportCategory(label)
    except ValueError as exc:
        raise InferenceError(f"model suggested an unknown category {label!r}") from exc


def _validated_confidence(probabilities: Any, index: int) -> Decimal:
    """The probability of the *selected* class, quantised and range-checked.

    Deliberately the selected class's probability rather than the vector's
    maximum. They are the same for any well-formed distribution, and differ
    only when the model has returned something it should not have — in which
    case reporting the maximum would quietly substitute a plausible number for
    the real one.

    Rejected rather than clamped when out of range. Clamping a nonsense value
    to 1.0 would store a confident-looking number produced by a broken model;
    a ``FAILED`` triage row is the honest record of what happened.
    """
    try:
        value = Decimal(str(float(probabilities[index])))
    except (TypeError, ValueError, InvalidOperation) as exc:
        raise InferenceError("model returned a non-numeric confidence") from exc

    if not value.is_finite():
        raise InferenceError("model returned a non-finite confidence")

    quantised = value.quantize(CONFIDENCE_PLACES, rounding=ROUND_HALF_UP)

    if quantised < 0 or quantised > 1:
        raise InferenceError("model returned a confidence outside 0.0-1.0")

    return quantised


def _validated_priority(priority: str) -> TriagePriority:
    """The heuristic's level, checked against the official set."""
    try:
        return TriagePriority(priority)
    except ValueError as exc:
        raise InferenceError(f"heuristic produced an unknown priority {priority!r}") from exc


def sanitise_keywords(raw: Any) -> list[str]:
    """Coerce whatever came back into a bounded list of non-empty strings.

    Sanitised rather than rejected: keywords are decoration on the suggestion,
    and losing them is not worth discarding a usable category. The result is
    always a JSON array of strings, which is what the ``jsonb_typeof =
    'array'`` constraint on the column requires.
    """
    if not isinstance(raw, list | tuple):
        return []

    cleaned: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        term = item.strip()[:MAX_KEYWORD_LENGTH]
        if term and term not in cleaned:
            cleaned.append(term)
        if len(cleaned) >= MAX_KEYWORDS:
            break
    return cleaned
