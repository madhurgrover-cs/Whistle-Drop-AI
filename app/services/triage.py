"""Running triage for a report and recording the outcome.

The contract, and the reason this service exists
------------------------------------------------
**Nothing this service does may affect whether a report was filed.**

A reporter pressing submit is often taking a real risk. Their case code and
their report must not depend on a model file being present, a vectoriser
loading, or an inference succeeding. So :meth:`TriageService.run_for_report`
**never raises**: every failure path ends in a ``FAILED`` triage row and a
return, and even a failure to write *that* is swallowed and logged.

Where it runs in the flow
-------------------------
After the submission transaction has committed::

    validate -> create report + initial case update -> COMMIT -> infer -> write triage

Deliberately not inside that transaction. Holding one open across inference
would mean the model's latency is time a database connection sits idle in a
transaction, and — worse — that a model failure could roll back a report that
was already durable. Committing first makes the report unconditional and makes
triage a genuinely separate, failable step.

The cost is a brief window in which a report exists with no triage row. That is
the correct trade: a report with no suggestion is a report a moderator reads
unaided, which is how the system works anyway. A suggestion with no report is
nothing at all.
"""

import logging
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.ml.inference import InferenceError, TriageModel
from app.models import TriageStatus
from app.repositories.triage import ReportTriageRepository

logger = logging.getLogger(__name__)


class TriageService:
    """Produces and stores an advisory triage suggestion for a report."""

    def __init__(self, session: Session, *, model: TriageModel | None) -> None:
        self._session = session
        self._model = model
        self._triage = ReportTriageRepository(session)

    @property
    def is_enabled(self) -> bool:
        """Whether triage is configured to run at all."""
        return self._model is not None

    def run_for_report(self, report_id: UUID, description: str) -> TriageStatus | None:
        """Infer and store a suggestion. Never raises.

        Returns the triage status recorded, or ``None`` when triage is disabled
        and nothing was written. The return value is for callers that want to
        log or test the outcome; the report submission path ignores it.
        """
        if self._model is None:
            return None

        try:
            suggestion = self._model.predict(description)
        except InferenceError as exc:
            # The reason names the failure, never the text that caused it.
            logger.warning("Triage failed for a report: %s", exc)
            return self._record_failure(report_id)
        except Exception:
            # A defect rather than an expected failure. Logged with a
            # traceback for the operator; still must not affect the report.
            logger.exception("Unexpected error during triage inference.")
            return self._record_failure(report_id)

        try:
            self._triage.upsert(
                report_id,
                suggested_category=suggestion.suggested_category,
                category_confidence=suggestion.confidence,
                suggested_priority=suggestion.priority,
                keywords=list(suggestion.keywords),
                model_version=suggestion.model_version,
                status=TriageStatus.COMPLETED,
            )
            self._session.commit()
        except SQLAlchemyError:
            self._session.rollback()
            logger.exception("Could not store a completed triage result.")
            return self._record_failure(report_id)

        logger.info(
            "Triage completed with model %s (category suggested, priority by heuristic).",
            suggestion.model_version,
        )
        return TriageStatus.COMPLETED

    def _record_failure(self, report_id: UUID) -> TriageStatus | None:
        """Record that triage was attempted and did not produce a suggestion.

        A ``FAILED`` row is more useful than no row: it distinguishes "the
        model was tried and could not help" from "nobody has looked at this
        yet", which is what a missing row means. The prediction fields stay
        null — there is no prediction, and inventing one would be worse than
        having none.

        If even this cannot be written, the failure is logged and swallowed.
        The report is already committed and is not at risk either way.
        """
        try:
            self._triage.upsert(
                report_id,
                suggested_category=None,
                category_confidence=None,
                suggested_priority=None,
                keywords=[],
                model_version=self._model.model_version if self._model else None,
                status=TriageStatus.FAILED,
            )
            self._session.commit()
        except SQLAlchemyError:
            self._session.rollback()
            logger.exception("Could not store a failed triage result either.")
            return None
        except Exception:
            self._session.rollback()
            logger.exception("Unexpected error while recording a triage failure.")
            return None

        return TriageStatus.FAILED
