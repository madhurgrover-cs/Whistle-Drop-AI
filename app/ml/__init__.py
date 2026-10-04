"""The application's machine-learning boundary.

Everything the API knows about the model lives behind this package. joblib, the
fitted estimators and numpy are imported here and nowhere else in ``app/``, so
the rest of the application neither depends on a scientific stack nor has any
way to reach the model except through :class:`~app.ml.inference.TriageModel`.

The offline training pipeline in ``ml/`` stays separate: this package imports
its *inference-side* helpers — text normalisation, keyword extraction, the
priority heuristic — so that runtime behaviour matches what Phase 6 actually
evaluated, and imports nothing that trains.
"""

from app.ml.artifacts import ArtifactUnavailableError, CategoryArtifact
from app.ml.inference import InferenceError, TriageModel, TriageSuggestion

__all__ = [
    "ArtifactUnavailableError",
    "CategoryArtifact",
    "InferenceError",
    "TriageModel",
    "TriageSuggestion",
]
