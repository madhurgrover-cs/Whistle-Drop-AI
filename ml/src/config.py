"""Shared configuration: paths, the seed, the model version, the label set.

One place for every constant the pipeline's stages agree on, so that a change
of seed or category set cannot apply to training but not to evaluation.
"""

from pathlib import Path
from typing import Final

# --- Paths -----------------------------------------------------------------

ML_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = ML_ROOT / "data"
ARTIFACTS_DIR: Final[Path] = ML_ROOT / "artifacts"
REPORTS_DIR: Final[Path] = ML_ROOT / "reports"

DEFAULT_DATASET: Final[Path] = DATA_DIR / "synthetic_reports.csv"

# --- Labels ----------------------------------------------------------------

#: The official WhistleDrop categories, and the only labels this model may
#: emit. Duplicated as plain strings rather than imported from
#: ``app.models.enums`` on purpose: the training pipeline must run from a bare
#: checkout with no application dependencies, no settings and no database. The
#: test suite asserts the two lists stay identical, so the duplication cannot
#: drift unnoticed.
OFFICIAL_CATEGORIES: Final[tuple[str, ...]] = (
    "SECURITY",
    "HARASSMENT",
    "CORRUPTION",
    "TECHNICAL",
    "OTHER",
)

#: The priority levels the heuristic may suggest. Same reasoning as above.
PRIORITY_LEVELS: Final[tuple[str, ...]] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# --- Reproducibility -------------------------------------------------------

#: Fixed seed for every random operation: the split, the solver, and the
#: synthetic data generator. Recorded in the artifact metadata.
RANDOM_SEED: Final[int] = 20260925

#: Held-out proportions. The test set is untouched until final evaluation;
#: validation is what any model choice is made against.
TEST_SIZE: Final[float] = 0.20
VALIDATION_SIZE: Final[float] = 0.20

# --- Model identity --------------------------------------------------------

#: Explicit, non-"latest" version, written into the artifact metadata and
#: destined for ``report_triage.model_version`` when Phase 7 wires inference in.
CATEGORY_MODEL_VERSION: Final[str] = "whistledrop-category-v1"

#: The priority suggester is a documented heuristic, not a trained model. The
#: name says so, so that a stored value can never imply otherwise.
PRIORITY_HEURISTIC_VERSION: Final[str] = "whistledrop-priority-heuristic-v1"

# --- Validation thresholds -------------------------------------------------

#: Shorter than this is not a report; the API enforces the same floor.
MIN_DESCRIPTION_LENGTH: Final[int] = 10

#: The API's ceiling. Anything longer did not come from this system.
MAX_DESCRIPTION_LENGTH: Final[int] = 20_000
