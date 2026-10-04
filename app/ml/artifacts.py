"""Loading the trained artifact, once, safely.

The only module in the application that imports joblib. scikit-learn is never
imported here by name at all: the fitted objects arrive inside the artifact,
and joblib pulls in whatever they need. That keeps the failure mode clean — if
scikit-learn is not installed, loading fails like any other artifact problem
and triage degrades, rather than the application failing to import.

Loading rules
-------------
* **Lazy.** Nothing is read from disk until the first inference. Startup never
  touches the file, so a missing or corrupt artifact cannot stop the
  application serving reports — which is the one thing that must always work.
* **Once.** The result is cached, success or failure. A broken artifact is not
  retried on every request; the operator fixes it and restarts.
* **Thread-safe.** A lock guards the load. Afterwards the objects are only ever
  read, and scikit-learn's ``predict``/``transform`` on a fitted estimator do
  not mutate it, so no lock is needed per inference.
* **Server-controlled.** The path is built from settings and a pinned version.
  No value from a request reaches it, so nothing a client sends can cause an
  arbitrary file to be deserialised.

On the trust boundary
---------------------
``joblib.load`` unpickles, and unpickling arbitrary data is arbitrary code
execution. That is safe here for exactly one reason: **the artifact is
application-owned data, produced by this project's own training pipeline and
shipped with the deployment.** It is never uploaded, never fetched, never named
by a request. If that ever stops being true, this loader is the wrong shape and
a non-executable format is required.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Layout written by ``ml/src/train_category.py``:
#: ``<root>/<version>/category/model.joblib``
_MODEL_FILENAME = "model.joblib"
_CATEGORY_SUBDIR = "category"

#: Keys the bundle must contain. Checked on load so that an artifact from a
#: different or future pipeline is rejected as incompatible rather than
#: producing an obscure AttributeError at inference time.
_REQUIRED_KEYS = frozenset({"vectorizer", "classifier", "labels", "model_version"})


class ArtifactUnavailableError(RuntimeError):
    """The artifact could not be loaded.

    Carries a short, operator-facing reason. It is logged, never returned to a
    client: a filesystem path or a library traceback in an API response would
    be exactly the disclosure Phase 5 exists to prevent.
    """


@dataclass(frozen=True, slots=True)
class CategoryArtifact:
    """A loaded, ready-to-use model bundle."""

    vectorizer: Any
    classifier: Any
    labels: tuple[str, ...]
    model_version: str
    source: Path


def artifact_path(root: Path, version: str) -> Path:
    """Where a given model version lives.

    ``version`` comes from settings and is a fixed identifier, not a path
    fragment from a request. It is still validated by the caller before use —
    see :func:`load_category_artifact`.
    """
    return root / version / _CATEGORY_SUBDIR / _MODEL_FILENAME


class ArtifactLoader:
    """Loads and caches one model version.

    One instance per process, held by ``app.ml.inference``. Constructed with
    plain values rather than a ``Settings`` object so it can be exercised
    directly in tests.
    """

    def __init__(self, *, root: Path, version: str) -> None:
        self._root = Path(root)
        self._version = version
        self._lock = threading.Lock()
        self._loaded: CategoryArtifact | None = None
        self._failure: str | None = None
        self._attempted = False

    @property
    def version(self) -> str:
        return self._version

    @property
    def failure_reason(self) -> str | None:
        """Why loading failed, for logs and the operator-facing status."""
        return self._failure

    def get(self) -> CategoryArtifact:
        """Return the artifact, loading it on first use.

        Raises :class:`ArtifactUnavailableError` if it cannot be loaded — the
        same error every time, without re-reading the file.
        """
        if self._loaded is not None:
            return self._loaded
        if self._attempted and self._failure is not None:
            raise ArtifactUnavailableError(self._failure)

        with self._lock:
            # Re-check: another thread may have loaded it while we waited.
            if self._loaded is not None:
                return self._loaded
            if self._attempted and self._failure is not None:
                raise ArtifactUnavailableError(self._failure)

            self._attempted = True
            try:
                self._loaded = self._load()
            except ArtifactUnavailableError as exc:
                self._failure = str(exc)
                # The reason is operator-facing and contains no report data.
                logger.error("Triage model unavailable: %s", self._failure)
                raise

            logger.info(
                "Loaded triage model %s (%d classes).",
                self._loaded.model_version,
                len(self._loaded.labels),
            )
            return self._loaded

    def _load(self) -> CategoryArtifact:
        path = artifact_path(self._root, self._version)

        if not path.is_file():
            raise ArtifactUnavailableError(f"artifact not found for version {self._version!r}")

        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - environment problem
            raise ArtifactUnavailableError("joblib is not installed") from exc

        try:
            bundle = joblib.load(path)
        except Exception as exc:
            # Deliberately broad: a corrupt or truncated file raises almost
            # anything, and every one of them means the same thing here. The
            # exception type is logged; nothing from it reaches a client.
            raise ArtifactUnavailableError(
                f"artifact could not be deserialised ({type(exc).__name__})"
            ) from exc

        if not isinstance(bundle, dict):
            raise ArtifactUnavailableError("artifact is not a model bundle")

        missing = _REQUIRED_KEYS - set(bundle)
        if missing:
            raise ArtifactUnavailableError(f"artifact is missing {', '.join(sorted(missing))}")

        stored_version = bundle["model_version"]
        if stored_version != self._version:
            # Refusing a mismatch is the point of pinning a version: a triage
            # row must never record a version the artifact does not match.
            raise ArtifactUnavailableError(
                f"artifact declares version {stored_version!r}, expected {self._version!r}"
            )

        vectorizer, classifier = bundle["vectorizer"], bundle["classifier"]
        for name, obj, method in (
            ("vectorizer", vectorizer, "transform"),
            ("classifier", classifier, "predict_proba"),
        ):
            if not hasattr(obj, method):
                raise ArtifactUnavailableError(f"{name} does not implement {method}()")

        labels = tuple(str(label) for label in bundle["labels"])
        if not labels:
            raise ArtifactUnavailableError("artifact declares no labels")

        return CategoryArtifact(
            vectorizer=vectorizer,
            classifier=classifier,
            labels=labels,
            model_version=str(stored_version),
            source=path,
        )
