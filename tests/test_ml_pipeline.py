"""Tests for the offline training pipeline.

These need no database, no FastAPI and no network — which is itself one of the
things asserted, since a training pipeline that quietly depends on the
application is a training pipeline that cannot be run from a clean checkout.

Training runs into a temporary directory so the committed artifact is never
disturbed by a test run.
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.src import evaluate as evaluation
from ml.src.config import (
    CATEGORY_MODEL_VERSION,
    DEFAULT_DATASET,
    OFFICIAL_CATEGORIES,
    PRIORITY_HEURISTIC_VERSION,
    PRIORITY_LEVELS,
    RANDOM_SEED,
)
from ml.src.data import (
    FORBIDDEN_FEATURE_COLUMNS,
    check_split_leakage,
    dataset_fingerprint,
    load_dataset,
    normalise_text,
    split,
    validate,
)
from ml.src.features import build_vectorizer, describe
from ml.src.keywords import extract_keywords, top_features_per_class
from ml.src.priority import PrioritySuggestion, suggest_priority
from ml.src.train_category import build_classifier
from ml.src.train_category import main as train_main

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def frame_from(records: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=["description", "category"])


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return load_dataset(DEFAULT_DATASET)


@pytest.fixture(scope="module")
def validated(dataset: pd.DataFrame):
    return validate(dataset, DEFAULT_DATASET)


@pytest.fixture(scope="module")
def trained(validated):
    """Train once for the whole module; several tests share the result."""
    valid, _ = validated
    train, _validation, test = split(valid, seed=RANDOM_SEED)

    vectorizer = build_vectorizer(len(train))
    X_train = vectorizer.fit_transform(train["_text"])
    classifier = build_classifier(class_weight=None, seed=RANDOM_SEED)
    classifier.fit(X_train, train["_label"].to_numpy())

    return vectorizer, classifier, train, test


# ---------------------------------------------------------------------------
# 1-4. Dataset validation
# ---------------------------------------------------------------------------


def test_the_committed_dataset_validates(validated) -> None:
    _valid, report = validated

    assert report.total_records > 0
    assert report.valid_records == report.total_records
    assert report.is_trainable


def test_required_columns_are_enforced(tmp_path: Path) -> None:
    path = tmp_path / "no_label.csv"
    path.write_text("description\nsomething happened here\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required column"):
        load_dataset(path)


def test_an_invalid_label_is_detected_and_excluded() -> None:
    frame = frame_from(
        [
            ("A perfectly ordinary report about a security issue.", "SECURITY"),
            ("A report carrying a label nobody defined.", "ESPIONAGE"),
            ("Another report with an empty label.", ""),
        ]
    )

    valid, report = validate(frame, Path("memory.csv"))

    assert report.total_records == 3
    assert report.valid_records == 1
    assert report.invalid_records == 2
    assert report.issues["label_not_in_official_categories"] == 1
    assert report.issues["missing_label"] == 1
    assert set(valid["_label"]) == {"SECURITY"}


def test_empty_and_too_short_text_is_detected() -> None:
    frame = frame_from(
        [
            ("A report of a perfectly reasonable length.", "OTHER"),
            ("", "OTHER"),
            ("   ", "OTHER"),
            ("short", "OTHER"),
        ]
    )

    _valid, report = validate(frame, Path("memory.csv"))

    assert report.valid_records == 1
    assert report.issues["empty_or_missing_text"] == 2
    assert report.issues["text_below_minimum_length"] == 1


def test_excessively_long_text_is_detected() -> None:
    frame = frame_from([("x" * 25_000, "OTHER"), ("A normal report body.", "OTHER")])

    _valid, report = validate(frame, Path("memory.csv"))

    assert report.issues["text_above_maximum_length"] == 1
    assert report.valid_records == 1


def test_duplicate_descriptions_are_counted() -> None:
    frame = frame_from(
        [
            ("The very same report text, filed twice over.", "SECURITY"),
            ("The very same report text, filed twice over.", "SECURITY"),
            ("A different report entirely, filed once.", "OTHER"),
        ]
    )

    _valid, report = validate(frame, Path("memory.csv"))

    assert report.duplicate_descriptions == 2
    assert report.valid_records == 3  # counted, not discarded


def test_identical_text_with_conflicting_labels_is_detected() -> None:
    """Two moderators reading one report differently. Real, and worth knowing."""
    frame = frame_from(
        [
            ("The supplier engineer still has an admin login here.", "SECURITY"),
            ("The supplier engineer still has an admin login here.", "CORRUPTION"),
            ("An unrelated report about the canteen.", "OTHER"),
        ]
    )

    _valid, report = validate(frame, Path("memory.csv"))

    assert report.conflicting_label_groups == 1
    assert report.issues["identical_text_with_different_labels"] == 2


def test_the_committed_dataset_contains_its_planted_conflicts(validated) -> None:
    _valid, report = validated

    assert report.conflicting_label_groups == 3
    assert report.duplicate_descriptions > 0


def test_nothing_is_silently_discarded() -> None:
    """Valid + invalid must always account for every input row."""
    frame = frame_from(
        [
            ("A perfectly fine report body here.", "SECURITY"),
            ("", "SECURITY"),
            ("A body with a bad label.", "NOPE"),
            ("short", "OTHER"),
        ]
    )

    _valid, report = validate(frame, Path("memory.csv"))

    assert report.valid_records + report.invalid_records == report.total_records == 4


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_columns_that_would_leak_are_dropped() -> None:
    """A dataset exported from the database would carry the answer."""
    frame = pd.DataFrame(
        [
            {
                "description": "A report about credentials in a public channel.",
                "category": "SECURITY",
                "status": "RESOLVED",
                "moderator_id": "7c9d1b2a-3e4f-456a-8b9c-1d2e3f4a5b6c",
                "case_code_hash": "a" * 64,
                "id": "some-uuid",
            }
        ]
    )

    valid, report = validate(frame, Path("memory.csv"))

    assert report.issues["dropped_leaking_column"] == 4
    for column in ("status", "moderator_id", "case_code_hash", "id"):
        assert column not in valid.columns


def test_the_forbidden_list_covers_the_obvious_leaks() -> None:
    for column in ("status", "moderator_id", "case_code_hash", "id", "priority"):
        assert column in FORBIDDEN_FEATURE_COLUMNS


def test_only_the_description_is_used_as_a_feature(validated) -> None:
    valid, _ = validated

    # The working frame carries exactly the original columns plus the two
    # derived ones, and the vectoriser is only ever fitted on _text.
    assert "_text" in valid.columns
    assert "_label" in valid.columns
    assert set(valid.columns) == {"description", "category", "_text", "_label"}


def test_no_text_appears_in_more_than_one_split(validated) -> None:
    """The leakage that inflates a score most convincingly."""
    valid, _ = validated

    train, validation, test = split(valid, seed=RANDOM_SEED)
    overlap = check_split_leakage(train["_text"], validation["_text"], test["_text"])

    assert all(count == 0 for count in overlap.values()), overlap
    assert not set(validation["_text"]) & set(test["_text"])


# ---------------------------------------------------------------------------
# 5. Splitting
# ---------------------------------------------------------------------------


def test_the_split_is_reproducible(validated) -> None:
    valid, _ = validated

    first = split(valid, seed=RANDOM_SEED)
    second = split(valid, seed=RANDOM_SEED)

    for a, b in zip(first, second, strict=True):
        assert list(a.index) == list(b.index)


def test_a_different_seed_gives_a_different_split(validated) -> None:
    valid, _ = validated

    default = split(valid, seed=RANDOM_SEED)[2]
    other = split(valid, seed=RANDOM_SEED + 1)[2]

    assert set(default["_text"]) != set(other["_text"])


def test_the_split_preserves_class_proportions(validated) -> None:
    valid, _ = validated
    train, _validation, test = split(valid, seed=RANDOM_SEED)

    overall = valid["_label"].value_counts(normalize=True)
    in_test = test["_label"].value_counts(normalize=True)

    for label in overall.index:
        assert abs(overall[label] - in_test.get(label, 0)) < 0.08, label
    assert set(train["_label"]) == set(valid["_label"])


def test_every_split_is_non_empty(validated) -> None:
    valid, _ = validated

    train, validation, test = split(valid, seed=RANDOM_SEED)

    assert len(train) > len(test)
    assert len(validation) > 0
    assert len(train) + len(validation) + len(test) == len(valid)


# ---------------------------------------------------------------------------
# 6. Features
# ---------------------------------------------------------------------------


def test_tfidf_produces_a_sensible_matrix(validated) -> None:
    valid, _ = validated
    vectorizer = build_vectorizer(len(valid))

    matrix = vectorizer.fit_transform(valid["_text"])

    assert matrix.shape[0] == len(valid)
    assert matrix.shape[1] > 100
    assert matrix.max() <= 1.0 + 1e-9  # L2-normalised rows
    assert matrix.min() >= 0.0


def test_the_vectoriser_uses_word_unigrams_and_bigrams(validated) -> None:
    valid, _ = validated
    vectorizer = build_vectorizer(len(valid))
    vectorizer.fit(valid["_text"])

    features = list(vectorizer.get_feature_names_out())
    assert vectorizer.ngram_range == (1, 2)
    assert any(" " in feature for feature in features), "no bigrams learned"
    assert any(" " not in feature for feature in features), "no unigrams learned"


def test_the_feature_description_records_no_vocabulary(validated) -> None:
    """Metadata must describe the configuration, not reproduce the corpus."""
    valid, _ = validated
    vectorizer = build_vectorizer(len(valid))
    vectorizer.fit(valid["_text"])

    described = describe(vectorizer)

    assert described["vocabulary_size"] > 0
    assert "vocabulary" not in described
    serialised = json.dumps(described)
    for term in list(vectorizer.get_feature_names_out())[:50]:
        assert f'"{term}"' not in serialised


def test_normalisation_is_conservative() -> None:
    assert normalise_text("  Mixed   CASE\n\ntext  ") == "mixed case text"
    # NFC: composed and decomposed accents compare equal.
    assert normalise_text("café") == normalise_text("café")
    # Nothing is stemmed or stripped.
    assert "reporting" in normalise_text("Reporting this")
    assert "not" in normalise_text("I was told not to report this")


# ---------------------------------------------------------------------------
# 7-10. Training and prediction
# ---------------------------------------------------------------------------


def test_the_model_trains(trained) -> None:
    _vectorizer, classifier, _train, _test = trained

    assert hasattr(classifier, "coef_")
    assert classifier.coef_.shape[0] == len(classifier.classes_)


def test_predictions_are_official_categories(trained) -> None:
    vectorizer, classifier, _train, test = trained

    predictions = classifier.predict(vectorizer.transform(test["_text"]))

    assert set(predictions) <= set(OFFICIAL_CATEGORIES)
    assert set(classifier.classes_) <= set(OFFICIAL_CATEGORIES)


def test_probabilities_are_between_zero_and_one(trained) -> None:
    vectorizer, classifier, _train, test = trained

    probabilities = classifier.predict_proba(vectorizer.transform(test["_text"]))

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_probability_rows_sum_to_one(trained) -> None:
    vectorizer, classifier, _train, test = trained

    probabilities = classifier.predict_proba(vectorizer.transform(test["_text"]))

    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert evaluation.probabilities_are_valid(probabilities)


def test_the_confidence_is_the_probability_of_the_predicted_class(trained) -> None:
    vectorizer, classifier, _train, test = trained
    matrix = vectorizer.transform(test["_text"][:20])

    predictions = classifier.predict(matrix)
    probabilities = classifier.predict_proba(matrix)

    for index, prediction in enumerate(predictions):
        position = list(classifier.classes_).index(prediction)
        assert probabilities[index].argmax() == position


def test_the_model_beats_the_majority_baseline(trained) -> None:
    """The question accuracy alone cannot answer."""
    vectorizer, classifier, train, test = trained
    labels = list(OFFICIAL_CATEGORIES)

    X_train = vectorizer.transform(train["_text"])
    X_test = vectorizer.transform(test["_text"])
    y_train = train["_label"].to_numpy()
    y_test = test["_label"].to_numpy()

    baseline = evaluation.majority_baseline(X_train, y_train, X_test, y_test, labels=labels)
    metrics = evaluation.evaluate(y_test, classifier.predict(X_test), labels=labels, split="test")

    assert metrics.macro_f1 > baseline.macro_f1
    assert metrics.accuracy > baseline.accuracy


def test_training_is_reproducible(validated) -> None:
    valid, _ = validated
    train, _validation, test = split(valid, seed=RANDOM_SEED)

    def fit_and_predict():
        vectorizer = build_vectorizer(len(train))
        matrix = vectorizer.fit_transform(train["_text"])
        classifier = build_classifier(class_weight=None, seed=RANDOM_SEED)
        classifier.fit(matrix, train["_label"].to_numpy())
        return classifier.predict_proba(vectorizer.transform(test["_text"]))

    assert np.allclose(fit_and_predict(), fit_and_predict(), atol=1e-10)


# ---------------------------------------------------------------------------
# 16. Keywords
# ---------------------------------------------------------------------------


def test_keyword_extraction_returns_terms_from_the_document(trained) -> None:
    vectorizer, classifier, _train, _test = trained
    text = "Production database credentials are being shared in a public chat channel."

    predicted = classifier.predict(vectorizer.transform([text]))[0]
    keywords = extract_keywords(
        text, vectorizer=vectorizer, classifier=classifier, predicted_class=predicted
    )

    assert keywords
    assert len(keywords) <= 6

    # Every keyword must be built from words the document actually contains.
    # Substring matching would be wrong: a bigram joins consecutive *tokens*,
    # and the token pattern drops single-letter words, so "shared in a public
    # channel" legitimately yields the bigram "in public".
    document_words = set(normalise_text(text).replace(".", "").split())
    for keyword in keywords:
        for word in keyword.split():
            assert word in document_words, f"{keyword!r} uses a word not in the document"


def test_keyword_extraction_is_deterministic(trained) -> None:
    vectorizer, classifier, _train, _test = trained
    text = "The nightly export job has been failing silently for two weeks."
    predicted = classifier.predict(vectorizer.transform([text]))[0]

    first = extract_keywords(
        text, vectorizer=vectorizer, classifier=classifier, predicted_class=predicted
    )
    second = extract_keywords(
        text, vectorizer=vectorizer, classifier=classifier, predicted_class=predicted
    )

    assert first == second


def test_keyword_extraction_handles_unknown_vocabulary(trained) -> None:
    """An honest empty list beats an invented keyword."""
    vectorizer, classifier, _train, _test = trained

    keywords = extract_keywords(
        "zzzqqq xxxyyy wwwvvv",
        vectorizer=vectorizer,
        classifier=classifier,
        predicted_class=str(classifier.classes_[0]),
    )

    assert keywords == []


def test_keyword_extraction_rejects_an_unknown_class(trained) -> None:
    vectorizer, classifier, _train, _test = trained

    assert (
        extract_keywords(
            "Some report text here.",
            vectorizer=vectorizer,
            classifier=classifier,
            predicted_class="NOT_A_CATEGORY",
        )
        == []
    )


def test_per_class_top_features_covers_every_class(trained) -> None:
    vectorizer, classifier, _train, _test = trained

    top = top_features_per_class(vectorizer=vectorizer, classifier=classifier, top_k=5)

    assert set(top) == set(classifier.classes_)
    assert all(len(terms) == 5 for terms in top.values())


# ---------------------------------------------------------------------------
# 17-18. Priority heuristic
# ---------------------------------------------------------------------------


def test_priority_is_deterministic() -> None:
    text = "Urgent: payroll data is publicly accessible and this is ongoing every day."

    results = [suggest_priority(text, "SECURITY") for _ in range(10)]

    assert len({(r.priority, r.score, tuple(r.signals)) for r in results}) == 1


@pytest.mark.parametrize(
    "text,category",
    [
        ("The canteen supplier was changed without consultation.", "OTHER"),
        ("The nightly export produces totals that do not match.", "TECHNICAL"),
        ("Credentials are in a public channel.", "SECURITY"),
        ("A manager threatened staff repeatedly; I have screenshots.", "HARASSMENT"),
        ("", "OTHER"),
        ("x" * 5000, "SECURITY"),
        ("!!!???", None),
    ],
)
def test_priority_is_always_one_of_the_official_levels(text: str, category) -> None:
    suggestion = suggest_priority(text, category)

    assert suggestion.priority in PRIORITY_LEVELS


def test_priority_rises_with_severity_signals() -> None:
    mild = suggest_priority("The staff handbook has not been updated.", "OTHER")
    severe = suggest_priority(
        "Urgent: customer data is publicly accessible right now, this is ongoing, "
        "and it affects all customers.",
        "SECURITY",
    )

    assert mild.priority == "LOW"
    assert severe.priority == "CRITICAL"
    assert severe.score > mild.score


def test_critical_needs_more_than_one_signal() -> None:
    """One emotive word must not push a report to the top of the queue."""
    suggestion = suggest_priority("There was a threat.", "SECURITY")

    assert suggestion.priority != "CRITICAL"
    assert len(suggestion.signals) < 2


def test_priority_explains_itself() -> None:
    suggestion = suggest_priority(
        "Passwords are in a public channel and this is ongoing.", "SECURITY"
    )

    assert suggestion.signals
    assert "credential_or_data_exposure" in suggestion.signals
    assert suggestion.priority in suggestion.explain()
    for signal in suggestion.signals:
        assert signal in suggestion.explain()


def test_priority_is_not_described_as_a_trained_model() -> None:
    from ml.src.priority import describe as describe_priority

    described = describe_priority()

    assert described["is_trained_model"] is False
    assert described["type"] == "deterministic_rule_heuristic"
    assert "heuristic" in described["version"]
    assert described["version"] == PRIORITY_HEURISTIC_VERSION


def test_priority_returns_a_frozen_suggestion() -> None:
    suggestion = suggest_priority("Some text about a password leak.", "SECURITY")

    assert isinstance(suggestion, PrioritySuggestion)
    with pytest.raises(AttributeError):
        suggestion.priority = "CRITICAL"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 11-15, 19-21. The artifact
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory) -> tuple[Path, dict]:
    """Run the real training entrypoint into a temporary directory."""
    root = tmp_path_factory.mktemp("ml-artifacts")
    artifacts = root / "artifacts"
    reports = root / "reports"

    exit_code = train_main(
        [
            "--dataset",
            str(DEFAULT_DATASET),
            "--artifacts",
            str(artifacts),
            "--reports",
            str(reports),
        ]
    )
    assert exit_code == 0

    metadata_path = artifacts / CATEGORY_MODEL_VERSION / "metadata" / "model-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return artifacts / CATEGORY_MODEL_VERSION, metadata


def test_the_artifact_is_written(trained_artifact) -> None:
    artifact_dir, _metadata = trained_artifact

    assert (artifact_dir / "category" / "model.joblib").exists()
    assert (artifact_dir / "metadata" / "model-metadata.json").exists()


def test_the_artifact_loads_and_predicts(trained_artifact) -> None:
    artifact_dir, _metadata = trained_artifact

    bundle = joblib.load(artifact_dir / "category" / "model.joblib")

    assert set(bundle) == {"vectorizer", "classifier", "labels", "model_version"}
    assert bundle["model_version"] == CATEGORY_MODEL_VERSION

    text = "Credentials for the production database are in a public chat channel."
    probabilities = bundle["classifier"].predict_proba(bundle["vectorizer"].transform([text]))[0]

    assert bundle["classifier"].predict(bundle["vectorizer"].transform([text]))[0] in (
        OFFICIAL_CATEGORIES
    )
    assert abs(probabilities.sum() - 1.0) < 1e-6


def test_the_artifact_is_small(trained_artifact) -> None:
    """A model that trains in a second should not be megabytes."""
    artifact_dir, _metadata = trained_artifact

    size_kb = (artifact_dir / "category" / "model.joblib").stat().st_size / 1024

    assert size_kb < 2_048, f"artifact is {size_kb:.0f} KB"


def test_the_metadata_records_the_model_version(trained_artifact) -> None:
    _artifact_dir, metadata = trained_artifact

    assert metadata["model_version"] == CATEGORY_MODEL_VERSION
    assert metadata["model_version"] != "latest"
    assert "v" in metadata["model_version"]


def test_the_metadata_records_a_dataset_fingerprint(trained_artifact) -> None:
    _artifact_dir, metadata = trained_artifact

    fingerprint = metadata["dataset"]["sha256"]

    assert len(fingerprint) == 64
    assert fingerprint == dataset_fingerprint(DEFAULT_DATASET)


def test_the_metadata_records_everything_needed_to_reproduce(trained_artifact) -> None:
    _artifact_dir, metadata = trained_artifact

    assert metadata["random_seed"] == RANDOM_SEED
    assert metadata["trained_at_utc"]
    assert metadata["features"]["ngram_range"] == [1, 2]
    assert metadata["classifier"]["type"] == "LogisticRegression"
    assert metadata["environment"]["scikit_learn"]
    assert metadata["classifier"]["classes"]
    assert metadata["metrics"]["test"]["macro_f1"] > 0


def test_the_metadata_marks_the_data_as_synthetic(trained_artifact) -> None:
    """A reader must never mistake these numbers for real-world performance."""
    _artifact_dir, metadata = trained_artifact

    assert metadata["dataset"]["synthetic"] is True
    assert "not real-world performance" in metadata["dataset"]["synthetic_warning"]


def test_the_metadata_marks_the_model_advisory(trained_artifact) -> None:
    _artifact_dir, metadata = trained_artifact

    assert metadata["advisory_only"] is True
    assert metadata["authoritative"] is False


def test_no_raw_report_text_is_stored_in_the_metadata(trained_artifact) -> None:
    """The metadata identifies the data by hash; it does not reproduce it."""
    _artifact_dir, metadata = trained_artifact
    serialised = json.dumps(metadata)

    descriptions = pd.read_csv(DEFAULT_DATASET)["description"].tolist()
    for description in descriptions:
        assert description not in serialised
        assert description[:40] not in serialised

    assert metadata["artifact"]["contains_training_text"] is False


def test_no_secrets_are_written_to_the_artifact(trained_artifact) -> None:
    artifact_dir, metadata = trained_artifact
    serialised = json.dumps(metadata).lower()

    for marker in (
        "password",
        "secret",
        "pepper",
        "jwt",
        "postgresql://",
        "postgres",
        "database_url",
        "authorization",
        "bearer",
        "$2b$",
        "case_code",
    ):
        assert marker not in serialised, f"metadata mentions {marker!r}"

    blob = (artifact_dir / "category" / "model.joblib").read_bytes()
    from dotenv import dotenv_values

    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for key, value in dotenv_values(env_file).items():
            if (
                value
                and len(value) >= 16
                and any(m in key for m in ("SECRET", "PEPPER", "PASSWORD"))
            ):
                assert value.encode() not in blob, f"{key} found in the model artifact"


def test_an_evaluation_report_is_generated(trained_artifact, tmp_path_factory) -> None:
    artifact_dir, _metadata = trained_artifact
    report_path = artifact_dir.parent.parent / "reports" / "model-evaluation.md"

    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    for expected in ("Macro F1", "Confusion matrix", "baseline", "Limitations"):
        assert expected.lower() in text.lower()
    assert "synthetic" in text.lower()


# ---------------------------------------------------------------------------
# 19. Independence from the application
# ---------------------------------------------------------------------------


def test_the_pipeline_imports_nothing_from_the_application() -> None:
    """Training must run from a clean checkout with no app dependencies."""
    import ast

    offenders: list[str] = []
    for path in (PROJECT_ROOT / "ml").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("app"):
                offenders.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.Import):
                offenders += [
                    f"{path.name}:{node.lineno}"
                    for alias in node.names
                    if alias.name.startswith("app.")
                ]

    assert offenders == [], f"ml/ imports from app/: {offenders}"


def test_training_runs_without_fastapi_importable() -> None:
    """A subprocess that cannot import FastAPI must still train.

    Proves the pipeline has no hidden dependency on the web application.
    """
    script = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name.split('.')[0] in {'fastapi', 'starlette', 'uvicorn'}:\n"
        "            raise ImportError(f'blocked: {name}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from ml.src.train_category import build_classifier\n"
        "from ml.src.data import load_dataset, validate\n"
        "from ml.src.priority import suggest_priority\n"
        "print('OK', suggest_priority('a password leak', 'SECURITY').priority)\n"
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_the_label_sets_match_the_application() -> None:
    """The one place ml/ and app/ duplicate a definition, pinned.

    ``ml/src/config.py`` restates the categories rather than importing them, so
    training needs no application dependencies. This is what stops the two
    drifting apart.
    """
    from app.models.enums import ReportCategory, TriagePriority

    assert set(OFFICIAL_CATEGORIES) == {member.value for member in ReportCategory}
    assert set(PRIORITY_LEVELS) == {member.value for member in TriagePriority}


def test_only_the_ml_boundary_imports_the_pipeline() -> None:
    """Phase 7 lets ``app/ml/`` reuse the inference helpers — and only it.

    Runtime inference must use the *same* keyword extraction, priority
    heuristic and text normalisation that Phase 6 evaluated, so ``app/ml/``
    imports them. Nothing else under ``app/`` may: that boundary is what keeps
    the scientific stack out of the rest of the application.
    """
    import ast

    offenders: list[str] = []
    for path in (PROJECT_ROOT / "app").rglob("*.py"):
        if path.parent.name == "ml":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ml."):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")

    assert offenders == [], f"ml/ imported outside app/ml/: {offenders}"


def test_the_application_never_imports_training_code() -> None:
    """Inference reuses the helpers; it must not pull in the trainer.

    ``train_category``, ``generate_synthetic`` and ``data`` import pandas and
    scikit-learn's model-selection machinery, and none of them has any
    business running inside a web request.
    """
    import ast

    forbidden = {"ml.src.train_category", "ml.src.generate_synthetic", "ml.src.data"}
    offenders: list[str] = []

    for path in (PROJECT_ROOT / "app").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "") in forbidden:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
            if isinstance(node, ast.Import):
                offenders += [
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
                    for alias in node.names
                    if alias.name in forbidden
                ]

    assert offenders == [], f"training code imported by the application: {offenders}"


def test_the_scientific_stack_stays_inside_the_ml_boundary() -> None:
    """The API must not need sklearn or pandas to serve a health check.

    ``joblib`` and ``numpy`` are permitted inside ``app/ml/`` alone — that is
    how the artifact is read — and joblib is imported lazily even there.
    """
    import ast

    heavy = {"sklearn", "pandas", "scipy", "joblib", "numpy"}
    offenders: list[str] = []

    for path in (PROJECT_ROOT / "app").rglob("*.py"):
        inside_boundary = path.parent.name == "ml"
        if inside_boundary:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]

            offenders += [
                f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} {name}"
                for name in names
                if name in heavy
            ]

    assert offenders == [], f"scientific stack imported outside app/ml/: {offenders}"
