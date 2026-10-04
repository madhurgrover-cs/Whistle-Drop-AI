"""Train the advisory category classifier.

    python -m ml.src.train_category

One command: validate, split, choose between two variants on *validation*,
train, evaluate once on the untouched test set, write the artifact, the
metadata and the evaluation report. No FastAPI, no database, no settings —
this runs from a checkout and a CSV.

The test set is read exactly once, at the end. The only model choice made
anywhere is whether to use balanced class weights, and that is decided on the
validation split; the test numbers are therefore an estimate of performance on
unseen data rather than the best of several attempts.
"""

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

from ml.src import evaluate as evaluation
from ml.src.config import (
    ARTIFACTS_DIR,
    CATEGORY_MODEL_VERSION,
    DEFAULT_DATASET,
    OFFICIAL_CATEGORIES,
    RANDOM_SEED,
    REPORTS_DIR,
)
from ml.src.data import (
    check_split_leakage,
    dataset_fingerprint,
    load_dataset,
    split,
    validate,
)
from ml.src.features import build_vectorizer
from ml.src.features import describe as describe_features
from ml.src.keywords import top_features_per_class
from ml.src.priority import describe as describe_priority

# Written into the metadata so a reader of an artifact knows whether its
# numbers describe generated scaffolding or real reports.
SYNTHETIC_MARKER = "synthetic_reports.csv"


def build_classifier(*, class_weight: str | None, seed: int) -> LogisticRegression:
    """The category model.

    **LogisticRegression**, multinomial by default in scikit-learn 1.7 — one
    softmax over the five classes rather than five independent one-vs-rest
    fits, so the probabilities are a proper distribution and the confidence
    reported to a moderator means what it appears to mean.

    **solver="lbfgs"** — the default, and appropriate for a dense-enough
    multinomial problem of this size. ``liblinear`` cannot do multinomial at
    all; ``saga`` is for far larger corpora.

    **C=1.0** — scikit-learn's default regularisation, left alone deliberately.
    Tuning it against this dataset would be tuning against a generator.

    **max_iter=2000** — comfortably above what convergence needs here, so the
    run never ends on a convergence warning that would make the result depend
    on where the optimiser happened to stop.
    """
    return LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=2_000,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ml.src.train_category",
        description="Validate, train, evaluate and save the category model.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--artifacts", type=Path, default=ARTIFACTS_DIR, help="Artifact root directory."
    )
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        print(
            "Generate the synthetic development set with:\n    python -m ml.src.generate_synthetic",
            file=sys.stderr,
        )
        return 2

    # --- 1. Validate -------------------------------------------------------
    print("== Validating ==")
    frame = load_dataset(args.dataset)
    valid, report = validate(frame, args.dataset)
    print(report.summary())

    if not report.is_trainable:
        print(
            "\nNot enough usable data to train. Need at least 50 valid records, "
            "two or more classes, and at least 5 records in the smallest class.",
            file=sys.stderr,
        )
        return 3

    fingerprint = dataset_fingerprint(args.dataset)
    is_synthetic = SYNTHETIC_MARKER in args.dataset.name

    # --- 2. Split ----------------------------------------------------------
    print("\n== Splitting ==")
    train, validation, test = split(valid, seed=args.seed)
    leakage = check_split_leakage(train["_text"], validation["_text"], test["_text"])
    print(
        f"train={len(train)}  validation={len(validation)}  test={len(test)}  "
        f"(grouped by text, stratified)"
    )
    print(f"text overlap train/validation and train/test: {list(leakage.values())}")
    if any(leakage.values()):
        print("Split leakage detected. Refusing to train.", file=sys.stderr)
        return 4

    labels = [label for label in OFFICIAL_CATEGORIES if label in set(valid["_label"])]

    # --- 3. Features -------------------------------------------------------
    print("\n== Features ==")
    vectorizer = build_vectorizer(len(train))
    X_train = vectorizer.fit_transform(train["_text"])
    X_validation = vectorizer.transform(validation["_text"])
    X_test = vectorizer.transform(test["_text"])
    print(f"TF-IDF matrix: {X_train.shape[0]} x {X_train.shape[1]} features")

    y_train = train["_label"].to_numpy()
    y_validation = validation["_label"].to_numpy()
    y_test = test["_label"].to_numpy()

    # --- 4. Choose a variant, on VALIDATION only ---------------------------
    print("\n== Selecting class weighting (validation split) ==")
    candidates: dict[str, tuple[LogisticRegression, evaluation.Metrics]] = {}
    for name, class_weight in (("unweighted", None), ("balanced", "balanced")):
        model = build_classifier(class_weight=class_weight, seed=args.seed)
        model.fit(X_train, y_train)
        metrics = evaluation.evaluate(
            y_validation, model.predict(X_validation), labels=labels, split="validation"
        )
        candidates[name] = (model, metrics)
        print(f"  {name:11} macro F1 = {metrics.macro_f1:.4f}   accuracy = {metrics.accuracy:.4f}")

    chosen_name = max(candidates, key=lambda name: candidates[name][1].macro_f1)
    classifier, validation_metrics = candidates[chosen_name]
    class_weight = None if chosen_name == "unweighted" else "balanced"
    print(f"  -> chose {chosen_name!r} on validation macro F1")

    # --- 5. Baseline and final evaluation, on the untouched test set -------
    print("\n== Final evaluation (test split, read once) ==")
    baseline = evaluation.majority_baseline(X_train, y_train, X_test, y_test, labels=labels)
    test_predictions = classifier.predict(X_test)
    test_metrics = evaluation.evaluate(y_test, test_predictions, labels=labels, split="test")

    probabilities = classifier.predict_proba(X_test)
    probabilities_valid = evaluation.probabilities_are_valid(probabilities)

    print(f"  baseline  macro F1 = {baseline.macro_f1:.4f}  accuracy = {baseline.accuracy:.4f}")
    print(
        f"  model     macro F1 = {test_metrics.macro_f1:.4f}  "
        f"accuracy = {test_metrics.accuracy:.4f}"
    )
    print(f"  probabilities form valid distributions: {probabilities_valid}")
    print()
    print(evaluation.render_confusion(test_metrics))

    # --- 6. Save the artifact ---------------------------------------------
    print("\n== Saving ==")
    artifact_dir = args.artifacts / CATEGORY_MODEL_VERSION
    (artifact_dir / "category").mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata").mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "category" / "model.joblib"
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "labels": list(classifier.classes_),
            "model_version": CATEGORY_MODEL_VERSION,
        },
        model_path,
        compress=3,
    )

    metadata = build_metadata(
        dataset_path=args.dataset,
        fingerprint=fingerprint,
        is_synthetic=is_synthetic,
        seed=args.seed,
        validation_report=report,
        split_sizes={
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        leakage=leakage,
        vectorizer=vectorizer,
        classifier=classifier,
        class_weight=class_weight,
        baseline=baseline,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        probabilities_valid=probabilities_valid,
        model_path=model_path,
    )

    metadata_path = artifact_dir / "metadata" / "model-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    size_kb = model_path.stat().st_size / 1024
    print(f"  model:    {model_path}  ({size_kb:.1f} KB)")
    print(f"  metadata: {metadata_path}")

    # --- 7. Reports --------------------------------------------------------
    args.reports.mkdir(parents=True, exist_ok=True)
    evaluation_report_path = args.reports / "model-evaluation.md"
    evaluation_report_path.write_text(
        render_evaluation_report(
            metadata=metadata,
            validation_report=report,
            baseline=baseline,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            keywords=top_features_per_class(vectorizer=vectorizer, classifier=classifier, top_k=10),
            sklearn_text=evaluation.sklearn_report(y_test, test_predictions, labels),
        ),
        encoding="utf-8",
    )
    print(f"  report:   {evaluation_report_path}")

    if is_synthetic:
        print(
            "\nNOTE: trained on SYNTHETIC data. These metrics describe the "
            "generator,\nnot real-world performance. See ml/reports/model-card.md."
        )
    return 0


def build_metadata(
    *,
    dataset_path: Path,
    fingerprint: str,
    is_synthetic: bool,
    seed: int,
    validation_report: Any,
    split_sizes: dict[str, int],
    leakage: dict[str, int],
    vectorizer: Any,
    classifier: LogisticRegression,
    class_weight: str | None,
    baseline: evaluation.Metrics,
    validation_metrics: evaluation.Metrics,
    test_metrics: evaluation.Metrics,
    probabilities_valid: bool,
    model_path: Path,
) -> dict[str, Any]:
    """Assemble the artifact metadata.

    Records the dataset's *name and hash*, never its contents. No report text,
    no vocabulary, no secrets, no environment values — a reader can verify
    which data produced the model without being handed any of it.
    """
    return {
        "model_version": CATEGORY_MODEL_VERSION,
        "model_type": "TfidfVectorizer + LogisticRegression (multinomial)",
        "purpose": "Advisory category suggestion for moderator triage.",
        "advisory_only": True,
        "authoritative": False,
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "random_seed": seed,
        "dataset": {
            "filename": dataset_path.name,
            "sha256": fingerprint,
            "synthetic": is_synthetic,
            "synthetic_warning": (
                "Trained on generated scaffolding. Metrics describe the generator, "
                "not real-world performance."
            )
            if is_synthetic
            else None,
            "total_records": validation_report.total_records,
            "valid_records": validation_report.valid_records,
            "invalid_records": validation_report.invalid_records,
            "class_counts": validation_report.class_counts,
            "duplicate_descriptions": validation_report.duplicate_descriptions,
            "conflicting_label_groups": validation_report.conflicting_label_groups,
        },
        "split": {
            "strategy": "grouped by normalised text, stratified by label",
            "sizes": split_sizes,
            "test_size": test_metrics.n_samples,
            "text_overlap_between_splits": leakage,
        },
        "features": describe_features(vectorizer),
        "classifier": {
            "type": "LogisticRegression",
            "solver": classifier.solver,
            "C": classifier.C,
            "max_iter": classifier.max_iter,
            "class_weight": class_weight,
            "n_iter": int(np.max(classifier.n_iter_)),
            "classes": [str(label) for label in classifier.classes_],
        },
        "priority_heuristic": describe_priority(),
        "metrics": {
            "baseline_majority_class": baseline.to_dict(),
            "validation": validation_metrics.to_dict(),
            "test": test_metrics.to_dict(),
            "probabilities_form_valid_distributions": probabilities_valid,
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "artifact": {
            "path": model_path.name,
            "size_bytes": model_path.stat().st_size,
            "format": "joblib",
            "contains_training_text": False,
        },
    }


def render_evaluation_report(
    *,
    metadata: dict[str, Any],
    validation_report: Any,
    baseline: evaluation.Metrics,
    validation_metrics: evaluation.Metrics,
    test_metrics: evaluation.Metrics,
    keywords: dict[str, list[str]],
    sklearn_text: str,
) -> str:
    """Write ml/reports/model-evaluation.md from this run's actual numbers."""
    dataset = metadata["dataset"]
    synthetic = dataset["synthetic"]

    warning = (
        "> ## ⚠ These numbers describe a generator, not the real world\n>\n"
        "> This model was trained on **synthetic data**: report text assembled\n"
        "> from templates, because no labelled corpus of real whistleblowing\n"
        "> reports exists in this project. Templated text is far easier to\n"
        "> separate than anything a person writes under stress, so the scores\n"
        "> below are close to perfect and **say nothing about how this model\n"
        "> would perform on real reports**.\n>\n"
        "> What they do establish is that the pipeline — validation, leakage\n"
        "> checks, splitting, vectorising, training, evaluation, artifact\n"
        "> writing — runs correctly end to end.\n"
        if synthetic
        else ""
    )

    improvement = test_metrics.macro_f1 - baseline.macro_f1

    lines = [
        "# Category model — evaluation report",
        "",
        "*Generated by `python -m ml.src.train_category`. Every number below "
        "comes from that run; none is written by hand.*",
        "",
        warning,
        "## Dataset",
        "",
        f"- **File**: `{dataset['filename']}`",
        f"- **SHA-256**: `{dataset['sha256']}`",
        f"- **Synthetic**: {'yes — generated scaffolding' if synthetic else 'no'}",
        f"- **Total records**: {dataset['total_records']}",
        f"- **Valid records**: {dataset['valid_records']}",
        f"- **Invalid records**: {dataset['invalid_records']}",
        "",
        "### Class distribution",
        "",
        "| Category | Count | Share |",
        "|---|---:|---:|",
    ]
    total = dataset["valid_records"] or 1
    for label, count in sorted(dataset["class_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {label} | {count} | {count / total * 100:.1f}% |")

    lines += [
        "",
        "### Validation findings",
        "",
        f"- Duplicate descriptions: **{dataset['duplicate_descriptions']}**",
        f"- Identical text with conflicting labels: "
        f"**{dataset['conflicting_label_groups']}** group(s)",
    ]
    if validation_report.issues:
        for issue, count in sorted(validation_report.issues.items()):
            lines.append(f"- `{issue}`: {count}")
    else:
        lines.append("- No other issues found.")

    if validation_report.text_length:
        length = validation_report.text_length
        lines += [
            "",
            f"- Text length (characters): min {length['min']:.0f}, "
            f"median {length['median']:.0f}, mean {length['mean']}, "
            f"max {length['max']:.0f}",
        ]

    split_info = metadata["split"]
    lines += [
        "",
        "## Train / validation / test split",
        "",
        f"- **Strategy**: {split_info['strategy']}",
        f"- **Sizes**: train {split_info['sizes']['train']}, "
        f"validation {split_info['sizes']['validation']}, "
        f"test {split_info['sizes']['test']}",
        f"- **Seed**: {metadata['random_seed']}",
        f"- **Text overlap between splits**: "
        f"{list(split_info['text_overlap_between_splits'].values())} "
        "(zero by construction — splits are grouped by normalised text, so "
        "duplicated reports cannot straddle a boundary)",
        "",
        "Three splits rather than two: the one model choice made anywhere — "
        "whether to use balanced class weights — was decided on **validation**. "
        "The test set was read once, at the end.",
        "",
        "## Feature extraction",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    for key, value in metadata["features"].items():
        lines.append(f"| `{key}` | {value} |")

    classifier_info = metadata["classifier"]
    lines += [
        "",
        "## Model",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    for key, value in classifier_info.items():
        lines.append(f"| `{key}` | {value} |")

    lines += [
        "",
        "## Baseline comparison",
        "",
        "The baseline always predicts the most common training class. A model "
        "that cannot beat it is not learning anything from the text.",
        "",
        "| Metric | Majority baseline | This model | Difference |",
        "|---|---:|---:|---:|",
        f"| Accuracy | {baseline.accuracy:.4f} | {test_metrics.accuracy:.4f} | "
        f"{test_metrics.accuracy - baseline.accuracy:+.4f} |",
        f"| Macro F1 | {baseline.macro_f1:.4f} | {test_metrics.macro_f1:.4f} | "
        f"{improvement:+.4f} |",
        f"| Weighted F1 | {baseline.weighted_f1:.4f} | {test_metrics.weighted_f1:.4f} | "
        f"{test_metrics.weighted_f1 - baseline.weighted_f1:+.4f} |",
        "",
        "## Test metrics",
        "",
        f"- **Samples**: {test_metrics.n_samples}",
        f"- **Accuracy**: {test_metrics.accuracy:.4f}",
        f"- **Macro precision**: {test_metrics.macro_precision:.4f}",
        f"- **Macro recall**: {test_metrics.macro_recall:.4f}",
        f"- **Macro F1**: {test_metrics.macro_f1:.4f}",
        f"- **Weighted F1**: {test_metrics.weighted_f1:.4f}",
        "",
        "Macro F1 is the headline: it weighs every category equally, so a model "
        "that ignores the rarest class cannot hide behind the common ones.",
        "",
        "### Per-class performance",
        "",
        evaluation.render_per_class(test_metrics),
        "",
        "### Confusion matrix",
        "",
        "```",
        evaluation.render_confusion(test_metrics),
        "```",
        "",
        "### scikit-learn's own report (cross-check)",
        "",
        "```",
        sklearn_text.rstrip(),
        "```",
        "",
        "### Validation split, for reference",
        "",
        f"- Macro F1: {validation_metrics.macro_f1:.4f} · "
        f"Accuracy: {validation_metrics.accuracy:.4f}",
        "",
        "## Model-associated keywords",
        "",
        "The highest-weighted terms per class. These describe how this model "
        "weighs this vocabulary — they are **not** causal explanations, and not "
        "reasons a report belongs to a category.",
        "",
    ]
    for label, terms in keywords.items():
        lines.append(f"- **{label}**: {', '.join(f'`{term}`' for term in terms)}")

    lines += [
        "",
        "## Priority",
        "",
        "Priority is **not** a trained model. No priority or severity labels "
        "exist in any dataset here, so a documented, deterministic heuristic is "
        "used instead — see `ml/src/priority.py`. It returns the signals that "
        "fired alongside the level, and it is advisory: it cannot change a "
        "report's category or status.",
        "",
        "## Limitations",
        "",
    ]

    if synthetic:
        lines += [
            "1. **The training data is synthetic.** This is the limitation that "
            "subsumes the rest. Every metric above measures how well the model "
            "recovers the rules of a template generator. Real reports are "
            "misspelled, elliptical, emotional, written in a second language, "
            "and often belong to two categories at once.",
            "2. **The scores are therefore not evidence of usefulness.** They "
            "are evidence the pipeline works.",
        ]
    else:
        lines.append("1. Evaluated on a single held-out split; no cross-validation.")

    lines += [
        f"{3 if synthetic else 2}. **Small vocabulary.** "
        f"{metadata['features']['vocabulary_size']} features learned from "
        f"{split_info['sizes']['train']} training documents. Any real report "
        "using vocabulary outside it will be classified on very little signal.",
        f"{4 if synthetic else 3}. **Class imbalance persists.** The rarest "
        "category has the least training signal, which is exactly where a "
        "moderator most needs help.",
        f"{5 if synthetic else 4}. **Confidence is a model probability, not a "
        "certainty.** Logistic regression is reasonably calibrated but was not "
        "calibration-tested here; a 0.9 does not mean 'right nine times in ten'.",
        f"{6 if synthetic else 5}. **No fairness or bias evaluation.** There is "
        "no demographic data to evaluate against — by design, since the system "
        "collects none — so differential performance across groups of reporters "
        "is unmeasured and unmeasurable here.",
        f"{7 if synthetic else 6}. **English only.**",
        "",
        "## Future improvements",
        "",
        "1. **Replace the synthetic data with real, moderator-labelled reports.** "
        "Nothing else on this list matters until this happens. Phase 7 records "
        "moderator overrides, which is the mechanism that accumulates them.",
        "2. Cross-validation once the dataset is large enough to spare the folds.",
        "3. Calibration analysis, so the confidence shown to moderators means "
        "what it appears to mean.",
        "4. Per-class thresholds, so a low-confidence suggestion can be withheld "
        "rather than shown.",
        "5. Revisit linear TF-IDF only if it demonstrably underperforms on real "
        "data — it is fast, explainable and auditable, which are worth a great "
        "deal in this setting.",
        "",
        "---",
        "",
        f"*Model version `{metadata['model_version']}` · "
        f"trained {metadata['trained_at_utc']} · "
        f"scikit-learn {metadata['environment']['scikit_learn']} · "
        f"seed {metadata['random_seed']}*",
    ]

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
