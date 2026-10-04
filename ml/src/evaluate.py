"""Metrics, the baseline comparison, and confusion-matrix rendering.

Accuracy alone is not reported anywhere. On a corpus where the largest class
holds 29% of rows, accuracy flatters any model that leans towards it, and the
question that matters — does this beat simply guessing the most common
category — is answered by macro F1 against an explicit baseline.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class Metrics:
    """Everything measured on one split."""

    split: str
    n_samples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "n_samples": self.n_samples,
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "per_class": self.per_class,
            "confusion_matrix": self.confusion,
            "labels": self.labels,
        }


def evaluate(y_true, y_pred, *, labels: list[str], split: str) -> Metrics:
    """Compute the full metric set for one split.

    ``zero_division=0`` throughout: a class the model never predicts has a
    precision of zero, not an error and not a silently dropped row.
    """
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class = {
        label: {
            "precision": round(float(per_p[index]), 4),
            "recall": round(float(per_r[index]), 4),
            "f1": round(float(per_f1[index]), 4),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }

    return Metrics(
        split=split,
        n_samples=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        weighted_f1=float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        per_class=per_class,
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        labels=list(labels),
    )


def majority_baseline(X_train, y_train, X_eval, y_eval, *, labels: list[str]) -> Metrics:
    """Score a classifier that always predicts the most common training class.

    The number every trained model must beat. A model that cannot is not
    learning anything from the text, whatever its accuracy looks like.
    """
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    return evaluate(y_eval, dummy.predict(X_eval), labels=labels, split="baseline")


def render_confusion(metrics: Metrics) -> str:
    """The confusion matrix as a readable text table, rows = true labels."""
    labels = metrics.labels
    width = max(len(label) for label in labels) + 2
    header = " " * (width + 2) + "".join(f"{label[:9]:>11}" for label in labels)
    lines = [
        "rows = actual, columns = predicted",
        header,
    ]
    for index, label in enumerate(labels):
        row = "".join(f"{count:>11}" for count in metrics.confusion[index])
        lines.append(f"  {label:<{width}}{row}")
    return "\n".join(lines)


def render_per_class(metrics: Metrics) -> str:
    """Per-class metrics as a markdown table."""
    lines = [
        "| Category | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in metrics.labels:
        scores = metrics.per_class[label]
        lines.append(
            f"| {label} | {scores['precision']:.3f} | {scores['recall']:.3f} "
            f"| {scores['f1']:.3f} | {scores['support']} |"
        )
    return "\n".join(lines)


def sklearn_report(y_true, y_pred, labels: list[str]) -> str:
    """scikit-learn's own text report, kept for cross-checking our numbers."""
    return classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=3)


def probabilities_are_valid(probabilities: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Whether every row is a proper probability distribution."""
    if probabilities.ndim != 2:
        return False
    within_range = bool(np.all(probabilities >= 0) and np.all(probabilities <= 1))
    sums_to_one = bool(np.all(np.abs(probabilities.sum(axis=1) - 1.0) < tolerance))
    return within_range and sums_to_one
