"""Loading, validation, leakage checks and splitting.

Nothing here silently drops a record. Every problem found is counted,
described and returned in a :class:`ValidationReport`; the caller decides what
to exclude, and the exclusion is recorded in the evaluation report. A pipeline
that quietly discards its awkward rows produces metrics nobody can interpret.
"""

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.src.config import (
    MAX_DESCRIPTION_LENGTH,
    MIN_DESCRIPTION_LENGTH,
    OFFICIAL_CATEGORIES,
    RANDOM_SEED,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from ml.src.text import normalise_text

__all__ = [  # noqa: RUF022 - grouped by role, not alphabetised
    "normalise_text",
    "ValidationReport",
    "FORBIDDEN_FEATURE_COLUMNS",
    "check_split_leakage",
    "dataset_fingerprint",
    "load_dataset",
    "split",
    "validate",
]

TEXT_COLUMN = "description"
LABEL_COLUMN = "category"
REQUIRED_COLUMNS = (TEXT_COLUMN, LABEL_COLUMN)

#: Columns that must never become features, whatever a dataset file contains.
#: Each either *is* the target, is derived from it, or identifies the record —
#: and every one of them would leak.
FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "category",  # the target itself
        "status",  # set by a moderator after reading the report
        "moderator_id",
        "moderator_username",
        "id",
        "report_id",
        "case_code",
        "case_code_hash",
        "created_at",
        "updated_at",
        "suggested_category",
        "suggested_priority",
        "priority",
    }
)


@dataclass
class ValidationReport:
    """What was found in the dataset, and what was excluded from training."""

    path: str
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    issues: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)
    conflicting_label_groups: int = 0
    duplicate_descriptions: int = 0
    text_length: dict[str, float] = field(default_factory=dict)

    @property
    def is_trainable(self) -> bool:
        """Whether enough usable data survived to attempt training at all."""
        return (
            self.valid_records >= 50
            and len(self.class_counts) >= 2
            and min(self.class_counts.values(), default=0) >= 5
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "class_counts": self.class_counts,
            "issues": self.issues,
            "conflicting_label_groups": self.conflicting_label_groups,
            "duplicate_descriptions": self.duplicate_descriptions,
            "text_length": self.text_length,
            "is_trainable": self.is_trainable,
        }

    def summary(self) -> str:
        lines = [
            f"Dataset:           {self.path}",
            f"Total records:     {self.total_records}",
            f"Valid records:     {self.valid_records}",
            f"Invalid records:   {self.invalid_records}",
            f"Duplicate texts:   {self.duplicate_descriptions}",
            f"Conflicting texts: {self.conflicting_label_groups} "
            "(identical text, different labels)",
            "Class counts:",
        ]
        for label, count in sorted(self.class_counts.items(), key=lambda kv: -kv[1]):
            share = count / self.valid_records * 100 if self.valid_records else 0
            lines.append(f"  {label:12} {count:5}  {share:5.1f}%")
        if self.issues:
            lines.append("Issues:")
            for issue, count in sorted(self.issues.items()):
                lines.append(f"  {issue:34} {count}")
        return "\n".join(lines)


def dataset_fingerprint(path: Path) -> str:
    """SHA-256 of the dataset file.

    Recorded in the artifact metadata so that a model can always be traced to
    the exact bytes it was trained on. A hash, not a copy: the fingerprint
    identifies the data without reproducing any of it.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    """Read the dataset and drop any column that must never be a feature.

    The drop is loud, not silent: the caller sees which columns were removed
    in the validation report's issues.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {', '.join(missing)}. "
            f"Expected at least {', '.join(REQUIRED_COLUMNS)}."
        )

    return frame


def validate(frame: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, ValidationReport]:
    """Check the dataset and return the usable rows plus a report.

    Rows are excluded only for reasons that make them unusable — empty text, a
    label outside the official set, text too short or too long to be a real
    report. Everything excluded is counted.
    """
    report = ValidationReport(path=str(path), total_records=len(frame))
    issues: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}

    def note(issue: str, hint: str | None = None) -> None:
        issues[issue] += 1
        if hint and len(examples.setdefault(issue, [])) < 3:
            examples[issue].append(hint)

    # Columns that would leak, if a future dataset file carries them.
    leaking = sorted(set(frame.columns) & FORBIDDEN_FEATURE_COLUMNS - {LABEL_COLUMN})
    for column in leaking:
        note("dropped_leaking_column", column)
    working = frame.drop(columns=leaking, errors="ignore").copy()

    working["_text"] = working[TEXT_COLUMN].map(normalise_text)
    working["_label"] = working[LABEL_COLUMN].fillna("").str.strip().str.upper()

    keep = pd.Series(True, index=working.index)

    empty_text = working["_text"].str.len() == 0
    for _ in range(int(empty_text.sum())):
        note("empty_or_missing_text")
    keep &= ~empty_text

    too_short = (~empty_text) & (working["_text"].str.len() < MIN_DESCRIPTION_LENGTH)
    for _ in range(int(too_short.sum())):
        note("text_below_minimum_length")
    keep &= ~too_short

    too_long = working["_text"].str.len() > MAX_DESCRIPTION_LENGTH
    for _ in range(int(too_long.sum())):
        note("text_above_maximum_length")
    keep &= ~too_long

    missing_label = working["_label"].str.len() == 0
    for _ in range(int(missing_label.sum())):
        note("missing_label")
    keep &= ~missing_label

    unknown_label = (~missing_label) & (~working["_label"].isin(OFFICIAL_CATEGORIES))
    for label in working.loc[unknown_label, "_label"].unique()[:3]:
        examples.setdefault("label_not_in_official_categories", []).append(str(label))
    issues["label_not_in_official_categories"] += int(unknown_label.sum())
    keep &= ~unknown_label

    valid = working[keep].copy()

    # Duplicates and conflicts, measured on the surviving rows.
    duplicate_mask = valid.duplicated(subset=["_text"], keep=False)
    report.duplicate_descriptions = int(duplicate_mask.sum())

    labels_per_text = valid.groupby("_text")["_label"].nunique()
    conflicting_texts = set(labels_per_text[labels_per_text > 1].index)
    report.conflicting_label_groups = len(conflicting_texts)
    if conflicting_texts:
        issues["identical_text_with_different_labels"] = int(
            valid["_text"].isin(conflicting_texts).sum()
        )
        for text in list(conflicting_texts)[:3]:
            examples.setdefault("identical_text_with_different_labels", []).append(text[:70] + "…")

    report.valid_records = len(valid)
    report.invalid_records = report.total_records - report.valid_records
    report.class_counts = {
        label: int(count) for label, count in valid["_label"].value_counts().items()
    }
    report.issues = {issue: int(count) for issue, count in issues.items() if count}
    report.examples = examples

    lengths = valid["_text"].str.len()
    if len(lengths):
        report.text_length = {
            "min": float(lengths.min()),
            "median": float(lengths.median()),
            "mean": round(float(lengths.mean()), 1),
            "max": float(lengths.max()),
        }

    return valid, report


def check_split_leakage(train_texts: pd.Series, *others: pd.Series) -> dict[str, int]:
    """Count normalised texts that appear in training *and* a held-out split.

    This is the leakage that inflates a score most convincingly, because
    nothing about the metrics looks wrong: the model is simply being asked to
    recall text it has already seen. :func:`split` prevents it by construction;
    this is the assertion that the prevention worked.
    """
    train_set = set(train_texts)
    return {
        f"overlap_with_split_{index + 1}": len(train_set & set(other))
        for index, other in enumerate(others)
    }


def split(
    frame: pd.DataFrame,
    *,
    seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / validation / test, stratified and leak-free.

    **Grouped by normalised text, not by row.** A plain stratified split would
    put two copies of one duplicated report on opposite sides, and the model
    would then be scored on text it had memorised. Every copy of a given text
    therefore lands in exactly one split, which is why the split proportions
    come out close to but not exactly at the requested sizes.

    Three splits rather than two: any choice between model variants — class
    weighting, say — is made against *validation*. The test set is not looked
    at until the final evaluation, and is scored once.
    """
    # One row per distinct text, labelled by its most common label, so that
    # stratification is over groups rather than rows.
    groups = (
        frame.groupby("_text")["_label"]
        .agg(lambda labels: labels.value_counts().idxmax())
        .reset_index()
    )

    # Stratify only where every class has enough groups for it to be possible.
    counts = groups["_label"].value_counts()
    stratify_all = counts.min() >= 3

    train_groups, holdout_groups = train_test_split(
        groups,
        test_size=test_size + validation_size,
        random_state=seed,
        stratify=groups["_label"] if stratify_all else None,
        shuffle=True,
    )

    holdout_counts = holdout_groups["_label"].value_counts()
    relative_test = test_size / (test_size + validation_size)
    validation_groups, test_groups = train_test_split(
        holdout_groups,
        test_size=relative_test,
        random_state=seed,
        stratify=holdout_groups["_label"] if holdout_counts.min() >= 2 else None,
        shuffle=True,
    )

    def rows_for(group_frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["_text"].isin(set(group_frame["_text"]))].copy()

    return rows_for(train_groups), rows_for(validation_groups), rows_for(test_groups)
