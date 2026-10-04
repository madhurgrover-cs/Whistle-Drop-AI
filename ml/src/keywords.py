"""Model-associated keyword extraction.

What these keywords are
-----------------------
For a document and a predicted class, the terms with the largest positive
contribution to that class's score — that is, ``tfidf_value * coefficient``,
restricted to terms actually present in the document.

For a linear model this is exact, not an approximation: a logistic regression's
score for a class *is* the sum of those products plus an intercept, so the
ranking says precisely which terms moved the decision.

What they are not
-----------------
**Not causal explanations, and not reasons.** They describe how this model
weighs words in this vocabulary, nothing more. A term can rank highly because
it genuinely carries the signal, or because it happens to correlate with the
class in the training data — the arithmetic cannot tell those apart, and
neither can a reader of the output.

The wording used everywhere downstream is "model-associated keywords" or
"important model features". Never "reasons", never "because".
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - annotations only
    # Imported for typing alone, so this module can be imported by the API
    # without scikit-learn being present. The objects themselves arrive
    # already fitted, inside the artifact.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

DEFAULT_TOP_K = 6


def extract_keywords(
    text: str,
    *,
    vectorizer: "TfidfVectorizer",
    classifier: "LogisticRegression",
    predicted_class: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[str]:
    """The terms in ``text`` that most pushed the model towards ``predicted_class``.

    Returns at most ``top_k`` terms, most influential first, and only terms
    with a *positive* contribution — a word that argued against the predicted
    class is not a keyword for it.

    Returns an empty list when the document shares no vocabulary with the
    model, which is the honest answer rather than an invented one.
    """
    contributions = keyword_contributions(
        text,
        vectorizer=vectorizer,
        classifier=classifier,
        predicted_class=predicted_class,
        top_k=top_k,
    )
    return [term for term, _ in contributions]


def keyword_contributions(
    text: str,
    *,
    vectorizer: "TfidfVectorizer",
    classifier: "LogisticRegression",
    predicted_class: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[str, float]]:
    """As :func:`extract_keywords`, but with each term's contribution."""
    classes = list(classifier.classes_)
    if predicted_class not in classes:
        return []

    row = vectorizer.transform([text])
    if row.nnz == 0:
        return []

    class_index = classes.index(predicted_class)
    coefficients = classifier.coef_[class_index]
    feature_names = vectorizer.get_feature_names_out()

    present = row.indices
    values = row.data
    contributions = values * coefficients[present]

    positive = [
        (str(feature_names[feature]), float(contribution))
        for feature, contribution in zip(present, contributions, strict=True)
        if contribution > 0
    ]
    positive.sort(key=lambda pair: pair[1], reverse=True)
    return positive[:top_k]


def top_features_per_class(
    *,
    vectorizer: "TfidfVectorizer",
    classifier: "LogisticRegression",
    top_k: int = 12,
) -> dict[str, list[str]]:
    """The highest-weighted terms for each class, across the whole model.

    A global view for the evaluation report — useful for spotting a model that
    has latched onto an artefact of the data rather than anything meaningful.
    """
    feature_names = vectorizer.get_feature_names_out()
    result: dict[str, list[str]] = {}

    for index, label in enumerate(classifier.classes_):
        coefficients = classifier.coef_[index]
        top = np.argsort(coefficients)[::-1][:top_k]
        result[str(label)] = [str(feature_names[i]) for i in top]

    return result
