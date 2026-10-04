"""Text normalisation, in a module with no heavy dependencies.

Split out of ``ml/src/data.py`` so that runtime inference can reuse the *exact*
normalisation used in training without importing pandas and scikit-learn's
model-selection machinery. ``data.py`` re-exports it, so nothing that already
imported it from there has to change.

One implementation, used by the vectoriser at training time, by the priority
heuristic, and by the API at inference time. Two would eventually diverge, and
a divergence here means the model sees text that differs from what it was
fitted on.
"""

import unicodedata


def normalise_text(text: str) -> str:
    """Normalise for comparison and for vectorising.

    Deliberately conservative: NFC so that two spellings of one accented
    character compare equal, and whitespace collapsed so that a line-wrapped
    copy of a report is recognised as the same text. Case is lowered, which
    the vectoriser would do anyway.

    Nothing else. No stemming, no stop-word stripping, no punctuation removal.
    Report text is evidence, and the signal that separates "a manager shouts at
    junior staff" from "the export job fails" lives in ordinary words that an
    aggressive filter would throw away.
    """
    if not isinstance(text, str):
        return ""
    collapsed = " ".join(unicodedata.normalize("NFC", text).split())
    return collapsed.strip().lower()
