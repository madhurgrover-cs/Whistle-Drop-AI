"""TF-IDF configuration, chosen for this dataset's actual size.

The numbers below are not copied from a tutorial; each is justified against
the ~950-row, ~140-character corpus this pipeline was built on, and
:func:`build_vectorizer` scales the two that should move with corpus size.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

from ml.src.data import normalise_text


def build_vectorizer(n_documents: int) -> TfidfVectorizer:
    """A vectoriser sized for a corpus of ``n_documents`` documents.

    **ngram_range=(1, 2)** — unigrams plus bigrams, as specified. Bigrams earn
    their place here: "shared drive", "without tender", "access control" and
    "line manager" each carry more signal than either word alone, and the
    vocabulary stays small enough that the extra columns cost nothing.

    **min_df** — 2 for a corpus this size, rising to 3 above 5,000 documents.
    A term appearing once cannot generalise; it can only memorise the document
    it came from. Kept low because the corpus is small and raising it further
    would discard most bigrams.

    **max_df=0.9** — drop terms in more than 90% of documents. With no
    stop-word list (see below) this is what removes "the" and "and", and it
    does so from the data rather than from an English word list that might
    also remove something meaningful.

    **sublinear_tf=True** — use 1 + log(tf) instead of raw counts. A report
    that says "password" eight times is not eight times more about passwords
    than one that says it once, and reports vary a lot in length.

    **max_features=30000** — a ceiling, not a target. This corpus produces far
    fewer; the cap exists so a much larger dataset cannot silently grow the
    artifact without a deliberate change here.

    **No stop-word list and no stemming.** Both are destructive, and this is
    report text. "I was told not to report this" becomes meaningless once the
    function words are stripped, and stemming would merge "reporting" with
    "reporter" — words that mean different things in this domain.
    """
    return TfidfVectorizer(
        preprocessor=normalise_text,
        ngram_range=(1, 2),
        min_df=3 if n_documents > 5_000 else 2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=30_000,
        strip_accents=None,  # normalise_text already applies NFC
        lowercase=False,  # normalise_text already lowers
        # Keeps hyphenated and possessive forms intact. S106 reads the argument
        # name as "token" and calls it a hardcoded password; it is a regex.
        token_pattern=r"(?u)\b\w[\w'-]+\b",  # noqa: S106
    )


def describe(vectorizer: TfidfVectorizer) -> dict[str, object]:
    """The configuration as plain data, for the artifact metadata.

    Records the settings and the resulting vocabulary size — never the
    vocabulary itself, and never any document text.
    """
    return {
        "type": "TfidfVectorizer",
        "ngram_range": list(vectorizer.ngram_range),
        "min_df": vectorizer.min_df,
        "max_df": vectorizer.max_df,
        "sublinear_tf": vectorizer.sublinear_tf,
        "max_features": vectorizer.max_features,
        "token_pattern": vectorizer.token_pattern,
        "stop_words": None,
        "stemming": False,
        "vocabulary_size": (
            len(vectorizer.vocabulary_) if hasattr(vectorizer, "vocabulary_") else None
        ),
    }
