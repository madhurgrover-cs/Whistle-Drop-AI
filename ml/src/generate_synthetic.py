"""Generate the SYNTHETIC development dataset.

    python -m ml.src.generate_synthetic

Read this before you read any metric produced from the output
==============================================================
**This is not real data, and no evaluation number derived from it says
anything about real-world performance.**

No labelled corpus of whistleblowing reports exists in this project, and one
cannot be invented honestly. What this script produces is scaffolding: text
assembled from templates and vocabulary lists so that the *pipeline* —
validation, leakage checks, splitting, vectorising, training, evaluation,
artifact writing — can be built, run and tested end to end.

Templated text is far easier to separate than anything a person writes under
stress. A model trained on it will score close to perfect, and that score
measures the generator, not the model. The evaluation report and the model
card both say so; so does the ``synthetic`` flag written into the artifact
metadata.

What is deliberately *not* clean
--------------------------------
A generator that emitted five tidy, perfectly separable classes would let
every validation and leakage check pass vacuously. So the output contains, on
purpose and in known quantities:

* **class imbalance** — roughly the shape a real queue has, with ``OTHER``
  and ``SECURITY`` common and ``CORRUPTION`` rare;
* **exact duplicate descriptions**, some within a class and some across two,
  so the duplicate and conflicting-label checks have something to find;
* **genuinely ambiguous reports** that borrow vocabulary from two categories,
  so accuracy cannot reach 1.0 and the confusion matrix has structure;
* **vocabulary shared between categories**, so TF-IDF cannot separate on a
  single giveaway term.

Everything is drawn from a seeded :class:`random.Random`, so the same seed
produces byte-identical output.
"""

import argparse
import csv
import random
from pathlib import Path

from ml.src.config import DEFAULT_DATASET, RANDOM_SEED

# --- Vocabulary -------------------------------------------------------------
#
# Written by hand. Deliberately overlapping in places: "access", "data",
# "manager" and "system" appear under more than one category, so the classifier
# has to use combinations rather than a single giveaway word.

SUBJECTS = {
    "SECURITY": [
        "production database credentials",
        "an administrator password",
        "the customer data export",
        "an unencrypted backup",
        "the internal API keys",
        "a shared service account",
        "access logs for the payments system",
        "the signing certificate",
    ],
    "HARASSMENT": [
        "a senior manager",
        "a team lead",
        "someone on the operations floor",
        "a colleague in my department",
        "a supervisor on the night shift",
        "a member of the leadership team",
    ],
    "CORRUPTION": [
        "a supplier contract",
        "the procurement process",
        "an expense claim",
        "a consultancy invoice",
        "the vendor selection",
        "a facilities tender",
    ],
    "TECHNICAL": [
        "the payment reconciliation job",
        "the nightly export",
        "the reporting dashboard",
        "the order processing queue",
        "the customer search page",
        "the invoice generator",
    ],
    "OTHER": [
        "the building access policy",
        "our overtime recording",
        "the recycling contract",
        "the staff handbook",
        "a parking allocation",
        "the canteen supplier",
    ],
}

PREDICATES = {
    "SECURITY": [
        "is being shared in a public chat channel",
        "was posted to a group anyone can join",
        "has been reused across three separate systems",
        "was left readable on a shared drive",
        "is still active for someone who left last year",
        "has not been rotated since the last incident",
    ],
    "HARASSMENT": [
        "repeatedly makes demeaning remarks in team meetings",
        "has been singling out one colleague for months",
        "shouts at junior staff in front of the team",
        "keeps making comments about a colleague's appearance",
        "has threatened people's shifts when they disagree",
        "excludes one person from every meeting deliberately",
    ],
    "CORRUPTION": [
        "was awarded without any tender process",
        "went to a company owned by a relative of the approver",
        "was approved without receipts or any second signature",
        "was signed off twice for the same work",
        "was decided after a weekend trip paid by the supplier",
        "was split into smaller amounts to stay under the approval limit",
    ],
    "TECHNICAL": [
        "has been failing silently every night for two weeks",
        "produces totals that do not match the ledger",
        "times out whenever the queue goes above a thousand items",
        "shows stale numbers after every deployment",
        "drops records without logging anything",
        "returns duplicated rows for some accounts",
    ],
    "OTHER": [
        "does not match what staff were told at the last briefing",
        "seems to be applied differently depending on the team",
        "has not been updated since the merger",
        "is causing confusion across two departments",
        "was changed without anyone being consulted",
        "contradicts the policy published on the intranet",
    ],
}

CONTEXTS = [
    "",
    " This has been going on for several weeks.",
    " I raised it with my line manager and nothing changed.",
    " Several people on the team have noticed the same thing.",
    " I have screenshots but I am not comfortable attaching them.",
    " It started after the reorganisation in the spring.",
    " I am reporting this anonymously because I am worried about retaliation.",
]

OPENERS = [
    "I want to report that ",
    "I need to raise a concern: ",
    "Reporting an issue - ",
    "",
    "Please look into this. ",
]

# Reports that borrow from two categories on purpose. The label is the one a
# moderator would most likely choose, but the text genuinely supports another,
# which is what stops accuracy reaching 1.0 and gives the confusion matrix
# something real to show.
AMBIGUOUS = [
    (
        "SECURITY",
        "A manager asked me to share my login so they could approve their own "
        "expense claim while the approver was away. I did not do it but I think "
        "it has happened before with someone else.",
    ),
    (
        "CORRUPTION",
        "The vendor was given an administrator account on our systems as part of "
        "the contract, and the contract itself was never put out to tender.",
    ),
    (
        "TECHNICAL",
        "The access control list on the reporting dashboard resets every "
        "deployment, so for about an hour anyone in the company can see payroll "
        "figures. I think it is a bug rather than anything deliberate.",
    ),
    (
        "HARASSMENT",
        "My team lead keeps reading other people's messages over their shoulder "
        "and then bringing up what he saw in meetings to embarrass them.",
    ),
    (
        "OTHER",
        "Overtime is being recorded on a spreadsheet that anyone can edit, and "
        "the totals people submit do not match what is approved.",
    ),
    (
        "SECURITY",
        "Someone from the supplier still has a badge for the server room months "
        "after the contract ended, and nobody will say who authorised it.",
    ),
    (
        "TECHNICAL",
        "The invoice generator rounds every line up, which over a year adds up "
        "to a lot of money going to the wrong place. It might be deliberate.",
    ),
    (
        "CORRUPTION",
        "Expenses are approved by whoever is on shift rather than a manager, so "
        "people effectively sign off their own claims.",
    ),
]

# The shape of a real queue rather than a uniform one: most reports are either
# security concerns or things that do not fit a category cleanly.
CLASS_WEIGHTS = {
    "SECURITY": 0.28,
    "OTHER": 0.24,
    "TECHNICAL": 0.20,
    "HARASSMENT": 0.16,
    "CORRUPTION": 0.12,
}


def _compose(rng: random.Random, category: str) -> str:
    """Assemble one report body for ``category``."""
    opener = rng.choice(OPENERS)
    subject = rng.choice(SUBJECTS[category])
    predicate = rng.choice(PREDICATES[category])
    context = rng.choice(CONTEXTS)

    body = f"{opener}{subject} {predicate}.{context}"
    return body[0].upper() + body[1:] if body else body


def generate(rows: int = 900, seed: int = RANDOM_SEED) -> list[dict[str, str]]:
    """Build the synthetic dataset as a list of records.

    Returns records with exactly two fields — ``description`` and
    ``category``. Nothing else: no status, no id, no timestamp, no moderator.
    A generator that emitted those would invite them into the feature set, and
    the description is the only legitimate model input.
    """
    # A seeded Mersenne Twister is precisely what is wanted here: the dataset
    # must be byte-identical for a given seed. S311 warns against it for
    # cryptographic use, which this is not — nothing generated here is a secret.
    rng = random.Random(seed)  # noqa: S311
    records: list[dict[str, str]] = []

    categories = list(CLASS_WEIGHTS)
    weights = [CLASS_WEIGHTS[category] for category in categories]

    for _ in range(rows):
        category = rng.choices(categories, weights=weights, k=1)[0]
        records.append({"description": _compose(rng, category), "category": category})

    # The ambiguous cases, repeated a few times each so they carry weight.
    for category, text in AMBIGUOUS:
        for _ in range(4):
            records.append({"description": text, "category": category})

    # Exact duplicates within one class — the ordinary kind, where the same
    # report was filed twice.
    for record in rng.sample(records[:rows], 12):
        records.append(dict(record))

    # And three pairs of identical text carrying *different* labels, so the
    # conflicting-label check has something real to find. This happens for
    # genuine reasons — two moderators reading one report differently.
    conflicting = [
        (
            "The same supplier keeps winning every contract and their engineer "
            "still has an admin login to our systems.",
            ("CORRUPTION", "SECURITY"),
        ),
        (
            "Someone keeps changing the rota after it is published and will not "
            "say who authorised it.",
            ("OTHER", "HARASSMENT"),
        ),
        (
            "The export job writes customer records to a folder the whole company can read.",
            ("TECHNICAL", "SECURITY"),
        ),
    ]
    for text, (first, second) in conflicting:
        records.append({"description": text, "category": first})
        records.append({"description": text, "category": second})

    rng.shuffle(records)
    return records


def write_csv(records: list[dict[str, str]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["description", "category"])
        writer.writeheader()
        writer.writerows(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ml.src.generate_synthetic",
        description="Generate the synthetic development dataset. NOT real data.",
    )
    parser.add_argument("--rows", type=int, default=900, help="Base rows before extras.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args(argv)

    records = generate(rows=args.rows, seed=args.seed)
    write_csv(records, args.out)

    print(f"Wrote {len(records)} SYNTHETIC records to {args.out}")
    print("This is generated scaffolding, not real reports. Metrics derived from")
    print("it describe the generator, not real-world performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
