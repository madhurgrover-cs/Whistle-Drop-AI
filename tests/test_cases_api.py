"""Integration tests for ``POST /api/v1/cases/lookup``.

Covers the reporter's side of the system: that a real case code resolves, that
a wrong one fails without telling the caller anything, and — the part that
matters most — that the response contains only what a reporter is meant to see.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.case_codes import canonicalise_case_code, generate_case_code, hash_case_code
from app.models import CaseUpdate, Moderator, Report, ReportStatus
from app.services.reports import INITIAL_UPDATE_NOTE
from tests.conftest import TEST_CASE_CODE_PEPPER

pytestmark = pytest.mark.integration

LOOKUP_URL = "/api/v1/cases/lookup"
REPORTS_URL = "/api/v1/reports"

HIDDEN_NOTE = "INTERNAL: complainant may be the team lead; escalate to legal before contact."
PUBLISHED_NOTE = "A moderator has begun reviewing this report."


def _report_for(session: Session, case_code: str) -> Report:
    canonical = canonicalise_case_code(case_code)
    assert canonical is not None
    return session.scalars(
        select(Report).where(
            Report.case_code_hash == hash_case_code(canonical, TEST_CASE_CODE_PEPPER)
        )
    ).one()


# ---------------------------------------------------------------------------
# 18, 20, 21, 26. The happy path
# ---------------------------------------------------------------------------


def test_lookup_with_the_correct_case_code_succeeds(
    api_client: TestClient, submitted_report: dict
) -> None:
    response = api_client.post(LOOKUP_URL, json={"case_code": submitted_report["case_code"]})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "SUBMITTED"
    assert body["category"] == "CORRUPTION"
    assert body["submitted_at"] == submitted_report["submitted_at"]


def test_lookup_requires_no_authentication(api_client: TestClient, submitted_report: dict) -> None:
    """The case code is the only credential, and no other is accepted."""
    assert "authorization" not in {k.lower() for k in api_client.headers}

    response = api_client.post(LOOKUP_URL, json={"case_code": submitted_report["case_code"]})

    assert response.status_code == 200


def test_reporter_sees_the_current_status_after_it_changes(
    api_client: TestClient, db_session: Session, submitted_report: dict
) -> None:
    report = _report_for(db_session, submitted_report["case_code"])
    report.status = ReportStatus.UNDER_REVIEW
    db_session.commit()

    body = api_client.post(LOOKUP_URL, json={"case_code": submitted_report["case_code"]}).json()

    assert body["status"] == "UNDER_REVIEW"


def test_reporter_sees_the_visible_timeline(api_client: TestClient, submitted_report: dict) -> None:
    body = api_client.post(LOOKUP_URL, json={"case_code": submitted_report["case_code"]}).json()

    assert len(body["timeline"]) == 1
    entry = body["timeline"][0]
    assert entry["status"] == "SUBMITTED"
    assert entry["note"] == INITIAL_UPDATE_NOTE
    assert entry["occurred_at"]


def test_timeline_is_ordered_oldest_first(
    api_client: TestClient, db_session: Session, submitted_report: dict
) -> None:
    report = _report_for(db_session, submitted_report["case_code"])
    db_session.add(
        CaseUpdate(
            report_id=report.id,
            from_status=ReportStatus.SUBMITTED,
            to_status=ReportStatus.UNDER_REVIEW,
            note=PUBLISHED_NOTE,
            visible_to_reporter=True,
        )
    )
    db_session.commit()

    timeline = api_client.post(
        LOOKUP_URL, json={"case_code": submitted_report["case_code"]}
    ).json()["timeline"]

    assert [entry["status"] for entry in timeline] == ["SUBMITTED", "UNDER_REVIEW"]
    # Ordering is real, not incidental: case_updates.created_at defaults to
    # clock_timestamp(), so entries written inside one transaction still carry
    # distinct, increasing timestamps.
    stamps = [entry["occurred_at"] for entry in timeline]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)


def test_lookup_accepts_a_sloppily_copied_code(
    api_client: TestClient, submitted_report: dict
) -> None:
    """Lower case, no hyphens, stray spaces — all still find the report."""
    code = submitted_report["case_code"]

    spellings = (
        code.lower(),
        code.replace("-", ""),
        f"  {code}  ",
        code.lower().replace("-", ""),
    )
    for spelling in spellings:
        response = api_client.post(LOOKUP_URL, json={"case_code": spelling})

        assert response.status_code == 200, f"{spelling!r} did not resolve"


def test_lookup_response_is_not_cacheable(api_client: TestClient, submitted_report: dict) -> None:
    """The body holds report contents; no shared cache should keep it."""
    response = api_client.post(LOOKUP_URL, json={"case_code": submitted_report["case_code"]})

    assert response.headers["cache-control"] == "no-store"


# ---------------------------------------------------------------------------
# 19, 25. Failing safely
# ---------------------------------------------------------------------------


def test_lookup_with_an_unknown_but_valid_code_returns_404(api_client: TestClient) -> None:
    response = api_client.post(LOOKUP_URL, json={"case_code": generate_case_code()})

    assert response.status_code == 404
    assert response.json() == {
        "error": {"code": "CASE_NOT_FOUND", "message": "No report matches that case code."}
    }


@pytest.mark.parametrize(
    "case_code",
    [
        "not-a-case-code",
        "WD-00000-00000-00000",
        "'; DROP TABLE reports; --",
        "../../etc/passwd",
        "<script>alert(1)</script>",
        "WD-4K7PQ-92MRT-XJ3HN-B8VZ6U",
        "%00",
    ],
    ids=["prose", "too-short", "sql", "traversal", "xss", "bad-letter", "null-byte"],
)
def test_malformed_codes_fail_exactly_like_unknown_ones(
    api_client: TestClient, case_code: str
) -> None:
    """A malformed code and an unknown one are indistinguishable.

    Same status, same code, same message — so the endpoint cannot be used as an
    oracle that confirms when a guess has the right shape.
    """
    malformed = api_client.post(LOOKUP_URL, json={"case_code": case_code})
    unknown = api_client.post(LOOKUP_URL, json={"case_code": generate_case_code()})

    assert malformed.status_code == unknown.status_code == 404
    assert malformed.json() == unknown.json()


def test_a_wrong_code_never_reaches_someone_elses_report(
    api_client: TestClient, submitted_report: dict
) -> None:
    other = api_client.post(
        REPORTS_URL,
        json={"category": "SECURITY", "description": "A different report entirely, unrelated."},
    ).json()

    body = api_client.post(LOOKUP_URL, json={"case_code": other["case_code"]}).json()

    # Resolves to the second report, not the first: the codes are not
    # interchangeable and neither reaches the other's record.
    assert body["category"] == "SECURITY"
    assert body["category"] != "CORRUPTION"


def test_error_body_never_echoes_the_submitted_code(api_client: TestClient) -> None:
    """A mistyped code must not be reflected into the response or any log of it."""
    attempted = "WD-MYTYPO-BADCODE-9999"

    response = api_client.post(LOOKUP_URL, json={"case_code": attempted})

    assert attempted not in response.text
    assert "details" not in response.json()["error"]


def test_missing_and_empty_case_codes_are_rejected(api_client: TestClient) -> None:
    assert api_client.post(LOOKUP_URL, json={}).status_code == 422
    assert api_client.post(LOOKUP_URL, json={"case_code": ""}).status_code == 422


def test_oversized_case_code_is_rejected_without_a_query(api_client: TestClient) -> None:
    response = api_client.post(LOOKUP_URL, json={"case_code": "A" * 5_000})

    assert response.status_code == 422


def test_unknown_fields_in_the_lookup_body_are_rejected(api_client: TestClient) -> None:
    response = api_client.post(
        LOOKUP_URL, json={"case_code": generate_case_code(), "email": "me@example.com"}
    )

    assert response.status_code == 422


def test_get_is_not_allowed_on_lookup(api_client: TestClient) -> None:
    """There is deliberately no GET form that would put the code in a URL."""
    assert api_client.get(LOOKUP_URL).status_code == 405
    assert api_client.get(f"{LOOKUP_URL}/WD-4K7PQ-92MRT-XJ3HN-B8VZ6").status_code == 404


# ---------------------------------------------------------------------------
# 22-25. What the reporter must never see
# ---------------------------------------------------------------------------


def test_hidden_timeline_entries_never_reach_the_reporter(
    api_client: TestClient, db_session: Session, submitted_report: dict
) -> None:
    """The central confidentiality guarantee of the tracking endpoint."""
    report = _report_for(db_session, submitted_report["case_code"])
    db_session.add_all(
        [
            CaseUpdate(
                report_id=report.id,
                from_status=ReportStatus.SUBMITTED,
                to_status=ReportStatus.UNDER_REVIEW,
                note=HIDDEN_NOTE,
                visible_to_reporter=False,
            ),
            CaseUpdate(
                report_id=report.id,
                from_status=ReportStatus.SUBMITTED,
                to_status=ReportStatus.UNDER_REVIEW,
                note=PUBLISHED_NOTE,
                visible_to_reporter=True,
            ),
        ]
    )
    db_session.commit()

    response = api_client.post(LOOKUP_URL, json={"case_code": submitted_report["case_code"]})
    body = response.json()

    notes = [entry["note"] for entry in body["timeline"]]
    assert PUBLISHED_NOTE in notes
    assert HIDDEN_NOTE not in notes
    # Belt and braces: the string must not appear anywhere in the payload.
    assert HIDDEN_NOTE not in response.text
    assert "INTERNAL" not in response.text


def test_a_fully_hidden_timeline_yields_an_empty_list_not_a_leak(
    api_client: TestClient, db_session: Session, submitted_report: dict
) -> None:
    report = _report_for(db_session, submitted_report["case_code"])
    for update in db_session.scalars(select(CaseUpdate).where(CaseUpdate.report_id == report.id)):
        update.visible_to_reporter = False
    db_session.commit()

    body = api_client.post(LOOKUP_URL, json={"case_code": submitted_report["case_code"]}).json()

    assert body["timeline"] == []
    assert body["status"] == "SUBMITTED"  # status itself is still shown


def test_moderator_identity_is_never_returned(
    api_client: TestClient, db_session: Session, submitted_report: dict
) -> None:
    moderator = Moderator(username="reviewer-alex", password_hash="placeholder-not-a-real-hash")
    db_session.add(moderator)
    db_session.flush()

    report = _report_for(db_session, submitted_report["case_code"])
    db_session.add(
        CaseUpdate(
            report_id=report.id,
            from_status=ReportStatus.SUBMITTED,
            to_status=ReportStatus.UNDER_REVIEW,
            note=PUBLISHED_NOTE,
            visible_to_reporter=True,
            moderator_id=moderator.id,
        )
    )
    db_session.commit()

    response = api_client.post(LOOKUP_URL, json={"case_code": submitted_report["case_code"]})

    assert response.status_code == 200
    # What must never appear is the moderator's *identity*. The word itself is
    # fine in a published note ("queued for review by a moderator") — that is
    # the role, not the person.
    assert "reviewer-alex" not in response.text
    assert str(moderator.id) not in response.text
    for entry in response.json()["timeline"]:
        assert "moderator_id" not in entry
        assert "moderator" not in entry


def test_internal_identifiers_are_never_returned(
    api_client: TestClient, db_session: Session, submitted_report: dict
) -> None:
    report = _report_for(db_session, submitted_report["case_code"])
    response = api_client.post(LOOKUP_URL, json={"case_code": submitted_report["case_code"]})
    body = response.json()

    # The report's primary key and every timeline entry's key.
    assert str(report.id) not in response.text
    for update in db_session.scalars(select(CaseUpdate).where(CaseUpdate.report_id == report.id)):
        assert str(update.id) not in response.text

    assert "id" not in body
    assert "report_id" not in body
    for entry in body["timeline"]:
        assert set(entry) == {"status", "note", "occurred_at"}


def test_the_case_code_hash_is_never_returned(
    api_client: TestClient, db_session: Session, submitted_report: dict
) -> None:
    report = _report_for(db_session, submitted_report["case_code"])

    response = api_client.post(LOOKUP_URL, json={"case_code": submitted_report["case_code"]})

    assert report.case_code_hash not in response.text
    assert "case_code_hash" not in response.text
    assert "hash" not in response.text.lower()


def test_the_case_code_is_never_returned_again(
    api_client: TestClient, submitted_report: dict
) -> None:
    """Submission is the only time the plaintext code is ever emitted."""
    response = api_client.post(LOOKUP_URL, json={"case_code": submitted_report["case_code"]})

    assert "case_code" not in response.json()
    assert submitted_report["case_code"] not in response.text


def test_the_description_is_not_echoed_back(api_client: TestClient, submitted_report: dict) -> None:
    """A stolen code yields progress, not the full text of the report."""
    response = api_client.post(LOOKUP_URL, json={"case_code": submitted_report["case_code"]})

    assert "description" not in response.json()
    assert "Procurement contracts" not in response.text


def test_the_response_contains_exactly_the_documented_fields(
    api_client: TestClient, submitted_report: dict
) -> None:
    """A column added to `reports` later cannot silently join the response."""
    body = api_client.post(LOOKUP_URL, json={"case_code": submitted_report["case_code"]}).json()

    assert set(body) == {
        "status",
        "category",
        "submitted_at",
        "last_updated_at",
        "timeline",
    }
