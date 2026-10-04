"""Integration tests for the moderator report-detail view.

The detail view is the widest disclosure in the system — the full report body,
every internal note, and who wrote them — so these tests care as much about
what it withholds as about what it returns.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import CaseUpdate, Moderator, Report, ReportStatus, ReportTriage, TriageStatus

pytestmark = pytest.mark.integration

QUEUE_URL = "/api/v1/moderation/reports"
LOOKUP_URL = "/api/v1/cases/lookup"

INTERNAL_NOTE = "INTERNAL: the reporter may be on the affected team; do not contact directly."
PUBLISHED_NOTE = "A moderator is now reviewing this report."


def detail_url(report_id) -> str:
    return f"{QUEUE_URL}/{report_id}"


# ---------------------------------------------------------------------------
# 13-15. Fetching
# ---------------------------------------------------------------------------


def test_an_existing_report_can_be_read(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    body_text = "Expense approvals are being signed off without any supporting receipts."
    submitted = submit_report(
        category="CORRUPTION",
        description=body_text,
        evidence_url="https://example.com/evidence",
    )
    report_id = report_id_for(submitted["case_code"])

    response = api_client.get(detail_url(report_id), headers=auth_headers)

    assert response.status_code == 200, response.text
    detail = response.json()
    assert detail["id"] == str(report_id)
    assert detail["category"] == "CORRUPTION"
    assert detail["status"] == "SUBMITTED"
    assert detail["description"] == body_text  # the full body, not a preview
    assert detail["evidence_url"] == "https://example.com/evidence"
    assert set(detail) == {
        "id",
        "category",
        "status",
        "description",
        "evidence_url",
        "created_at",
        "updated_at",
        "triage",
        "timeline",
    }


def test_the_detail_requires_authentication(
    api_client: TestClient, submit_report, report_id_for
) -> None:
    submitted = submit_report()
    report_id = report_id_for(submitted["case_code"])

    assert api_client.get(detail_url(report_id)).status_code == 401


def test_a_missing_report_returns_404(api_client: TestClient, auth_headers: dict) -> None:
    response = api_client.get(detail_url(uuid.uuid4()), headers=auth_headers)

    assert response.status_code == 404
    assert response.json() == {
        "error": {"code": "REPORT_NOT_FOUND", "message": "No report exists with that id."}
    }


@pytest.mark.parametrize(
    "bad_id",
    [
        "not-a-uuid",
        "12345",
        "' OR 1=1 --",
        "<script>alert(1)",
        "00000000-0000-0000-0000-00000000000",
        "a1b2c3d4-5e6f-4a7b-8c9d-0e1f2a3b4c5dEXTRA",
    ],
    ids=["prose", "number", "sql", "xss", "truncated-uuid", "trailing-junk"],
)
def test_a_malformed_report_id_is_rejected_safely(
    api_client: TestClient, auth_headers: dict, bad_id: str
) -> None:
    """A bad id fails path validation and never reaches the database as a string."""
    response = api_client.get(f"{QUEUE_URL}/{bad_id}", headers=auth_headers)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert bad_id not in response.text  # the value is never echoed back


@pytest.mark.parametrize(
    "bad_path",
    ["../../etc/passwd", "..%2f..%2fetc%2fpasswd", "a/b/c", "<script>alert(1)</script>"],
    ids=["traversal", "encoded-traversal", "extra-segments", "xss-with-slash"],
)
def test_a_path_shaped_id_matches_no_route_at_all(
    api_client: TestClient, auth_headers: dict, bad_path: str
) -> None:
    """Input containing separators changes the path rather than the parameter.

    It therefore matches no route and gets a plain 404 — which is equally safe:
    nothing reaches a handler, and no filesystem or database path is touched.
    """
    response = api_client.get(f"{QUEUE_URL}/{bad_path}", headers=auth_headers)

    assert response.status_code == 404
    assert "Traceback" not in response.text
    assert "passwd" not in response.text
    assert "sqlalchemy" not in response.text.lower()


def test_a_malformed_id_is_rejected_before_authentication_matters(
    api_client: TestClient,
) -> None:
    """Even unauthenticated, a bad id must not produce a stack trace."""
    response = api_client.get(f"{QUEUE_URL}/not-a-uuid")

    assert response.status_code in (401, 422)
    assert "Traceback" not in response.text
    assert "sqlalchemy" not in response.text.lower()


def test_the_detail_is_not_cacheable(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    report_id = report_id_for(submit_report()["case_code"])

    response = api_client.get(detail_url(report_id), headers=auth_headers)

    assert response.headers["cache-control"] == "no-store"


# ---------------------------------------------------------------------------
# 16-18. The timeline
# ---------------------------------------------------------------------------


def test_the_initial_submission_entry_is_present(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    report_id = report_id_for(submit_report()["case_code"])

    timeline = api_client.get(detail_url(report_id), headers=auth_headers).json()["timeline"]

    assert len(timeline) == 1
    entry = timeline[0]
    assert entry["from_status"] is None
    assert entry["to_status"] == "SUBMITTED"
    assert entry["visible_to_reporter"] is True
    assert entry["moderator_id"] is None
    assert entry["moderator_username"] is None


def test_a_moderator_sees_both_internal_and_published_entries(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    """The moderator view is complete; the reporter's is filtered."""
    report_id = report_id_for(submit_report()["case_code"])

    api_client.patch(
        f"{detail_url(report_id)}/status",
        headers=auth_headers,
        json={"status": "UNDER_REVIEW", "note": INTERNAL_NOTE, "visible_to_reporter": False},
    )
    api_client.patch(
        f"{detail_url(report_id)}/status",
        headers=auth_headers,
        json={"status": "RESOLVED", "note": PUBLISHED_NOTE, "visible_to_reporter": True},
    )

    timeline = api_client.get(detail_url(report_id), headers=auth_headers).json()["timeline"]

    notes = [entry["note"] for entry in timeline]
    assert INTERNAL_NOTE in notes
    assert PUBLISHED_NOTE in notes

    visibility = {entry["note"]: entry["visible_to_reporter"] for entry in timeline}
    assert visibility[INTERNAL_NOTE] is False
    assert visibility[PUBLISHED_NOTE] is True


def test_the_timeline_is_ordered_oldest_first(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    report_id = report_id_for(submit_report()["case_code"])

    api_client.patch(
        f"{detail_url(report_id)}/status", headers=auth_headers, json={"status": "UNDER_REVIEW"}
    )
    api_client.patch(
        f"{detail_url(report_id)}/status", headers=auth_headers, json={"status": "RESOLVED"}
    )

    timeline = api_client.get(detail_url(report_id), headers=auth_headers).json()["timeline"]

    assert [entry["to_status"] for entry in timeline] == [
        "SUBMITTED",
        "UNDER_REVIEW",
        "RESOLVED",
    ]
    stamps = [entry["created_at"] for entry in timeline]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps), "entries share a timestamp and cannot be ordered"


def test_the_acting_moderator_is_recorded_and_shown(
    api_client: TestClient,
    auth_headers: dict,
    moderator: tuple[Moderator, str],
    submit_report,
    report_id_for,
) -> None:
    account, _ = moderator
    report_id = report_id_for(submit_report()["case_code"])

    api_client.patch(
        f"{detail_url(report_id)}/status", headers=auth_headers, json={"status": "UNDER_REVIEW"}
    )

    timeline = api_client.get(detail_url(report_id), headers=auth_headers).json()["timeline"]
    transition = timeline[-1]

    assert transition["moderator_id"] == str(account.id)
    assert transition["moderator_username"] == account.username


def test_history_is_append_only(
    api_client: TestClient, db_session: Session, auth_headers: dict, submit_report, report_id_for
) -> None:
    """Three moves produce three entries, and the earlier ones are untouched."""
    report_id = report_id_for(submit_report()["case_code"])

    before = api_client.get(detail_url(report_id), headers=auth_headers).json()["timeline"]
    assert len(before) == 1
    original_entry = before[0]

    api_client.patch(
        f"{detail_url(report_id)}/status", headers=auth_headers, json={"status": "UNDER_REVIEW"}
    )
    api_client.patch(
        f"{detail_url(report_id)}/status", headers=auth_headers, json={"status": "DISMISSED"}
    )

    after = api_client.get(detail_url(report_id), headers=auth_headers).json()["timeline"]

    assert len(after) == 3
    assert after[0] == original_entry, "the original entry was modified"
    assert [e["from_status"] for e in after] == [None, "SUBMITTED", "UNDER_REVIEW"]
    assert [e["to_status"] for e in after] == ["SUBMITTED", "UNDER_REVIEW", "DISMISSED"]

    stored = db_session.scalars(select(CaseUpdate).where(CaseUpdate.report_id == report_id)).all()
    assert len(stored) == 3


# ---------------------------------------------------------------------------
# AI triage: reported, never invented
# ---------------------------------------------------------------------------


def test_triage_is_null_when_no_triage_row_exists(
    api_client: TestClient, db_session: Session, auth_headers: dict, submit_report, report_id_for
) -> None:
    """A report with no triage row reports null, not an invented suggestion.

    Since Phase 7 a submission normally creates one, so the row is removed
    here to reach the case this asserts — which still happens in practice: a
    report filed while triage was switched off, or before the feature existed.
    """
    from sqlalchemy import delete

    report_id = report_id_for(submit_report()["case_code"])
    db_session.execute(delete(ReportTriage).where(ReportTriage.report_id == report_id))
    db_session.commit()

    detail = api_client.get(detail_url(report_id), headers=auth_headers).json()

    assert detail["triage"] is None


def test_a_pending_triage_row_is_reported_as_pending(
    api_client: TestClient, db_session: Session, auth_headers: dict, submit_report, report_id_for
) -> None:
    """A queued job is shown as queued, not dressed up as a prediction.

    Phase 7 fills the row in at submission, so it is reset to PENDING here —
    the state a row is in between being created and inference completing.
    """
    from sqlalchemy import select

    report_id = report_id_for(submit_report()["case_code"])
    triage = db_session.scalars(
        select(ReportTriage).where(ReportTriage.report_id == report_id)
    ).one_or_none() or ReportTriage(report_id=report_id)
    triage.status = TriageStatus.PENDING
    triage.suggested_category = None
    triage.category_confidence = None
    triage.suggested_priority = None
    triage.keywords = []
    triage.model_version = None
    db_session.add(triage)
    db_session.commit()

    detail = api_client.get(detail_url(report_id), headers=auth_headers).json()

    assert detail["triage"] == {
        "status": "PENDING",
        "suggested_category": None,
        "category_confidence": None,
        "suggested_priority": None,
        "keywords": [],
        "model_version": None,
    }


def test_triage_never_overrides_the_official_category(
    api_client: TestClient, db_session: Session, auth_headers: dict, submit_report, report_id_for
) -> None:
    """A model's disagreement is shown beside the decision, never as it.

    Written by hand rather than left to the real model so the disagreement is
    guaranteed: the stored suggestion says SECURITY with high confidence while
    the official category is OTHER. Phase 7 fills this row in at submission, so
    it is overwritten here.
    """
    from decimal import Decimal

    from sqlalchemy import select

    from app.models import ReportCategory, TriagePriority

    report_id = report_id_for(submit_report(category="OTHER")["case_code"])
    triage = db_session.scalars(
        select(ReportTriage).where(ReportTriage.report_id == report_id)
    ).one_or_none() or ReportTriage(report_id=report_id)
    triage.suggested_category = ReportCategory.SECURITY
    triage.category_confidence = Decimal("0.970")
    triage.suggested_priority = TriagePriority.CRITICAL
    triage.status = TriageStatus.COMPLETED
    triage.keywords = ["credentials", "leak"]
    triage.model_version = "tfidf-logreg-v1"
    db_session.add(triage)
    db_session.commit()

    detail = api_client.get(detail_url(report_id), headers=auth_headers).json()

    assert detail["category"] == "OTHER"  # the human decision stands
    assert detail["status"] == "SUBMITTED"
    assert detail["triage"]["suggested_category"] == "SECURITY"
    assert detail["triage"]["keywords"] == ["credentials", "leak"]


# ---------------------------------------------------------------------------
# 19, 40-43. What the detail must never contain
# ---------------------------------------------------------------------------


def test_the_case_code_hash_is_never_exposed(
    api_client: TestClient, db_session: Session, auth_headers: dict, submit_report, report_id_for
) -> None:
    """The one stored value that could be tested offline against a guess."""
    submitted = submit_report()
    report_id = report_id_for(submitted["case_code"])
    stored = db_session.scalars(select(Report).where(Report.id == report_id)).one()

    response = api_client.get(detail_url(report_id), headers=auth_headers)

    assert stored.case_code_hash not in response.text
    assert "case_code" not in response.text
    assert submitted["case_code"] not in response.text
    assert "hash" not in response.text.lower()


def test_the_detail_carries_no_reporter_identity(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    """There is nothing to withhold: the schema records no identity at all."""
    report_id = report_id_for(submit_report()["case_code"])

    body = api_client.get(detail_url(report_id), headers=auth_headers).json()

    for forbidden in ("reporter", "email", "name", "phone", "ip", "user_agent", "account"):
        assert forbidden not in body


def test_no_moderator_password_hash_is_ever_returned(
    api_client: TestClient,
    auth_headers: dict,
    moderator: tuple[Moderator, str],
    submit_report,
    report_id_for,
) -> None:
    account, password = moderator
    report_id = report_id_for(submit_report()["case_code"])
    api_client.patch(
        f"{detail_url(report_id)}/status", headers=auth_headers, json={"status": "UNDER_REVIEW"}
    )

    response = api_client.get(detail_url(report_id), headers=auth_headers)

    assert account.password_hash not in response.text
    assert password not in response.text
    assert "$2b$" not in response.text


def test_the_public_lookup_shows_none_of_what_the_detail_shows(
    api_client: TestClient,
    db_session: Session,
    auth_headers: dict,
    submit_report,
    report_id_for,
    moderator: tuple[Moderator, str],
) -> None:
    """The two views of one report, compared side by side.

    This is the clearest statement of the privacy boundary: the same report,
    read by a moderator and by its reporter, and what differs between them.
    """
    account, _ = moderator
    submitted = submit_report()
    report_id = report_id_for(submitted["case_code"])

    api_client.patch(
        f"{detail_url(report_id)}/status",
        headers=auth_headers,
        json={"status": "UNDER_REVIEW", "note": INTERNAL_NOTE, "visible_to_reporter": False},
    )

    moderator_view = api_client.get(detail_url(report_id), headers=auth_headers)
    reporter_view = api_client.post(LOOKUP_URL, json={"case_code": submitted["case_code"]})

    # The moderator sees the internal note, the actor, the id and the body.
    assert INTERNAL_NOTE in moderator_view.text
    assert str(account.id) in moderator_view.text
    assert str(report_id) in moderator_view.text

    # The reporter sees none of them.
    assert INTERNAL_NOTE not in reporter_view.text
    assert str(account.id) not in reporter_view.text
    assert account.username not in reporter_view.text
    assert str(report_id) not in reporter_view.text
    assert "moderator_id" not in reporter_view.text
    assert "visible_to_reporter" not in reporter_view.text

    # But the status change itself is public.
    assert reporter_view.json()["status"] == ReportStatus.UNDER_REVIEW.value
