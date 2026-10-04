"""Integration tests for status transitions and the audit trail they produce.

The forbidden-transition cases are derived from the transition map rather than
listed by hand, so a future change to the workflow cannot leave a stale list of
"rejected" moves quietly passing.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import CaseUpdate, Moderator, Report, ReportStatus
from app.schemas.moderation import NOTE_MAX_LENGTH
from app.services.auth import AuthService
from app.services.moderation import ALLOWED_TRANSITIONS, forbidden_transitions
from tests.conftest import make_moderator_password, make_moderator_username

pytestmark = pytest.mark.integration

QUEUE_URL = "/api/v1/moderation/reports"
LOOKUP_URL = "/api/v1/cases/lookup"

INTERNAL_NOTE = "INTERNAL: cross-check with the finance team before contacting anyone."
PUBLISHED_NOTE = "Your report has been reviewed and the issue has been addressed."


def status_url(report_id) -> str:
    return f"{QUEUE_URL}/{report_id}/status"


def advance(client: TestClient, headers: dict, report_id, to: str, **extra):
    """Drive one transition through the API."""
    return client.patch(status_url(report_id), headers=headers, json={"status": to, **extra})


def put_in_state(client: TestClient, headers: dict, report_id, target: ReportStatus) -> None:
    """Walk a freshly submitted report to ``target`` by legal moves only."""
    if target is ReportStatus.SUBMITTED:
        return
    assert advance(client, headers, report_id, ReportStatus.UNDER_REVIEW.value).status_code == 200
    if target is ReportStatus.UNDER_REVIEW:
        return
    assert advance(client, headers, report_id, target.value).status_code == 200


# ---------------------------------------------------------------------------
# 20-22. Legal moves
# ---------------------------------------------------------------------------


def test_submitted_to_under_review(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    report_id = report_id_for(submit_report()["case_code"])

    response = advance(api_client, auth_headers, report_id, "UNDER_REVIEW")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["previous_status"] == "SUBMITTED"
    assert body["status"] == "UNDER_REVIEW"
    assert body["report_id"] == str(report_id)
    assert body["update"]["from_status"] == "SUBMITTED"
    assert body["update"]["to_status"] == "UNDER_REVIEW"


def test_under_review_to_resolved(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    report_id = report_id_for(submit_report()["case_code"])
    put_in_state(api_client, auth_headers, report_id, ReportStatus.UNDER_REVIEW)

    response = advance(api_client, auth_headers, report_id, "RESOLVED")

    assert response.status_code == 200
    assert response.json()["status"] == "RESOLVED"


def test_under_review_to_dismissed(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    report_id = report_id_for(submit_report()["case_code"])
    put_in_state(api_client, auth_headers, report_id, ReportStatus.UNDER_REVIEW)

    response = advance(api_client, auth_headers, report_id, "DISMISSED")

    assert response.status_code == 200
    assert response.json()["status"] == "DISMISSED"


def test_the_full_workflow_end_to_end(
    api_client: TestClient, db_session: Session, auth_headers: dict, submit_report, report_id_for
) -> None:
    report_id = report_id_for(submit_report()["case_code"])

    advance(api_client, auth_headers, report_id, "UNDER_REVIEW")
    advance(api_client, auth_headers, report_id, "RESOLVED")

    report = db_session.scalars(select(Report).where(Report.id == report_id)).one()
    assert report.status is ReportStatus.RESOLVED

    entries = db_session.scalars(
        select(CaseUpdate).where(CaseUpdate.report_id == report_id).order_by(CaseUpdate.created_at)
    ).all()
    assert [(e.from_status, e.to_status) for e in entries] == [
        (None, ReportStatus.SUBMITTED),
        (ReportStatus.SUBMITTED, ReportStatus.UNDER_REVIEW),
        (ReportStatus.UNDER_REVIEW, ReportStatus.RESOLVED),
    ]


# ---------------------------------------------------------------------------
# 23-29. Illegal moves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "current,requested",
    forbidden_transitions(),
    ids=lambda value: value.value if isinstance(value, ReportStatus) else str(value),
)
def test_every_forbidden_transition_is_rejected(
    api_client: TestClient,
    db_session: Session,
    auth_headers: dict,
    submit_report,
    report_id_for,
    current: ReportStatus,
    requested: ReportStatus,
) -> None:
    """Exhaustive: every pair the map does not allow, including same-to-same.

    Generated from ``ALLOWED_TRANSITIONS`` so this cannot fall out of step with
    the policy it is testing.
    """
    report_id = report_id_for(submit_report()["case_code"])
    put_in_state(api_client, auth_headers, report_id, current)

    response = advance(api_client, auth_headers, report_id, requested.value)

    assert response.status_code == 409, f"{current.value} -> {requested.value} was allowed"
    assert response.json()["error"]["code"] == "INVALID_STATUS_TRANSITION"

    # And the report did not move.
    db_session.expire_all()
    report = db_session.scalars(select(Report).where(Report.id == report_id)).one()
    assert report.status is current


def test_a_same_status_move_is_rejected(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    """Re-applying the current status is not a decision, and would forge history."""
    report_id = report_id_for(submit_report()["case_code"])

    response = advance(api_client, auth_headers, report_id, "SUBMITTED")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INVALID_STATUS_TRANSITION"


def test_a_rejected_transition_writes_no_audit_entry(
    api_client: TestClient, db_session: Session, auth_headers: dict, submit_report, report_id_for
) -> None:
    report_id = report_id_for(submit_report()["case_code"])

    advance(api_client, auth_headers, report_id, "RESOLVED", note="should never be recorded")

    entries = db_session.scalars(select(CaseUpdate).where(CaseUpdate.report_id == report_id)).all()
    assert len(entries) == 1  # only the original submission
    assert "should never be recorded" not in str([e.note for e in entries])


def test_a_closed_report_cannot_be_reopened(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    for terminal in ("RESOLVED", "DISMISSED"):
        report_id = report_id_for(submit_report()["case_code"])
        put_in_state(api_client, auth_headers, report_id, ReportStatus.UNDER_REVIEW)
        advance(api_client, auth_headers, report_id, terminal)

        for attempt in ("SUBMITTED", "UNDER_REVIEW", "RESOLVED", "DISMISSED"):
            response = advance(api_client, auth_headers, report_id, attempt)

            assert response.status_code == 409, f"{terminal} -> {attempt} was allowed"


def test_the_rejection_explains_what_would_be_legal(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    """A useful message. The workflow is documented, so naming it leaks nothing."""
    report_id = report_id_for(submit_report()["case_code"])

    message = advance(api_client, auth_headers, report_id, "RESOLVED").json()["error"]["message"]

    assert "SUBMITTED" in message
    assert "RESOLVED" in message
    assert "UNDER_REVIEW" in message  # the move that would have been allowed


def test_the_transition_map_matches_the_documented_workflow() -> None:
    """The policy itself, asserted directly."""
    assert ALLOWED_TRANSITIONS[ReportStatus.SUBMITTED] == frozenset({ReportStatus.UNDER_REVIEW})
    assert ALLOWED_TRANSITIONS[ReportStatus.UNDER_REVIEW] == frozenset(
        {ReportStatus.RESOLVED, ReportStatus.DISMISSED}
    )
    assert ALLOWED_TRANSITIONS[ReportStatus.RESOLVED] == frozenset()
    assert ALLOWED_TRANSITIONS[ReportStatus.DISMISSED] == frozenset()
    # No status may transition to itself.
    for current, allowed in ALLOWED_TRANSITIONS.items():
        assert current not in allowed


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"status": "NOT_A_STATUS"},
        {"status": "under_review"},
        {"status": None},
        {"status": "UNDER_REVIEW", "note": "x" * (NOTE_MAX_LENGTH + 1)},
        {"status": "UNDER_REVIEW", "visible_to_reporter": "maybe"},
        {"status": "UNDER_REVIEW", "unknown_field": 1},
    ],
    ids=[
        "empty",
        "bad-status",
        "lowercase-status",
        "null-status",
        "note-too-long",
        "bad-boolean",
        "unknown-field",
    ],
)
def test_a_malformed_status_request_is_rejected(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for, body: dict
) -> None:
    report_id = report_id_for(submit_report()["case_code"])

    response = api_client.patch(status_url(report_id), headers=auth_headers, json=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_a_status_change_on_a_missing_report_returns_404(
    api_client: TestClient, auth_headers: dict
) -> None:
    response = advance(api_client, auth_headers, uuid.uuid4(), "UNDER_REVIEW")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "REPORT_NOT_FOUND"


# ---------------------------------------------------------------------------
# 30-34. The audit entry
# ---------------------------------------------------------------------------


def test_a_transition_records_everything_it_should(
    api_client: TestClient,
    db_session: Session,
    auth_headers: dict,
    moderator: tuple[Moderator, str],
    submit_report,
    report_id_for,
) -> None:
    account, _ = moderator
    report_id = report_id_for(submit_report()["case_code"])

    advance(
        api_client,
        auth_headers,
        report_id,
        "UNDER_REVIEW",
        note=INTERNAL_NOTE,
        visible_to_reporter=False,
    )

    entry = db_session.scalars(
        select(CaseUpdate)
        .where(CaseUpdate.report_id == report_id, CaseUpdate.from_status.is_not(None))
        .order_by(CaseUpdate.created_at)
    ).one()

    assert entry.from_status is ReportStatus.SUBMITTED
    assert entry.to_status is ReportStatus.UNDER_REVIEW
    assert entry.moderator_id == account.id
    assert entry.note == INTERNAL_NOTE
    assert entry.visible_to_reporter is False


def test_the_visibility_flag_is_stored_as_given(
    api_client: TestClient, db_session: Session, auth_headers: dict, submit_report, report_id_for
) -> None:
    report_id = report_id_for(submit_report()["case_code"])

    advance(api_client, auth_headers, report_id, "UNDER_REVIEW", visible_to_reporter=True)
    advance(api_client, auth_headers, report_id, "RESOLVED", visible_to_reporter=False)

    entries = db_session.scalars(
        select(CaseUpdate)
        .where(CaseUpdate.report_id == report_id, CaseUpdate.from_status.is_not(None))
        .order_by(CaseUpdate.created_at)
    ).all()

    assert [e.visible_to_reporter for e in entries] == [True, False]


def test_visibility_defaults_to_internal(
    api_client: TestClient, db_session: Session, auth_headers: dict, submit_report, report_id_for
) -> None:
    """Omitting the flag must keep the note internal, never publish it."""
    report_id = report_id_for(submit_report()["case_code"])

    advance(api_client, auth_headers, report_id, "UNDER_REVIEW", note="No flag was supplied.")

    entry = db_session.scalars(
        select(CaseUpdate).where(
            CaseUpdate.report_id == report_id, CaseUpdate.from_status.is_not(None)
        )
    ).one()
    assert entry.visible_to_reporter is False


def test_a_note_is_optional(
    api_client: TestClient, db_session: Session, auth_headers: dict, submit_report, report_id_for
) -> None:
    report_id = report_id_for(submit_report()["case_code"])

    response = advance(api_client, auth_headers, report_id, "UNDER_REVIEW")

    assert response.status_code == 200
    assert response.json()["update"]["note"] is None


# ---------------------------------------------------------------------------
# Moderator identity cannot be forged
# ---------------------------------------------------------------------------


def test_a_moderator_id_in_the_body_is_rejected(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    """Rejected, not ignored — so an attempt to impersonate fails loudly."""
    report_id = report_id_for(submit_report()["case_code"])

    response = api_client.patch(
        status_url(report_id),
        headers=auth_headers,
        json={"status": "UNDER_REVIEW", "moderator_id": str(uuid.uuid4())},
    )

    assert response.status_code == 422
    fields = {detail["field"] for detail in response.json()["error"]["details"]}
    assert "body.moderator_id" in fields


def test_the_recorded_moderator_is_the_token_holder(
    api_client: TestClient,
    db_session: Session,
    auth_headers: dict,
    auth_service: AuthService,
    moderator: tuple[Moderator, str],
    submit_report,
    report_id_for,
) -> None:
    """With a second moderator in the database, the token still decides."""
    account, _ = moderator

    other = auth_service.create_moderator(
        username=make_moderator_username(), password=make_moderator_password()
    )
    db_session.commit()

    report_id = report_id_for(submit_report()["case_code"])
    api_client.patch(status_url(report_id), headers=auth_headers, json={"status": "UNDER_REVIEW"})

    entry = db_session.scalars(
        select(CaseUpdate).where(
            CaseUpdate.report_id == report_id, CaseUpdate.from_status.is_not(None)
        )
    ).one()

    assert entry.moderator_id == account.id
    assert entry.moderator_id != other.id


# ---------------------------------------------------------------------------
# 35-36. What the reporter sees afterwards
# ---------------------------------------------------------------------------


def test_an_internal_note_never_reaches_the_reporter(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    submitted = submit_report()
    report_id = report_id_for(submitted["case_code"])

    advance(
        api_client,
        auth_headers,
        report_id,
        "UNDER_REVIEW",
        note=INTERNAL_NOTE,
        visible_to_reporter=False,
    )

    lookup = api_client.post(LOOKUP_URL, json={"case_code": submitted["case_code"]})

    assert lookup.status_code == 200
    assert INTERNAL_NOTE not in lookup.text
    assert "INTERNAL" not in lookup.text
    assert [entry["status"] for entry in lookup.json()["timeline"]] == ["SUBMITTED"]
    # The status change itself is still public — only the note is withheld.
    assert lookup.json()["status"] == "UNDER_REVIEW"


def test_a_published_note_does_reach_the_reporter(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    submitted = submit_report()
    report_id = report_id_for(submitted["case_code"])

    advance(api_client, auth_headers, report_id, "UNDER_REVIEW", visible_to_reporter=False)
    advance(
        api_client,
        auth_headers,
        report_id,
        "RESOLVED",
        note=PUBLISHED_NOTE,
        visible_to_reporter=True,
    )

    body = api_client.post(LOOKUP_URL, json={"case_code": submitted["case_code"]}).json()

    assert body["status"] == "RESOLVED"
    notes = [entry["note"] for entry in body["timeline"]]
    assert PUBLISHED_NOTE in notes
    assert INTERNAL_NOTE not in notes


def test_the_reporter_view_is_unchanged_in_shape_after_moderation(
    api_client: TestClient,
    auth_headers: dict,
    moderator: tuple[Moderator, str],
    submit_report,
    report_id_for,
) -> None:
    """Moderation must not widen what case lookup returns."""
    account, _ = moderator
    submitted = submit_report()
    report_id = report_id_for(submitted["case_code"])

    advance(
        api_client,
        auth_headers,
        report_id,
        "UNDER_REVIEW",
        note=PUBLISHED_NOTE,
        visible_to_reporter=True,
    )

    response = api_client.post(LOOKUP_URL, json={"case_code": submitted["case_code"]})
    body = response.json()

    assert set(body) == {"status", "category", "submitted_at", "last_updated_at", "timeline"}
    for entry in body["timeline"]:
        assert set(entry) == {"status", "note", "occurred_at"}

    assert str(report_id) not in response.text
    assert str(account.id) not in response.text
    assert account.username not in response.text
    assert "moderator" not in str(set(body)).lower()
