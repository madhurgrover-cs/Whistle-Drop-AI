"""Integration tests for ``POST /api/v1/reports``.

Full stack: real HTTP request, real service, real PostgreSQL. The handler and
the assertions share one session inside one transaction (see the ``api_client``
fixture), so a test can file a report over HTTP and then read the rows it
produced, and nothing outlives the test.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.case_codes import CASE_CODE_PATTERN, canonicalise_case_code, hash_case_code
from app.models import CaseUpdate, Report, ReportCategory, ReportStatus
from app.schemas.reports import DESCRIPTION_MAX_LENGTH, DESCRIPTION_MIN_LENGTH
from app.services.reports import INITIAL_UPDATE_NOTE
from tests.conftest import TEST_CASE_CODE_PEPPER

pytestmark = pytest.mark.integration

REPORTS_URL = "/api/v1/reports"

VALID_DESCRIPTION = "Expense claims are being approved without any receipts or oversight."


def _body(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "category": "CORRUPTION",
        "description": VALID_DESCRIPTION,
    }
    payload.update(overrides)
    return payload


def _stored_report(session: Session, case_code: str) -> Report:
    """Fetch the row a case code points at, the same way the service would."""
    canonical = canonicalise_case_code(case_code)
    assert canonical is not None
    return session.scalars(
        select(Report).where(
            Report.case_code_hash == hash_case_code(canonical, TEST_CASE_CODE_PEPPER)
        )
    ).one()


# ---------------------------------------------------------------------------
# 1-2. Successful submission, no authentication
# ---------------------------------------------------------------------------


def test_report_can_be_submitted(api_client: TestClient) -> None:
    response = api_client.post(REPORTS_URL, json=_body())

    assert response.status_code == 201, response.text
    body = response.json()
    assert set(body) == {"case_code", "status", "submitted_at", "message"}
    assert body["status"] == "SUBMITTED"
    assert CASE_CODE_PATTERN.match(body["case_code"])


def test_submission_requires_no_authentication(api_client: TestClient) -> None:
    """No credential of any kind is sent, and the request still succeeds."""
    assert "authorization" not in {k.lower() for k in api_client.headers}

    response = api_client.post(REPORTS_URL, json=_body())

    assert response.status_code == 201


def test_submission_is_not_documented_as_requiring_a_security_scheme(
    client: TestClient,
) -> None:
    """The OpenAPI contract must not advertise auth on an anonymous endpoint.

    A ``securitySchemes`` block does exist from Phase 4 onwards, for the
    moderation routes. What matters is that this operation does not reference
    it: a client reading the spec must not conclude that filing a report needs
    a credential.
    """
    schema = client.get("/openapi.json").json()

    assert "security" not in schema["paths"][REPORTS_URL]["post"]
    assert not schema["paths"][REPORTS_URL]["post"].get("security")


# ---------------------------------------------------------------------------
# 3-4. Category
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", [c.value for c in ReportCategory])
def test_every_official_category_is_accepted(api_client: TestClient, category: str) -> None:
    response = api_client.post(REPORTS_URL, json=_body(category=category))

    assert response.status_code == 201, response.text


@pytest.mark.parametrize(
    "category",
    ["ESPIONAGE", "security", "Security", "", "OTHER ", 42, None],
    ids=["unknown", "lowercase", "mixed-case", "empty", "trailing-space", "number", "null"],
)
def test_invalid_category_is_rejected(api_client: TestClient, category: object) -> None:
    response = api_client.post(REPORTS_URL, json=_body(category=category))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_category_is_required(api_client: TestClient) -> None:
    response = api_client.post(REPORTS_URL, json={"description": VALID_DESCRIPTION})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 5-7. Description
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "description",
    ["", "   ", "\n\n\t  \r\n", "short"],
    ids=["empty", "spaces", "newlines-and-tabs", "below-minimum"],
)
def test_blank_or_tiny_description_is_rejected(api_client: TestClient, description: str) -> None:
    response = api_client.post(REPORTS_URL, json=_body(description=description))

    assert response.status_code == 422
    fields = {detail["field"] for detail in response.json()["error"]["details"]}
    assert "body.description" in fields


def test_description_is_required(api_client: TestClient) -> None:
    response = api_client.post(REPORTS_URL, json={"category": "OTHER"})

    assert response.status_code == 422


def test_excessively_long_description_is_rejected(api_client: TestClient) -> None:
    response = api_client.post(
        REPORTS_URL, json=_body(description="x" * (DESCRIPTION_MAX_LENGTH + 1))
    )

    assert response.status_code == 422


def test_description_at_the_maximum_is_accepted(api_client: TestClient) -> None:
    """The boundary is inclusive; only the character past it fails."""
    response = api_client.post(REPORTS_URL, json=_body(description="x" * DESCRIPTION_MAX_LENGTH))

    assert response.status_code == 201


def test_description_at_the_minimum_is_accepted(api_client: TestClient) -> None:
    response = api_client.post(REPORTS_URL, json=_body(description="x" * DESCRIPTION_MIN_LENGTH))

    assert response.status_code == 201


def test_surrounding_whitespace_is_trimmed_but_the_text_is_untouched(
    api_client: TestClient, db_session: Session
) -> None:
    """Only the edges are normalised; a report's own formatting is evidence."""
    written = "Line one.\n\n    Indented line two.\n\nLine three."
    response = api_client.post(REPORTS_URL, json=_body(description=f"\n\n  {written}   \n"))

    assert response.status_code == 201
    stored = _stored_report(db_session, response.json()["case_code"])
    assert stored.description == written


def test_windows_line_endings_are_normalised(api_client: TestClient, db_session: Session) -> None:
    response = api_client.post(
        REPORTS_URL, json=_body(description="First line.\r\nSecond line.\r\nThird line.")
    )

    assert response.status_code == 201
    stored = _stored_report(db_session, response.json()["case_code"])
    assert "\r" not in stored.description
    assert stored.description == "First line.\nSecond line.\nThird line."


def test_null_bytes_are_rejected(api_client: TestClient) -> None:
    """PostgreSQL text cannot hold a NUL, and no person typed one."""
    response = api_client.post(REPORTS_URL, json=_body(description=f"{VALID_DESCRIPTION}\x00"))

    assert response.status_code == 422


def test_description_is_stored_verbatim_and_not_escaped(
    api_client: TestClient, db_session: Session
) -> None:
    """Report text is data, not markup. Mangling it would destroy evidence."""
    written = "The <script> tag & the 'quote' were in the config: a > b."
    response = api_client.post(REPORTS_URL, json=_body(description=written))

    assert response.status_code == 201
    assert _stored_report(db_session, response.json()["case_code"]).description == written


# ---------------------------------------------------------------------------
# 8-9. Evidence URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/evidence.png",
        "http://example.com/a/b?c=d#e",
        "https://sub.domain.example.co.uk:8443/path",
    ],
    ids=["https", "http-with-query", "port-and-subdomain"],
)
def test_valid_evidence_url_is_accepted(api_client: TestClient, url: str) -> None:
    response = api_client.post(REPORTS_URL, json=_body(evidence_url=url))

    assert response.status_code == 201, response.text


@pytest.mark.parametrize(
    "url",
    [
        "not a url",
        "example.com",
        "javascript:alert(1)",
        "file:///etc/passwd",
        "data:text/html;base64,PHNjcmlwdD4=",
        "ftp://example.com/x",
        "https://" + "a" * 3000 + ".com",
        "",
    ],
    ids=[
        "prose",
        "no-scheme",
        "javascript",
        "file",
        "data-uri",
        "ftp",
        "over-length",
        "empty-string",
    ],
)
def test_invalid_evidence_url_is_rejected(api_client: TestClient, url: str) -> None:
    response = api_client.post(REPORTS_URL, json=_body(evidence_url=url))

    assert response.status_code == 422


def test_evidence_url_is_optional(api_client: TestClient, db_session: Session) -> None:
    response = api_client.post(REPORTS_URL, json=_body())

    assert response.status_code == 201
    assert _stored_report(db_session, response.json()["case_code"]).evidence_url is None


def test_explicit_null_evidence_url_is_accepted(api_client: TestClient) -> None:
    response = api_client.post(REPORTS_URL, json=_body(evidence_url=None))

    assert response.status_code == 201


# ---------------------------------------------------------------------------
# 10-11, 14. The case code and what is stored
# ---------------------------------------------------------------------------


def test_case_code_is_returned_on_success(api_client: TestClient) -> None:
    body = api_client.post(REPORTS_URL, json=_body()).json()

    assert CASE_CODE_PATTERN.match(body["case_code"])
    assert "cannot be recovered" in body["message"]


def test_every_submission_gets_a_different_case_code(api_client: TestClient) -> None:
    codes = {api_client.post(REPORTS_URL, json=_body()).json()["case_code"] for _ in range(25)}

    assert len(codes) == 25


def test_the_stored_value_is_a_hash_and_not_the_case_code(
    api_client: TestClient, db_session: Session
) -> None:
    case_code = api_client.post(REPORTS_URL, json=_body()).json()["case_code"]
    stored = _stored_report(db_session, case_code)

    canonical = canonicalise_case_code(case_code)
    assert canonical is not None
    assert stored.case_code_hash == hash_case_code(canonical, TEST_CASE_CODE_PEPPER)
    assert stored.case_code_hash != case_code
    assert len(stored.case_code_hash) == 64


def test_the_plaintext_case_code_appears_nowhere_in_the_database(
    api_client: TestClient, db_session: Session
) -> None:
    """Scan every text column of every row for the code, in any spelling.

    This is the phase's central security claim, so it is checked against the
    real tables rather than inferred from the code path that wrote them.
    """
    case_code = api_client.post(REPORTS_URL, json=_body()).json()["case_code"]
    compact = case_code.replace("-", "")

    report = _stored_report(db_session, case_code)
    haystacks = [
        str(report.case_code_hash),
        str(report.description),
        str(report.evidence_url),
        str(report.category),
        str(report.status),
    ]
    haystacks += [
        f"{update.note} {update.to_status} {update.from_status}"
        for update in db_session.scalars(
            select(CaseUpdate).where(CaseUpdate.report_id == report.id)
        )
    ]

    blob = " ".join(haystacks).upper()
    assert case_code.upper() not in blob
    assert compact.upper() not in blob


def test_hashing_is_deterministic_across_submissions(
    api_client: TestClient, db_session: Session
) -> None:
    """Same code plus same pepper always reaches the same row.

    This is exactly the property that makes an indexed lookup possible, and it
    is what a salted password hash could not provide.
    """
    case_code = api_client.post(REPORTS_URL, json=_body()).json()["case_code"]
    canonical = canonicalise_case_code(case_code)
    assert canonical is not None

    digests = {hash_case_code(canonical, TEST_CASE_CODE_PEPPER) for _ in range(10)}

    assert len(digests) == 1
    assert _stored_report(db_session, case_code).case_code_hash == digests.pop()


# ---------------------------------------------------------------------------
# 15-17. Initial state and timeline
# ---------------------------------------------------------------------------


def test_initial_status_is_submitted(api_client: TestClient, db_session: Session) -> None:
    response = api_client.post(REPORTS_URL, json=_body())

    assert response.json()["status"] == "SUBMITTED"
    assert _stored_report(db_session, response.json()["case_code"]).status is ReportStatus.SUBMITTED


def test_an_initial_case_update_is_created(api_client: TestClient, db_session: Session) -> None:
    case_code = api_client.post(REPORTS_URL, json=_body()).json()["case_code"]
    report = _stored_report(db_session, case_code)

    updates = db_session.scalars(select(CaseUpdate).where(CaseUpdate.report_id == report.id)).all()

    assert len(updates) == 1
    entry = updates[0]
    assert entry.from_status is None  # nothing preceded submission
    assert entry.to_status is ReportStatus.SUBMITTED
    assert entry.note == INITIAL_UPDATE_NOTE
    assert entry.visible_to_reporter is True
    assert entry.moderator_id is None  # no moderator has seen this yet


def test_the_initial_note_does_not_claim_a_moderator_reviewed_anything(
    api_client: TestClient, db_session: Session
) -> None:
    """The first entry describes receipt, not review. Overstating it would lie."""
    case_code = api_client.post(REPORTS_URL, json=_body()).json()["case_code"]
    report = _stored_report(db_session, case_code)
    note = (
        db_session.scalars(select(CaseUpdate).where(CaseUpdate.report_id == report.id)).one().note
    )

    assert note is not None
    lowered = note.lower()
    assert "received" in lowered
    assert "reviewed" not in lowered
    assert "resolved" not in lowered


# ---------------------------------------------------------------------------
# Response hygiene
# ---------------------------------------------------------------------------


def test_response_leaks_no_internal_fields(api_client: TestClient) -> None:
    body = api_client.post(REPORTS_URL, json=_body()).json()
    keys = {key.lower() for key in body}

    assert "case_code_hash" not in keys
    assert "id" not in keys
    assert not any("uuid" in key for key in keys)
    assert "report_id" not in keys
    assert "moderator_id" not in keys
    assert "description" not in keys


def test_identifying_fields_are_rejected_rather_than_ignored(api_client: TestClient) -> None:
    """A client that tries to attach an identity gets an error, not silence.

    Silently dropping the field would leave a caller believing it had been
    stored, and would let a future schema change quietly start accepting it.
    """
    for field in ("name", "email", "phone", "reporter_id", "account_id", "reporter_ip", "address"):
        response = api_client.post(REPORTS_URL, json=_body(**{field: "anything"}))

        assert response.status_code == 422, f"{field} was not rejected"
        fields = {detail["field"] for detail in response.json()["error"]["details"]}
        assert f"body.{field}" in fields


def test_malformed_body_is_rejected_cleanly(api_client: TestClient) -> None:
    response = api_client.post(
        REPORTS_URL, content=b"{not json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_empty_body_is_rejected(api_client: TestClient) -> None:
    assert api_client.post(REPORTS_URL, json={}).status_code == 422


def test_non_object_body_is_rejected(api_client: TestClient) -> None:
    assert api_client.post(REPORTS_URL, json=["not", "an", "object"]).status_code == 422


def test_get_is_not_allowed_on_the_submission_endpoint(api_client: TestClient) -> None:
    response = api_client.get(REPORTS_URL)

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "METHOD_NOT_ALLOWED"


# ---------------------------------------------------------------------------
# 28. Atomicity
# ---------------------------------------------------------------------------


def test_report_and_initial_update_are_written_together(
    api_client: TestClient, db_session: Session
) -> None:
    """Every report has exactly one timeline entry — no orphans either way."""
    for _ in range(5):
        api_client.post(REPORTS_URL, json=_body())

    report_count = db_session.scalar(select(func.count()).select_from(Report))
    update_count = db_session.scalar(select(func.count()).select_from(CaseUpdate))

    assert report_count == 5
    assert update_count == 5

    orphaned_reports = db_session.scalars(select(Report).where(~Report.updates.any())).all()
    assert orphaned_reports == []
