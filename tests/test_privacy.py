"""Privacy tests: what the system structurally cannot record.

These deliberately go beyond "the response happened not to contain an email
address". A response that omits a field today can include it tomorrow. So these
tests assert on *structure*:

* the persisted schema has no column an identity could be written into;
* the request schema rejects such a field rather than dropping it;
* the OpenAPI contract advertises exactly three input fields and no more.

What these tests do NOT establish
---------------------------------
Nothing here demonstrates anonymity in any absolute sense, and the project does
not claim it. What is verified is *application-level* non-collection: this
service does not ask for, store or derive reporter identity. Outside that
boundary, all of the following remain true and are out of scope:

* **Network layer.** The reporter's IP address reaches whatever terminates TLS
  in front of the API — a load balancer, a CDN, a reverse proxy — and those
  commonly log it. This application never sees or stores it, and equally cannot
  prevent the infrastructure in front of it from doing so. Tor or a VPN is the
  reporter's answer, not ours.
* **Traffic analysis.** Submission timing, request size and access patterns are
  observable to anyone positioned on the network, and can be correlated with
  other events.
* **Report contents.** A report can identify its author perfectly well through
  what it says. The system stores the text as written, by design, because it is
  evidence — so this is a limit no schema can remove.
* **Writing style.** Stylometry works, and a long report is a large sample.
* **Operational reality.** A database administrator, a backup, or a subpoena
  reaches the stored rows. What those rows contain is the guarantee; that they
  are reachable is assumed.

"Anonymous" here means the system holds nothing that identifies the reporter.
It does not mean the reporter cannot be identified.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, inspect
from sqlalchemy.orm import Session

from app.models import Report
from app.schemas.reports import ReportSubmissionRequest

pytestmark = pytest.mark.integration

REPORTS_URL = "/api/v1/reports"
LOOKUP_URL = "/api/v1/cases/lookup"

# Every shape an identity could plausibly take in a schema like this.
IDENTITY_FIELD_NAMES = {
    "name",
    "full_name",
    "first_name",
    "last_name",
    "username",
    "email",
    "email_address",
    "phone",
    "phone_number",
    "mobile",
    "address",
    "postcode",
    "reporter",
    "reporter_id",
    "reporter_name",
    "reporter_email",
    "reporter_ip",
    "account_id",
    "user_id",
    "submitter",
    "submitted_by",
    "ip",
    "ip_address",
    "client_ip",
    "user_agent",
    "session_id",
    "device_id",
    "fingerprint",
}


# ---------------------------------------------------------------------------
# The persisted structure
# ---------------------------------------------------------------------------


def test_no_table_can_hold_a_reporter_identity(db_engine: Engine) -> None:
    """Inspect the live schema, column by column, not just the model classes."""
    inspector = inspect(db_engine)

    offenders: list[str] = []
    for table in inspector.get_table_names():
        if table == "alembic_version":
            continue
        for column in inspector.get_columns(table):
            if column["name"].lower() in IDENTITY_FIELD_NAMES:
                # moderators.username is the one legitimate identity in the
                # system: staff are identified, reporters are not.
                if table == "moderators" and column["name"] == "username":
                    continue
                offenders.append(f"{table}.{column['name']}")

    assert offenders == [], f"identity columns present in the schema: {offenders}"


def test_there_is_no_reporters_table(db_engine: Engine) -> None:
    tables = {name.lower() for name in inspect(db_engine).get_table_names()}

    assert "reporters" not in tables
    assert "reporter" not in tables
    assert "users" not in tables


def test_reports_table_holds_exactly_the_expected_columns(db_engine: Engine) -> None:
    """A new column on `reports` fails this test until it is reviewed.

    That is the point: adding somewhere to put an identity should require a
    deliberate decision, not slip in with a migration.
    """
    columns = {column["name"] for column in inspect(db_engine).get_columns("reports")}

    assert columns == {
        "id",
        "case_code_hash",
        "category",
        "description",
        "evidence_url",
        "status",
        "created_at",
        "updated_at",
    }


def test_a_persisted_report_carries_nothing_but_its_own_content(
    api_client: TestClient, db_session: Session
) -> None:
    """Read back a real row and check every attribute it actually has."""
    api_client.post(
        REPORTS_URL,
        json={
            "category": "HARASSMENT",
            "description": "A manager repeatedly makes demeaning remarks in team meetings.",
        },
    )

    report = db_session.query(Report).one()
    ignored = {"metadata", "registry"}
    attributes = {name for name in dir(report) if not name.startswith("_") and name not in ignored}

    assert not attributes & IDENTITY_FIELD_NAMES


def test_no_request_metadata_is_recorded(api_client: TestClient, db_session: Session) -> None:
    """Headers that could identify a reporter are not persisted anywhere."""
    api_client.post(
        REPORTS_URL,
        json={"category": "OTHER", "description": "Something is wrong with the process here."},
        headers={
            "User-Agent": "IdentifiableBrowser/1.0 (some-unique-machine)",
            "X-Forwarded-For": "203.0.113.47",
            "Referer": "https://internal.example.com/staff/directory",
        },
    )

    report = db_session.query(Report).one()
    stored = " ".join(
        str(value) for value in (report.description, report.evidence_url, report.case_code_hash)
    )

    assert "IdentifiableBrowser" not in stored
    assert "203.0.113.47" not in stored
    assert "internal.example.com" not in stored


# ---------------------------------------------------------------------------
# The request contract
# ---------------------------------------------------------------------------


def test_the_request_schema_accepts_exactly_three_fields() -> None:
    assert set(ReportSubmissionRequest.model_fields) == {
        "category",
        "description",
        "evidence_url",
    }


def test_the_request_schema_is_closed() -> None:
    """Unknown keys are an error, not something silently discarded."""
    assert ReportSubmissionRequest.model_config["extra"] == "forbid"


@pytest.mark.parametrize("field", sorted(IDENTITY_FIELD_NAMES))
def test_every_identity_field_is_refused_by_the_api(api_client: TestClient, field: str) -> None:
    response = api_client.post(
        REPORTS_URL,
        json={
            "category": "OTHER",
            "description": "A description long enough to pass validation on its own.",
            field: "identifying-value",
        },
    )

    assert response.status_code == 422, f"{field!r} was accepted"


def test_the_openapi_contract_advertises_no_identity_field(client: TestClient) -> None:
    """Whatever a client reads from the spec, it cannot conclude we want a name."""
    schema = client.get("/openapi.json").json()
    request_schema = schema["components"]["schemas"]["ReportSubmissionRequest"]

    assert set(request_schema["properties"]) == {"category", "description", "evidence_url"}
    assert request_schema.get("additionalProperties") is False


# ---------------------------------------------------------------------------
# The documented limits
# ---------------------------------------------------------------------------


def test_the_api_documentation_states_the_network_limitation(client: TestClient) -> None:
    """The docs must not overstate the guarantee.

    A reporter deciding whether it is safe to file needs the real boundary, not
    a marketing claim.
    """
    schema = client.get("/openapi.json").json()
    prose = (
        schema["info"]["description"] + schema["paths"][REPORTS_URL]["post"]["description"]
    ).lower()

    assert "ip address" in prose
    assert "tor" in prose or "vpn" in prose
    assert "100% anonymous" not in prose
    assert "completely anonymous" not in prose
    assert "fully anonymous" not in prose


def test_the_documentation_states_that_a_lost_code_is_unrecoverable(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    prose = (
        schema["paths"][REPORTS_URL]["post"]["description"]
        + schema["paths"][LOOKUP_URL]["post"]["description"]
    ).lower()

    assert "recover" in prose
