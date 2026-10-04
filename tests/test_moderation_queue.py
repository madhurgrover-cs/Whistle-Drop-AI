"""Integration tests for the moderator queue and report detail.

Real HTTP, real tokens, real PostgreSQL. Reports are filed through the public
endpoint rather than inserted as rows, so these operate on exactly the shape
the system produces — initial timeline entry included.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.tokens import create_access_token
from app.models import Moderator, Report
from app.schemas.moderation import DEFAULT_PAGE_SIZE, DESCRIPTION_PREVIEW_LENGTH, MAX_PAGE_SIZE
from tests.conftest import TEST_JWT_SECRET_KEY

pytestmark = pytest.mark.integration

QUEUE_URL = "/api/v1/moderation/reports"


def detail_url(report_id) -> str:
    return f"{QUEUE_URL}/{report_id}"


# ---------------------------------------------------------------------------
# 1-5. Authorisation
# ---------------------------------------------------------------------------


def test_the_queue_requires_authentication(api_client: TestClient) -> None:
    response = api_client.get(QUEUE_URL)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTHENTICATION_FAILED"
    assert response.headers["www-authenticate"] == 'Bearer realm="whistledrop"'


@pytest.mark.parametrize(
    "header",
    [
        {},
        {"Authorization": "Bearer not-a-real-token"},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": "some-token-without-a-scheme"},
        {"Authorization": "Bearer"},
    ],
    ids=["none", "garbage", "basic", "no-scheme", "scheme-only"],
)
def test_every_moderation_endpoint_refuses_a_bad_credential(
    api_client: TestClient, header: dict
) -> None:
    # Only PATCH takes a body; httpx rejects json= on a GET.
    attempts = (
        api_client.get(QUEUE_URL, headers=header),
        api_client.get(detail_url(uuid.uuid4()), headers=header),
        api_client.patch(
            f"{detail_url(uuid.uuid4())}/status",
            headers=header,
            json={"status": "RESOLVED"},
        ),
    )

    for response in attempts:
        assert response.status_code == 401, (
            f"{response.request.method} {response.request.url.path} with {header}"
        )


def test_an_expired_token_is_refused(
    api_client: TestClient, moderator: tuple[Moderator, str]
) -> None:
    account, _ = moderator
    stale = create_access_token(
        subject=account.id,
        secret_key=TEST_JWT_SECRET_KEY,
        now=datetime.now(UTC) - timedelta(hours=2),
    )

    response = api_client.get(QUEUE_URL, headers={"Authorization": f"Bearer {stale}"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "TOKEN_EXPIRED"


def test_a_deactivated_moderator_is_refused(
    api_client: TestClient, inactive_moderator: tuple[Moderator, str]
) -> None:
    """A valid token is not enough; the account must still be active."""
    account, _ = inactive_moderator
    token = create_access_token(subject=account.id, secret_key=TEST_JWT_SECRET_KEY)

    response = api_client.get(QUEUE_URL, headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401


def test_a_moderator_deactivated_mid_session_loses_access_immediately(
    api_client: TestClient,
    db_session: Session,
    moderator: tuple[Moderator, str],
    auth_headers: dict,
) -> None:
    account, _ = moderator
    assert api_client.get(QUEUE_URL, headers=auth_headers).status_code == 200

    account.is_active = False
    db_session.commit()

    assert api_client.get(QUEUE_URL, headers=auth_headers).status_code == 401


def test_a_valid_moderator_is_allowed(api_client: TestClient, auth_headers: dict) -> None:
    assert api_client.get(QUEUE_URL, headers=auth_headers).status_code == 200


# ---------------------------------------------------------------------------
# 6-12. The queue
# ---------------------------------------------------------------------------


def test_the_queue_returns_reports(
    api_client: TestClient, auth_headers: dict, submit_report
) -> None:
    submit_report()
    submit_report(category="HARASSMENT")

    body = api_client.get(QUEUE_URL, headers=auth_headers).json()

    assert body["page"]["total_items"] == 2
    assert len(body["items"]) == 2
    assert set(body["items"][0]) == {
        "id",
        "category",
        "status",
        "description_preview",
        "evidence_url",
        "created_at",
        "updated_at",
    }


def test_an_empty_queue_is_a_valid_answer(api_client: TestClient, auth_headers: dict) -> None:
    body = api_client.get(QUEUE_URL, headers=auth_headers).json()

    assert body["items"] == []
    assert body["page"]["total_items"] == 0
    assert body["page"]["total_pages"] == 0
    assert body["page"]["has_next"] is False
    assert body["page"]["has_previous"] is False


def test_filtering_by_status(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    submit_report()
    advanced = submit_report()
    api_client.patch(
        f"{detail_url(report_id_for(advanced['case_code']))}/status",
        headers=auth_headers,
        json={"status": "UNDER_REVIEW"},
    )

    submitted = api_client.get(QUEUE_URL, headers=auth_headers, params={"status": "SUBMITTED"})
    reviewing = api_client.get(QUEUE_URL, headers=auth_headers, params={"status": "UNDER_REVIEW"})

    assert submitted.json()["page"]["total_items"] == 1
    assert reviewing.json()["page"]["total_items"] == 1
    assert reviewing.json()["items"][0]["status"] == "UNDER_REVIEW"
    assert submitted.json()["filters"] == {"status": "SUBMITTED", "category": None}


def test_filtering_by_category(api_client: TestClient, auth_headers: dict, submit_report) -> None:
    submit_report(category="SECURITY")
    submit_report(category="SECURITY")
    submit_report(category="CORRUPTION")

    security = api_client.get(QUEUE_URL, headers=auth_headers, params={"category": "SECURITY"})
    corruption = api_client.get(QUEUE_URL, headers=auth_headers, params={"category": "CORRUPTION"})

    assert security.json()["page"]["total_items"] == 2
    assert corruption.json()["page"]["total_items"] == 1
    assert all(item["category"] == "SECURITY" for item in security.json()["items"])


def test_the_two_filters_compose(
    api_client: TestClient, auth_headers: dict, submit_report, report_id_for
) -> None:
    submit_report(category="HARASSMENT")  # SUBMITTED + HARASSMENT
    submit_report(category="SECURITY")  # SUBMITTED + SECURITY
    target = submit_report(category="HARASSMENT")
    api_client.patch(
        f"{detail_url(report_id_for(target['case_code']))}/status",
        headers=auth_headers,
        json={"status": "UNDER_REVIEW"},
    )

    response = api_client.get(
        QUEUE_URL, headers=auth_headers, params={"status": "UNDER_REVIEW", "category": "HARASSMENT"}
    )

    body = response.json()
    assert body["page"]["total_items"] == 1
    assert body["items"][0]["status"] == "UNDER_REVIEW"
    assert body["items"][0]["category"] == "HARASSMENT"
    assert body["filters"] == {"status": "UNDER_REVIEW", "category": "HARASSMENT"}


def test_a_filter_matching_nothing_returns_an_empty_page(
    api_client: TestClient, auth_headers: dict, submit_report
) -> None:
    submit_report(category="SECURITY")

    body = api_client.get(QUEUE_URL, headers=auth_headers, params={"status": "RESOLVED"}).json()

    assert body["items"] == []
    assert body["page"]["total_items"] == 0


@pytest.mark.parametrize(
    "params",
    [
        {"status": "NOT_A_STATUS"},
        {"category": "NOT_A_CATEGORY"},
        {"status": "submitted"},
        {"page": 0},
        {"page": -1},
        {"page_size": 0},
        {"page_size": MAX_PAGE_SIZE + 1},
        {"page": "abc"},
    ],
    ids=[
        "bad-status",
        "bad-category",
        "lowercase-status",
        "page-zero",
        "page-negative",
        "size-zero",
        "size-over-max",
        "page-not-a-number",
    ],
)
def test_invalid_query_parameters_are_rejected(
    api_client: TestClient, auth_headers: dict, params: dict
) -> None:
    response = api_client.get(QUEUE_URL, headers=auth_headers, params=params)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_pagination_walks_the_whole_set_without_repeats(
    api_client: TestClient, auth_headers: dict, submit_report
) -> None:
    for index in range(7):
        submit_report(description=f"Paginated fixture report number {index} for the queue.")

    seen: list[str] = []
    for page in (1, 2, 3):
        body = api_client.get(
            QUEUE_URL, headers=auth_headers, params={"page": page, "page_size": 3}
        ).json()
        seen.extend(item["id"] for item in body["items"])

        assert body["page"]["page"] == page
        assert body["page"]["total_items"] == 7
        assert body["page"]["total_pages"] == 3

    assert len(seen) == 7
    assert len(set(seen)) == 7, "a report appeared on two pages"


def test_pagination_metadata_is_correct(
    api_client: TestClient, auth_headers: dict, submit_report
) -> None:
    for _ in range(5):
        submit_report()

    first = api_client.get(QUEUE_URL, headers=auth_headers, params={"page_size": 2}).json()["page"]
    last = api_client.get(
        QUEUE_URL, headers=auth_headers, params={"page": 3, "page_size": 2}
    ).json()["page"]

    assert first == {
        "page": 1,
        "page_size": 2,
        "total_items": 5,
        "total_pages": 3,
        "has_next": True,
        "has_previous": False,
    }
    assert last["has_next"] is False
    assert last["has_previous"] is True


def test_the_default_page_size_applies(
    api_client: TestClient, auth_headers: dict, submit_report
) -> None:
    submit_report()

    assert (
        api_client.get(QUEUE_URL, headers=auth_headers).json()["page"]["page_size"]
        == DEFAULT_PAGE_SIZE
    )


def test_the_page_size_ceiling_cannot_be_exceeded(
    api_client: TestClient, auth_headers: dict
) -> None:
    """No single request may pull the whole table."""
    at_limit = api_client.get(QUEUE_URL, headers=auth_headers, params={"page_size": MAX_PAGE_SIZE})
    over_limit = api_client.get(
        QUEUE_URL, headers=auth_headers, params={"page_size": MAX_PAGE_SIZE + 1}
    )
    absurd = api_client.get(QUEUE_URL, headers=auth_headers, params={"page_size": 10_000})

    assert at_limit.status_code == 200
    assert at_limit.json()["page"]["page_size"] == MAX_PAGE_SIZE
    assert over_limit.status_code == 422
    assert absurd.status_code == 422


def test_a_page_past_the_end_is_empty_rather_than_an_error(
    api_client: TestClient, auth_headers: dict, submit_report
) -> None:
    submit_report()

    body = api_client.get(QUEUE_URL, headers=auth_headers, params={"page": 99}).json()

    assert body["items"] == []
    assert body["page"]["total_items"] == 1


def test_the_ordering_is_newest_first_and_deterministic(
    api_client: TestClient, auth_headers: dict, submit_report
) -> None:
    """Repeating the request must give the same order every time.

    Reports filed in one transaction share a ``created_at`` exactly, so the
    ``id`` tiebreaker is what makes this stable rather than incidental.
    """
    for index in range(6):
        submit_report(description=f"Ordering fixture report number {index} for the queue.")

    orders = [
        [item["id"] for item in api_client.get(QUEUE_URL, headers=auth_headers).json()["items"]]
        for _ in range(4)
    ]

    assert len({tuple(order) for order in orders}) == 1, "the queue order varied between requests"

    stamps = [
        item["created_at"]
        for item in api_client.get(QUEUE_URL, headers=auth_headers).json()["items"]
    ]
    assert stamps == sorted(stamps, reverse=True)


def test_the_queue_is_not_cacheable(api_client: TestClient, auth_headers: dict) -> None:
    response = api_client.get(QUEUE_URL, headers=auth_headers)

    assert response.headers["cache-control"] == "no-store"


# ---------------------------------------------------------------------------
# Queue hygiene
# ---------------------------------------------------------------------------


def test_the_queue_never_exposes_the_case_code_hash(
    api_client: TestClient, db_session: Session, auth_headers: dict, submit_report
) -> None:
    submitted = submit_report()

    response = api_client.get(QUEUE_URL, headers=auth_headers)

    stored = db_session.scalars(select(Report)).one()
    assert stored.case_code_hash not in response.text
    assert "case_code" not in response.text
    assert submitted["case_code"] not in response.text


def test_the_queue_shortens_long_descriptions(
    api_client: TestClient, auth_headers: dict, submit_report
) -> None:
    long_body = "x" * (DESCRIPTION_PREVIEW_LENGTH + 500)
    submit_report(description=long_body)

    item = api_client.get(QUEUE_URL, headers=auth_headers).json()["items"][0]

    assert len(item["description_preview"]) <= DESCRIPTION_PREVIEW_LENGTH + 1
    assert item["description_preview"].endswith("…")
    assert item["description_preview"] != long_body


def test_a_short_description_is_not_marked_as_shortened(
    api_client: TestClient, auth_headers: dict, submit_report
) -> None:
    body = "A short but complete report body."
    submit_report(description=body)

    item = api_client.get(QUEUE_URL, headers=auth_headers).json()["items"][0]

    assert item["description_preview"] == body
    assert not item["description_preview"].endswith("…")


def test_the_queue_filters_in_sql_not_in_python(
    api_client: TestClient, auth_headers: dict, submit_report
) -> None:
    """A filtered page must not read rows it will then discard.

    Counted by watching the statements the connection actually executes: a
    filtered query and its count, not a full table scan narrowed afterwards.
    """
    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    for _ in range(5):
        submit_report(category="SECURITY")
    submit_report(category="OTHER")

    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany) -> None:
        statements.append(statement)

    event.listen(Engine, "before_cursor_execute", record)
    try:
        api_client.get(QUEUE_URL, headers=auth_headers, params={"category": "SECURITY"})
    finally:
        event.remove(Engine, "before_cursor_execute", record)

    report_queries = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("SELECT") and "reports" in statement
    ]
    assert report_queries, "no report query was issued"

    # The page query must carry both the filter and the limit. If filtering
    # happened in Python there would be an unfiltered, unlimited SELECT here.
    assert any(
        "category" in statement and "LIMIT" in statement.upper() for statement in report_queries
    ), f"expected a filtered, limited query; saw: {report_queries}"
    assert not any(
        "category" not in statement and "LIMIT" not in statement.upper()
        for statement in report_queries
    ), f"an unfiltered, unlimited report scan was issued: {report_queries}"
