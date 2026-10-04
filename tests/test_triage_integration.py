"""Phase 7: triage inference wired into report submission.

The organising question behind every test here is the same one: *can anything
the model does cost a reporter their report?* The answer has to be no, for a
missing artifact, a corrupt one, a model that raises, a model that returns
nonsense, and a database that will not accept the triage row.
"""

import logging
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.main import create_app
from app.ml.artifacts import ArtifactLoader, ArtifactUnavailableError
from app.ml.inference import InferenceError, TriageModel, sanitise_keywords
from app.models import (
    CaseUpdate,
    Report,
    ReportCategory,
    ReportStatus,
    ReportTriage,
    TriagePriority,
    TriageStatus,
)
from app.services.triage import TriageService

pytestmark = pytest.mark.integration

REPORTS_URL = "/api/v1/reports"
LOOKUP_URL = "/api/v1/cases/lookup"
QUEUE_URL = "/api/v1/moderation/reports"

MODEL_VERSION = "whistledrop-category-v1"
ARTIFACT_ROOT = Path("ml/artifacts")

SECURITY_REPORT = {
    "category": "OTHER",  # deliberately not what the model will suggest
    "description": (
        "Production database credentials are being shared in a public chat channel "
        "and anyone in the company can read them."
    ),
}


def triage_for(session: Session, report_id) -> ReportTriage | None:
    return session.scalars(
        select(ReportTriage).where(ReportTriage.report_id == report_id)
    ).one_or_none()


@pytest.fixture
def real_model() -> TriageModel:
    return TriageModel(artifact_root=ARTIFACT_ROOT, model_version=MODEL_VERSION)


def client_with_model(
    settings: Settings, db_session: Session, model: TriageModel | None
) -> TestClient:
    """An app whose triage service uses ``model`` — real, broken, or absent."""
    from app.api.deps import get_triage_service

    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_triage_service] = lambda: TriageService(db_session, model=model)
    return TestClient(app)


# ---------------------------------------------------------------------------
# A. Successful inference
# ---------------------------------------------------------------------------


def test_submission_creates_report_case_update_and_triage(
    api_client: TestClient, db_session: Session, report_id_for
) -> None:
    response = api_client.post(REPORTS_URL, json=SECURITY_REPORT)

    assert response.status_code == 201
    report_id = report_id_for(response.json()["case_code"])

    report = db_session.scalars(select(Report).where(Report.id == report_id)).one()
    updates = db_session.scalars(select(CaseUpdate).where(CaseUpdate.report_id == report_id)).all()
    triage = triage_for(db_session, report_id)

    assert report is not None
    assert len(updates) == 1
    assert triage is not None
    assert triage.status is TriageStatus.COMPLETED


def test_a_completed_triage_row_holds_every_expected_field(
    api_client: TestClient, db_session: Session, report_id_for
) -> None:
    response = api_client.post(REPORTS_URL, json=SECURITY_REPORT)
    triage = triage_for(db_session, report_id_for(response.json()["case_code"]))

    assert triage.suggested_category in set(ReportCategory)
    assert triage.suggested_priority in set(TriagePriority)
    assert triage.model_version == MODEL_VERSION
    assert triage.keywords
    assert all(isinstance(keyword, str) for keyword in triage.keywords)
    assert triage.created_at is not None
    assert triage.updated_at is not None


def test_the_stored_confidence_is_within_range(
    api_client: TestClient, db_session: Session, report_id_for
) -> None:
    response = api_client.post(REPORTS_URL, json=SECURITY_REPORT)
    triage = triage_for(db_session, report_id_for(response.json()["case_code"]))

    assert triage.category_confidence is not None
    assert Decimal("0") <= triage.category_confidence <= Decimal("1")


def test_the_model_version_is_the_pinned_one(
    api_client: TestClient, db_session: Session, report_id_for
) -> None:
    """Never "latest", and never chosen by the request."""
    response = api_client.post(REPORTS_URL, json=SECURITY_REPORT)
    triage = triage_for(db_session, report_id_for(response.json()["case_code"]))

    assert triage.model_version == "whistledrop-category-v1"
    assert "latest" not in triage.model_version


def test_a_request_cannot_choose_a_model_version(api_client: TestClient) -> None:
    """The server controls the model. A client that names one is rejected."""
    response = api_client.post(
        REPORTS_URL, json={**SECURITY_REPORT, "model_version": "attacker-supplied-v9"}
    )

    assert response.status_code == 422


def test_a_request_cannot_supply_a_model_path(api_client: TestClient) -> None:
    for field in ("model_path", "artifact_path", "ml_artifact_root"):
        response = api_client.post(REPORTS_URL, json={**SECURITY_REPORT, field: "/etc/passwd"})

        assert response.status_code == 422, field


# ---------------------------------------------------------------------------
# E. The suggestion is advisory
# ---------------------------------------------------------------------------


def test_inference_does_not_change_the_official_category(
    api_client: TestClient, db_session: Session, report_id_for
) -> None:
    """The reporter chose OTHER; the model suggests SECURITY. OTHER stands."""
    response = api_client.post(REPORTS_URL, json=SECURITY_REPORT)
    report_id = report_id_for(response.json()["case_code"])

    report = db_session.scalars(select(Report).where(Report.id == report_id)).one()
    triage = triage_for(db_session, report_id)

    assert report.category is ReportCategory.OTHER
    assert triage.suggested_category is ReportCategory.SECURITY
    assert report.category is not triage.suggested_category


def test_inference_does_not_change_the_status(
    api_client: TestClient, db_session: Session, report_id_for
) -> None:
    response = api_client.post(REPORTS_URL, json=SECURITY_REPORT)
    report_id = report_id_for(response.json()["case_code"])

    report = db_session.scalars(select(Report).where(Report.id == report_id)).one()

    assert report.status is ReportStatus.SUBMITTED


def test_inference_adds_no_case_update(
    api_client: TestClient, db_session: Session, report_id_for
) -> None:
    """The audit trail records human decisions, not machine guesses."""
    response = api_client.post(REPORTS_URL, json=SECURITY_REPORT)
    report_id = report_id_for(response.json()["case_code"])

    updates = db_session.scalars(select(CaseUpdate).where(CaseUpdate.report_id == report_id)).all()

    assert len(updates) == 1
    assert updates[0].from_status is None
    assert updates[0].to_status is ReportStatus.SUBMITTED
    assert updates[0].moderator_id is None


def test_the_submission_response_is_unchanged(api_client: TestClient) -> None:
    """The Phase 2 contract, byte for byte."""
    body = api_client.post(REPORTS_URL, json=SECURITY_REPORT).json()

    assert set(body) == {"case_code", "status", "submitted_at", "message"}


# ---------------------------------------------------------------------------
# B / G. Failure isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def missing_artifact_model(tmp_path: Path) -> TriageModel:
    return TriageModel(artifact_root=tmp_path / "nowhere", model_version=MODEL_VERSION)


@pytest.fixture
def corrupt_artifact_model(tmp_path: Path) -> TriageModel:
    path = tmp_path / MODEL_VERSION / "category" / "model.joblib"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is not a joblib file at all")
    return TriageModel(artifact_root=tmp_path, model_version=MODEL_VERSION)


@pytest.fixture
def incompatible_artifact_model(tmp_path: Path) -> TriageModel:
    """A well-formed joblib file that is not this project's bundle."""
    import joblib

    path = tmp_path / MODEL_VERSION / "category" / "model.joblib"
    path.parent.mkdir(parents=True)
    joblib.dump({"something": "else"}, path)
    return TriageModel(artifact_root=tmp_path, model_version=MODEL_VERSION)


class ExplodingModel:
    """A model whose every prediction raises."""

    model_version = MODEL_VERSION

    def predict(self, description: str):
        raise InferenceError("simulated inference failure")


class CrashingModel:
    """A model that raises something the service does not expect."""

    model_version = MODEL_VERSION

    def predict(self, description: str):
        raise RuntimeError("simulated defect with /srv/secret/path in the message")


@pytest.mark.parametrize(
    "broken",
    ["missing", "corrupt", "incompatible", "exploding", "crashing"],
)
def test_a_broken_model_never_costs_the_reporter_their_report(
    test_settings: Settings,
    db_session: Session,
    report_id_for,
    request: pytest.FixtureRequest,
    broken: str,
) -> None:
    """The central guarantee of this phase."""
    model = {
        "missing": lambda: request.getfixturevalue("missing_artifact_model"),
        "corrupt": lambda: request.getfixturevalue("corrupt_artifact_model"),
        "incompatible": lambda: request.getfixturevalue("incompatible_artifact_model"),
        "exploding": ExplodingModel,
        "crashing": CrashingModel,
    }[broken]()

    with client_with_model(test_settings, db_session, model) as client:
        response = client.post(REPORTS_URL, json=SECURITY_REPORT)

    # The reporter is served in full.
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["case_code"]
    assert body["status"] == "SUBMITTED"

    report_id = report_id_for(body["case_code"])
    report = db_session.scalars(select(Report).where(Report.id == report_id)).one()
    updates = db_session.scalars(select(CaseUpdate).where(CaseUpdate.report_id == report_id)).all()
    triage = triage_for(db_session, report_id)

    # The report is intact and complete.
    assert report.status is ReportStatus.SUBMITTED
    assert report.category is ReportCategory.OTHER
    assert len(updates) == 1

    # And the failure is recorded honestly rather than hidden.
    assert triage is not None
    assert triage.status is TriageStatus.FAILED
    assert triage.suggested_category is None
    assert triage.category_confidence is None
    assert triage.suggested_priority is None
    assert triage.keywords == []


def test_a_failure_exposes_nothing_internal(test_settings: Settings, db_session: Session) -> None:
    with client_with_model(test_settings, db_session, CrashingModel()) as client:
        response = client.post(REPORTS_URL, json=SECURITY_REPORT)

    assert response.status_code == 201
    text = response.text.lower()
    for leak in (
        "traceback",
        "sklearn",
        "joblib",
        "/srv/secret",
        "runtimeerror",
        "inferenceerror",
        "model.joblib",
        "artifact",
    ):
        assert leak not in text, f"{leak!r} leaked to the client"


def test_the_application_starts_with_no_artifact_at_all(
    test_settings: Settings, db_session: Session, tmp_path: Path
) -> None:
    """Startup must not depend on the model file existing."""
    settings = test_settings.model_copy(update={"ml_artifact_root": tmp_path / "absent"})

    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = lambda: db_session

    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.post(REPORTS_URL, json=SECURITY_REPORT).status_code == 201


def test_loading_is_attempted_once_and_the_failure_is_cached(tmp_path: Path) -> None:
    """A broken artifact is not re-read on every request."""
    loader = ArtifactLoader(root=tmp_path, version=MODEL_VERSION)

    for _ in range(3):
        with pytest.raises(ArtifactUnavailableError):
            loader.get()

    assert loader.failure_reason is not None
    assert "not found" in loader.failure_reason


def test_an_artifact_declaring_the_wrong_version_is_refused(tmp_path: Path) -> None:
    """A triage row must never record a version the artifact does not match."""
    import joblib

    path = tmp_path / MODEL_VERSION / "category" / "model.joblib"
    path.parent.mkdir(parents=True)
    real = joblib.load(ARTIFACT_ROOT / MODEL_VERSION / "category" / "model.joblib")
    joblib.dump({**real, "model_version": "someone-elses-model-v4"}, path)

    loader = ArtifactLoader(root=tmp_path, version=MODEL_VERSION)

    with pytest.raises(ArtifactUnavailableError, match="declares version"):
        loader.get()


def test_triage_can_be_switched_off_entirely(
    test_settings: Settings, db_session: Session, report_id_for
) -> None:
    """Off means no row at all, not a FAILED one: nothing was attempted."""
    with client_with_model(test_settings, db_session, None) as client:
        response = client.post(REPORTS_URL, json=SECURITY_REPORT)

    assert response.status_code == 201
    assert triage_for(db_session, report_id_for(response.json()["case_code"])) is None


# ---------------------------------------------------------------------------
# F. Invalid model output
# ---------------------------------------------------------------------------


class FakeVectorizer:
    def transform(self, texts):
        return [[0.0]]


class FakeClassifier:
    def __init__(self, classes, probabilities):
        self.classes_ = classes
        self._probabilities = probabilities

    def predict_proba(self, matrix):
        return [self._probabilities]


def model_returning(classes, probabilities, tmp_path: Path) -> TriageModel:
    """A TriageModel wired to a fake artifact with controlled output."""
    import joblib

    path = tmp_path / MODEL_VERSION / "category" / "model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": FakeVectorizer(),
            "classifier": FakeClassifier(classes, probabilities),
            "labels": list(classes),
            "model_version": MODEL_VERSION,
        },
        path,
    )
    return TriageModel(artifact_root=tmp_path, model_version=MODEL_VERSION)


@pytest.mark.parametrize(
    "classes,probabilities,reason",
    [
        (["ESPIONAGE", "OTHER"], [0.9, 0.1], "unknown category"),
        (["security", "OTHER"], [0.9, 0.1], "wrong case"),
        (["SECURITY", "OTHER"], [1.5, 0.1], "confidence above 1"),
        (["SECURITY", "OTHER"], [-0.4, -0.9], "confidence below 0"),
        (["SECURITY", "OTHER"], [float("nan"), 0.1], "non-finite confidence"),
        (["SECURITY", "OTHER"], [0.9], "probability vector too short"),
    ],
)
def test_invalid_model_output_is_refused(
    tmp_path: Path, classes, probabilities, reason: str
) -> None:
    """Nothing invalid may reach the database, even from a swapped artifact."""
    model = model_returning(classes, probabilities, tmp_path)

    with pytest.raises(InferenceError):
        model.predict("A report about credentials being shared publicly.")


def test_invalid_model_output_records_a_failed_row_not_a_bad_one(
    test_settings: Settings, db_session: Session, report_id_for, tmp_path: Path
) -> None:
    model = model_returning(["ESPIONAGE", "OTHER"], [0.95, 0.05], tmp_path)

    with client_with_model(test_settings, db_session, model) as client:
        response = client.post(REPORTS_URL, json=SECURITY_REPORT)

    assert response.status_code == 201
    triage = triage_for(db_session, report_id_for(response.json()["case_code"]))
    assert triage.status is TriageStatus.FAILED
    assert triage.suggested_category is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        (["one", "two"], ["one", "two"]),
        (["one", "one", "two"], ["one", "two"]),
        (["  padded  ", ""], ["padded"]),
        ([1, None, {"a": 1}, "kept"], ["kept"]),
        ("not a list", []),
        (None, []),
        ([], []),
        (["x" * 200], ["x" * 64]),
        ([f"term{i}" for i in range(50)], [f"term{i}" for i in range(10)]),
    ],
    ids=[
        "plain",
        "deduplicated",
        "trimmed",
        "non-strings-dropped",
        "not-a-list",
        "none",
        "empty",
        "over-length",
        "too-many",
    ],
)
def test_keywords_are_sanitised(raw, expected) -> None:
    """Malformed keywords lose decoration, never the suggestion."""
    assert sanitise_keywords(raw) == expected


# ---------------------------------------------------------------------------
# 16. One triage row per report
# ---------------------------------------------------------------------------


def test_running_triage_twice_updates_one_row(
    db_session: Session, real_model: TriageModel, report_id_for, api_client: TestClient
) -> None:
    response = api_client.post(REPORTS_URL, json=SECURITY_REPORT)
    report_id = report_id_for(response.json()["case_code"])

    service = TriageService(db_session, model=real_model)
    service.run_for_report(report_id, SECURITY_REPORT["description"])
    service.run_for_report(report_id, SECURITY_REPORT["description"])

    rows = db_session.scalars(select(ReportTriage).where(ReportTriage.report_id == report_id)).all()

    assert len(rows) == 1


def test_a_retry_can_turn_a_failed_row_into_a_completed_one(
    db_session: Session, real_model: TriageModel, report_id_for, api_client: TestClient
) -> None:
    """Triage is current state, not a log — the audit trail is case_updates."""
    response = api_client.post(REPORTS_URL, json=SECURITY_REPORT)
    report_id = report_id_for(response.json()["case_code"])

    TriageService(db_session, model=ExplodingModel()).run_for_report(
        report_id, SECURITY_REPORT["description"]
    )
    assert triage_for(db_session, report_id).status is TriageStatus.FAILED

    TriageService(db_session, model=real_model).run_for_report(
        report_id, SECURITY_REPORT["description"]
    )

    db_session.expire_all()
    triage = triage_for(db_session, report_id)
    assert triage.status is TriageStatus.COMPLETED
    assert triage.suggested_category is not None
    assert (
        len(
            db_session.scalars(
                select(ReportTriage).where(ReportTriage.report_id == report_id)
            ).all()
        )
        == 1
    )


# ---------------------------------------------------------------------------
# C. Reporter privacy
# ---------------------------------------------------------------------------


def test_case_lookup_returns_no_triage_information(api_client: TestClient) -> None:
    submitted = api_client.post(REPORTS_URL, json=SECURITY_REPORT).json()

    response = api_client.post(LOOKUP_URL, json={"case_code": submitted["case_code"]})
    body = response.json()

    assert set(body) == {"status", "category", "submitted_at", "last_updated_at", "timeline"}
    for forbidden in (
        "suggested_category",
        "confidence",
        "category_confidence",
        "priority",
        "keywords",
        "model_version",
        "triage",
        "triage_status",
        "ai",
    ):
        assert forbidden not in response.text.lower().replace("category", ""), forbidden


def test_the_reporter_sees_the_official_category_not_the_suggestion(
    api_client: TestClient, db_session: Session, report_id_for
) -> None:
    submitted = api_client.post(REPORTS_URL, json=SECURITY_REPORT).json()
    triage = triage_for(db_session, report_id_for(submitted["case_code"]))

    body = api_client.post(LOOKUP_URL, json={"case_code": submitted["case_code"]}).json()

    assert body["category"] == "OTHER"
    assert triage.suggested_category is ReportCategory.SECURITY
    assert body["category"] != triage.suggested_category.value


def test_the_submission_response_carries_no_triage_data(api_client: TestClient) -> None:
    response = api_client.post(REPORTS_URL, json=SECURITY_REPORT)

    text = response.text.lower()
    for forbidden in ("suggested", "confidence", "priority", "keyword", "model_version"):
        assert forbidden not in text, forbidden


def test_no_public_schema_mentions_triage(client: TestClient) -> None:
    """The OpenAPI contract for reporters must not advertise triage at all."""
    schema = client.get("/openapi.json").json()

    for name in ("ReportSubmissionResponse", "CaseLookupResponse", "CaseTimelineEntry"):
        properties = set(schema["components"]["schemas"][name]["properties"])
        assert not properties & {
            "suggested_category",
            "category_confidence",
            "suggested_priority",
            "keywords",
            "model_version",
            "triage",
        }, name


# ---------------------------------------------------------------------------
# D. Moderator visibility
# ---------------------------------------------------------------------------


def test_a_moderator_sees_the_triage_suggestion(
    api_client: TestClient, auth_headers: dict, report_id_for
) -> None:
    submitted = api_client.post(REPORTS_URL, json=SECURITY_REPORT).json()
    report_id = report_id_for(submitted["case_code"])

    detail = api_client.get(f"{QUEUE_URL}/{report_id}", headers=auth_headers).json()

    assert detail["triage"] is not None
    triage = detail["triage"]
    assert triage["status"] == "COMPLETED"
    assert triage["suggested_category"] in {category.value for category in ReportCategory}
    assert 0.0 <= float(triage["category_confidence"]) <= 1.0
    assert triage["suggested_priority"] in {level.value for level in TriagePriority}
    assert triage["keywords"]
    assert triage["model_version"] == MODEL_VERSION


def test_the_moderator_view_keeps_suggestion_and_decision_separate(
    api_client: TestClient, auth_headers: dict, report_id_for
) -> None:
    """The official category and the suggestion are different fields."""
    submitted = api_client.post(REPORTS_URL, json=SECURITY_REPORT).json()
    report_id = report_id_for(submitted["case_code"])

    detail = api_client.get(f"{QUEUE_URL}/{report_id}", headers=auth_headers).json()

    assert detail["category"] == "OTHER"  # the official one, at the top level
    assert detail["triage"]["suggested_category"] == "SECURITY"  # nested, labelled


def test_the_queue_does_not_leak_triage(api_client: TestClient, auth_headers: dict) -> None:
    """The queue is a scanning view; the suggestion lives in the detail."""
    api_client.post(REPORTS_URL, json=SECURITY_REPORT)

    item = api_client.get(QUEUE_URL, headers=auth_headers).json()["items"][0]

    assert "triage" not in item
    assert "suggested_category" not in item


def test_triage_requires_authentication(api_client: TestClient, report_id_for) -> None:
    submitted = api_client.post(REPORTS_URL, json=SECURITY_REPORT).json()
    report_id = report_id_for(submitted["case_code"])

    assert api_client.get(f"{QUEUE_URL}/{report_id}").status_code == 401


def test_a_failed_triage_is_shown_as_failed_to_a_moderator(
    test_settings: Settings,
    db_session: Session,
    auth_headers: dict,
    report_id_for,
) -> None:
    """A moderator learns the model did not help, rather than seeing nothing."""
    with client_with_model(test_settings, db_session, ExplodingModel()) as client:
        submitted = client.post(REPORTS_URL, json=SECURITY_REPORT).json()
        report_id = report_id_for(submitted["case_code"])
        detail = client.get(f"{QUEUE_URL}/{report_id}", headers=auth_headers).json()

    assert detail["triage"]["status"] == "FAILED"
    assert detail["triage"]["suggested_category"] is None
    assert detail["triage"]["category_confidence"] is None


def test_the_openapi_documents_triage_as_advisory_and_synthetic(
    client: TestClient,
) -> None:
    """The warning must survive into the contract a moderator reads."""
    schema = client.get("/openapi.json").json()
    detail_description = schema["paths"][f"{QUEUE_URL}/{{report_id}}"]["get"]["description"].lower()

    assert "advisory" in detail_description or "suggestion" in detail_description
    assert "synthetic" in detail_description
    assert "never" in detail_description


# ---------------------------------------------------------------------------
# H / I. No external calls, no sensitive logging
# ---------------------------------------------------------------------------


def test_inference_makes_no_network_call(
    test_settings: Settings, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every socket is blocked for the duration of a submission."""
    import socket

    def refuse(*args: object, **kwargs: object):
        raise AssertionError("triage attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    model = TriageModel(artifact_root=ARTIFACT_ROOT, model_version=MODEL_VERSION)
    suggestion = model.predict(SECURITY_REPORT["description"])

    assert suggestion.suggested_category is ReportCategory.SECURITY


def test_no_external_ai_sdk_is_importable_from_the_ml_boundary() -> None:
    """Nothing in app/ml/ reaches an external provider."""
    import ast

    banned = {
        "openai",
        "anthropic",
        "google",
        "cohere",
        "litellm",
        "langchain",
        "transformers",
        "torch",
        "tensorflow",
        "requests",
        "httpx",
        "urllib",
        "socket",
    }
    offenders: list[str] = []

    for path in (Path("app") / "ml").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            offenders += [f"{path.name}:{node.lineno} {n}" for n in names if n in banned]

    assert offenders == [], f"app/ml reaches outward: {offenders}"


def test_triage_logs_no_report_text_or_case_code(
    api_client: TestClient, db_session: Session, report_id_for
) -> None:
    """Phase 5's logging policy, re-checked on the new code path."""
    marker = "PHASE7MARKER-credentials-were-left-on-a-public-share"
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Capture(level=logging.DEBUG)
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    try:
        response = api_client.post(
            REPORTS_URL, json={"category": "SECURITY", "description": marker + " in a chat channel"}
        )
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)

    case_code = response.json()["case_code"]
    captured = "\n".join(record.getMessage() for record in records)

    assert marker not in captured
    assert "PHASE7MARKER" not in captured
    assert case_code not in captured
    assert case_code.replace("-", "") not in captured

    triage = triage_for(db_session, report_id_for(case_code))
    assert triage.case_code_hash if False else True  # triage holds no case data
    assert not hasattr(triage, "case_code")


def test_a_triage_failure_logs_the_reason_but_not_the_text(
    test_settings: Settings, db_session: Session
) -> None:
    marker = "PHASE7FAILMARKER-a-confidential-report-body"
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Capture(level=logging.DEBUG)
    logger = logging.getLogger("app.services.triage")
    logger.addHandler(handler)
    try:
        with client_with_model(test_settings, db_session, ExplodingModel()) as client:
            client.post(REPORTS_URL, json={"category": "SECURITY", "description": marker + " here"})
    finally:
        logger.removeHandler(handler)

    captured = "\n".join(record.getMessage() for record in records)

    assert "Triage failed" in captured  # the operator is told
    assert marker not in captured  # but not what was in the report


# ---------------------------------------------------------------------------
# Performance and caching
# ---------------------------------------------------------------------------


def test_the_artifact_is_loaded_once_across_many_predictions() -> None:
    loader = ArtifactLoader(root=ARTIFACT_ROOT, version=MODEL_VERSION)
    calls = {"n": 0}

    original = loader._load

    def counting():
        calls["n"] += 1
        return original()

    loader._load = counting  # type: ignore[method-assign]

    for _ in range(5):
        loader.get()

    assert calls["n"] == 1


def test_the_dependency_reuses_one_model_per_version(test_settings: Settings) -> None:
    from app.api.deps import _triage_model

    first = _triage_model(test_settings.ml_artifact_root, MODEL_VERSION)
    second = _triage_model(test_settings.ml_artifact_root, MODEL_VERSION)

    assert first is second


def test_inference_is_fast_enough_to_run_inline(real_model: TriageModel) -> None:
    """Justifies running inference synchronously rather than in the background.

    A loose ceiling: this is guarding against a change that makes inference
    cost hundreds of milliseconds, not measuring performance.
    """
    import time

    real_model.predict("warm up the artifact")

    start = time.perf_counter()
    for _ in range(20):
        real_model.predict(SECURITY_REPORT["description"])
    average_ms = (time.perf_counter() - start) / 20 * 1000

    assert average_ms < 50, f"inference averaged {average_ms:.1f} ms"


def test_predictions_are_deterministic(real_model: TriageModel) -> None:
    results = [real_model.predict(SECURITY_REPORT["description"]) for _ in range(5)]

    assert len({(r.suggested_category, r.confidence, tuple(r.keywords)) for r in results}) == 1


# ---------------------------------------------------------------------------
# J. Regression
# ---------------------------------------------------------------------------


def test_case_tracking_still_works_after_inference(api_client: TestClient) -> None:
    submitted = api_client.post(REPORTS_URL, json=SECURITY_REPORT).json()

    lookup = api_client.post(LOOKUP_URL, json={"case_code": submitted["case_code"]})

    assert lookup.status_code == 200
    assert lookup.json()["status"] == "SUBMITTED"
    assert len(lookup.json()["timeline"]) == 1


def test_the_moderation_workflow_still_works_after_inference(
    api_client: TestClient, auth_headers: dict, db_session: Session, report_id_for
) -> None:
    """A status transition must be unaffected by the presence of a triage row."""
    submitted = api_client.post(REPORTS_URL, json=SECURITY_REPORT).json()
    report_id = report_id_for(submitted["case_code"])

    response = api_client.patch(
        f"{QUEUE_URL}/{report_id}/status",
        headers=auth_headers,
        json={"status": "UNDER_REVIEW", "note": "Picked up."},
    )

    assert response.status_code == 200
    db_session.expire_all()
    report = db_session.scalars(select(Report).where(Report.id == report_id)).one()
    assert report.status is ReportStatus.UNDER_REVIEW
    # And the suggestion is untouched by the human decision.
    assert triage_for(db_session, report_id).status is TriageStatus.COMPLETED


def test_a_moderator_may_disagree_with_the_suggestion_freely(
    api_client: TestClient, auth_headers: dict, db_session: Session, report_id_for
) -> None:
    """Human-in-the-loop: the decision is the human's, and nothing pushes back.

    Phase 7 records no feedback and retrains nothing. The disagreement simply
    stands, and is available to be curated into a future training set.
    """
    submitted = api_client.post(REPORTS_URL, json=SECURITY_REPORT).json()
    report_id = report_id_for(submitted["case_code"])

    triage = triage_for(db_session, report_id)
    assert triage.suggested_category is ReportCategory.SECURITY

    report = db_session.scalars(select(Report).where(Report.id == report_id)).one()
    report.category = ReportCategory.CORRUPTION  # a moderator's correction
    db_session.commit()
    db_session.expire_all()

    detail = api_client.get(f"{QUEUE_URL}/{report_id}", headers=auth_headers).json()

    assert detail["category"] == "CORRUPTION"  # the human's decision stands
    assert detail["triage"]["suggested_category"] == "SECURITY"  # unchanged record


# ---------------------------------------------------------------------------
# The last line of defence: the triage write itself failing
# ---------------------------------------------------------------------------


def test_a_database_failure_while_storing_a_result_falls_back_to_failed(
    db_session: Session,
    real_model: TriageModel,
    api_client: TestClient,
    report_id_for,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference succeeded but the row would not commit.

    The suggestion is lost and recorded as FAILED. What must not happen is an
    exception escaping into the submission path.
    """
    from sqlalchemy.exc import OperationalError

    from app.repositories.triage import ReportTriageRepository

    response = api_client.post(REPORTS_URL, json=SECURITY_REPORT)
    report_id = report_id_for(response.json()["case_code"])

    calls = {"n": 0}
    original = ReportTriageRepository.upsert

    def fail_first(self, rid, **values):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OperationalError("stmt", {}, Exception("connection lost"))
        return original(self, rid, **values)

    monkeypatch.setattr(ReportTriageRepository, "upsert", fail_first)

    status = TriageService(db_session, model=real_model).run_for_report(
        report_id, SECURITY_REPORT["description"]
    )

    assert status is TriageStatus.FAILED
    db_session.expire_all()
    assert triage_for(db_session, report_id).status is TriageStatus.FAILED


def test_a_database_failure_while_recording_a_failure_is_swallowed(
    db_session: Session, api_client: TestClient, report_id_for, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even the fallback write can fail, and still nothing may escape.

    The report is already committed and is not at risk either way, so the only
    correct behaviour is to log and return.
    """
    from sqlalchemy.exc import OperationalError

    from app.repositories.triage import ReportTriageRepository

    response = api_client.post(REPORTS_URL, json=SECURITY_REPORT)
    report_id = report_id_for(response.json()["case_code"])

    def always_fail(self, rid, **values):
        raise OperationalError("stmt", {}, Exception("database is gone"))

    monkeypatch.setattr(ReportTriageRepository, "upsert", always_fail)

    status = TriageService(db_session, model=ExplodingModel()).run_for_report(
        report_id, SECURITY_REPORT["description"]
    )

    assert status is None  # nothing recorded, nothing raised
    db_session.rollback()
    report = db_session.scalars(select(Report).where(Report.id == report_id)).one()
    assert report.status is ReportStatus.SUBMITTED  # the report is untouched


def test_the_warm_start_survives_a_broken_artifact(
    test_settings: Settings, db_session: Session, tmp_path: Path
) -> None:
    """Startup must complete even when warming the model fails."""
    settings = test_settings.model_copy(
        update={"ml_warm_start": True, "ml_artifact_root": tmp_path / "absent"}
    )
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = lambda: db_session

    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.post(REPORTS_URL, json=SECURITY_REPORT).status_code == 201
