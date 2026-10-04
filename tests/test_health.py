"""Tests for the health endpoint and the OpenAPI surface."""

from fastapi.testclient import TestClient

from app import __version__

HEALTH_URL = "/api/v1/health"


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get(HEALTH_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["environment"] == "test"


def test_health_is_public(client: TestClient) -> None:
    """Health must not require authentication."""
    response = client.get(HEALTH_URL)

    assert response.status_code == 200


def test_unknown_route_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404


def test_openapi_schema_is_served(client: TestClient) -> None:
    """The GDG task requires Swagger/OpenAPI documentation."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["version"] == __version__
    assert HEALTH_URL in schema["paths"]


def test_swagger_ui_is_served(client: TestClient) -> None:
    response = client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
