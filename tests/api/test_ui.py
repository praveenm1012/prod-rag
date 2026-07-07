"""UI route tests."""

from fastapi.testclient import TestClient


def test_test_console_page(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "API Test Console" in response.text


def test_test_console_alias(client: TestClient) -> None:
    response = client.get("/ui")

    assert response.status_code == 200
    assert "API Test Console" in response.text


def test_static_assets(client: TestClient) -> None:
    for asset in ("styles.css", "app.js"):
        response = client.get(f"/static/{asset}")
        assert response.status_code == 200
        assert response.text
