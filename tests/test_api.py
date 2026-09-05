from fastapi.testclient import TestClient

from assessment.api import create_app
from assessment.runtime import Runtime


def test_health_and_access_control(settings):
    settings.app_access_token = "private-app-token"
    # Assignment validation is intentionally not relied upon by settings; construct a SecretStr.
    from pydantic import SecretStr

    settings.app_access_token = SecretStr("private-app-token")
    client = TestClient(create_app(Runtime(settings)))
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer private-app-token"}).status_code == 200


def test_changed_endpoint_cannot_reuse_default_key(settings):
    client = TestClient(create_app(Runtime(settings)))
    response = client.post("/qa", json={"question": "Hello"}, headers={"X-Provider-Url": "https://other.example/v1"})
    assert response.status_code == 422
    assert "own credential" in response.text


def test_invalid_upload_is_rejected(settings):
    client = TestClient(create_app(Runtime(settings)))
    response = client.post("/knowledge/ingest", files={"file": ("program.exe", b"not a document")})
    assert response.status_code == 422
