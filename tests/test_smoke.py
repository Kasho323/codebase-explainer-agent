from fastapi.testclient import TestClient

from codebase_explainer.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_placeholder_returns_question():
    response = client.post(
        "/chat",
        json={"repo_url": "https://github.com/encode/httpx", "question": "what is this?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["repo_url"] == "https://github.com/encode/httpx"
    assert "question" in body
