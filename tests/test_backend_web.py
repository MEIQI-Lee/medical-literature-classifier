"""Smoke tests for the member-5 HTTP facade without loading the large model."""

from fastapi.testclient import TestClient

from backend import main


class _Result:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


class _Service:
    def __init__(self, payload):
        self.payload = payload

    def classify_text(self, *args, **kwargs):
        if callback := kwargs.get("on_step"):
            callback({"step_id": "prepare:1", "name": "prepare_text", "status": "done", "detail": "文本已整理"})
        return _Result(self.payload)

    def classify_file(self, *args, **kwargs):
        if callback := kwargs.get("on_step"):
            callback({"step_id": "document:1", "name": "parse_document", "status": "done", "detail": "文档已解析"})
        return _Result(self.payload)

    def run(self, *args, **kwargs):
        return _Result(self.payload)


def _payload():
    return {
        "success": True,
        "data": {
            "prediction": {
                "top1": {"code": "R54", "name": "cardiovascular", "confidence": 0.9},
                "topk": [],
            }
        },
        "error": None,
        "steps": [],
        "logs": [],
    }


def test_web_routes_return_saved_reports_and_static_ui():
    client = TestClient(main.app)
    assert client.get("/").status_code == 200
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/statistics").json()["dataset"]["total"] == 5450
    assert {item["id"] for item in client.get("/api/metrics").json()["models"]} == {
        "medbert_v20",
        "macbert_v20",
    }


def test_classification_and_upload_delegate_to_service(monkeypatch):
    monkeypatch.setattr(main, "get_agent", lambda: _Service(_payload()))
    client = TestClient(main.app)
    text_response = client.post("/api/classify", json={"text": "sufficient medical text", "top_k": 3})
    file_response = client.post(
        "/api/classify/file?top_k=3",
        files={"file": ("case.txt", b"sufficient medical text", "text/plain")},
    )
    assert text_response.status_code == 200
    assert text_response.json()["data"]["prediction"]["top1"]["code"] == "R54"
    assert file_response.status_code == 200


def test_upload_rejects_unknown_suffix():
    client = TestClient(main.app)
    response = client.post(
        "/api/classify/file",
        files={"file": ("unsafe.exe", b"not a document", "application/octet-stream")},
    )
    assert response.status_code == 422


def test_streaming_routes_send_agent_steps_and_final_result(monkeypatch):
    monkeypatch.setattr(main, "get_agent", lambda: _Service(_payload()))
    client = TestClient(main.app)
    text_response = client.post("/api/classify/stream", json={"text": "sufficient medical text"})
    file_response = client.post(
        "/api/classify/file/stream",
        files={"file": ("case.txt", b"sufficient medical text", "text/plain")},
    )
    assert text_response.status_code == 200
    assert "event: step" in text_response.text
    assert "event: result" in text_response.text
    assert "prepare_text" in text_response.text
    assert file_response.status_code == 200
    assert "parse_document" in file_response.text
