from fastapi.testclient import TestClient

import src.app as app_module


class FakeAnalyzer:
    """Stands in for the real model so tests stay fast and offline."""

    model_name = "fake-model"

    def load(self):
        return None

    def analyze(self, text):
        return {"label": "POSITIVE", "score": 0.99}

    def analyze_batch(self, texts):
        return [{"label": "POSITIVE", "score": 0.99} for _ in texts]


# swap the real model out before the app boots
app_module.get_analyzer = lambda: FakeAnalyzer()

client = TestClient(app_module.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["task"] == "sentiment-analysis"


def test_analyze_returns_label_and_score():
    r = client.post("/analyze", json={"text": "I love this product"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= body["score"] <= 1.0


def test_analyze_batch_matches_input_length():
    texts = ["great day", "terrible service", "it was fine"]
    r = client.post("/analyze/batch", json={"texts": texts})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert len(body["results"]) == 3


def test_analyze_rejects_blank_text():
    r = client.post("/analyze", json={"text": "   "})
    assert r.status_code == 422


def test_analyze_rejects_missing_text():
    r = client.post("/analyze", json={})
    assert r.status_code == 422


def test_batch_rejects_too_many_texts():
    r = client.post("/analyze/batch", json={"texts": ["ok"] * 51})
    assert r.status_code == 422
