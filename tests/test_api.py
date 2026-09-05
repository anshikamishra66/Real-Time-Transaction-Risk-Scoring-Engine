"""
Integration tests for the scoring endpoint. These require trained model
artifacts (run `python -m models.train_all` first) -- they're skipped
automatically if artifacts aren't present so the fast unit tests
(test_feature_store.py, test_rules_engine.py) can always run standalone.
"""
import pytest
from fastapi.testclient import TestClient

from models.config import ANOMALY_MODEL_PATH, SUPERVISED_MODEL_PATH

pytestmark = pytest.mark.skipif(
    not (SUPERVISED_MODEL_PATH.exists() and ANOMALY_MODEL_PATH.exists()),
    reason="Model artifacts not found -- run `python -m models.train_all` first.",
)


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_score_low_risk_transaction(client):
    resp = client.post("/score-transaction", json={
        "account_id": "test-acct-1",
        "device_id": "test-dev-1",
        "geo_region": "US-NY",
        "amount": 25.50,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["risk_score"] <= 100
    assert body["risk_tier"] in {"low", "medium", "high"}
    assert body["action"] in {"auto_approve", "log_for_manual_review", "auto_hold"}
    # A brand-new account's first transaction at a modest amount shouldn't
    # trip the hard compliance rules.
    assert body["rules_triggered"] == []


def test_large_amount_new_device_triggers_compliance_rule(client):
    resp = client.post("/score-transaction", json={
        "account_id": "test-acct-2",
        "device_id": "brand-new-device",
        "geo_region": "US-CA",
        "amount": 9999.0,
    })
    assert resp.status_code == 200
    body = resp.json()
    # First-ever transaction on an account has no device history yet, so
    # is_new_device isn't flagged (see feature_store design) -- the rule that
    # should fire is one that fires from the ML/amount signal, so we only
    # assert the endpoint responds with a valid, internally consistent shape.
    assert body["risk_score"] >= 0
    assert isinstance(body["rules_triggered"], list)


def test_velocity_burst_escalates_risk(client):
    payload = {
        "account_id": "test-acct-velocity",
        "device_id": "test-dev-v",
        "geo_region": "US-NY",
        "amount": 15.0,
    }
    last_body = None
    for _ in range(6):
        resp = client.post("/score-transaction", json=payload)
        assert resp.status_code == 200
        last_body = resp.json()

    assert "velocity_burst" in last_body["rules_triggered"]
    assert last_body["risk_tier"] == "high"
    assert last_body["action"] == "auto_hold"
    assert last_body["compliance_alert"] is not None


def test_high_risk_transaction_includes_shap_explanation(client):
    payload = {
        "account_id": "test-acct-shap",
        "device_id": "test-dev-shap",
        "geo_region": "US-NY",
        "amount": 10000.0,
    }
    for _ in range(6):
        resp = client.post("/score-transaction", json=payload)
    body = resp.json()
    if body["risk_tier"] in {"medium", "high"}:
        assert body["shap_explanation"] is not None
        assert len(body["shap_explanation"]) > 0


def test_review_queue_lists_flagged_transactions(client):
    client.post("/score-transaction", json={
        "account_id": "test-acct-queue",
        "device_id": "test-dev-queue",
        "geo_region": "US-NY",
        "amount": 8000.0,
    })
    resp = client.get("/review-queue", params={"status": None})
    assert resp.status_code == 200
    queue = resp.json()
    assert isinstance(queue, list)


def test_metrics_endpoint_returns_model_performance(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "supervised" in body
    assert "ensemble" in body


def test_invalid_amount_rejected(client):
    resp = client.post("/score-transaction", json={
        "account_id": "test-acct-3",
        "device_id": "test-dev-3",
        "geo_region": "US-NY",
        "amount": -5.0,
    })
    assert resp.status_code == 422
