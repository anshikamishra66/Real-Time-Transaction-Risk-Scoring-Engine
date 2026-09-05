from api.rules_engine import apply_rules

BASE_FEATURES = {
    "velocity_count_1m": 0.0,
    "velocity_count_5m": 0.0,
    "velocity_count_1h": 0.0,
    "amount_zscore": 0.0,
    "time_since_last_tx_sec": 3600.0,
    "is_new_device": 0.0,
    "is_new_geo": 0.0,
}


def test_no_rules_triggered_leaves_score_unchanged():
    score, triggered = apply_rules(BASE_FEATURES, amount=50.0, ml_risk_score=10)
    assert score == 10
    assert triggered == []


def test_large_amount_new_device_forces_high_floor():
    features = {**BASE_FEATURES, "is_new_device": 1.0}
    score, triggered = apply_rules(features, amount=5000.0, ml_risk_score=5)
    assert score >= 95
    assert "large_amount_new_device" in triggered


def test_velocity_burst_forces_floor_even_with_low_ml_score():
    features = {**BASE_FEATURES, "velocity_count_1m": 6}
    score, triggered = apply_rules(features, amount=20.0, ml_risk_score=2)
    assert score >= 85
    assert "velocity_burst" in triggered


def test_rules_never_lower_a_higher_ml_score():
    score, triggered = apply_rules(BASE_FEATURES, amount=10.0, ml_risk_score=98)
    assert score == 98
    assert triggered == []


def test_multiple_rules_can_trigger_at_once():
    features = {**BASE_FEATURES, "is_new_device": 1.0, "is_new_geo": 1.0}
    score, triggered = apply_rules(features, amount=5000.0, ml_risk_score=0)
    assert "large_amount_new_device" in triggered
    assert "moderate_amount_new_device_new_geo" in triggered
    assert score >= 95
