from datetime import datetime, timedelta

import pandas as pd

from features.feature_store import FeatureStore, compute_batch_features


def test_first_transaction_has_no_history_signal():
    store = FeatureStore()
    feats = store.compute_and_update(
        account_id="a1", device_id="d1", geo_region="US-NY",
        timestamp=datetime(2024, 1, 1), amount=50.0,
    )
    assert feats["velocity_count_1m"] == 0
    assert feats["velocity_count_5m"] == 0
    assert feats["velocity_count_1h"] == 0
    assert feats["is_new_device"] == 0.0
    assert feats["is_new_geo"] == 0.0
    assert feats["time_since_last_tx_sec"] > 0


def test_velocity_counts_respect_window_boundaries():
    store = FeatureStore()
    base = datetime(2024, 1, 1)
    store.compute_and_update("a1", "d1", "US-NY", base, 10.0)
    store.compute_and_update("a1", "d1", "US-NY", base + timedelta(seconds=30), 10.0)
    store.compute_and_update("a1", "d1", "US-NY", base + timedelta(minutes=3), 10.0)
    feats = store.compute_and_update("a1", "d1", "US-NY", base + timedelta(minutes=4, seconds=30), 10.0)

    assert feats["velocity_count_1m"] == 0  # nothing within 60s before this tx
    assert feats["velocity_count_5m"] == 3  # all 3 prior tx are within 5 minutes
    assert feats["velocity_count_1h"] == 3


def test_new_device_flag_triggers_only_on_unseen_device():
    store = FeatureStore()
    base = datetime(2024, 1, 1)
    store.compute_and_update("a1", "d1", "US-NY", base, 10.0)
    same_device = store.compute_and_update("a1", "d1", "US-NY", base + timedelta(minutes=1), 10.0)
    new_device = store.compute_and_update("a1", "d2", "US-NY", base + timedelta(minutes=2), 10.0)

    assert same_device["is_new_device"] == 0.0
    assert new_device["is_new_device"] == 1.0


def test_new_geo_flag_triggers_only_on_unseen_region():
    store = FeatureStore()
    base = datetime(2024, 1, 1)
    store.compute_and_update("a1", "d1", "US-NY", base, 10.0)
    new_geo = store.compute_and_update("a1", "d1", "RU-MOW", base + timedelta(minutes=1), 10.0)

    assert new_geo["is_new_geo"] == 1.0


def test_amount_zscore_flags_outlier_relative_to_own_history():
    store = FeatureStore()
    base = datetime(2024, 1, 1)
    for i, amount in enumerate([18.0, 19.0, 20.0, 21.0, 22.0]):
        store.compute_and_update("a1", "d1", "US-NY", base + timedelta(seconds=i), amount)
    outlier = store.compute_and_update("a1", "d1", "US-NY", base + timedelta(seconds=10), 5000.0)

    assert outlier["amount_zscore"] > 5  # clipped at 10, but should be large and positive


def test_accounts_are_isolated_from_each_other():
    store = FeatureStore()
    base = datetime(2024, 1, 1)
    store.compute_and_update("a1", "d1", "US-NY", base, 10.0)
    feats_a2 = store.compute_and_update("a2", "d1", "US-NY", base + timedelta(seconds=1), 10.0)

    # a2's first transaction should show no history even though a1 just
    # transacted a moment ago on the same device/geo.
    assert feats_a2["velocity_count_1h"] == 0
    assert feats_a2["is_new_device"] == 0.0


def test_compute_batch_features_matches_incremental_semantics():
    base = datetime(2024, 1, 1)
    df = pd.DataFrame({
        "account_id": ["a1", "a1", "a1"],
        "device_id": ["d1", "d1", "d2"],
        "geo_region": ["US-NY", "US-NY", "US-NY"],
        "timestamp": [base, base + timedelta(seconds=30), base + timedelta(minutes=1)],
        "Amount": [10.0, 12.0, 14.0],
    })
    out = compute_batch_features(df)
    assert list(out["is_new_device"]) == [0.0, 0.0, 1.0]
    assert len(out) == 3
