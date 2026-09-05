"""
Trains the unsupervised anomaly detector (Isolation Forest).

This is deliberately fit only on transactions labeled legitimate, so it
learns a model of "normal" behavior rather than "normal vs. the specific
fraud patterns in this dataset." That is what lets it act as a secondary
signal for fraud typologies the supervised model has never seen -- a new
fraud pattern still looks anomalous relative to normal behavior even though
no similar fraud example existed in the supervised model's training data.

Isolation Forest's raw decision_function isn't on a stable, interpretable
scale, so we fit a MinMaxScaler on the training scores and persist it
alongside the model -- at serving time we always know how to map a raw score
into the same [0, 1] anomaly-score range the ensemble expects.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from features.feature_store import compute_batch_features
from models.config import ANOMALY_MODEL_PATH, ANOMALY_SCALER_PATH, FEATURE_COLUMNS

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    if "velocity_count_1h" not in df.columns:
        df = compute_batch_features(df)
    return df


def anomaly_score_01(model: IsolationForest, scaler: MinMaxScaler, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous, scaled to [0, 1]."""
    raw = -model.decision_function(X)  # sklearn: higher decision_function = more normal
    scaled = scaler.transform(raw.reshape(-1, 1)).ravel()
    return np.clip(scaled, 0.0, 1.0)


def train(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_df = _prep(train_df)
    test_df = _prep(test_df)

    normal_train = train_df[train_df["Class"] == 0]
    X_train_normal = normal_train[FEATURE_COLUMNS]

    contamination = max(train_df["Class"].mean(), 0.001)
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=min(contamination, 0.05),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_normal)

    train_scores_raw = -model.decision_function(X_train_normal)
    scaler = MinMaxScaler()
    scaler.fit(train_scores_raw.reshape(-1, 1))

    joblib.dump(model, ANOMALY_MODEL_PATH)
    joblib.dump(scaler, ANOMALY_SCALER_PATH)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Class"].to_numpy()
    scores = anomaly_score_01(model, scaler, X_test)

    metrics = {
        "mean_anomaly_score_fraud": float(scores[y_test == 1].mean()) if (y_test == 1).any() else None,
        "mean_anomaly_score_legit": float(scores[y_test == 0].mean()) if (y_test == 0).any() else None,
        "n_train_normal": int(len(X_train_normal)),
        "contamination": float(min(contamination, 0.05)),
    }
    return metrics


def main() -> None:
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    metrics = train(train_df, test_df)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
