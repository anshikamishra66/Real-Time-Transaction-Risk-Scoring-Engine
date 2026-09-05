"""
Orchestrates the full training pipeline: supervised classifier, anomaly
detector, then evaluates the combined ensemble on the held-out test set and
persists everything the serving layer and dashboard need:
  - models/artifacts/supervised_xgb.json      (XGBoost model)
  - models/artifacts/isolation_forest.joblib  (Isolation Forest)
  - models/artifacts/anomaly_scaler.joblib    (raw score -> [0,1] scaler)
  - models/artifacts/ensemble_config.json     (weights + supervised threshold)
  - models/artifacts/metrics.json             (everything the dashboard shows)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from features.feature_store import compute_batch_features
from models import train_anomaly, train_supervised
from models.config import (
    DEFAULT_ANOMALY_WEIGHT,
    DEFAULT_SUPERVISED_WEIGHT,
    ENSEMBLE_CONFIG_PATH,
    METRICS_PATH,
)
from models.ensemble import RiskEnsemble

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def evaluate_ensemble(ensemble: RiskEnsemble, test_df: pd.DataFrame) -> dict:
    y_test = test_df["Class"].to_numpy()
    scored = ensemble.score_batch(test_df)

    flagged = (scored["risk_tier"] != "low").to_numpy()  # medium or high -> some action taken
    high = (scored["risk_tier"] == "high").to_numpy()

    def _prf(pred: "pd.Series | pd.array") -> dict:
        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        tn = int(((pred == 0) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        fpr = fp / max(fp + tn, 1)
        return {
            "precision": precision, "recall": recall, "f1": f1,
            "false_positive_rate": fpr,
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        }

    return {
        "flagged_medium_or_high": _prf(flagged.astype(int)),
        "flagged_high_only": _prf(high.astype(int)),
        "risk_tier_counts": scored["risk_tier"].value_counts().to_dict(),
    }


def main() -> None:
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    print("Computing behavioral features (train)...")
    train_df = compute_batch_features(train_df)
    print("Computing behavioral features (test)...")
    test_df = compute_batch_features(test_df)

    print("\n=== Training supervised classifier ===")
    supervised_metrics = train_supervised.train(train_df, test_df)
    print(json.dumps(supervised_metrics, indent=2))

    print("\n=== Training anomaly detector ===")
    anomaly_metrics = train_anomaly.train(train_df, test_df)
    print(json.dumps(anomaly_metrics, indent=2))

    ensemble_config = {
        "supervised_weight": DEFAULT_SUPERVISED_WEIGHT,
        "anomaly_weight": DEFAULT_ANOMALY_WEIGHT,
        "supervised_threshold": supervised_metrics["threshold"],
    }
    ENSEMBLE_CONFIG_PATH.write_text(json.dumps(ensemble_config, indent=2))

    print("\n=== Evaluating ensemble on test set ===")
    ensemble = RiskEnsemble.load()
    ensemble_metrics = evaluate_ensemble(ensemble, test_df)
    print(json.dumps(ensemble_metrics, indent=2))

    metrics = {
        "supervised": supervised_metrics,
        "anomaly": anomaly_metrics,
        "ensemble": ensemble_metrics,
        "ensemble_config": ensemble_config,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
