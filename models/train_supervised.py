"""
Trains the supervised fraud classifier (XGBoost) on labeled transactions.

Class imbalance handling: fraud is <1% of transactions here (real dataset:
~0.17%). We use `scale_pos_weight` (class weighting) rather than SMOTE,
because SMOTE synthesizes new points by interpolating between existing fraud
examples in feature space -- fine for a handful of dense numeric features,
but our fraud examples are sparse in a mostly-PCA'd 35-dimensional space, and
interpolated points there don't correspond to any transaction that could
plausibly occur. `scale_pos_weight` instead reweights the *real* fraud
examples we have, which is both cheaper and avoids inventing data. See
README "How class imbalance was handled" for the full rationale.

Model selection uses precision-recall AUC, not accuracy or ROC-AUC: with
<1% positives, a model that always predicts "legit" gets ~99.8% accuracy and
a deceptively high ROC-AUC while catching zero fraud. PR-AUC is sensitive to
exactly the failure mode we care about (precision collapsing as we try to
catch more fraud).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
)

from features.feature_store import compute_batch_features
from models.config import FEATURE_COLUMNS, SUPERVISED_MODEL_PATH

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    if "velocity_count_1h" not in df.columns:
        df = compute_batch_features(df)
    return df


def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    """Chooses the probability threshold that maximizes F1 on the PR curve,
    then reports precision/recall/F1 at that operating point."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best_idx = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
    pr_auc = float(auc(recall, precision))
    return best_threshold, {
        "precision_at_best_f1": float(precision[best_idx]),
        "recall_at_best_f1": float(recall[best_idx]),
        "f1_at_best_f1": float(f1[best_idx]),
        "pr_auc": pr_auc,
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }


def train(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_df = _prep(train_df)
    test_df = _prep(test_df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["Class"].to_numpy()
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["Class"].to_numpy()

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold, metrics = pick_threshold(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    metrics.update({
        "threshold": threshold,
        "scale_pos_weight": scale_pos_weight,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "false_positive_rate": fp / max(fp + tn, 1),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_fraud_train": n_pos,
        "n_fraud_test": int(y_test.sum()),
    })

    model.save_model(str(SUPERVISED_MODEL_PATH))
    return metrics


def main() -> None:
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    metrics = train(train_df, test_df)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
