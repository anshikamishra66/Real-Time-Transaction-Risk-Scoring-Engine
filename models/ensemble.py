"""
Combines the supervised classifier and the unsupervised anomaly detector into
a single 0-100 risk score, and maps that score to a risk tier.

Why an ensemble instead of a single model: the supervised model is precise
about fraud patterns it has seen labeled examples of, but by definition it
cannot flag a genuinely novel fraud typology as fraud -- it was never
penalized for missing that pattern during training. The Isolation Forest
doesn't need labels at all; it flags "this doesn't look like normal behavior
for this account" regardless of whether that specific pattern ever appeared
in the fraud-labeled training data. Blending them (weighted, supervised-
dominant) gives one signal from "matches known fraud" and another from
"deviates from this account's normal behavior," which is strictly more
information than either alone -- at the cost of one extra inference call,
which is cheap relative to the SHAP call that only runs for flagged
transactions.

Why a 0-100 tiered score instead of a binary fraud/not-fraud flag: a binary
flag forces the same action (block) onto a transaction that is 51% likely
fraud and one that is 99% likely fraud, and forces a compliance team to
review every single flag with no prioritization. Tiers let cheap, reversible
actions (log for review) trigger at a lower bar than expensive, customer-
visible ones (auto-hold + compliance alert), which is how real payments risk
systems are actually operated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from models.config import (
    ANOMALY_MODEL_PATH,
    ANOMALY_SCALER_PATH,
    DEFAULT_ANOMALY_WEIGHT,
    DEFAULT_SUPERVISED_WEIGHT,
    ENSEMBLE_CONFIG_PATH,
    FEATURE_COLUMNS,
    LOW_RISK_MAX,
    MEDIUM_RISK_MAX,
    SUPERVISED_MODEL_PATH,
)
from models.train_anomaly import anomaly_score_01


def risk_tier(score: float) -> str:
    if score < LOW_RISK_MAX:
        return "low"
    if score < MEDIUM_RISK_MAX:
        return "medium"
    return "high"


@dataclass
class EnsembleResult:
    supervised_prob: float
    anomaly_score: float
    risk_score: int
    risk_tier: str


class RiskEnsemble:
    def __init__(
        self,
        supervised_model: xgb.XGBClassifier,
        anomaly_model,
        anomaly_scaler,
        supervised_weight: float = DEFAULT_SUPERVISED_WEIGHT,
        anomaly_weight: float = DEFAULT_ANOMALY_WEIGHT,
    ) -> None:
        self.supervised_model = supervised_model
        self.anomaly_model = anomaly_model
        self.anomaly_scaler = anomaly_scaler
        self.supervised_weight = supervised_weight
        self.anomaly_weight = anomaly_weight

    @classmethod
    def load(cls) -> "RiskEnsemble":
        supervised_model = xgb.XGBClassifier()
        supervised_model.load_model(str(SUPERVISED_MODEL_PATH))
        anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
        anomaly_scaler = joblib.load(ANOMALY_SCALER_PATH)

        supervised_weight = DEFAULT_SUPERVISED_WEIGHT
        anomaly_weight = DEFAULT_ANOMALY_WEIGHT
        if ENSEMBLE_CONFIG_PATH.exists():
            cfg = json.loads(ENSEMBLE_CONFIG_PATH.read_text())
            supervised_weight = cfg.get("supervised_weight", supervised_weight)
            anomaly_weight = cfg.get("anomaly_weight", anomaly_weight)

        return cls(supervised_model, anomaly_model, anomaly_scaler, supervised_weight, anomaly_weight)

    def score_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[FEATURE_COLUMNS]
        supervised_prob = self.supervised_model.predict_proba(X)[:, 1]
        anomaly = anomaly_score_01(self.anomaly_model, self.anomaly_scaler, X)

        blended = self.supervised_weight * supervised_prob + self.anomaly_weight * anomaly
        risk_score = np.clip(np.round(blended * 100), 0, 100).astype(int)
        tiers = [risk_tier(s) for s in risk_score]

        return pd.DataFrame({
            "supervised_prob": supervised_prob,
            "anomaly_score": anomaly,
            "risk_score": risk_score,
            "risk_tier": tiers,
        })

    def score_one(self, features: dict[str, float]) -> EnsembleResult:
        row = pd.DataFrame([features])[FEATURE_COLUMNS]
        result = self.score_batch(row).iloc[0]
        return EnsembleResult(
            supervised_prob=float(result["supervised_prob"]),
            anomaly_score=float(result["anomaly_score"]),
            risk_score=int(result["risk_score"]),
            risk_tier=str(result["risk_tier"]),
        )
