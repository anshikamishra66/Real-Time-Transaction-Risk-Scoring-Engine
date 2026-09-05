"""Shared configuration so training and serving can never drift apart on
feature order, model paths, or risk-tier thresholds."""
from __future__ import annotations

from pathlib import Path

from features.feature_store import FeatureStore

MODELS_DIR = Path(__file__).resolve().parent / "artifacts"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
BEHAVIORAL_COLUMNS = FeatureStore.FEATURE_NAMES
# Exact column order fed to both models. Changing this requires retraining.
FEATURE_COLUMNS = PCA_COLUMNS + ["Amount"] + BEHAVIORAL_COLUMNS

SUPERVISED_MODEL_PATH = MODELS_DIR / "supervised_xgb.json"
ANOMALY_MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"
ANOMALY_SCALER_PATH = MODELS_DIR / "anomaly_scaler.joblib"
ENSEMBLE_CONFIG_PATH = MODELS_DIR / "ensemble_config.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.parquet"

# Ensemble weights: supervised model is the primary, well-calibrated signal;
# the anomaly detector is a secondary vote for patterns the supervised model
# was never trained on (it only ever saw historical fraud typologies).
DEFAULT_SUPERVISED_WEIGHT = 0.7
DEFAULT_ANOMALY_WEIGHT = 0.3

# Risk tiers over the final 0-100 ensemble score.
LOW_RISK_MAX = 30
MEDIUM_RISK_MAX = 70

# SHAP is only computed for transactions at/above this tier to bound latency
# on the hot path -- see README "Why SHAP only on flagged transactions".
SHAP_MIN_TIER = "medium"
