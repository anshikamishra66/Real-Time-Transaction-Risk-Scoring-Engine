"""
SHAP explainability for the supervised model, computed only for transactions
that already cleared the "medium" risk tier or above.

Why gate SHAP behind the risk tier instead of always computing it: TreeSHAP on
a single row is cheap in isolation, but it is not free, and on the hot scoring
path (POST /score-transaction) the overwhelming majority of transactions are
low risk and get auto-approved -- there is no compliance or customer-facing
reason to explain a decision nobody will ever review. Computing SHAP for
100% of traffic would multiply the average request latency for zero benefit
on ~97%+ of requests (fraud/flagged rates are typically single-digit percent
at most). Gating it behind the tier means the only requests that pay the SHAP
cost are exactly the ones a human (or a compliance alert) will actually look
at, where an auditable "why" is the whole point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from models.config import FEATURE_COLUMNS

_explainer_cache: dict[int, shap.TreeExplainer] = {}


def get_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    key = id(model)
    explainer = _explainer_cache.get(key)
    if explainer is None:
        explainer = shap.TreeExplainer(model)
        _explainer_cache[key] = explainer
    return explainer


def explain_transaction(model: xgb.XGBClassifier, features: dict[str, float], top_k: int = 5) -> list[dict]:
    """Returns the top-k features driving this transaction's fraud probability,
    ranked by |SHAP value|, as a list of {feature, value, shap_contribution}."""
    row = pd.DataFrame([features])[FEATURE_COLUMNS]
    explainer = get_explainer(model)
    shap_values = explainer.shap_values(row)
    if isinstance(shap_values, list):  # multiclass API shape guard
        shap_values = shap_values[-1]
    values = np.asarray(shap_values).reshape(-1)

    order = np.argsort(-np.abs(values))[:top_k]
    return [
        {
            "feature": FEATURE_COLUMNS[i],
            "value": float(row.iloc[0, i]),
            "shap_contribution": float(values[i]),
        }
        for i in order
    ]
