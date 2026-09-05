"""
Ties together the live feature store, the ML ensemble, the rules engine,
SHAP explainability, and automated actions into one function per transaction.

Holds process-wide singletons (the live FeatureStore and the loaded models)
so state persists across requests within one running API process -- this is
what makes velocity/new-device/new-geo features meaningful in a live demo:
each account's history accumulates across calls to /score-transaction.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from api import actions, db
from api.rules_engine import apply_rules
from api.schemas import ScoreResponse, ShapContribution, TransactionRequest
from features.feature_store import FeatureStore
from models.config import PCA_COLUMNS, SHAP_MIN_TIER
from models.ensemble import RiskEnsemble, risk_tier
from models.explain import explain_transaction

_TIER_RANK = {"low": 0, "medium": 1, "high": 2}

feature_store = FeatureStore()
_ensemble: RiskEnsemble | None = None


def load_models() -> None:
    global _ensemble
    _ensemble = RiskEnsemble.load()


def get_ensemble() -> RiskEnsemble:
    if _ensemble is None:
        raise RuntimeError("Models not loaded yet -- call load_models() at startup.")
    return _ensemble


def score_transaction(request: TransactionRequest) -> ScoreResponse:
    transaction_id = request.transaction_id or f"tx-{uuid.uuid4().hex[:12]}"
    timestamp = request.timestamp or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    behavioral = feature_store.compute_and_update(
        account_id=request.account_id,
        device_id=request.device_id,
        geo_region=request.geo_region,
        timestamp=timestamp,
        amount=request.amount,
    )

    pca_features = {col: 0.0 for col in PCA_COLUMNS}
    if request.pca_features:
        pca_features.update({k: v for k, v in request.pca_features.items() if k in pca_features})

    all_features = {**pca_features, "Amount": request.amount, **behavioral}

    ensemble = get_ensemble()
    ml_result = ensemble.score_one(all_features)

    final_score, rules_triggered = apply_rules(behavioral, request.amount, ml_result.risk_score)
    final_tier = risk_tier(final_score)
    # A rule can only push the tier up, never down, from what the raw ML
    # score already implied.
    if _TIER_RANK[ml_result.risk_tier] > _TIER_RANK[final_tier]:
        final_tier = ml_result.risk_tier

    action = actions.decide_action(final_tier)

    shap_explanation = None
    if _TIER_RANK[final_tier] >= _TIER_RANK[SHAP_MIN_TIER]:
        raw = explain_transaction(ensemble.supervised_model, all_features)
        shap_explanation = [ShapContribution(**item) for item in raw]

    compliance_alert = actions.build_compliance_alert(
        transaction_id, final_score, final_tier, rules_triggered
    )

    response = ScoreResponse(
        transaction_id=transaction_id,
        risk_score=final_score,
        risk_tier=final_tier,
        action=action,
        supervised_prob=ml_result.supervised_prob,
        anomaly_score=ml_result.anomaly_score,
        rules_triggered=rules_triggered,
        features=behavioral,
        shap_explanation=shap_explanation,
        compliance_alert=compliance_alert,
    )

    db.insert_transaction({
        "transaction_id": transaction_id,
        "account_id": request.account_id,
        "device_id": request.device_id,
        "geo_region": request.geo_region,
        "amount": request.amount,
        "timestamp": timestamp.isoformat(),
        "risk_score": final_score,
        "risk_tier": final_tier,
        "action": action,
        "supervised_prob": ml_result.supervised_prob,
        "anomaly_score": ml_result.anomaly_score,
        "rules_triggered": rules_triggered,
        "features": behavioral,
        "shap_explanation": [c.model_dump() for c in shap_explanation] if shap_explanation else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    return response
