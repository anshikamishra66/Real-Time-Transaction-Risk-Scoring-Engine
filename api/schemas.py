"""Pydantic request/response models for the scoring API."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = Field(
        default=None, description="Generated if omitted."
    )
    account_id: str
    device_id: str
    geo_region: str
    amount: float = Field(gt=0)
    timestamp: Optional[datetime] = Field(
        default=None, description="Defaults to now if omitted."
    )
    # Stand-in for whatever upstream, proprietary risk features a real payment
    # processor already computes per transaction (in the reference dataset
    # these are the anonymized PCA components V1..V28). Optional so a
    # hand-typed demo transaction can still be scored; defaults to a zero
    # vector when omitted. The "replay test set" simulator in the dashboard
    # always supplies the real values from held-out data.
    pca_features: Optional[dict[str, float]] = None


class ShapContribution(BaseModel):
    feature: str
    value: float
    shap_contribution: float


class ComplianceAlert(BaseModel):
    transaction_id: str
    reason: str
    risk_score: int
    risk_tier: str
    rules_triggered: list[str]
    created_at: datetime


class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: int
    risk_tier: str
    action: str
    supervised_prob: float
    anomaly_score: float
    rules_triggered: list[str]
    features: dict[str, float]
    shap_explanation: Optional[list[ShapContribution]] = None
    compliance_alert: Optional[ComplianceAlert] = None
