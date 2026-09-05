"""Maps a final risk tier to an automated compliance action."""
from __future__ import annotations

from datetime import datetime, timezone

from api.schemas import ComplianceAlert

TIER_TO_ACTION = {
    "low": "auto_approve",
    "medium": "log_for_manual_review",
    "high": "auto_hold",
}


def decide_action(risk_tier: str) -> str:
    return TIER_TO_ACTION[risk_tier]


def build_compliance_alert(
    transaction_id: str, risk_score: int, risk_tier: str, rules_triggered: list[str]
) -> ComplianceAlert | None:
    if risk_tier != "high":
        return None
    if rules_triggered:
        reason = "Compliance rule(s) triggered: " + ", ".join(rules_triggered)
    else:
        reason = "ML ensemble risk score exceeded the high-risk threshold."
    return ComplianceAlert(
        transaction_id=transaction_id,
        reason=reason,
        risk_score=risk_score,
        risk_tier=risk_tier,
        rules_triggered=rules_triggered,
        created_at=datetime.now(timezone.utc),
    )
