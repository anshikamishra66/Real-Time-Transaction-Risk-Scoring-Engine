"""
Hard compliance rules layered on top of the ML ensemble score.

Why rules on top of a model at all: an ML score is a statistical estimate,
and a compliance program needs to be able to say "we always escalate X"
regardless of what any model outputs -- both for genuine risk reasons (a
large transaction from a brand-new device is exactly the account-takeover
pattern regulators expect to see hard-coded controls for) and for auditability
(a rule is a deterministic, explainable statement a regulator can verify
directly, where a model score requires trusting the model). Rules here can
only escalate risk, never suppress it below what the model found -- a rule
is a floor, not a ceiling, so the ML signal is never silently overridden
downward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

LARGE_AMOUNT_THRESHOLD = 3000.0
MODERATE_AMOUNT_THRESHOLD = 1000.0
VELOCITY_BURST_THRESHOLD = 5  # transactions in the last 1 minute


@dataclass
class Rule:
    name: str
    description: str
    condition: Callable[[dict, float], bool]  # (features, amount) -> bool
    floor_score: int  # minimum risk_score (0-100) this rule forces


RULES: list[Rule] = [
    Rule(
        name="large_amount_new_device",
        description="Amount above the large-transaction threshold combined with a device "
                     "never seen before on this account -- classic account-takeover pattern.",
        condition=lambda f, amount: amount >= LARGE_AMOUNT_THRESHOLD and f["is_new_device"] >= 1,
        floor_score=95,
    ),
    Rule(
        name="moderate_amount_new_device_new_geo",
        description="A moderately large amount from both a new device and a new "
                     "geography at once -- two independent anomaly signals compounding.",
        condition=lambda f, amount: (
            amount >= MODERATE_AMOUNT_THRESHOLD and f["is_new_device"] >= 1 and f["is_new_geo"] >= 1
        ),
        floor_score=90,
    ),
    Rule(
        name="velocity_burst",
        description="Five or more transactions on the same account within the last minute "
                     "-- consistent with automated card-testing or a compromised account.",
        condition=lambda f, amount: f["velocity_count_1m"] >= VELOCITY_BURST_THRESHOLD,
        floor_score=85,
    ),
    Rule(
        name="extreme_amount_zscore",
        description="Transaction amount is a statistical outlier (>6 std dev) versus this "
                     "account's own historical spending pattern.",
        condition=lambda f, amount: abs(f["amount_zscore"]) >= 6,
        floor_score=75,
    ),
]


def apply_rules(features: dict, amount: float, ml_risk_score: int) -> tuple[int, list[str]]:
    """Returns (final_risk_score, triggered_rule_names). Rules only raise the
    score -- the final score is max(ml_risk_score, best rule floor triggered)."""
    triggered = []
    final_score = ml_risk_score
    for rule in RULES:
        if rule.condition(features, amount):
            triggered.append(rule.name)
            final_score = max(final_score, rule.floor_score)
    return min(final_score, 100), triggered
