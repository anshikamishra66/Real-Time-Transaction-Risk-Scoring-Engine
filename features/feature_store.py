"""
Rolling behavioral feature store, shared by both training and the live API.

The same `FeatureStore` class is used to build training features (by replaying
historical transactions in chronological order) and to score live transactions
one at a time in the FastAPI service. This is a deliberate design choice: if
training computed features one way (e.g. vectorized pandas rolling windows)
and serving computed them another way (e.g. incremental state), any subtle
mismatch between the two would silently degrade the model in production --
the classic "train/serve skew" bug. Using one implementation for both closes
that gap by construction.

Per account/card, the store maintains just enough state to answer the
required behavioral questions in O(1)-ish amortized time per transaction:
  - a time-ordered deque of (timestamp, amount) capped at a 1 hour window,
    used for transaction-velocity counts (1 min / 5 min / 1 hr) and for the
    rolling mean/std used in the amount z-score
  - the timestamp of the account's previous transaction
  - the set of devices and geo regions previously seen on the account

All features are computed from state *before* the current transaction is
folded in, so there is no leakage from a transaction into its own features.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Optional

import numpy as np
import pandas as pd

ONE_MIN = timedelta(minutes=1)
FIVE_MIN = timedelta(minutes=5)
ONE_HOUR = timedelta(hours=1)

# Sentinel for "no prior transaction on this account" -- large enough that it
# never gets confused with a real gap, but finite so it can feed a numeric model.
NO_HISTORY_SECONDS = 24 * 3600.0


@dataclass
class AccountState:
    history: Deque[tuple[datetime, float]] = field(default_factory=deque)  # capped at 1hr
    last_timestamp: Optional[datetime] = None
    devices_seen: set[str] = field(default_factory=set)
    geos_seen: set[str] = field(default_factory=set)

    def prune(self, now: datetime) -> None:
        cutoff = now - ONE_HOUR
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()


class FeatureStore:
    """Stateful, per-account rolling feature computation.

    Call `compute_and_update(...)` once per transaction, in chronological
    order per account (globally chronological order is sufficient and is what
    both the batch replay and the live API naturally provide).
    """

    FEATURE_NAMES = [
        "velocity_count_1m",
        "velocity_count_5m",
        "velocity_count_1h",
        "amount_zscore",
        "time_since_last_tx_sec",
        "is_new_device",
        "is_new_geo",
    ]

    def __init__(self) -> None:
        self._accounts: dict[str, AccountState] = {}

    def reset(self) -> None:
        self._accounts.clear()

    def _get(self, account_id: str) -> AccountState:
        state = self._accounts.get(account_id)
        if state is None:
            state = AccountState()
            self._accounts[account_id] = state
        return state

    def compute_and_update(
        self,
        account_id: str,
        device_id: str,
        geo_region: str,
        timestamp: datetime,
        amount: float,
    ) -> dict[str, float]:
        state = self._get(account_id)
        state.prune(timestamp)

        times = [t for t, _ in state.history]
        amounts = np.array([a for _, a in state.history], dtype=float)

        velocity_1m = sum(1 for t in times if t >= timestamp - ONE_MIN)
        velocity_5m = sum(1 for t in times if t >= timestamp - FIVE_MIN)
        velocity_1h = len(times)

        if amounts.size >= 2:
            mean, std = float(amounts.mean()), float(amounts.std())
            amount_zscore = (amount - mean) / std if std > 1e-6 else 0.0
        elif amounts.size == 1:
            amount_zscore = 0.0
        else:
            amount_zscore = 0.0

        if state.last_timestamp is None:
            time_since_last = NO_HISTORY_SECONDS
        else:
            time_since_last = max(0.0, (timestamp - state.last_timestamp).total_seconds())

        is_new_device = float(device_id not in state.devices_seen) if state.devices_seen else 0.0
        is_new_geo = float(geo_region not in state.geos_seen) if state.geos_seen else 0.0

        features = {
            "velocity_count_1m": float(velocity_1m),
            "velocity_count_5m": float(velocity_5m),
            "velocity_count_1h": float(velocity_1h),
            "amount_zscore": float(np.clip(amount_zscore, -10, 10)),
            "time_since_last_tx_sec": float(min(time_since_last, NO_HISTORY_SECONDS)),
            "is_new_device": is_new_device,
            "is_new_geo": is_new_geo,
        }

        # Update state *after* computing features so the current transaction
        # never contributes to its own feature values.
        state.history.append((timestamp, amount))
        state.last_timestamp = timestamp
        state.devices_seen.add(device_id)
        state.geos_seen.add(geo_region)

        return features


def compute_batch_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replays a historical transaction log through a fresh FeatureStore, in
    chronological order, and returns the input df with feature columns added.

    Expects columns: account_id, device_id, geo_region, timestamp, Amount.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    store = FeatureStore()

    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            store.compute_and_update(
                account_id=row.account_id,
                device_id=row.device_id,
                geo_region=row.geo_region,
                timestamp=row.timestamp,
                amount=float(row.Amount),
            )
        )

    feature_df = pd.DataFrame(rows, columns=FeatureStore.FEATURE_NAMES)
    return pd.concat([df.reset_index(drop=True), feature_df], axis=1)
