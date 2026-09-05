"""
Turns data/raw/creditcard.csv into data/processed/transactions.parquet plus a
chronological train/test split.

Why this exists: the Kaggle Credit Card Fraud dataset is PCA-anonymized
(V1..V28) and has no card/account id, device id, or geolocation -- exactly the
fields a real-time behavioral feature store needs (velocity, new-device,
new-geo). Rather than fabricate a whole new dataset, we deterministically
attach synthetic account/device/geo metadata on top of the *real* Time /
Amount / V1..V28 / Class columns, seeded so results are reproducible.

The synthetic metadata is generated so it correlates with the real fraud
label the way real-world account-takeover fraud does -- fraudulent
transactions are more likely to come from a new device and an unusual
geography -- which gives the downstream velocity/device/geo features genuine
predictive signal instead of being pure noise. This is a documented design
choice (see README "Dataset & synthetic metadata"), not an attempt to
misrepresent synthetic data as ground truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAW_CSV = Path(__file__).resolve().parent / "raw" / "creditcard.csv"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

GEO_REGIONS = [
    "US-NY", "US-CA", "US-TX", "US-FL", "US-IL",
    "GB-LON", "DE-BER", "FR-PAR", "CA-ON", "AU-NSW",
    "BR-SP", "IN-MH", "NG-LA", "RU-MOW", "SG-SG",
]


def _assign_accounts(n_rows: int, avg_tx_per_account: int, rng: np.random.Generator) -> np.ndarray:
    """Zipf-weighted account assignment so a minority of accounts are heavy users,
    which is realistic and gives the velocity features real variance to learn from."""
    n_accounts = max(1, n_rows // avg_tx_per_account)
    weights = rng.zipf(a=1.3, size=n_accounts).astype(float)
    weights = np.clip(weights, 1, 200)
    weights /= weights.sum()
    return rng.choice(n_accounts, size=n_rows, p=weights)


def _account_profiles(n_accounts: int, rng: np.random.Generator) -> pd.DataFrame:
    home_geo = rng.choice(GEO_REGIONS, size=n_accounts)
    primary_device = np.array([f"dev-{i:06d}-a" for i in range(n_accounts)])
    secondary_device = np.array([f"dev-{i:06d}-b" for i in range(n_accounts)])
    return pd.DataFrame({
        "account_id": [f"acct-{i:06d}" for i in range(n_accounts)],
        "home_geo": home_geo,
        "primary_device": primary_device,
        "secondary_device": secondary_device,
    })


def augment(df: pd.DataFrame, seed: int = 7, avg_tx_per_account: int = 35) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.sort_values("Time").reset_index(drop=True)
    n_rows = len(df)

    account_idx = _assign_accounts(n_rows, avg_tx_per_account, rng)
    n_accounts = account_idx.max() + 1
    profiles = _account_profiles(n_accounts, rng)

    df["account_id"] = profiles["account_id"].to_numpy()[account_idx]
    home_geo = profiles["home_geo"].to_numpy()[account_idx]
    primary_device = profiles["primary_device"].to_numpy()[account_idx]
    secondary_device = profiles["secondary_device"].to_numpy()[account_idx]

    is_fraud = df["Class"].to_numpy() == 1
    # Fraud rows use an unrecognized device/geo far more often than legit rows,
    # mirroring account-takeover / card-not-present fraud patterns.
    new_device_prob = np.where(is_fraud, 0.65, 0.06)
    new_geo_prob = np.where(is_fraud, 0.55, 0.03)

    uses_new_device = rng.random(n_rows) < new_device_prob
    uses_secondary = rng.random(n_rows) < 0.5
    other_region = rng.choice(GEO_REGIONS, size=n_rows)

    device_id = np.where(
        uses_new_device,
        np.where(uses_secondary, secondary_device, [f"dev-anon-{i}" for i in range(n_rows)]),
        primary_device,
    )
    geo_region = np.where(rng.random(n_rows) < new_geo_prob, other_region, home_geo)

    df["device_id"] = device_id
    df["geo_region"] = geo_region

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df["timestamp"] = [base_time + timedelta(seconds=float(s)) for s in df["Time"]]

    df["transaction_id"] = [f"tx-{i:08d}" for i in range(n_rows)]

    ordered_cols = ["transaction_id", "account_id", "device_id", "geo_region", "timestamp",
                    "Time", "Amount"] + [f"V{i}" for i in range(1, 29)] + ["Class"]
    return df[ordered_cols]


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered split (not random) -- fraud detection must be validated on
    the future relative to training data, never shuffled, or the evaluation
    leaks information the production model would never have."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_at = int(len(df) * (1 - test_frac))
    return df.iloc[:split_at].copy(), df.iloc[split_at:].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=RAW_CSV)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not args.raw_csv.exists():
        raise SystemExit(
            f"{args.raw_csv} not found. Run data/download_data.py (needs Kaggle "
            f"credentials) or scripts/generate_synthetic_data.py first."
        )

    raw = pd.read_csv(args.raw_csv)
    df = augment(raw, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_dir / "transactions.parquet", index=False)

    train_df, test_df = chronological_split(df, test_frac=args.test_frac)
    train_df.to_parquet(args.out_dir / "train.parquet", index=False)
    test_df.to_parquet(args.out_dir / "test.parquet", index=False)

    print(f"Processed {len(df):,} transactions across {df['account_id'].nunique():,} accounts")
    print(f"  train: {len(train_df):,} rows ({train_df['Class'].sum()} fraud)")
    print(f"  test:  {len(test_df):,} rows ({test_df['Class'].sum()} fraud)")


if __name__ == "__main__":
    main()
