"""
Generates a schema-compatible stand-in for the Kaggle "Credit Card Fraud
Detection" dataset (mlg-ulb/creditcardfraud) so the full pipeline can be
built, run, and demoed without Kaggle API credentials.

Output columns match the real dataset exactly: Time, V1..V28, Amount, Class.
V1..V28 are synthesized as correlated Gaussian features with a handful of
dimensions shifted for fraud rows, so a real classifier trained on them shows
a genuine (not perfect) separation -- same qualitative shape as the real PCA
components, which are known to carry most of the fraud signal in V14, V17,
V12, V10, etc.

This script is a deliberate substitute, not a silent fallback: it never runs
automatically. Swap it for scripts/../data/download_data.py + real
creditcard.csv at any time -- every downstream script only depends on the
column schema, not on which source produced it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def generate(n_rows: int, fraud_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_fraud = max(1, int(round(n_rows * fraud_rate)))
    # Fraud positions are scattered uniformly at random across the whole
    # timeline -- NOT concatenated as a legit block followed by a fraud block,
    # which would silently cluster all fraud at one end of the chronological
    # order once Time is sorted, and corrupt any time-based train/test split.
    labels = np.zeros(n_rows, dtype=int)
    fraud_positions = rng.choice(n_rows, size=n_fraud, replace=False)
    labels[fraud_positions] = 1

    # Time: seconds elapsed over a 2-day window, matching the real dataset's span.
    time_seconds = np.sort(rng.uniform(0, 2 * 24 * 3600, size=n_rows))

    n_components = 28
    base = rng.normal(loc=0.0, scale=1.0, size=(n_rows, n_components))

    # Correlate components a little (PCA components aren't independent in practice).
    mix = rng.normal(0, 0.15, size=(n_components, n_components))
    np.fill_diagonal(mix, 1.0)
    base = base @ mix

    # Shift a handful of dimensions for fraud rows to create learnable, imperfect
    # separation -- mirrors the well-known real-world signal concentrated in
    # V14, V17, V12, V10, V16.
    fraud_idx = np.where(labels == 1)[0]
    signal_dims = [9, 11, 13, 15, 16]  # zero-based -> V10, V12, V14, V16, V17
    for dim in signal_dims:
        shift = rng.choice([-1, 1]) * rng.uniform(2.5, 4.5)
        base[fraud_idx, dim] += shift

    amount = np.empty(n_rows)
    legit_idx = np.where(labels == 0)[0]
    amount[legit_idx] = rng.lognormal(mean=3.0, sigma=1.1, size=len(legit_idx))

    # Fraud amounts: bimodal -- small "card testing" probes and occasional large hits.
    fraud_small = rng.lognormal(mean=1.0, sigma=0.6, size=len(fraud_idx))
    fraud_large = rng.lognormal(mean=5.5, sigma=0.8, size=len(fraud_idx))
    is_large = rng.random(len(fraud_idx)) < 0.35
    amount[fraud_idx] = np.where(is_large, fraud_large, fraud_small)

    amount = np.round(np.clip(amount, 0.01, 25000.0), 2)

    df = pd.DataFrame(base, columns=[f"V{i}" for i in range(1, n_components + 1)])
    df.insert(0, "Time", time_seconds)
    df["Amount"] = amount
    df["Class"] = labels
    # Rows are already in chronological (Time-ascending) order by construction,
    # with fraud scattered throughout -- no reshuffle/resort needed.
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=250_000)
    parser.add_argument("--fraud-rate", type=float, default=0.00173, help="~real dataset rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=RAW_DIR / "creditcard.csv")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.n_rows, args.fraud_rate, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows ({df['Class'].sum():,} fraud, "
          f"{df['Class'].mean() * 100:.3f}%) to {args.out}")


if __name__ == "__main__":
    main()
