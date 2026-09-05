# %% [markdown]
# # EDA: Credit Card Fraud Detection
#
# Written in Jupytext "percent" format -- open it directly in Jupyter/VS Code
# as a notebook (each `# %%` starts a new cell), or just run it as a plain
# script: `python notebooks/01_eda.py`. Either way it reads the already-
# augmented dataset from `data/preprocess.py` and writes figures to
# `notebooks/figures/`.
#
# Goal: understand (1) how extreme the class imbalance is, (2) whether the
# synthetic account/device/geo metadata we attached actually carries signal,
# and (3) what the amount distributions look like per class -- all of which
# directly motivated the modeling choices in `models/train_supervised.py` and
# `models/train_anomaly.py`.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURES_DIR = REPO_ROOT / "notebooks" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

df = pd.read_parquet(PROCESSED_DIR / "transactions.parquet")
df.head()

# %% [markdown]
# ## 1. Class imbalance

# %%
fraud_rate = df["Class"].mean()
print(f"Total transactions: {len(df):,}")
print(f"Fraud transactions: {df['Class'].sum():,}")
print(f"Fraud rate: {fraud_rate:.4%}")

fig, ax = plt.subplots(figsize=(4, 4))
df["Class"].value_counts().rename({0: "legit", 1: "fraud"}).plot.bar(ax=ax, color=["#2ecc71", "#e74c3c"])
ax.set_title(f"Class balance (fraud = {fraud_rate:.3%})")
ax.set_yscale("log")
ax.set_ylabel("count (log scale)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "class_balance.png", dpi=120)

# %% [markdown]
# At well under 1% positive, accuracy is a useless metric here -- a
# classifier that predicts "legit" for everything scores ~99.8% accuracy
# while catching zero fraud. This is why training/evaluation throughout this
# project uses precision-recall curves and PR-AUC instead of accuracy or
# ROC-AUC (see `models/train_supervised.py` docstring for the full argument).

# %% [markdown]
# ## 2. Transaction amount by class

# %%
fig, ax = plt.subplots(figsize=(6, 4))
df[df.Class == 0]["Amount"].clip(upper=2000).hist(ax=ax, bins=60, alpha=0.6, label="legit", color="#2ecc71")
df[df.Class == 1]["Amount"].clip(upper=2000).hist(ax=ax, bins=60, alpha=0.6, label="fraud", color="#e74c3c")
ax.set_xlabel("Amount (clipped at 2000)")
ax.set_ylabel("count")
ax.set_title("Transaction amount distribution by class")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "amount_by_class.png", dpi=120)

print(df.groupby("Class")["Amount"].describe())

# %% [markdown]
# ## 3. Does the synthetic device/geo metadata carry signal?
#
# This dataset's account/device/geo fields are synthetic (see
# `data/preprocess.py` docstring for why), deliberately correlated with the
# real fraud label so the behavioral feature store has something real to
# learn from. Sanity-check that correlation actually shows up before trusting
# any downstream model trained on it.

# %%
from features.feature_store import compute_batch_features  # noqa: E402

sample = df.sample(n=min(60_000, len(df)), random_state=0)
featurized = compute_batch_features(sample)

behavioral_summary = featurized.groupby("Class")[
    ["velocity_count_1h", "amount_zscore", "is_new_device", "is_new_geo", "time_since_last_tx_sec"]
].mean()
print(behavioral_summary)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
featurized.groupby("Class")["is_new_device"].mean().rename({0: "legit", 1: "fraud"}).plot.bar(
    ax=axes[0], color=["#2ecc71", "#e74c3c"]
)
axes[0].set_title("P(new device) by class")
featurized.groupby("Class")["is_new_geo"].mean().rename({0: "legit", 1: "fraud"}).plot.bar(
    ax=axes[1], color=["#2ecc71", "#e74c3c"]
)
axes[1].set_title("P(new geo) by class")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "device_geo_by_class.png", dpi=120)

# %% [markdown]
# Fraudulent transactions should show a markedly higher rate of new-device
# and new-geo flags than legitimate ones -- that gap is exactly the signal
# `api/rules_engine.py`'s `large_amount_new_device` and
# `moderate_amount_new_device_new_geo` rules are built to catch, and what the
# supervised model learns to weight via the `is_new_device` / `is_new_geo`
# feature columns.

# %% [markdown]
# ## 4. Which PCA components separate fraud best?

# %%
pca_cols = [f"V{i}" for i in range(1, 29)]
corr_with_class = df[pca_cols + ["Class"]].corr()["Class"].drop("Class").sort_values(key=abs, ascending=False)
print(corr_with_class.head(10))

fig, ax = plt.subplots(figsize=(6, 5))
corr_with_class.head(10).plot.barh(ax=ax, color="#3498db")
ax.set_title("Top 10 |correlation with Class| among V1..V28")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "pca_correlation.png", dpi=120)

print("\nSaved figures to", FIGURES_DIR)
