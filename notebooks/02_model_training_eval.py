# %% [markdown]
# # Model training & evaluation walkthrough
#
# Percent-format notebook -- open in Jupyter/VS Code, or run directly:
# `python notebooks/02_model_training_eval.py`.
#
# This notebook is the "show your work" companion to
# `python -m models.train_all` (which is what actually produces the artifacts
# the API loads). Here we additionally plot the precision-recall curve that
# justifies the classifier's decision threshold, and a SHAP summary plot
# across flagged transactions, to make the "why" behind the two headline
# design choices -- PR-AUC over accuracy, and SHAP only where needed --
# visible rather than asserted.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIGURES_DIR = REPO_ROOT / "notebooks" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

from features.feature_store import compute_batch_features  # noqa: E402
from models.config import FEATURE_COLUMNS, SUPERVISED_MODEL_PATH  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# %% [markdown]
# ## 1. Load data + features
#
# Assumes `python -m models.train_all` has already been run at least once
# (so `models/artifacts/supervised_xgb.json` exists). If not, run that first.

# %%
train_df = compute_batch_features(pd.read_parquet(PROCESSED_DIR / "train.parquet"))
test_df = compute_batch_features(pd.read_parquet(PROCESSED_DIR / "test.parquet"))

import xgboost as xgb  # noqa: E402

model = xgb.XGBClassifier()
model.load_model(str(SUPERVISED_MODEL_PATH))

X_test, y_test = test_df[FEATURE_COLUMNS], test_df["Class"].to_numpy()
y_prob = model.predict_proba(X_test)[:, 1]

# %% [markdown]
# ## 2. Precision-recall curve
#
# We tune the decision threshold on this curve (maximizing F1) rather than
# using accuracy or the default 0.5 cutoff, and rather than ROC-AUC, because
# with <1% positives ROC-AUC stays deceptively high even for a model that's
# bad at the precision/recall tradeoff that actually matters operationally.

# %%
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall, precision, color="#3498db")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curve -- supervised classifier (test set)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "pr_curve.png", dpi=120)

# %% [markdown]
# ## 3. SHAP summary across flagged transactions
#
# In production (see `models/explain.py`), SHAP only runs for transactions
# that already cleared the medium-risk tier, to keep the hot path fast. Here,
# for analysis purposes only (not on the serving path), we compute it for a
# batch of the highest-scored test transactions to see which features drive
# the model globally -- this is exactly the kind of check a compliance team
# would want before trusting the model's stated reasons in production.

# %%
import shap  # noqa: E402

top_idx = pd.Series(y_prob).nlargest(200).index
X_top = X_test.iloc[top_idx]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_top)

fig = plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_top, show=False, max_display=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")

print("Saved figures to", FIGURES_DIR)
