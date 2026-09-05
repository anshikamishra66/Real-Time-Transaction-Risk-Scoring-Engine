"""
Streamlit dashboard for the Real-Time Transaction Risk Scoring Engine.

Talks to the FastAPI service over HTTP only (never touches the DB or models
directly) -- the dashboard is a thin client, exactly like a real ops console
would be, so it can be pointed at any deployment of the API.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = st.sidebar.text_input("API base URL", value="http://localhost:8000")
TEST_PARQUET = Path(__file__).resolve().parent.parent / "data" / "processed" / "test.parquet"

st.set_page_config(page_title="Risk Scoring Engine", layout="wide")
st.title("Real-Time Transaction Risk Scoring Engine")

TIER_COLORS = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}


def tier_badge(tier: str) -> str:
    color = TIER_COLORS.get(tier, "#888")
    return f"<span style='background-color:{color};color:white;padding:2px 10px;border-radius:10px;font-weight:600'>{tier.upper()}</span>"


@st.cache_data(ttl=30)
def load_replay_pool() -> pd.DataFrame:
    if not TEST_PARQUET.exists():
        return pd.DataFrame()
    df = pd.read_parquet(TEST_PARQUET)
    return df.sort_values("timestamp").reset_index(drop=True)


def api_get(path: str, **params):
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None


def api_post(path: str, json_body: dict):
    try:
        resp = requests.post(f"{API_BASE}{path}", json=json_body, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None


def replay_batch(n: int) -> list[dict]:
    pool = load_replay_pool()
    if pool.empty:
        st.warning("No replay data found -- run data/preprocess.py first.")
        return []

    idx = st.session_state.get("replay_idx", 0)
    results = []
    for _ in range(n):
        row = pool.iloc[idx % len(pool)]
        pca_features = {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)}
        payload = {
            "account_id": row["account_id"],
            "device_id": row["device_id"],
            "geo_region": row["geo_region"],
            "amount": float(row["Amount"]),
            "pca_features": pca_features,
        }
        result = api_post("/score-transaction", payload)
        if result:
            result["_ground_truth_fraud"] = bool(row["Class"])
            results.append(result)
        idx += 1
    st.session_state["replay_idx"] = idx
    return results


tab_feed, tab_queue, tab_metrics = st.tabs(["Live Feed", "Review Queue", "Model Performance"])

with tab_feed:
    col_a, col_b, col_c = st.columns([1, 1, 3])
    with col_a:
        batch_size = st.number_input("Batch size", min_value=1, max_value=50, value=5)
    with col_b:
        if st.button("Replay next batch"):
            replay_batch(int(batch_size))
    with col_c:
        auto_replay = st.checkbox("Auto-replay (streams continuously)")

    if st.button("Reset demo (clears log + account history)"):
        requests.post(f"{API_BASE}/admin/reset-demo", timeout=5)
        st.session_state["replay_idx"] = 0
        st.success("Demo reset.")

    recent = api_get("/transactions/recent", limit=100)
    if recent:
        df = pd.DataFrame(recent)
        df["tier_badge"] = df["risk_tier"].apply(tier_badge)
        display_cols = ["created_at", "transaction_id", "account_id", "amount",
                         "risk_score", "risk_tier", "action", "rules_triggered"]
        st.write(
            df[display_cols].to_html(escape=False, index=False,
                formatters={"risk_tier": lambda t: tier_badge(t)}),
            unsafe_allow_html=True,
        )
    else:
        st.info("No transactions scored yet -- click 'Replay next batch' to start the demo.")

    if auto_replay:
        replay_batch(int(batch_size))
        time.sleep(1.5)
        st.rerun()

with tab_queue:
    st.subheader("Medium / High risk transactions awaiting review")
    status_filter = st.selectbox("Status", ["pending", "resolved", "all"])
    queue = api_get("/review-queue", status=(None if status_filter == "all" else status_filter))

    if queue:
        for item in queue:
            with st.expander(
                f"[{item['risk_tier'].upper()}]  {item['transaction_id']}  "
                f"— ${item['amount']:.2f}  (score {item['risk_score']})",
                expanded=False,
            ):
                st.markdown(tier_badge(item["risk_tier"]), unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Account:**", item["account_id"])
                    st.write("**Device:**", item["device_id"])
                    st.write("**Geo:**", item["geo_region"])
                    st.write("**Action taken:**", item["action"])
                    st.write("**Rules triggered:**", ", ".join(item["rules_triggered"]) or "none")
                with c2:
                    st.write("**Behavioral features:**")
                    st.json(item["features"])

                if item.get("shap_explanation"):
                    shap_df = pd.DataFrame(item["shap_explanation"])
                    fig = px.bar(
                        shap_df.sort_values("shap_contribution"),
                        x="shap_contribution", y="feature", orientation="h",
                        color="shap_contribution", color_continuous_scale=["#2ecc71", "#e74c3c"],
                        title="Why this transaction was flagged (SHAP)",
                    )
                    st.plotly_chart(fig, width='stretch', key=f"shap-{item['transaction_id']}")

                if item["review_status"] == "pending":
                    if st.button("Mark resolved", key=f"resolve-{item['transaction_id']}"):
                        requests.post(
                            f"{API_BASE}/review-queue/{item['transaction_id']}/resolve",
                            params={"status": "resolved"}, timeout=5,
                        )
                        st.rerun()
    else:
        st.info("Review queue is empty.")

with tab_metrics:
    st.subheader("Model performance (held-out test set)")
    metrics = api_get("/metrics")
    if metrics:
        sup = metrics["supervised"]
        ens = metrics["ensemble"]["flagged_medium_or_high"]

        st.markdown("#### Supervised classifier (at its tuned decision threshold)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{sup['precision_at_best_f1']:.1%}")
        c2.metric("Recall", f"{sup['recall_at_best_f1']:.1%}")
        c3.metric("F1", f"{sup['f1_at_best_f1']:.2f}")
        c4.metric("PR-AUC", f"{sup['pr_auc']:.3f}")
        st.caption(f"False positive rate: {sup['false_positive_rate']:.3%} · "
                   f"threshold: {sup['threshold']:.3f} · "
                   f"trained on {sup['n_train']:,} tx ({sup['n_fraud_train']} fraud)")

        st.markdown("#### Full ensemble + rules (medium/high tier = flagged)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{ens['precision']:.1%}")
        c2.metric("Recall", f"{ens['recall']:.1%}")
        c3.metric("F1", f"{ens['f1']:.2f}")
        c4.metric("False positive rate", f"{ens['false_positive_rate']:.3%}")

        cm = sup["confusion_matrix"]
        st.markdown("#### Supervised confusion matrix")
        cm_df = pd.DataFrame(
            [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            index=["Actual legit", "Actual fraud"],
            columns=["Predicted legit", "Predicted fraud"],
        )
        st.dataframe(cm_df)

        st.markdown("#### Risk tier distribution (test set)")
        tier_counts = metrics["ensemble"]["risk_tier_counts"]
        fig = px.pie(names=list(tier_counts.keys()), values=list(tier_counts.values()),
                     color=list(tier_counts.keys()),
                     color_discrete_map=TIER_COLORS)
        st.plotly_chart(fig, width='stretch', key="risk-tier-distribution")

        anomaly = metrics["anomaly"]
        st.caption(
            f"Anomaly detector: mean score on fraud = {anomaly['mean_anomaly_score_fraud']:.3f}, "
            f"on legit = {anomaly['mean_anomaly_score_legit']:.3f} "
            f"(higher = more anomalous, [0,1] scale)"
        )
    else:
        st.info("No metrics found -- run `python -m models.train_all` first.")
