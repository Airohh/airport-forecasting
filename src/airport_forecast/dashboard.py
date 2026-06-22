"""Streamlit dashboard for airport PAX forecasting results."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
DATA = ROOT / "data" / "processed"

from airport_forecast.constants import SHORT_NAMES as SHORT

st.set_page_config(page_title="Airport PAX Forecasting", layout="wide")
st.title("Airport PAX Forecasting - VINCI Airports Network")

# Load data
@st.cache_data
def load_data():
    pax = pd.read_parquet(DATA / "pax_enriched.parquet")
    pax["date"] = pd.to_datetime(pax["date"])
    results = pd.read_csv(REPORTS / "model_results.csv")
    fi = pd.read_csv(REPORTS / "feature_importance.csv")
    return pax, results, fi

pax, results, fi = load_data()
results["airport_name"] = results["airport"].map(SHORT)
pax_core = pax[pax["airport"].isin(SHORT.keys())]

# Sidebar
st.sidebar.header("Filters")
selected_airports = st.sidebar.multiselect(
    "Airports", list(SHORT.values()), default=list(SHORT.values())
)
selected_codes = [k for k, v in SHORT.items() if v in selected_airports]

# ─── Tab 1: Time Series ───
tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Model Comparison", "Feature Importance", "EDA Gallery"])

with tab1:
    st.subheader("Monthly Passenger Traffic")
    fig, ax = plt.subplots(figsize=(14, 6))
    for code in selected_codes:
        sub = pax_core[pax_core["airport"] == code]
        ax.plot(sub["date"], sub["pax"] / 1e6, label=SHORT[code])
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-06-01"),
               alpha=0.1, color="red", label="COVID")
    ax.axvline(pd.Timestamp("2024-01-01"), color="orange", linestyle="--", label="Val start")
    ax.axvline(pd.Timestamp("2025-01-01"), color="red", linestyle="--", label="Test start")
    ax.set_ylabel("Passengers (millions)")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    # Summary stats
    st.subheader("Summary Statistics")
    summary = pax_core[pax_core["airport"].isin(selected_codes)].groupby("airport").agg(
        months=("pax", "count"),
        mean_pax=("pax", "mean"),
        max_pax=("pax", "max"),
    ).reset_index()
    summary["airport"] = summary["airport"].map(SHORT)
    summary["mean_pax"] = summary["mean_pax"].apply(lambda x: f"{x:,.0f}")
    summary["max_pax"] = summary["max_pax"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(summary, use_container_width=True)

with tab2:
    st.subheader("Model Comparison (MAPE %)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Validation (2024)**")
        val_res = results[results["airport_name"].isin(selected_airports)]
        # Approximate: val has horizon=12 for 2024
        pivot_val = val_res.pivot_table(index="airport_name", columns="model", values="mape")
        if not pivot_val.empty:
            st.dataframe(pivot_val.round(1).style.highlight_min(axis=1, color="lightgreen"),
                        use_container_width=True)

    with col2:
        st.markdown("**Average MAPE by Model**")
        avg = results.groupby("model")["mape"].mean().sort_values()
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colors = ["green" if v == avg.min() else "steelblue" for v in avg.values]
        ax2.barh(avg.index, avg.values, color=colors)
        ax2.set_xlabel("MAPE (%)")
        for i, v in enumerate(avg.values):
            ax2.text(v + 0.2, i, f"{v:.1f}%", va="center")
        st.pyplot(fig2)
        plt.close()

    # Predictions vs actual plots
    st.subheader("Predictions vs Actual")
    pred_plot = FIGS / "24_predictions_vs_actual.png"
    if pred_plot.exists():
        st.image(str(pred_plot), use_container_width=True)

with tab3:
    st.subheader("LightGBM Global - Feature Importance")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    fi_top = fi.head(15)
    ax3.barh(fi_top["feature"], fi_top["importance"], color="steelblue")
    ax3.set_xlabel("Importance (split count)")
    ax3.invert_yaxis()
    st.pyplot(fig3)
    plt.close()

    st.markdown("""
    **Key findings:**
    - `pax_lag_12` dominates: traffic 12 months ago is the strongest predictor
    - `pax_yoy_growth` captures momentum
    - `oil_price_usd` has real predictive power from macro enrichment
    - `month_sin/cos` captures seasonality
    """)

with tab4:
    st.subheader("EDA Gallery")
    eda_files = sorted(FIGS.glob("*.png"))
    cols = st.columns(2)
    for i, f in enumerate(eda_files):
        with cols[i % 2]:
            st.image(str(f), caption=f.stem, use_container_width=True)
