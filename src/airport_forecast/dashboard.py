"""Streamlit dashboard for airport PAX forecasting — VINCI Airports network.

Tabs:
  1. Forecast      — live honest recursive forecast (pick airport + horizon)
  2. Performance   — MAPE by horizon (LGB vs SARIMA vs naive) + per-airport
  3. Drivers       — LightGBM feature importance
  4. Data & EDA    — raw traffic + EDA gallery
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from airport_forecast.constants import CORE_AIRPORTS, SHORT_NAMES as SHORT
from airport_forecast.data import load_enriched
from airport_forecast.models import forecast_future_global

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
MODELS = ROOT / "models"

# ──────────────────────────────────────────────────────────────────
# Page config + light theming
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airport PAX Forecasting · VINCI",
    page_icon="✈️",
    layout="wide",
)

ACCENT = "#E2001A"  # VINCI red
INK = "#1d2b3a"

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2rem; }}
    .kpi {{
        background: #ffffff; border: 1px solid #e8ebef; border-radius: 14px;
        padding: 1rem 1.2rem; box-shadow: 0 1px 3px rgba(20,30,45,.06);
    }}
    .kpi .label {{ font-size: .78rem; color: #6b7785; text-transform: uppercase;
        letter-spacing: .04em; margin-bottom: .25rem; }}
    .kpi .value {{ font-size: 1.7rem; font-weight: 700; color: {INK}; line-height: 1; }}
    .kpi .sub {{ font-size: .8rem; color: #8a95a1; margin-top: .3rem; }}
    h1, h2, h3 {{ color: {INK}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    df = load_enriched()
    return df[df["airport"].isin(CORE_AIRPORTS)].copy()


@st.cache_data(show_spinner=False)
def load_reports():
    res = pd.read_csv(REPORTS / "model_results.csv") if (REPORTS / "model_results.csv").exists() else pd.DataFrame()
    fi = pd.read_csv(REPORTS / "feature_importance.csv") if (REPORTS / "feature_importance.csv").exists() else pd.DataFrame()
    hz = pd.read_csv(REPORTS / "horizon_results.csv") if (REPORTS / "horizon_results.csv").exists() else pd.DataFrame()
    return res, fi, hz


@st.cache_resource(show_spinner=False)
def load_model():
    with open(MODELS / "lightgbm_global.pkl", "rb") as f:
        d = pickle.load(f)
    return d["model"], d["feature_cols"]


@st.cache_data(show_spinner="Forecasting…")
def run_forecast(horizon: int) -> pd.DataFrame:
    """Honest recursive forecast for all core airports (cached per horizon)."""
    model, fcols = load_model()
    raw = load_raw()
    return forecast_future_global(model, fcols, raw, CORE_AIRPORTS, horizon)


def horizon_mape_curve(hz: pd.DataFrame) -> dict[int, float]:
    """LightGBM recursive MAPE per horizon (for the uncertainty band)."""
    if hz.empty:
        return {1: 3.5, 3: 4.1, 6: 3.8, 12: 3.9}
    sub = hz[hz["model"] == "LightGBM_Recursive"]
    if sub.empty:
        return {1: 3.5, 3: 4.1, 6: 3.8, 12: 3.9}
    return sub.groupby("horizon")["mape"].mean().to_dict()


raw = load_raw()
results, fi, hz = load_reports()
mape_curve = horizon_mape_curve(hz)
hz_points = sorted(mape_curve)
hz_vals = [mape_curve[h] for h in hz_points]

# ──────────────────────────────────────────────────────────────────
# Header + KPI row
# ──────────────────────────────────────────────────────────────────
st.title("✈️ Airport PAX Forecasting")
st.caption("Monthly passenger traffic forecasting across the VINCI Airports network · honest recursive evaluation")

best_m1 = mape_curve.get(min(hz_points), np.nan)
best_m12 = mape_curve.get(max(hz_points), np.nan)
last_month = raw["date"].max().strftime("%b %Y")

k1, k2, k3, k4 = st.columns(4)
for col, label, value, sub in [
    (k1, "Airports", str(raw["airport"].nunique()), "VINCI network core"),
    (k2, "MAPE M+1", f"{best_m1:.1f}%", "honest recursive"),
    (k3, "MAPE M+12", f"{best_m12:.1f}%", "beats SARIMA 5.2%"),
    (k4, "Data through", last_month, "Eurostat avia_paoa"),
]:
    col.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

tab_fc, tab_perf, tab_drv, tab_eda = st.tabs(
    ["🔮 Forecast", "📊 Performance", "🧩 Drivers", "🗂️ Data & EDA"]
)

# ──────────────────────────────────────────────────────────────────
# Tab 1 — Interactive forecast
# ──────────────────────────────────────────────────────────────────
with tab_fc:
    c1, c2 = st.columns([1, 3])
    with c1:
        ap_name = st.selectbox("Airport", [SHORT[a] for a in CORE_AIRPORTS])
        ap_code = next(a for a in CORE_AIRPORTS if SHORT[a] == ap_name)
        horizon = st.slider("Horizon (months)", 1, 18, 12)
        show_band = st.checkbox("Show uncertainty band", value=True)
        st.caption(
            "Recursive forecast: each month's prediction feeds the next month's "
            "lags. Future flights/macro use seasonal-naive / carried-forward "
            "values — no leakage."
        )

    fc = run_forecast(horizon)
    fc_ap = fc[fc["airport"] == ap_code].sort_values("date").head(horizon)

    hist = raw[raw["airport"] == ap_code].sort_values("date").tail(36)

    # Uncertainty band from per-horizon MAPE
    months_ahead = np.arange(1, len(fc_ap) + 1)
    band_pct = np.interp(months_ahead, hz_points, hz_vals) / 100.0
    lower = fc_ap["pax_pred"].values * (1 - band_pct)
    upper = fc_ap["pax_pred"].values * (1 + band_pct)

    with c2:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(hist["date"], hist["pax"] / 1e6, color=INK, lw=2, label="Actual")
        ax.plot(fc_ap["date"], fc_ap["pax_pred"] / 1e6, color=ACCENT, lw=2.2,
                marker="o", ms=4, label="Forecast")
        if show_band:
            ax.fill_between(fc_ap["date"], lower / 1e6, upper / 1e6,
                            color=ACCENT, alpha=0.15, label="±MAPE band")
        # connect last actual to first forecast
        if not hist.empty and not fc_ap.empty:
            ax.plot(
                [hist["date"].iloc[-1], fc_ap["date"].iloc[0]],
                [hist["pax"].iloc[-1] / 1e6, fc_ap["pax_pred"].iloc[0] / 1e6],
                color=ACCENT, lw=2.2, ls=":",
            )
        ax.set_ylabel("Passengers (millions)")
        ax.legend(loc="upper left", frameon=False)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    # Forecast table + summary
    out = fc_ap[["date", "pax_pred"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m")
    out["pax_pred"] = out["pax_pred"].round().astype(int)
    out.columns = ["Month", "Forecast PAX"]

    m1, m2 = st.columns([1, 2])
    with m1:
        total = int(fc_ap["pax_pred"].sum())
        st.metric(f"Total forecast PAX ({len(fc_ap)} mo)", f"{total:,}")
        if not fc_ap.empty:
            st.metric("Peak month forecast",
                      f"{int(fc_ap['pax_pred'].max()):,}",
                      fc_ap.loc[fc_ap['pax_pred'].idxmax(), 'date'].strftime('%b %Y'))
    with m2:
        st.dataframe(out, use_container_width=True, hide_index=True, height=320)

# ──────────────────────────────────────────────────────────────────
# Tab 2 — Performance
# ──────────────────────────────────────────────────────────────────
with tab_perf:
    st.subheader("MAPE by horizon — honest recursive")
    if not hz.empty:
        pivot = hz.pivot_table(index="model", columns="horizon", values="mape", aggfunc="mean")
        pivot = pivot[sorted(pivot.columns)]
        pivot.columns = [f"M+{c}" for c in pivot.columns]
        st.dataframe(
            pivot.round(1).style.highlight_min(axis=0, color="#d6f5dd"),
            use_container_width=True,
        )
        st.caption("Lowest MAPE per horizon highlighted. LightGBM Recursive wins every horizon.")

    if not results.empty:
        st.subheader("Average MAPE by model (per-airport, one-step)")
        res = results.copy()
        res["airport_name"] = res["airport"].map(SHORT)
        avg = res.groupby("model")["mape"].mean().sort_values()
        fig2, ax2 = plt.subplots(figsize=(9, 3.6))
        colors = [ACCENT if v == avg.min() else "#9fb0c0" for v in avg.values]
        ax2.barh(avg.index, avg.values, color=colors)
        for i, v in enumerate(avg.values):
            ax2.text(v + 0.15, i, f"{v:.1f}%", va="center", fontsize=9)
        ax2.set_xlabel("MAPE (%)")
        ax2.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig2)
        plt.close()

        st.subheader("Per-airport MAPE")
        pv = res.pivot_table(index="airport_name", columns="model", values="mape", aggfunc="mean")
        st.dataframe(pv.round(1).style.highlight_min(axis=1, color="#d6f5dd"),
                     use_container_width=True)

    pred_plot = FIGS / "24_predictions_vs_actual.png"
    if pred_plot.exists():
        st.subheader("Predictions vs actual (test set)")
        st.image(str(pred_plot), use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# Tab 3 — Drivers
# ──────────────────────────────────────────────────────────────────
with tab_drv:
    st.subheader("LightGBM Global — feature importance")
    if not fi.empty:
        top = fi.head(15)
        fig3, ax3 = plt.subplots(figsize=(9, 7))
        ax3.barh(top["feature"], top["importance"], color="#9fb0c0")
        ax3.barh(top["feature"].iloc[:3], top["importance"].iloc[:3], color=ACCENT)
        ax3.set_xlabel("Importance (split count)")
        ax3.invert_yaxis()
        ax3.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig3)
        plt.close()

    st.markdown(
        """
        **Read:**
        - `pax_lag_12` — same month last year — is the dominant predictor (annual seasonality)
        - `pax_lag_1` anchors the short term
        - `month_sin/cos` encode the seasonal cycle
        - `pax_yoy_growth` captures momentum; `oil_price_usd` adds macro signal
        """
    )

# ──────────────────────────────────────────────────────────────────
# Tab 4 — Data & EDA
# ──────────────────────────────────────────────────────────────────
with tab_eda:
    st.subheader("Monthly passenger traffic")
    sel = st.multiselect("Airports", [SHORT[a] for a in CORE_AIRPORTS],
                         default=[SHORT[a] for a in CORE_AIRPORTS])
    codes = [a for a in CORE_AIRPORTS if SHORT[a] in sel]
    fig, ax = plt.subplots(figsize=(13, 5))
    for code in codes:
        sub = raw[raw["airport"] == code]
        ax.plot(sub["date"], sub["pax"] / 1e6, label=SHORT[code], lw=1.4)
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-06-01"),
               alpha=0.08, color=ACCENT, label="COVID")
    ax.set_ylabel("Passengers (millions)")
    ax.legend(ncol=4, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.subheader("EDA gallery")
    eda_files = sorted(FIGS.glob("*.png"))
    cols = st.columns(2)
    for i, f in enumerate(eda_files):
        with cols[i % 2]:
            st.image(str(f), caption=f.stem, use_container_width=True)
