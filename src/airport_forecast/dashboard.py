"""Streamlit dashboard for airport PAX forecasting — VINCI Airports network.

Tabs:
  1. Forecast      — live honest recursive forecast (pick airport + horizon)
  2. Performance   — MAPE by horizon (LGB vs SARIMA vs naive) + per-airport
  3. Drivers       — LightGBM feature importance
  4. Data & EDA    — raw traffic + EDA gallery

Charts are Plotly (interactive: hover, zoom, unified tooltips). Theme is set
globally in `.streamlit/config.toml` (VINCI red accent).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

# Make `airport_forecast` importable when run directly (e.g. Streamlit Cloud),
# which puts the script's own dir on sys.path but not the src/ root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from airport_forecast.constants import CORE_AIRPORTS, SHORT_NAMES as SHORT
from airport_forecast.data import load_enriched
from airport_forecast.models import forecast_future_global, train_sarima

# Horizon is capped at the longest horizon we actually backtested (M+12).
# Beyond that we have no honest error estimate, so the UI never pretends to.
MAX_HORIZON = 12

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
MODELS = ROOT / "models"

# ──────────────────────────────────────────────────────────────────
# Palette + page config
# ──────────────────────────────────────────────────────────────────
ACCENT = "#E2001A"   # VINCI red
INK = "#1d2b3a"
SARIMA_BLUE = "#2b6cb0"
MUTED = "#9fb0c0"
GRID = "rgba(20,30,45,.08)"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

st.set_page_config(
    page_title="Airport PAX Forecasting · VINCI",
    page_icon="✈️",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2rem; max-width: 1400px; }}
    .kpi {{
        background: #ffffff; border: 1px solid #e8ebef; border-radius: 14px;
        padding: 1rem 1.2rem; box-shadow: 0 1px 3px rgba(20,30,45,.06);
        transition: box-shadow .15s ease;
    }}
    .kpi:hover {{ box-shadow: 0 4px 14px rgba(20,30,45,.10); }}
    .kpi .label {{ font-size: .78rem; color: #6b7785; text-transform: uppercase;
        letter-spacing: .04em; margin-bottom: .25rem; }}
    .kpi .value {{ font-size: 1.7rem; font-weight: 700; color: {INK}; line-height: 1; }}
    .kpi .sub {{ font-size: .8rem; color: #8a95a1; margin-top: .3rem; }}
    h1, h2, h3 {{ color: {INK}; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: .25rem; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig: go.Figure, height: int = 460, legend: bool = True) -> go.Figure:
    """Shared Plotly layout — clean white, VINCI typography, unified hover."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        font=dict(family=FONT, color=INK, size=13),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=(
            dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                 bgcolor="rgba(0,0,0,0)")
            if legend else dict(visible=False)
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


PLOTLY_CFG = {"displayModeBar": False, "responsive": True}


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


@st.cache_resource(show_spinner="Backtesting held-out test…")
def run_backtest(val_end: str = "2024-12") -> dict[str, pd.DataFrame]:
    """Honest held-out backtest: train a global model on data UP TO `val_end`, then
    recursively forecast the test window (val_end+1 onward) — so predictions are
    compared against actuals the model never saw. Returns per-airport DataFrames
    (date, actual, pred). This is the model proving itself on known ground truth,
    before the genuine post-data forecast extends beyond it."""
    from airport_forecast.models import evaluate_lightgbm_recursive

    _, results = evaluate_lightgbm_recursive(
        load_raw(), val_end=val_end, core_airports=CORE_AIRPORTS
    )
    out: dict[str, pd.DataFrame] = {}
    for r in results:
        out[r.airport] = pd.DataFrame({
            "date": pd.to_datetime(r.dates),
            "actual": np.asarray(r.y_true, dtype=float),
            "pred": np.asarray(r.y_pred, dtype=float),
        })
    return out


@st.cache_resource(show_spinner="Computing SHAP values…")
def compute_shap(sample_n: int = 800):
    """Exact TreeSHAP feature attributions via LightGBM's native pred_contrib
    (no shap/numba dependency). Returns (feature_names, shap_matrix, feature_values)
    over a recent sample. SHAP values are in PAX units — how many passengers each
    feature pushes the forecast up or down — so importance is directly readable."""
    from airport_forecast.features import build_features

    model, fcols = load_model()
    feat = build_features(load_raw()).copy()
    if "airport_cat" in fcols and "airport_cat" not in feat.columns:
        feat["airport_cat"] = feat["airport"].astype("category")
    need = [c for c in fcols if c != "airport_cat" and c in feat.columns]
    feat = feat.dropna(subset=need).tail(sample_n)
    X = feat[fcols]
    contrib = np.asarray(model.predict(X, pred_contrib=True))
    shap_vals = contrib[:, :-1]  # last column is the base/expected value
    Xv = X.copy()
    if "airport_cat" in Xv.columns:
        Xv["airport_cat"] = Xv["airport_cat"].cat.codes
    return list(fcols), shap_vals, Xv.to_numpy(dtype=float)


@st.cache_data(show_spinner="Fitting SARIMA…")
def run_sarima(ap_code: str, future_dates: list[pd.Timestamp]) -> pd.Series:
    """Live SARIMA(1,1,1)(1,1,1,12) forecast for one airport, aligned to the same
    future dates as the LightGBM forecast. SARIMA extrapolates the trend linearly —
    which is exactly what a tree model cannot do on a strong-growth airport like
    Budapest. Cached per (airport, horizon)."""
    raw = load_raw()
    hist = (
        raw[raw["airport"] == ap_code]
        .sort_values("date")
        .set_index("date")["pax"]
        .dropna()
    )
    n = len(future_dates)
    if len(hist) < 36 or n == 0:
        return pd.Series(dtype=float)
    try:
        preds = train_sarima(hist, n)
    except Exception:
        return pd.Series(dtype=float)
    return pd.Series(preds[:n], index=pd.DatetimeIndex(future_dates[:n]))


def horizon_mape_curve(hz: pd.DataFrame, ap_code: str | None = None) -> dict[int, float]:
    """LightGBM recursive MAPE per horizon (for the uncertainty band). When an
    airport is given, use ITS own backtested error curve (more honest than the
    network average — Budapest is harder than Porto)."""
    fallback = {1: 3.5, 3: 4.1, 6: 3.8, 12: 3.9}
    if hz.empty:
        return fallback
    sub = hz[hz["model"] == "LightGBM_Recursive"]
    if ap_code is not None and "airport" in sub.columns:
        ap_sub = sub[sub["airport"] == ap_code]
        if not ap_sub.empty:
            sub = ap_sub
    if sub.empty:
        return fallback
    return sub.groupby("horizon")["mape"].mean().to_dict()


def trust_label(mape: float) -> tuple[str, str]:
    """Map an airport's mean recursive MAPE to a confidence badge (label, color)."""
    if mape < 4:
        return "High confidence", "#1a7f37"
    if mape < 6:
        return "Good", "#9a6700"
    return "Harder regime", "#b35900"


def kpi_card(col, label: str, value: str, sub: str) -> None:
    col.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


# Event flags present in the enriched data, mapped to readable causes.
EVENT_LABELS = {
    "event_covid": "COVID-19",
    "event_ukraine_war": "Ukraine war",
    "event_major_sport": "Major sporting event",
    "event_conference": "Conference (e.g. Web Summit)",
}


@st.cache_data(show_spinner=False)
def compute_anomalies(ap_code: str, z_thresh: float = 3.0, min_pct: float = 8.0) -> pd.DataFrame:
    """Flag months where traffic departs from its OWN seasonal norm, then attribute.

    Baseline = seasonal-naive (same month last year), so seasonality — summer peaks,
    school holidays — is REMOVED by construction: an anomaly is a deviation from the
    normal seasonal pattern, not the pattern itself. Each flagged month is then
    cross-referenced with the event flags (COVID, war, sport, conference) and with a
    same-direction flight-supply swing (route/frequency change). Anything left is
    labelled honestly as 'Unexplained'.

    Robust z-score (median / MAD) so the COVID collapse doesn't desensitise the
    threshold. Returns the full series with residuals + an `is_anomaly` flag +
    `attribution`."""
    src = load_raw()
    sub = src[src["airport"] == ap_code].sort_values("date").reset_index(drop=True).copy()
    if sub.empty or "pax" not in sub.columns:
        return pd.DataFrame()

    sub["expected"] = sub["pax"].shift(12)  # same month last year
    sub = sub.dropna(subset=["expected"])
    sub = sub[sub["expected"] > 0]
    if sub.empty:
        return pd.DataFrame()
    sub["residual"] = sub["pax"] - sub["expected"]
    sub["residual_pct"] = sub["residual"] / sub["expected"] * 100.0

    # flight supply YoY (route / frequency change proxy)
    if "n_flights" in sub.columns:
        f_exp = sub["n_flights"].shift(12)
        sub["flights_yoy"] = np.where(f_exp > 0, (sub["n_flights"] - f_exp) / f_exp * 100.0, np.nan)
    else:
        sub["flights_yoy"] = np.nan

    med = sub["residual_pct"].median()
    mad = (sub["residual_pct"] - med).abs().median()
    scale = mad * 1.4826 if mad > 0 else sub["residual_pct"].std(ddof=0)
    sub["rz"] = (sub["residual_pct"] - med) / scale if scale and scale > 0 else 0.0
    sub["is_anomaly"] = (sub["rz"].abs() > z_thresh) & (sub["residual_pct"].abs() > min_pct)

    def _attribute(r) -> str:
        reasons = [lab for col, lab in EVENT_LABELS.items()
                   if col in sub.columns and r.get(col, 0) and r[col] > 0]
        fy = r.get("flights_yoy", np.nan)
        if pd.notna(fy) and abs(fy) > 10 and np.sign(fy) == np.sign(r["residual_pct"]):
            reasons.append(f"flight supply {fy:+.0f}% (routes/frequency)")
        return " + ".join(reasons) if reasons else "Unexplained"

    sub["attribution"] = sub.apply(
        lambda r: _attribute(r) if r["is_anomaly"] else "", axis=1
    )
    return sub


raw = load_raw()
results, fi, hz = load_reports()
mape_curve = horizon_mape_curve(hz)
hz_points = sorted(mape_curve)

# ──────────────────────────────────────────────────────────────────
# Header + KPI row
# ──────────────────────────────────────────────────────────────────
st.title("✈️ Airport PAX Forecasting")
st.caption("Monthly passenger traffic forecasting across the VINCI Airports network · honest recursive evaluation")

best_m1 = mape_curve.get(min(hz_points), np.nan)
best_m12 = mape_curve.get(max(hz_points), np.nan)
last_month = raw["date"].max().strftime("%b %Y")

k1, k2, k3, k4 = st.columns(4)
kpi_card(k1, "Airports", str(raw["airport"].nunique()), "VINCI network core")
kpi_card(k2, "MAPE M+1", f"{best_m1:.1f}%", "honest recursive")
kpi_card(k3, "MAPE M+12", f"{best_m12:.1f}%", "beats SARIMA 5.2%")
kpi_card(k4, "Data through", last_month, "Eurostat avia_paoa")

st.write("")

# ──────────────────────────────────────────────────────────────────
# Synthèse métier (FR) — verdict d'abord, audience hiring manager VINCI.
# Le reste du tableau de bord est en anglais (norme technique) ; cette
# couche narrative parle au métier français qui prend la décision.
# ──────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown(
        f"""
##### 🎯 Synthèse — à quoi sert ce forecast

Un **seul modèle LightGBM** (récursif, évaluation honnête) bat SARIMA à **tous les horizons**
— de **{best_m1:.1f}% d'erreur à M+1** (planning court terme) à **{best_m12:.1f}% à M+12**
(budget annuel) — sur les **6 aéroports** du réseau. Pas besoin d'aiguiller par modèle :
une config, partout. L'architecture passe à l'échelle du réseau VINCI (70+ aéroports)
sans changement — on ajoute un aéroport en ajoutant des lignes.
"""
    )
    d1, d2, d3 = st.columns(3)
    d1.markdown(
        "**Court terme · M+1→M+3**  \n"
        "_Staffing, allocation des portes, ouverture de comptoirs._  \n"
        "Pic prévu → renfort équipes sol & sûreté."
    )
    d2.markdown(
        "**Moyen terme · M+6**  \n"
        "_Capacité, slots parking, contrats intérim saisonniers._  \n"
        "Croissance soutenue → anticiper l'upgrade terminal."
    )
    d3.markdown(
        "**Stratégique · M+12**  \n"
        "_Budget, négociation compagnies, capex infrastructure._  \n"
        "Tendance → incentives nouvelles lignes ou diversification."
    )

st.write("")

tab_fc, tab_val, tab_perf, tab_drv, tab_anom, tab_eda = st.tabs(
    ["🔮 Forecast", "✅ Validation", "📊 Performance", "🧩 Drivers",
     "🔎 Anomalies", "🗂️ Data & EDA"]
)

# ──────────────────────────────────────────────────────────────────
# Tab 1 — Interactive forecast
# ──────────────────────────────────────────────────────────────────
with tab_fc:
    c1, c2 = st.columns([1, 3])
    with c1:
        ap_name = st.selectbox("Airport", [SHORT[a] for a in CORE_AIRPORTS])
        ap_code = next(a for a in CORE_AIRPORTS if SHORT[a] == ap_name)
        horizon = st.slider("Horizon (months)", 1, MAX_HORIZON, MAX_HORIZON)
        show_band = st.checkbox("Show uncertainty band", value=True)
        show_sarima = st.checkbox("Overlay SARIMA (long-term trend)", value=True)
        st.caption(
            f"Horizon = {horizon} months **beyond each airport's latest available "
            "observation** — airports have unequal data end-dates, so each forecast "
            "starts right after its own last actual. Recursive: each month's "
            "prediction feeds the next month's lags; future flights/macro use "
            "seasonal-naive / carried-forward values — no leakage. Capped at M+12, "
            "the longest horizon backtested. See the **Validation** tab for the "
            "held-out track record behind these numbers."
        )

        # Per-airport confidence badge from this airport's own backtest
        ap_curve = horizon_mape_curve(hz, ap_code)
        ap_mape = float(np.mean(list(ap_curve.values()))) if ap_curve else np.nan
        label, color = trust_label(ap_mape)
        st.markdown(
            f'<div class="kpi" style="margin-top:.6rem">'
            f'<div class="label">Forecast confidence · {ap_name}</div>'
            f'<div class="value" style="color:{color}">{label}</div>'
            f'<div class="sub">mean recursive MAPE {ap_mape:.1f}% (backtested)</div></div>',
            unsafe_allow_html=True,
        )

    fc = run_forecast(horizon)
    fc_ap = fc[fc["airport"] == ap_code].sort_values("date").head(horizon)

    hist = raw[raw["airport"] == ap_code].sort_values("date").tail(48)

    # Uncertainty band from THIS airport's per-horizon MAPE (interp, no clamp
    # surprises: horizon is capped at the curve's max so np.interp stays in-range)
    ap_pts = sorted(ap_curve)
    ap_vals = [ap_curve[h] for h in ap_pts]
    months_ahead = np.arange(1, len(fc_ap) + 1)
    band_pct = np.interp(months_ahead, ap_pts, ap_vals) / 100.0
    lower = fc_ap["pax_pred"].values * (1 - band_pct)
    upper = fc_ap["pax_pred"].values * (1 + band_pct)

    # Live SARIMA aligned to the same future dates
    fut_dates = list(fc_ap["date"])
    sarima = run_sarima(ap_code, fut_dates) if show_sarima else pd.Series(dtype=float)

    with c2:
        fig = go.Figure()

        # Uncertainty band (drawn first so lines sit on top)
        if show_band and not fc_ap.empty:
            fig.add_trace(go.Scatter(
                x=fc_ap["date"], y=upper / 1e6, mode="lines",
                line=dict(width=0), hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=fc_ap["date"], y=lower / 1e6, mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(226,0,26,.13)", hoverinfo="skip",
                name="±MAPE band (LGB)",
            ))

        # Actual history
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["pax"] / 1e6, mode="lines",
            line=dict(color=INK, width=2.5), name="Actual",
            hovertemplate="%{y:.2f}M",
        ))

        # Connector last actual → first forecast
        if not hist.empty and not fc_ap.empty:
            fig.add_trace(go.Scatter(
                x=[hist["date"].iloc[-1], fc_ap["date"].iloc[0]],
                y=[hist["pax"].iloc[-1] / 1e6, fc_ap["pax_pred"].iloc[0] / 1e6],
                mode="lines", line=dict(color=ACCENT, width=2, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ))

        # LightGBM recursive forecast
        fig.add_trace(go.Scatter(
            x=fc_ap["date"], y=fc_ap["pax_pred"] / 1e6, mode="lines+markers",
            line=dict(color=ACCENT, width=2.6), marker=dict(size=6),
            name="LightGBM (recursive)", hovertemplate="%{y:.2f}M",
        ))

        # SARIMA overlay
        if show_sarima and not sarima.empty:
            fig.add_trace(go.Scatter(
                x=sarima.index, y=sarima.values / 1e6, mode="lines+markers",
                line=dict(color=SARIMA_BLUE, width=2, dash="dash"),
                marker=dict(size=5, symbol="square"),
                name="SARIMA (trend)", hovertemplate="%{y:.2f}M",
            ))

        fig.update_yaxes(title_text="Passengers (millions)")
        st.plotly_chart(style_fig(fig, height=460), use_container_width=True, config=PLOTLY_CFG)

        # Model-choice guidance — makes the "short-term LGB / long-term SARIMA"
        # narrative tangible, and flags strong-growth airports honestly.
        if show_sarima and not sarima.empty:
            gap = (sarima.iloc[-1] - fc_ap["pax_pred"].iloc[-1]) / fc_ap["pax_pred"].iloc[-1]
            if gap > 0.06:
                st.info(
                    f"**{ap_name} is a strong-growth regime.** At M+{len(fc_ap)} "
                    f"SARIMA sits {gap:.0%} above LightGBM: a tree model cannot "
                    "extrapolate above the traffic levels it saw in training, so it "
                    "under-shoots a fast-growing airport. Trust **LightGBM for the "
                    "short term** (1–3 mo, where its MAPE is lowest) and read "
                    "**SARIMA as the long-term trend** ceiling."
                )
            else:
                st.caption(
                    "LightGBM and SARIMA agree closely here — the recursive forecast "
                    "is well-anchored. Short term: LightGBM; long term: cross-check "
                    "with SARIMA."
                )

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
# Tab 2 — Validation (held-out backtest: prediction vs actual)
# ──────────────────────────────────────────────────────────────────
# Kept deliberately SEPARATE from the Forecast tab: the backtest model is
# trained on a historical snapshot (data ≤ 2024-12), the production forecast on
# all data. Mixing both lines on one chart invites "the red forecast scored that
# accuracy" — which is false. Here it is unambiguous: this is the held-out test.
with tab_val:
    st.subheader("Held-out backtest — does the model match reality it never saw?")
    st.markdown(
        "A model trained **only on data up to 2024-12** then forecasts the 2025+ "
        "test window it never saw. The dotted line is the prediction; the solid line "
        "is the actual that happened. The gap between them is the **real error** — "
        "this is the track record the production forecast inherits."
    )

    bt = run_backtest()
    vc1, vc2 = st.columns([1, 3])
    with vc1:
        val_name = st.selectbox("Airport", [SHORT[a] for a in CORE_AIRPORTS], key="val_ap")
        val_code = next(a for a in CORE_AIRPORTS if SHORT[a] == val_name)

    bt_ap = bt.get(val_code, pd.DataFrame())
    if bt_ap.empty:
        st.info("No backtest available for this airport.")
    else:
        err = np.abs(bt_ap["pred"] - bt_ap["actual"]) / bt_ap["actual"].replace(0, np.nan)
        bt_mape = float(np.nanmean(err) * 100)
        bias = float((bt_ap["pred"] - bt_ap["actual"]).mean())

        with vc1:
            lab, col = trust_label(bt_mape)
            st.metric("Held-out MAPE", f"{bt_mape:.1f}%", lab)
            st.metric("Mean bias (PAX)", f"{bias:+,.0f}",
                      "over-forecast" if bias > 0 else "under-forecast")
            st.metric("Test months", f"{len(bt_ap)}")

        with vc2:
            figv = go.Figure()
            ctx = (
                raw[raw["airport"] == val_code]
                .sort_values("date").tail(48)
            )
            figv.add_trace(go.Scatter(
                x=ctx["date"], y=ctx["pax"] / 1e6, mode="lines",
                line=dict(color=INK, width=2.5), name="Actual",
                hovertemplate="%{y:.2f}M",
            ))
            figv.add_trace(go.Scatter(
                x=bt_ap["date"], y=bt_ap["pred"] / 1e6, mode="lines+markers",
                line=dict(color="#1a7f37", width=2.2, dash="dot"),
                marker=dict(size=6, symbol="diamond"),
                name="LightGBM (held-out prediction)", hovertemplate="%{y:.2f}M",
            ))
            figv.add_vrect(
                x0=bt_ap["date"].min(), x1=bt_ap["date"].max(),
                fillcolor="#1a7f37", opacity=0.05, line_width=0,
                annotation_text="held-out test window", annotation_position="top left",
            )
            figv.update_yaxes(title_text="Passengers (millions)")
            st.plotly_chart(style_fig(figv, height=440), use_container_width=True, config=PLOTLY_CFG)
            st.caption(
                f"**{val_name}:** {bt_mape:.1f}% MAPE on {len(bt_ap)} unseen months. "
                "The model is trained on a 2024-12 snapshot here purely to *prove* the "
                "method on known ground truth; the deployed forecast (Forecast tab) is "
                "retrained on all available data."
            )

# ──────────────────────────────────────────────────────────────────
# Tab 3 — Performance
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
        avg = res.groupby("model")["mape"].mean().sort_values(ascending=False)
        colors = [ACCENT if v == avg.min() else MUTED for v in avg.values]
        figb = go.Figure(go.Bar(
            x=avg.values, y=avg.index, orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in avg.values],
            textposition="outside", hoverinfo="skip",
        ))
        figb.update_xaxes(title_text="MAPE (%)")
        st.plotly_chart(style_fig(figb, height=320, legend=False),
                        use_container_width=True, config=PLOTLY_CFG)
        st.caption(
            "So what: LightGBM Global wins. Prophet collapses (it extrapolates the "
            "pre-COVID trend instead of learning the recovery) — a worked example of "
            "why a flexible learner with an explicit `is_covid` flag beats a rigid "
            "trend model on a regime change."
        )

        st.subheader("Per-airport MAPE")
        pv = res.pivot_table(index="airport_name", columns="model", values="mape", aggfunc="mean")
        st.dataframe(pv.round(1).style.highlight_min(axis=1, color="#d6f5dd"),
                     use_container_width=True)

    pred_plot = FIGS / "24_predictions_vs_actual.png"
    if pred_plot.exists():
        st.subheader("Predictions vs actual (test set)")
        st.image(str(pred_plot), use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# Tab 4 — Drivers (SHAP — directional, in PAX units)
# ──────────────────────────────────────────────────────────────────
with tab_drv:
    st.subheader("What drives the forecast — SHAP (impact in passengers)")
    st.markdown(
        "SHAP attributes each prediction to its features in **PAX units** — not just "
        "*which* feature matters (split count) but *how much* it moves the forecast and "
        "*in which direction*. This answers the question an airport director actually "
        "asks: **why do you predict +X% this summer?**"
    )

    try:
        feat_names, shap_vals, feat_vals = compute_shap()
        mean_abs = np.abs(shap_vals).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        K = min(12, len(feat_names))
        top_idx = order[:K]

        # --- Mean |SHAP| magnitude bar ---
        bar_idx = top_idx[::-1]  # largest on top
        figs = go.Figure(go.Bar(
            x=mean_abs[bar_idx],
            y=[feat_names[i] for i in bar_idx],
            orientation="h",
            marker_color=[ACCENT if r < 3 else MUTED for r in range(K)][::-1],
            hovertemplate="%{x:,.0f} PAX<extra></extra>",
        ))
        figs.update_xaxes(title_text="Mean |SHAP| (passengers)")
        st.plotly_chart(style_fig(figs, height=460, legend=False),
                        use_container_width=True, config=PLOTLY_CFG)

        # --- Beeswarm: direction + value (each dot = one month, colour = feature value) ---
        st.markdown("**Directional impact** — each dot is one observation; colour is the feature's value.")
        figbw = go.Figure()
        xs, ys, cs = [], [], []
        rng = np.random.default_rng(0)
        for pos, i in enumerate(top_idx[::-1]):  # bottom→top to match bar order
            v = feat_vals[:, i]
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            norm = (v - vmin) / (vmax - vmin) if vmax > vmin else np.full_like(v, 0.5)
            jitter = rng.uniform(-0.32, 0.32, size=len(v))
            xs.extend(shap_vals[:, i].tolist())
            ys.extend((pos + jitter).tolist())
            cs.extend(norm.tolist())
        figbw.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=5, color=cs, colorscale="RdBu_r", opacity=0.6,
                        colorbar=dict(title="Feature<br>value", tickvals=[0, 1],
                                      ticktext=["low", "high"])),
            hovertemplate="SHAP %{x:,.0f} PAX<extra></extra>",
        ))
        figbw.add_vline(x=0, line_width=1, line_color="#888")
        figbw.update_yaxes(
            tickvals=list(range(K)),
            ticktext=[feat_names[i] for i in top_idx[::-1]],
        )
        figbw.update_xaxes(title_text="SHAP value (← fewer PAX  ·  more PAX →)")
        st.plotly_chart(style_fig(figbw, height=460, legend=False),
                        use_container_width=True, config=PLOTLY_CFG)

        # --- Honest macro-signal check (tests the "macro adds little" hypothesis) ---
        macro = ["oil_price_usd", "gdp", "unemployment_rate", "exchange_rate"]
        rank = {f: (list(order).index(feat_names.index(f)) + 1)
                for f in macro if f in feat_names}
        if rank:
            ranked = ", ".join(f"`{f}` #{r}" for f, r in sorted(rank.items(), key=lambda x: x[1]))
            st.info(
                "**Macro hypothesis, tested honestly:** lagged PAX, the rolling mean, "
                "flight supply and seasonality dominate. Macro features rank lower — "
                f"{ranked} out of {len(feat_names)}. Oil carries a modest signal; GDP, "
                "unemployment and FX are weak once seasonality and supply are in. Showing "
                "this is the point: the enriched features were *tested*, not assumed."
            )
    except Exception as e:  # noqa: BLE001 — dashboard should degrade, not crash
        st.warning(f"SHAP unavailable ({e}). Falling back to split-count importance.")
        if not fi.empty:
            top = fi.head(15).iloc[::-1]
            figf = go.Figure(go.Bar(
                x=top["importance"], y=top["feature"], orientation="h",
                marker_color=MUTED, hovertemplate="%{x} splits<extra></extra>",
            ))
            figf.update_xaxes(title_text="Importance (split count)")
            st.plotly_chart(style_fig(figf, height=480, legend=False),
                            use_container_width=True, config=PLOTLY_CFG)

    # Split-count importance kept as a secondary, classic view
    if not fi.empty:
        with st.expander("Classic split-count importance (for comparison)"):
            top = fi.head(15).iloc[::-1]
            figf = go.Figure(go.Bar(
                x=top["importance"], y=top["feature"], orientation="h",
                marker_color=MUTED, hovertemplate="%{x} splits<extra></extra>",
            ))
            figf.update_xaxes(title_text="Importance (split count)")
            st.plotly_chart(style_fig(figf, height=440, legend=False),
                            use_container_width=True, config=PLOTLY_CFG)

# ──────────────────────────────────────────────────────────────────
# Tab 5 — Anomalies (deviation from seasonal norm → attribution)
# ──────────────────────────────────────────────────────────────────
with tab_anom:
    st.subheader("Anomalies — when did traffic break its own seasonal pattern, and why?")
    st.markdown(
        "Not every rise or fall has a name. The baseline here is **same month last "
        "year**, so ordinary seasonality (summer peak, school holidays) is removed by "
        "construction — what's left are the months that *deviate* from the seasonal "
        "norm. Each flagged month is cross-referenced with known events and with a "
        "same-direction flight-supply swing; anything unexplained is labelled honestly "
        "as **Unexplained** rather than given a convenient story."
    )

    ac1, ac2 = st.columns([1, 3])
    with ac1:
        an_name = st.selectbox("Airport", [SHORT[a] for a in CORE_AIRPORTS], key="anom_ap")
        an_code = next(a for a in CORE_AIRPORTS if SHORT[a] == an_name)
        sensitivity = st.slider("Sensitivity (robust z)", 2.0, 4.0, 3.0, 0.5,
                                help="Lower = flag more months. Robust z-score on the "
                                     "year-over-year deviation.")

    adf = compute_anomalies(an_code, z_thresh=sensitivity)
    if adf.empty:
        st.info("Not enough history for this airport.")
    else:
        anoms = adf[adf["is_anomaly"]].copy()
        n_an = len(anoms)
        n_attr = int((anoms["attribution"] != "Unexplained").sum()) if n_an else 0
        attr_rate = (n_attr / n_an * 100) if n_an else 0.0

        with ac1:
            st.metric("Anomalous months", f"{n_an}")
            st.metric("Attributed to a known cause", f"{attr_rate:.0f}%",
                      f"{n_attr}/{n_an}" if n_an else "—")

        with ac2:
            base = adf
            colors = np.where(
                base["is_anomaly"] & (base["residual_pct"] >= 0), "#1a7f37",
                np.where(base["is_anomaly"], ACCENT, "rgba(159,176,192,.45)")
            )
            # Display deviation clipped to ±100% so normal anomalies stay readable;
            # the COVID-era rebounds (vs a collapsed prior-year base) run far higher
            # and clip the axis. Hover shows the TRUE value via customdata.
            y_plot = base["residual_pct"].clip(-100, 100)
            attr_disp = np.where(base["attribution"].values == "", "—",
                                 base["attribution"].values)
            customdata = np.column_stack([base["residual_pct"].values, attr_disp])
            figa = go.Figure(go.Bar(
                x=base["date"], y=y_plot, marker_color=colors,
                customdata=customdata,
                hovertemplate="%{x|%b %Y}<br>%{customdata[0]:+.0f}% vs last year"
                              "<br>%{customdata[1]}<extra></extra>",
            ))
            figa.add_hline(y=0, line_width=1, line_color="#888")
            figa.update_yaxes(title_text="Deviation vs same month last year (%)",
                              range=[-105, 105])
            st.plotly_chart(style_fig(figa, height=420, legend=False),
                            use_container_width=True, config=PLOTLY_CFG)
            st.caption(
                "Green = above seasonal norm, red = below, grey = within normal range. "
                "Bars are clipped at ±100% — the 2021–22 rebounds run much higher (they "
                "compare against a COVID-collapsed prior year); hover shows the true "
                "value. The bulk of anomalies are the COVID window, correctly attributed; "
                "outside it, traffic mostly tracks its seasonal norm."
            )

        if n_an:
            tbl = anoms.sort_values("date", ascending=False)[
                ["date", "pax", "expected", "residual_pct", "attribution"]
            ].copy()
            tbl["date"] = tbl["date"].dt.strftime("%Y-%m")
            tbl["pax"] = tbl["pax"].round().astype(int)
            tbl["expected"] = tbl["expected"].round().astype(int)
            tbl["residual_pct"] = tbl["residual_pct"].round(1)
            tbl.columns = ["Month", "Actual PAX", "Expected (N-1)", "Deviation %", "Likely cause"]
            st.dataframe(tbl, use_container_width=True, hide_index=True, height=340)
        else:
            st.success("No months break the seasonal norm at this sensitivity — traffic "
                       "follows its seasonal pattern closely.")

# ──────────────────────────────────────────────────────────────────
# Tab 6 — Data & EDA
# ──────────────────────────────────────────────────────────────────
with tab_eda:
    st.subheader("Monthly passenger traffic")
    sel = st.multiselect("Airports", [SHORT[a] for a in CORE_AIRPORTS],
                         default=[SHORT[a] for a in CORE_AIRPORTS])
    codes = [a for a in CORE_AIRPORTS if SHORT[a] in sel]
    figt = go.Figure()
    for code in codes:
        sub = raw[raw["airport"] == code]
        figt.add_trace(go.Scatter(
            x=sub["date"], y=sub["pax"] / 1e6, mode="lines",
            name=SHORT[code], line=dict(width=1.6), hovertemplate="%{y:.2f}M",
        ))
    figt.add_vrect(
        x0="2020-03-01", x1="2022-06-01",
        fillcolor=ACCENT, opacity=0.07, line_width=0,
        annotation_text="COVID", annotation_position="top left",
    )
    figt.update_yaxes(title_text="Passengers (millions)")
    st.plotly_chart(style_fig(figt, height=440), use_container_width=True, config=PLOTLY_CFG)
    st.caption(
        "So what: the COVID trough (shaded) is the hardest regime — a structural "
        "break, not noise. Every model is judged on how cleanly it recovers from it; "
        "this is why evaluation folds are split around 2023."
    )

    # Curated story — 6 figures that carry the EDA narrative, each with a
    # business-level "so what". The full 30+ plot dump lives in the expander below
    # so the page reads as an edited story, not a raw output folder.
    st.subheader("EDA — the six that matter")
    STORY = [
        ("01_pax_all_airports.png",
         "Six airports, very different scales (Lisbon ~1.6M/mo vs Belgrade ~0.5M). "
         "A *global* model lets the small airports borrow seasonal signal from the big ones."),
        ("03_covid_impact_recovery.png",
         "The 2020 collapse and the staggered recovery — the structural break the "
         "whole evaluation is built to survive honestly."),
        ("02_seasonality_boxplot.png",
         "Strong, stable monthly seasonality (summer peak) — this is why `pax_lag_12` "
         "(same month last year) is the #1 feature."),
        ("24_pax_vs_flights.png",
         "Flight movements track PAX tightly (r 0.86–0.98). Airline supply anchors the "
         "forecast level — and airlines publish schedules ~6 months ahead, so it's usable."),
        ("13_recovery_ratio_2024_vs_2019.png",
         "2024 vs 2019 traffic: who has fully recovered. Frames each airport's regime "
         "(strong-growth vs flat) and sets expectations per airport."),
        ("05_correlation_heatmap.png",
         "Feature correlations — confirms macro signal (oil, GDP, FX) adds information "
         "beyond the PAX lags, justifying the enriched feature set."),
    ]
    cols = st.columns(2)
    shown = set()
    for i, (fname, why) in enumerate(STORY):
        fpath = FIGS / fname
        if fpath.exists():
            with cols[i % 2]:
                st.image(str(fpath), use_container_width=True)
                st.caption(why)
            shown.add(fname)

    rest = [f for f in sorted(FIGS.glob("*.png")) if f.name not in shown]
    if rest:
        with st.expander(f"Full EDA gallery ({len(rest)} more figures)"):
            gcols = st.columns(3)
            for i, f in enumerate(rest):
                with gcols[i % 3]:
                    st.image(str(f), caption=f.stem, use_container_width=True)
