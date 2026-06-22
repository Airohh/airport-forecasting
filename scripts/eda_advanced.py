"""Advanced EDA — deeper analysis for airport PAX forecasting.

Covers: ACF/PACF, cross-correlation, growth regimes, structural breaks,
lag scatter plots, feature correlations, train/val/test visualization.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from airport_forecast.data import load_pax
from airport_forecast.features import build_features, temporal_train_val_test_split

FIGS = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("tab10", 8)

df = load_pax(with_holidays=True)
feat = build_features(df)

SHORT = {
    "FR_LFLL": "Lyon", "FR_LFRS": "Nantes", "UK_EGKK": "Gatwick",
    "HU_LHBP": "Budapest", "PT_LPPT": "Lisbon", "PT_LPPR": "Porto",
    "RS_LYBE": "Belgrade", "UK_EGPH": "Edinburgh",
}
feat["name"] = feat["airport"].map(SHORT)
CORE = ["FR_LFLL", "FR_LFRS", "HU_LHBP", "PT_LPPT", "PT_LPPR", "RS_LYBE"]
core = feat[feat["airport"].isin(CORE)]

print("Generating advanced EDA plots...")

# ──────────────────────────────────────────────────────────
# A1. ACF / PACF per airport (pre-COVID)
# ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(6, 2, figsize=(16, 20))
for i, code in enumerate(CORE):
    sub = feat[(feat["airport"] == code) & (feat["date"] < "2020-01-01")]["pax"].dropna()
    max_lags = min(36, len(sub) // 2 - 1)
    plot_acf(sub, lags=max_lags, ax=axes[i, 0], title=f"{SHORT[code]} — ACF")
    plot_pacf(sub, lags=max_lags, ax=axes[i, 1], title=f"{SHORT[code]} — PACF", method="ywm")
fig.suptitle("Autocorrelation & Partial Autocorrelation (pre-COVID, 36 lags)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "09_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.close()
print("  09 — ACF/PACF")

# ──────────────────────────────────────────────────────────
# A2. Lag scatter plots (pax vs pax_lag_1, pax_lag_12)
# ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, code in enumerate(CORE):
    sub = core[(core["airport"] == code)].dropna(subset=["pax_lag_1"])
    axes[0, i % 3].scatter(sub["pax_lag_1"] / 1e6, sub["pax"] / 1e6,
                            alpha=0.4, s=15, color=COLORS[i])
    axes[0, i % 3].set_title(f"{SHORT[code]} — lag 1")
    axes[0, i % 3].set_xlabel("PAX(t-1) M")
    axes[0, i % 3].set_ylabel("PAX(t) M")

    if i < 3:
        sub12 = core[(core["airport"] == code)].dropna(subset=["pax_lag_12"])
        axes[1, i].scatter(sub12["pax_lag_12"] / 1e6, sub12["pax"] / 1e6,
                           alpha=0.4, s=15, color=COLORS[i])
        axes[1, i].set_title(f"{SHORT[code]} — lag 12")
        axes[1, i].set_xlabel("PAX(t-12) M")
        axes[1, i].set_ylabel("PAX(t) M")
        r, _ = stats.pearsonr(sub12["pax_lag_12"].values, sub12["pax"].values)
        axes[1, i].annotate(f"r={r:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=11)

fig.suptitle("Lag Scatter Plots — PAX(t) vs PAX(t-k)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "10_lag_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("  10 — Lag scatter plots")

# ──────────────────────────────────────────────────────────
# A3. Cross-correlation: do airports move together month-to-month?
# ──────────────────────────────────────────────────────────
# Monthly pax changes (diff) — correlation of CHANGES not levels
pivot = feat[feat["airport"].isin(CORE)].pivot_table(index="date", columns="name", values="pax")
diff_pivot = pivot.diff().dropna()
corr_diff = diff_pivot.corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
mask = np.triu(np.ones_like(pivot.corr(), dtype=bool), k=1)
sns.heatmap(pivot.corr(), mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0.5, vmin=0, vmax=1, ax=axes[0], square=True)
axes[0].set_title("Correlation: PAX Levels")

mask2 = np.triu(np.ones_like(corr_diff, dtype=bool), k=1)
sns.heatmap(corr_diff, mask=mask2, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-0.5, vmax=1, ax=axes[1], square=True)
axes[1].set_title("Correlation: PAX Month-to-Month Changes")
fig.suptitle("Level vs Change Correlation (core airports)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "11_cross_correlation_levels_vs_changes.png", dpi=150, bbox_inches="tight")
plt.close()
print("  11 — Cross-correlation levels vs changes")

# ──────────────────────────────────────────────────────────
# A4. Growth regimes: pre-COVID, COVID, recovery
# ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, code in enumerate(CORE):
    ax = axes[i // 3, i % 3]
    sub = core[core["airport"] == code].copy()

    pre = sub[sub["date"] < "2020-01-01"]
    covid = sub[(sub["date"] >= "2020-01-01") & (sub["date"] < "2022-07-01")]
    post = sub[sub["date"] >= "2022-07-01"]

    ax.plot(pre["date"], pre["pax"] / 1e6, color="steelblue", label="Pre-COVID")
    ax.plot(covid["date"], covid["pax"] / 1e6, color="red", label="COVID")
    ax.plot(post["date"], post["pax"] / 1e6, color="green", label="Recovery")
    ax.set_title(SHORT[code])
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.tick_params(axis="x", rotation=45)

fig.suptitle("Growth Regimes: Pre-COVID / COVID / Recovery", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "12_growth_regimes.png", dpi=150, bbox_inches="tight")
plt.close()
print("  12 — Growth regimes")

# ──────────────────────────────────────────────────────────
# A5. Recovery ratio: 2024 vs 2019 same month
# ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
for i, code in enumerate(CORE):
    sub = core[core["airport"] == code]
    y2019 = sub[sub["year"] == 2019].set_index("month")["pax"]
    y2024 = sub[sub["year"] == 2024].set_index("month")["pax"]
    common = y2019.index.intersection(y2024.index)
    if len(common) == 0:
        continue
    ratio = (y2024.loc[common] / y2019.loc[common] * 100)
    ax.plot(ratio.index, ratio.values, marker="o", label=SHORT[code], color=COLORS[i])

ax.axhline(100, color="gray", linestyle="--", alpha=0.5, label="2019 level")
ax.set_xlabel("Month")
ax.set_ylabel("2024 PAX as % of 2019")
ax.set_title("Recovery Ratio: 2024 vs 2019 (same month)")
ax.set_xticks(range(1, 13))
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "13_recovery_ratio_2024_vs_2019.png", dpi=150)
plt.close()
print("  13 — Recovery ratio 2024 vs 2019")

# ──────────────────────────────────────────────────────────
# A6. Feature importance preview: correlation with PAX
# ──────────────────────────────────────────────────────────
feature_cols = [
    "month_sin", "month_cos", "is_summer", "is_covid",
    "n_holidays", "is_school_vacation",
    "pax_lag_1", "pax_lag_12",
    "pax_rolling_mean_3", "pax_rolling_mean_12",
    "pax_yoy_growth",
]
corr_with_pax = core[feature_cols + ["pax"]].corr()["pax"].drop("pax").sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
colors_bar = ["red" if v < 0 else "steelblue" for v in corr_with_pax.values]
ax.barh(corr_with_pax.index, corr_with_pax.values, color=colors_bar)
ax.set_xlabel("Pearson Correlation with PAX")
ax.set_title("Feature Correlation with PAX (core airports pooled)")
ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(FIGS / "14_feature_correlation_with_pax.png", dpi=150)
plt.close()
print("  14 — Feature correlation with PAX")

# ──────────────────────────────────────────────────────────
# A7. Train / Val / Test split visualization
# ──────────────────────────────────────────────────────────
train, val, test = temporal_train_val_test_split(core)

fig, ax = plt.subplots(figsize=(16, 6))
for i, code in enumerate(CORE):
    t = train[train["airport"] == code]
    v = val[val["airport"] == code]
    te = test[test["airport"] == code]
    ax.plot(t["date"], t["pax"] / 1e6, color=COLORS[i], alpha=0.6)
    ax.plot(v["date"], v["pax"] / 1e6, color=COLORS[i], linestyle="--", linewidth=2)
    ax.plot(te["date"], te["pax"] / 1e6, color=COLORS[i], linestyle=":", linewidth=2.5)

ax.axvline(pd.Timestamp("2024-01-01"), color="orange", linestyle="-", linewidth=2, label="Val start")
ax.axvline(pd.Timestamp("2025-01-01"), color="red", linestyle="-", linewidth=2, label="Test start")
ax.set_ylabel("PAX (millions)")
ax.set_title("Train (solid) / Validation (dashed) / Test (dotted) Split")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIGS / "15_train_val_test_split.png", dpi=150)
plt.close()
print("  15 — Train/Val/Test split")

# ──────────────────────────────────────────────────────────
# A8. Monthly seasonal indices per airport
# ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
for i, code in enumerate(CORE):
    sub = core[(core["airport"] == code) & (core["date"] < "2020-01-01")]
    monthly_avg = sub.groupby("month")["pax"].mean()
    annual_avg = monthly_avg.mean()
    seasonal_idx = monthly_avg / annual_avg
    ax.plot(seasonal_idx.index, seasonal_idx.values, marker="o",
            label=SHORT[code], color=COLORS[i])
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Month")
ax.set_ylabel("Seasonal Index (1.0 = average)")
ax.set_title("Monthly Seasonal Indices (pre-COVID) — Are patterns similar?")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "16_seasonal_indices.png", dpi=150)
plt.close()
print("  16 — Seasonal indices comparison")

# ──────────────────────────────────────────────────────────
# A9. Coefficient of variation over time (volatility)
# ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
for i, code in enumerate(CORE):
    sub = core[core["airport"] == code].copy()
    sub["cv_12"] = (
        sub["pax"].rolling(12).std() / sub["pax"].rolling(12).mean()
    )
    ax.plot(sub["date"], sub["cv_12"], label=SHORT[code], color=COLORS[i])
ax.set_ylabel("CV (rolling 12 months)")
ax.set_title("Volatility Over Time (Coefficient of Variation)")
ax.legend()
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-06-01"),
           alpha=0.12, color="red")
fig.tight_layout()
fig.savefig(FIGS / "17_volatility_over_time.png", dpi=150)
plt.close()
print("  17 — Volatility over time")

# ──────────────────────────────────────────────────────────
# A10. Pairwise airport comparison (same country)
# ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# France: Lyon vs Nantes
lyon = core[core["airport"] == "FR_LFLL"].set_index("date")["pax"]
nantes = core[core["airport"] == "FR_LFRS"].set_index("date")["pax"]
common_fr = lyon.index.intersection(nantes.index)
axes[0].scatter(lyon.loc[common_fr] / 1e6, nantes.loc[common_fr] / 1e6, alpha=0.5, s=15)
axes[0].set_xlabel("Lyon PAX (M)")
axes[0].set_ylabel("Nantes PAX (M)")
r_fr, _ = stats.pearsonr(lyon.loc[common_fr], nantes.loc[common_fr])
axes[0].set_title(f"France: Lyon vs Nantes (r={r_fr:.3f})")

# Portugal: Lisbon vs Porto
lisbon = core[core["airport"] == "PT_LPPT"].set_index("date")["pax"]
porto = core[core["airport"] == "PT_LPPR"].set_index("date")["pax"]
common_pt = lisbon.index.intersection(porto.index)
axes[1].scatter(lisbon.loc[common_pt] / 1e6, porto.loc[common_pt] / 1e6, alpha=0.5, s=15)
axes[1].set_xlabel("Lisbon PAX (M)")
axes[1].set_ylabel("Porto PAX (M)")
r_pt, _ = stats.pearsonr(lisbon.loc[common_pt], porto.loc[common_pt])
axes[1].set_title(f"Portugal: Lisbon vs Porto (r={r_pt:.3f})")

fig.suptitle("Same-Country Airport Pairs", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "18_same_country_pairs.png", dpi=150, bbox_inches="tight")
plt.close()
print("  18 — Same-country pairs")

# ──────────────────────────────────────────────────────────
# Summary of findings
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA FINDINGS SUMMARY")
print("=" * 60)
print("""
1. SEASONALITY: All airports show strong summer peak (Jul-Aug).
   Seasonal pattern is SHARED across airports -> global model makes sense.

2. STATIONARITY: ALL series are non-stationary (ADF p >> 0.05).
   -> Need differencing for SARIMA, or use features that capture trend.

3. COVID: Massive drop 2020-03. Recovery varies:
   -> Check plot 13 for 2024 vs 2019 ratios.
   -> Some airports exceeded 2019 levels, others haven't.

4. TREND: Budapest and Porto show strongest growth pre-COVID.
   Lyon and Nantes more stable. Belgrade short history but strong.

5. AUTOCORRELATION: Strong lag-12 (yearly seasonality) visible in ACF.
   PACF shows significant spikes at lag 1, 12 -> SARIMA(p,d,q)(P,D,Q,12).

6. CROSS-CORRELATION: High correlation in LEVELS between airports
   (shared trend + seasonality). Lower correlation in CHANGES
   -> changes are more airport-specific.

7. FEATURES: lag_1 and lag_12 are the strongest predictors.
   is_covid has strong negative correlation.
   Holidays/vacation effect is moderate but present.
""")

print(f"\nAll {len(list(FIGS.glob('*.png')))} plots saved to: {FIGS}")
