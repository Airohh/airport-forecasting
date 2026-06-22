"""Full EDA on enriched dataset (PAX + macro + events)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from airport_forecast.features import build_features, temporal_train_val_test_split

FIGS = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("tab10", 8)

# Load enriched data
df = pd.read_parquet(
    Path(__file__).resolve().parent.parent / "data" / "processed" / "pax_enriched.parquet"
)
df["date"] = pd.to_datetime(df["date"])
feat = build_features(df)

SHORT = {
    "FR_LFLL": "Lyon", "FR_LFRS": "Nantes", "UK_EGKK": "Gatwick",
    "HU_LHBP": "Budapest", "PT_LPPT": "Lisbon", "PT_LPPR": "Porto",
    "RS_LYBE": "Belgrade", "UK_EGPH": "Edinburgh",
}
feat["name"] = feat["airport"].map(SHORT)
CORE = ["FR_LFLL", "FR_LFRS", "HU_LHBP", "PT_LPPT", "PT_LPPR", "RS_LYBE"]
core = feat[feat["airport"].isin(CORE)]

print("Generating full EDA on enriched dataset...")
print(f"Dataset: {feat.shape[0]} rows, {feat.shape[1]} columns")
print(f"Columns: {feat.columns.tolist()}\n")

# ══════════════════════════════════════════════════════════
# PART 1 — TIME SERIES BASICS
# ══════════════════════════════════════════════════════════

# 01. Full PAX time series all airports
fig, ax = plt.subplots(figsize=(16, 7))
for i, (code, grp) in enumerate(feat.groupby("airport")):
    ax.plot(grp["date"], grp["pax"] / 1e6, label=SHORT[code], color=COLORS[i], linewidth=1.2)
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-06-01"),
           alpha=0.15, color="red", label="COVID")
ax.set_ylabel("Passengers (millions)")
ax.set_title("Monthly PAX - VINCI Airports Network")
ax.legend(loc="upper left", ncol=2)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig(FIGS / "01_pax_all_airports.png", dpi=150)
plt.close()
print("  01 - Full time series")

# 02. Seasonality boxplot by month (pre-COVID)
fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=False)
for ax, code in zip(axes.flat, CORE):
    sub = core[(core["airport"] == code) & (core["date"] < "2020-01-01")]
    sns.boxplot(data=sub, x="month", y="pax", ax=ax, color=COLORS[0], fliersize=2)
    ax.set_title(SHORT[code])
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
fig.suptitle("Seasonality by Month (pre-COVID)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "02_seasonality_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  02 - Seasonality boxplots")

# 03. COVID impact normalized to Jan 2020
fig, ax = plt.subplots(figsize=(14, 7))
window = feat[(feat["date"] >= "2019-01-01") & (feat["date"] <= "2025-12-01")]
for i, code in enumerate(CORE):
    sub = window[window["airport"] == code].copy()
    jan = sub.loc[sub["date"] == "2020-01-01", "pax"]
    if len(jan) == 0:
        continue
    sub["pax_norm"] = sub["pax"] / jan.values[0] * 100
    ax.plot(sub["date"], sub["pax_norm"], label=SHORT[code], color=COLORS[i], linewidth=1.5)
ax.axhline(100, color="gray", linestyle="--", alpha=0.5, label="Jan 2020 baseline")
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-06-01"), alpha=0.12, color="red")
ax.set_ylabel("PAX (% of Jan 2020)")
ax.set_title("COVID Impact & Recovery")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGS / "03_covid_impact_recovery.png", dpi=150)
plt.close()
print("  03 - COVID impact & recovery")

# 04. Annual trend
annual = core.groupby(["airport", "year"])["pax"].sum().reset_index()
fig, ax = plt.subplots(figsize=(14, 7))
for i, code in enumerate(CORE):
    sub = annual[annual["airport"] == code]
    ax.plot(sub["year"], sub["pax"] / 1e6, marker="o", markersize=4,
            label=SHORT[code], color=COLORS[i], linewidth=1.5)
ax.set_ylabel("Annual PAX (millions)")
ax.set_title("Annual Passenger Trend")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "04_annual_trend.png", dpi=150)
plt.close()
print("  04 - Annual trend")

# ══════════════════════════════════════════════════════════
# PART 2 — CORRELATIONS & STRUCTURE
# ══════════════════════════════════════════════════════════

# 05. Correlation heatmap (levels)
pivot = feat.pivot_table(index="date", columns="name", values="pax")
corr = pivot.corr()
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0.5, vmin=0, vmax=1, ax=ax, square=True)
ax.set_title("PAX Correlation Between Airports (monthly)")
fig.tight_layout()
fig.savefig(FIGS / "05_correlation_heatmap.png", dpi=150)
plt.close()
print("  05 - Correlation heatmap")

# 06. YoY growth distribution
yoy = core.dropna(subset=["pax_yoy_growth"])
yoy = yoy[(yoy["pax_yoy_growth"] > -1) & (yoy["pax_yoy_growth"] < 3)]
fig, ax = plt.subplots(figsize=(12, 6))
for i, code in enumerate(CORE):
    sub = yoy[yoy["airport"] == code]
    ax.hist(sub["pax_yoy_growth"] * 100, bins=30, alpha=0.4, label=SHORT[code], color=COLORS[i])
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.set_xlabel("Year-over-Year Growth (%)")
ax.set_title("Distribution of Monthly YoY PAX Growth")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "06_yoy_growth_distribution.png", dpi=150)
plt.close()
print("  06 - YoY growth distribution")

# 07. Seasonal decomposition (Lyon)
lyon = feat[(feat["airport"] == "FR_LFLL") & (feat["date"] < "2020-01-01")].set_index("date")["pax"]
decomp = seasonal_decompose(lyon, model="multiplicative", period=12)
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
decomp.observed.plot(ax=axes[0], title="Observed")
decomp.trend.plot(ax=axes[1], title="Trend")
decomp.seasonal.plot(ax=axes[2], title="Seasonal")
decomp.resid.plot(ax=axes[3], title="Residual")
for ax in axes:
    ax.set_xlabel("")
fig.suptitle("Multiplicative Decomposition - Lyon (pre-COVID)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "07_decomposition_lyon.png", dpi=150, bbox_inches="tight")
plt.close()
print("  07 - Seasonal decomposition (Lyon)")

# 08. Holiday effect
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, code in enumerate(CORE):
    sub = core[(core["airport"] == code) & (core["date"] < "2020-01-01")]
    vac = sub[sub["is_school_vacation"] == 1]["pax"].mean()
    novac = sub[sub["is_school_vacation"] == 0]["pax"].mean()
    axes[0].bar(i, (vac / novac - 1) * 100, color=COLORS[i])
axes[0].set_xticks(range(len(CORE)))
axes[0].set_xticklabels([SHORT[c] for c in CORE], rotation=45)
axes[0].set_ylabel("PAX uplift (%)")
axes[0].set_title("School Vacation Effect on PAX")
axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)

for i, code in enumerate(CORE):
    sub = core[(core["airport"] == code) & (core["date"] < "2020-01-01")]
    r, _ = stats.pearsonr(sub["n_holidays"], sub["pax"])
    axes[1].bar(i, r, color=COLORS[i])
axes[1].set_xticks(range(len(CORE)))
axes[1].set_xticklabels([SHORT[c] for c in CORE], rotation=45)
axes[1].set_ylabel("Pearson r")
axes[1].set_title("Correlation: N Holidays vs PAX")
fig.tight_layout()
fig.savefig(FIGS / "08_holiday_effect.png", dpi=150)
plt.close()
print("  08 - Holiday effect")

# ══════════════════════════════════════════════════════════
# PART 3 — AUTOCORRELATION
# ══════════════════════════════════════════════════════════

# 09. ACF / PACF
fig, axes = plt.subplots(6, 2, figsize=(16, 20))
for i, code in enumerate(CORE):
    sub = feat[(feat["airport"] == code) & (feat["date"] < "2020-01-01")]["pax"].dropna()
    max_lags = min(36, len(sub) // 2 - 1)
    plot_acf(sub, lags=max_lags, ax=axes[i, 0], title=f"{SHORT[code]} - ACF")
    plot_pacf(sub, lags=max_lags, ax=axes[i, 1], title=f"{SHORT[code]} - PACF", method="ywm")
fig.suptitle("Autocorrelation & Partial Autocorrelation (pre-COVID)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "09_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.close()
print("  09 - ACF/PACF")

# 10. Lag scatter (lag 1 and lag 12)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, code in enumerate(CORE):
    row, col = i // 3, i % 3
    sub = core[core["airport"] == code].dropna(subset=["pax_lag_12"])
    axes[row, col].scatter(sub["pax_lag_12"] / 1e6, sub["pax"] / 1e6, alpha=0.4, s=15, color=COLORS[i])
    r, _ = stats.pearsonr(sub["pax_lag_12"], sub["pax"])
    axes[row, col].set_title(f"{SHORT[code]} - lag 12 (r={r:.3f})")
    axes[row, col].set_xlabel("PAX(t-12) M")
    axes[row, col].set_ylabel("PAX(t) M")
fig.suptitle("Lag-12 Scatter: PAX(t) vs PAX(t-12)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "10_lag_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("  10 - Lag scatter")

# ══════════════════════════════════════════════════════════
# PART 4 — MACRO-ECONOMIC FEATURES (NEW)
# ══════════════════════════════════════════════════════════

# 11. Oil price vs total PAX
fig, ax1 = plt.subplots(figsize=(16, 7))
total_pax = core.groupby("date")["pax"].sum().reset_index()
ax1.plot(total_pax["date"], total_pax["pax"] / 1e6, color="steelblue", label="Total PAX (6 airports)")
ax1.set_ylabel("Total PAX (millions)", color="steelblue")
ax2 = ax1.twinx()
oil = feat.drop_duplicates(subset=["date"])[["date", "oil_price_usd"]].dropna().sort_values("date")
ax2.plot(oil["date"], oil["oil_price_usd"], color="orange", alpha=0.7, label="Brent USD/barrel")
ax2.set_ylabel("Brent Oil (USD/barrel)", color="orange")
ax1.set_title("Total PAX vs Brent Oil Price")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-06-01"), alpha=0.1, color="red")
fig.tight_layout()
fig.savefig(FIGS / "11_oil_vs_pax.png", dpi=150)
plt.close()
print("  11 - Oil price vs PAX")

# 12. Unemployment vs PAX per country
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
country_airports = {"FR": ["FR_LFLL", "FR_LFRS"], "HU": ["HU_LHBP"],
                    "PT": ["PT_LPPT", "PT_LPPR"], "RS": ["RS_LYBE"]}
COUNTRY_MAP = {"FR_LFLL": "FR", "FR_LFRS": "FR", "HU_LHBP": "HU",
               "PT_LPPT": "PT", "PT_LPPR": "PT", "RS_LYBE": "RS"}
feat["country"] = feat["airport"].map(COUNTRY_MAP)

for idx, (country, aps) in enumerate(country_airports.items()):
    ax = axes[idx // 2, idx % 2]
    sub = feat[feat["airport"].isin(aps)].dropna(subset=["unemployment_rate"])
    country_pax = sub.groupby("date").agg(pax=("pax", "sum"), unemp=("unemployment_rate", "mean")).reset_index()
    ax_twin = ax.twinx()
    ax.plot(country_pax["date"], country_pax["pax"] / 1e6, color="steelblue", label="PAX")
    ax_twin.plot(country_pax["date"], country_pax["unemp"], color="red", alpha=0.6, label="Unemployment %")
    ax.set_title(f"{country}: PAX vs Unemployment")
    ax.set_ylabel("PAX (M)", color="steelblue")
    ax_twin.set_ylabel("Unemployment %", color="red")
fig.tight_layout()
fig.savefig(FIGS / "12_unemployment_vs_pax.png", dpi=150)
plt.close()
print("  12 - Unemployment vs PAX per country")

# 13. GDP vs PAX per country
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, (country, aps) in enumerate(country_airports.items()):
    ax = axes[idx // 2, idx % 2]
    sub = feat[feat["airport"].isin(aps)].dropna(subset=["gdp"])
    if sub.empty:
        ax.set_title(f"{country}: No GDP data")
        continue
    country_data = sub.groupby("date").agg(pax=("pax", "sum"), gdp=("gdp", "mean")).reset_index()
    ax_twin = ax.twinx()
    ax.plot(country_data["date"], country_data["pax"] / 1e6, color="steelblue")
    ax_twin.plot(country_data["date"], country_data["gdp"] / 1e3, color="green", alpha=0.6)
    ax.set_title(f"{country}: PAX vs GDP")
    ax.set_ylabel("PAX (M)", color="steelblue")
    ax_twin.set_ylabel("GDP (B EUR)", color="green")
fig.tight_layout()
fig.savefig(FIGS / "13_gdp_vs_pax.png", dpi=150)
plt.close()
print("  13 - GDP vs PAX per country")

# 14. Exchange rate vs PAX (HU only - most interesting)
fig, ax1 = plt.subplots(figsize=(14, 6))
hu = feat[(feat["airport"] == "HU_LHBP")].dropna(subset=["exchange_rate"]).sort_values("date")
ax1.plot(hu["date"], hu["pax"] / 1e6, color="steelblue", label="Budapest PAX")
ax1.set_ylabel("PAX (M)", color="steelblue")
ax2 = ax1.twinx()
ax2.plot(hu["date"], hu["exchange_rate"], color="purple", alpha=0.6, label="EUR/HUF")
ax2.set_ylabel("EUR/HUF", color="purple")
ax1.set_title("Budapest: PAX vs EUR/HUF Exchange Rate")
lines1, l1 = ax1.get_legend_handles_labels()
lines2, l2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, l1 + l2, loc="upper left")
fig.tight_layout()
fig.savefig(FIGS / "14_exchange_rate_budapest.png", dpi=150)
plt.close()
print("  14 - Exchange rate vs PAX (Budapest)")

# ══════════════════════════════════════════════════════════
# PART 5 — CROSS-AIRPORT ANALYSIS
# ══════════════════════════════════════════════════════════

# 15. Cross-correlation: levels vs changes
pivot_core = feat[feat["airport"].isin(CORE)].pivot_table(index="date", columns="name", values="pax")
diff_pivot = pivot_core.diff().dropna()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
mask1 = np.triu(np.ones_like(pivot_core.corr(), dtype=bool), k=1)
sns.heatmap(pivot_core.corr(), mask=mask1, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0.5, vmin=0, vmax=1, ax=axes[0], square=True)
axes[0].set_title("Correlation: PAX Levels")
mask2 = np.triu(np.ones_like(diff_pivot.corr(), dtype=bool), k=1)
sns.heatmap(diff_pivot.corr(), mask=mask2, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-0.5, vmax=1, ax=axes[1], square=True)
axes[1].set_title("Correlation: PAX Changes (diff)")
fig.suptitle("Level vs Change Correlation", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "15_cross_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  15 - Cross-correlation levels vs changes")

# 16. Growth regimes
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, code in enumerate(CORE):
    ax = axes[i // 3, i % 3]
    sub = core[core["airport"] == code]
    pre = sub[sub["date"] < "2020-01-01"]
    covid = sub[(sub["date"] >= "2020-01-01") & (sub["date"] < "2022-07-01")]
    post = sub[sub["date"] >= "2022-07-01"]
    ax.plot(pre["date"], pre["pax"] / 1e6, color="steelblue", label="Pre-COVID")
    ax.plot(covid["date"], covid["pax"] / 1e6, color="red", label="COVID")
    ax.plot(post["date"], post["pax"] / 1e6, color="green", label="Recovery")
    ax.set_title(SHORT[code])
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
fig.suptitle("Growth Regimes: Pre-COVID / COVID / Recovery", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "16_growth_regimes.png", dpi=150, bbox_inches="tight")
plt.close()
print("  16 - Growth regimes")

# 17. Recovery ratio 2024 vs 2019
fig, ax = plt.subplots(figsize=(14, 6))
for i, code in enumerate(CORE):
    sub = core[core["airport"] == code]
    y19 = sub[sub["year"] == 2019].set_index("month")["pax"]
    y24 = sub[sub["year"] == 2024].set_index("month")["pax"]
    common = y19.index.intersection(y24.index)
    if len(common) == 0:
        continue
    ratio = y24.loc[common] / y19.loc[common] * 100
    ax.plot(ratio.index, ratio.values, marker="o", label=SHORT[code], color=COLORS[i])
ax.axhline(100, color="gray", linestyle="--", alpha=0.5, label="2019 level")
ax.set_xlabel("Month")
ax.set_ylabel("2024 PAX as % of 2019")
ax.set_title("Recovery: 2024 vs 2019")
ax.set_xticks(range(1, 13))
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "17_recovery_ratio.png", dpi=150)
plt.close()
print("  17 - Recovery ratio 2024 vs 2019")

# 18. Seasonal indices comparison
fig, ax = plt.subplots(figsize=(12, 6))
for i, code in enumerate(CORE):
    sub = core[(core["airport"] == code) & (core["date"] < "2020-01-01")]
    monthly_avg = sub.groupby("month")["pax"].mean()
    seasonal_idx = monthly_avg / monthly_avg.mean()
    ax.plot(seasonal_idx.index, seasonal_idx.values, marker="o", label=SHORT[code], color=COLORS[i])
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Month")
ax.set_ylabel("Seasonal Index (1.0 = average)")
ax.set_title("Monthly Seasonal Indices (pre-COVID)")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "18_seasonal_indices.png", dpi=150)
plt.close()
print("  18 - Seasonal indices")

# ══════════════════════════════════════════════════════════
# PART 6 — FEATURE IMPORTANCE PREVIEW
# ══════════════════════════════════════════════════════════

# 19. All feature correlations with PAX
feature_cols = [c for c in feat.columns if c not in
                ["airport", "airport_name", "date", "pax", "name", "country"]]
numeric_feats = feat[feature_cols + ["pax"]].select_dtypes(include=[np.number])
corr_pax = numeric_feats.corr()["pax"].drop("pax").dropna().sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
colors_bar = ["red" if v < 0 else "steelblue" for v in corr_pax.values]
ax.barh(corr_pax.index, corr_pax.values, color=colors_bar)
ax.set_xlabel("Pearson Correlation with PAX")
ax.set_title("All Features: Correlation with PAX")
ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(FIGS / "19_all_feature_correlations.png", dpi=150)
plt.close()
print("  19 - All feature correlations")

# 20. Same-country pairs
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
lyon = core[core["airport"] == "FR_LFLL"].set_index("date")["pax"]
nantes = core[core["airport"] == "FR_LFRS"].set_index("date")["pax"]
common_fr = lyon.index.intersection(nantes.index)
axes[0].scatter(lyon.loc[common_fr] / 1e6, nantes.loc[common_fr] / 1e6, alpha=0.5, s=15)
r_fr, _ = stats.pearsonr(lyon.loc[common_fr], nantes.loc[common_fr])
axes[0].set_xlabel("Lyon PAX (M)")
axes[0].set_ylabel("Nantes PAX (M)")
axes[0].set_title(f"France: Lyon vs Nantes (r={r_fr:.3f})")

lisbon = core[core["airport"] == "PT_LPPT"].set_index("date")["pax"]
porto = core[core["airport"] == "PT_LPPR"].set_index("date")["pax"]
common_pt = lisbon.index.intersection(porto.index)
axes[1].scatter(lisbon.loc[common_pt] / 1e6, porto.loc[common_pt] / 1e6, alpha=0.5, s=15)
r_pt, _ = stats.pearsonr(lisbon.loc[common_pt], porto.loc[common_pt])
axes[1].set_xlabel("Lisbon PAX (M)")
axes[1].set_ylabel("Porto PAX (M)")
axes[1].set_title(f"Portugal: Lisbon vs Porto (r={r_pt:.3f})")
fig.suptitle("Same-Country Airport Pairs", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "20_same_country_pairs.png", dpi=150, bbox_inches="tight")
plt.close()
print("  20 - Same-country pairs")

# 21. Volatility over time (CV rolling 12)
fig, ax = plt.subplots(figsize=(14, 6))
for i, code in enumerate(CORE):
    sub = core[core["airport"] == code].copy()
    sub["cv_12"] = sub["pax"].rolling(12).std() / sub["pax"].rolling(12).mean()
    ax.plot(sub["date"], sub["cv_12"], label=SHORT[code], color=COLORS[i])
ax.set_ylabel("CV (rolling 12 months)")
ax.set_title("Volatility Over Time")
ax.legend()
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-06-01"), alpha=0.12, color="red")
fig.tight_layout()
fig.savefig(FIGS / "21_volatility.png", dpi=150)
plt.close()
print("  21 - Volatility")

# 22. Train / Val / Test split
train, val, test = temporal_train_val_test_split(core)
fig, ax = plt.subplots(figsize=(16, 6))
for i, code in enumerate(CORE):
    t = train[train["airport"] == code]
    v = val[val["airport"] == code]
    te = test[test["airport"] == code]
    ax.plot(t["date"], t["pax"] / 1e6, color=COLORS[i], alpha=0.6)
    ax.plot(v["date"], v["pax"] / 1e6, color=COLORS[i], linestyle="--", linewidth=2)
    ax.plot(te["date"], te["pax"] / 1e6, color=COLORS[i], linestyle=":", linewidth=2.5)
ax.axvline(pd.Timestamp("2024-01-01"), color="orange", linewidth=2, label="Val start")
ax.axvline(pd.Timestamp("2025-01-01"), color="red", linewidth=2, label="Test start")
ax.set_ylabel("PAX (millions)")
ax.set_title("Train (solid) / Validation (dashed) / Test (dotted)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIGS / "22_train_val_test.png", dpi=150)
plt.close()
print("  22 - Train/Val/Test split")

# ══════════════════════════════════════════════════════════
# PART 7 — STATIONARITY + SUMMARY
# ══════════════════════════════════════════════════════════

# ADF test
print("\n=== Stationarity (ADF test, pre-COVID) ===")
print(f"{'Airport':<12} {'ADF stat':>10} {'p-value':>10} {'Stationary?':>12}")
print("-" * 48)
for code in CORE:
    sub = feat[(feat["airport"] == code) & (feat["date"] < "2020-01-01")]["pax"].dropna()
    if len(sub) < 24:
        continue
    result = adfuller(sub, autolag="AIC")
    stationary = "YES" if result[1] < 0.05 else "NO"
    print(f"{SHORT[code]:<12} {result[0]:>10.3f} {result[1]:>10.4f} {stationary:>12}")

# Summary stats
print("\n=== Summary Stats ===")
summary = core.groupby("airport").agg(
    months=("pax", "count"),
    mean_pax=("pax", lambda x: f"{x.mean():,.0f}"),
    min_pax=("pax", lambda x: f"{x.min():,.0f}"),
    max_pax=("pax", lambda x: f"{x.max():,.0f}"),
    cv=("pax", lambda x: f"{x.std()/x.mean():.3f}"),
).reset_index()
summary["airport"] = summary["airport"].map(SHORT)
print(summary.to_string(index=False))

# Macro coverage
print("\n=== Macro Feature Coverage ===")
for c in ["unemployment_rate", "gdp", "oil_price_usd", "exchange_rate"]:
    ok = feat[c].notna().sum()
    print(f"  {c}: {ok}/{len(feat)} ({ok/len(feat)*100:.0f}%)")

print(f"\n{'='*60}")
print(f"TOTAL: 22 plots saved to {FIGS}")
print(f"{'='*60}")
