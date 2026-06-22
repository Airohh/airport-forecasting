"""EDA — Airport PAX Forecasting.

Generates all exploration plots in reports/figures/.
Run: python scripts/eda.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from airport_forecast.data import load_pax
from airport_forecast.features import build_features

FIGS = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("tab10", 8)

df = load_pax(with_holidays=True)
feat = build_features(df)

# Short names for plots
SHORT = {
    "FR_LFLL": "Lyon",
    "FR_LFRS": "Nantes",
    "UK_EGKK": "Gatwick",
    "HU_LHBP": "Budapest",
    "PT_LPPT": "Lisbon",
    "PT_LPPR": "Porto",
    "RS_LYBE": "Belgrade",
    "UK_EGPH": "Edinburgh",
}
feat["name"] = feat["airport"].map(SHORT)

# Only airports with post-2020 data for modeling plots
CORE_AIRPORTS = ["FR_LFLL", "FR_LFRS", "HU_LHBP", "PT_LPPT", "PT_LPPR", "RS_LYBE"]
core = feat[feat["airport"].isin(CORE_AIRPORTS)]

print("Generating EDA plots...")

# ──────────────────────────────────────────────────────────
# 1. Full PAX time series — all airports
# ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))
for i, (code, grp) in enumerate(feat.groupby("airport")):
    ax.plot(grp["date"], grp["pax"] / 1e6, label=SHORT[code], color=COLORS[i], linewidth=1.2)
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-06-01"),
           alpha=0.15, color="red", label="COVID")
ax.set_ylabel("Passengers (millions)")
ax.set_title("Monthly PAX — VINCI Airports Network")
ax.legend(loc="upper left", ncol=2)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig(FIGS / "01_pax_all_airports.png", dpi=150)
plt.close()
print("  01 — Full time series")

# ──────────────────────────────────────────────────────────
# 2. Seasonality — boxplot PAX by month (core airports)
# ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=False)
for ax, code in zip(axes.flat, CORE_AIRPORTS):
    sub = core[core["airport"] == code]
    sub_pre = sub[sub["date"] < "2020-01-01"]
    sns.boxplot(data=sub_pre, x="month", y="pax", ax=ax, color=COLORS[0], fliersize=2)
    ax.set_title(SHORT[code])
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
fig.suptitle("Seasonality by Month (pre-COVID)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "02_seasonality_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  02 — Seasonality boxplots")

# ──────────────────────────────────────────────────────────
# 3. COVID impact — normalized to Jan 2020
# ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
covid_window = feat[(feat["date"] >= "2019-01-01") & (feat["date"] <= "2025-12-01")]
for i, code in enumerate(CORE_AIRPORTS):
    sub = covid_window[covid_window["airport"] == code].copy()
    jan2020 = sub.loc[sub["date"] == "2020-01-01", "pax"]
    if len(jan2020) == 0:
        continue
    baseline = jan2020.values[0]
    sub["pax_norm"] = sub["pax"] / baseline * 100
    ax.plot(sub["date"], sub["pax_norm"], label=SHORT[code], color=COLORS[i], linewidth=1.5)
ax.axhline(100, color="gray", linestyle="--", alpha=0.5, label="Jan 2020 baseline")
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-06-01"),
           alpha=0.12, color="red")
ax.set_ylabel("PAX (% of Jan 2020)")
ax.set_title("COVID Impact & Recovery — Normalized to Jan 2020")
ax.legend(loc="lower right")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig(FIGS / "03_covid_impact_recovery.png", dpi=150)
plt.close()
print("  03 — COVID impact & recovery")

# ──────────────────────────────────────────────────────────
# 4. Trend — annual PAX (core airports)
# ──────────────────────────────────────────────────────────
annual = core.groupby(["airport", "year"])["pax"].sum().reset_index()
annual["name"] = annual["airport"].map(SHORT)
fig, ax = plt.subplots(figsize=(14, 7))
for i, code in enumerate(CORE_AIRPORTS):
    sub = annual[annual["airport"] == code]
    ax.plot(sub["year"], sub["pax"] / 1e6, marker="o", markersize=4,
            label=SHORT[code], color=COLORS[i], linewidth=1.5)
ax.set_ylabel("Annual PAX (millions)")
ax.set_title("Annual Passenger Trend — VINCI Airports")
ax.legend()
ax.set_xlabel("Year")
fig.tight_layout()
fig.savefig(FIGS / "04_annual_trend.png", dpi=150)
plt.close()
print("  04 — Annual trend")

# ──────────────────────────────────────────────────────────
# 5. Correlation heatmap between airports (monthly PAX)
# ──────────────────────────────────────────────────────────
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
print("  05 — Correlation heatmap")

# ──────────────────────────────────────────────────────────
# 6. Year-over-Year growth distribution
# ──────────────────────────────────────────────────────────
yoy = core.dropna(subset=["pax_yoy_growth"])
yoy = yoy[(yoy["pax_yoy_growth"] > -1) & (yoy["pax_yoy_growth"] < 3)]
fig, ax = plt.subplots(figsize=(12, 6))
for i, code in enumerate(CORE_AIRPORTS):
    sub = yoy[yoy["airport"] == code]
    ax.hist(sub["pax_yoy_growth"] * 100, bins=30, alpha=0.4,
            label=SHORT[code], color=COLORS[i])
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.set_xlabel("Year-over-Year Growth (%)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Monthly YoY PAX Growth")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "06_yoy_growth_distribution.png", dpi=150)
plt.close()
print("  06 — YoY growth distribution")

# ──────────────────────────────────────────────────────────
# 7. Decomposition: one airport example (Lyon)
# ──────────────────────────────────────────────────────────
from statsmodels.tsa.seasonal import seasonal_decompose

lyon = feat[(feat["airport"] == "FR_LFLL") & (feat["date"] < "2020-01-01")].set_index("date")["pax"]
decomp = seasonal_decompose(lyon, model="multiplicative", period=12)
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
decomp.observed.plot(ax=axes[0], title="Observed")
decomp.trend.plot(ax=axes[1], title="Trend")
decomp.seasonal.plot(ax=axes[2], title="Seasonal")
decomp.resid.plot(ax=axes[3], title="Residual")
for ax in axes:
    ax.set_xlabel("")
fig.suptitle("Multiplicative Decomposition — Lyon (pre-COVID)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "07_decomposition_lyon.png", dpi=150, bbox_inches="tight")
plt.close()
print("  07 — Seasonal decomposition (Lyon)")

# ──────────────────────────────────────────────────────────
# 8. Stationarity test (ADF) per airport
# ──────────────────────────────────────────────────────────
from statsmodels.tsa.stattools import adfuller

print("\n=== Stationarity (ADF test, pre-COVID) ===")
print(f"{'Airport':<12} {'ADF stat':>10} {'p-value':>10} {'Stationary?':>12}")
print("-" * 48)
for code in CORE_AIRPORTS:
    sub = feat[(feat["airport"] == code) & (feat["date"] < "2020-01-01")]["pax"].dropna()
    if len(sub) < 24:
        continue
    result = adfuller(sub, autolag="AIC")
    stationary = "YES" if result[1] < 0.05 else "NO"
    print(f"{SHORT[code]:<12} {result[0]:>10.3f} {result[1]:>10.4f} {stationary:>12}")

# ──────────────────────────────────────────────────────────
# 9. Holiday effect visualization
# ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# School vacation effect
for i, code in enumerate(CORE_AIRPORTS):
    sub = core[(core["airport"] == code) & (core["date"] < "2020-01-01")]
    vac = sub[sub["is_school_vacation"] == 1]["pax"].mean()
    novac = sub[sub["is_school_vacation"] == 0]["pax"].mean()
    axes[0].bar(i, (vac / novac - 1) * 100, color=COLORS[i])
axes[0].set_xticks(range(len(CORE_AIRPORTS)))
axes[0].set_xticklabels([SHORT[c] for c in CORE_AIRPORTS], rotation=45)
axes[0].set_ylabel("PAX uplift (%)")
axes[0].set_title("School Vacation Effect on PAX")
axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)

# Holiday count correlation with PAX
for i, code in enumerate(CORE_AIRPORTS):
    sub = core[(core["airport"] == code) & (core["date"] < "2020-01-01")]
    r, p = stats.pearsonr(sub["n_holidays"], sub["pax"])
    axes[1].bar(i, r, color=COLORS[i])
axes[1].set_xticks(range(len(CORE_AIRPORTS)))
axes[1].set_xticklabels([SHORT[c] for c in CORE_AIRPORTS], rotation=45)
axes[1].set_ylabel("Pearson r")
axes[1].set_title("Correlation: N Holidays vs PAX")
axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(FIGS / "08_holiday_effect.png", dpi=150)
plt.close()
print("\n  08 — Holiday effect")

# ──────────────────────────────────────────────────────────
# 10. Summary stats table
# ──────────────────────────────────────────────────────────
print("\n=== Summary Stats (core airports, full period) ===")
summary = core.groupby("airport").agg(
    months=("pax", "count"),
    mean_pax=("pax", "mean"),
    std_pax=("pax", "std"),
    min_pax=("pax", "min"),
    max_pax=("pax", "max"),
    cv=("pax", lambda x: x.std() / x.mean()),
).reset_index()
summary["airport"] = summary["airport"].map(SHORT)
print(summary.to_string(index=False))

print(f"\nAll plots saved to: {FIGS}")
print("Done.")
