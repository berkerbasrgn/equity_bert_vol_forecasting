"""
Exploratory Data Analysis — r_t (Log-Range Volatility)

r_t = ln(H_t / L_t), the Parkinson (1980) log-range estimator applied to
hourly E-mini S&P 500 (ES) futures bars from Databento, 2019–2026.

Figures produced
----------------
1. eda_timeseries.png   — Full 7-year hourly time series of r_t with a
                          rolling 24 h average overlay.
2. eda_histogram.png    — Histogram + KDE of r_t with right-skew annotations.
3. eda_stats_table.png  — Summary-statistics table rendered as a figure.
4. eda_intraday.png     — Mean r_t ± 1 std by hour-of-day, session-coloured.

All figures are also saved as PDF for LaTeX inclusion.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy import stats
from statsmodels.tsa.stattools import acf
import os

DATA_PATH  = "data/processed/ES_1h.parquet"
OUT_DIR    = "eda_figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})

SESSION_COLORS = {
    "overnight":   "#4C72B0",
    "pre_market":  "#DD8452",
    "regular":     "#55A868",
    "after_hours": "#C44E52",
}
SESSION_LABELS = {
    "overnight":   "Overnight  (20:00–03:59 ET)",
    "pre_market":  "Pre-market (04:00–09:29 ET)",
    "regular":     "Regular    (09:30–15:59 ET)",
    "after_hours": "After-hours(16:00–19:59 ET)",
}

# Load & compute r_t
print("Loading data …")
df = pd.read_parquet(DATA_PATH)
df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_convert("America/New_York")
df = df.sort_values("Datetime").reset_index(drop=True)

df["r_t"] = np.log(df["High"] / df["Low"])
df = df.dropna(subset=["r_t"])
df = df[df["r_t"] > 0]          # drop degenerate flat bars (H == L)

print(f"  Rows: {len(df):,}  |  {df['Datetime'].iloc[0].date()} → {df['Datetime'].iloc[-1].date()}")

# Session label (vectorised)
h = df["Datetime"].dt.hour
m = df["Datetime"].dt.minute
mins = h * 60 + m
dow  = df["Datetime"].dt.dayofweek  # 0=Mon … 6=Sun

conditions = [
    (dow == 6),
    (mins >= 240) & (mins < 570),    # 04:00–09:29
    (mins >= 570) & (mins < 960),    # 09:30–15:59
    (mins >= 960) & (mins < 1200),   # 16:00–19:59
]
choices = ["overnight", "pre_market", "regular", "after_hours"]
df["session"] = np.select(conditions, choices, default="overnight")
df["hour"]    = df["Datetime"].dt.hour

r = df["r_t"].values
dates = df["Datetime"]

# Time-series plot
print("Plotting time series …")

daily = df.groupby(df["Datetime"].dt.date)["r_t"].mean().reset_index()
daily.columns = ["date", "r_t_mean"]
daily["date"] = pd.to_datetime(daily["date"])

# Annotate major volatility events
events = {
    "COVID crash\n(Mar 2020)":   "2020-03-16",
    "Fed hike cycle\n(Mar 2022)":"2022-03-16",
    "SVB collapse\n(Mar 2023)":  "2023-03-13",
    "Aug 2024\nunkwind":         "2024-08-05",
}

fig, axes = plt.subplots(2, 1, figsize=(13, 6.5),
                         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

ax1, ax2 = axes

# Upper panel: hourly r_t (thin) + daily rolling mean (thick)
ax1.plot(dates, r, color="#9EB9D4", linewidth=0.25, alpha=0.6, label="Hourly $r_t$")
ax1.plot(daily["date"], daily["r_t_mean"], color="#1B4F8A", linewidth=1.1,
         label="Daily mean $r_t$")

ymax = np.percentile(r, 99.5)
for label, d in events.items():
    ts = pd.Timestamp(d, tz="America/New_York")
    ax1.axvline(ts, color="#B03A2E", linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.text(ts, ymax * 0.97, label, fontsize=7.5, color="#B03A2E",
             ha="center", va="top", rotation=0,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

ax1.set_ylim(0, ymax * 1.08)
ax1.set_ylabel("$r_t = \\ln(H_t / L_t)$")
ax1.set_title("E-mini S&P 500 (ES) Hourly Log-Range Volatility  $r_t$  —  2019–2026",
              fontweight="bold")
ax1.legend(loc="upper left", framealpha=0.8)
ax1.set_xticklabels([])

# Lower panel: 30-day rolling std (regime indicator)
roll_std = (pd.Series(r, index=dates)
            .rolling("30D")
            .std())
ax2.fill_between(roll_std.index, roll_std.values, alpha=0.55, color="#5B8DB8")
ax2.set_ylabel("30-day\nrolling std", fontsize=8)
ax2.set_xlabel("Date")
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

# Train/val/test boundaries (70/15/15 split)
n = len(df)
idx_val  = int(n * 0.70)
idx_test = int(n * 0.85)
for idx, lbl, col in [(idx_val, "Val", "#E59866"), (idx_test, "Test", "#A93226")]:
    ts = df["Datetime"].iloc[idx]
    for ax in axes:
        ax.axvline(ts, color=col, linewidth=1.0, linestyle=":", alpha=0.9)
    ax1.text(ts, ymax * 0.87, f" {lbl}", fontsize=7.5, color=col, va="top")

fig.savefig(f"{OUT_DIR}/eda_timeseries.png", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/eda_timeseries.pdf", bbox_inches="tight")
plt.close(fig)
print(f"  → {OUT_DIR}/eda_timeseries.png")

#  Histogram + KDE
print("Plotting histogram …")

skew_val = float(pd.Series(r).skew())
kurt_val = float(pd.Series(r).kurtosis())   # excess kurtosis

fig, ax = plt.subplots(figsize=(7, 4.5))

n_bins = 120
counts, bins, patches = ax.hist(r, bins=n_bins, density=True,
                                color="#5B8DB8", alpha=0.65, edgecolor="white",
                                linewidth=0.3, label="Hourly $r_t$")

# KDE
from scipy.stats import gaussian_kde
kde = gaussian_kde(r, bw_method="scott")
xgrid = np.linspace(0, np.percentile(r, 99.8), 400)
ax.plot(xgrid, kde(xgrid), color="#1B4F8A", linewidth=1.8, label="KDE")

# Normal with same mean/std for reference
mu, sigma = r.mean(), r.std()
ax.plot(xgrid, stats.norm.pdf(xgrid, mu, sigma),
        color="#C0392B", linewidth=1.4, linestyle="--",
        label=f"Normal($\\mu$={mu:.4f}, $\\sigma$={sigma:.4f})")

# Annotations
ax.set_xlim(left=0)
yref = ax.get_ylim()[1]
ax.text(0.97, 0.95,
        f"Skewness  = {skew_val:+.3f}\nExc. kurtosis = {kurt_val:+.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="#AAAAAA"))

ax.set_xlabel("$r_t = \\ln(H_t / L_t)$")
ax.set_ylabel("Density")
ax.set_title("Distribution of Hourly Log-Range Volatility  $r_t$", fontweight="bold")
ax.legend(framealpha=0.8)

fig.savefig(f"{OUT_DIR}/eda_histogram.png", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/eda_histogram.pdf", bbox_inches="tight")
plt.close(fig)
print(f"  → {OUT_DIR}/eda_histogram.png")

# Summary statistics table
print("Building summary statistics …")

acf_vals = acf(r, nlags=24, fft=True)

rows = [
    ("Observations",           f"{len(r):,}"),
    ("Date range",             f"{df['Datetime'].iloc[0].date()}  →  {df['Datetime'].iloc[-1].date()}"),
    ("Mean",                   f"{np.mean(r):.6f}"),
    ("Median",                 f"{np.median(r):.6f}"),
    ("Std dev",                f"{np.std(r, ddof=1):.6f}"),
    ("Min",                    f"{np.min(r):.6f}"),
    ("Max",                    f"{np.max(r):.6f}"),
    ("5th percentile",         f"{np.percentile(r, 5):.6f}"),
    ("95th percentile",        f"{np.percentile(r, 95):.6f}"),
    ("Skewness",               f"{skew_val:+.4f}"),
    ("Excess kurtosis",        f"{kurt_val:+.4f}"),
    ("Autocorrelation lag 1",  f"{acf_vals[1]:.4f}"),
    ("Autocorrelation lag 5",  f"{acf_vals[5]:.4f}"),
    ("Autocorrelation lag 24", f"{acf_vals[24]:.4f}"),
]

# Print to console
print("\n  Summary Statistics for r_t = ln(H/L)")
print(f"  {'Statistic':<28} {'Value':>18}")
print("  " + "-" * 48)
for stat, val in rows:
    print(f"  {stat:<28} {val:>18}")

# Render as figure
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.axis("off")

col_labels  = ["Statistic", "Value"]
cell_text   = [[s, v] for s, v in rows]
col_widths  = [0.60, 0.40]

tbl = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    colWidths=col_widths,
    loc="center",
    cellLoc="left",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.45)

# Style header
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor("#1B4F8A")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

# Alternating row shading
for i in range(1, len(rows) + 1):
    fc = "#EBF2FA" if i % 2 == 0 else "white"
    for j in range(len(col_labels)):
        tbl[(i, j)].set_facecolor(fc)

ax.set_title("Summary Statistics  —  $r_t = \\ln(H_t / L_t)$, ES Futures 1 h",
             fontweight="bold", pad=14)

fig.savefig(f"{OUT_DIR}/eda_stats_table.png", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/eda_stats_table.pdf", bbox_inches="tight")
plt.close(fig)
print(f"\n  → {OUT_DIR}/eda_stats_table.png")

# Also save as CSV
stats_df = pd.DataFrame(rows, columns=["Statistic", "Value"])
stats_df.to_csv(f"{OUT_DIR}/eda_stats_table.csv", index=False)

# Intraday seasonality
print("Plotting intraday seasonality …")

hourly_stats = (df.groupby("hour")["r_t"]
                .agg(mean="mean", std="std", median="median", count="count")
                .reset_index())

# Session for each hour-of-day (reference: weekday, non-Sunday)
def hour_to_session(h):
    mins = h * 60
    if 240 <= mins < 570:
        return "pre_market"
    elif 570 <= mins < 960:
        return "regular"
    elif 960 <= mins < 1200:
        return "after_hours"
    else:
        return "overnight"

hourly_stats["session"] = hourly_stats["hour"].apply(hour_to_session)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
ax_mean, ax_med = axes

for ax, col, title in [
    (ax_mean, "mean",   "Mean $r_t$ by Hour-of-Day (ET)"),
    (ax_med,  "median", "Median $r_t$ by Hour-of-Day (ET)"),
]:
    for sess, grp in hourly_stats.groupby("session"):
        color = SESSION_COLORS[sess]
        # Bar chart segment
        ax.bar(grp["hour"], grp[col],
               color=color, alpha=0.75, width=0.85, zorder=2)
        # Error ribbon (mean ± std only on left panel)
        if ax is ax_mean:
            ax.errorbar(grp["hour"], grp["mean"],
                        yerr=grp["std"] / np.sqrt(grp["count"]),
                        fmt="none", ecolor=color, elinewidth=1.0,
                        capsize=2.5, zorder=3)

    ax.set_xlabel("Hour of day (ET)")
    ax.set_ylabel("$r_t$")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.6, 23.6)

# Shade session regions (upper panel only, using ax_mean)
session_spans = [
    ("overnight",   [(0, 3), (20, 23)]),
    ("pre_market",  [(4, 9)]),
    ("regular",     [(9, 15)]),
    ("after_hours", [(16, 19)]),
]
for sess, spans in session_spans:
    col = SESSION_COLORS[sess]
    for x0, x1 in spans:
        ax_mean.axvspan(x0 - 0.5, x1 + 0.5, alpha=0.08, color=col, zorder=0)
        ax_med.axvspan(x0 - 0.5,  x1 + 0.5, alpha=0.08, color=col, zorder=0)

# added 9:30 open and 16:00 close markers
for ax in axes:
    ax.axvline(9.5,  color="#55A868", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axvline(16.0, color="#C44E52", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.text(9.5,  ax.get_ylim()[1] * 0.97, " Open",  fontsize=7.5,
            color="#55A868", va="top")
    ax.text(16.0, ax.get_ylim()[1] * 0.97, " Close", fontsize=7.5,
            color="#C44E52", va="top")

handles = [mpatches.Patch(color=SESSION_COLORS[s], alpha=0.75,
                           label=SESSION_LABELS[s])
           for s in SESSION_LABELS]
ax_mean.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85)

fig.suptitle("Intraday Seasonality of Log-Range Volatility  $r_t$  —  ES Futures 1 h",
             fontweight="bold", fontsize=11, y=1.01)
fig.tight_layout()

fig.savefig(f"{OUT_DIR}/eda_intraday.png", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/eda_intraday.pdf", bbox_inches="tight")
plt.close(fig)
print(f"  → {OUT_DIR}/eda_intraday.png")


# ---------------------------------------------------------------------------
# 5. ACF / PACF
# ---------------------------------------------------------------------------
print("Plotting ACF / PACF ...")

from statsmodels.tsa.stattools import pacf as pacf_fn

MAX_LAG = 48
CI_BOUND = 1.96 / np.sqrt(len(r))

acf_full  = acf(r,  nlags=MAX_LAG, fft=True)
pacf_full = pacf_fn(r, nlags=MAX_LAG, method="ywm")

fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(13, 4.2))

for ax, vals, title in [
    (ax_acf,  acf_full[1:],  "Autocorrelation Function (ACF)"),
    (ax_pacf, pacf_full[1:], "Partial ACF (PACF)"),
]:
    lags = np.arange(1, MAX_LAG + 1)
    ax.bar(lags, vals, color=np.where(np.abs(vals) > CI_BOUND, "#1B4F8A", "#9EB9D4"),
           width=0.7, zorder=2)
    ax.axhline(0,          color="black", linewidth=0.6)
    ax.axhline( CI_BOUND,  color="#C0392B", linewidth=0.9, linestyle="--",
                label="95% CI")
    ax.axhline(-CI_BOUND,  color="#C0392B", linewidth=0.9, linestyle="--")

    for lag_mark, col in [(1, "#55A868"), (5, "#DD8452"), (24, "#9B59B6")]:
        v = vals[lag_mark - 1]
        ax.axvline(lag_mark, color=col, linewidth=0.9, linestyle=":", alpha=0.8,
                   label=f"Lag {lag_mark} ({v:+.3f})")

    ax.set_xlim(0, MAX_LAG + 1)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Correlation")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.85)

fig.suptitle("Autocorrelation Structure of $r_t$  ---  ES Futures 1 h  (lags 1-48)",
             fontweight="bold", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/eda_acf_pacf.png", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/eda_acf_pacf.pdf", bbox_inches="tight")
plt.close(fig)
print(f"  -> {OUT_DIR}/eda_acf_pacf.png")

# ---------------------------------------------------------------------------
# 6. Event-window volatility response
# ---------------------------------------------------------------------------
print("Plotting event-window analysis ...")

EVENTS_PATH = "data/NEW_macro_events.csv"
EVENT_COLORS = {"CPI": "#DD8452", "PPI": "#4C72B0",
                "NFP": "#55A868", "FOMC": "#C44E52"}
WINDOW = 12  # hours either side of each release

events_raw = pd.read_csv(EVENTS_PATH)
events_raw["datetime"] = (pd.to_datetime(events_raw["datetime"], utc=True)
                          .dt.tz_convert("America/New_York"))

df_rt = df.set_index("Datetime")["r_t"]

rows_ev = []
for _, ev in events_raw.iterrows():
    et, etype = ev["datetime"], ev["event_type"]
    pos = df_rt.index.searchsorted(et)
    for lag in range(-WINDOW, WINDOW + 1):
        idx = pos + lag
        if 0 <= idx < len(df_rt):
            rows_ev.append({"event_type": etype, "lag_h": lag,
                            "r_t": df_rt.iloc[idx]})

ev_df = pd.DataFrame(rows_ev)
ev_agg = (ev_df.groupby(["event_type", "lag_h"])["r_t"]
          .agg(mean="mean", se=lambda x: x.std(ddof=1) / np.sqrt(len(x)))
          .reset_index())

mu_all = df["r_t"].mean()

fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey=True)
axes = axes.flatten()

for ax, etype in zip(axes, ["CPI", "PPI", "NFP", "FOMC"]):
    sub = ev_agg[ev_agg["event_type"] == etype].sort_values("lag_h")
    col = EVENT_COLORS[etype]
    n_ev = int((ev_df["event_type"] == etype).sum() / (2 * WINDOW + 1))

    ax.fill_between(sub["lag_h"], sub["mean"] - sub["se"],
                    sub["mean"] + sub["se"], alpha=0.25, color=col)
    ax.plot(sub["lag_h"], sub["mean"], color=col, linewidth=1.8,
            label=f"{etype} (n={n_ev})")
    ax.axhline(mu_all, color="grey", linewidth=0.9, linestyle="--",
               label=f"Unconditional mean ({mu_all:.4f})")
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    yhi = sub["mean"].max() * 1.05
    ax.text(0.3, yhi, "Release", fontsize=7.5, va="top", color="black")

    ax.set_xlim(-WINDOW, WINDOW)
    ax.set_xlabel("Hours relative to release")
    ax.set_ylabel("Mean $r_t$")
    ax.set_title(f"{etype} Release Window", fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.85)

fig.suptitle("Volatility Response Around Macro Releases  (+-12 h)  ---  ES Futures 1 h",
             fontweight="bold", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/eda_event_window.png", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/eda_event_window.pdf", bbox_inches="tight")
plt.close(fig)
print(f"  -> {OUT_DIR}/eda_event_window.png")

print(f"\nAll EDA figures written to '{OUT_DIR}/'")
