"""
HAR-RV Baseline for EquityBERT thesis.

Implements Corsi (2009) HAR-RV adapted to hourly ES futures data:
  r_{t+1} = c + beta_d * mean(r, 24h) + beta_w * mean(r, 120h) + beta_m * mean(r, 528h) + eps

Key design: HAR features are computed on the FULL dataset first, then the
SAME date-based split boundaries as the LSTM/EquityBERT are applied:
  Train: 2019-01-03 → 2024-01-26
  Val:   2024-01-26 → 2025-02-27
  Test:  2025-02-27 → 2026-03-31

This ensures all models are evaluated on identical test observations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#  CONFIG 
DATA_PATH = "/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/vola-bert/data/processed/ES_1h.parquet"
OUT_DIR   = "runs/har_baseline"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [
    {"lookback": 24, "forecast": 5},
    {"lookback": 50, "forecast": 10},
]

# HAR window sizes at hourly frequency
HAR_DAILY   = 24        # 1 day   = 24h
HAR_WEEKLY  = 24 * 5    # 5 days  = 120h
HAR_MONTHLY = 24 * 22   # 22 days = 528h

# These are the EXACT same split dates produced by Dataset_SP500_1H (70/15/15)
# Confirmed from: python silinecek.py output
VAL_START_DATE  = "2024-01-26"
TEST_START_DATE = "2025-02-27"


# ─ LOAD & FEATURE ENGINEERING 
def load_and_build_features():
    df = pd.read_parquet(DATA_PATH)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df = df.sort_values("Datetime").reset_index(drop=True)
    df = df.dropna(subset=["High", "Low"])

    # Target: log high-low range (same as LSTM/EquityBERT)
    df["r"] = np.log(df["High"] / df["Low"])

    # Compute HAR features on the FULL dataset
    # Rolling means carry across all rows — no per-week warmup dropout
    r = df["r"].values
    N = len(r)

    rv_d = np.full(N, np.nan)
    rv_w = np.full(N, np.nan)
    rv_m = np.full(N, np.nan)

    for t in range(HAR_MONTHLY, N):
        rv_d[t] = r[t - HAR_DAILY  : t].mean()
        rv_w[t] = r[t - HAR_WEEKLY : t].mean()
        rv_m[t] = r[t - HAR_MONTHLY: t].mean()

    df["rv_d"] = rv_d
    df["rv_w"] = rv_w
    df["rv_m"] = rv_m

    # Drop only the initial warmup NaNs at the very start
    df = df.dropna(subset=["rv_d", "rv_w", "rv_m"]).reset_index(drop=True)

    print(f"  Total bars after warmup: {len(df):,}")
    print(f"  Full range: {df['Datetime'].iloc[0].date()} → {df['Datetime'].iloc[-1].date()}")

    return df


#  DATE-BASED SPLIT (matches LSTM/EquityBERT exactly) 
def make_splits(df):
    """
    Split by the same date boundaries as Dataset_SP500_1H.
    Any HAR warmup rows before the LSTM train start are simply not used for training
    but their feature values are available as lookback for the training windows.
    """
    val_dt  = pd.Timestamp(VAL_START_DATE,  tz="UTC")
    test_dt = pd.Timestamp(TEST_START_DATE, tz="UTC")

    train = df[df["Datetime"] <  val_dt ].reset_index(drop=True)
    val   = df[(df["Datetime"] >= val_dt) & (df["Datetime"] < test_dt)].reset_index(drop=True)
    test  = df[df["Datetime"] >= test_dt].reset_index(drop=True)

    print(f"\n  Train: {len(train):,} bars  "
          f"({train['Datetime'].iloc[0].date()} → {train['Datetime'].iloc[-1].date()})")
    print(f"  Val:   {len(val):,} bars  "
          f"({val['Datetime'].iloc[0].date()} → {val['Datetime'].iloc[-1].date()})")
    print(f"  Test:  {len(test):,} bars  "
          f"({test['Datetime'].iloc[0].date()} → {test['Datetime'].iloc[-1].date()})")

    return train, val, test


#  FIT 
def fit_har(train_df):
    X = train_df[["rv_d", "rv_w", "rv_m"]].values
    y = train_df["r"].values
    model = LinearRegression().fit(X, y)
    print(f"\n  HAR-RV OLS coefficients (fit on training split only):")
    print(f"    intercept = {model.intercept_:.6f}")
    print(f"    beta_d    = {model.coef_[0]:.6f}  (daily,   24h mean)")
    print(f"    beta_w    = {model.coef_[1]:.6f}  (weekly,  120h mean)")
    print(f"    beta_m    = {model.coef_[2]:.6f}  (monthly, 528h mean)")
    return model


#  PREDICT 
def predict_har(model, split_df, lookback, forecast):
    """
    Sliding window over the split.
    At each window: use the HAR features of the LAST bar in the lookback
    as the forecast for ALL forecast steps (HAR is a 1-step model applied multi-step).
    Clip predictions to zero since volatility cannot be negative.
    """
    rows = []
    n_windows = len(split_df) - lookback - forecast + 1

    if n_windows <= 0:
        print(f"    Warning: not enough rows for lookback={lookback} forecast={forecast}")
        return pd.DataFrame()

    for i in range(n_windows):
        last_input_idx = i + lookback - 1
        row_input = split_df.iloc[last_input_idx]
        X_in = np.array([[row_input["rv_d"], row_input["rv_w"], row_input["rv_m"]]])
        y_hat = float(model.predict(X_in)[0])
        y_hat = max(y_hat, 0.0)

        for step in range(forecast):
            target_idx = i + lookback + step
            actual_r   = split_df.iloc[target_idx]["r"]
            dt         = split_df.iloc[target_idx]["Datetime"]
            rows.append({
                "Datetime":     dt,
                "Horizon_step": step + 1,
                "Actual_r":     actual_r,
                "Predicted_r":  y_hat,
            })

    df_out = pd.DataFrame(rows)
    df_out["AE"] = np.abs(df_out["Actual_r"] - df_out["Predicted_r"])
    df_out["SE"] = (df_out["Actual_r"] - df_out["Predicted_r"]) ** 2
    return df_out


def naive_mae_calc(df_pred):
    actual = df_pred["Actual_r"].values
    naive  = df_pred["Actual_r"].shift(1).bfill().values
    return np.mean(np.abs(actual - naive))


#  MAIN 
def main():
    print("=" * 60)
    print("HAR-RV Baseline Evaluation")
    print("=" * 60)

    print("\nLoading data and computing HAR features...")
    df = load_and_build_features()

    print("\nSplitting by date (same boundaries as LSTM/EquityBERT)...")
    train_df, val_df, test_df = make_splits(df)

    # Fit HAR-RV on training data only
    model = fit_har(train_df)

    all_results = {}

    for hz in HORIZONS:
        lookback = hz["lookback"]
        forecast = hz["forecast"]
        tag      = f"{lookback}to{forecast}"

        print(f"\n{'='*60}")
        print(f"  Horizon: {lookback}h → {forecast}h")
        print(f"{'='*60}")

        split_dfs = {}
        for split_name, split_data in [
            ("train", train_df),
            ("val",   val_df),
            ("test",  test_df),
        ]:
            df_pred = predict_har(model, split_data, lookback, forecast)
            if df_pred.empty:
                continue
            split_dfs[split_name] = df_pred

            mae   = df_pred["AE"].mean()
            n_mae = naive_mae_calc(df_pred)
            rmae  = mae / n_mae
            n_windows = len(df_pred) // forecast

            print(f"  {split_name.upper():5s}  "
                  f"windows={n_windows:,}  "
                  f"pred_pairs={len(df_pred):,}  "
                  f"MAE={mae:.6f}  NaiveMAE={n_mae:.6f}  rMAE={rmae:.3f}")

        all_results[tag] = split_dfs

        # Save Excel
        xlsx_path = os.path.join(OUT_DIR, f"har_predictions_{tag}.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for split, df_p in split_dfs.items():
                df_out = df_p.copy()
                try:
                    df_out["Datetime"] = df_out["Datetime"].dt.tz_localize(None)
                except Exception:
                    df_out["Datetime"] = df_out["Datetime"].dt.tz_convert(None)
                df_out.to_excel(writer, sheet_name=split.capitalize(), index=False)
        print(f"  ✓ Excel → {os.path.basename(xlsx_path)}")

        # Time-series plot
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
        for ax, (split, df_p) in zip(axes, split_dfs.items()):
            df1 = df_p[df_p["Horizon_step"] == 1].sort_values("Datetime")
            mae1 = df1["AE"].mean()
            ax.plot(df1["Datetime"], df1["Actual_r"],
                    label="Actual r", linewidth=0.7, alpha=0.8, color="steelblue")
            ax.plot(df1["Datetime"], df1["Predicted_r"],
                    label="HAR-RV", linewidth=0.7, alpha=0.8, color="green")
            ax.set_title(f"{split.upper()} — 1-step-ahead (MAE={mae1:.5f})")
            ax.set_ylabel("log(H/L)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle(f"HAR-RV Baseline  {lookback}h→{forecast}h  |  Actual vs Predicted",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig_path = os.path.join(OUT_DIR, f"har_vol_forecast_{tag}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  ✓ Plot → {os.path.basename(fig_path)}")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Horizon':<12} {'Split':<8} {'MAE':>10} {'NaiveMAE':>10} {'rMAE':>8}")
    print("-" * 52)
    for tag, sdfs in all_results.items():
        for split, df_p in sdfs.items():
            mae   = df_p["AE"].mean()
            n_mae = naive_mae_calc(df_p)
            rmae  = mae / n_mae
            print(f"{tag:<12} {split:<8} {mae:>10.6f} {n_mae:>10.6f} {rmae:>8.3f}")

    print(f"\nAll results saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()