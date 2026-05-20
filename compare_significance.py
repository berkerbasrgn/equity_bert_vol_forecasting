"""
Statistical significance comparison: EquityBERT vs LSTM Baseline

For each configured EquityBERT variant and horizon:
  1. Loads LSTM test predictions from the xlsx Test sheet (original scale).
  2. Runs EquityBERT inference on the test set and inverse-transforms predictions
     to the same original log-range scale.
  3. Aligns both on (Datetime, Horizon_step) via an inner join.
  4. Applies:
       - Diebold-Mariano (DM) test — proper for nested/non-nested forecast comparison,
         with Newey-West long-run variance using pred_len - 1 autocorrelation lags.
       - Paired two-sided t-test on absolute / squared errors.
  5. Saves a summary TXT and CSV to runs/significance_tests/.

Usage:
    cd /path/to/vola-bert
    source venv/bin/activate
    python compare_significance.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy import stats

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "src"))

from src.mydataset import Dataset_SP500_1H, SEMANTIC_TOKEN_VOCAB
from src.model_bert import EquityBERT

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH         = os.path.join(current_dir, "data", "processed", "ES_1h.parquet")
EVENTS_CSV        = os.path.join(current_dir, "data", "NEW_macro_events.csv")
EQUITYBERT_DIR    = os.path.join(current_dir, "runs", "equitybert", "v27")
LSTM_DIR          = os.path.join(current_dir, "runs", "lstm_baseline", "v2")
OUTPUT_DIR        = os.path.join(current_dir, "runs", "significance_tests")

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256

# ── Experiment matrix ──────────────────────────────────────────────────────────
CONFIGS = [
    {
        "folder":           "No Events_24to5_full",
        "label":            "24→5 | No Events",
        "horizon_tag":      "24to5",
        "seq_len":          24,
        "pred_len":         5,
        "use_events":       False,
        "use_event_type":   False,
        "use_event_impact": False,
    },
    {
        "folder":           "Event Type Only_24to5_full",
        "label":            "24→5 | Event Type Only",
        "horizon_tag":      "24to5",
        "seq_len":          24,
        "pred_len":         5,
        "use_events":       True,
        "use_event_type":   True,
        "use_event_impact": False,
    },
    {
        "folder":           "Event Timing Only_24to5_full",
        "label":            "24→5 | Event Timing Only",
        "horizon_tag":      "24to5",
        "seq_len":          24,
        "pred_len":         5,
        "use_events":       True,
        "use_event_type":   False,
        "use_event_impact": False,
    },
    {
        "folder":           "No Events_50to10_full",
        "label":            "50→10 | No Events",
        "horizon_tag":      "50to10",
        "seq_len":          50,
        "pred_len":         10,
        "use_events":       False,
        "use_event_type":   False,
        "use_event_impact": False,
    },
    {
        "folder":           "Event Type Only_50to10_full",
        "label":            "50→10 | Event Type Only",
        "horizon_tag":      "50to10",
        "seq_len":          50,
        "pred_len":         10,
        "use_events":       True,
        "use_event_type":   True,
        "use_event_impact": False,
    },
    {
        "folder":           "Event Timing Only_50to10_full",
        "label":            "50→10 | Event Timing Only",
        "horizon_tag":      "50to10",
        "seq_len":          50,
        "pred_len":         10,
        "use_events":       True,
        "use_event_type":   False,
        "use_event_impact": False,
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_events_df():
    df = pd.read_csv(EVENTS_CSV)
    # Column is already named 'datetime' in data/NEW_macro_events.csv
    if "event_time_et" in df.columns:
        df = df.rename(columns={"event_time_et": "datetime"})
    df["impact"] = "NONE"
    df["datetime"] = (
        pd.to_datetime(df["datetime"], utc=True)
        .dt.tz_convert("America/New_York")
    )
    return df


def get_r_scaler_params(dataset):
    """Return (mean, std) of the target column from the fitted StandardScaler."""
    scaler = dataset.scaler
    if hasattr(scaler, "mean") and hasattr(scaler, "std"):
        return float(scaler.mean[-1]), float(scaler.std[-1])
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        return float(scaler.mean_[-1]), float(scaler.scale_[-1])
    if hasattr(scaler, "_mean") and hasattr(scaler, "_std"):
        return float(scaler._mean[-1]), float(scaler._std[-1])
    raise AttributeError(
        "Cannot find mean/std attributes on the scaler. "
        "Check src/utils.py and update this function."
    )


def run_equitybert_inference(cfg, events_df):
    """
    Run the trained EquityBERT on the test split and return a DataFrame
    with columns: Datetime (tz-naive), Horizon_step, Actual_r, Predicted_r, AE, SE.
    Both Actual_r and Predicted_r are on the original log-range scale
    (inverse-StandardScaler applied; RevIN is already undone inside the model).
    """
    seq_len  = cfg["seq_len"]
    pred_len = cfg["pred_len"]

    common_kwargs = dict(
        data_path        = DATA_PATH,
        events_df        = events_df if cfg["use_events"] else None,
        size             = (seq_len, pred_len),
        use_events       = cfg["use_events"],
        use_event_type   = cfg["use_event_type"],
        use_event_impact = cfg["use_event_impact"],
        use_explainable  = True,
        mode             = "24h",
    )
    test_ds = Dataset_SP500_1H(flag="test", **common_kwargs)
    print(f"    EquityBERT test period : {test_ds.start_date} → {test_ds.end_date}")
    print(f"    EquityBERT test windows: {len(test_ds):,}")

    r_mean, r_std = get_r_scaler_params(test_ds)

    # Build model (architecture must match training)
    sample     = test_ds[0]
    num_series = sample[0][0].shape[0]
    sem_tokens = {"market_session": SEMANTIC_TOKEN_VOCAB["market_session"]}
    if cfg["use_event_type"]:
        sem_tokens["event_type"]   = SEMANTIC_TOKEN_VOCAB["event_type"]
    if cfg["use_event_impact"]:
        sem_tokens["event_impact"] = SEMANTIC_TOKEN_VOCAB["event_impact"]

    model = EquityBERT(
        num_series      = num_series,
        input_len       = seq_len,
        pred_len        = pred_len,
        n_layer         = 4,
        revin           = True,
        semantic_tokens = sem_tokens,
    ).to(DEVICE)

    ckpt_dir  = os.path.join(EQUITYBERT_DIR, cfg["folder"], "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "model.pth")   # fallback
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found in: {ckpt_dir}")

    print(f"    Checkpoint: {os.path.basename(ckpt_path)}")
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=(DEVICE != "cpu"),
    )

    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            (x, tokens), y = batch
            x      = x.to(DEVICE)
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
            out    = model((x, tokens))          # (B, 1, pred_len)
            all_preds.append(out.cpu().numpy())
            all_labels.append(y.numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)  # (N, 1, pred_len)
    all_labels = np.concatenate(all_labels, axis=0)   # (N, 1, pred_len)

    # Inverse-transform: undo StandardScaler (RevIN already undone inside model)
    preds_orig  = all_preds.squeeze(1).reshape(-1)  * r_std + r_mean
    labels_orig = all_labels.squeeze(1).reshape(-1) * r_std + r_mean

    # Attach datetimes — sample i predicts rows [i+seq_len .. i+seq_len+pred_len)
    rows = []
    n_windows = all_preds.shape[0]
    for i in range(n_windows):
        r_begin = i + seq_len
        for step in range(pred_len):
            idx = r_begin + step
            if idx >= len(test_ds.raw_data):
                break
            dt   = test_ds.raw_data["Datetime"].iloc[idx]
            flat = i * pred_len + step
            rows.append({
                "Datetime":    dt,
                "Horizon_step": step + 1,
                "Actual_r":    labels_orig[flat],
                "Predicted_r": preds_orig[flat],
            })

    df = pd.DataFrame(rows)
    df["AE"] = np.abs(df["Actual_r"] - df["Predicted_r"])
    df["SE"] = (df["Actual_r"] - df["Predicted_r"]) ** 2

    # Strip timezone for merge compatibility
    df["Datetime"] = df["Datetime"].dt.tz_localize(None) if df["Datetime"].dt.tz else df["Datetime"]
    return df


def load_lstm_test(horizon_tag):
    """
    Load the Test sheet from lstm_predictions_{horizon_tag}.xlsx.
    Returns a DataFrame with the same column set as run_equitybert_inference.
    The xlsx has a trailing summary row with "MEAN" in the Datetime cell;
    errors="coerce" turns it into NaT which we then drop.
    """
    xlsx_path = os.path.join(LSTM_DIR, f"lstm_predictions_{horizon_tag}.xlsx")
    df = pd.read_excel(xlsx_path, sheet_name="Test")
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])          # drops the MEAN summary row
    # evaluate_lstm.py saves tz-naive datetimes; guard in case they're not
    if df["Datetime"].dt.tz is not None:
        df["Datetime"] = df["Datetime"].dt.tz_localize(None)
    df["Horizon_step"] = df["Horizon_step"].astype(int)
    return df


# ── Statistical tests ──────────────────────────────────────────────────────────

def diebold_mariano(e1, e2, h, criterion):
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    Parameters
    ----------
    e1, e2   : 1-D arrays of raw (signed) forecast errors for model 1 (LSTM)
               and model 2 (EquityBERT).
    h        : forecast horizon — used to set the Newey-West lag truncation
               to h - 1, which accounts for the MA(h-1) error structure in
               multi-step-ahead forecasts.
    criterion: "MAE"  → loss differential d_t = |e1_t| − |e2_t|
               "MSE"  → loss differential d_t = e1_t² − e2_t²

    Returns
    -------
    dm_stat  : float — positive means model 1 (LSTM) has higher loss, i.e.,
               EquityBERT is better.
    p_value  : float — two-sided p-value under the asymptotic N(0,1) null.
    """
    if criterion == "MAE":
        d = np.abs(e1) - np.abs(e2)
    elif criterion == "MSE":
        d = e1 ** 2 - e2 ** 2
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    T     = len(d)
    d_bar = np.mean(d)

    # Newey-West long-run variance with max lag = h - 1
    max_lag = max(h - 1, 0)
    gamma_0 = np.var(d, ddof=0)
    lrv = gamma_0
    for lag in range(1, max_lag + 1):
        gamma_k = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        lrv += 2 * gamma_k

    lrv = max(lrv, 1e-30)     # guard against numerical zero
    dm_stat = d_bar / np.sqrt(lrv / T)
    p_value = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
    return float(dm_stat), p_value


def run_tests(lstm_df, eb_df, pred_len, label):
    """
    Align predictions on (Datetime, Horizon_step) and run significance tests.
    Returns a results dict.
    """
    merged = pd.merge(
        lstm_df.rename(columns={
            "Actual_r": "Actual_lstm", "Predicted_r": "Pred_lstm",
            "AE": "AE_lstm",           "SE": "SE_lstm",
        }),
        eb_df.rename(columns={
            "Actual_r": "Actual_eb",   "Predicted_r": "Pred_eb",
            "AE": "AE_eb",             "SE": "SE_eb",
        }),
        on=["Datetime", "Horizon_step"],
        how="inner",
    )

    N = len(merged)
    if N == 0:
        print(f"  WARNING: no aligned samples for {label}. Skipping.")
        return None

    # Use LSTM actual as common ground truth (same datetimes → same r values)
    e_lstm = (merged["Actual_lstm"] - merged["Pred_lstm"]).values
    e_eb   = (merged["Actual_lstm"] - merged["Pred_eb"]).values

    # ── Aggregate metrics ────────────────────────────────────────────────────
    lstm_mae = merged["AE_lstm"].mean()
    eb_mae   = merged["AE_eb"].mean()
    lstm_mse = merged["SE_lstm"].mean()
    eb_mse   = merged["SE_eb"].mean()

    # ── DM test ─────────────────────────────────────────────────────────────
    dm_mae,  p_dm_mae  = diebold_mariano(e_lstm, e_eb, h=pred_len, criterion="MAE")
    dm_mse,  p_dm_mse  = diebold_mariano(e_lstm, e_eb, h=pred_len, criterion="MSE")

    # ── Paired t-test ────────────────────────────────────────────────────────
    t_mae, p_t_mae = stats.ttest_rel(merged["AE_lstm"].values, merged["AE_eb"].values)
    t_mse, p_t_mse = stats.ttest_rel(merged["SE_lstm"].values, merged["SE_eb"].values)

    winner_mae = "LSTM" if lstm_mae < eb_mae else "EquityBERT"
    winner_mse = "LSTM" if lstm_mse < eb_mse else "EquityBERT"

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "n.s."

    print(f"\n  ─── {label} ───")
    print(f"  Aligned pairs : {N:,}")
    print(f"  {'Model':<14} {'MAE':>12} {'MSE':>12}")
    print(f"  {'LSTM':<14} {lstm_mae:>12.6f} {lstm_mse:>12.6f}")
    print(f"  {'EquityBERT':<14} {eb_mae:>12.6f} {eb_mse:>12.6f}")
    print(f"  {'Winner':<14} {winner_mae:>12} {winner_mse:>12}")
    print()
    print(f"  {'Test':<22} {'MAE stat':>10} {'MAE p':>10} {'sig':>5}  "
          f"{'MSE stat':>10} {'MSE p':>10} {'sig':>5}")
    print(f"  {'DM (Newey-West)':<22} {dm_mae:>10.4f} {p_dm_mae:>10.4f} {sig(p_dm_mae):>5}  "
          f"{dm_mse:>10.4f} {p_dm_mse:>10.4f} {sig(p_dm_mse):>5}")
    print(f"  {'Paired t-test':<22} {t_mae:>10.4f} {p_t_mae:>10.4f} {sig(p_t_mae):>5}  "
          f"{t_mse:>10.4f} {p_t_mse:>10.4f} {sig(p_t_mse):>5}")
    print()
    print("  Significance codes: *** p<0.001  ** p<0.01  * p<0.05  n.s. p≥0.05")
    print("  DM stat > 0  →  LSTM has higher loss (EquityBERT wins)")
    print("  DM stat < 0  →  EquityBERT has higher loss (LSTM wins)")

    return {
        "label":       label,
        "N":           N,
        "lstm_mae":    lstm_mae,
        "eb_mae":      eb_mae,
        "lstm_mse":    lstm_mse,
        "eb_mse":      eb_mse,
        "winner_mae":  winner_mae,
        "winner_mse":  winner_mse,
        "dm_stat_mae": dm_mae,
        "dm_p_mae":    p_dm_mae,
        "dm_sig_mae":  sig(p_dm_mae),
        "dm_stat_mse": dm_mse,
        "dm_p_mse":    p_dm_mse,
        "dm_sig_mse":  sig(p_dm_mse),
        "t_stat_mae":  t_mae,
        "t_p_mae":     p_t_mae,
        "t_sig_mae":   sig(p_t_mae),
        "t_stat_mse":  t_mse,
        "t_p_mse":     p_t_mse,
        "t_sig_mse":   sig(p_t_mse),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading macro event calendar…")
    events_df = load_events_df()

    # Cache LSTM test sheets (loaded once per horizon tag)
    lstm_cache: dict[str, pd.DataFrame] = {}

    all_results = []

    for cfg in CONFIGS:
        print(f"\n{'='*65}")
        print(f"Config: {cfg['label']}")

        # Load LSTM test predictions (once per horizon)
        tag = cfg["horizon_tag"]
        if tag not in lstm_cache:
            print(f"  Loading LSTM Test sheet for horizon {tag}…")
            lstm_df = load_lstm_test(tag)
            lstm_cache[tag] = lstm_df
            print(f"  LSTM test rows: {len(lstm_df):,}  "
                  f"({lstm_df['Datetime'].min()} → {lstm_df['Datetime'].max()})")
        lstm_df = lstm_cache[tag]

        # Run EquityBERT inference
        print(f"  Running EquityBERT inference…")
        try:
            eb_df = run_equitybert_inference(cfg, events_df)
        except FileNotFoundError as exc:
            print(f"  SKIP: {exc}")
            continue

        print(f"  EquityBERT test rows: {len(eb_df):,}  "
              f"({eb_df['Datetime'].min()} → {eb_df['Datetime'].max()})")

        result = run_tests(lstm_df, eb_df, pred_len=cfg["pred_len"], label=cfg["label"])
        if result:
            all_results.append(result)

    if not all_results:
        print("\nNo results produced. Check checkpoint paths.")
        return

    # ── Save outputs ──────────────────────────────────────────────────────────
    txt_path = os.path.join(OUTPUT_DIR, "significance_results.txt")
    csv_path = os.path.join(OUTPUT_DIR, "significance_results.csv")

    with open(txt_path, "w") as f:
        f.write("EquityBERT vs LSTM — Statistical Significance Tests\n")
        f.write("=" * 70 + "\n\n")
        f.write("DM test uses Newey-West long-run variance with (pred_len - 1) lags.\n")
        f.write("Both tests are two-sided. All metrics are on the original log-range scale.\n")
        f.write("DM stat > 0  →  LSTM has higher loss (EquityBERT is better).\n\n")
        f.write("Significance codes: *** p<0.001  ** p<0.01  * p<0.05  n.s. p≥0.05\n\n")

        for r in all_results:
            f.write(f"{'─'*65}\n")
            f.write(f"Config  : {r['label']}\n")
            f.write(f"Samples : {r['N']:,}\n\n")
            f.write(f"  {'Model':<14} {'MAE':>12} {'MSE':>12}\n")
            f.write(f"  {'LSTM':<14} {r['lstm_mae']:>12.6f} {r['lstm_mse']:>12.6f}\n")
            f.write(f"  {'EquityBERT':<14} {r['eb_mae']:>12.6f} {r['eb_mse']:>12.6f}\n")
            f.write(f"  {'Winner':<14} {r['winner_mae']:>12} {r['winner_mse']:>12}\n\n")
            f.write(f"  DM test (MAE) : stat={r['dm_stat_mae']:+.4f}  p={r['dm_p_mae']:.4f}  {r['dm_sig_mae']}\n")
            f.write(f"  DM test (MSE) : stat={r['dm_stat_mse']:+.4f}  p={r['dm_p_mse']:.4f}  {r['dm_sig_mse']}\n")
            f.write(f"  Paired t (MAE): stat={r['t_stat_mae']:+.4f}  p={r['t_p_mae']:.4f}  {r['t_sig_mae']}\n")
            f.write(f"  Paired t (MSE): stat={r['t_stat_mse']:+.4f}  p={r['t_p_mse']:.4f}  {r['t_sig_mse']}\n\n")

    pd.DataFrame(all_results).to_csv(csv_path, index=False)

    print(f"\n{'='*65}")
    print("DONE")
    print(f"  TXT → {txt_path}")
    print(f"  CSV → {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        import traceback
        print(f"\nError: {exc}")
        traceback.print_exc()
