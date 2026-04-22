"""
 Extract predictions, build Excel report & visualise.

What it produces (inside runs/lstm_baseline/):
    1. lstm_predictions.xlsx   — one sheet per split (Train / Val / Test),
       columns: Datetime, Actual_r, Predicted_r, AE, SE  + summary row
    2. lstm_vol_forecast.png   — actual vs predicted time-series per split
    3. lstm_scatter.png        — scatter plot (actual vs predicted) per split
"""

 
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime
import glob
 
from src.mydataset import Dataset_SP500_1H
from src.model_lstm import LSTMModel
 
 
# CONFIG
DATA_PATH = "/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/vola-bert/data/processed/ES_1h.parquet"
BASE_RUN_DIR = "runs"
 
MODEL_CFG = dict(
    hidden_size=64,
    num_layers=2,
    dropout=0.4,
)
 
HORIZONS = [
    {"lookback": 24, "forecast": 5},
    {"lookback": 50, "forecast": 10},
]
 
BATCH_SIZE    = 128
USE_TECHNICAL = True
USE_INTERDAY  = True
 
 
# HELPERS
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
 
 
def find_run_folder(date_str=None):
    """
    Find the run folder in BASE_RUN_DIR.
    If date_str is provided, looks for lstm_baseline_YYYY-MM-DD.
    Otherwise, returns the latest (most recent) folder.
    """
    pattern = os.path.join(BASE_RUN_DIR, "lstm_baseline_*")
    folders = sorted(glob.glob(pattern), reverse=True)
    
    if not folders:
        raise FileNotFoundError(f"No lstm_baseline_* folders found in {BASE_RUN_DIR}/")
    
    if date_str:
        target = os.path.join(BASE_RUN_DIR, f"lstm_baseline_{date_str}")
        if os.path.exists(target):
            return target
        else:
            raise FileNotFoundError(f"Folder {target} not found")
    else:
        latest = folders[0]
        print(f"Found run folders: {[os.path.basename(f) for f in folders]}")
        print(f"Using latest: {os.path.basename(latest)}")
        return latest
 
 
def build_dataset(flag, lookback, forecast):
    return Dataset_SP500_1H(
        data_path=DATA_PATH,
        events_df=None,
        flag=flag,
        size=(lookback, forecast),
        use_events=False,
        use_explainable=False,
        use_technical=USE_TECHNICAL,
        use_interday=USE_INTERDAY,
    )
 
 
def extract_predictions(model, dataset, device, pred_len):
    """
    Run the trained model over every sample in `dataset` and collect:
      - per-step actual r values
      - per-step predicted r values
      - the datetime at each prediction step
 
    Returns a DataFrame with columns:
        Datetime | Horizon_step | Actual_r | Predicted_r | AE | SE
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
 
    all_preds = []
    all_trues = []
 
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            all_preds.append(out.cpu())
            all_trues.append(y.cpu())
 
    preds = torch.cat(all_preds, dim=0).squeeze(1).numpy()
    trues = torch.cat(all_trues, dim=0).squeeze(1).numpy()
 
    # Inverse-transform back to original scale
    num_features = dataset.scaler.mean.shape[0]
    target_idx   = num_features - 1
 
    def inv(arr_2d):
        N, P = arr_2d.shape
        dummy = np.zeros((N * P, num_features))
        dummy[:, target_idx] = arr_2d.reshape(-1)
        inv_full = dataset.scaler.inverse_transform(dummy)
        return inv_full[:, target_idx].reshape(N, P)
 
    preds_orig = inv(preds)
    trues_orig = inv(trues)
 
    # Attach datetimes
    rows = []
    sample_idx = 0
    for x_tensor, y_tensor in DataLoader(dataset, batch_size=1, shuffle=False):
        week_idx = dataset._find_week_index(sample_idx)
        day_idx  = sample_idx - (dataset.cumsum[week_idx - 1] if week_idx > 0 else 0)
        r_begin  = day_idx + dataset.seq_len
 
        raw_week = dataset.raw_data[week_idx]
 
        for step in range(pred_len):
            dt_step = raw_week["Datetime"].iloc[r_begin + step]
            rows.append({
                "Datetime":      dt_step,
                "Horizon_step":  step + 1,
                "Actual_r":      trues_orig[sample_idx, step],
                "Predicted_r":   preds_orig[sample_idx, step],
            })
        sample_idx += 1
 
    df = pd.DataFrame(rows)
    df["AE"] = np.abs(df["Actual_r"] - df["Predicted_r"])
    df["SE"] = (df["Actual_r"] - df["Predicted_r"]) ** 2
    return df
 
 
# MAIN
def main(date_str=None):
    device = get_device()
    print(f"Device: {device}\n")
 
    # Find the run folder
    run_dir = find_run_folder(date_str)
    print(f"Using run directory: {run_dir}\n")
 
    for hz in HORIZONS:
        lookback = hz["lookback"]
        forecast = hz["forecast"]
        tag      = f"{lookback}to{forecast}"
 
        print(f"\n{'='*60}")
        print(f"  Horizon: {lookback}h → {forecast}h")
        print(f"{'='*60}")
 
        # Load model
        ckpt_path = os.path.join(run_dir, f"best_model_{tag}.pth")
        if not os.path.exists(ckpt_path):
            print(f"  ✗ Checkpoint not found: {ckpt_path}")
            continue
 
        sample_ds = build_dataset("train", lookback, forecast)
        num_series = sample_ds[0][0].shape[0]
 
        model = LSTMModel(
            num_series=num_series,
            pred_len=forecast,
            **MODEL_CFG,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  ✓ Loaded checkpoint")
 
        # Extract predictions per split
        split_dfs = {}
        for split in ["train", "val", "test"]:
            print(f"  Running inference on {split}…")
            ds = build_dataset(split, lookback, forecast)
            df = extract_predictions(model, ds, device, forecast)
            split_dfs[split] = df
            mae = df["AE"].mean()
            mse = df["SE"].mean()
            print(f"    {split.upper():5s}  samples={len(df):>7,}  "
                  f"MAE={mae:.6f}  MSE={mse:.6f}")
 
        # Save to Excel
        xlsx_path = os.path.join(run_dir, f"lstm_predictions_{tag}.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for split, df in split_dfs.items():
                df_out = df.copy()
                df_out["Datetime"] = df_out["Datetime"].dt.tz_localize(None)
                df_out.to_excel(writer, sheet_name=split.capitalize(), index=False)
 
                ws = writer.sheets[split.capitalize()]
                summary_row = ws.max_row + 2
                ws.cell(row=summary_row, column=1, value="MEAN")
                ws.cell(row=summary_row, column=5, value=df["AE"].mean())
                ws.cell(row=summary_row, column=6, value=df["SE"].mean())
 
        print(f"  ✓ Excel saved → {os.path.basename(xlsx_path)}")
 
        # Visualise: time-series
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
        for ax, (split, df) in zip(axes, split_dfs.items()):
            df1 = df[df["Horizon_step"] == 1].copy()
            df1 = df1.sort_values("Datetime")
 
            ax.plot(df1["Datetime"], df1["Actual_r"],
                    label="Actual r", linewidth=0.7, alpha=0.8)
            ax.plot(df1["Datetime"], df1["Predicted_r"],
                    label="Predicted r", linewidth=0.7, alpha=0.8)
            ax.set_title(f"{split.upper()} — 1-step-ahead forecast "
                         f"(MAE={df1['AE'].mean():.5f})")
            ax.set_ylabel("log(H/L)")
            ax.legend()
            ax.grid(True, alpha=0.3)
 
        plt.suptitle(f"LSTM Baseline  {lookback}h→{forecast}h  |  Actual vs Predicted",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        ts_path = os.path.join(run_dir, f"lstm_vol_forecast_{tag}.png")
        plt.savefig(ts_path, dpi=150)
        plt.close()
        print(f"  ✓ Time-series plot → {os.path.basename(ts_path)}")
 
        # Visualise: scatter
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (split, df) in zip(axes, split_dfs.items()):
            df1 = df[df["Horizon_step"] == 1]
            ax.scatter(df1["Actual_r"], df1["Predicted_r"],
                       s=4, alpha=0.3, edgecolors="none")
            lo = min(df1["Actual_r"].min(), df1["Predicted_r"].min())
            hi = max(df1["Actual_r"].max(), df1["Predicted_r"].max())
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
            ax.set_xlabel("Actual r")
            ax.set_ylabel("Predicted r")
            ax.set_title(f"{split.upper()}")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.3)
 
        plt.suptitle(f"LSTM {lookback}h→{forecast}h  |  Scatter",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        sc_path = os.path.join(run_dir, f"lstm_scatter_{tag}.png")
        plt.savefig(sc_path, dpi=150)
        plt.close()
        print(f"  ✓ Scatter plot → {os.path.basename(sc_path)}")
 
    print("\n" + "="*60)
    print(f"All results saved to: {run_dir}/")
    print("="*60)
 
 
if __name__ == "__main__":
    import sys
    
    date_str = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        main(date_str)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
