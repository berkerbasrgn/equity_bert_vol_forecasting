"""
Post-hoc evaluation script for EquityBERT.

Loads saved checkpoints and computes metrics on the ORIGINAL (inverse-transformed)
log-range scale without re-running training.

Matches the architecture and data loading in:
  - train_equitybert.py  (your training script)
  - model_equitybert.py  (EquityBERT class)
  - src/mydataset.py     (Dataset_SP500_1H)

Usage:
    python evaluate_inverse.py

Outputs:
    - equitybert_results_original_scale.txt
    - equitybert_results_original_scale.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "src"))

from src.mydataset import Dataset_SP500_1H, SEMANTIC_TOKEN_VOCAB
from src.model_bert import EquityBERT

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
RUNS_DIR   = "/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/vola-bert/runs/equityBERT/v28"
DATA_PATH  = "/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/vola-bert/data/processed/ES_1h.parquet"
EVENTS_CSV = "/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/vola-bert/data/NEW_macro_events.csv"
OUTPUT_TXT = "equitybert_results_original_scale.txt"
OUTPUT_CSV = "equitybert_results_original_scale.csv"
BATCH_SIZE = 256

#  exact same configurations that were trained in v27 
# (folder_name, horizon_label, seq_len, pred_len, use_events, use_event_type, use_event_impact)
CONFIGS = [
    ("No Events_24to5_full",          "24→5",  24,  5, False, False, False),
    ("Event Type Only_24to5_full",    "24→5",  24,  5, True,  True,  False),
    ("Event Timing Only_24to5_full",  "24→5",  24,  5, True,  False, False),
    ("No Events_50to10_full",         "50→10", 50, 10, False, False, False),
    ("Event Type Only_50to10_full",   "50→10", 50, 10, True,  True,  False),
    ("Event Timing Only_50to10_full", "50→10", 50, 10, True,  False, False),
]


def load_events_df():
    """Load macro event calendar exactly as in the training script."""
    events_df = pd.read_csv(EVENTS_CSV)
    events_df = events_df.rename(columns={"event_time_et": "datetime"})
    events_df["impact"] = "NONE"
    events_df["datetime"] = pd.to_datetime(
        events_df["datetime"], utc=True
    ).dt.tz_convert("America/New_York")
    return events_df


def build_semantic_tokens(use_event_type, use_event_impact):
    """
    Builds the semantic_tokens dict exactly as in train_equitybert_single_config.
    market_session is always included; event_type and event_impact are optional.
    """
    semantic_tokens = {
        "market_session": SEMANTIC_TOKEN_VOCAB["market_session"],
    }
    if use_event_type:
        semantic_tokens["event_type"] = SEMANTIC_TOKEN_VOCAB["event_type"]
    if use_event_impact:
        semantic_tokens["event_impact"] = SEMANTIC_TOKEN_VOCAB["event_impact"]
    return semantic_tokens


def get_r_scaler_params(dataset):
    """
    Extract mean and std for the target column 'r' from the fitted scaler.
    r is always the last column in the feature matrix.

    Tries the three most common naming conventions used in custom
    StandardScaler implementations. Check src/utils.py if none match
    and add the correct attribute names below.
    """
    scaler = dataset.scaler

    if hasattr(scaler, "mean") and hasattr(scaler, "std"):
        return float(scaler.mean[-1]), float(scaler.std[-1])
    elif hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        # sklearn-style
        return float(scaler.mean_[-1]), float(scaler.scale_[-1])
    elif hasattr(scaler, "_mean") and hasattr(scaler, "_std"):
        return float(scaler._mean[-1]), float(scaler._std[-1])
    else:
        raise AttributeError(
            "Cannot find mean/std attributes on your StandardScaler. "
            "Open src/utils.py, find the attribute names, and update "
            "get_r_scaler_params() in this script accordingly."
        )


def evaluate_config(folder_name, horizon_label, seq_len, pred_len,
                    use_events, use_event_type, use_event_impact,
                    events_df):

    print(f"\n{'='*65}")
    print(f"Config  : {folder_name}")
    print(f"Horizon : {horizon_label}  ({seq_len}h lookback → {pred_len}h forecast)")

    # Rebuild dataset 
    # Instantiating Dataset_SP500_1H with flag="test" triggers __read_data__,
    # which fits the StandardScaler on the training rows of the full dataset
    # (rows 0 to 0.70*N) — identical to what happened during training.
    # So the scaler here is the same scaler used at training time.
    common_kwargs = dict(
        data_path        = DATA_PATH,
        events_df        = events_df if use_events else None,
        size             = (seq_len, pred_len),
        use_events       = use_events,
        use_event_type   = use_event_type,
        use_event_impact = use_event_impact,
        use_explainable  = True,
        mode             = "24h",
    )
    test_dataset = Dataset_SP500_1H(flag="test", **common_kwargs)

    print(f"Test period  : {test_dataset.start_date} → {test_dataset.end_date}")
    print(f"Test samples : {len(test_dataset):,}")

    # Get scaler params for r 
    r_mean, r_std = get_r_scaler_params(test_dataset)
    print(f"Scaler (r)   : mean={r_mean:.6f}  std={r_std:.6f}")

    # Build model 
    # Infer num_series from dataset exactly as the training script does:
    #   sample = dataset[0]  →  ((x, tokens), y)  →  x.shape = (N, L)
    sample     = test_dataset[0]
    num_series = sample[0][0].shape[0]

    semantic_tokens = build_semantic_tokens(use_event_type, use_event_impact)

    model = EquityBERT(
        num_series      = num_series,
        input_len       = seq_len,
        pred_len        = pred_len,
        n_layer         = 4,       # must match training config base_config["n_layer"]
        revin           = True,
        semantic_tokens = semantic_tokens,
    ).to(DEVICE)

    #  Load checkpoint 
    checkpoint_path = os.path.join(
        RUNS_DIR, folder_name, "checkpoints", "checkpoint.pth"
    )
    if not os.path.exists(checkpoint_path):
        print(f"  ERROR: checkpoint not found at {checkpoint_path}")
        print(f"  Skipping this config.")
        return None

    state = torch.load(checkpoint_path, map_location=DEVICE)
    # Handle both raw state_dict and trainer-wrapped checkpoints
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    print(f"Checkpoint   : {checkpoint_path}")

    #  Inference on scaled outputs 
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Dataset returns ((x, tokens), y) when use_explainable=True
            (x, tokens), y = batch

            x      = x.to(DEVICE)
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
            # model.forward(x_data) where x_data is the tuple (x, tokens)
            preds  = model((x, tokens))   # (B, 1, pred_len)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)  # (N, 1, pred_len)
    all_labels = np.concatenate(all_labels, axis=0)  # (N, 1, pred_len)

    # Inverse transform: scaled → original log-range scale 
    # StandardScaler: x_orig = x_scaled * std + mean
    # RevIN is applied inside the model's forward pass and already undone there,
    # so what comes out of model() is still on the StandardScaler scale.
    # We only need to undo the StandardScaler here.
    preds_orig  = all_preds.reshape(-1)  * r_std + r_mean
    labels_orig = all_labels.reshape(-1) * r_std + r_mean

    # Naive baseline on original scale 
    # Naive = last observed r in the lookback window, repeated for all pred steps.
    # raw_data["r"] is on the original (pre-StandardScaler) log-range scale.
    r_raw = test_dataset.raw_data["r"].values

    naive_preds  = []
    naive_labels = []
    for i in range(test_dataset.tot_len):
        s_end   = i + seq_len
        r_begin = s_end
        r_end   = r_begin + pred_len
        if r_end > len(r_raw):
            break
        naive_preds.append(np.full(pred_len, r_raw[s_end - 1]))
        naive_labels.append(r_raw[r_begin:r_end])

    naive_preds  = np.array(naive_preds).reshape(-1)
    naive_labels = np.array(naive_labels).reshape(-1)

    #  Align lengths 
    n = min(len(preds_orig), len(naive_preds))
    preds_orig   = preds_orig[:n]
    labels_orig  = labels_orig[:n]
    naive_preds  = naive_preds[:n]
    naive_labels = naive_labels[:n]

    #  Metrics 
    model_mae = float(np.mean(np.abs(labels_orig  - preds_orig)))
    model_mse = float(np.mean((labels_orig  - preds_orig) ** 2))
    naive_mae = float(np.mean(np.abs(naive_labels - naive_preds)))
    naive_mse = float(np.mean((naive_labels - naive_preds) ** 2))
    rmae      = model_mae / naive_mae
    rmse_rel  = model_mse / naive_mse

    print(f"\nNaive MAE (original scale) : {naive_mae:.6f}   MSE: {naive_mse:.6f}")
    print(f"Model MAE (original scale) : {model_mae:.6f}   MSE: {model_mse:.6f}")
    print(f"rMAE : {rmae:.4f}   rMSE : {rmse_rel:.4f}")
    print(f"vs Naive : -{(1 - rmae)*100:.1f}%")

    return {
        "config":    folder_name,
        "horizon":   horizon_label,
        "naive_mae": naive_mae,
        "naive_mse": naive_mse,
        "model_mae": model_mae,
        "model_mse": model_mse,
        "rMAE":      rmae,
        "rMSE":      rmse_rel,
        "n_samples": n // pred_len,
        "test_start": str(test_dataset.start_date),
        "test_end":   str(test_dataset.end_date),
    }


def main():
    events_df = load_events_df()
    results   = []

    for cfg in CONFIGS:
        folder, horizon, seq_len, pred_len, use_events, use_et, use_ei = cfg
        result = evaluate_config(
            folder_name      = folder,
            horizon_label    = horizon,
            seq_len          = seq_len,
            pred_len         = pred_len,
            use_events       = use_events,
            use_event_type   = use_et,
            use_event_impact = use_ei,
            events_df        = events_df,
        )
        if result is not None:
            results.append(result)

    if not results:
        print("\nNo results — check checkpoint paths above.")
        return

    with open(OUTPUT_TXT, "w") as f:
        f.write("EquityBERT — Original Scale Results (after inverse transform)\n")
        f.write("=" * 65 + "\n\n")
        for r in results:
            f.write(f"{r['horizon']}  {r['config']}\n")
            f.write(f"  Test period : {r['test_start']} → {r['test_end']}\n")
            f.write(f"  Naive MAE   : {r['naive_mae']:.6f}   MSE: {r['naive_mse']:.6f}\n")
            f.write(f"  Model MAE   : {r['model_mae']:.6f}   MSE: {r['model_mse']:.6f}\n")
            f.write(f"  rMAE        : {r['rMAE']:.4f}\n")
            f.write(f"  rMSE        : {r['rMSE']:.4f}\n")
            f.write(f"  vs Naive    : -{(1 - r['rMAE'])*100:.1f}%\n\n")
    print(f"\nTXT saved : {OUTPUT_TXT}")

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"CSV saved : {OUTPUT_CSV}")

    print("SUMMARY — Original log-range scale (after inverse transform)")
    print(f"{'Config':<32} {'Horizon':<8} {'MAE':>10} {'rMAE':>8} {'rMSE':>8}")
    print("-" * 100)
    for r in results:
        name = r['config'].replace('_full', '').replace('_', ' ')
        print(f"{name:<32} {r['horizon']:<8} "
              f"{r['model_mae']:>10.6f} {r['rMAE']:>8.4f} {r['rMSE']:>8.4f}")


if __name__ == "__main__":
    main()