# Baseline LSTM training script for S&P 500 hourly volatility forecasting.
#
# This script trains a simple LSTM model as a non-Transformer baseline for
# comparison with EquityBERT.  It uses the same Dataset_SP500_1H class but
# with use_explainable=False (plain tensor output, no semantic tokens).
#
# The LSTM baseline is important for the thesis because:
#   - It isolates the contribution of the BERT architecture
#   - It shows whether pre-trained language model representations actually
#     help beyond a standard sequential model
#   - rMAE/rMSE relative to naive are directly comparable with EquityBERT
#

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
 
from src.mydataset import Dataset_SP500_1H
from src.model_lstm import LSTMModel


class NaiveBaseline:
    def evaluate(self, dataset):
        predictions = []
        targets = []
 
        for i in range(len(dataset)):
            x, y = dataset[i]  # (x,y) no tokens for LSTM datasets

            last_val = x[0, -1].item()
            horizon = y.shape[1]
            pred = torch.full((1, horizon), last_val)
            predictions.append(pred)
            targets.append(y)
 
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        return {
            "MAE": torch.mean(torch.abs(predictions - targets)).item(),
            "MSE": torch.mean((predictions - targets) ** 2).item(),
        }


# ── METRICS ──

def relative_metrics(model_mae, model_mse, naive_mae, naive_mse):
    return model_mae / naive_mae, model_mse / naive_mse


# ── PLOTTING ──

def save_loss_plot(result, save_path):
    """Save a 2x2 panel plot for one horizon with split info."""
    lookback = result["lookback"]
    forecast = result["forecast"]
    label = f"{lookback}h → {forecast}h"

    train_losses = result["train_losses"]
    val_losses = result["val_losses"]
    train_maes = result["train_maes"]
    val_maes = result["val_maes"]

    # Split info text
    total = result["train_samples"] + result["val_samples"] + result["test_samples"]
    train_pct = result["train_samples"] / total * 100
    val_pct = result["val_samples"] / total * 100
    test_pct = result["test_samples"] / total * 100

    split_text = (
        f"Train: {result['train_samples']:,} ({train_pct:.0f}%)  "
        f"{result['train_start']} → {result['train_end']}\n"
        f"Val:     {result['val_samples']:,} ({val_pct:.0f}%)  "
        f"{result['val_start']} → {result['val_end']}\n"
        f"Test:   {result['test_samples']:,} ({test_pct:.0f}%)  "
        f"{result['test_start']} → {result['test_end']}"
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Top-left: MSE curves
    axes[0, 0].plot(train_losses, label="Train MSE")
    axes[0, 0].plot(val_losses, label="Val MSE", linestyle="--")
    axes[0, 0].set(xlabel="Epoch", ylabel="MSE", title="MSE Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Top-right: MAE curves
    axes[0, 1].plot(train_maes, label="Train MAE")
    axes[0, 1].plot(val_maes, label="Val MAE", linestyle="--")
    axes[0, 1].set(xlabel="Epoch", ylabel="MAE", title="MAE Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Bottom-left: Train-Val gap MSE
    axes[1, 0].plot(train_losses, label="Train", alpha=0.7)
    axes[1, 0].plot(val_losses, label="Val", alpha=0.7)
    axes[1, 0].fill_between(
        range(len(train_losses)), train_losses, val_losses,
        alpha=0.15, color="red", label="Gap"
    )
    axes[1, 0].set(xlabel="Epoch", ylabel="MSE", title="Train-Val Gap (MSE)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Bottom-right: Train-Val gap MAE
    axes[1, 1].plot(train_maes, label="Train", alpha=0.7)
    axes[1, 1].plot(val_maes, label="Val", alpha=0.7)
    axes[1, 1].fill_between(
        range(len(train_maes)), train_maes, val_maes,
        alpha=0.15, color="red", label="Gap"
    )
    axes[1, 1].set(xlabel="Epoch", ylabel="MAE", title="Train-Val Gap (MAE)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.suptitle(f"LSTM Baseline — {label}", fontsize=14, fontweight="bold")

    # Add split info as text below the title
    fig.text(
        0.5, 0.93, split_text,
        ha="center", va="top", fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── TRAIN ──

def run_experiment(run_dir, config):
    """
    Train and evaluate an LSTM baseline for one (lookback, forecast) pair.

    Unlike EquityBERT, the LSTM uses a simple manual training loop rather
    than the Trainer class, because:
      - No semantic tokens → no input_to_device dispatch needed
      - MSE loss (not MAE) is more standard for LSTM baselines
      - No OneCycleLR — Adam + ReduceLROnPlateau is sufficient for this model
      - Simpler code is easier to debug and explain in the thesis
    """
    print("\n" + "=" * 60)
    print(f"LSTM Baseline  {config['lookback']}→{config['forecast']}")
    print("=" * 60)

    # ---- Datasets (use_explainable=False → plain (x, y) tuples) ----------
    common = dict(
        data_path=config["data_path"],
        events_df=None,
        size=(config["lookback"], config["forecast"]),
        use_events=False,
        use_explainable=False,
        use_technical=config.get("use_technical", True),
        use_interday=config.get("use_interday", True),
    )

    train_ds = Dataset_SP500_1H(flag="train", **common)
    val_ds   = Dataset_SP500_1H(flag="val",   **common)
    test_ds  = Dataset_SP500_1H(flag="test",  **common)

    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Test: {len(test_ds):,}")
    print(f"Train: {train_ds.start_date} → {train_ds.end_date}")
    print(f"Val:   {val_ds.start_date} → {val_ds.end_date}")
    print(f"Test:  {test_ds.start_date} → {test_ds.end_date}")

    # ── BASELINE ──
    naive_results = NaiveBaseline().evaluate(test_ds)
    naive_mae, naive_mse = naive_results["MAE"], naive_results["MSE"]
    print(f"Naive MAE: {naive_mae:.6f}  |  Naive MSE: {naive_mse:.6f}")

    # ── MODEL ──
    sample_x, _ = train_ds[0]
    num_series = sample_x.shape[0]
 
    model = LSTMModel(
        num_series=num_series,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        pred_len=config["forecast"],
        dropout=config.get("dropout", 0.2),
    )
    params = model.num_params
    print(f"Parameters — total: {params['total']:,}  trainable: {params['trainable']:,}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = model.to(device)
 
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = torch.nn.MSELoss()
 
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"])

    train_losses = []
    val_losses   = []
    train_maes   = []
    val_maes     = []

    # ── TRAIN LOOP ──
    best_val = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(
        run_dir, f"best_model_{config['lookback']}to{config['forecast']}.pth"
    )
    print("\nTraining LSTM...")

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        epoch_mae_sum = 0
 
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
 
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mae_sum += torch.mean(torch.abs(output - y)).item()
 
        train_losses.append(epoch_loss / len(train_loader))
        train_maes.append(epoch_mae_sum / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        val_mae_sum = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
                val_mae_sum += torch.mean(torch.abs(output - y)).item()
 
        val_losses.append(val_loss / len(val_loader))
        val_maes.append(val_mae_sum / len(val_loader))

        # Step the LR scheduler
        scheduler.step(val_losses[-1])

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:3d}/{config['epochs']}  "
            f"Train MSE: {train_losses[-1]:.6f}  Val MSE: {val_losses[-1]:.6f}  "
            f"Train MAE: {train_maes[-1]:.6f}  Val MAE: {val_maes[-1]:.6f}  "
            f"LR: {current_lr:.2e}"
        )

        # Early stopping
        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # ── TEST ──
    print("\nTesting...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
 
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"])
    preds = []
    trues = []
 
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            preds.append(output)
            trues.append(y)
 
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    save_dir = os.path.join(run_dir, "predictions")
    os.makedirs(save_dir, exist_ok=True)

    horizon_tag = f"{config['lookback']}to{config['forecast']}" 
    np.save(os.path.join(save_dir, f"y_true_{horizon_tag}.npy"), trues.cpu().numpy())
    np.save(os.path.join(save_dir, f"lstm_pred_{horizon_tag}.npy"), preds.cpu().numpy())
    print(f"Predictions saved to {save_dir}")
 
    model_mae = torch.mean(torch.abs(preds - trues)).item()
    model_mse = torch.mean((preds - trues) ** 2).item()
 
    print(f"\nLSTM MAE: {model_mae:.6f}")
    print(f"LSTM MSE: {model_mse:.6f}")
 
    rmae, rmse = relative_metrics(model_mae, model_mse, naive_mae, naive_mse)
 
    print(f"\nrMAE: {rmae:.4f}")
    print(f"rMSE: {rmse:.4f}")
 
    if rmae < 1.0:
        print(f"LSTM is {(1 - rmae) * 100:.1f}% better than naive")
    else:
        print(f"LSTM is {(rmae - 1) * 100:.1f}% worse than naive")

    # Format dates for display
    def fmt_date(d):
        return str(d).split(" ")[0] if d else "N/A"

    return {
        "lookback": config["lookback"],
        "forecast": config["forecast"],
        "naive_mae": naive_mae,
        "naive_mse": naive_mse,
        "model_mae": model_mae,
        "model_mse": model_mse,
        "rmae": rmae,
        "rmse": rmse,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_maes": train_maes,
        "val_maes": val_maes,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "train_start": fmt_date(train_ds.start_date),
        "train_end": fmt_date(train_ds.end_date),
        "val_start": fmt_date(val_ds.start_date),
        "val_end": fmt_date(val_ds.end_date),
        "test_start": fmt_date(test_ds.start_date),
        "test_end": fmt_date(test_ds.end_date),
    }


# ── MAIN ──

def main():
    base_dir = "runs/lstm_baseline"
    os.makedirs(base_dir, exist_ok=True)

    # Run versioning
    existing = [d for d in os.listdir(base_dir) if d.startswith("v")]
    run_id = len(existing) + 1
    run_dir = os.path.join(base_dir, f"v{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nSaving results to {run_dir}")

    config = {
        "data_path": "/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/vola-bert/data/processed/ES_1h.parquet",
        "hidden_size": 64,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "batch_size": 128,
        "epochs": 100,
        "lr": 3e-4,
        "weight_decay": 5e-4,
        "lr": 3e-4,
        "weight_decay": 5e-4,
        "patience": 15,
        "use_technical": True,
        "use_interday": True,
    }

    horizons = [
        {"lookback": 24, "forecast": 5},
        {"lookback": 50, "forecast": 10},
    ]

    all_results = []
    for horizon in horizons:
        result = run_experiment(run_dir, {**config, **horizon})
        all_results.append(result)

        # Save per-horizon plot
        plot_path = os.path.join(
            run_dir, f"loss_{horizon['lookback']}to{horizon['forecast']}.png"
        )
        save_loss_plot(result, plot_path)
        print(f"Loss curves saved to {plot_path}")

    # Save results summary
    results_path = os.path.join(run_dir, f"lstm_results_v{run_id}.txt")
    with open(results_path, "w") as f:
        f.write("LSTM Baseline — Results Summary\n")
        f.write("=" * 60 + "\n\n")

        # Config
        f.write("Configuration:\n")
        for k, v in config.items():
            if k != "data_path":
                f.write(f"  {k}: {v}\n")
        f.write("\n")

        # Per-horizon results
        for r in all_results:
            total = r["train_samples"] + r["val_samples"] + r["test_samples"]
            train_pct = r["train_samples"] / total * 100
            val_pct = r["val_samples"] / total * 100
            test_pct = r["test_samples"] / total * 100

            f.write(f"Horizon: {r['lookback']}h → {r['forecast']}h\n")
            f.write(f"  Train: {r['train_samples']:,} ({train_pct:.0f}%)  "
                    f"{r['train_start']} → {r['train_end']}\n")
            f.write(f"  Val:   {r['val_samples']:,} ({val_pct:.0f}%)  "
                    f"{r['val_start']} → {r['val_end']}\n")
            f.write(f"  Test:  {r['test_samples']:,} ({test_pct:.0f}%)  "
                    f"{r['test_start']} → {r['test_end']}\n")
            f.write(f"\n")
            f.write(f"  Naive MAE: {r['naive_mae']:.6f}   MSE: {r['naive_mse']:.6f}\n")
            f.write(f"  Model MAE: {r['model_mae']:.6f}   MSE: {r['model_mse']:.6f}\n")
            f.write(f"  rMAE: {r['rmae']:.4f}   rMSE: {r['rmse']:.4f}\n")
            if r["rmae"] < 1.0:
                f.write(f"  LSTM is {(1 - r['rmae']) * 100:.1f}% better than naive\n")
            f.write("\n" + "-" * 60 + "\n\n")

    print(f"Results saved to {results_path}")

    # Final summary to terminal
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for r in all_results:
        print(
            f"  {r['lookback']}h → {r['forecast']}h  |  "
            f"rMAE: {r['rmae']:.4f}  rMSE: {r['rmse']:.4f}  |  "
            f"MAE: {r['model_mae']:.6f}  MSE: {r['model_mse']:.6f}"
        )


if __name__ == "__main__":
    try:
        main()
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()