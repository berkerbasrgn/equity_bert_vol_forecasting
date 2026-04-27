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
            x, y = dataset[i]
            last_val = x[0, -1].item()
            horizon = y.shape[1]
            pred = torch.full((1, horizon), last_val)
            predictions.append(pred)
            targets.append(y)
 
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
 
        mae = torch.mean(torch.abs(predictions - targets)).item()
        mse = torch.mean((predictions - targets) ** 2).item()
 
        return {
            "MAE": mae,
            "MSE": mse
        }
 
 
def relative_metrics(model_mae, model_mse, naive_mae, naive_mse):
    return model_mae / naive_mae, model_mse / naive_mse
 
 
def save_loss_plot(all_results, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
 
    for r in all_results:
        label = f"{r['lookback']}h→{r['forecast']}h"
 
        axes[0, 0].plot(r["train_losses"], label=f"{label} Train")
        axes[0, 0].plot(r["val_losses"], label=f"{label} Val", linestyle="--")
 
        axes[0, 1].plot(r["train_maes"], label=f"{label} Train")
        axes[0, 1].plot(r["val_maes"], label=f"{label} Val", linestyle="--")
 
        axes[1, 0].plot(r["train_losses"], label=f"{label} Train", alpha=0.7)
        axes[1, 0].plot(r["val_losses"], label=f"{label} Val", alpha=0.7)
        axes[1, 0].fill_between(range(len(r["train_losses"])), r["train_losses"], r["val_losses"], alpha=0.1)
 
        axes[1, 1].plot(r["train_maes"], label=f"{label} Train", alpha=0.7)
        axes[1, 1].plot(r["val_maes"], label=f"{label} Val", alpha=0.7)
        axes[1, 1].fill_between(range(len(r["train_maes"])), r["train_maes"], r["val_maes"], alpha=0.1)
 
    axes[0, 0].set(xlabel="Epoch", ylabel="MSE", title="MSE Loss")
    axes[0, 1].set(xlabel="Epoch", ylabel="MAE", title="MAE Loss")
    axes[1, 0].set(xlabel="Epoch", ylabel="MSE", title="Train-Val Gap (MSE)")
    axes[1, 1].set(xlabel="Epoch", ylabel="MAE", title="Train-Val Gap (MAE)")
 
    for ax in axes.flat:
        ax.legend()
        ax.grid(True)
 
    plt.suptitle("LSTM Baseline Training — All Horizons", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
 
 
def run_experiment(run_dir, config):
    """
    Train and evaluate an LSTM baseline for one (lookback, forecast) pair.
    """
    print("\n" + "=" * 60)
    print(f"LSTM Baseline  {config['lookback']}→{config['forecast']}")
    print("=" * 60)
 
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
 
    naive_results = NaiveBaseline().evaluate(test_ds)
    naive_mae, naive_mse = naive_results['MAE'], naive_results['MSE']
    print(f"Naive MAE: {naive_mae:.6f}  |  Naive MSE: {naive_mse:.6f}")
 
    sample_x, _ = train_ds[0]
    num_series = sample_x.shape[0]
 
    model = LSTMModel(
        num_series=num_series,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        pred_len=config["forecast"],
        dropout=config.get("dropout", 0.2)
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
        weight_decay=config.get("weight_decay", 0)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )
    criterion = torch.nn.MSELoss()
 
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])
 
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
 
    best_val = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(run_dir, f"best_model_{config['lookback']}to{config['forecast']}.pth")
    
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
            optimizer.step()
            epoch_loss += loss.item()
            epoch_mae_sum += torch.mean(torch.abs(output - y)).item()
 
        train_losses.append(epoch_loss / len(train_loader))
        train_maes.append(epoch_mae_sum / len(train_loader))
 
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
        scheduler.step(val_losses[-1])
 
        print(f"Epoch {epoch + 1:3d}/{config['epochs']}  "
              f"Train MSE: {train_losses[-1]:.6f}   Val MSE: {val_losses[-1]:.6f}  "
              f"Train MAE: {train_maes[-1]:.6f}   Val MAE: {val_maes[-1]:.6f}  "
              f"LR : {scheduler.optimizer.param_groups[0]['lr']:.2e}")
 
        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break
 
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
        "train_end_date": train_ds.end_date,
        "val_start_date": val_ds.start_date,
        "val_end_date": val_ds.end_date,
        "test_start_date": test_ds.start_date,
        "test_end_date": test_ds.end_date,
    }
 
 
def main():
    base_run_dir = "runs"
    os.makedirs(base_run_dir, exist_ok=True)
 
    config = {
        "data_path": "/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/vola-bert/data/processed/ES_1h.parquet",
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.4,
        "batch_size": 128,
        "epochs": 100,
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
 
    # Create a temporary dataset to get date ranges
    temp_ds = Dataset_SP500_1H(
        data_path=config["data_path"],
        events_df=None,
        flag="train",
        size=(24, 5),
        use_events=False,
        use_explainable=False,
        use_technical=True,
        use_interday=True,
    )
    
    train_end = temp_ds.end_date.strftime("%Y-%m-%d")
    run_dir_name = f"lstm_baseline_{train_end}"
    run_dir = os.path.join(base_run_dir, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)
 
    print(f"\n{'='*70}")
    print(f"  Results will be saved to: {run_dir}/")
    print(f"{'='*70}")
 
    all_results = []
    for horizon in horizons:
        result = run_experiment(run_dir, {**config, **horizon})
        all_results.append(result)
 
    plot_path = os.path.join(run_dir, "lstm_training_curves.png")
    save_loss_plot(all_results, plot_path)
    print(f"Loss curves saved to {plot_path}")
 
    # Save results summary
    results_path = os.path.join(run_dir, "lstm_results.txt")
    with open(results_path, "w") as f:
        f.write("LSTM Baseline — Results Summary\n")
        f.write("=" * 70 + "\n\n")
        
        # Write data split information
        f.write("DATA SPLIT INFORMATION:\n")
        f.write("-" * 70 + "\n")
        for r in all_results:
            f.write(f"Horizon: {r['lookback']}h → {r['forecast']}h\n")
            f.write(f"  Train: {r['train_end_date']}\n")
            f.write(f"  Val:   {r['val_start_date']} → {r['val_end_date']}\n")
            f.write(f"  Test:  {r['test_start_date']} → {r['test_end_date']}\n\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("RESULTS:\n")
        f.write("=" * 70 + "\n\n")
        
        for r in all_results:
            f.write(f"Horizon: {r['lookback']}h → {r['forecast']}h\n")
            f.write(f"  Naive MAE: {r['naive_mae']:.6f}   MSE: {r['naive_mse']:.6f}\n")
            f.write(f"  Model MAE: {r['model_mae']:.6f}   MSE: {r['model_mse']:.6f}\n")
            f.write(f"  rMAE: {r['rmae']:.4f}   rMSE: {r['rmse']:.4f}\n")
            if r['rmae'] < 1.0:
                f.write(f"  LSTM is {(1 - r['rmae']) * 100:.1f}% better than naive\n")
            f.write("\n")
    
    print(f"Results saved to {results_path}")
 
    print("\n" + "=" * 70)
    print("FINAL RESULTS:")
    print("=" * 70)
    for r in all_results:
        print(f"\nHorizon: {r['lookback']}h → {r['forecast']}h")
        print(f"  Model MAE: {r['model_mae']:.6f}   |   RMSE: {np.sqrt(r['model_mse']):.6f}")
        print(f"  Naive MAE: {r['naive_mae']:.6f}   |   rMAE: {r['rmae']:.4f}")
 
 
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
