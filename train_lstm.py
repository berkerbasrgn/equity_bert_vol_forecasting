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

from src.mydataset import Dataset_SP500_1H
from src.model_lstm import LSTMModel
####### from src.trainer import Trainer


class NaiveBaseline:
    def evaluate(self, dataset):
        predictions = []
        targets = []

        for i in range(len(dataset)):
            x, y = dataset[i] #(x,y) no tokens for LSTM datasets

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
            "MAE": torch.mean(torch.abs(predictions - targets)).item(),
            "MSE": torch.mean((predictions - targets) ** 2).item()
        }


#  METRICS 

def relative_metrics(model_mae, model_mse, naive_mae, naive_mse):
    return model_mae / naive_mae, model_mse / naive_mse

def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Baseline - Training Curves")
    plt.legend()
    plt.grid(True)  
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

#  TRAIN 

def run_experiment(run_dir, config):
    """
    Train and evaluate an LSTM baseline for one (lookback, forecast) pair.
 
    Unlike EquityBERT, the LSTM uses a simple manual training loop rather
    than the Trainer class, because:
      - No semantic tokens → no input_to_device dispatch needed
      - MSE loss (not MAE) is more standard for LSTM baselines
      - No OneCycleLR — plain Adam is sufficient for this model size
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

    #  BASELINE
    naive_results = NaiveBaseline().evaluate(test_ds)

    naive_mae, naive_mse = naive_results['MAE'], naive_results['MSE']

    print(f"Naive MAE: {naive_mae:.6f}  |  Naive MSE: {naive_mse:.6f}")

    #  MODEL
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

    #  TRAIN LOOP
    best_val = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(run_dir, "best_model.pth")
    print("\nTraining LSTM...")
    train_maes = []
    val_maes = []
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            epoch_mae = torch.mean(torch.abs(output - y)).item()
            epoch_loss += loss.item()
            epoch_mae_sum += torch.mean(torch.abs(output - y)).item()

        train_losses.append(epoch_loss / len(train_loader))
        train_maes.append(epoch_mae_sum / len(train_loader))

        # validation
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
        # step the scheduler with the latest validation loss
        scheduler.step(val_losses[-1])


        print(f"Epoch {epoch + 1:3d}/{config['epochs']}  "
              f"Train MSE: {train_losses[-1]:.6f}   Val MSE: {val_losses[-1]:.6f}  "
              f"Train MAE: {train_maes[-1]:.6f}   Val MAE: {val_maes[-1]:.6f}  "
              f"LR : {scheduler.optimizer.param_groups[0]['lr']:.2e}")
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


    #  TEST
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

    }


def main():
    run_dir = "runs/lstm_baseline"
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "data_path": "/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/vola-bert/data/processed/ES_1h.parquet",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "batch_size": 64,
        "epochs": 100,
        "lr": 5e-4,
        "weight_decay": 1e-4,
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


    print("\nFINAL RESULTS:")
    print(all_results)


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
