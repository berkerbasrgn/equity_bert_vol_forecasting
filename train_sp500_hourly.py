
"""
Training script for EquityBERT on S&P 500 HOURLY DATA
Following ICAIF 2025 Paper: "Repurposing Language Models for FX Volatility Forecasting"
This script:
 1. Loads the pre-processed ES.FUT hourly Parquet via Dataset_SP500_1H
 2. Evaluates a naive (last-value persistence) baseline on the test set
 3. Trains EquityBERT with semantic tokens (market session, event type/impact)
 4. Computes rMAE and rMSE relative to the naive baseline
 5. Saves loss curves, checkpoints, and a results summary

The goal is to prove EquityBERT's effectiveness on S&P 500 data,
not just FX data as in the original paper.
The script creates a versioned run directory under runs/ (e.g. runs/v1/)
containing checkpoints, loss plots, and results files.

"""

import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "src"))

# Import model components
from src.mydataset import Dataset_SP500_1H, SEMANTIC_TOKEN_VOCAB
from src.model_bert import EquityBERT
from src.trainer import Trainer

#  NAIVE BASELINE (Critical for Paper Comparison) 

class NaiveBaseline:
    """
    Last-value persistence baseline: predicts the most recent observed
    volatility for all future steps.
 
    This is the denominator in the rMAE / rMSE metrics used in the paper.
    rMAE < 1.0 means the model beats naive; rMAE > 1.0 means it's worse.
    """
    def __init__(self):
        self.name = "Naive Baseline"

    def evaluate(self, dataset):
        """Evaluate on a dataset that returns ((x, tokens), y) or (x, y).
 
        Uses the last value of the TARGET feature (last column before target
        split = the last value of r in the input window) as the naive forecast.
        Because the dataset places the target 'r' last and __getitem__ slices
        seq_x = data[:, :-1], the target's lagged values are NOT in seq_x.
        Instead, I use the last value of the first feature as a proxy for
        the most recent scaled observation , matching the original Vola-BERT
        naive baseline implementation.
 
        Returns:
            dict with 'MAE' and 'MSE' keys
        """
    
        predictions = []
        targets = []
    
        print(f"\nEvaluating {self.name} on {len(dataset)} samples...")
    

        # for i in range(len(dataset)):
        #     # (seq_x, tokens), seq_y = dataset[i]
        #     seq_x, seq_y = dataset[i]  # Assuming dataset returns (seq_x, tokens), seq_y but we only need seq_x and seq_y for naive baseline
        
        # Last volatility (scaled) - first feature, last timestep
            # last_vola = seq_x[0, -1].item()  # Assuming first feature is volatility and it's scaled
        
        # Repeat for forecast horizon
        for i in range(len(dataset)):
            sample = dataset[i]
            #handle both ((seq_x, tokens), seq_y) and (seq_x, seq_y)
            if isinstance(sample[0], tuple):
                (x, y) = sample[0][0], sample[1]
            else:
                x, y = sample[0], sample[1]

            horizon = y.shape[1]
            last_vola = x[0, -1].item()  # Assuming first feature is volatility and it's scaled
            naive_pred = torch.full((1, horizon), last_vola)
        
            predictions.append(naive_pred)
            targets.append(y)

    # Stack
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

                # Calculate metrics
        mae = torch.mean(torch.abs(predictions - targets)).item()
        mse = torch.mean((predictions - targets) ** 2).item()

        print(f"  MAE: {mae:.6f}")
        print(f"  MSE: {mse:.6f}")

        return {"MAE": mae, "MSE": mse}


#  EVALUATION METRICS 

def calculate_relative_metrics(model_mae, model_mse, naive_mae, naive_mse):
    """
    Calculate rMAE and rMSE as used in the paper
    
    rMAE = model_MAE / naive_MAE
    rMSE = model_MSE / naive_MSE
    
    Values < 1.0 indicate the model outperforms naive baseline
    The paper's Vola-BERT achieves rMAE around 0.65-0.70 (35-30% better than naive)
    """
    rmae = model_mae / naive_mae
    rmse = model_mse / naive_mse
    
    return rmae, rmse

#  TRAINING FUNCTION 

def train_equitybert_single_config(config, run_dir, data_scenario="full", ablation_name="default"):
    """
    Train EquityBERT for a single configuration (lookback, forecast)

    Args:
        config        : dict with lookback, forecast, and other hyperparameters
        run_dir       : versioned output directory
        data_scenario : "full" (100% training data) or "scarce" (10%)
        ablation_name : label for this ablation variant

    Returns:
        dict with all metrics and paths
    """
    
    print("\n" + "=" * 80)
    print(f"Training Configuration: {config['lookback']}→{config['forecast']}")
    print(f"Data Scenario: {data_scenario.upper()}")
    print(f"Ablation: {ablation_name}")
    print("=" * 80)

    fine_tuning_pct = None if data_scenario == "full" else 0.1

    #  Load Datasets 
    print("\nLoading datasets...")

    common_kwargs = dict(
        data_path=config["data_path"],
        events_df=config.get("events_df"),
        size=(config["lookback"], config["forecast"]),
        use_events=config["use_events"],
        use_event_type=config["use_event_type"],
        use_event_impact=config["use_event_impact"],
        use_explainable=True,
        mode=config.get("mode", "24h"),
    )

    train_dataset = Dataset_SP500_1H(flag="train", fine_tuning_pct=fine_tuning_pct, **common_kwargs)
    val_dataset   = Dataset_SP500_1H(flag="val",   **common_kwargs)
    test_dataset  = Dataset_SP500_1H(flag="test",  **common_kwargs)

    print(f"Train: {len(train_dataset):,} samples  ({train_dataset.start_date} → {train_dataset.end_date})")
    print(f"Val:   {len(val_dataset):,} samples  ({val_dataset.start_date} → {val_dataset.end_date})")
    print(f"Test:  {len(test_dataset):,} samples  ({test_dataset.start_date} → {test_dataset.end_date})")

    #  Evaluate Naive Baseline
    print("\n" + "-" * 80)
    print("Evaluating Naive Baseline")
    print("-" * 80)

    naive_model = NaiveBaseline()
    naive_results = naive_model.evaluate(test_dataset)

    # Setup EquityBERT Model 
    print("\n" + "-" * 80)
    print("Setting up EquityBERT Model")
    print("-" * 80)

    sample = train_dataset[0]
    num_series = sample[0][0].shape[0]

    print(f"Input features: {num_series}")
    print(f"Lookback: {config['lookback']} hours")
    print(f"Forecast: {config['forecast']} hours")
    print(f"BERT layers: {config['n_layer']}")

    semantic_tokens = {
        "market_session": SEMANTIC_TOKEN_VOCAB["market_session"],
    }
    if config["use_event_type"]:
        semantic_tokens["event_type"] = SEMANTIC_TOKEN_VOCAB["event_type"]
    if config["use_event_impact"]:
        semantic_tokens["event_impact"] = SEMANTIC_TOKEN_VOCAB["event_impact"]

    model = EquityBERT(
        num_series=num_series,
        input_len=config["lookback"],
        pred_len=config["forecast"],
        n_layer=config["n_layer"],
        revin=True,
        semantic_tokens=semantic_tokens,
    )

    params = model.num_params
    print(f"\nModel parameters:")
    print(f"  Total:     {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")

    print("=" * 80)
    print("Training EquityBERT")
    print("-" * 80)

    #  Train Model 
 
    
    use_amp = torch.cuda.is_available()
    scenario_name = f"{ablation_name}_{config['lookback']}to{config['forecast']}_{data_scenario}"
    checkpoint_dir = os.path.join(
        run_dir, scenario_name, "checkpoints"
    )

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    trainer = Trainer(
        model=model,
        use_amp=use_amp,
        features=config["features"],
        inverse=False,
        save_path=checkpoint_dir,
        patience=config["patience"],
        verbose=True,
        num_workers=2,  # Use 2 workers for data loading (adjust based on my CPU)
    )

    train_mae, val_mae, train_mse, val_mse = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config["batch_size"],
        max_epochs=config["max_epochs"],
        lr=config["lr"],
    )
     # Loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_mae, label="Train MAE")
    ax1.plot(val_mae, label="Val MAE")
    ax1.set(xlabel="Epoch", ylabel="MAE",
            title=f"MAE — {config['lookback']}→{config['forecast']}")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_mse, label="Train MSE")
    ax2.plot(val_mse, label="Val MSE")
    ax2.set(xlabel="Epoch", ylabel="MSE",
            title=f"MSE — {config['lookback']}→{config['forecast']}")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, f"loss_{scenario_name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nTraining complete. Loss curves saved to {plot_path}")

    # Overfitting check
    print("\nOverfitting Check:")
    print(f"  Final Train MAE: {train_mae[-1]:.6f}")
    print(f"  Final Val MAE:   {val_mae[-1]:.6f}")
    if val_mae[-1] > min(val_mae):
        print("  Warning: Validation MAE increased — possible overfitting.")
    else:
        print("  No overfitting detected.")


    
    # Test Model 
    print("\n" + "-" * 80)
    print("Evaluating EquityBERT on Test Set")
    print("-" * 80)
    
    model_mae = trainer.test(
        test_dataset=test_dataset,
        batch_size=config["batch_size"],
    )
    
    # Calculate MSE separately (trainer only returns MAE)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, num_workers=2)
    mse_list = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = trainer.input_to_device(batch_x)
            # Forward pass
            outputs = model(batch_x)
            batch_y = batch_y.to(outputs.device)
            # Calculate MSE
            mse = torch.mean((outputs - batch_y) ** 2)
            mse_list.append(mse.item())
    
    model_mse = np.mean(mse_list)

    print(f"\nEquityBERT Test MAE: {model_mae:.6f}")
    print(f"EquityBERT Test MSE: {model_mse:.6f}")

    #  Calculate Relative Metrics 
    print("\n" + "-" * 80)
    print("Calculating Relative Metrics (vs Naive Baseline)")
    print("-" * 80)
    
    rmae, rmse = calculate_relative_metrics(
        model_mae, model_mse,
        naive_results["MAE"], naive_results["MSE"]
    )
    
    print(f"\nrMAE = {rmae:.4f} (lower is better)")
    print(f"rMSE = {rmse:.4f} (lower is better)")

    if rmae < 1.0:
        improvement = (1 - rmae) * 100
        print(f"\nEquityBERT is {improvement:.1f}% better than naive baseline!")
    else:
        degradation = (rmae - 1) * 100
        print(f"\nEquityBERT is {degradation:.1f}% worse than naive baseline")

    #  Return Results 
    results = {
        "config": config,
        "data_scenario": data_scenario,
        "naive_mae": naive_results["MAE"],
        "naive_mse": naive_results["MSE"],
        "model_mae": model_mae,
        "model_mse": model_mse,
        "rmae": rmae,
        "rmse": rmse,
        "checkpoint_dir": checkpoint_dir,
        "ablation_name": ablation_name,
        "train_mae_history": train_mae,
        "val_mae_history": val_mae,
        "train_mse_history": train_mse,
        "val_mse_history": val_mse,
    }
    
    return results


#  MAIN ENTRY POINT 

def main():
    """
    Main training pipeline
    
    Trains Vola-BERT on multiple horizons following the paper:
    - (40→5): Short-term forecasting
    Ablation configs loop over event feature combinations.
    
    For each horizon, tests both:
    - Full data scenario (100% of training data)
    """
    # run versioning

    base_runs_dir = "runs"
    os.makedirs(base_runs_dir, exist_ok=True)

    existing = [d for d in os.listdir(base_runs_dir) if d.startswith("v")]
    run_id = len(existing) + 1
    run_dir = os.path.join(base_runs_dir, f"v{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nsaving results to {run_dir}")
    print("\n" + "-" * 80)
    print("EquityBERT Training on S&P 500 Hourly Data")
    print("Following ICAIF 2025 Paper Methodology")
    print("*" * 80)
    
    # Base configuration (shared across all experiments)
    events_df = pd.read_csv("/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/vola-bert/NEW_macro_events_us.csv")
    events_df = events_df.rename(columns={"event_time_et": "datetime"})
    events_df["impact"]= "NONE"  # Placeholder impact since we don't have actual impact data for these events
    events_df["datetime"] = pd.to_datetime(
        events_df["datetime"], utc=True
    ).dt.tz_convert("America/New_York")

    base_config = {
        "data_path": "/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/vola-bert/data/processed/ES_1h.parquet",
        "root_path": current_dir,
        "events_df": events_df,
        "n_layer": 4,               # BERT layers
        "batch_size": 32,
        "max_epochs": 150,
        "lr": 1e-4,
        "patience": 10,
        "features": "MS",           # Multivariate to univariate
        # "use_technical": True,
        # "use_events": False,
        # "use_event_type": False,
        # "use_event_impact": False,
        # "use_interday": True,
        # "use_explainable": True,
        # "use_volume_state": False,
    }
    
    # Horizons to test (matching paper Table 2 and Table 3)
    horizons = [
        {"lookback": 24, "forecast": 5},    # Short-term
        {"lookback": 50, "forecast": 10},   # Medium-term
        #{"lookback": 60, "forecast": 20},   # Long-term
    ]
    ablation_configs = [
        {
            "name": "No Events",
            "use_events": False,
            "use_event_type": False,
            "use_event_impact": False,
        },
        {
            "name": "Event Type Only",
            "use_events": True,
            "use_event_type": True,
            "use_event_impact": False,
        },
        {
            "name": "Event Timing Only",
            "use_events": True,
            "use_event_type": False,
            "use_event_impact": False,
        }
    ]


    # Store all results
    all_results = []
    
    # Train on each horizon and data scenario
    for horizon in horizons:
        for ablation in ablation_configs:
            if ablation["use_events"] and events_df is None:
                print(f"\nSkipping {ablation['name']} for Horizon {horizon['lookback']}→{horizon['forecast']} because events_df is not available.")
                continue
            print("\n" + "=" * 80)
            print(f"Running Ablation: {ablation['name']} for Horizon: {horizon['lookback']}→{horizon['forecast']}")
            print("=" * 80)
        # Merge horizon into base config
            config = {**base_config, **horizon, **ablation}
        
        # Full-data scenario
            results_full = train_equitybert_single_config(
                config, run_dir,
                data_scenario="full",
                ablation_name=ablation["name"]
            )
            all_results.append(results_full)

    n = len(all_results)
    if n > 0:
        fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
        if n == 1:
            axes = axes.reshape(1, -1)
 
        for i, r in enumerate(all_results):
            label = f"{r['ablation_name']} {r['config']['lookback']}→{r['config']['forecast']}"
 
            axes[i, 0].plot(r["train_mae_history"], label="Train MAE", linewidth=0.8)
            axes[i, 0].plot(r["val_mae_history"], label="Val MAE", linewidth=0.8)
            axes[i, 0].set(xlabel="Epoch", ylabel="MAE", title=f"MAE — {label}")
            axes[i, 0].legend(fontsize=8)
            axes[i, 0].grid(True, alpha=0.3)
 
            axes[i, 1].plot(r["train_mse_history"], label="Train MSE", linewidth=0.8)
            axes[i, 1].plot(r["val_mse_history"], label="Val MSE", linewidth=0.8)
            axes[i, 1].set(xlabel="Epoch", ylabel="MSE", title=f"MSE — {label}")
            axes[i, 1].legend(fontsize=8)
            axes[i, 1].grid(True, alpha=0.3)
 
        plt.tight_layout()
        combined_path = os.path.join(run_dir, f"all_loss_curves_v{run_id}.png")
        plt.savefig(combined_path, dpi=150)
        plt.close()
        print(f"\nCombined loss curves saved to: {combined_path}")
        
    #  Summary Table
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print("=" * 80)
    
    print("\nResults saved to: vola_bert_sp500_results.txt")
    
    # Save detailed results
    results_path = os.path.join(run_dir, f"equitybert_results_v{run_id}.txt")
    with open(results_path, "w") as f:
        f.write("EquityBERT on S&P 500 Hourly Data — Results Summary\n")
        f.write("=" * 60 + "\n\n")
        for r in all_results:
            f.write(f"{r['config']['lookback']}→{r['config']['forecast']}  "
                    f"[{r['data_scenario']}]  Ablation: {r['ablation_name']}\n")
            f.write(f"  Naive MAE: {r['naive_mae']:.6f}   MSE: {r['naive_mse']:.6f}\n")
            f.write(f"  Model MAE: {r['model_mae']:.6f}   MSE: {r['model_mse']:.6f}\n")
            f.write(f"  rMAE: {r['rmae']:.4f}   rMSE: {r['rmse']:.4f}\n")
            f.write(f"  Checkpoint: {r['checkpoint_dir']}\n\n")
    print(f"Results TXT: {results_path}")
    csv_rows = []
    for r in all_results:
        csv_rows.append({
            "horizon": f"{r['config']['lookback']}→{r['config']['forecast']}",
            "ablation": r["ablation_name"],
            "data_scenario": r["data_scenario"],
            "naive_mae": r["naive_mae"],
            "naive_mse": r["naive_mse"],
            "model_mae": r["model_mae"],
            "model_mse": r["model_mse"],
            "rmae": r["rmae"],
            "rmse": r["rmse"],
        })

    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(run_dir, f"equitybert_results_v{run_id}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Results CSV: {csv_path}")

    return all_results



if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()