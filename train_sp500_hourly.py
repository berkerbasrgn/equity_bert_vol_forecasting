"""
Training script for EquityBERT on S&P 500 HOURLY DATA
Following ICAIF 2025 Paper: "Repurposing Language Models for FX Volatility Forecasting"

This script implements:
1. Multiple forecast horizons: (40→5), (50→10), (60→20)
2. Naive baseline comparison (last-value persistence)
3. Data-scarce scenario (10% of training data)
4. rMAE and rMSE metrics (relative to naive baseline)

The goal is to prove EquityBERT's effectiveness on S&P 500 data,
not just FX data as in the original paper.
"""

import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt
from datetime import datetime

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "src"))

# Import model components
from src.mydataset import Dataset_SP500_1H 
from src.model_bert import EquityBERT
from src.model_lstm import LSTMModel
from src.trainer import Trainer

# ================== NAIVE BASELINE (Critical for Paper Comparison) ==================

class NaiveBaseline:
    """
    Naive baseline: persistence model that predicts the last observed volatility
    
    This is the baseline used in the paper (Table 2 and Table 3).
    rMAE and rMSE are calculated by dividing model errors by naive baseline errors.
    
    The naive baseline serves as a sanity check:
    - If rMAE < 1.0: Model is better than naive
    - If rMAE > 1.0: Model is worse than naive (problem!)
    """
    
    def __init__(self):
        self.name = "Naive (Last Value)"
    
    def evaluate(self, dataset):
        """Evaluate naive baseline on the given dataset."""
    
        predictions = []
        targets = []
    
        print(f"\nEvaluating {self.name} on {len(dataset)} samples...")
    
        for i in range(len(dataset)):
            # (seq_x, tokens), seq_y = dataset[i]
            seq_x, seq_y = dataset[i]  # Assuming dataset returns (seq_x, tokens), seq_y but we only need seq_x and seq_y for naive baseline
        
        # Last volatility (scaled) - first feature, last timestep
            last_vola = seq_x[0, -1].item()  # Assuming first feature is volatility and it's scaled
        
        # Repeat for forecast horizon
            horizon = seq_y.shape[1]
            naive_pred = torch.full((1, horizon), last_vola)
        
            predictions.append(naive_pred)
            targets.append(seq_y)
    
    # Stack
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

                # Calculate metrics
        mae = torch.mean(torch.abs(predictions - targets)).item()
        mse = torch.mean((predictions - targets) ** 2).item()

        print(f"  MAE: {mae:.6f}")
        print(f"  MSE: {mse:.6f}")

        return {"MAE": mae, "MSE": mse}


# ================== EVALUATION METRICS ==================

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

# events_df = pd.read_csv("/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/vola-bert/data/macro_events_us.csv")

# events_df = events_df.rename(columns={
#     "event_time_et": "datetime"
# })

# events_df["datetime"] = pd.to_datetime(events_df["datetime"], utc=True).dt.tz_convert("America/New_York")
# ================== TRAINING FUNCTION ==================

def train_volabert_single_config(config, run_dir, data_scenario="full", ablation_name="default"):
    """
    Train Vola-BERT for a single configuration (lookback, forecast)
    
    Args:
        config: Dictionary with lookback, forecast, and other hyperparameters
        data_scenario: "full" (100% training data) or "scarce" (10% training data)
        
    Returns:
        Dictionary with results including model and naive baseline metrics
    """
    
    print("\n" + "=" * 80)
    print(f"Training Configuration: {config['lookback']}→{config['forecast']}")
    print(f"Data Scenario: {data_scenario.upper()}")
    print("=" * 80)
    
    # Determine fine-tuning percentage based on scenario
    fine_tuning_pct = None if data_scenario == "full" else 0.1
    
    # ========== STEP 1: Load Datasets ==========
    print("\nLoading datasets...")
    
    train_dataset = Dataset_SP500_1H(
        data_path=config["data_path"],
        events_df=None,  # Assuming events_df is defined elsewhere
        flag="train",
        size=(config["lookback"], config["forecast"]),
#features=config["features"],
#use_technical=config["use_technical"],
        use_events=config["use_events"],
        use_event_type=config["use_event_type"],
        use_event_impact=config["use_event_impact"],
        use_explainable=False,
#use_interday=config["use_interday"],
#use_explainable=config["use_explainable"],
        fine_tuning_pct=fine_tuning_pct,  # None for full, 0.1 for scarce
       # scaler=None,
       # use_volume_state=config["use_volume_state"],
       mode="24h",  # Use 24h mode to keep all hours (including overnight)
    )
    
    # Get shared scaler from training set
    #shared_scaler = train_dataset.scaler
    
    val_dataset = Dataset_SP500_1H(
        data_path=config["data_path"],
        flag="val",
        size=(config["lookback"], config["forecast"]),
        events_df=None,  # Assuming events_df is defined elsewhere
       # features=config["features"],
       # use_technical=config["use_technical"],
        use_events=config["use_events"],
        use_event_type=config["use_event_type"],
        use_event_impact=config["use_event_impact"],
        use_explainable=False,
       # use_interday=config["use_interday"],
       # use_explainable=config["use_explainable"],
        fine_tuning_pct=None,
        #scaler=shared_scaler,
       # use_volume_state=config["use_volume_state"],
        mode="24h",  # Use 24h mode to keep all hours (including overnight
    )
    
    test_dataset = Dataset_SP500_1H(
        data_path=config["data_path"],
        flag="test",
        size=(config["lookback"], config["forecast"]),
        events_df=None,  # Assuming events_df is defined elsewhere
        #features=config["features"],
        #use_technical=config["use_technical"],
        use_events=config["use_events"],
        use_event_type=config["use_event_type"],
        use_event_impact=config["use_event_impact"],
        use_explainable=False,
        #use_interday=config["use_interday"],
        #use_explainable=config["use_explainable"],
        fine_tuning_pct=None,
        #scaler=shared_scaler,
        #use_volume_state=config["use_volume_state"],
        mode="24h",  # Use 24h mode to keep all hours (including overnight
    )
    # start_date = min(train_dataset.start_date, val_dataset.start_date, test_dataset.start_date)
    # end_date = max(train_dataset.end_date, val_dataset.end_date, test_dataset.end_date)
    # print(f"Overall date range: {start_date} to {end_date}")
    # print(f"Train: {len(train_dataset)} samples")
    # print(f"Train date range: {train_dataset.start_date} to {train_dataset.end_date}")
    # print(f"Val:   {len(val_dataset)} samples")
    # print(f"Val date range: {val_dataset.start_date} to {val_dataset.end_date}")
    # print(f"Test:  {len(test_dataset)} samples")
    # print(f"Test date range: {test_dataset.start_date} to {test_dataset.end_date}")
    # #sys.exit(0)
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")
    print(f"Test:  {len(test_dataset)} samples")
    # ========== STEP 2: Evaluate Naive Baseline ==========
    print("\n" + "-" * 80)
    print("Evaluating Naive Baseline")
    print("-" * 80)
    
    naive_model = NaiveBaseline()
    naive_results = naive_model.evaluate(test_dataset)
    
    # ========== STEP 3: Setup Vola-BERT Model ==========
    print("\n" + "-" * 80)
    print("Setting up EquityBERT Model")
    print("-" * 80)
    
    # Infer number of input features from sample
    sample_x, sample_y = train_dataset[0]
    if isinstance(sample_x, tuple):
        num_series = sample_x[0].shape[0]
    else:
        num_series = sample_x.shape[0]
    
    print(f"Input features: {num_series}")
    print(f"Lookback: {config['lookback']} hours")
    print(f"Forecast: {config['forecast']} hours")
    print(f"BERT layers: {config['n_layer']}")

    # Semantic tokens configuration
    semantic_tokens = {
        "market_session": 4,  # None, Early, Mid, Late
    }
    if config["use_event_type"]:
        semantic_tokens["event_type"] = 5  # None, FOMC, NFP, CPI, Unemployment
    if config["use_event_impact"]:
        semantic_tokens["event_impact"] = 4  # None, Low, Medium, High
    # # # Create model
    # # model = EquityBERT(
    # #     num_series=num_series,
    # #     input_len=config["lookback"],
    # #     pred_len=config["forecast"],
    # #     n_layer=config["n_layer"],
    # #     revin=True,
    # #     semantic_tokens=semantic_tokens,
    # )

    model = LSTMModel(
        num_series=num_series,
        hidden_size=128,
        num_layers=2,
        pred_len=config["forecast"]
)
    
    print(f"\n✓ Model parameters:")
    # print(f"  Total:     {model.num_params['total']:,}")
    print(f"  Trainable: {model.num_params['trainable']:,}")
    
    # ========== STEP 4: Train Model ==========
    print("\n" + "-" * 80)
    print("Training EquityBERT")
    print("-" * 80)
    
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
        num_workers=2,  # Use 2 workers for data loading (adjust based on your CPU)
    )

    train_mae, val_mae, train_mse, val_mse = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config["batch_size"],
        max_epochs=config["max_epochs"],
        lr=config["lr"],
    )
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_mae, label="Train MAE")
    plt.plot(val_mae, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title(f"MAE Curves - {config['lookback']}→{config['forecast']})")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_mse, label="Train MSE")
    plt.plot(val_mse, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"MSE Curves - {config['lookback']}→{config['forecast']})")
    plt.legend()
    plt.grid()

    plot_path = os.path.join(checkpoint_dir, f"loss_{scenario_name}.png")
    
    plt.savefig(plot_path)
    plt.close()
    print(f"\n Training complete. Loss curves saved to {plot_path}")


    print("Overfitting Check:")
    print(f"  Final Train MAE: {train_mae[-1]:.6f}")
    print(f"  Final Val MAE:   {val_mae[-1]:.6f}")
    if val_mae[-1] > min(val_mae):
        print("  Warning: Validation MAE increased at the end of training. Possible overfitting.")
    else:
        print("   No  overfitting detected.")
    print(f"  Final Train MSE: {train_mse[-1]:.6f}")
    print(f"  Final Val MSE:   {val_mse[-1]:.6f}")
    if val_mse[-1] > min(val_mse):
        print("  Warning: Validation MSE increased at the end of training. Possible overfitting.")

    
    # ========== STEP 5: Test Model ==========
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

    # ========== STEP 6: Calculate Relative Metrics ==========
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

    # ========== Return Results ==========
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
    }
    
    return results


# ================== MAIN ENTRY POINT ==================

def main():
    """
    Main training pipeline
    
    Trains Vola-BERT on multiple horizons following the paper:
    - (40→5): Short-term forecasting
    - (50→10): Medium-term forecasting
    - (60→20): Long-term forecasting
    
    For each horizon, tests both:
    - Full data scenario (100% of training data)
    - Data-scarce scenario (10% of training data)
    """
    # run versioning

    base_runs_dir = "runs"
    os.makedirs(base_runs_dir, exist_ok=True)

    #mevcut runlari say
    existing = [d for d in os.listdir(base_runs_dir) if d.startswith("v")]
    run_id = len(existing) + 1
    run_dir = os.path.join(base_runs_dir, f"v{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nsaving results to {run_dir}")
    print("\n" + "=" * 80)
    print("EquityBERT Training on S&P 500 Hourly Data")
    print("Following ICAIF 2025 Paper Methodology")
    print("=" * 80)
    
    # Base configuration (shared across all experiments)
    base_config = {
        "data_path": "/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/vola-bert/data/processed/ES_1h.parquet",
        "root_path": current_dir,
        "n_layer": 4,               # BERT layers
        "batch_size": 32,
        "max_epochs": 50,
        "lr": 1e-4,
        "patience": 10,
        "features": "MS",           # Multivariate to univariate
        "use_technical": True,
        "use_events": False,
        "use_event_type": False,
        "use_event_impact": False,
        "use_interday": True,
        "use_explainable": True,
        "use_volume_state": False,
    }
    
    # Horizons to test (matching paper Table 2 and Table 3)
    horizons = [
        {"lookback": 24, "forecast": 5},    # Short-term
        #{"lookback": 50, "forecast": 10},   # Medium-term
        #{"lookback": 60, "forecast": 20},   # Long-term
    ]
    # ablation_configs = [
    #     {
    #         "name": "No Events",
    #         "use_events": False,
    #         "use_event_type": False,
    #         "use_event_impact": False,
    #     },
    #     {
    #         "name": "Event Type Only",
    #         "use_events": True,
    #         "use_event_type": True,
    #         "use_event_impact": False,
    #     },
    #     {
    #         "name": "FULL EVENTS",
    #         "use_events": True,
    #         "use_event_type": True,
    #         "use_event_impact": False,
    #     },
    #     {
    #         "name": "Improved Timing Events",
    #         "use_events": True,
    #         "use_event_type": False,
    #         "use_event_impact": False,
    #     }
    # ]

    ablation_configs = [
    {
        "name": "LSTM_BASE",
        "use_events": False,
        "use_event_type": False,
        "use_event_impact": False,
    }
]

    # Store all results
    all_results = []
    
    # Train on each horizon and data scenario
    for horizon in horizons:
        for ablation in ablation_configs:
            print("\n" + "=" * 80)
            print(f"Running Ablation: {ablation['name']} for Horizon: {horizon['lookback']}→{horizon['forecast']}")
            print("=" * 80)
        # Merge horizon into base config
            config = {**base_config, **horizon, **ablation}
        
        # Full-data scenario
            results_full = train_volabert_single_config(config, run_dir, data_scenario="full"
                                                        , ablation_name=ablation["name"])
            all_results.append(results_full)
        
        # Data-scarce scenario
        # print("DATA-SCARCE SCENARIO (10% training data)")
        # results_scarce = train_volabert_single_config(config, data_scenario="scarce")
        # all_results.append(results_scarce)
    
    # ========== Print Summary Table ==========
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print("=" * 80)
    
    print("\nResults saved to: vola_bert_sp500_results.txt")
    
    # Save detailed results
    results_path = os.path.join(run_dir, f"run_version_{run_id}_equity_bert_sp500_results.txt")
    with open(results_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EquityBERT on S&P 500 Hourly Data - Results Summary\n")
        f.write("=" * 80 + "\n\n")
        
        for result in all_results:
            f.write(f"\nConfiguration: {result['config']['lookback']}→{result['config']['forecast']}\n")
            f.write(f"Ablation: {result.get('ablation_name')}\n")
            f.write(f"Data Scenario: {result['data_scenario'].upper()}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Naive Baseline MAE: {result['naive_mae']:.6f}\n")
            f.write(f"Naive Baseline MSE: {result['naive_mse']:.6f}\n")
            f.write(f"EquityBERT MAE:      {result['model_mae']:.6f}\n")
            f.write(f"EquityBERT MSE:      {result['model_mse']:.6f}\n")
            f.write(f"rMAE:               {result['rmae']:.4f}\n")
            f.write(f"rMSE:               {result['rmse']:.4f}\n")
            f.write(f"Checkpoint:         {result['checkpoint_dir']}/\n")
            f.write("\n")
    
    print("\nAll experiments completed successfully!")
    print("Check 'vola_bert_sp500_results.txt' for detailed results")
    print("Model checkpoints saved in 'checkpoints_sp500_*' directories")
    
    #  Export to CSV #
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(run_dir, f"run_version_{run_id}_equity_bert_sp500_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved to: {csv_path}")

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