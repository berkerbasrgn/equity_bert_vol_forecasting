# EquityBERT: S&P 500 Volatility Forecasting

A PyTorch-based volatility forecasting framework for E-mini S&P 500 futures (ES.FUT) at hourly frequency, adapted from the original Vola-BERT model for FX rate forecasting.

## Overview

This project implements two complementary deep learning architectures for forecasting log-range volatility of ES.FUT hourly bars:

- **EquityBERT**: Transformer-based model with semantic token embeddings for market regime awareness
- **LSTM Baseline**: Recurrent neural network baseline for comparison

The model ingests:
- Technical indicators (Bollinger Bands, RSI, EMA, momentum)
- Market session tokens (overnight, pre-market, regular, after-hours)
- US macro event signals (FOMC, CPI, PPI, NFP)
- Lagged volatility features (1h, 2h, 4h, 8h, 24h)

**Target variable**: Log-range volatility `r_t = ln(H_t / L_t)`, the Parkinson (1980) volatility estimator

## Project Structure

```
vola-bert/
├── data/
│   ├── raw/                              # Raw Databento Parquet (output of download_databento.py)
│   │   ├── ES_1h.parquet
│   │   └── ES_1min.parquet
│   ├── processed/
│   │   ├── ES_1h.parquet                 # Preprocessed 1-hour OHLCV bars
│   ├── NEW_macro_events.csv              # US macro event calendar (output of macro_event_builder.py)
│   └── dataset_nb.ipynb                  # Exploratory notebook for dataset inspection
├── data_preprocessing/
│   ├── calendar_merging.ipynb            # Macro event calendar construction notebook
│   └── firstrate_data_processing.ipynb  # Alternative data source processing notebook
├── src/
│   ├── __init__.py
│   ├── model_bert.py                     # EquityBERT (BERT-based) architecture
│   ├── model_lstm.py                     # LSTM baseline model
│   ├── mydataset.py                      # PyTorch Dataset class with token generation
│   ├── preprocess_data.py                # Raw → processed Parquet pipeline
│   ├── trainer.py                        # Training and evaluation loop (EquityBERT)
│   ├── utils.py                          # StandardScaler, EarlyStopping
│   ├── loss.py                           # MAE / MSE loss functions
│   ├── plot_utils.py                     # Visualisation helpers
│   ├── download_databento.py             # Databento data download script
│   └── macro_event_builder.py            # US macro event calendar builder (FRED API)
├── checkpoints_sp500_24to5_full/         # Saved EquityBERT checkpoints (root-level)
├── train_sp500_hourly.py                 # EquityBERT training entry point (with ablation study)
├── train_lstm.py                         # LSTM baseline training entry point
├── evaluate_lstm.py                      # LSTM inference, Excel export, and plots
├── pyproject.toml                        # Project metadata and dependencies (uv)
├── uv.lock                               # Locked dependency graph (committed for reproducibility)
└── README.md
```

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and sync

```bash
git clone <repository_url>
cd vola-bert
uv sync
```

This creates a `.venv`, resolves the locked dependency graph from `uv.lock`, and installs everything. No separate pip step needed.

If you also need to download raw data (first-time setup), install the optional data extras:

```bash
uv sync --extra data
```

**Key dependencies** (managed via `pyproject.toml`):
- PyTorch 2.0+
- HuggingFace `transformers` (for `bert-base-uncased`)
- pandas, numpy, pyarrow, scikit-learn
- `databento` + `fredapi` (optional, data download only)

### 3. Environment Variables

Create a `.env` file in the project root:

```
FRED_API_KEY=your_fred_api_key_here
```

A free FRED API key can be obtained at https://fred.stlouisfed.org/docs/api/api_key.html

### 4. Run scripts

Prefix any script with `uv run` to use the managed environment:

```bash
uv run python train_sp500_hourly.py
uv run python train_lstm.py
uv run python evaluate_lstm.py
```

## Data Preparation
####
### Prerequisites

The project expects preprocessed data at `data/processed/ES_1h.parquet` with the following structure:

| Column   | Type     | Description                      |
|----------|----------|----------------------------------|
| Datetime | datetime | Timezone-aware timestamp (ET)    |
| Open     | float    | Opening price                    |
| High     | float    | Session high                     |
| Low      | float    | Session low                      |
| Close    | float    | Closing price                    |
| Volume   | int      | Trade volume or count            |

### Running Preprocessing

Download raw data from Databento, then run preprocessing:

```bash
python src/download_databento.py          # produces data/raw/ES_1h.parquet
python src/preprocess_data.py             # produces data/processed/ES_1h.parquet
```

**What preprocessing does**:
1. Resets the `ts_event` DatetimeIndex to a `Datetime` column
2. Removes calendar-spread instruments (symbols containing `-`)
3. Renames lowercase OHLCV columns to title-case
4. Keeps only the most liquid contract per timestamp (highest volume)
5. Drops bars with `Low < 1000` or `High > 10000` (data errors / placeholder rows)
6. Outputs cleaned Parquet to `data/processed/ES_1h.parquet`

### Building the Macro Event Calendar

```bash
python src/macro_event_builder.py
```

This pulls actual first-release dates from the FRED API for CPI, PPI, and NFP, combines them with hardcoded FOMC statement dates (2019–2026), and writes `data/NEW_macro_events.csv`. Requires a valid `FRED_API_KEY` in `.env`.

Technical indicator computation and session labelling happen inside `Dataset_SP500_1H` at load time, not in preprocessing.

## Training

### EquityBERT

```bash
python train_sp500_hourly.py
```

The script runs an **ablation study** over three event-feature configurations for each forecast horizon, creating a new versioned run directory (e.g. `runs/v21/`) automatically:

| Ablation         | `use_events` | `use_event_type` | `use_event_impact` |
|-----------------|-------------|------------------|--------------------|
| No Events        | False       | False            | False              |
| Event Type Only  | True        | True             | False              |
| Event Timing Only| True        | False            | False              |

**Default horizons**:

```python
horizons = [
    {"lookback": 24, "forecast": 5},    # Short-term
    {"lookback": 50, "forecast": 10},   # Medium-term
]
```

**Base hyperparameters** (edit `base_config` in `train_sp500_hourly.py`):

```python
base_config = {
    "n_layer":     4,       # BERT encoder layers (sweep: {2, 4, 6})
    "batch_size":  32,
    "max_epochs":  150,
    "lr":          1e-4,
    "patience":    10,      # early stopping patience (monitors val MSE)
}
```

### LSTM Baseline

```bash
python train_lstm.py
```

**Default hyperparameters** (edit `config` in `train_lstm.py`):

```python
config = {
    "hidden_size":   64,
    "num_layers":    2,
    "dropout":       0.4,
    "batch_size":    128,
    "epochs":        100,
    "lr":            3e-4,
    "weight_decay":  5e-4,
    "patience":      15,
    "use_technical": True,
    "use_interday":  True,
}
```

### Dataset Parameters

The `Dataset_SP500_1H` class accepts:

| Parameter          | Type  | Default | Description                                    |
|--------------------|-------|---------|------------------------------------------------|
| `data_path`        | str   | –       | Path to `ES_1h.parquet`                        |
| `events_df`        | DF    | None    | US macro event calendar (if `use_events=True`) |
| `flag`             | str   | 'train' | 'train', 'val', or 'test'                      |
| `size`             | tuple | (48,12) | (seq_len, pred_len)                            |
| `use_technical`    | bool  | True    | Include technical indicators                   |
| `use_events`       | bool  | False   | Include macro event proximity features         |
| `use_event_type`   | bool  | False   | Enable event type tokens                       |
| `use_event_impact` | bool  | False   | Enable event impact tokens                     |
| `use_interday`     | bool  | True    | Include lagged volatility (1h–24h)             |
| `use_explainable`  | bool  | True    | Return semantic tokens (required for EquityBERT) |
| `fine_tuning_pct`  | float | None    | Fraction of training data to use (None = 100%) |

## Model Architectures

### EquityBERT

**Input**:
- Numerical features: `(batch_size, num_series, seq_len)`
- Semantic tokens: dict of scalar `int64` tensors — `{"market_session": ..., "event_type": ..., "event_impact": ...}`

**Architecture** (three-stage pipeline adapted from Vola-BERT / Nguyen et al., ICAIF 2025):

1. **Input encoding** — each of the N feature time series (length L) is projected to a 768-dim vector by a shared linear layer (`wte`), producing one token per feature.
2. **BERT encoder with PEFT** — semantic tokens (market session, event type, event impact) are prepended to the feature token sequence and passed through the first `n_layer` layers of `bert-base-uncased`. Core attention (Q/K/V) and FFN weights are **frozen**; only LayerNorm, positional embeddings, `wte`, semantic embeddings, and the forecast head are trained.
3. **Forecast head** — hidden states of the semantic tokens and the last feature token are concatenated and projected to `pred_len` scalar forecasts. **RevIN** (Reversible Instance Normalisation) is applied before encoding and inverted after decoding.

Note: BERT uses full bidirectional attention (no causal masking), as in the original Vola-BERT design.

**Hyperparameters**:
- `n_layer`: Number of BERT encoder layers to use (default: 4; sweep over {2, 4, 6})
- `revin`: Enable Reversible Instance Normalisation (default: `True`)
- `head_drop_rate`: Dropout before the forecast head (default: 0.2)
- `semantic_tokens`: Dict mapping token name → vocabulary size (e.g. `{"market_session": 4, "event_type": 5, "event_impact": 3}`); pass `{}` to disable

### LSTM Baseline

**Input**: Numerical features only, `(batch_size, num_series, seq_len)`

**Architecture**:
1. Input permuted from `(B, N, L)` to `(B, L, N)` for LSTM convention
2. Unidirectional stacked LSTM layers with dropout between layers
3. Last time-step hidden state taken, dropout applied
4. Single linear layer → `(batch_size, 1, pred_len)` predictions

**Hyperparameters**:
- `hidden_size`: LSTM hidden dimension (default: 64)
- `num_layers`: Stacked LSTM layers (default: 2)
- `dropout`: Dropout between LSTM layers and before output (default: 0.4)

## Evaluation

### Metrics

Both scripts compute:
- **MAE**: Mean absolute error (primary training loss for EquityBERT)
- **MSE**: Mean squared error (used for early stopping in EquityBERT)
- **rMAE**: `model_MAE / naive_MAE` — ratio relative to naive persistence baseline
- **rMSE**: `model_MSE / naive_MSE` — ratio relative to naive persistence baseline

Values below 1.0 indicate the model outperforms the naive baseline.

### LSTM Inference and Export

```bash
python evaluate_lstm.py
```

Loads the best LSTM checkpoint from `runs/lstm_baseline_*/`, runs inference on all splits, and writes to `runs/lstm_baseline/`:
- `lstm_predictions_{lookback}to{forecast}.xlsx` — Datetime, Actual_r, Predicted_r, AE, SE per split
- `lstm_vol_forecast_{lookback}to{forecast}.png` — actual vs predicted time-series
- `lstm_scatter_{lookback}to{forecast}.png` — scatter plot per split

### Output Structure

```
runs/
├── v{N}/                                      # EquityBERT versioned run
│   ├── {ablation}_{lookback}to{forecast}_full/
│   │   └── checkpoints/
│   │       ├── model.pth                      # Best model weights
│   │       └── loss_{name}.png                # MAE and MSE training curves
│   ├── all_loss_curves_v{N}.png               # Combined curves for all ablations
│   └── equitybert_results_v{N}.txt            # Metrics summary
├── lstm_baseline_{YYYY-MM-DD}/
│   ├── best_model_{lookback}to{forecast}.pth  # Best LSTM checkpoint
│   ├── lstm_training_curves.png               # Loss curves across all horizons
│   └── lstm_results.txt                       # rMAE / rMSE summary
└── lstm_baseline/
    ├── lstm_predictions_{lookback}to{forecast}.xlsx
    ├── lstm_vol_forecast_{lookback}to{forecast}.png
    └── lstm_scatter_{lookback}to{forecast}.png
```

### Loading an EquityBERT Checkpoint

```python
import torch
from src.model_bert import EquityBERT
from src.mydataset import SEMANTIC_TOKEN_VOCAB

model = EquityBERT(
    num_series=...,
    input_len=24,
    pred_len=5,
    n_layer=4,
    revin=True,
    semantic_tokens={"market_session": SEMANTIC_TOKEN_VOCAB["market_session"]},
)
model.load_state_dict(torch.load('runs/v1/.../checkpoints/model.pth'))
model.eval()

# x: (B, N, L), tokens: dict of scalar int64 tensors
with torch.no_grad():
    predictions = model((x, tokens))  # (B, 1, pred_len)
```

## Feature Engineering

### Technical Indicators (T=20 hourly bars)

```python
# Bollinger Bands
middle_band = SMA(log_return, 20)
upper_band  = middle_band + 2 * STD(log_return, 20)
lower_band  = middle_band - 2 * STD(log_return, 20)

# Momentum & Acceleration
momentum     = log_return[t] - log_return[t-20]
acceleration = momentum[t] - momentum[t-20]

# EMA & RSI
ema = EMA(log_return, span=20)
rsi = Wilder_RSI(log_return, period=14)
```

### Market Sessions

| Session     | Hours (ET)              | Description                        |
|------------|-------------------------|------------------------------------|
| overnight  | 20:00–04:00, all Sunday | Thin futures continuation market   |
| pre_market | 04:00–09:30             | Institutional pre-open order flow  |
| regular    | 09:30–16:00             | NYSE core session, highest liquidity|
| after_hours| 16:00–20:00             | Post-close, earnings releases      |

### Lagged Volatility

Captures volatility clustering and intraday periodicity:

```python
prev_r_1h  = r[t-1]
prev_r_2h  = r[t-2]
prev_r_4h  = r[t-4]
prev_r_8h  = r[t-8]
prev_r_24h = r[t-24]
```

## Troubleshooting

### Issue: "No training data available for scaling"

**Cause**: `fine_tuning_pct` is too small or the data split is misaligned.

**Solution**:
```python
train_ds = Dataset_SP500_1H(..., fine_tuning_pct=0.5)
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size or sequence length:
```python
batch_size = 16  # Down from 32
seq_len = 24     # Down from 50
```

### Issue: NaN losses during training

**Cause**: Learning rate too high or unstable gradients. Gradient clipping is already applied in the Trainer (`max_norm=1.0`).

**Solution**:
```python
lr = 1e-5  # Reduce from 1e-4
```

## Citation

```
@article{EquityBERT2026,
  title={EquityBERT: Transformer-based Volatility Forecasting for S\&P 500 Futures},
  author={Berker Basergun},
  year={2026}
}
```

## License

See LICENSE file for details.

---

**Last Updated**: April 2026  
**Python Version**: 3.10+  
**PyTorch Version**: 2.0+
