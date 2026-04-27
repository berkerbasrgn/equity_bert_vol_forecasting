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
│   ├── raw/                          # Raw Databento Parquet (output of download_databento.py)
│   └── processed/
│       └── ES_1h.parquet             # Preprocessed 1-hour OHLCV bars
├── src/
│   ├── __init__.py
│   ├── model_bert.py                 # EquityBERT (BERT-based) architecture
│   ├── model_lstm.py                 # LSTM baseline model
│   ├── mydataset.py                  # PyTorch Dataset class with token generation
│   ├── preprocess_data.py            # Raw → processed Parquet pipeline
│   ├── trainer.py                    # Training and evaluation loop (EquityBERT)
│   ├── utils.py                      # StandardScaler, EarlyStopping
│   ├── loss.py                       # MAE / MSE loss functions
│   ├── plot_utils.py                 # Visualisation helpers
│   ├── download_databento.py         # Databento data download script
│   ├── macro_event_builder.py        # US macro event calendar builder
│   └── macroeventcoder.py            # Event type / impact encoder
├── train_sp500_hourly_RAW.py         # EquityBERT training entry point
├── train_lstm.py                     # LSTM baseline training entry point
├── evaluate_lstm.py                  # LSTM inference, Excel export, and plots
├── README.md                         # This file
└── req.txt                           # Python dependencies
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository_url>
cd vola-bert
```

### 2. Install Dependencies

```bash
pip install -r req.txt
```

**Key dependencies**:
- PyTorch (CPU/CUDA)
- pandas, numpy
- scikit-learn
- Databento (for data download)

## Data Preparation

### Prerequisites

The project expects preprocessed data at `data/processed/ES_1h.parquet` with the following structure:

| Column      | Type      | Description                          |
|------------|-----------|--------------------------------------|
| Datetime   | datetime  | Timezone-aware timestamp (ET)        |
| Open       | float     | Opening price                        |
| High       | float     | Session high                         |
| Low        | float     | Session low                          |
| Close      | float     | Closing price                        |
| Volume     | int       | Trade volume or count                |

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

Technical indicator computation and session labelling happen inside `Dataset_SP500_1H` at load time, not in preprocessing.

## Training

### Quick Start

Train EquityBERT:

```bash
python train_sp500_hourly_RAW.py
```

Train the LSTM baseline:

```bash
python train_lstm.py
```

### Configuration

Edit the `base_config` and `horizons` dicts at the bottom of `train_sp500_hourly_RAW.py`:

```python
# Forecast horizon (lookback → forecast)
horizons = [
    {"lookback": 24, "forecast": 5},   # Short-term (default)
    {"lookback": 50, "forecast": 10},  # Medium-term
    {"lookback": 60, "forecast": 20},  # Long-term
]

# Feature toggles
use_technical = True   # Bollinger Bands, RSI, EMA, momentum
use_events = False     # US macro calendar proximity
use_interday = True    # Lagged volatility (1h, 2h, 4h, 8h, 24h)
use_explainable = False  # Semantic tokens (set True to enable EquityBERT tokens)

# Market filter
mode = "24h"           # "24h" (all hours) or "trading" (09:00-16:00 ET)
```

### Dataset Parameters

The `Dataset_SP500_1H` class accepts:

| Parameter          | Type  | Default | Description                                    |
|--------------------|-------|---------|------------------------------------------------|
| `data_path`        | str   | –       | Path to `ES_1h.parquet`                        |
| `events_df`        | DF    | None    | US macro event calendar (if `use_events=True`) |
| `flag`             | str   | 'train' | 'train', 'val', or 'test'                      |
| `size`             | tuple | (48,12) | (seq_len, pred_len); training scripts default to (24, 5) |
| `scale`            | bool  | True    | Standardize features                           |
| `use_technical`    | bool  | True    | Include technical indicators                   |
| `use_events`       | bool  | False   | Include macro event features                   |
| `use_interday`     | bool  | True    | Include lagged volatility                      |
| `use_explainable`  | bool  | True    | Return semantic tokens                         |
| `mode`             | str   | '24h'   | '24h' or 'trading'                             |
| `use_event_type`   | bool  | False   | Enable event type tokens                       |
| `use_event_impact` | bool  | False   | Enable event impact tokens                     |



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
- `dropout`: Dropout between LSTM layers and before output (default: 0.2–0.4)

## Evaluation

### Metrics

The trainer computes per epoch:
- **MAE**: Mean absolute error (primary training loss)
- **MSE**: Mean squared error (used for early stopping)

After training, both scripts additionally compute:
- **rMAE**: `model_MAE / naive_MAE` — ratio relative to the naive persistence baseline
- **rMSE**: `model_MSE / naive_MSE` — ratio relative to the naive persistence baseline

Values below 1.0 indicate the model outperforms the naive baseline. rMAE and rMSE are the primary evaluation metrics for comparing EquityBERT against the LSTM and naive baselines, consistent with the Vola-BERT paper methodology.

### Viewing Results

After training, checkpoints and logs are saved to:

```
runs/
├── v1/
│   ├── BERT_RAW_24to5_full/
│   │   └── checkpoints/
│   │       ├── model.pth                    # Best model weights (saved by EarlyStopping)
│   │       └── loss_BERT_RAW_24to5_full.png # MAE and MSE training curves
│   └── run_version_1_equity_bert_sp500_results.txt   # Metrics summary (MAE, MSE, rMAE, rMSE)
├── lstm_baseline_YYYY-MM-DD/
│   ├── best_model_24to5.pth                 # Best LSTM checkpoint
│   ├── lstm_results.txt                     # rMAE / rMSE summary
│   └── lstm_training_curves.png             # Loss curves across all horizons
```

Load and evaluate an EquityBERT checkpoint:

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
    semantic_tokens=SEMANTIC_TOKEN_VOCAB,  # or {} to disable
)
model.load_state_dict(torch.load('runs/v1/BERT_RAW_24to5_full/checkpoints/model.pth'))
model.eval()

# Make predictions — x: (B, N, L), tokens: dict of scalar int64 tensors
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

| Session     | Hours (ET) | Description            |
|------------|-----------|------------------------|
| overnight  | 20:00-04:00, Sunday | Thin futures market    |
| pre_market | 04:00-09:30        | Institutional orders   |
| regular    | 09:30-16:00        | NYSE core hours        |
| after_hours| 16:00-20:00        | Post-close activity    |

### Lagged Volatility

Captures clustering and intraday periodicity:

```python
prev_r-1h  = r[t-1]
prev_r-2h  = r[t-2]
prev_r-4h  = r[t-4]
prev_r-8h  = r[t-8]
prev_r-24h = r[t-24]
```

## Troubleshooting

### Issue: "No training data available for scaling"

**Cause**: `fine_tuning_pct` is too small or data split is misaligned.

**Solution**:
```python
# Reduce fine_tuning_pct or check data path
train_ds = Dataset_SP500_1H(..., fine_tuning_pct=0.5)
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size or sequence length:
```python
batch_size = 16  # Down from 32
seq_len = 24     # Down from 48
```

### Issue: NaN losses during training

**Cause**: Learning rate too high or unstable gradients.

**Solution**:
```python
learning_rate = 1e-4  # Reduce
# Enable gradient clipping in trainer
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Citation


@article{ES_Futures,
  title={EquityBERT: Transformer-based Volatility Forecasting for S&P 500 Futures},
  author={Berker Basergun},
  year={2026}
}
```

## License

See LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Last Updated**: April 2026  
**Python Version**: 3.8+  
**PyTorch Version**: 1.9+
