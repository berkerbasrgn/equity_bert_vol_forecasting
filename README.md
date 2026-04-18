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
│   └── processed/
│       └── ES_1h.parquet         # Preprocessed 1-hour OHLCV bars
├── src/
│   ├── __init__.py
│   ├── model_bert.py             # EquityBERT transformer architecture
│   ├── model_lstm.py             # LSTM baseline model
│   ├── mydataset.py              # PyTorch Dataset class with token generation
│   ├── preprocess_data.py        # Data preprocessing pipeline
│   ├── trainer.py                # Training and evaluation loop
│   └── utils.py                  # Utility functions (scaling, metrics)
├── train_sp500_hourly.py         # Main training entry point
├── README.md                      # This file
└── req.txt                        # Python dependencies
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

If you have raw data:

```bash
python src/preprocess_data.py \
    --input_path <path_to_raw_data> \
    --output_path data/processed/ES_1h.parquet
```

**What preprocessing does**:
1. Loads 1-hour OHLCV bars from Databento or CSV
2. Filters bad bars and outliers
3. Computes technical indicators (Bollinger Bands, RSI, EMA)
4. Aligns macro event calendar
5. Generates market session labels
6. Outputs cleaned Parquet

## Training

### Quick Start

Run the main training script:

```bash
python train_sp500_hourly.py
```

### Configuration

Edit `train_sp500_hourly.py` to customize:

```python
# Data parameters
seq_len = 48           # 48-hour lookback
pred_len = 12          # 12-hour forecast horizon

use_explainable = True # Enable semantic tokens


# Feature toggles
use_technical = True   # Bollinger Bands, RSI, EMA, momentum
use_events = False     # US macro calendar proximity
use_interday = True    # Lagged volatility (1h, 2h, 4h, 8h, 24h)

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
| `size`             | tuple | (48,12) | (seq_len, pred_len)                            |
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
- Semantic tokens: `(batch_size, 3)` — [market_session, event_type, event_impact]

**Architecture**:
1. Token embeddings concatenated to feature matrix
2. Multi-head self-attention transformer stack (configurable layers/heads)
3. Temporal attention masking (causal, for valid forecasting)
4. Linear decoder → `(batch_size, 1, pred_len)` predictions

**Hyperparameters**:
- `d_model`: Embedding dimension (default: 64)
- `nhead`: Number of attention heads (default: 4)
- `num_layers`: Transformer depth (default: 2)
- `dim_feedforward`: FFN hidden size (default: 256)
- `dropout`: Regularization (default: 0.1)

### LSTM Baseline

**Input**: Numerical features only, `(batch_size, num_series, seq_len)`

**Architecture**:
1. Bidirectional LSTM layers
2. Dense layers with dropout
3. Output layer → predictions

**Hyperparameters**:
- `hidden_size`: LSTM units (default: 64)
- `num_layers`: Stacked LSTM layers (default: 2)
- `dropout`: Regularization (default: 0.1)

## Evaluation

### Metrics

The trainer computes:
- **MAE**: Mean absolute error
- **RMSE**: Root mean squared error
- **MAPE**: Mean absolute percentage error
- **Direction Accuracy**: % correct up/down forecasts

### Viewing Results
50/20/30 split (train/val/test):
- **2019-01-08 19:00:00-05:00 → 2022-08-07 23:00:00-04:00**
- **2022-08-09 19:00:00-04:00 → 2024-01-14 23:00:00-05:00**
- **2024-01-16 23:00:00-05:00 → 2026-03-29 23:00:00-04:00**


<img width="2100" height="3600" alt="all_loss_curves_v22" src="https://github.com/user-attachments/assets/5542e2be-dd26-4694-8861-1b9347b0cf28" />

| Horizon | Ablation Type           | Val MSE | Test MAE | Test rMAE | Test rMSE |
|--------|------------------------|--------|----------|-----------|-----------|
| 24→5   | No Events              | 0.3564 | 0.3206   | 0.3739    | 0.2074    |
| 24→5   | Event Type + Timing    | 0.3596 | 0.3239   | 0.3777    | 0.2087    |
| 24→5   | Event Timing Only      | 0.3593 | 0.3234   | 0.3772    | 0.2083    |
| 50→10  | No Events              | 0.4284 | 0.3705   | 0.4253    | 0.2522    |
| 50→10  | Event Type + Timing    | 0.4364 | 0.3722   | 0.4273    | 0.2544    |
| 50→10  | Event Timing Only      | 0.4329 | 0.3648   | 0.4188    | 0.2522    |


After training, checkpoints and logs are saved to:

```
runs/
├── v1/
│   ├── BERT_RAW_24to5_full/
│   │   └── checkpoints/
│   │       ├── model.pth           # Best model weights
│   │       ├── checkpoint.pth      # Latest checkpoint (for resume)
│   │       └── loss_*.png          # Training curves
│   └── run_version_1_results.txt   # Metrics summary
```

Load and evaluate a checkpoint:

```python
import torch
from src.model_bert import EquityBERT

model = EquityBERT(...)
checkpoint = torch.load('runs/v1/BERT_RAW_24to5_full/checkpoints/model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(x, tokens)
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
