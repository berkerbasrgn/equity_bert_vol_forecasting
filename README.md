# EquityBERT: S&P 500 Volatility Forecasting

A PyTorch framework for forecasting log-range volatility of E-mini S&P 500 futures (ES.FUT) at hourly frequency, adapted from the Vola-BERT architecture (Nguyen et al., ICAIF 2025).

---

## Key Results

All metrics are on the **original log-range scale** (`r = ln(H/L)`), test period **2025-03-01 → 2026-03-31**.

### EquityBERT vs LSTM — Test Set Performance

| Horizon | Model | MAE | MSE | MAE vs LSTM | DM p-value |
|---------|-------|-----|-----|-------------|------------|
| 24→5 h | LSTM Baseline | 0.001601 | 6.0×10⁻⁶ | — | — |
| 24→5 h | EquityBERT (No Events) | 0.001195 | 6.0×10⁻⁶ | **−25.4%** | <0.001 *** |
| 24→5 h | EquityBERT (Event Type) | 0.001195 | 6.0×10⁻⁶ | **−25.4%** | <0.001 *** |
| 24→5 h | EquityBERT (Event Timing) | 0.001190 | 6.0×10⁻⁶ | **−25.7%** | <0.001 *** |
| 50→10 h | LSTM Baseline | 0.001560 | 6.0×10⁻⁶ | — | — |
| 50→10 h | EquityBERT (No Events) | 0.001192 | 6.0×10⁻⁶ | **−23.6%** | <0.001 *** |
| 50→10 h | EquityBERT (Event Type) | 0.001188 | 6.0×10⁻⁶ | **−23.8%** | <0.001 *** |
| 50→10 h | EquityBERT (Event Timing) | 0.001177 | 6.0×10⁻⁶ | **−24.6%** | <0.001 *** |

Significance codes: `***` p<0.001 · `**` p<0.01 · `*` p<0.05 · `n.s.` p≥0.05  
Diebold-Mariano test with Newey-West long-run variance (lags = pred\_len − 1).

### EquityBERT vs Naive Persistence Baseline (original scale)

| Horizon | Variant | Naive MAE | Model MAE | rMAE | rMSE | vs Naive | ∆MAE vs No Events |
|---------|---------|-----------|-----------|------|------|----------|-------------------|
| 24→5 h | No Events | 0.001518 | 0.001238 | 0.815 | 0.794 | −18.5% | — |
| 24→5 h | Event Type Only | 0.001518 | 0.001227 | 0.808 | 0.785 | −19.2% | −0.000011 (−0.9%) |
| 24→5 h | Event Timing Only | 0.001518 | 0.001227 | 0.808 | 0.784 | −19.2% | −0.000011 (−0.9%) |
| 50→10 h | No Events | 0.001762 | 0.001344 | 0.763 | 0.661 | −23.7% | — |
| 50→10 h | Event Type Only | 0.001762 | 0.001319 | 0.749 | 0.657 | **−25.2%** | −0.000025 (−1.9%) |
| 50→10 h | Event Timing Only | 0.001762 | 0.001368 | 0.776 | 0.668 | −22.4% | +0.000024 (+1.8%) |

rMAE < 1.0 means the model beats the naive last-value persistence baseline. rMAE and rMSE follow the Vola-BERT paper (Nguyen et al., 2025) evaluation protocol. ∆MAE vs No Events shows the marginal contribution of each event token variant relative to the session-token-only baseline.

### Statistical Significance Summary

| Horizon | Variant | DM stat (MAE) | DM p (MAE) | DM stat (MSE) | DM p (MSE) |
|---------|---------|---------------|------------|---------------|------------|
| 24→5 | No Events | +30.43 | <0.001 *** | −0.47 | n.s. |
| 24→5 | Event Type Only | +30.00 | <0.001 *** | −0.57 | n.s. |
| 24→5 | Event Timing Only | +31.02 | <0.001 *** | +0.16 | n.s. |
| 50→10 | No Events | +32.52 | <0.001 *** | −1.13 | n.s. |
| 50→10 | Event Type Only | +32.68 | <0.001 *** | −1.08 | n.s. |
| 50→10 | Event Timing Only | +33.95 | <0.001 *** | −0.85 | n.s. |

**Conclusion**: EquityBERT significantly outperforms the LSTM on MAE across all configurations and horizons (DM p<0.001). MSE differences are not statistically significant, indicating both models produce similarly-sized large errors during extreme market moves. Event semantic tokens (Event Type, Event Timing) provide marginal but consistent MAE gains over the No Events baseline.

---

## Overview

```
Raw ES futures data (Databento)
          │
          ▼
  src/preprocess_data.py      ← OHLCV cleaning, contract roll, quality filters
          │
          ▼
  data/processed/ES_1h.parquet
          │
          ▼
  src/mydataset.py            ← Feature engineering, StandardScaler, token generation
          │   (technical indicators, lagged vol, session tokens, event tokens)
          │
     ┌────┴────┐
     ▼         ▼
train_sp500_hourly.py    train_lstm.py
(EquityBERT)             (LSTM Baseline)
     │                        │
     ▼                        ▼
runs/equitybert/v*/      runs/lstm_baseline/v*/
     │                        │
     └─────────┬──────────────┘
               ▼
    evaluate_inverse.py      ← EquityBERT: original-scale metrics
    evaluate_lstm.py         ← LSTM: per-step Excel export, plots
    evaluation/evaluate_shap.py  ← SHAP feature importance
    compare_significance.py  ← Diebold-Mariano + paired t-test
               │
               ▼
    runs/significance_tests/
```

Two complementary architectures are trained on the same data and evaluated on the same test split:

- **EquityBERT** — BERT backbone with frozen attention weights, semantic token embeddings for market session and macro event awareness, RevIN instance normalisation.
- **LSTM Baseline** — Stacked unidirectional LSTM with technical and interday features; no event awareness.

**Target variable**: Log-range volatility `r_t = ln(H_t / L_t)` (Parkinson, 1980), computed from hourly ES.FUT OHLCV bars.

---

## Project Structure

```
vola-bert/
├── data/
│   ├── raw/
│   │   ├── ES_1h.parquet          # Raw Databento hourly bars
│   │   └── ES_1min.parquet        # Raw 1-minute bars (optional)
│   ├── processed/
│   │   └── ES_1h.parquet          # Cleaned hourly OHLCV for model input
│   └── NEW_macro_events.csv       # US macro event calendar (FRED + FOMC)
│
├── src/
│   ├── model_bert.py              # EquityBERT architecture (BERT + PEFT + RevIN)
│   ├── model_lstm.py              # LSTM baseline
│   ├── mydataset.py               # Dataset_SP500_1H with token generation
│   ├── trainer.py                 # Training loop, early stopping, checkpointing
│   ├── utils.py                   # StandardScaler, EarlyStopping
│   ├── loss.py                    # MAE / MSE loss functions
│   ├── preprocess_data.py         # Raw → processed Parquet pipeline
│   ├── macro_event_builder.py     # US macro calendar builder (FRED API)
│   └── plot_utils.py              # Visualisation helpers
│
├── evaluation/
│   ├── dm_test.py                 # Standalone Diebold-Mariano implementation
│   ├── evaluate_lstm.py           # LSTM evaluation (per-step Excel + SHAP + plots)
│   ├── evaluate_shap.py           # SHAP feature importance for LSTM
│   └── main_dm_test_run.py        # CLI entry point for DM test
│
├── train_sp500_hourly.py          # EquityBERT training entry point
├── train_lstm.py                  # LSTM baseline training entry point
├── evaluate_lstm.py               # LSTM inference, Excel export, plots (root-level)
├── evaluate_inverse.py            # EquityBERT post-hoc eval on original scale
├── compare_significance.py        # Cross-model statistical significance tests
│
├── runs/
│   ├── equitybert/
│   │   └── v27/                   # Latest EquityBERT run (6 variants × 2 horizons)
│   │       ├── No Events_24to5_full/checkpoints/
│   │       ├── Event Type Only_24to5_full/checkpoints/
│   │       ├── Event Timing Only_24to5_full/checkpoints/
│   │       ├── No Events_50to10_full/checkpoints/
│   │       ├── Event Type Only_50to10_full/checkpoints/
│   │       ├── Event Timing Only_50to10_full/checkpoints/
│   │       ├── equitybert_results_v25.csv
│   │       └── all_loss_curves_v25.png
│   ├── lstm_baseline/
│   │   └── v2/                    # LSTM baseline run
│   │       ├── best_model_24to5.pth
│   │       ├── best_model_50to10.pth
│   │       ├── lstm_predictions_24to5.xlsx   # Train/Val/Test sheets
│   │       ├── lstm_predictions_50to10.xlsx
│   │       └── lstm_results_v2.txt
│   └── significance_tests/
│       ├── significance_results.txt
│       └── significance_results.csv
│
├── eda_rt.py                      # Exploratory Data Analysis script
├── eda_figures/                   # Generated EDA figures (PNG + PDF)
│   ├── eda_timeseries.png         # Full 7-year r_t time series
│   ├── eda_histogram.png          # Distribution histogram + KDE
│   ├── eda_stats_table.png        # Summary statistics table
│   ├── eda_stats_table.csv        # Machine-readable summary stats
│   ├── eda_intraday.png           # Intraday seasonality by hour-of-day
│   ├── eda_acf_pacf.png           # ACF / PACF up to lag 48 h
│   └── eda_event_window.png       # Volatility response ±12 h around macro releases
├── equitybert_results_original_scale.txt  # EquityBERT inverse-scale eval
├── equitybert_results_original_scale.csv
├── req.txt                        # Python dependencies
└── pyproject.toml
```

---

## Installation

```bash
git clone <repository_url>
cd vola-bert

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r req.txt
```

**Key dependencies**:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0 | Model training |
| transformers | latest | BERT backbone (bert-base-uncased) |
| pandas | ≥2.0 | Data handling |
| numpy | ≥1.24 | Numerical ops |
| scikit-learn | ≥1.3 | StandardScaler |
| shap | ≥0.45 | Feature importance |
| scipy | latest | Statistical tests (DM, t-test) |
| pyarrow | ≥14.0 | Parquet I/O |
| matplotlib | ≥3.7 | Plots |

---

## Data Preparation

### Parquet Schema

`data/processed/ES_1h.parquet` must contain:

| Column | Type | Description |
|--------|------|-------------|
| Datetime | datetime (tz-aware, ET) | Bar open timestamp |
| Open | float | Opening price |
| High | float | Session high |
| Low | float | Session low |
| Close | float | Closing price |
| Volume | int | Trade volume |

### Preprocessing Pipeline

```bash
# 1. Download raw hourly bars from Databento
python src/download_databento.py          # → data/raw/ES_1h.parquet

# 2. Clean and normalise
python src/preprocess_data.py             # → data/processed/ES_1h.parquet
```

Preprocessing steps:
1. Reset `ts_event` DatetimeIndex to `Datetime` column
2. Remove calendar-spread instruments (symbols containing `-`)
3. Rename lowercase OHLCV columns to title-case
4. Keep only the most liquid contract per timestamp (highest volume)
5. Drop bars with `Low < 1000` or `High > 10000` (data errors)
6. Output cleaned Parquet

Technical indicators, session tokens, and event features are computed inside `Dataset_SP500_1H` at load time.

### Macro Event Calendar

The event calendar (`data/NEW_macro_events.csv`) is built from FRED vintage dates and hardcoded FOMC release times:

```bash
# Requires FRED_API_KEY in .env
python src/macro_event_builder.py         # → data/NEW_macro_events.csv
```

| Column | Values |
|--------|--------|
| datetime | timezone-aware ET timestamp |
| event_type | CPI, PPI, NFP, FOMC |

The calendar covers 2019-01-01 → 2026-03-31 with four event types at their standard release times (BLS releases at 08:30 ET; FOMC statements at 14:00 ET).

---

## Exploratory Data Analysis

```bash
python eda_rt.py
```

Produces six publication-ready figures (PNG at 300 dpi + PDF) in `eda_figures/`:

| Figure | File | Description |
|--------|------|-------------|
| Time series | `eda_timeseries.png` | Hourly r_t over the full 7-year period with daily mean overlay, 30-day rolling std, and major event annotations (COVID, Fed hike cycle, SVB, Aug 2024 unwind). Train / Val / Test boundaries shown. |
| Histogram | `eda_histogram.png` | Density histogram + KDE with Normal reference curve. Right-skew (+5.5) and excess kurtosis (+67) annotated. |
| Stats table | `eda_stats_table.png` | Summary statistics table (also saved as `eda_stats_table.csv`). |
| Intraday seasonality | `eda_intraday.png` | Mean and median r_t by hour-of-day (ET), session-coloured, with NYSE open / close markers. |
| ACF / PACF | `eda_acf_pacf.png` | Autocorrelation and partial ACF up to lag 48 h. Significant lags highlighted; lags 1, 5, 24 marked. Directly motivates the 48 h lookback window. |
| Event-window | `eda_event_window.png` | Mean r_t ±12 h around each macro release type (CPI, PPI, NFP, FOMC), with ±1 SE ribbon and unconditional mean reference. Empirically motivates the event semantic tokens. |

### Summary Statistics — r_t = ln(H_t / L_t), ES Futures 1 h

| Statistic | Value |
|-----------|-------|
| Observations | 42,742 |
| Date range | 2019-01-01 → 2026-03-31 |
| Mean | 0.002846 |
| Median | 0.001981 |
| Std dev | 0.003036 |
| Min | 0.000098 |
| Max | 0.083833 |
| 5th percentile | 0.000595 |
| 95th percentile | 0.007756 |
| Skewness | +5.495 |
| Excess kurtosis | +66.745 |
| Autocorrelation lag 1 | 0.698 |
| Autocorrelation lag 5 | 0.515 |
| Autocorrelation lag 24 | 0.580 |

**Key observations:**

- **Right-skewed, fat-tailed**: skewness +5.5 and excess kurtosis +67 confirm the distribution is far from Gaussian; the maximum bar (0.084, during the COVID crash) is ~30× the mean.
- **Persistent autocorrelation with a 24 h seasonal cycle**: the ACF decays slowly (ACF(1)=0.70, ACF(24)=0.58) and shows a clear trough–peak oscillation at multiples of 24 lags, reflecting the daily open-close volatility cycle. The PACF cuts off sharply after lag 2–3, confirming that the long memory is driven by a seasonal AR structure rather than long-range dependence. This directly motivates the 48 h lookback window.
- **Macro releases cause sharp volatility spikes**: all four event types (CPI, PPI, NFP, FOMC) show a pronounced spike at lag 0 (release hour), with FOMC producing the largest response (~3–4× the unconditional mean) and CPI/NFP showing elevated volatility persisting for 2–3 h post-release. This empirically validates the inclusion of event semantic tokens in EquityBERT.
- **Intraday U-shape**: r_t peaks at the NYSE open (09:30 ET), remains elevated into the early afternoon, and drops to overnight lows after 20:00 ET. This regime structure is explicitly captured by the market session semantic token.
- **Volatility clustering**: the 30-day rolling standard deviation panel shows sustained elevated regimes around COVID (Mar 2020), the Fed hike cycle (2022), and the Aug 2024 carry-unwind, consistent with GARCH-type clustering.

---

## Training

### EquityBERT

```bash
python train_sp500_hourly.py
```

Outputs a versioned run directory under `runs/equitybert/v{N}/` containing checkpoints, loss plots, and a results CSV.

**Key hyperparameters** (edit `base_config` in `train_sp500_hourly.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 24 | Input window length (hours) |
| `forecast` | 5 | Prediction horizon (hours) |
| `n_layer` | 4 | BERT encoder layers to use |
| `batch_size` | 32 | Training batch size |
| `max_epochs` | 200 | Maximum training epochs |
| `lr` | 1e-4 | Learning rate |
| `patience` | 10 | Early stopping patience |
| `weight_decay` | 1e-3 | AdamW weight decay |

**Sensitivity analysis variants** (controlled via `sensitivity_configs`):

| Variant | `use_events` | `use_event_type` | Description |
|---------|-------------|-----------------|-------------|
| No Events | False | False | Market session tokens only |
| Event Type Only | True | True | Session + event type tokens |
| Event Timing Only | True | False | Session + event proximity only |

**Forecast horizons**:

| Label | Lookback | Forecast | Description |
|-------|----------|----------|-------------|
| Short-term | 24 h | 5 h | 1-day context → 5-hour ahead |
| Medium-term | 50 h | 10 h | ~2-day context → 10-hour ahead |

### LSTM Baseline

```bash
python train_lstm.py
```

Outputs checkpoints to `runs/lstm_baseline/v{N}/`.

**LSTM configuration**:

| Parameter | Value |
|-----------|-------|
| hidden_size | 64 |
| num_layers | 2 |
| dropout | 0.2–0.4 |
| batch_size | 128 |
| lr | 3e-4 |
| weight_decay | 5e-4 |
| patience | 15 |

### Data Split

Both models use the same 70 / 15 / 15 weekly split on the full dataset:

| Split | Period | Fraction |
|-------|--------|----------|
| Train | 2019-01-03 → 2024-01-21 | 70% |
| Val | 2024-01-23 → 2025-02-23 | 15% |
| Test | 2025-02-25 → 2026-03-31 | 15% |

---

## Reproducibility

| Item | Value |
|------|-------|
| Random seed | PyTorch default (no manual seed set — results may vary by ±0.0001 MAE across runs) |
| Hardware | Apple M-series (MPS backend); SHAP evaluation forced to CPU |
| EquityBERT training time | ~25–40 min per variant per horizon (M-series, batch=32) |
| LSTM training time | ~5–8 min per horizon |
| SHAP evaluation time | ~15–20 min per horizon (CPU, 500 background samples) |
| Python | 3.10+ |
| PyTorch | ≥2.0 |
| Transformers | bert-base-uncased weights from HuggingFace Hub |

To reproduce the reported results exactly, use the checkpoints in `runs/equityBERT/v27/` and `runs/lstm_baseline/v2/` together with `evaluate_inverse.py` and `compare_significance.py`. Retraining from scratch will produce statistically equivalent but not bit-identical numbers due to the absence of a fixed global seed.

---

## Evaluation Pipeline

### 1. EquityBERT — Original-Scale Metrics

Loads saved checkpoints and inverse-transforms predictions back to the raw `ln(H/L)` scale:

```bash
python evaluate_inverse.py
```

Outputs `equitybert_results_original_scale.txt` and `.csv` with naive MAE, model MAE, rMAE, and rMSE for each variant.

### 2. LSTM — Per-Step Excel Export and Plots

```bash
python evaluate_lstm.py
```

Produces inside `runs/lstm_baseline/v{N}/`:
- `lstm_predictions_{tag}.xlsx` — three sheets (Train / Val / Test) with columns `Datetime, Horizon_step, Actual_r, Predicted_r, AE, SE`
- `lstm_vol_forecast_{tag}.png` — actual vs predicted time series
- `lstm_scatter_{tag}.png` — scatter plot per split

### 3. SHAP Feature Importance (LSTM)

```bash
python evaluation/evaluate_shap.py
```

Produces inside `runs/lstm_baseline/v{N}/`:
- `shap_bar_{tag}.png` — ranked mean |SHAP| per feature
- `shap_heatmap_{tag}.png` — feature × time-step importance heatmap
- `shap_summary_{tag}.png` — beeswarm coloured by feature value

**Results — feature ranking by mean |SHAP value|** (GradientExplainer, 500 test samples):

| Rank | Feature | 24→5 h | 50→10 h | Interpretation |
|------|---------|--------|---------|----------------|
| 1 | `prev_r_8h` | 0.00632 | 0.00300 | Previous US session volatility — dominant signal at both horizons |
| 2 | `log_volume` | 0.00601 | 0.00283 | Volume–volatility relationship; second-most important across horizons |
| 3 | `prev_r_1h` | 0.00390 | 0.00088 | Very recent memory — critical short-term, decays at longer horizon |
| 4 | `rsi` | 0.00224 | 0.00080 | Momentum state; more useful at 24→5 where regime is fresh |
| 3† | `prev_r_24h` | 0.00202 | 0.00180 | Daily cycle — rises to #3 at 50→10 (full day lookback matters more) |
| 4† | `momentum` | 0.00099 | 0.00173 | Price-change acceleration; gains importance at longer horizon |

†Rank at 50→10 horizon.

**Key findings:**
- `prev_r_8h` and `log_volume` are the top-2 features at both horizons, together accounting for ~60% of total SHAP weight at 24→5 and ~50% at 50→10.
- At the **short horizon (24→5)**, very recent lags (`prev_r_1h`, RSI) dominate after the top-2 — the model is primarily an intraday mean-reversion/persistence signal.
- At the **long horizon (50→10)**, `prev_r_24h` and `momentum` rise significantly — the model begins to exploit the daily volatility cycle and directional momentum.
- Technical indicators (Bollinger bands, EMA) contribute modestly at both horizons, confirming their secondary role relative to the raw lagged volatility features.

### 4. Statistical Significance Testing

```bash
python compare_significance.py
```

For each EquityBERT variant and horizon, the script:
1. Loads LSTM test predictions from the xlsx Test sheet (original scale)
2. Runs EquityBERT inference and inverse-transforms to the same scale
3. Aligns both on `(Datetime, Horizon_step)` via an inner join
4. Runs Diebold-Mariano test (Newey-West, `h−1` lags) and paired t-test
5. Saves `runs/significance_tests/significance_results.txt` and `.csv`

```
runs/significance_tests/
├── significance_results.txt    # Full report with DM stats and p-values
└── significance_results.csv    # Machine-readable summary
```

---

## Model Architectures

### EquityBERT

Adapted from Vola-BERT (Nguyen et al., ICAIF 2025) for equity index volatility.

```
Input: (B, N, L)  — B=batch, N=features, L=seq_len
       + tokens: {"market_session": int64, "event_type": int64, ...}

┌────────────────────────────────────────────────┐
│  1. RevIN  (instance normalisation per series) │
├────────────────────────────────────────────────┤
│  2. Input encoding                             │
│     shared Linear: (L,) → 768-dim token        │
│     → N feature tokens + K semantic tokens     │
├────────────────────────────────────────────────┤
│  3. BERT encoder (first n_layer layers)        │
│     ✗ Frozen: Q/K/V, FFN dense weights        │
│     ✓ Trained: LayerNorm, pos. embed., wte,   │
│                semantic embed., forecast head  │
├────────────────────────────────────────────────┤
│  4. Token selection                            │
│     [semantic token outputs] + [last feat tok] │
│     → concatenate → Linear → (B, 1, pred_len) │
├────────────────────────────────────────────────┤
│  5. RevIN⁻¹  (denormalise)                    │
└────────────────────────────────────────────────┘
Output: (B, 1, pred_len)
```

**Semantic token vocabulary**:

| Token | Values | Description |
|-------|--------|-------------|
| market_session | 4 | overnight / pre-market / regular / after-hours |
| event_type | 5 | none / CPI / PPI / NFP / FOMC |
| event_impact | 3 | none / low / high |

**Trainable parameter count** (n_layer=4): ~2.5M out of ~110M total BERT parameters.

### LSTM Baseline

```
Input: (B, N, L)

Permute → (B, L, N)
→ LSTM (hidden=64, layers=2, dropout=0.2–0.4)
→ Last hidden state h_T: (B, 64)
→ Dropout
→ Linear: 64 → pred_len
→ Reshape → (B, 1, pred_len)

Output: (B, 1, pred_len)
```

---

## Feature Engineering

### Dataset_SP500_1H Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str | — | Path to `ES_1h.parquet` |
| `events_df` | DataFrame | None | Macro event calendar |
| `flag` | str | `'train'` | `'train'`, `'val'`, or `'test'` |
| `size` | tuple | `(48, 12)` | `(seq_len, pred_len)` |
| `use_technical` | bool | True | Bollinger Bands, RSI, EMA, momentum |
| `use_events` | bool | False | Event proximity features |
| `use_event_type` | bool | False | Event type semantic token |
| `use_event_impact` | bool | False | Event impact semantic token |
| `use_interday` | bool | True | Lagged volatility (1h–24h) |
| `use_explainable` | bool | True | Return semantic token dict |
| `mode` | str | `'24h'` | `'24h'` (all hours) or `'trading'` (09:00–16:00 ET) |
| `fine_tuning_pct` | float | None | Fraction of training data to use (scarce-data experiments) |

### Technical Indicators (window T=20 bars)

```python
r_t         = ln(High_t / Low_t)           # log-range volatility (target)
log_ret     = ln(Close_t / Close_{t-1})    # log return

middle_band = SMA(log_ret, 20)
upper_band  = middle_band + 2 × STD(log_ret, 20)   # Bollinger upper
lower_band  = middle_band − 2 × STD(log_ret, 20)   # Bollinger lower

momentum    = log_ret_t − log_ret_{t-20}
acceleration= momentum_t − momentum_{t-20}

ema         = EMA(log_ret, span=20)
rsi         = Wilder_RSI(log_ret, period=14)
```

### Lagged Volatility (Interday Features)

Captures volatility clustering across intraday and overnight horizons:

| Feature | Lag |
|---------|-----|
| `prev_r_1h` | r[t−1] |
| `prev_r_2h` | r[t−2] |
| `prev_r_4h` | r[t−4] |
| `prev_r_8h` | r[t−8] |
| `prev_r_24h` | r[t−24] |

### Market Session Tokens

| Session | Hours (ET) | Token ID |
|---------|-----------|----------|
| overnight | 20:00–03:59 (+ Sun) | 0 |
| pre_market | 04:00–09:29 | 1 |
| regular | 09:30–15:59 | 2 |
| after_hours | 16:00–19:59 | 3 |

### Macro Event Tokens

Event proximity features (when `use_events=True`):
- `hours_to_event` — hours until next scheduled macro release (clipped at 999)
- `hours_since_event` — hours since last release (clipped at 48)
- `is_event_window` — binary flag within ±2 h of an event

Event type token maps to: `{none: 0, CPI: 1, PPI: 2, NFP: 3, FOMC: 4}`

---

## Loading a Saved Checkpoint

```python
import torch
from src.model_bert import EquityBERT
from src.mydataset import SEMANTIC_TOKEN_VOCAB

model = EquityBERT(
    num_series=14,        # number of input features (excluding target r)
    input_len=24,
    pred_len=5,
    n_layer=4,
    revin=True,
    semantic_tokens={"market_session": SEMANTIC_TOKEN_VOCAB["market_session"]},
)

state = torch.load("runs/equitybert/v27/No Events_24to5_full/checkpoints/checkpoint.pth",
                   map_location="cpu")
# checkpoint.pth is a raw state_dict; model.pth is also a raw state_dict
model.load_state_dict(state)
model.eval()

# x: (B, N, L)  tokens: dict of int64 tensors
with torch.no_grad():
    predictions = model((x, tokens))   # → (B, 1, pred_len)
```

---

## Metrics Reference

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| MAE | mean\|ŷ − y\| | Absolute forecast error (primary) |
| MSE | mean(ŷ − y)² | Penalises large errors more |
| rMAE | MAE_model / MAE_naive | <1.0 beats naive persistence |
| rMSE | MSE_model / MSE_naive | <1.0 beats naive persistence |
| DM stat | d̄ / √(LRV/T) | >0 means EquityBERT wins on loss |

**Naive baseline**: last observed `r` value in the input window repeated for all forecast steps.

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `No training data available for scaling` | `fine_tuning_pct` too small or path error | Check `data_path`; raise `fine_tuning_pct` |
| CUDA out of memory | Batch or sequence too large | Reduce `batch_size` (try 16) or `seq_len` |
| NaN losses | Learning rate too high | Set `lr=1e-4`; gradient clip added in Trainer |
| `Checkpoint not found` | Wrong run version path | Check `EQUITYBERT_DIR` in `evaluate_inverse.py` / `compare_significance.py` |
| MPS error in SHAP | GradientExplainer not fully MPS-compatible | `evaluation/evaluate_shap.py` forces CPU for SHAP computation |

---

## Limitations

| Limitation | Detail |
|------------|--------|
| **MSE not improved** | EquityBERT reduces MAE significantly (−25% vs LSTM) but DM tests on MSE are uniformly n.s. Both models produce similarly-sized errors on extreme bars (COVID, Aug 2024). Tail-event forecasting remains an open problem. |
| **BERT frozen weights** | Q/K/V and FFN weights are fixed at bert-base-uncased pre-trained values. Only LayerNorm, positional embeddings, and the forecast head are fine-tuned. The attention structure was not pre-trained on financial sequences and may not be optimal for this domain. |
| **Single instrument** | All experiments use ES.FUT (front-month E-mini S&P 500). Transferability to other equity indices, commodities, or FX has not been tested. |
| **Event impact not modelled** | The macro calendar includes release type (CPI/PPI/NFP/FOMC) but not the surprise component (actual minus consensus). Surprise magnitude is the primary driver of event-hour volatility spikes, and is not captured by timing tokens alone. |
| **No exogenous macro features** | VIX, yield-curve slope, and other macro regime indicators are excluded. These could improve performance during sustained regime shifts (e.g., 2022 rate cycle). |
| **Fixed lookback** | The 48 h / 50 h lookback windows are fixed hyperparameters. An adaptive or attention-based window selection was not explored. |
| **No global seed** | Weights are randomly initialised without a fixed seed. Reported metrics are from a single training run; ensemble averaging across seeds was not performed. |

---

## Citation

```bibtex
@mastersthesis{Basergun2026EquityBERT,
  title  = {EquityBERT: Transformer-based Volatility Forecasting for S\&P 500 Futures},
  author = {Basergun, Berker},
  year   = {2026}
}

@inproceedings{Nguyen2025VolaBERT,
  title     = {Repurposing Language Models for FX Volatility Forecasting},
  author    = {Nguyen et al.},
  booktitle = {ICAIF},
  year      = {2025}
}
```

---

**Last Updated**: May 2026 · **Python**: 3.10+ · **PyTorch**: 2.0+ · **EDA figures**: 6
