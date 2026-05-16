
# Adapted from the original Vola-BERT dataset (Dataset_Rates_30M) for S&P 500
# E-mini futures hourly volatility forecasting.
#
# Data source:
#   Databento — CME Globex MDP 3.0 feed (dataset: GLBX.MDP3)
#   Instrument : ES.FUT  (E-mini S&P 500 continuous front-month contract)
#   Schema     : ohlcv-1h  (pre-aggregated 1-hour OHLCV bars)
#   Coverage   : 2019-01-01 – 2026-04-01
#   Pipeline   : download_databento.py → preprocess_data.py → this module
#
#   The processed Parquet (data/processed/ES_1h.parquet) has:
#     - A 'Datetime' column (timezone-aware, America/New_York)
#     - Title-case OHLCV columns (Open, High, Low, Close, Volume)
#     - One row per timestamp (most-liquid contract already selected)
#     - Spread instruments and bad bars already removed



# Key differences from the original Vola-BERT dataset (Dataset_Rates_30M):
#   - Source     : Databento GLBX.MDP3 pre-aggregated 1h OHLCV Parquet
#                  (vs pre-built 30-minute FX CSV from FirstRate Data)
#   - Instrument : ES E-mini S&P 500 futures (vs spot FX pairs)
#   - Frequency  : 1-hour bars, no resampling needed (vs 30-minute)
#   - Sessions   : US equity market regimes (overnight / pre-market / regular /
#                  after-hours) rather than geographic FX sessions
#                  (Tokyo / London / NY overlap)
#   - Events     : US macro calendar (FOMC, CPI, PPI, NFP) rather than
#                  FXStreet per-currency impact events
#   - Week filter: 60-row minimum (vs 230 in the 30-minute FX version)
#   - Tech windows: scaled to hourly — Bollinger / EMA / momentum T=20h
#                   (vs T=12 × 30min = 6h in Vola-BERT)
#   - Lagged vol : row-shifts of 1h, 2h, 4h, 8h, 24h (vs prev_vola-1..5 days)
#   - Target     : r_t = ln(H_t / L_t), proportional to the Parkinson (1980)
#                  log-range volatility estimator (same definition as Vola-BERT)
#   - Volume     : log_volume derived from the bar's trade count / volume field

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils import StandardScaler


# ---------------------------------------------------------------------------
# Token vocabularies
# ---------------------------------------------------------------------------
# Semantic tokens encode the qualitative market regime at the first step of
# each prediction window.  They are passed to EquityBERT's embedding layers
# and concatenated to the numerical input representation, following the same
# design as Vola-BERT's session / event tokens.
#
# market_session — reflects US equity market structure:
#   0 = overnight   (20:00–04:00 ET, and all Sunday bars; thin, gapped market)
#   1 = pre_market  (04:00–09:30 ET; institutional order flow, low liquidity)
#   2 = regular     (09:30–16:00 ET; NYSE core session, highest liquidity)
#   3 = after_hours (16:00–20:00 ET; earnings releases, post-close prints)
#
# event_type — US macro releases that drive broad equity volatility:
#   0 = NONE, 1 = CPI, 2 = PPI, 3 = NFP, 4 = FOMC
#
# event_impact — severity tier of the macro event:
#   0 = NONE, 1 = MEDIUM, 2 = HIGH
TOKEN_MAPPINGS = {
    "market_session": {
        "overnight":   0,
        "pre_market":  1,
        "regular":     2,
        "after_hours": 3,
    },
    "event_type": {
        "NONE": 0,
        "CPI":  1,
        "PPI":  2,
        "NFP":  3,
        "FOMC": 4,
    },
    "event_impact": {
        "NONE":   0,
        "MEDIUM": 1,
        "HIGH":   2,
    },
}

# Vocabulary sizes — pass this dict to EquityBERT's SemanticTokenEmbedding
# so it can build correctly-sized nn.Embedding tables for each token type.
SEMANTIC_TOKEN_VOCAB = {
    "market_session": 4,
    "event_type":     5,
    "event_impact":   3,
}


# ---------------------------------------------------------------------------
# Helper: US equity session classifier (reference implementation)
# ---------------------------------------------------------------------------
# NOT USED IN THIS VERSION — session labels are computed via vectorised
# numpy.select in __read_data__ (10-50× faster than row-wise .apply()).
# Retained for reference and unit-testing individual timestamps.
def session_label(dt):
    """
    Classifies a US/Eastern timestamp into one of four equity market sessions.

    Arguments:
        dt (pd.Timestamp): timezone-aware timestamp in America/New_York

    Returns:
        str: one of 'overnight', 'pre_market', 'regular', 'after_hours'

    Session boundaries (ET):
        pre_market  : 04:00 – 09:29  (pre-open institutional flow)
        regular     : 09:30 – 15:59  (NYSE core session)
        after_hours : 16:00 – 19:59  (post-close prints, earnings releases)
        overnight   : 20:00 – 03:59  and all Sunday bars (thin futures continuation)

    Difference from Vola-BERT:
        Vola-BERT used geographic FX sessions (Tokyo / London / NY overlaps),
        which have no meaning for US equity index volatility.  This function
        replaces them with boundaries that reflect actual US equity market
        microstructure — liquidity levels, institutional flow patterns, and
        earnings announcement timing all vary materially across these four regimes.
    """
    if dt.dayofweek == 6:
        return "overnight"

    mins = dt.hour * 60 + dt.minute

    if 4 * 60 <= mins < 9 * 60 + 30:
        return "pre_market"
    elif 9 * 60 + 30 <= mins < 16 * 60:
        return "regular"
    elif 16 * 60 <= mins < 20 * 60:
        return "after_hours"
    else:
        return "overnight"


# ---------------------------------------------------------------------------
# Helper: RSI (unchanged from Vola-BERT)
# ---------------------------------------------------------------------------
def calculate_rsi(prices, period=14):
    """
    Calculates the Relative Strength Index (RSI) for a given price series.

    Uses Wilder's exponential smoothing (alpha = 1/period) to match the
    conventional RSI definition — identical to the implementation in Vola-BERT.

    Arguments:
        prices (pd.Series): closing prices or log returns
        period (int)      : lookback window (default: 14)

    Returns:
        pd.Series: RSI values in the range [0, 100]; NaN for the first
                   `period` rows due to the EMA warm-up period
    """
    delta    = prices.diff().dropna()
    up       = delta.clip(lower=0)
    down     = delta.clip(upper=0, lower=None)
    ema_up   = up.ewm(alpha=1 / period, min_periods=period).mean()
    ema_down = down.abs().ewm(alpha=1 / period, min_periods=period).mean()
    rs       = ema_up / ema_down
    rsi      = 100 - 100 / (1 + rs)
    return rsi


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------
class Dataset_SP500_1H(Dataset):
    """
    PyTorch Dataset for S&P 500 hourly volatility forecasting.

    Adapted from Vola-BERT's Dataset_Rates_30M to support US equity index data
    at hourly frequency.  The public interface (constructor arguments, __getitem__
    return shape, inverse_transform) is kept compatible with Vola-BERT so that
    EquityBERT training scripts require minimal changes.

    Key adaptations from Dataset_Rates_30M:
        - Source data is a pre-processed Databento Parquet with title-case OHLCV
          columns (output of preprocess_data.py); no resampling or column
          renaming needed inside __read_data__
        - US equity session tokens replace Vola-BERT's geographic FX sessions
        - US macro event calendar replaces FXStreet per-currency event flags
        - Technical indicator windows scaled to T=20h for hourly data
        - Minimum week-row filter is 60 (vs 230 for 30-minute FX data)
        - Lagged volatility features use row-shifts of 1h, 2h, 4h, 8h, 24h

    Each __getitem__ sample uses a flat sliding window over the split's
    contiguous row array — sequences can span week boundaries, which is
    correct for near-continuous ES futures data.

    Each __getitem__ sample (use_explainable=True):
        ((x, tokens), y)
        x      : (num_series, seq_len)  float32 — scaled numerical features
        tokens : dict of scalar int64 tensors:
                 'market_session', 'event_type', 'event_impact'
        y      : (1, pred_len)          float32 — future log high-low range

    Each __getitem__ sample (use_explainable=False):
        (x, y)
        x      : (num_series, seq_len)  float32
        y      : (1, pred_len)          float32
    """

    TECH_INDICATORS = [
        "middle_band", "upper_band", "lower_band",
        "momentum", "acceleration",
        "ema", "rsi",
    ]

    # Lags chosen to capture short-term clustering (1h, 2h),
    # medium-term persistence (4h, 8h), and intraday cyclicality (24h).
    INTERDAY_VOLAS = [f"prev_r-{lag}h" for lag in [1, 2, 4, 8, 24]]

    def __init__(
        self,
        data_path,
        events_df=None,
        flag="train",
        size=None,
        scale=True,
        use_technical=True,
        use_events=False,
        use_interday=True,
        use_explainable=True,
        fine_tuning_pct=None,
        mode="24h",
        use_event_type=False,
        use_event_impact=False,
    ):
        """
        Arguments:
            data_path (str)          : path to the pre-processed ES.FUT 1h Parquet
                                       (output of preprocess_data.py)
            events_df (pd.DataFrame  : US macro event calendar with columns
                      | None)          ['datetime', 'event_type', 'impact'];
                                       'datetime' must be timezone-aware (ET).
                                       Required when use_events=True; ignored
                                       (and safely None) otherwise.
            flag (str)               : split selector — 'train', 'val', or 'test'
            size (tuple | None)      : (seq_len, pred_len); defaults to (48, 12)
                                       → 48h lookback, 12h forecast horizon
            scale (bool)             : standardise all features using mean and
                                       std computed on the training split only
            use_technical (bool)     : include Bollinger / EMA / RSI / momentum
                                       indicator columns in x
            use_events (bool)        : include numeric event-proximity features
                                       (hours_to_event, is_event_window, etc.)
                                       in x.  Requires events_df to be a valid
                                       DataFrame; raises ValueError if None.
            use_interday (bool)      : include lagged log-range columns in x
            use_explainable (bool)   : compute and return semantic tokens in
                                       __getitem__.  When False, returns plain
                                       (x, y) tuples compatible with LSTMModel.
            fine_tuning_pct (float)  : fraction of training weeks to retain for
                                       data-scarce experiments; None = use all
            mode (str)               : 'trading' restricts to 09:00–15:59 ET;
                                       '24h' (default) retains all hours
            use_event_type (bool)    : if False, event_type token forced to 0;
                                       only affects use_explainable=True mode
            use_event_impact (bool)  : if False, event_impact token forced to 0;
                                       only affects use_explainable=True mode
        """
        if size is None:
            self.seq_len  = 48   # 48h lookback (~2 trading days of 24h futures)

            self.pred_len = 12    # 12h forecast horizon
        else:
            self.seq_len, self.pred_len = size

        assert flag in ("train", "val", "test"), \
            f"flag must be 'train', 'val', or 'test'; got '{flag}'"
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]

        # Fail fast if events are requested but no calendar is provided
        if use_events and events_df is None:
            raise ValueError(
                "use_events=True but events_df is None. "
                "Pass a macro-event DataFrame or set use_events=False."
            )

        self.data_path        = data_path
        self.events_df        = events_df
        self.scale            = scale
        self.use_technical    = use_technical
        self.use_events       = use_events
        self.use_event_type   = use_event_type
        self.use_event_impact = use_event_impact
        self.use_interday     = use_interday
        self.use_explainable  = use_explainable
        self.fine_tuning_pct  = fine_tuning_pct
        self.mode             = mode

        if flag != "train" and fine_tuning_pct is not None:
            print(
                "Warning: fine_tuning_pct is only applied to the training split "
                "and has no effect on val/test."
            )

        self.__read_data__()

    # ------------------------------------------------------------------
    def __read_data__(self):
        """
        Loads and preprocesses the processed ES.FUT Parquet into a contiguous
        feature array ready for sliding-window sampling.

        Processing pipeline:
            1. Load processed Parquet (title-case columns, one row/timestamp)
            2. Ensure Datetime is timezone-aware and sort chronologically
            3. Optionally restrict to regular trading hours (mode='trading')
            4. If use_events: compute event-proximity features via merge_asof
            5. If use_explainable: compute session labels via numpy.select
            6. Compute target (r), log-return, log-volume, Bollinger Bands,
               momentum, acceleration, EMA, RSI, and lagged r on the FULL
               dataset — rolling windows carry across week boundaries so no
               Monday/Tuesday rows are lost to per-week warmup dropout
            7. Drop only the initial NaN warmup rows at the start of the series
            8. Assemble feature matrix (target 'r' always last column)
            9. Chronological 70/15/15 train-val-test split by row count,
               producing contiguous splits with no calendar gaps
           10. Fit StandardScaler on training rows; transform all splits
           11. Store split slice in self.data (ndarray) and self.raw_data (DataFrame)
        """
        self.scaler = StandardScaler()

        # ---- 1. Load and parse -----------------------------------------------
        df_raw = pd.read_parquet(self.data_path)
        df_raw["Datetime"] = pd.to_datetime(df_raw["Datetime"], utc=True)
        if df_raw["Datetime"].dt.tz is None:
            df_raw["Datetime"] = df_raw["Datetime"].dt.tz_localize("America/New_York")
        else:
            df_raw["Datetime"] = df_raw["Datetime"].dt.tz_convert("America/New_York")

        df_raw = df_raw.sort_values("Datetime").reset_index(drop=True)
        df_raw = df_raw.dropna(subset=["Open", "High", "Low", "Close"])

        # ---- 2. Optional: restrict to regular trading hours ------------------
        if self.mode == "trading":
            df_raw = df_raw[
                (df_raw["Datetime"].dt.hour >= 9) &
                (df_raw["Datetime"].dt.hour < 16)
            ].reset_index(drop=True)

        # ---- 3. Event lookup (forward + backward merge_asof) -----------------
        # Forward merge: next upcoming event → hours_to_event
        # Backward merge: most recent past event → hours_since_event
        if self.use_events:
            events_sorted = (
                self.events_df
                .sort_values("datetime")
                .reset_index(drop=True)
                [["datetime", "event_type", "impact"]]
            )

            merged_fwd = pd.merge_asof(
                df_raw[["Datetime"]].reset_index(drop=True),
                events_sorted.rename(columns={"datetime": "next_event_dt"}),
                left_on="Datetime",
                right_on="next_event_dt",
                direction="forward",
            )

            merged_bwd = pd.merge_asof(
                df_raw[["Datetime"]].reset_index(drop=True),
                events_sorted.rename(columns={"datetime": "prev_event_dt"}),
                left_on="Datetime",
                right_on="prev_event_dt",
                direction="backward",
            )

            df_raw = df_raw.reset_index(drop=True)
            df_raw["hours_to_event"] = (
                (merged_fwd["next_event_dt"].values - df_raw["Datetime"].values)
                / np.timedelta64(1, "h")
            )
            df_raw["hours_to_event"] = df_raw["hours_to_event"].clip(upper=999.0).fillna(999.0)

            df_raw["hours_since_event"] = (
                (df_raw["Datetime"].values - merged_bwd["prev_event_dt"].values)
                / np.timedelta64(1, "h")
            )
            df_raw["hours_since_event"] = (
                df_raw["hours_since_event"].clip(lower=0, upper=48).fillna(999.0)
            )
            df_raw["is_event_recent"] = (df_raw["hours_since_event"] <= 5).astype(int)
            df_raw["is_event"]        = merged_fwd["next_event_dt"].notna().astype(int)
            df_raw["is_event_window"] = (df_raw["hours_to_event"].abs() <= 3).astype(int)
            df_raw["time_to_event"]   = df_raw["hours_to_event"].clip(-12, 12).fillna(999.0)
            df_raw["event_type_raw"]   = merged_fwd["event_type"].fillna("NONE").str.upper().values
            df_raw["event_impact_raw"] = merged_fwd["impact"].fillna("NONE").str.upper().values

        # ---- 4. Session labels (vectorised) ----------------------------------
        if self.use_explainable:
            dow  = df_raw["Datetime"].dt.dayofweek
            mins = df_raw["Datetime"].dt.hour * 60 + df_raw["Datetime"].dt.minute
            conditions = [
                dow == 6,
                mins < 4 * 60,
                mins >= 20 * 60,
                (mins >= 4 * 60) & (mins < 9 * 60 + 30),
                (mins >= 9 * 60 + 30) & (mins < 16 * 60),
                (mins >= 16 * 60) & (mins < 20 * 60),
            ]
            choices = [
                "overnight", "overnight", "overnight",
                "pre_market", "regular", "after_hours",
            ]
            df_raw["market_session"] = np.select(conditions, choices, default="overnight")

        # ---- 5. Compute ALL features on the full dataset ---------------------
        # Rolling windows carry across week boundaries — no per-week warmup
        # reset, so Monday/Tuesday rows are no longer lost to NaN dropout.
        # Only the initial warmup period at the very start of the dataset is
        # dropped (rows where acceleration or prev_r-24h is still NaN).
        df_raw = df_raw.reset_index(drop=True)

        df_raw["r"]          = np.log(df_raw["High"] / df_raw["Low"])
        df_raw["log_return"] = np.log(df_raw["Close"] / df_raw["Close"].shift(1))
        df_raw["log_volume"] = np.log1p(df_raw["Volume"])

        T = 20
        D = 2
        df_raw["middle_band"] = df_raw["log_return"].rolling(window=T).mean()
        rolling_std           = df_raw["log_return"].rolling(window=T).std()
        df_raw["upper_band"]  = df_raw["middle_band"] + D * rolling_std
        df_raw["lower_band"]  = df_raw["middle_band"] - D * rolling_std

        df_raw["momentum"]     = df_raw["log_return"] - df_raw["log_return"].shift(T)
        df_raw["acceleration"] = df_raw["momentum"]   - df_raw["momentum"].shift(T)

        df_raw["ema"] = df_raw["log_return"].ewm(span=T, adjust=False).mean()
        df_raw["rsi"] = calculate_rsi(df_raw["log_return"], period=14)

        for lag in [1, 2, 4, 8, 24]:
            df_raw[f"prev_r-{lag}h"] = df_raw["r"].shift(lag)

        # Drop only the initial NaN warmup (first ~40 rows of the full series)
        df_raw = df_raw.dropna().reset_index(drop=True)

        # ---- 6. Assemble feature matrix (target 'r' always last) -------------
        used_features = []
        if self.use_technical:
            used_features += self.TECH_INDICATORS + ["log_return"]
        if self.use_events:
            used_features += [
                "hours_to_event", "hours_since_event", "is_event_recent",
                "is_event", "is_event_window", "time_to_event",
            ]
        if self.use_interday:
            used_features += self.INTERDAY_VOLAS
        used_features += ["log_volume"]
        used_features += ["r"]

        df_features = df_raw[used_features]
        N = len(df_features)

        # ---- 7. Temporal 70/15/15 split — by row count, no calendar gaps ----
        train_start = 0
        if self.fine_tuning_pct is not None and self.fine_tuning_pct < 1.0:
            train_start = int(
                0.70 * (1 - self.fine_tuning_pct - 0.001) * N
            )

        val_start  = int(0.70 * N)
        test_start = int(0.85 * N)
        borders    = [train_start, val_start, test_start, N]

        # ---- 8. Fit scaler on training rows only ----------------------------
        if self.scale:
            train_values = df_features.iloc[train_start:val_start].values
            if len(train_values) == 0:
                raise ValueError(
                    "No training data available for scaling. "
                    "Check fine_tuning_pct or the date range of data_path."
                )
            self.scaler.fit(train_values)

        # ---- 9. Select this split's contiguous slice ------------------------
        lo, hi = borders[self.set_type], borders[self.set_type + 1]

        split_values = df_features.iloc[lo:hi].values
        if self.scale:
            split_values = self.scaler.transform(split_values)

        self.data     = split_values                              # (N_split, num_features)
        self.raw_data = df_raw.iloc[lo:hi].reset_index(drop=True)  # for token lookups

        self.num_series = df_features.shape[1] - 1
        self.tot_len    = max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    # ------------------------------------------------------------------
    def __len__(self):
        return self.tot_len

    # ------------------------------------------------------------------
    def __getitem__(self, index):
        """
        Returns one sample from the contiguous sliding window.

        When use_explainable=True  → ((x, tokens), y)
        When use_explainable=False → (x, y)

        x : (num_series, seq_len) float32
        y : (1, pred_len)         float32

        Token lookup uses the first row of the prediction window (r_begin) —
        the regime the model is forecasting INTO.
        """
        s_begin = index
        s_end   = s_begin + self.seq_len
        r_begin = s_end
        r_end   = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end, :-1]
        seq_y = self.data[r_begin:r_end, -1:]

        x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)
        y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)

        if self.use_explainable:
            next_row = self.raw_data.iloc[r_begin]

            session_str = next_row.get("market_session", "overnight")
            session_tok = torch.tensor(
                TOKEN_MAPPINGS["market_session"].get(session_str, 0),
                dtype=torch.long,
            )

            event_type_str = str(next_row.get("event_type_raw", "NONE")).upper()
            if event_type_str not in TOKEN_MAPPINGS["event_type"]:
                event_type_str = "NONE"
            event_type_tok = torch.tensor(
                TOKEN_MAPPINGS["event_type"][event_type_str] if self.use_event_type else 0,
                dtype=torch.long,
            )

            event_impact_str = str(next_row.get("event_impact_raw", "NONE")).upper()
            if event_impact_str not in TOKEN_MAPPINGS["event_impact"]:
                event_impact_str = "NONE"
            event_impact_tok = torch.tensor(
                TOKEN_MAPPINGS["event_impact"][event_impact_str] if self.use_event_impact else 0,
                dtype=torch.long,
            )

            tokens = {
                "market_session": session_tok,
                "event_type":     event_type_tok,
                "event_impact":   event_impact_tok,
            }
            return (x, tokens), y

        return x, y

    # ------------------------------------------------------------------
    def inverse_transform(self, data):
        """Re-scales predictions back to the original log-range scale."""
        return self.scaler.inverse_transform(data)

    # ------------------------------------------------------------------
    @property
    def start_date(self):
        """Earliest datetime in this split."""
        if len(self.raw_data) > 0:
            return self.raw_data["Datetime"].iloc[0]
        return None

    @property
    def end_date(self):
        """Latest datetime in this split."""
        if len(self.raw_data) > 0:
            return self.raw_data["Datetime"].iloc[-1]
        return None