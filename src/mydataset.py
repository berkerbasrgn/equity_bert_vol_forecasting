

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
#
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

import os
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
# Helper: US equity session classifier

# NOT USED IN THIS VERSION — session labels are computed via vectorised
# numpy.select in __read_data__ (10-50× faster than row-wise .apply()).
# Retained here for reference and unit-testing individual timestamps.
# ---------------------------------------------------------------------------
def session_label(dt):
    """
    Classifies a US/Eastern timestamp into one of four equity market sessions.

    Arguments:
        dt: a timezone-aware pandas Timestamp in America/New_York

    Returns:
        str: one of 'overnight', 'pre_market', 'regular', 'after_hours'

    Session boundaries (ET):
        pre_market  : 04:00 -- 09:29
        regular     : 09:30 -- 15:59  (NYSE core session)
        after_hours : 16:00 -- 19:59
        overnight   : 20:00 -- 03:59  and all Sunday bars

    Improvement over original:
        The original Vola-BERT used geographic FX sessions (Tokyo/London/NY
        overlaps). These have no meaning for US equity futures. This function
        replaces them with session boundaries that reflect actual US equity
        market microstructure — liquidity, institutional flow, and earnings
        announcement timing all vary materially across these four regimes.
    """
    # Sunday bars belong to the overnight futures session regardless of hour
    if dt.dayofweek == 6:
        return "overnight"

    h, m = dt.hour, dt.minute
    mins = h * 60 + m

    if 4 * 60 <= mins < 9 * 60 + 30:
        return "pre_market"
    elif 9 * 60 + 30 <= mins < 16 * 60:
        return "regular"
    elif 16 * 60 <= mins < 20 * 60:
        return "after_hours"
    else:
        return "overnight"


# ---------------------------------------------------------------------------
# Helper: RSI (unchanged from original)
# ---------------------------------------------------------------------------
def calculate_rsi(prices, period=14):
    """
    Calculates RSI for a given price series.

    Arguments:
        prices (pd.Series): closing prices or log returns
        period (int): lookback period (default 14)

    Returns:
        pd.Series: RSI values
    """
    delta = prices.diff().dropna()
    up   = delta.clip(lower=0)
    down = delta.clip(upper=0, lower=None)
    ema_up   = up.ewm(alpha=1 / period, min_periods=period).mean()
    ema_down = down.abs().ewm(alpha=1 / period, min_periods=period).mean()
    rs  = ema_up / ema_down
    rsi = 100 - 100 / (1 + rs)
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
        - Source data is a Databento GLBX.MDP3 Parquet with pre-aggregated 1h
          OHLCV bars for ES.FUT; no resampling needed inside __read_data__
          (Vola-BERT received a pre-built 30-minute FX CSV from FirstRate Data)
        - US equity session tokens replace Vola-BERT's geographic FX sessions
        - US macro event calendar replaces FXStreet per-currency event flags
        - Technical indicator windows scaled to T=20h for hourly data
          (Vola-BERT used T=12 × 30min = 6h)
        - Minimum week-row filter is 60 (vs 230 for 30-minute FX data)
        - Lagged volatility features use row-shifts of 1h, 2h, 4h, 8h, 24h
 
 
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
 
    Technical indicator columns (TECH_INDICATORS):
        middle_band, upper_band, lower_band  — Bollinger Bands on log_return
        momentum, acceleration               — rate of change and its derivative
        ema                                  — exponential moving average of log_return
        rsi                                  — Wilder RSI on log_return
 
    Lagged volatility columns (INTERDAY_VOLAS):
        prev_r-{1,2,4,8,24}h  — row-shifts of the target r_t
    """

    # Technical indicator column names (computed per week)
    TECH_INDICATORS = [
        "middle_band", "upper_band", "lower_band",
        "momentum", "acceleration",
        "ema", "rsi",
    ]

    # Lagged log-range column names
    # Lags chosen to capture short-term clustering (1h, 2h),
    # medium-term persistence (4h, 8h), and intraday cyclicality (24h)
    INTERDAY_VOLAS = [f"prev_r-{lag}h" for lag in [1, 2, 4, 8, 24]]

    def __init__(
        self,
        data_path,
        events_df,
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
            self.pred_len = 12   # 12h forecast horizon
        else:
            self.seq_len, self.pred_len = size

        assert flag in ("train", "val", "test"), \
            f"flag must be 'train', 'val', or 'test'; got '{flag}'"
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]

        self.data_path       = data_path
        self.events_df       = events_df
        self.scale           = scale
        self.use_technical   = use_technical
        self.use_events      = use_events
        self.use_event_type   = use_event_type
        self.use_event_impact = use_event_impact
        self.use_interday    = use_interday
        self.use_explainable = use_explainable
        self.fine_tuning_pct = fine_tuning_pct
        self.mode            = mode

        if flag != "train" and fine_tuning_pct is not None:
            print("Warning: fine_tuning_pct is only applied to the training split and"
            "has no effect on the validation or test splits.")

        self.__read_data__()

    # ------------------------------------------------------------------
    def __read_data__(self):
        """
        Loads and preprocesses the processed ES.FUT Parquet into per-week
        feature tensors ready for model consumption.
 
        Processing pipeline:
            1. Load processed Parquet (title-case columns, one row/timestamp)
            2. Ensure Datetime is timezone-aware and sort chronologically
            3. Optionally restrict to regular trading hours (mode='trading')
            4. If use_events: compute event-proximity features via merge_asof
            5. If use_explainable: compute session labels via numpy.select
            6. Group by ISO week; skip incomplete weeks (< 60 bars)
            7. Per-week: compute target (r), log-return, log-volume,
               Bollinger Bands, momentum, acceleration, EMA, RSI, lagged r
            8. Drop NaN warm-up rows; skip if < seq_len + pred_len remain
            9. Assemble feature matrix (target 'r' always last column)
           10. Chronological 70/15/15 train-val-test split by week count
           11. Fit StandardScaler on training weeks; transform all splits
           12. Store per-week arrays in self.data and self.raw_data
 
        Key differences from Dataset_Rates_30M.__read_data__:
            - Input is a pre-processed Parquet (not raw CSV)
            - MIN_WEEK_ROWS = 60 (vs 230 for 30-minute FX data)
            - Technical window T = 20h (vs T = 12 × 30min in Vola-BERT)
            - Event proximity via two merge_asof passes (forward + backward)
            - Session labels via numpy.select (10-50× faster than .apply())
            - Lagged features: row-shifts of 1h, 2h, 4h, 8h, 24h
        """


        self.scaler = StandardScaler()

        # ---- Load and parse -----------------------------------------------
        # preprocess_data.py has already: reset index, renamed columns to
        # title-case, selected most-liquid contract, filtered bad bars.
        df_raw = pd.read_parquet(self.data_path)
        df_raw["Datetime"] = pd.to_datetime(df_raw["Datetime"], utc=True).dt.tz_convert("America/New_York")

        df_raw = df_raw.sort_values("Datetime").reset_index(drop=True)

        df_raw = df_raw[df_raw["Datetime"] >= pd.Timestamp("2019-01-01", tz="America/New_York")].copy()
        df_raw = df_raw.dropna(subset=["Open", "High", "Low", "Close"])

        # ---- Optional: restrict to regular trading hours ------

        if self.mode == "trading":
            df_raw = df_raw[
                (df_raw["Datetime"].dt.hour >= 9) &

                (df_raw["Datetime"].dt.hour < 16)            
            ]
        # ---- Event lookup (forward + backward merge_asof) -----------------
        # Forward merge: next upcoming event → hours_to_event
        # Backward merge: most recent past event → hours_since_event 
        # The original code used only a forward merge for hours_since_event,
        # which produced all-zero values (bar_time − future_event ≤ 0, clipped
        # to 0).  The backward merge fixes this.


        if self.use_events:
            events_sorted = (
                self.events_df
                .sort_values("datetime")
                .reset_index(drop=True)
                [["datetime", "event_type", "impact"]]
            )
            df_raw = df_raw.sort_values("Datetime")

            merged_fwd = pd.merge_asof(
                df_raw[["Datetime"]],
                events_sorted.rename(columns={"datetime": "next_event_dt"}),
                left_on="Datetime",
                right_on="next_event_dt",
                direction="forward",
            )
            merged_bwd = pd.merge_asof(
                df_raw[["Datetime"]],
                events_sorted.rename(columns={"datetime": "prev_event_dt"}),
                left_on="Datetime",
                right_on="prev_event_dt",
                direction="backward",
            )


            df_raw["hours_to_event"] = (
                (merged_fwd["next_event_dt"] - df_raw["Datetime"])
                .dt.total_seconds()
                .div(3600.0)
                .clip(upper=999.0)
                .fillna(999.0)
            )
            df_raw["hours_since_event"] = (
                (df_raw["Datetime"] - merged_bwd["prev_event_dt"])
                .dt.total_seconds()
                .div(3600.0)
                .clip(lower=-0.0, upper=48.0)
                .fillna(999.0)
            )

            df_raw["is_event_recent"] = (df_raw["hours_since_event"] <= 5).astype(int)
            df_raw["is_event"] = (merged_fwd["next_event_dt"].notna()).astype(int)
            df_raw["event_type_raw"]   = merged_fwd["event_type"].fillna("NONE").str.upper()
            df_raw["event_impact_raw"] = merged_fwd["impact"].fillna("NONE").str.upper()

            df_raw["is_event_window"] = ((df_raw["hours_to_event"].abs() <= 3)).astype(int)
            df_raw["time_to_event"] = (
                (merged_fwd["next_event_dt"] - df_raw["Datetime"])
                .dt.total_seconds()
                .div(3600.0)
                .clip(-12, 12)
                .fillna(999.0)
            )
        # ---- Vectorised session label (replaces per-row .apply()) --------
        # SPEEDUP: numpy select on integer arrays is 10-50x faster than
        # calling a Python function row by row via .apply().
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

        # ---- Group by ISO week -------------------------------------------
        df_raw["iso_year"] = df_raw["Datetime"].dt.isocalendar().year.astype(int)
        df_raw["iso_week"] = df_raw["Datetime"].dt.isocalendar().week.astype(int)
        df_raw["week_key"] = (
            df_raw["iso_year"].astype(str)
            + "-W"
            + df_raw["iso_week"].astype(str).str.zfill(2)
        )

        MIN_WEEK_ROWS = 60

        df_data     = []
        raw_df_data = []

        for week_key, week_data in df_raw.groupby("week_key"):

            if len(week_data) < MIN_WEEK_ROWS:
                continue

            week_data = week_data.copy().reset_index(drop=True)
            # target variable: log-range (Parkinson) volatility estimator, log high-low range
            week_data["r"] = np.log(week_data["High"] / week_data["Low"])
            # log return and log volume features (log return is the basis for most technical indicators)
            week_data["log_return"] = np.log(
                week_data["Close"] / week_data["Close"].shift(1)
            )

            week_data["log_volume"] = np.log1p(week_data["Volume"])

            T = 20
            D = 2

            week_data["middle_band"] = week_data["log_return"].rolling(window=T).mean()
            rolling_std = week_data["log_return"].rolling(window=T).std()
            week_data["upper_band"]  = week_data["middle_band"] + D * rolling_std
            week_data["lower_band"]  = week_data["middle_band"] - D * rolling_std

            week_data["momentum"]     = (
                week_data["log_return"] - week_data["log_return"].shift(T)
            )
            week_data["acceleration"] = (
                week_data["momentum"] - week_data["momentum"].shift(T)
            )

            week_data["ema"] = week_data["log_return"].ewm(span=T, adjust=False).mean()
            week_data["rsi"] = calculate_rsi(week_data["log_return"], period=14)
            # lagged log-range features to capture volatility clustering and intraday cyclicality; lags of 1h, 2h, 4h, 8h, 24h
            for lag in [1, 2, 4, 8, 24]:
                week_data[f"prev_r-{lag}h"] = week_data["r"].shift(lag)

            # ---- Drop NaN rows (rolling indicator warmup) ----------------
            week_data = week_data.dropna().reset_index(drop=True)

            if len(week_data) < (self.seq_len + self.pred_len):
                continue

            # ---- Build feature matrix ------------------------------------
            # Target 'r' is always placed last so that slicing [:,:-1] gives
            # features and [:,-1:] gives the target with no ambiguity.
            used_features = []

            if self.use_technical:
                used_features += self.TECH_INDICATORS + ["log_return"]

            if self.use_events:
                used_features += ["hours_to_event",
                                   "hours_since_event",
                                     "is_event_recent",
                                     "is_event",
                                     "is_event_window",
                                     "time_to_event"]

            if self.use_interday:
                used_features += self.INTERDAY_VOLAS

            used_features += ["log_volume"]

            used_features.append("r")

            raw_df_data.append(week_data)
            df_data.append(week_data[used_features])

        # ---- Temporal train / val / test split ---------------------------
        # 70 / 15 / 15 by week count, strictly chronological.
        train_start = 0
        if self.fine_tuning_pct is not None and self.fine_tuning_pct < 1.0:
            train_start = int(
                0.7 * (1 - self.fine_tuning_pct - 0.001) * len(df_data)
            )

        val_start  = int(0.70 * len(df_data))
        test_start = int(0.85 * len(df_data))
        borders    = [train_start, val_start, test_start, len(df_data)]

        # ---- Fit scaler on training weeks only ----------------------------
        # Scaler statistics computed from training weeks only.
        # Training stats applied to val/test — never the reverse.
        if self.scale:
            train_frames = [df_data[i] for i in range(train_start, val_start)]
            if len(train_frames) == 0:
                raise ValueError("No training data available for scaling. Check fine_tuning_pct.")
            train_concat = pd.concat(train_frames).values
            self.scaler.fit(train_concat)

        # ---- Store split-specific weeks ----------------------------------
        lo, hi = borders[self.set_type], borders[self.set_type + 1]

        self.data     = []
        self.raw_data = []

        for i in range(lo, hi):
            week_values = df_data[i].values
            if self.scale:
                week_values = self.scaler.transform(week_values)
            self.data.append(week_values)
            if self.use_explainable:
                self.raw_data.append(raw_df_data[i])

        # ---- Index helpers -----------------------------------------------
        self.week_lens = [len(w) for w in self.data]
        self.data_len  = [
            max(0, wl - self.seq_len - self.pred_len + 1)
            for wl in self.week_lens
        ]
        self.cumsum  = np.cumsum(self.data_len)
        self.tot_len = sum(self.data_len)

        # num_series = feature count excluding target 'r'
        # Pass this value to EquityBERT as num_series at construction time.
        self.num_series = df_data[0].shape[1] - 1 if df_data else 0

    # ------------------------------------------------------------------
    def __len__(self):
        return self.tot_len

    # ------------------------------------------------------------------
    def _find_week_index(self, index):
        """Binary search for the week containing the given flat sample index."""
        l, r = 0, len(self.cumsum) - 1
        while l <= r:
            mid = (l + r) // 2
            if index >= self.cumsum[mid]:
                l = mid + 1
            else:
                r = mid - 1
        return l

    # ------------------------------------------------------------------
    def __getitem__(self, index):
        """
        Returns one sample.
 
        When use_explainable=True  → ((x, tokens), y)
        When use_explainable=False → (x, y)
 
        x : (num_series, seq_len) float32
        y : (1, pred_len)         float32
 
        Token lookup uses the first row of the prediction window (r_begin) —
        the regime the model is forecasting INTO.
        """
        week_idx = self._find_week_index(index)
        day_idx  = index - (self.cumsum[week_idx - 1] if week_idx > 0 else 0)

        s_begin = day_idx
        s_end   = s_begin + self.seq_len
        r_begin = s_end
        r_end   = r_begin + self.pred_len

        seq_x = self.data[week_idx][s_begin:s_end, :-1]   # (seq_len, num_series)
        seq_y = self.data[week_idx][r_begin:r_end, -1:]   # (pred_len, 1)

        x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # (num_series, seq_len)
        y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # (1, pred_len)

        if self.use_explainable:
            next_row = self.raw_data[week_idx].iloc[r_begin]

            session_str = next_row.get("market_session", "overnight")
            session_tok = torch.tensor(
                TOKEN_MAPPINGS["market_session"].get(session_str, 0), dtype=torch.long
            )

            event_type_str = next_row.get("event_type_raw", "NONE")
            event_type_str = str(event_type_str).upper()
            if event_type_str not in TOKEN_MAPPINGS["event_type"]:
                event_type_str = "NONE"
            event_type_tok = torch.tensor(
                TOKEN_MAPPINGS["event_type"][event_type_str] if self.use_event_type else 0, dtype=torch.long
            )

            event_impact_str = next_row.get("event_impact_raw", "NONE")
            event_impact_str = str(event_impact_str).upper()
            if event_impact_str not in TOKEN_MAPPINGS["event_impact"]:
                event_impact_str = "NONE"
            event_impact_tok = torch.tensor(
                TOKEN_MAPPINGS["event_impact"][event_impact_str] if self.use_event_impact else 0, dtype=torch.long
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
        """Re-scales model predictions back to the original log-range scale."""
        return self.scaler.inverse_transform(data)
    
    @property
    def start_date(self):
        """Returns the earliest datetime in the dataset split (train/val/test)."""
        if self.raw_data:
            return self.raw_data[0]["Datetime"].iloc[0]
        return None
    
    @property
    def end_date(self):
        """Returns the latest datetime in the dataset split (train/val/test)."""
        if self.raw_data:
            return self.raw_data[-1]["Datetime"].iloc[-1]
        return None
    
