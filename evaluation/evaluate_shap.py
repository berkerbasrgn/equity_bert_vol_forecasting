"""
SHAP feature-importance analysis for the LSTM volatility baseline.

Run after training:
    python evaluate_shap.py

What it produces inside runs/lstm_baseline/:
    shap_bar_{tag}.png       — ranked mean |SHAP| per feature (bar chart)
    shap_heatmap_{tag}.png   — feature × time-step importance heatmap
    shap_summary_{tag}.png   — beeswarm summary coloured by feature value

How it works:
    - Wraps LSTMModel to output a single scalar (mean prediction over horizon)
      so that shap.GradientExplainer produces one SHAP value per input element.
    - SHAP values have shape (N_explain, num_features, seq_len).
    - Feature-level importance = mean |SHAP| averaged over samples AND time steps.
    - Heatmap keeps the time-step dimension to reveal recency effects.
    - Beeswarm averages SHAP over time steps to reduce to 2-D for the plot.

Note on device:
    GradientExplainer requires autograd hooks that are not fully supported on
    Apple MPS. The script forces CPU for the SHAP computation.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from torch.utils.data import DataLoader

from src.mydataset import Dataset_SP500_1H
from src.model_lstm import LSTMModel


# CONFIG — must match train_lstm.py exactly

DATA_PATH = "/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/vola-bert/data/processed/ES_1h.parquet"
RUN_DIR = "runs/lstm_baseline/v2"

MODEL_CFG = dict(
    hidden_size=64,   # must match train_lstm.py
    num_layers=2,      # must match train_lstm.py
    dropout=0.2,       # must match train_lstm.py
)

HORIZONS = [
    {"lookback": 24, "forecast": 5},
    {"lookback": 50, "forecast": 10},
]

USE_TECHNICAL = True
USE_INTERDAY = True

# Number of background samples for baseline expectation
N_BACKGROUND = 100
# Number of test samples to compute SHAP values for
N_EXPLAIN = 300


# Helpers
def get_feature_names(use_technical=True, use_interday=True, use_events=False):
    """Returns ordered feature names matching the dataset's used_features list."""
    names = []
    if use_technical:
        names += [
            "middle_band", "upper_band", "lower_band",
            "momentum", "acceleration",
            "ema", "rsi", "log_return",
        ]
    if use_events:
        names += [
            "hours_to_event", "hours_since_event", "is_event_recent",
            "is_event", "is_event_window", "time_to_event",
        ]
    if use_interday:
        names += ["prev_r-1h", "prev_r-2h", "prev_r-4h", "prev_r-8h", "prev_r-24h"]
    names += ["log_volume"]
    return names


def build_test_dataset(lookback, forecast):
    return Dataset_SP500_1H(
        data_path=DATA_PATH,
        events_df=None,
        flag="test",
        size=(lookback, forecast),
        use_events=False,
        use_explainable=False,
        use_technical=USE_TECHNICAL,
        use_interday=USE_INTERDAY,
    )


# LSTM wrapper: collapses (B, 1, pred_len) → (B, 1) for GradientExplainer
class _LSTMScalarWrapper(nn.Module):
    """Returns the mean predicted volatility across the forecast horizon."""

    def __init__(self, model: LSTMModel):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)  # (B, 1, pred_len)
        return out.squeeze(1).mean(dim=-1, keepdim=True)  # (B, 1 ) !


# SHAP analysis
def run_shap(model, test_ds, feature_names, run_dir, tag):
    """
    Computes GradientExplainer SHAP values and saves three diagnostic plots.
    """
    # ── Collect all test tensors ──
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    all_x = torch.cat([x for x, _ in loader], dim=0)  # (N, F, L)
    N, F, L = all_x.shape
    print(f"  Test set: {N} samples, {F} features, {L} time steps")

    # Verify feature count
    if F != len(feature_names):
        print(f"  WARNING: model has {F} features but feature_names has "
              f"{len(feature_names)} entries. Results may be mislabeled.")

    rng = np.random.default_rng(42)
    n_bg = min(N_BACKGROUND, N)
    n_ex = min(N_EXPLAIN, N)
    bg_idx = rng.choice(N, size=n_bg, replace=False)
    ex_idx = rng.choice(N, size=n_ex, replace=False)

    background = all_x[bg_idx]  # (n_bg, F, L)
    explain_x = all_x[ex_idx]   # (n_ex, F, L)

    #  GradientExplainer expects a single scalar output per sample, so we wrap the model to average
    wrapper = _LSTMScalarWrapper(model).eval()
    print(f"  Computing SHAP values for {n_ex} samples "
          f"(background={n_bg}) ...")

    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(explain_x)

    # shap_values: list of length 1 (single output) → each (n_ex, F, L)
    if isinstance(shap_values, list):
        sv = np.array(shap_values[0])
    else:
        sv = np.array(shap_values)

    sv = np.squeeze(sv)  # (n_ex, F, L) if output had extra dim — should be (n_ex, F, L) for time-step analysis

    print(f"  SHAP values shape: {sv.shape}")

    #  1. Bar chart: mean |SHAP| per feature 
    feat_imp = np.abs(sv).mean(axis=(0, 2)).astype(np.float64)  # (F,)
    sorted_idx = np.argsort(feat_imp)  # ascending for horizontal bar

    fig, ax = plt.subplots(figsize=(9, max(4, F * 0.35)))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        [feat_imp[i] for i in sorted_idx],
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Mean |SHAP value| (averaged over samples & time steps)")
    ax.set_title(f"SHAP Feature Importance — LSTM {tag}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(run_dir, f"shap_bar_{tag}.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"  Saved: {bar_path}")

    #  2. Heatmap: feature × time-step 
    heatmap_vals = np.abs(sv).mean(axis=0)  # (F, L)

    fig, ax = plt.subplots(figsize=(max(10, L * 0.18), max(4, F * 0.35)))
    im = ax.imshow(heatmap_vals, aspect="auto", cmap="YlOrRd", origin="upper")
    ax.set_yticks(range(F))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel("Time step in lookback window (0 = oldest)")
    ax.set_title(f"SHAP Feature × Time-Step Importance — LSTM {tag}")
    plt.colorbar(im, ax=ax, label="Mean |SHAP|", shrink=0.8)
    plt.tight_layout()
    hm_path = os.path.join(run_dir, f"shap_heatmap_{tag}.png")
    plt.savefig(hm_path, dpi=150)
    plt.close()
    print(f"  Saved: {hm_path}")

    #  3. Beeswarm summary plot 
    # Average over time steps to reduce to 2-D
    sv_2d = sv.mean(axis=2)              # (n_ex, F)
    x_2d = explain_x.numpy().mean(axis=2)  # (n_ex, F)

    shap.summary_plot(
        sv_2d,
        x_2d,
        feature_names=feature_names,
        show=False,
        max_display=F,
        plot_size=(9, max(4, F * 0.3)),
    )
    plt.title(f"SHAP Beeswarm Summary — LSTM {tag}", pad=14)
    plt.tight_layout()
    sw_path = os.path.join(run_dir, f"shap_summary_{tag}.png")
    plt.savefig(sw_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {sw_path}")

    #  Console ranking
    desc_idx = np.argsort(feat_imp)[::-1]
    print(f"\n  Feature ranking by mean |SHAP| (top {min(F, 10)}):")
    max_imp = feat_imp[desc_idx[0]]
    for rank, i in enumerate(desc_idx[:10]):
        bar = "█" * int(feat_imp[i] / max_imp * 20)
        print(f"    {rank+1:>2}. {feature_names[i]:<22}  {feat_imp[i]:.5f}  {bar}")

    return feat_imp, sv


def main():
    device = torch.device("cpu")
    print(f"Device: {device} (forced CPU for SHAP autograd compatibility)\n")

    if not os.path.exists(RUN_DIR):
        print(f"Run directory not found: {RUN_DIR}")
        print("Run train_lstm.py first.")
        return

    feature_names = get_feature_names(
        use_technical=USE_TECHNICAL,
        use_interday=USE_INTERDAY,
        use_events=False,
    )
    print(f"Features ({len(feature_names)}): {feature_names}\n")

    for hz in HORIZONS:
        lookback = hz["lookback"]
        forecast = hz["forecast"]
        tag = f"{lookback}to{forecast}"

        print(f"\n{'=' * 60}")
        print(f"  Horizon: {lookback}h → {forecast}h")
        print(f"{'=' * 60}")

        # Look for checkpoint — trying tagged name first, fall back to generic
        ckpt_path = os.path.join(RUN_DIR, f"best_model_{tag}.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(RUN_DIR, "best_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found, skipping. Tried:")
            print(f"    {os.path.join(RUN_DIR, f'best_model_{tag}.pth')}")
            print(f"    {os.path.join(RUN_DIR, 'best_model.pth')}")
            continue

        # Build test dataset and load model
        test_ds = build_test_dataset(lookback, forecast)
        num_series = test_ds[0][0].shape[0]

        model = LSTMModel(
            num_series=num_series,
            pred_len=forecast,
            **MODEL_CFG,
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        print(f"  Loaded checkpoint: {ckpt_path}")
        print(f"  Model: {num_series} features, {MODEL_CFG['hidden_size']} hidden, "
              f"{MODEL_CFG['num_layers']} layers")

        run_shap(model, test_ds, feature_names, RUN_DIR, tag)

    print(f"\n{'=' * 60}")
    print(f"SHAP analysis complete. Plots saved to: {RUN_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        import traceback
        traceback.print_exc()