
import numpy as np
from dm_test import diebold_mariano_test

#  load  saved predictions
y_true = np.load("/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/equity_bert_vol_forecasting/runs/v23/predictions/y_true_No_Events_24to5.npy")

bert_preds = np.load("//Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/equity_bert_vol_forecasting/runs/v23/predictions/bert_pred_Event_Type_Only_24to5.npy")
lstm_preds = np.load("/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/equity_bert_vol_forecasting/runs/lstm_baseline/v6/predictions/lstm_pred_24to5.npy")
naive_preds = np.load("/Users/burakberkerbasergun/Desktop/master thesis/VolaBERT/equity_bert_vol_forecasting/runs/lstm_baseline/v6/predictions/naive_preds_24to5.npy")


def interpret(name1, name2, dm, p):
    print(f"\n{name1} vs {name2}")
    print(f"DM: {dm:.4f} | p-value: {p:.6f}")

    if p < 0.05:
        if dm < 0:
            print(f" {name1} is significantly better")
        else:
            print(f" {name2} is significantly better")
    else:
        print(" No significant difference")


#  Comparisons 

# LSTM vs BERT
dm, p = diebold_mariano_test(y_true, lstm_preds, bert_preds, h=5)
interpret("LSTM", "BERT", dm, p)

# LSTM vs Naive
dm, p = diebold_mariano_test(y_true, lstm_preds, naive_preds, h=5)
interpret("LSTM", "Naive", dm, p)

# BERT vs Naive
dm, p = diebold_mariano_test(y_true, bert_preds, naive_preds, h=5)
interpret("BERT", "Naive", dm, p)