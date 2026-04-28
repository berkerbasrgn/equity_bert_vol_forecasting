
import numpy as np
from scipy.stats import norm

def diebold_mariano_test(y_true, y_pred_1, y_pred_2, loss="mae", h=1):
    y_true = np.array(y_true)
    y_pred_1 = np.array(y_pred_1)
    y_pred_2 = np.array(y_pred_2)

    if y_true.ndim > 1:
        y_true = y_true.reshape(-1)
        y_pred_1 = y_pred_1.reshape(-1)
        y_pred_2 = y_pred_2.reshape(-1)

    e1 = y_true - y_pred_1
    e2 = y_true - y_pred_2

    if loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        d = (e1 ** 2) - (e2 ** 2)

    T = len(d)
    mean_d = np.mean(d)

    def autocovariance(x, lag):
        return np.cov(x[:-lag], x[lag:])[0, 1] if lag < len(x) else 0

    gamma_0 = np.var(d, ddof=1)
    gamma = 0

    for lag in range(1, h):
        weight = 1 - lag / h
        gamma += weight * autocovariance(d, lag)

    var_d = gamma_0 + 2 * gamma

    DM_stat = mean_d / np.sqrt(var_d / T)
    p_value = 2 * (1 - norm.cdf(abs(DM_stat)))

    return DM_stat, p_value