import numpy as np


def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)


def r2(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    return 1 - ss_res / ss_tot
