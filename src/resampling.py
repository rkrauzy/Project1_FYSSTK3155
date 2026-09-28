import numpy as np
from sklearn.utils import resample

from src.data import design_matrix, scale_matrix, center_y
from src.models import ols

def bootstrap(x_train, x_test, y_train, y_test, max_degree, n_bootstraps):
    """
    Reused code fom week36tuesday. Modified to fit our repo structure.
    Had to change array size due to not using intercept in our design matrix. 

    """
    y_test = y_test.reshape(-1, 1)
    error, bias, variance = (np.zeros(max_degree) for _ in range(3))
    for degree in range(1, max_degree + 1):
        y_pred = np.empty((y_test.shape[0], n_bootstraps))
        for i in range(n_bootstraps):
            x_, y_ = resample(x_train, y_train)
            X_train_s, X_test_s = scale_matrix(design_matrix(x_, degree), design_matrix(x_test, degree))
            y_c, y_mean = center_y(y_)
            theta = ols(X_train_s, y_c)
            y_pred[:, i] = X_test_s @ theta + y_mean
        error[degree - 1] = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
        bias[degree - 1] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
        variance[degree - 1] = np.mean(np.var(y_pred, axis=1, keepdims=True))
    return error, bias, variance