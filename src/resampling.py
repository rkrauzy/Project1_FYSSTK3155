import numpy as np
from sklearn.utils import resample

from src.data import design_matrix, scale_matrix, center_y
from src.models import ols

def bootstrap(x_train, x_test, y_train, y_test, max_degree, n_bootstraps):
    """
    Bootstrap estimate of test error, bias^2 and variance vs polynomial degree (OLS).
    Reused code from week36tuesday, modified to fit our repo structure.
    Degrees start at 1 since our design matrix has no intercept column,
    so degree d is stored at index d - 1.

    LLM-assisted
    ------------
    Tool: Claude, Opus 5.5 (Anthropic, September 2026)
    Level: 2 - Snippet
    Role: Assisted with adapting the week 36 Tuesday bootstrap code to the
    repository structure, including scaling and centering of each bootstrap
    sample and reshaping y_test to a column vector.
    Verification: Reviewed and executed by the project authors, and checked
    that error = bias + variance to machine precision.
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