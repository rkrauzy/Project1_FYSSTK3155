import numpy as np
from sklearn.utils import resample

from src.data import design_matrix, scale_matrix, center_y, SEED
from src.models import ols, ridge, lasso_fit

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


def bootstrap_lambda(x_train, x_test, y_train, y_test, degree, lambdas, n_bootstraps, lasso=False):
    """
    Bootstrap estimate of test error, bias^2 and variance vs lambda at a fixed
    polynomial degree, for Ridge, or Lasso with lasso=True. A copy of bootstrap
    with the loop over degree replaced by a loop over lambda, following Case 1,
    Step 4 of week36tuesday.

    LLM-assisted
    ------------
    Tool: Claude, Opus 5.5 (Anthropic, September 2026)
    Level: 2 - Snippet
    Role: Assisted with copying bootstrap and replacing the loop over degree
    by a loop over lambda, with ridge or lasso_fit from src/models.py in place
    of ols, and with resetting the seed for every lambda so that all lambdas
    use the same bootstrap samples.
    Verification: Reviewed and executed by the project authors, and checked
    that error = bias + variance to machine precision.
    """
    y_test = y_test.reshape(-1, 1)
    error, bias, variance = (np.zeros(len(lambdas)) for _ in range(3))
    for j, lmbda in enumerate(lambdas):
        np.random.seed(SEED)          # the same bootstrap samples for every lambda
        y_pred = np.empty((y_test.shape[0], n_bootstraps))
        for i in range(n_bootstraps):
            x_, y_ = resample(x_train, y_train)
            X_train_s, X_test_s = scale_matrix(design_matrix(x_, degree), design_matrix(x_test, degree))
            y_c, y_mean = center_y(y_)
            theta = lasso_fit(X_train_s, y_c, lmbda) if lasso else ridge(X_train_s, y_c, lmbda)
            y_pred[:, i] = X_test_s @ theta + y_mean
        error[j] = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
        bias[j] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
        variance[j] = np.mean(np.var(y_pred, axis=1, keepdims=True))
    return error, bias, variance