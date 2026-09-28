import numpy as np


def ols(X, y):
    return np.linalg.pinv(X) @ y


def ridge(X, y, lmbda):
    n, p = X.shape
    return np.linalg.solve(X.T @ X + n * lmbda * np.eye(p), X.T @ y)
