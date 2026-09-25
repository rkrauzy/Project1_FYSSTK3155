import numpy as np


def ols(X, y):
    return np.linalg.pinv(X) @ y


def ridge(X, y, lmbda):
    I = np.eye(X.shape[1])
    I[0, 0] = 0

    return np.linalg.pinv(X.T @ X + lmbda * I) @ X.T @ y
