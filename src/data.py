import numpy as np
from sklearn.model_selection import train_test_split as _sklearn_split

SEED = 42


def runge(x):
    return 1 / (1 + 25 * x**2)


def generate_data(n, sigma=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n)
    y_true = runge(x)
    y = y_true + rng.normal(0, sigma, n)
    return x, y, y_true


def design_matrix(x, degree):
    """Polynomial design matrix without intercept: columns x, x², ..., x^degree."""
    return np.column_stack([x**k for k in range(1, degree + 1)])


def scale_matrix(X_train, X_test=None):
    """Standardize all columns using training statistics only."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train_s = (X_train - mean) / std

    if X_test is None:
        return X_train_s

    return X_train_s, (X_test - mean) / std


def center_y(y_train):
    """Center y using training mean. Returns (y_centered, mean)."""
    mean = y_train.mean()
    return y_train - mean, mean


def split_data(x, y, test_size=0.2, seed=SEED):
    return _sklearn_split(x, y, test_size=test_size, random_state=seed)
