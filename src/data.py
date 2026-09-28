import numpy as np


def runge(x):
    return 1 / (1 + 25 * x**2)


def generate_data(n, sigma=0.1, seed=42):
    rng = np.random.default_rng(seed)

    x = np.linspace(-1, 1, n)
    y_true = runge(x)

    noise = rng.normal(0, sigma, n)
    y = y_true + noise

    return x, y, y_true


def design_matrix(x, degree):
    X = np.ones((len(x), degree + 1))

    for i in range(1, degree + 1):
        X[:, i] = x**i

    return X