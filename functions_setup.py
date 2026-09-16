import numpy as np


def runge(x):
    """ Evaluate the Runge function. """

    return 1 / (1 + 25 * x**2)

def design_matrix(x, degree):
    """ Construct the polynomial design matrix. """
    
    x = np.asarray(x).reshape(-1)

    return np.column_stack([
        x**i for i in range(degree + 1)
    ])


def center_data(X_train, X_test):
    """ Center non-intercept features using training-set means. """

    mean = X_train[:, 1:].mean(axis=0)

    X_train_centered = X_train.copy()
    X_test_centered = X_test.copy()

    X_train_centered[:, 1:] -= mean
    X_test_centered[:, 1:] -= mean

    return X_train_centered, X_test_centered, mean


def ols(X, y):
    """ Compute OLS coefficients with the Moore-Penrose pseudoinverse. """

    return np.linalg.pinv(
        X,
        rcond=1e-15
    ) @ y

def mse(y, y_pred):
    """ Compute mean squared error. """

    return np.mean(
        (y - y_pred)**2
    )

def r2(y, y_pred):
    """ Compute the R-squared score. """

    return (
        1
        - np.sum((y - y_pred)**2)
        / np.sum((y - np.mean(y))**2)
    )


