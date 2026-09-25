import numpy as np


def ols_cost(X, y, theta):
    return np.mean((X @ theta - y) ** 2)


def ridge_cost(X, y, theta, lmbda):
    penalty = np.sum(theta[1:] ** 2)

    return np.mean((X @ theta - y) ** 2) + lmbda * penalty / len(y)


def ols_gradient(X, y, theta):
    n = len(y)

    return (2 / n) * X.T @ (X @ theta - y)


def ridge_gradient(X, y, theta, lmbda):
    n = len(y)

    penalty = theta.copy()
    penalty[0] = 0

    return (2 / n) * (
        X.T @ (X @ theta - y) + lmbda * penalty
    )


def gradient_descent(
    X,
    y,
    gradient,
    learning_rate,
    n_iterations,
    theta=None
):
    if theta is None:
        theta = np.zeros(X.shape[1])

    history = []

    for _ in range(n_iterations):
        grad = gradient(theta)

        theta = theta - learning_rate * grad
        history.append(theta.copy())

    return theta, np.array(history)