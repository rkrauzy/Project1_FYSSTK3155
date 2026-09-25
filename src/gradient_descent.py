import numpy as np


def ols_cost(X, y, theta):
    """
    Compute the OLS mean squared error cost.

    LLM-assisted
    ------------
    Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
    Level: 4 - Substantial
    Role: Assisted with the formulation and implementation of the OLS cost.
    Verification: Reviewed and tested by the project authors.
    """
    return np.mean((X @ theta - y) ** 2)


def ridge_cost(X, y, theta, lmbda):
    """
    Compute the Ridge cost with an unregularized intercept.

    LLM-assisted
    ------------
    Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
    Level: 4 - Substantial
    Role: Assisted with the formulation and implementation of the Ridge cost.
    Verification: Reviewed and tested by the project authors.
    """
    penalty = np.sum(theta[1:] ** 2)

    return np.mean((X @ theta - y) ** 2) + lmbda * penalty / len(y)


def ols_gradient(X, y, theta):
    """
    Compute the analytical gradient of the OLS cost.

    LLM-assisted
    ------------
    Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
    Level: 4 - Substantial
    Role: Assisted with the analytical gradient formulation and implementation.
    Verification: Reviewed and later checked against automatic differentiation
    by the project authors.
    """
    n = len(y)

    return (2 / n) * X.T @ (X @ theta - y)


def ridge_gradient(X, y, theta, lmbda):
    """
    Compute the analytical gradient of the Ridge cost.

    LLM-assisted
    ------------
    Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
    Level: 4 - Substantial
    Role: Assisted with the analytical Ridge gradient, including exclusion of
    the intercept from regularization.
    Verification: Reviewed and later checked against automatic differentiation
    by the project authors.
    """
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
    """
    Optimize model parameters using fixed-learning-rate gradient descent.

    LLM-assisted
    ------------
    Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
    Level: 4 - Substantial
    Role: Substantially assisted with the implementation of the reusable
    gradient descent routine and parameter-history tracking.
    Verification: Reviewed, executed and compared with closed-form OLS and
    Ridge solutions by the project authors.
    """
    if theta is None:
        theta = np.zeros(X.shape[1])

    history = []

    for _ in range(n_iterations):
        grad = gradient(theta)

        theta = theta - learning_rate * grad
        history.append(theta.copy())

    return theta, np.array(history)