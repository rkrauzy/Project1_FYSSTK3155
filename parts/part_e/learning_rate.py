# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 3 - Skeleton
# Role: Assisted with the overall structure of the learning-rate analysis,
# including Hessian-based stability limits, comparison of several learning
# rates, error tracking and plotting for OLS and Ridge.
# Verification: Reviewed, executed and adapted by the project authors.

import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ols, ridge
from src.plotting import save_fig
from src.gradient_descent import (
    gradient_descent,
    ols_gradient,
    ridge_gradient
)

x, y, y_true = generate_data(n=100, sigma=0.1)

x_train, x_test, y_train, y_test = split_data(x, y)

degree = 5
lmbda = 0.01
n_iterations = 500

X_train = scale_matrix(design_matrix(x_train, degree))
y_train, y_mean = center_y(y_train)

theta_ols = ols(X_train, y_train)
theta_ridge = ridge(X_train, y_train, lmbda)

H_ols = (2 / len(y_train)) * X_train.T @ X_train

H_ridge = (2 / len(y_train)) * X_train.T @ X_train + 2 * lmbda * np.eye(X_train.shape[1])

eta_max_ols = 2 / np.max(np.linalg.eigvalsh(H_ols))
eta_max_ridge = 2 / np.max(np.linalg.eigvalsh(H_ridge))

factors = [0.25, 0.5, 0.9, 1.01]

plt.figure(figsize=(9, 6))

for factor in factors:
    eta = factor * eta_max_ols

    theta, history = gradient_descent(
        X_train,
        y_train,
        lambda theta: ols_gradient(
            X_train,
            y_train,
            theta
        ),
        eta,
        n_iterations
    )

    error = np.linalg.norm(
        history - theta_ols,
        axis=1
    )

    plt.semilogy(
        error,
        linewidth=2,
        label=f"{factor} eta_max"
    )

plt.xlabel("Iteration")
plt.ylabel("Distance to closed-form solution")
plt.title("OLS: Effect of learning rate")
plt.legend()
plt.tight_layout()
save_fig("gd_learning_rate_ols")


plt.figure(figsize=(9, 6))

for factor in factors:
    eta = factor * eta_max_ridge

    theta, history = gradient_descent(
        X_train,
        y_train,
        lambda theta: ridge_gradient(
            X_train,
            y_train,
            theta,
            lmbda
        ),
        eta,
        n_iterations
    )

    error = np.linalg.norm(
        history - theta_ridge,
        axis=1
    )

    plt.semilogy(
        error,
        linewidth=2,
        label=f"{factor} eta_max"
    )

plt.xlabel("Iteration")
plt.ylabel("Distance to closed-form solution")
plt.title("Ridge: Effect of learning rate")
plt.legend()
plt.tight_layout()
save_fig("gd_learning_rate_ridge")

print("OLS eta_max:")
print(eta_max_ols)

print()

print("Ridge eta_max:")
print(eta_max_ridge)
