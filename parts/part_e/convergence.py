# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 3 - Skeleton
# Role: Assisted with the overall structure of the convergence analysis,
# including comparison of gradient descent with closed-form OLS and Ridge
# solutions, error tracking and plotting.
# Verification: Reviewed, executed and adapted by the project authors.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.data import generate_data, design_matrix
from src.models import ols, ridge
from src.gradient_descent import (
    gradient_descent,
    ols_gradient,
    ridge_gradient
)


x, y, y_true = generate_data(n=100, sigma=0.1, seed=42)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

degree = 5
lmbda = 0.01
n_iterations = 50000

X_train = design_matrix(x_train, degree)

mean_X = np.mean(X_train[:, 1:], axis=0)
X_train[:, 1:] = X_train[:, 1:] - mean_X

theta_ols = ols(X_train, y_train)
theta_ridge = ridge(X_train, y_train, lmbda)

H_ols = (2 / len(y_train)) * X_train.T @ X_train

I = np.eye(X_train.shape[1])
I[0, 0] = 0

H_ridge = (2 / len(y_train)) * (
    X_train.T @ X_train + lmbda * I
)

eta_ols = 1 / np.max(np.linalg.eigvalsh(H_ols))
eta_ridge = 1 / np.max(np.linalg.eigvalsh(H_ridge))

theta_gd_ols, history_ols = gradient_descent(
    X_train,
    y_train,
    lambda theta: ols_gradient(
        X_train,
        y_train,
        theta
    ),
    eta_ols,
    n_iterations
)

theta_gd_ridge, history_ridge = gradient_descent(
    X_train,
    y_train,
    lambda theta: ridge_gradient(
        X_train,
        y_train,
        theta,
        lmbda
    ),
    eta_ridge,
    n_iterations
)

error_ols = np.linalg.norm(
    history_ols - theta_ols,
    axis=1
)

error_ridge = np.linalg.norm(
    history_ridge - theta_ridge,
    axis=1
)

print("OLS final difference:")
print(np.linalg.norm(theta_gd_ols - theta_ols))

print()

print("Ridge final difference:")
print(np.linalg.norm(theta_gd_ridge - theta_ridge))

plt.figure(figsize=(9, 6))

plt.semilogy(
    error_ols,
    linewidth=2,
    label="OLS"
)

plt.semilogy(
    error_ridge,
    linewidth=2,
    label="Ridge"
)

plt.xlabel("Iteration")
plt.ylabel("Distance to closed-form solution")
plt.title("Gradient descent: Convergence to closed-form solutions")
plt.legend()
plt.tight_layout()
plt.show()