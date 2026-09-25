# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 2 - Snippet
# Role: Assisted with parts of the Ridge-versus-OLS comparison,
# including evaluation across lambda values and plotting.
# Verification: Reviewed and tested by the project authors.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.data import generate_data, design_matrix
from src.models import ols, ridge
from src.metrics import mse, r2


x, y, y_true = generate_data(n=100, sigma=0.1, seed=42)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

degrees = range(1, 16)
lambdas = [1e-6, 1e-4, 1e-2, 1, 100]

ols_mse = []
ols_r2 = []

mse_results = {}
r2_results = {}

for lmbda in lambdas:
    mse_results[lmbda] = []
    r2_results[lmbda] = []

for degree in degrees:
    X_train = design_matrix(x_train, degree)
    X_test = design_matrix(x_test, degree)

    mean_X = np.mean(X_train[:, 1:], axis=0)

    X_train[:, 1:] = X_train[:, 1:] - mean_X
    X_test[:, 1:] = X_test[:, 1:] - mean_X

    theta_ols = ols(X_train, y_train)
    y_test_ols = X_test @ theta_ols

    ols_mse.append(mse(y_test, y_test_ols))
    ols_r2.append(r2(y_test, y_test_ols))

    for lmbda in lambdas:
        theta_ridge = ridge(X_train, y_train, lmbda)
        y_test_ridge = X_test @ theta_ridge

        mse_results[lmbda].append(
            mse(y_test, y_test_ridge)
        )

        r2_results[lmbda].append(
            r2(y_test, y_test_ridge)
        )

plt.figure(figsize=(9, 6))

plt.plot(
    degrees,
    ols_mse,
    marker="o",
    linewidth=2,
    label="OLS"
)

for lmbda in lambdas:
    plt.plot(
        degrees,
        mse_results[lmbda],
        marker="o",
        linewidth=2,
        label=f"lambda = {lmbda:g}"
    )

plt.xlabel("Polynomial degree")
plt.ylabel("Test MSE")
plt.title("OLS and Ridge: Test MSE")
plt.xticks(degrees)
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))

plt.plot(
    degrees,
    ols_r2,
    marker="o",
    linewidth=2,
    label="OLS"
)

for lmbda in lambdas:
    plt.plot(
        degrees,
        r2_results[lmbda],
        marker="o",
        linewidth=2,
        label=f"lambda = {lmbda:g}"
    )

plt.xlabel("Polynomial degree")
plt.ylabel("R²")
plt.title("OLS and Ridge: Test R²")
plt.xticks(degrees)
plt.legend()
plt.tight_layout()
plt.show()
