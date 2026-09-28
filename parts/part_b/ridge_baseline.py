# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 2 - Snippet
# Role: Assisted with parts of the Ridge-versus-OLS comparison,
# including evaluation across lambda values and plotting.
# Verification: Reviewed and tested by the project authors.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ols, ridge
from src.plotting import save_fig


x, y, y_true = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)
y_train_c, y_mean = center_y(y_train)

degrees = range(1, 16)
lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0]

ols_mse, ols_r2 = [], []
mse_results, r2_results = {lmb: [] for lmb in lambdas}, {lmb: [] for lmb in lambdas}

for d in degrees:
    X_train = design_matrix(x_train, d)
    X_test = design_matrix(x_test, d)
    X_train_s, X_test_s = scale_matrix(X_train, X_test)

    theta_ols = ols(X_train_s, y_train_c)
    y_pred_ols = X_test_s @ theta_ols + y_mean

    ols_mse.append(mean_squared_error(y_test, y_pred_ols))
    ols_r2.append(r2_score(y_test, y_pred_ols))

    for lmb in lambdas:
        theta_r = ridge(X_train_s, y_train_c, lmb)
        y_pred_r = X_test_s @ theta_r + y_mean

        mse_results[lmb].append(mean_squared_error(y_test, y_pred_r))
        r2_results[lmb].append(r2_score(y_test, y_pred_r))

plt.figure(figsize=(9, 6))
plt.semilogy(degrees, ols_mse, "o-", linewidth=2, label="OLS")
for lmb in lambdas:
    plt.semilogy(degrees, mse_results[lmb], "o-", linewidth=2, label=f"λ={lmb:g}")
plt.xlabel("Polynomial degree")
plt.ylabel("Test MSE")
plt.title("OLS and Ridge: Test MSE")
plt.xticks(degrees)
plt.legend()
plt.tight_layout()
save_fig("ridge_mse")

plt.figure(figsize=(9, 6))
plt.plot(degrees, ols_r2, "o-", linewidth=2, label="OLS")
for lmb in lambdas:
    plt.plot(degrees, r2_results[lmb], "o-", linewidth=2, label=f"λ={lmb:g}")
plt.xlabel("Polynomial degree")
plt.ylabel("R²")
plt.title("OLS and Ridge: Test R²")
plt.xticks(degrees)
plt.legend()
plt.tight_layout()
save_fig("ridge_r2")
