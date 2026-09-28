# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 3 - Skeleton
# Role: Assisted with the overall structure of the OLS train/test analysis,
# including feature centering, evaluation and plotting.
# Verification: Reviewed, executed and adapted by the project authors.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ols
from src.plotting import save_fig


x, y, y_true = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)
y_train_c, y_mean = center_y(y_train)

degrees = range(1, 16)
mse_train, mse_test = [], []
r2_train, r2_test = [], []

for d in degrees:
    X_train = design_matrix(x_train, d)
    X_test = design_matrix(x_test, d)
    X_train_s, X_test_s = scale_matrix(X_train, X_test)

    theta = ols(X_train_s, y_train_c)

    y_pred_train = X_train_s @ theta + y_mean
    y_pred_test = X_test_s @ theta + y_mean

    mse_train.append(mean_squared_error(y_train, y_pred_train))
    mse_test.append(mean_squared_error(y_test, y_pred_test))
    r2_train.append(r2_score(y_train, y_pred_train))
    r2_test.append(r2_score(y_test, y_pred_test))

plt.figure(figsize=(9, 6))
plt.semilogy(degrees, mse_train, "o-", linewidth=2, label="Train MSE")
plt.semilogy(degrees, mse_test, "o-", linewidth=2, label="Test MSE")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.title("OLS: MSE as a function of polynomial degree")
plt.xticks(degrees)
plt.legend()
plt.tight_layout()
save_fig("ols_mse")

plt.figure(figsize=(9, 6))
plt.plot(degrees, r2_train, "o-", linewidth=2, label="Train R²")
plt.plot(degrees, r2_test, "o-", linewidth=2, label="Test R²")
plt.xlabel("Polynomial degree")
plt.ylabel("R²")
plt.title("OLS: R² as a function of polynomial degree")
plt.xticks(degrees)
plt.legend()
plt.tight_layout()
save_fig("ols_r2")
