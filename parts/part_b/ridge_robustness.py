# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 4 - Substantial
# Role: Suggested and substantially assisted with an additional robustness
# analysis of Ridge regression across different sample sizes and noise levels.
# The suggestion was reviewed and retained by the project authors to strengthen
# the empirical analysis and compare the results with Part A.
# Verification: Reviewed, executed and interpreted by the project authors.

import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ridge
from src.plotting import save_fig


degrees = range(1, 16)
lmbda = 1e-2
n_reps = 100


def repeated_test_mse(n, sigma):
    """
    Run repeated Ridge experiments and return median MSE with IQR.

    LLM-assisted
    ------------
    Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
    Level: 4 - Substantial
    Role: Substantially assisted with the repeated-experiment structure,
    aggregation of test errors, and calculation of median and interquartile
    ranges.
    Verification: Reviewed and tested by the project authors.
    """
    errors = np.zeros((n_reps, len(degrees)))

    for rep in range(n_reps):
        x, y, _ = generate_data(n=n, sigma=sigma, seed=rep)
        x_train, x_test, y_train, y_test = split_data(x, y, seed=rep)
        y_train_c, y_mean = center_y(y_train)

        for i, d in enumerate(degrees):
            X_train = design_matrix(x_train, d)
            X_test = design_matrix(x_test, d)
            X_train_s, X_test_s = scale_matrix(X_train, X_test)

            theta = ridge(X_train_s, y_train_c, lmbda)
            y_pred = X_test_s @ theta + y_mean

            residuals = y_test - y_pred
            errors[rep, i] = np.mean(residuals**2)

    return (
        np.median(errors, axis=0),
        np.percentile(errors, 25, axis=0),
        np.percentile(errors, 75, axis=0),
    )


n_values = [30, 60, 100, 200]

plt.figure(figsize=(9, 6))
for n in n_values:
    median_mse, q25, q75 = repeated_test_mse(n=n, sigma=0.1)
    line, = plt.plot(degrees, median_mse, "o-", linewidth=2, label=f"n = {n}")
    plt.fill_between(degrees, q25, q75, alpha=0.15, color=line.get_color())

plt.xlabel("Polynomial degree")
plt.ylabel("Test MSE")
plt.title(f"Ridge (λ={lmbda}): Effect of number of data points")
plt.xticks(degrees)
plt.yscale("log")
plt.legend()
plt.tight_layout()
save_fig("ridge_robustness_n")


sigma_values = [0.0, 0.05, 0.1, 0.2]

plt.figure(figsize=(9, 6))
for sigma in sigma_values:
    median_mse, q25, q75 = repeated_test_mse(n=100, sigma=sigma)
    line, = plt.plot(degrees, median_mse, "o-", linewidth=2, label=f"sigma = {sigma}")
    plt.fill_between(degrees, q25, q75, alpha=0.15, color=line.get_color())

plt.xlabel("Polynomial degree")
plt.ylabel("Test MSE")
plt.title(f"Ridge (λ={lmbda}): Effect of noise level")
plt.xticks(degrees)
plt.yscale("log")
plt.legend()
plt.tight_layout()
save_fig("ridge_robustness_sigma")
