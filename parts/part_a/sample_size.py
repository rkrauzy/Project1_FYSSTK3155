# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 2 - Snippet
# Role: Assisted with parts of the repeated-sampling analysis for studying
# the effect of sample size on OLS test error.
# Verification: Reviewed and tested by the project authors.

import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ols
from src.plotting import save_fig


n_values = [30, 60, 100, 200]
degrees = range(1, 16)
sigma = 0.1
n_reps = 100

plt.figure(figsize=(9, 6))

for n in n_values:
    errors = np.zeros((n_reps, len(degrees)))

    for rep in range(n_reps):
        x, y, y_true = generate_data(n=n, sigma=sigma, seed=rep)
        x_train, x_test, y_train, y_test = split_data(x, y, seed=rep)
        y_train_c, y_mean = center_y(y_train)

        for i, d in enumerate(degrees):
            X_train = design_matrix(x_train, d)
            X_test = design_matrix(x_test, d)
            X_train_s, X_test_s = scale_matrix(X_train, X_test)

            theta = ols(X_train_s, y_train_c)
            y_pred = X_test_s @ theta + y_mean

            residuals = y_test - y_pred
            errors[rep, i] = np.mean(residuals**2)

    median_mse = np.median(errors, axis=0)
    q25 = np.percentile(errors, 25, axis=0)
    q75 = np.percentile(errors, 75, axis=0)

    line, = plt.plot(degrees, median_mse, "o-", linewidth=2, label=f"n = {n}")
    plt.fill_between(degrees, q25, q75, alpha=0.15, color=line.get_color())

plt.xlabel("Polynomial degree")
plt.ylabel("Test MSE")
plt.title("OLS: Effect of number of data points")
plt.xticks(degrees)
plt.yscale("log")
plt.legend()
plt.tight_layout()
save_fig("ols_sample_size")
