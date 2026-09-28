# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 2 - Snippet
# Role: Assisted with adapting parts of the repeated-sampling analysis
# to study the effect of different noise levels on OLS test error.
# Verification: Reviewed and tested by the project authors.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.data import generate_data, design_matrix
from src.models import ols
from src.metrics import mse


sigma_values = [0.0, 0.05, 0.1, 0.2]
degrees = range(1, 16)
n = 100
n_reps = 100

plt.figure(figsize=(9, 6))

for sigma in sigma_values:
    errors = np.zeros((n_reps, len(degrees)))

    for rep in range(n_reps):
        x, y, y_true = generate_data(
            n=n,
            sigma=sigma,
            seed=rep
        )

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=rep
        )

        for i, degree in enumerate(degrees):
            X_train = design_matrix(x_train, degree)
            X_test = design_matrix(x_test, degree)

            mean_X = np.mean(X_train[:, 1:], axis=0)

            X_train[:, 1:] = X_train[:, 1:] - mean_X
            X_test[:, 1:] = X_test[:, 1:] - mean_X

            theta = ols(X_train, y_train)

            y_test_pred = X_test @ theta

            errors[rep, i] = mse(y_test, y_test_pred)

    median_mse = np.median(errors, axis=0)
    q25 = np.percentile(errors, 25, axis=0)
    q75 = np.percentile(errors, 75, axis=0)

    line, = plt.plot(
        degrees,
        median_mse,
        marker="o",
        linewidth=2,
        label=f"sigma = {sigma}"
    )

    plt.fill_between(
        degrees,
        q25,
        q75,
        alpha=0.15,
        color=line.get_color()
    )

plt.xlabel("Polynomial degree")
plt.ylabel("Test MSE")
plt.title("OLS: Effect of noise level")
plt.xticks(degrees)
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()
