# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 4 - Substantial
# Role: Suggested and substantially assisted with an additional condition-number
# analysis comparing centered and uncentered polynomial design matrices.
# The suggestion was reviewed and retained by the project authors to provide
# quantitative support for the discussion of centering and numerical stability.
# Verification: Reviewed, executed and interpreted by the project authors.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.data import generate_data, design_matrix


x, y, y_true = generate_data(n=100, sigma=0.1, seed=42)

x_train, x_test = train_test_split(
    x,
    test_size=0.2,
    random_state=42
)

degrees = range(1, 16)

cond_uncentered = []
cond_centered = []

for degree in degrees:
    X_train = design_matrix(x_train, degree)

    cond_uncentered.append(np.linalg.cond(X_train))

    X_centered = X_train.copy()

    mean_X = np.mean(X_centered[:, 1:], axis=0)
    X_centered[:, 1:] = X_centered[:, 1:] - mean_X

    cond_centered.append(np.linalg.cond(X_centered))

plt.figure(figsize=(9, 6))

plt.plot(
    degrees,
    cond_uncentered,
    marker="o",
    linewidth=2,
    label="Uncentered"
)

plt.plot(
    degrees,
    cond_centered,
    marker="o",
    linewidth=2,
    label="Centered"
)

plt.xlabel("Polynomial degree")
plt.ylabel("Condition number")
plt.title("OLS: Effect of centering on numerical conditioning")
plt.xticks(degrees)
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()
