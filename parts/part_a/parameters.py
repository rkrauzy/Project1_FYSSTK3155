# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 3 - Skeleton
# Role: Assisted with the overall structure of the coefficient analysis,
# including storing and plotting OLS coefficients across polynomial degrees.
# Verification: Reviewed, executed and interpreted by the project authors.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.data import generate_data, design_matrix
from src.models import ols


x, y, y_true = generate_data(n=100, sigma=0.1, seed=42)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

degrees = range(1, 16)

theta_values = np.full((15, 16), np.nan)

for degree in degrees:
    X_train = design_matrix(x_train, degree)

    mean_X = np.mean(X_train[:, 1:], axis=0)
    X_train[:, 1:] = X_train[:, 1:] - mean_X

    theta = ols(X_train, y_train)

    theta_values[degree - 1, :degree + 1] = theta

plt.figure(figsize=(10, 6))

for j in range(16):
    plt.plot(
        degrees,
        theta_values[:, j],
        marker="o",
        linewidth=2,
        label=f"theta_{j}"
    )

plt.xlabel("Polynomial degree")
plt.ylabel("Coefficient value")
plt.title("OLS: Coefficients as a function of polynomial degree")
plt.xticks(degrees)
plt.yscale("symlog", linthresh=1)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()
plt.show()
