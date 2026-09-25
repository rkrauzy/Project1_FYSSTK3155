# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 4 - Substantial
# Role: Suggested and substantially assisted with an additional visualization
# comparing OLS polynomial fits with the true Runge function.
# The suggestion was reviewed and retained by the project authors to provide
# additional visual evidence for the behaviour of higher-degree models.
# Verification: Reviewed, executed and interpreted by the project authors.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.data import generate_data, design_matrix, runge
from src.models import ols


x, y, y_true = generate_data(n=100, sigma=0.1, seed=42)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

x_plot = np.linspace(-1, 1, 500)
degrees = [8, 15]

plt.figure(figsize=(9, 6))

plt.scatter(
    x,
    y,
    alpha=0.6,
    label="Noisy data"
)

plt.plot(
    x_plot,
    runge(x_plot),
    linewidth=2,
    label="True Runge function"
)

for degree in degrees:
    X_train = design_matrix(x_train, degree)
    X_plot = design_matrix(x_plot, degree)

    mean_X = np.mean(X_train[:, 1:], axis=0)

    X_train[:, 1:] = X_train[:, 1:] - mean_X
    X_plot[:, 1:] = X_plot[:, 1:] - mean_X

    theta = ols(X_train, y_train)
    y_plot = X_plot @ theta

    plt.plot(
        x_plot,
        y_plot,
        linewidth=2,
        label=f"OLS degree {degree}"
    )

plt.xlabel("x")
plt.ylabel("y")
plt.title("OLS: Polynomial fits to the Runge function")
plt.legend()
plt.tight_layout()
plt.show()
