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

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data, runge
from src.models import ols
from src.plotting import save_fig

plt.rcdefaults()   # Matplotlib's default style for this figure, not the report style set in src/plotting.py

x, y, y_true = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)
y_train_c, y_mean = center_y(y_train)

x_plot = np.linspace(-1, 1, 500)
degrees = [2, 5, 8, 12]

plt.figure(figsize=(9, 6))
plt.scatter(x_train, y_train, s=10, alpha=0.5, color="gray", label="Train data")
plt.plot(x_plot, runge(x_plot), "k--", linewidth=2, label="True Runge function")

for d in degrees:
    X_train = design_matrix(x_train, d)
    X_plot = design_matrix(x_plot, d)
    X_train_s, X_plot_s = scale_matrix(X_train, X_plot)

    theta = ols(X_train_s, y_train_c)
    y_plot = X_plot_s @ theta + y_mean

    plt.plot(x_plot, y_plot, linewidth=2, label=f"OLS degree {d}")

plt.ylim(-0.3, 1.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("OLS: Polynomial fits to the Runge function")
plt.legend()
plt.tight_layout()
save_fig("ols_fitted_curves", print_size=False)
