import numpy as np
import matplotlib.pyplot as plt

from functions_setup import runge, design_matrix, center_data, ols

data = np.load("results_part_a.npz")

x = data["x"]
y = data["y"]
x_train = data["x_train"]
y_train = data["y_train"]

x_plot = np.linspace(-1, 1, 1000).reshape(-1, 1)

plt.figure(figsize=(8, 5))

plt.scatter(x, y, s=18, alpha=0.5, label="Noisy observations")
plt.plot(x_plot, runge(x_plot), linewidth=2, label="True Runge function")

for degree in [12, 15]:

    X_train = design_matrix(x_train, degree)
    X_plot = design_matrix(x_plot, degree)

    X_train_centered, X_plot_centered, _ = center_data(X_train, X_plot)

    theta = ols(X_train_centered, y_train)

    plt.plot(
        x_plot, 
        X_plot_centered @ theta, 
        linewidth=2, 
        label=f"OLS degree {degree}"
        )

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
