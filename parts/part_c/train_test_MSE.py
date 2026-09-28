import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ols
from src.plotting import save_fig

max_degree = 20
n_runs = 500

polydegree = np.arange(1, max_degree + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, n in zip(axes, (40, 100, 400)):
    TrainError = np.zeros(max_degree)
    TestError = np.zeros(max_degree)
    for run in range(n_runs):
        x, y, y_true = generate_data(n, seed=run)
        x_train, x_test, y_train, y_test = split_data(x, y, seed=run)
        y_c, y_mean = center_y(y_train)
        for degree in polydegree:
            X_train_s, X_test_s = scale_matrix(design_matrix(x_train, degree), design_matrix(x_test, degree))
            theta = ols(X_train_s, y_c)
            TrainError[degree - 1] += np.mean((y_train - (X_train_s @ theta + y_mean))**2) / n_runs
            TestError[degree - 1] += np.mean((y_test - (X_test_s @ theta + y_mean))**2) / n_runs

    print(f"n = {n}: minimum test MSE {TestError.min():.4f} at degree {np.argmin(TestError) + 1}")

    ax.plot(polydegree, TrainError, "o-", label="Training sample")
    ax.plot(polydegree, TestError, "o-", label="Test sample")
    ax.set_yscale("log")
    ax.set_title(f"$n = {n}$")
    ax.set_xlabel("Polynomial degree")
axes[0].set_ylabel("MSE")
axes[0].legend()
fig.suptitle("Test and training MSE as a function of model complexity")
plt.tight_layout()
save_fig("prediction_error_vs_complexity")