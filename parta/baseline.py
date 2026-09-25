import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.data import generate_data, design_matrix
from src.regression import ols, mse, r2


x, y, y_true = generate_data(n=100, sigma=0.1, seed=42)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

degrees = range(1, 16)

train_mse = []
test_mse = []
train_r2 = []
test_r2 = []

for degree in degrees:
    X_train = design_matrix(x_train, degree)
    X_test = design_matrix(x_test, degree)

    mean_X = np.mean(X_train[:, 1:], axis=0)

    X_train[:, 1:] = X_train[:, 1:] - mean_X
    X_test[:, 1:] = X_test[:, 1:] - mean_X

    theta = ols(X_train, y_train)

    y_train_pred = X_train @ theta
    y_test_pred = X_test @ theta

    train_mse.append(mse(y_train, y_train_pred))
    test_mse.append(mse(y_test, y_test_pred))

    train_r2.append(r2(y_train, y_train_pred))
    test_r2.append(r2(y_test, y_test_pred))

plt.figure(figsize=(9, 6))

plt.plot(
    degrees,
    train_mse,
    marker="o",
    linewidth=2,
    label="Training MSE"
)

plt.plot(
    degrees,
    test_mse,
    marker="o",
    linewidth=2,
    label="Test MSE"
)

plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.title("OLS: MSE as a function of polynomial degree")
plt.xticks(degrees)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))

plt.plot(
    degrees,
    train_r2,
    marker="o",
    linewidth=2,
    label="Training R²"
)

plt.plot(
    degrees,
    test_r2,
    marker="o",
    linewidth=2,
    label="Test R²"
)

plt.xlabel("Polynomial degree")
plt.ylabel("R²")
plt.title("OLS: R² as a function of polynomial degree")
plt.xticks(degrees)
plt.legend()
plt.tight_layout()
plt.show()