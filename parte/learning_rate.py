import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.data import generate_data, design_matrix
from src.regression import ols, ridge
from gradient_descent import (
    gradient_descent,
    ols_gradient,
    ridge_gradient
)


x, y, y_true = generate_data(n=100, sigma=0.1, seed=42)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

degree = 5
lmbda = 0.01
n_iterations = 500

X_train = design_matrix(x_train, degree)

mean_X = np.mean(X_train[:, 1:], axis=0)
X_train[:, 1:] = X_train[:, 1:] - mean_X

theta_ols = ols(X_train, y_train)
theta_ridge = ridge(X_train, y_train, lmbda)

H_ols = (2 / len(y_train)) * X_train.T @ X_train

I = np.eye(X_train.shape[1])
I[0, 0] = 0

H_ridge = (2 / len(y_train)) * (
    X_train.T @ X_train + lmbda * I
)

eta_max_ols = 2 / np.max(np.linalg.eigvalsh(H_ols))
eta_max_ridge = 2 / np.max(np.linalg.eigvalsh(H_ridge))

factors = [0.25, 0.5, 0.9, 1.01]

plt.figure(figsize=(9, 6))

for factor in factors:
    eta = factor * eta_max_ols

    theta, history = gradient_descent(
        X_train,
        y_train,
        lambda theta: ols_gradient(
            X_train,
            y_train,
            theta
        ),
        eta,
        n_iterations
    )

    error = np.linalg.norm(
        history - theta_ols,
        axis=1
    )

    plt.semilogy(
        error,
        linewidth=2,
        label=f"{factor} eta_max"
    )

plt.xlabel("Iteration")
plt.ylabel("Distance to closed-form solution")
plt.title("OLS: Effect of learning rate")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(9, 6))

for factor in factors:
    eta = factor * eta_max_ridge

    theta, history = gradient_descent(
        X_train,
        y_train,
        lambda theta: ridge_gradient(
            X_train,
            y_train,
            theta,
            lmbda
        ),
        eta,
        n_iterations
    )

    error = np.linalg.norm(
        history - theta_ridge,
        axis=1
    )

    plt.semilogy(
        error,
        linewidth=2,
        label=f"{factor} eta_max"
    )

plt.xlabel("Iteration")
plt.ylabel("Distance to closed-form solution")
plt.title("Ridge: Effect of learning rate")
plt.legend()
plt.tight_layout()
plt.show()

print("OLS eta_max:")
print(eta_max_ols)

print()

print("Ridge eta_max:")
print(eta_max_ridge)
