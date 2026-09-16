import numpy as np
from sklearn.model_selection import train_test_split

from functions_setup import runge, design_matrix, center_data, ols, mse, r2


rng = np.random.default_rng(2026)

n = 100
sigma = 0.1

degrees = np.arange(1, 16)

x = np.linspace(-1, 1, n).reshape(-1, 1)
epsilon = rng.normal(0, sigma, size=(n, 1))
y = runge(x) + epsilon

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2026
)

mse_train = []
mse_test = []

r2_train = []
r2_test = []

max_theta = []

condition_numbers = []
condition_numbers_uncentered = []

theta_values = np.full((len(degrees), 16), np.nan)

for degree in degrees:

    X_train = design_matrix(x_train, degree)
    X_test = design_matrix(x_test, degree)

    condition_numbers_uncentered.append(np.linalg.cond(X_train))

    X_train_centered, X_test_centered, _ = center_data(X_train, X_test)

    condition_numbers.append(np.linalg.cond(X_train_centered))

    theta = ols(X_train_centered, y_train)

    y_train_pred = X_train_centered @ theta
    y_test_pred = X_test_centered @ theta

    mse_train.append(mse(y_train, y_train_pred))
    mse_test.append(mse(y_test, y_test_pred))

    r2_train.append(r2(y_train, y_train_pred))
    r2_test.append(r2(y_test, y_test_pred))

    max_theta.append(np.max(np.abs(theta)))

    theta_values[degree - 1, :degree + 1] = theta.ravel()


np.savez(
    "results_part_a.npz",
    n=n,
    sigma=sigma,
    degrees=degrees,
    x=x,
    y=y,
    x_train=x_train,
    x_test=x_test,
    y_train=y_train,
    y_test=y_test,
    mse_train=mse_train,
    mse_test=mse_test,
    r2_train=r2_train,
    r2_test=r2_test,
    max_theta=max_theta,
    condition_numbers=condition_numbers,
    condition_numbers_uncentered=condition_numbers_uncentered,
    theta_values=theta_values,
)

print("Saved results_part_a.npz")

lowest_degree = degrees[np.argmin(mse_test)]

print("\nLowest observed test MSE for this particular split:")
print(f"degree {lowest_degree}, MSE = {np.min(mse_test):.6f}")

print("\nHigh-degree behaviour:")
for degree in [12, 13, 14, 15]:
    i = degree - 1
    print(
        f"degree {degree:2d}: "
        f"train MSE = {mse_train[i]:.6f}, "
        f"test MSE = {mse_test[i]:.6f}, "
        f"test R² = {r2_test[i]:.4f}"
    )

print("\nMaximum coefficient magnitude:")
print(f"degree 1  = {max_theta[0]:.3e}")
print(f"degree 15 = {max_theta[-1]:.3e}")
