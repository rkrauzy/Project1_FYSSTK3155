import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.data import generate_data, design_matrix
from src.regression import ols, ridge
from gradient_descent import (
    gradient_descent,
    ols_gradient,
    ridge_gradient
)


jax.config.update("jax_enable_x64", True)


x, y, y_true = generate_data(n=100, sigma=0.1, seed=42)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

degree = 5
lmbda = 0.01
n_iterations = 50000

X_train = design_matrix(x_train, degree)

mean_X = np.mean(X_train[:, 1:], axis=0)
X_train[:, 1:] = X_train[:, 1:] - mean_X

theta_ols = ols(X_train, y_train)
theta_ridge = ridge(X_train, y_train, lmbda)

X_jax = jnp.array(X_train)
y_jax = jnp.array(y_train)


def ols_cost_jax(theta):
    return jnp.mean((X_jax @ theta - y_jax) ** 2)


def ridge_cost_jax(theta):
    penalty = jnp.sum(theta[1:] ** 2)

    return (
        jnp.mean((X_jax @ theta - y_jax) ** 2)
        + lmbda * penalty / len(y_train)
    )


ols_grad_jax = jax.jit(jax.grad(ols_cost_jax))
ridge_grad_jax = jax.jit(jax.grad(ridge_cost_jax))

ols_grad_jax(jnp.zeros(X_train.shape[1])).block_until_ready()
ridge_grad_jax(jnp.zeros(X_train.shape[1])).block_until_ready()

H_ols = (2 / len(y_train)) * X_train.T @ X_train

I = np.eye(X_train.shape[1])
I[0, 0] = 0

H_ridge = (2 / len(y_train)) * (
    X_train.T @ X_train + lmbda * I
)

eta_ols = 1 / np.max(np.linalg.eigvalsh(H_ols))
eta_ridge = 1 / np.max(np.linalg.eigvalsh(H_ridge))

theta_ols_analytic, history_ols_analytic = gradient_descent(
    X_train,
    y_train,
    lambda theta: ols_gradient(
        X_train,
        y_train,
        theta
    ),
    eta_ols,
    n_iterations
)

theta_ols_jax, history_ols_jax = gradient_descent(
    X_train,
    y_train,
    lambda theta: np.asarray(
        ols_grad_jax(jnp.asarray(theta))
    ),
    eta_ols,
    n_iterations
)

theta_ridge_analytic, history_ridge_analytic = gradient_descent(
    X_train,
    y_train,
    lambda theta: ridge_gradient(
        X_train,
        y_train,
        theta,
        lmbda
    ),
    eta_ridge,
    n_iterations
)

theta_ridge_jax, history_ridge_jax = gradient_descent(
    X_train,
    y_train,
    lambda theta: np.asarray(
        ridge_grad_jax(jnp.asarray(theta))
    ),
    eta_ridge,
    n_iterations
)

print("OLS analytic vs JAX:")
print(np.linalg.norm(theta_ols_analytic - theta_ols_jax))

print()

print("Ridge analytic vs JAX:")
print(np.linalg.norm(theta_ridge_analytic - theta_ridge_jax))

print()

print("OLS JAX vs closed form:")
print(np.linalg.norm(theta_ols_jax - theta_ols))

print()

print("Ridge JAX vs closed form:")
print(np.linalg.norm(theta_ridge_jax - theta_ridge))

ols_analytic_error = np.linalg.norm(
    history_ols_analytic - theta_ols,
    axis=1
)

ols_jax_error = np.linalg.norm(
    history_ols_jax - theta_ols,
    axis=1
)

ridge_analytic_error = np.linalg.norm(
    history_ridge_analytic - theta_ridge,
    axis=1
)

ridge_jax_error = np.linalg.norm(
    history_ridge_jax - theta_ridge,
    axis=1
)

plt.figure(figsize=(9, 6))

plt.semilogy(
    ols_analytic_error,
    linewidth=2,
    label="Analytic gradient"
)

plt.semilogy(
    ols_jax_error,
    linestyle="--",
    linewidth=2,
    label="JAX gradient"
)

plt.xlabel("Iteration")
plt.ylabel("Distance to closed-form solution")
plt.title("OLS: Analytic vs automatic differentiation")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))

plt.semilogy(
    ridge_analytic_error,
    linewidth=2,
    label="Analytic gradient"
)

plt.semilogy(
    ridge_jax_error,
    linestyle="--",
    linewidth=2,
    label="JAX gradient"
)

plt.xlabel("Iteration")
plt.ylabel("Distance to closed-form solution")
plt.title("Ridge: Analytic vs automatic differentiation")
plt.legend()
plt.tight_layout()
plt.show()
