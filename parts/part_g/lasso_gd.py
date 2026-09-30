"""
Lasso regression with our own gradient descent methods from parts e) and f),
compared with Scikit-Learn's Lasso and with closed-form OLS and Ridge at
degree 5. Also checks the analytical Lasso gradient against JAX, including
at theta = 0 where |theta| is not differentiable.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, September 2026)
Level: 3 - Skeleton
Role: Assisted with assembling the script from existing code: the gradient
    check from parts/part_e/gradient_check.py, plain gradient descent with
    eta = 1/lambda_max of the OLS Hessian from parts/part_e/convergence.py,
    the Optax methods from parts/part_f/convergence_optax.py, the reference
    solution from lasso_fit in chapter 3 of the lecture notes (alpha = lambda/2),
    and the test predictions from cv_own_vs_sklearn.py.

Verification:
    Reviewed and executed by the project authors. The analytical and JAX
    gradients agree to 1e-16 away from theta = 0 and differ by exactly lambda
    at theta = 0, and all methods reach a cost within 2e-4 of Scikit-Learn's.
"""

import numpy as np
import jax
import jax.numpy as jnp
from sklearn.linear_model import Lasso

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ols, ridge
from src.gradient_descent import gradient_descent, lasso_cost, lasso_gradient
from src.optimiser import optimise_optax

jax.config.update("jax_enable_x64", True)

n = 100
x, y, y_true = generate_data(n)
x_train, x_test, y_train, y_test = split_data(x, y)

degree = 5
lmbda = 0.01
X_train_s, X_test_s = scale_matrix(design_matrix(x_train, degree), design_matrix(x_test, degree))
y_train_c, y_mean = center_y(y_train)

n_iterations = 10000
learning_rate = 0.01
tol = 1e-8

X_jax, y_jax = jnp.array(X_train_s), jnp.array(y_train_c)
lasso_cost_jax = lambda theta: jnp.mean((X_jax @ theta - y_jax) ** 2) + lmbda * jnp.sum(jnp.abs(theta))
for theta in (np.linspace(-0.5, 0.5, degree) + 0.05, np.zeros(degree)):
    diff = np.max(np.abs(lasso_gradient(X_train_s, y_train_c, theta, lmbda)
                         - np.array(jax.grad(lasso_cost_jax)(jnp.array(theta)))))
    print(f"gradient check at theta = {np.round(theta, 2)}: max difference {diff:.2e}")
print(f"jax.grad of |theta| at 0: {jax.grad(jnp.abs)(0.0)}, np.sign(0): {np.sign(0.0)}\n")

theta_sklearn = Lasso(alpha=lmbda / 2, fit_intercept=False, max_iter=100000,
                      tol=1e-10).fit(X_train_s, y_train_c).coef_
for name, theta in (("OLS", ols(X_train_s, y_train_c)), ("Ridge", ridge(X_train_s, y_train_c, lmbda)),
                    ("sklearn", theta_sklearn)):
    print(f"{name:10s} test MSE {np.mean((X_test_s @ theta + y_mean - y_test) ** 2):.6f}, "
          f"nonzero {np.sum(theta != 0)}")
print(f"sklearn Lasso cost {lasso_cost(X_train_s, y_train_c, theta_sklearn, lmbda):.6f}\n")

grad = lambda theta: lasso_gradient(X_train_s, y_train_c, theta, lmbda)
eta = 1 / np.max(np.linalg.eigvalsh((2 / len(y_train_c)) * X_train_s.T @ X_train_s))

for method in ["plain", "momentum", "adagrad", "rmsprop", "adam"]:
    if method == "plain":
        theta, history = gradient_descent(X_train_s, y_train_c, grad, eta, n_iterations)
        errors = np.linalg.norm(history - theta_sklearn, axis=1) / np.linalg.norm(theta_sklearn)
    else:
        theta, history, errors = optimise_optax(grad, n_iterations, learning_rate, method,
                                                np.zeros(degree), theta_sklearn, tol)
    print(f"{method:10s} test MSE {np.mean((X_test_s @ theta + y_mean - y_test) ** 2):.6f}, "
          f"nonzero {np.sum(theta != 0)}, iterations {len(errors):5d}, "
          f"relative error {errors[-1]:.3e}, cost {lasso_cost(X_train_s, y_train_c, theta, lmbda):.6f}")
