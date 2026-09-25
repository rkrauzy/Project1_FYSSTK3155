# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 4 - Substantial
# Role: Substantially assisted with the implementation of the gradient
# verification using JAX automatic differentiation for OLS and Ridge.
# Verification: Reviewed and executed by the project authors, with analytical
# and automatic gradients compared numerically to machine precision.

import numpy as np
import jax
import jax.numpy as jnp

from sklearn.model_selection import train_test_split

from src.data import generate_data, design_matrix
from src.gradient_descent import ols_gradient, ridge_gradient


jax.config.update("jax_enable_x64", True)


x, y, y_true = generate_data(n=100, sigma=0.1, seed=42)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

degree = 10
lmbda = 0.01

X_train = design_matrix(x_train, degree)

mean_X = np.mean(X_train[:, 1:], axis=0)
X_train[:, 1:] = X_train[:, 1:] - mean_X

theta = np.linspace(-0.5, 0.5, X_train.shape[1])

X_jax = jnp.array(X_train)
y_jax = jnp.array(y_train)
theta_jax = jnp.array(theta)


def ols_cost_jax(theta):
    """
    Compute the OLS cost using JAX.

    LLM-assisted
    ------------
    Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
    Level: 4 - Substantial
    Role: Assisted with the JAX-compatible cost implementation used for
    automatic differentiation.
    Verification: Compared against the independently implemented analytical
    OLS gradient by the project authors.
    """
    return jnp.mean((X_jax @ theta - y_jax) ** 2)


def ridge_cost_jax(theta):
    """
    Compute the Ridge cost using JAX.

    LLM-assisted
    ------------
    Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
    Level: 4 - Substantial
    Role: Assisted with the JAX-compatible Ridge cost implementation,
    including exclusion of the intercept from regularization.
    Verification: Compared against the independently implemented analytical
    Ridge gradient by the project authors.
    """
    penalty = jnp.sum(theta[1:] ** 2)

    return (
        jnp.mean((X_jax @ theta - y_jax) ** 2)
        + lmbda * penalty / len(y_train)
    )


ols_analytic = ols_gradient(
    X_train,
    y_train,
    theta
)

ols_autodiff = np.array(
    jax.grad(ols_cost_jax)(theta_jax)
)

ridge_analytic = ridge_gradient(
    X_train,
    y_train,
    theta,
    lmbda
)

ridge_autodiff = np.array(
    jax.grad(ridge_cost_jax)(theta_jax)
)

print("OLS maximum difference:")
print(np.max(np.abs(ols_analytic - ols_autodiff)))

print()

print("Ridge maximum difference:")
print(np.max(np.abs(ridge_analytic - ridge_autodiff)))
