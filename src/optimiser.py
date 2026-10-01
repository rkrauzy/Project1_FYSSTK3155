import numpy as np
import jax.numpy as jnp
import optax

from src.data import SEED
from src.gradient_descent import ridge_gradient, lasso_gradient

def optimise_optax(
    gradient,
    n_iterations,
    learning_rate,
    optimizer_name,
    theta,
    theta_exact=None,
    tol=None,
):
    """Optimise parameters using an Optax optimisation method.

    Parameters
    ----------
    gradient : callable
        Function that computes the gradient at the current parameter vector.
    n_iterations : int
        Maximum number of optimisation iterations.
    learning_rate : float
        Initial learning rate used by the optimiser.
    optimizer_name : str
        Optimisation method. Must be one of "momentum", "adagrad",
        "rmsprop", or "adam".
    theta : array_like
        Initial parameter vector.
    theta_exact : array_like, optional
        Closed-form solution used to calculate the relative error.
    tol : float, optional
        Relative error tolerance for early stopping.

    Returns
    -------
    theta : ndarray
        Final parameter vector.
    history : ndarray
        Parameter vector at each optimisation iteration.
    errors : ndarray
        Relative error with respect to the exact solution.

    LLM-assisted
    ------------
    Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
    Level: 4 - Substantial
    Role: Assisted with the implementation of the optimisation
        routine using Optax, including Momentum, AdaGrad, RMSprop and Adam.
    Verification: Reviewed and tested by the project authors.
    """

    theta = jnp.asarray(theta)

    if optimizer_name == "momentum":
        optimizer = optax.sgd(
            learning_rate=learning_rate,
            momentum=0.9,
        )

    elif optimizer_name == "adagrad":
        optimizer = optax.adagrad(
            learning_rate=learning_rate
        )

    elif optimizer_name == "rmsprop":
        optimizer = optax.rmsprop(
            learning_rate=learning_rate
        )

    elif optimizer_name == "adam":
        optimizer = optax.adam(
            learning_rate=learning_rate
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}"
        )

    opt_state = optimizer.init(theta)

    history = []
    errors = []

    if theta_exact is not None:
        theta_exact = jnp.asarray(theta_exact)

        exact_norm = float(jnp.linalg.norm(theta_exact))
        exact_norm = max(exact_norm, 1e-15)

    for _ in range(n_iterations):

        grad = jnp.asarray(
            gradient(np.asarray(theta))
        )

        updates, opt_state = optimizer.update(
            grad,
            opt_state,
            theta
        )

        theta = optax.apply_updates(
            theta,
            updates
        )

        theta_np = np.asarray(theta)

        history.append(theta_np.copy())

        if theta_exact is not None:

            error = (
                np.linalg.norm(
                    theta_np - np.asarray(theta_exact)
                )
                / exact_norm
            )

            errors.append(error)

            if tol is not None and error < tol:
                break

    return (
        np.asarray(theta),
        np.asarray(history),
        np.asarray(errors)
    )


def optimiser_step(method, theta, g, state, t, gamma, beta=0.9, rho=0.99,
                    beta1=0.9, beta2=0.999, eps=1e-8):
    """One update of theta from the gradient g at step t = 1, 2, ...; state carries the running quantities.

    From the week 38 exercises, following optimiser_step in the section
    Implementations of Chapter 4 in the lecture notes.
    """
    if method == "plain":                                    # Eq. (4.10)
        return theta - gamma * g, state
    if method == "momentum":                                 # Eq. (4.28)
        state["v"] = v = beta * state.get("v", 0.0) + gamma * g
        return theta - v, state
    if method == "adagrad":                                  # Eqs. (4.42) and (4.45)
        state["r"] = r = state.get("r", 0.0) + g * g
        return theta - gamma * g / (np.sqrt(r) + eps), state
    if method == "rmsprop":                                  # Eqs. (4.47) and (4.48)
        state["r"] = r = rho * state.get("r", 0.0) + (1.0 - rho) * g * g
        return theta - gamma * g / (np.sqrt(r) + eps), state
    if method == "adam":                                     # Eqs. (4.51), (4.52), (4.54), (4.55)
        state["m"] = m = beta1 * state.get("m", 0.0) + (1.0 - beta1) * g
        state["r"] = r = beta2 * state.get("r", 0.0) + (1.0 - beta2) * g * g
        m_hat, r_hat = m / (1.0 - beta1**t), r / (1.0 - beta2**t)
        return theta - gamma * m_hat / (np.sqrt(r_hat) + eps), state
    raise ValueError(f"unknown method {method}")


def make_batches(n, batch_size, rng):
    """Shuffle the indices and split them into minibatches. From the week 38 exercises."""
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def step_length(t, t0, t1):
    """The schedule gamma_t = t0 / (t + t1), Eq. (4.52). From the week 38 exercises."""
    return t0 / (t + t1)


def sgd(X, y, method="plain", n_epochs=50, batch_size=5, gamma=0.1, schedule=None,
        lmbda=0.0, lasso=False, seed=SEED, **kw):
    """Minibatch SGD with any optimiser. schedule=(t0, t1) replaces gamma by step_length.
    lmbda = 0 gives OLS, lmbda > 0 gives Ridge, or Lasso with lasso=True.
    Returns the iterate after every epoch.

    LLM-assisted
    ------------
    Tool: Claude, Opus 5.5 (Anthropic, September 2026)
    Level: 2 - Snippet
    Role: The function is our own sgd from the week 38 exercises. Claude
        assisted with moving it to src and replacing the notebook's gradient
        with ridge_gradient or lasso_gradient from src/gradient_descent.py.
    Verification: Reviewed by the project authors. With method="plain" and
        batch_size = n it reproduces full-batch gradient descent and converges
        to the closed-form OLS and Ridge solutions.
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    theta, state, t = np.zeros(p), {}, 0
    history = [theta.copy()]
    for epoch in range(n_epochs):
        for batch in make_batches(n, batch_size, rng):
            t += 1
            if lasso:
                g = lasso_gradient(X[batch], y[batch], theta, lmbda)
            else:
                g = ridge_gradient(X[batch], y[batch], theta, lmbda)
            gamma_t = gamma if schedule is None else step_length(t, *schedule)
            theta, state = optimiser_step(method, theta, g, state, t, gamma_t, **kw)
        history.append(theta.copy())
    return np.array(history)