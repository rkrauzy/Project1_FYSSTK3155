import numpy as np
import jax.numpy as jnp
import optax

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