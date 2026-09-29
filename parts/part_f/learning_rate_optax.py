import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ols
from src.gradient_descent import ols_gradient
from src.optimiser import optimise_optax
from src.plotting import save_fig

n = 100
degree = 5

n_iterations = 10000
tol = 1e-8

methods = [
    "momentum",
    "adagrad",
    "rmsprop",
    "adam",
]

learning_rates = [
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
]


def run_sensitivity(X, y):
    """
    Evaluate the sensitivity of each optimiser to the learning rate.

    Parameters
    ----------
    X : ndarray
        Design matrix.
    y : ndarray
        Target values.

    Returns
    -------
    results : dict
        Optimisation results for each method and learning rate.
    """

    theta_exact = ols(X, y)

    theta0 = np.zeros(X.shape[1])

    results = {}

    for method in methods:

        results[method] = {}

        for learning_rate in learning_rates:

            gradient = lambda theta: ols_gradient(
                X,
                y,
                theta,
            )

            theta, history, errors = optimise_optax(
                gradient = gradient,
                n_iterations = n_iterations,
                learning_rate = learning_rate,
                optimizer_name = method,
                theta = theta0,
                theta_exact = theta_exact,
                tol = tol,
            )

            results[method][learning_rate] = {
                "theta": theta,
                "history": history,
                "errors": errors,
                "iterations": len(errors),
            }

    return results


def plot_final_error(results):
    """
    Plot the final relative error for different learning rates.

    Parameters
    ----------
    results : dict
        Optimisation results for each method and learning rate.
    """

    plt.figure()

    for method in methods:

        final_errors = []

        for learning_rate in learning_rates:

            errors = results[method][learning_rate]["errors"]

            final_errors.append(errors[-1])

        plt.loglog(
            learning_rates,
            final_errors,
            marker="o",
            label=method,
        )

    plt.xlabel("Initial learning rate")
    plt.ylabel("Final relative error")
    plt.title("Sensitivity to initial learning rate")
    plt.legend()
    plt.grid(True)

    save_fig("Learning_rate_optax_methods")

def print_results(results):
    """
    Print optimisation results for each method and learning rate.

    Parameters
    ----------
    results : dict
        Optimisation results for each method and learning rate.

    LLM-assisted
    ------------
    Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
    Level: 3 - Skeleton
    Role: Copied the format from the previous usage in the convergence_optax.py
        file.
    Verification: Reviewed and tested by the project authors.
    """

    for method in methods:

        print(f"\n{method}")

        for learning_rate in learning_rates:

            result = results[method][learning_rate]

            print(
                f"eta = {learning_rate:.0e}, "
                f"iterations = {result['iterations']:5d}, "
                f"error = {result['errors'][-1]:.3e}"
            )


def main():

    x, y, y_true = generate_data(
        n = n,
        sigma = 0.1,
    )

    x_train, x_test, y_train, y_test = split_data(
        x,
        y,
    )

    X = scale_matrix(design_matrix(
        x_train,
        degree = degree,
    ))
    y, y_mean = center_y(y_train)

    results = run_sensitivity(
        X,
        y,
    )

    print_results(results)

    plot_final_error(results)


if __name__ == "__main__":
    main()