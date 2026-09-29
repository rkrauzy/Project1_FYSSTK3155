import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ols, ridge
from src.gradient_descent import ols_gradient, ridge_gradient
from src.optimiser import optimise_optax
from src.plotting import save_fig

n = 100
Degree = 5
Lambda = 0.01

n_iterations = 10000
learning_rate = 0.01
tol = 1e-8

methods = [
    "momentum",
    "adagrad",
    "rmsprop",
    "adam",
]

def run_ols(X, y):
    """Run the optimisation methods for OLS regression.

    Parameters
    ----------
    X : ndarray
        Design matrix.
    y : ndarray
        Target values.

    Returns
    -------
    theta_exact : ndarray
        Closed-form OLS solution.
    results : dict
        Optimisation results for each method, including the final
        parameters, parameter history, errors, and number of iterations.
    """

    theta_exact = ols(X, y)

    theta0 = np.zeros(X.shape[1])

    results = {}

    for method in methods:

        gradient = lambda theta: ols_gradient(
            X,
            y,
            theta,
        )

        theta, history, errors = optimise_optax(
            gradient = gradient,
            n_iterations= n_iterations,
            learning_rate = learning_rate,
            optimizer_name = method,
            theta = theta0,
            theta_exact = theta_exact,
            tol = tol,
        )

        results[method] = {
            "theta": theta,
            "history": history,
            "errors": errors,
            "iterations": len(errors),
        }

    return theta_exact, results

def run_ridge(X, y):
    """Run the optimisation methods for Ridge regression.

    Parameters
    ----------
    X : ndarray
        Design matrix.
    y : ndarray
        Target values.

    Returns
    -------
    theta_exact : ndarray
        Closed-form Ridge solution.
    results : dict
        Optimisation results for each method, including the final
        parameters, parameter history, errors, and number of iterations.
    """

    theta_exact = ridge(
        X,
        y,
        Lambda,
    )

    theta0 = np.zeros(X.shape[1])

    results = {}

    for method in methods:

        gradient = lambda theta: ridge_gradient(
            X,
            y,
            theta,
            Lambda,
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

        results[method] = {
            "theta": theta,
            "history": history,
            "errors": errors,
            "iterations": len(errors),
        }

    return theta_exact, results

def plot_convergence(results, title, name):
    """
    Plot the relative error as a function of optimisation iterations.

    Parameters
    ----------
    results : dict
        Optimisation results for each method.
    title : str
        Title of the plot.
    name : str
        File name of the saved figure.
    """

    plt.figure()

    for method, result in results.items():

        errors = result["errors"]

        plt.semilogy(
            np.arange(1, len(errors) + 1),
            errors,
            label=method,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Relative error")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    save_fig(name)

def print_results(results):
    """
    Print the number of iterations and final error for each method.

    Parameters
    ----------
    results : dict
        Optimisation results for each optimisation method.

    LLM-assisted
    ------------
    Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
    Level: 4 - Substantial
    Role: Assisted with the formulation and implementation of the result 
        format.
    Verification: Reviewed and tested by the project authors.
    """

    print("\nResults:")

    for method, result in results.items():

        errors = result["errors"]

        print(
            f"{method:10s} "
            f"iterations = {len(errors):5d}, "
            f"final error = {errors[-1]:.3e}"
        )

def main():

    x, y, y_true = generate_data(
        n = n,
        sigma = 0.1,
        seed = 42,
    )

    x_train, x_test, y_train, y_test = split_data(
        x,
        y,
        seed = 42,
    )

    X = scale_matrix(design_matrix(
        x_train,
        degree = Degree,
    ))
    y, y_mean = center_y(y_train)

    theta_ols, ols_results = run_ols(
        X,
        y,
    )

    print("\nOLS")
    print_results(ols_results)

    plot_convergence(
        ols_results,
        "OLS convergence",
        "convergence_optax_ols",
    )

    theta_ridge, ridge_results = run_ridge(
        X,
        y,
    )

    print("\nRidge")
    print_results(ridge_results)

    plot_convergence(
        ridge_results,
        "Ridge convergence",
        "convergence_optax_ridge",
    )


if __name__ == "__main__":
    main()