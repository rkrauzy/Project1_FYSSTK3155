"""
Bias-variance decomposition of the test error for Ridge and Lasso as a function
of lambda at fixed polynomial degree 12, estimated with the bootstrap, n = 100.
The OLS error at the same degree is shown as a reference.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, September 2026)
Level: 3 - Skeleton
Role: Assisted with adapting parts/part_c/bias_variance.py to lambda on the
    x-axis at fixed degree 12, with one panel for Ridge and one for Lasso and
    the OLS error as a reference line, using bootstrap_lambda from
    src/resampling.py. The idea of turning lambda at a fixed high degree
    follows Case 1, Step 4 of week36tuesday.

Verification:
    Reviewed and executed by the project authors. Error = bias^2 + variance
    to machine precision for both methods, and at the smallest lambda the
    Lasso error approaches the OLS error at degree 12.
"""
import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_data, split_data
from src.resampling import bootstrap, bootstrap_lambda
from src.plotting import save_fig

n, degree, n_bootstraps, sigma = 100, 12, 100, 0.1
lambdas = np.logspace(-6, 0, 20)

x, y, y_true = generate_data(n, sigma=sigma)
x_train, x_test, y_train, y_test = split_data(x, y)

np.random.seed(2026)
error_ols, bias_ols, variance_ols = bootstrap(x_train, x_test, y_train, y_test, degree, n_bootstraps)
print(f"OLS   degree {degree}: error {error_ols[-1]:.4f}  bias^2 {bias_ols[-1]:.4f}  var {variance_ols[-1]:.4f}")
print(f"   max relative difference |error - (bias^2 + var)| / error: "
      f"{np.max(np.abs(error_ols - (bias_ols + variance_ols)) / error_ols):.1e}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, name, lasso in zip(axes, ("Ridge", "Lasso"), (False, True)):
    error, bias, variance = bootstrap_lambda(x_train, x_test, y_train, y_test, degree, lambdas,
                                             n_bootstraps, lasso=lasso)

    best = np.argmin(error)
    print(f"{name:5s} degree {degree}: minimum error {error[best]:.4f} at lambda {lambdas[best]:.2e}, "
          f"bias^2 {bias[best]:.4f}  var {variance[best]:.4f}")
    print(f"   max relative difference |error - (bias^2 + var)| / error: "
          f"{np.max(np.abs(error - (bias + variance)) / error):.1e}")

    ax.plot(lambdas, error, "o-", label="Error")
    ax.plot(lambdas, bias, "s-", label=r"Bias$^2$ (+ $\sigma^2$)")
    ax.plot(lambdas, variance, "d-", label="Variance")
    ax.axhline(error_ols[-1], color="black", ls="--", label=f"OLS error, degree {degree}")
    ax.axhline(sigma**2, color="gray", ls=":", label=r"$\sigma^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(name)
    ax.set_xlabel(r"$\lambda$")
axes[0].set_ylabel("MSE decomposition")
axes[1].legend(loc="lower left")
fig.suptitle(f"Bias-variance tradeoff for Ridge and Lasso at degree {degree}")
plt.tight_layout()
save_fig("bias_variance_lambda")
