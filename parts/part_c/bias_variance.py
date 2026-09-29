"""
Bias-variance decomposition of the OLS test error as a function of polynomial
degree, estimated with the bootstrap, for n = 40, 100 and 400.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, September 2026)
Level: 3 - Skeleton
Role: Assisted with the overall structure of the analysis using the shared
    bootstrap function, the comparison of several n in one figure and the
    reference line for sigma^2. The structure follows Steps 2 and 3 of
    week36tuesday.
Verification: 
    Reviewed, executed and interpreted by the project authors.
"""
import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_data, split_data
from src.resampling import bootstrap
from src.plotting import save_fig

max_degree, n_bootstraps, sigma = 20, 100, 0.1
polydegree = np.arange(1, max_degree + 1)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, n in zip(axes, (40, 100, 400)):
    x, y, y_true = generate_data(n, sigma=sigma)
    x_train, x_test, y_train, y_test = split_data(x, y)
    np.random.seed(2026)
    error, bias, variance = bootstrap(x_train, x_test, y_train, y_test, max_degree, n_bootstraps)

    print(f"n = {n}: minimum error {error.min():.4f} at degree {np.argmin(error) + 1}")
    for d in (1, 5, 10, 15, 20):
        print(f"   degree {d:2d}: error {error[d - 1]:.4f}  bias^2 {bias[d - 1]:.4f}  var {variance[d - 1]:.4f}")

    ax.plot(polydegree, error, "o-", label="Error")
    ax.plot(polydegree, bias, "s-", label=r"Bias$^2$ (+ $\sigma^2$)")
    ax.plot(polydegree, variance, "d-", label="Variance")
    ax.axhline(sigma**2, color="gray", ls=":", label=r"$\sigma^2$")
    ax.set_yscale("log")
    ax.set_title(f"$n = {n}$")
    ax.set_xlabel("Polynomial degree")
axes[0].set_ylabel("MSE decomposition")
axes[0].legend()
fig.suptitle("Bias-variance tradeoff for OLS with bootstrap")
plt.tight_layout()
save_fig("bias_variance_tradeoff")
