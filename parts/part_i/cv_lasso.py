"""
Cross-validated MSE for Lasso regression as a function of both polynomial
degree and lambda, for k = 5 and k = 10, shown as heatmaps.

A copy of parts/part_d/cv_ridge.py with Ridge replaced by Lasso.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, September 2026)
Level: 2 - Snippet
Role: Assisted with adapting cv_ridge.py to Lasso: alpha = lambda / 2 since
    Scikit-Learn's Lasso divides the squared error by 2n (lasso_fit in
    Chapter 3 of the lecture notes), max_iter = 100000 as in the lecture notes,
    a lambda grid from 1e-6 to 1 that contains the CV minimum and ends above
    lambda_max = (2/n)||X^T y||_inf = 0.46 (Proposition 3.8), and n_jobs=-1 in
    cross_val_score to run the folds in parallel.

Verification:
    Reviewed and executed by the project authors. The CV minimum lies inside
    the lambda grid, at degree 12 for both k = 5 and k = 10, and above
    lambda_max every coefficient is zero and the model predicts the mean.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score

from src.data import generate_data, design_matrix
from src.plotting import save_fig

n = 100
x, y, y_true = generate_data(n)

max_degree = 20
polydegree = np.arange(1, max_degree + 1)

nlambdas = 50
lambdas = np.logspace(-6, 0, nlambdas)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, k in zip(axes, (5, 10)):
    kfold = KFold(n_splits=k, shuffle=True, random_state=2026)
    mse_sklearn = np.zeros((max_degree, nlambdas))
    for degree in polydegree:
        for i, lmb in enumerate(lambdas):
            pipe = make_pipeline(StandardScaler(), Lasso(alpha=lmb / 2, max_iter=100000))
            scores = cross_val_score(pipe, design_matrix(x, degree), y, cv=kfold,
                                     scoring="neg_mean_squared_error", n_jobs=-1)
            mse_sklearn[degree - 1, i] = np.mean(-scores)

    best_deg, best_lmb = np.unravel_index(np.argmin(mse_sklearn), mse_sklearn.shape)
    print(f"k = {k:2d}: best degree {best_deg + 1}, best lambda {lambdas[best_lmb]:.4g}, "
          f"cross-validated MSE {mse_sklearn[best_deg, best_lmb]:.4f}")

    im = ax.pcolormesh(np.log10(lambdas), polydegree, np.log10(mse_sklearn), shading="nearest")
    ax.plot(np.log10(lambdas[best_lmb]), best_deg + 1, "x", color="red", ms=10)
    ax.set_title(f"{k}-fold CV")
    ax.set_xlabel(r"$\log_{10}\lambda$")
    ax.set_ylabel("Polynomial degree")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ cross-validated MSE")
plt.tight_layout()
save_fig("cv_lasso_heatmap")
