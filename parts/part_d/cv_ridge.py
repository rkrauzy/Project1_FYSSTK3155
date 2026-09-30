"""
Cross-validated MSE for Ridge regression as a function of both polynomial
degree and lambda, for k = 5 and k = 10, shown as heatmaps.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, September 2026)
Level: 3 - Skeleton
Role: Assisted with extending the Ridge cross-validation code from
    week36tuesday (Case 2, Steps 2 and 3) from a fixed degree to a grid over
    degree and lambda, with StandardScaler inside the pipeline so that scaling is
    fitted on the training folds only, with choosing the lambda range, and with
    presenting the grid as heatmaps.

    Also Claude spotted minor inconsistency and fixed it: since in cv_own_vc_sklearn.py
    we use ridge from src.models/ which uses alpha = n * lmbdas so we got wrong scaling
    and best lambda. Adjusted for that here.

Verification: Reviewed, executed and interpreted by the project authors.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
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
lambdas = np.logspace(-10, 0, nlambdas)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, k in zip(axes, (5, 10)):
    kfold = KFold(n_splits=k, shuffle=True, random_state=2026)
    n_train = n * (k - 1) // k
    mse_sklearn = np.zeros((max_degree, nlambdas))
    for degree in polydegree:
        for i, lmb in enumerate(lambdas):
            pipe = make_pipeline(StandardScaler(), Ridge(alpha= n_train * lmb))
            scores = cross_val_score(pipe, design_matrix(x, degree), y, cv=kfold,
                                     scoring="neg_mean_squared_error")
            mse_sklearn[degree - 1, i] = np.mean(-scores)

    best_deg, best_lmb = np.unravel_index(np.argmin(mse_sklearn), mse_sklearn.shape)
    print(f"k = {k:2d}: best degree {best_deg + 1}, best lambda {lambdas[best_lmb]:.4g}, "
          f"cross-validated MSE {mse_sklearn[best_deg, best_lmb]:.4f}")

    im = ax.pcolormesh(np.log10(lambdas), polydegree, np.log10(mse_sklearn), shading="nearest")
    ax.plot(np.log10(lambdas[best_lmb]), best_deg + 1, "x", color="red", ms=10)
    ax.set_title(f"{k}-fold CV")
    ax.set_xlabel(r"$\log_{10}\lambda$")
    ax.set_ylabel("Polynomial degree")
    ax.set_yticks(polydegree[1::2])
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ cross-validated MSE")
plt.tight_layout()
save_fig("cv_ridge_heatmap")