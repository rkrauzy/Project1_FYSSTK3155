"""
Final model selection for the Runge function at degree 12, the degree selected
by cross-validation for OLS, Ridge and Lasso. 5-fold cross-validated MSE with
its standard error as a function of lambda for Ridge and Lasso, with OLS as a
reference, the minimum-CV and one-standard-error choices of lambda, and the
number of non-zero Lasso coefficients.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, September 2026)
Level: 3 - Skeleton
Role: Assisted with assembling the script from Case 2, Steps 5 and 6 of
    week36tuesday: cv_curve and select are copied from Step 5 and adapted to
    our design matrix, and the figure is a simplified version of the Ridge and
    Lasso comparison in Step 6. The models and lambda grids are the same as in
    parts/part_d/cv_ols.py, parts/part_d/cv_ridge.py and parts/part_i/cv_lasso.py.
    Claude also suggested the standard-error band for OLS and the markers for
    the minimum-CV and one-standard-error choices of lambda.

Verification:
    Reviewed and executed by the project authors. The minimum CV errors for
    OLS, Ridge and Lasso at degree 12 agree with cv_ols.py, cv_ridge.py and
    cv_lasso.py for k = 5.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score

from src.data import generate_data, design_matrix
from src.plotting import save_fig

n = 100
x, y, y_true = generate_data(n)

degree = 12
X = design_matrix(x, degree)
k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=2026)
n_train = n * (k - 1) // k


def cv_curve(make_model, lambdas):
    """Mean and standard error over folds of the CV-MSE, for every lambda."""
    mean, se = np.zeros(len(lambdas)), np.zeros(len(lambdas))
    for i, lmb in enumerate(lambdas):
        scores = -cross_val_score(make_model(lmb), X, y, cv=kfold,
                                  scoring="neg_mean_squared_error", n_jobs=-1)
        mean[i], se[i] = scores.mean(), scores.std(ddof=1) / np.sqrt(len(scores))
    return mean, se


def select(lambdas, mean, se):
    """The minimum-CV lambda, and the largest lambda within one standard error of it."""
    i = np.argmin(mean)
    return lambdas[i], lambdas[mean <= mean[i] + se[i]].max()


ols_scores = -cross_val_score(make_pipeline(StandardScaler(), LinearRegression()), X, y, cv=kfold,
                              scoring="neg_mean_squared_error")
ols_mean, ols_se = ols_scores.mean(), ols_scores.std(ddof=1) / np.sqrt(k)
print(f"{'method':6s} {'rule':7s} {'lambda':>9s} {'CV MSE':>8s} {'SE':>7s} {'non-zero':>9s}")
print(f"{'OLS':6s} {'-':7s} {'-':>9s} {ols_mean:8.4f} {ols_se:7.4f} {degree:>9d}")

models = {
    "Ridge": (np.logspace(-10, 0, 50),
              lambda lmb: make_pipeline(StandardScaler(), Ridge(alpha=n_train * lmb))),
    "Lasso": (np.logspace(-6, 0, 50),
              lambda lmb: make_pipeline(StandardScaler(), Lasso(alpha=lmb / 2, max_iter=100000))),
}

fig, ax = plt.subplots(figsize=(6, 4))      # one column of the report
for name, (lambdas, make_model) in models.items():
    mean, se = cv_curve(make_model, lambdas)
    lmb_min, lmb_1se = select(lambdas, mean, se)
    line, = ax.semilogx(lambdas, mean, lw=2, label=f"{name}, 5-fold CV")
    ax.fill_between(lambdas, mean - se, mean + se, color=line.get_color(), alpha=0.2)
    for rule, lmb, marker in (("min-CV", lmb_min, "o"), ("one-SE", lmb_1se, "s")):
        j = np.searchsorted(lambdas, lmb)
        nonzero = np.sum(make_model(lmb).fit(X, y)[-1].coef_ != 0)
        print(f"{name:6s} {rule:7s} {lmb:9.2e} {mean[j]:8.4f} {se[j]:7.4f} {nonzero:>9d}")
        ax.plot(lmb, mean[j], marker, color=line.get_color(), ms=6, mec="black")
ax.axhline(ols_mean, color="black", ls="--", label="OLS, 5-fold CV")
ax.axhspan(ols_mean - ols_se, ols_mean + ols_se, color="gray", alpha=0.2)
ax.plot([], [], "o", color="white", mec="black", label=r"$\lambda_{\min}$")
ax.plot([], [], "s", color="white", mec="black", label=r"$\lambda_{1\mathrm{SE}}$")
ax.set_yscale("log")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("Cross-validated MSE")
ax.set_title(f"Degree {degree}: CV error with one standard error")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
save_fig("model_selection")
