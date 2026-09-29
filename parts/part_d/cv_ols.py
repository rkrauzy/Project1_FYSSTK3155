"""
Cross-validated MSE for OLS as a function of polynomial degree, for k = 5 and
k = 10, compared with the bootstrap error from Part C.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, September 2026)
Level: 3 - Skeleton
    Role: Assisted with adapting the cross-validation code from week36tuesday
    (Case 2, Step 4) to the repository structure, with StandardScaler inside the
    pipeline so that scaling is fitted on the training folds only, and with
    plotting the CV MSE for k = 5 and k = 10 together with the bootstrap error.

Verification:  
    Reviewed, executed and interpreted by the project authors.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score

from src.data import generate_data, design_matrix, split_data
from src.resampling import bootstrap
from src.plotting import save_fig

BLUE, RED, YELLOW = "#004488", "#BB5566", "#DDAA33"

n = 100
x, y, y_true = generate_data(n)

max_degree = 20
polydegree = np.arange(1, max_degree + 1)

fig, ax = plt.subplots(figsize=(6.6, 4.0))
for k, color in ((5, BLUE), (10, RED)):
    kf = KFold(n_splits=k, shuffle=True, random_state=2026)
    cv_mse = np.zeros(max_degree)
    for degree in polydegree:
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        scores = -cross_val_score(pipe, design_matrix(x, degree), y, cv=kf,
                                  scoring="neg_mean_squared_error")
        cv_mse[degree - 1] = scores.mean()

    best_deg = np.argmin(cv_mse) + 1
    print(f"k = {k:2d}: CV selects degree {best_deg} (CV-MSE {cv_mse[best_deg - 1]:.4f})")
    ax.plot(polydegree, cv_mse, "o-", color=color, label=f"{k}-fold CV MSE")

x_train, x_test, y_train, y_test = split_data(x, y)
np.random.seed(2026)
error, bias, variance = bootstrap(x_train, x_test, y_train, y_test, max_degree, n_bootstraps=100)
best_boot = np.argmin(error) + 1
print(f"bootstrap selects degree {best_boot} (error {error[best_boot - 1]:.4f})")
ax.plot(polydegree, error, "s-", color=YELLOW, label="bootstrap error")

ax.set_yscale("log")
ax.set_xlabel("Polynomial degree")
ax.set_ylabel("MSE")
ax.legend(frameon=False)
save_fig("cv_vs_bootstrap_ols")